from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH, TILE_SIZE
from nesylink.cnn.components import CLASS_TO_ID, ID_TO_CLASS
from nesylink.vision.pixel_classifier import (
    EntityObservation,
    PixelObservation,
    TileObservation,
    classify_frame as classify_frame_pixels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "nesylink" / "cnn" / "checkpoints" / "tiny_hybrid_cnn_targeted_v3.weights.pt"
PLAYER_CENTER_ADJUST_PX = (-1.5, -4.5)


def classify_frame_cnn(
    frame: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    tile_threshold: float | None = None,
    dynamic_threshold: float | None = None,
    player_recovery_threshold: float | None = None,
    dynamic_top_k: int | None = None,
    fallback: bool = True,
) -> PixelObservation:
    """Use the trained TinyHybridCNN to convert a pixel frame to PixelObservation.

    The CNN consumes only the rendered RGB frame. If PyTorch/model loading fails
    or the CNN does not detect a player, the function can fall back to the
    deterministic palette classifier.
    """

    try:
        return _classify_frame_cnn(
            frame,
            checkpoint=checkpoint,
            device=device,
            tile_threshold=tile_threshold,
            dynamic_threshold=dynamic_threshold,
            player_recovery_threshold=player_recovery_threshold,
            dynamic_top_k=dynamic_top_k,
        )
    except Exception:
        if not fallback:
            raise
        return classify_frame_pixels(frame)


def _classify_frame_cnn(
    frame: np.ndarray,
    *,
    checkpoint: str | Path | None,
    device: str | None,
    tile_threshold: float | None,
    dynamic_threshold: float | None,
    player_recovery_threshold: float | None,
    dynamic_top_k: int | None,
) -> PixelObservation:
    import torch

    from nesylink.cnn.model import (
        DYNAMIC_CLASSES,
        dedupe_dynamic_boxes,
        dynamic_boxes_from_output,
        suppress_tile_classes,
    )

    map_frame = _map_only(frame)
    checkpoint_path = Path(
        checkpoint
        or os.environ.get("NESYLINK_CNN_CHECKPOINT")
        or DEFAULT_CHECKPOINT
    )
    selected_device = device or os.environ.get("NESYLINK_CNN_DEVICE", "cpu")
    tile_min_score = _float_setting("NESYLINK_CNN_TILE_THRESHOLD", tile_threshold, 0.50)
    dynamic_min_score = _float_setting("NESYLINK_CNN_DYNAMIC_THRESHOLD", dynamic_threshold, 0.50)
    player_recovery_min_score = _float_setting(
        "NESYLINK_CNN_PLAYER_RECOVERY_THRESHOLD",
        player_recovery_threshold,
        0.05,
    )
    top_k = _int_setting("NESYLINK_CNN_DYNAMIC_TOP_K", dynamic_top_k, 8)

    model = _load_model(str(checkpoint_path), selected_device)
    array = np.asarray(map_frame, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(selected_device)

    with torch.no_grad():
        output = model(tensor)
        tile_logits = suppress_tile_classes(output["tile_logits"], DYNAMIC_CLASSES)
        tile_probs = torch.softmax(tile_logits, dim=1)[0].cpu()
        tile_scores, tile_class_ids = tile_probs.max(dim=0)
        dynamic_boxes = dedupe_dynamic_boxes(
            dynamic_boxes_from_output(
                output["dynamic_heatmap_logits"],
                output["dynamic_box"],
                min_score=dynamic_min_score,
                top_k=top_k,
            )[0]
        )
        if not any(box.kind == "player" for box in dynamic_boxes):
            recovery_boxes = dynamic_boxes_from_output(
                output["dynamic_heatmap_logits"],
                output["dynamic_box"],
                min_score=player_recovery_min_score,
                top_k=top_k,
            )[0]
            player_candidates = [box for box in recovery_boxes if box.kind == "player"]
            if player_candidates:
                dynamic_boxes.append(max(player_candidates, key=lambda box: box.score))
                dynamic_boxes = dedupe_dynamic_boxes(dynamic_boxes)

    tile_class_grid = tile_class_ids.numpy()
    tile_observations = _tile_observations(tile_class_grid, tile_scores.numpy(), tile_min_score)
    adjust_player_center = bool(np.any(tile_class_grid == CLASS_TO_ID["npc"]))
    entities = _entity_observations(
        dynamic_boxes,
        DYNAMIC_CLASSES,
        adjust_player_center=adjust_player_center,
    )
    if not any(entity.kind == "player" for entity in entities):
        raise RuntimeError("CNN did not detect player")

    entity_by_tile: dict[tuple[int, int], EntityObservation] = {}
    for entity in entities:
        current = entity_by_tile.get(entity.tile)
        if current is None or entity.confidence > current.confidence:
            entity_by_tile[entity.tile] = entity

    tiles: list[TileObservation] = []
    grid_rows: list[list[str]] = [["floor" for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for tile_obs in tile_observations:
        entity = entity_by_tile.get(tile_obs.tile)
        if entity is not None:
            final_obs = TileObservation(
                kind=entity.kind,
                tile=entity.tile,
                confidence=entity.confidence,
                scores={entity.kind: entity.confidence},
            )
        else:
            final_obs = tile_obs
        tiles.append(final_obs)
        x, y = final_obs.tile
        grid_rows[y][x] = final_obs.kind

    player = next((entity for entity in entities if entity.kind == "player"), None)
    monsters = tuple(entity for entity in entities if entity.kind == "monster")
    return PixelObservation(
        grid=tuple(tuple(row) for row in grid_rows),
        tiles=tuple(tiles),
        player=player,
        monsters=monsters,
        entities=entities,
    )


@lru_cache(maxsize=4)
def _load_model(checkpoint_path: str, device: str):
    import torch

    from nesylink.cnn.model import TinyHybridCNN

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)
    model = TinyHybridCNN().to(device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _tile_observations(
    class_ids: np.ndarray,
    scores: np.ndarray,
    tile_threshold: float,
) -> tuple[TileObservation, ...]:
    observations: list[TileObservation] = []
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            score = float(scores[row, col])
            kind = ID_TO_CLASS.get(int(class_ids[row, col]), "unknown")
            if score < tile_threshold:
                kind = "floor"
            observations.append(
                TileObservation(
                    kind=kind,
                    tile=(col, row),
                    confidence=score,
                    scores={kind: score},
                )
            )
    return tuple(observations)


def _entity_observations(
    dynamic_boxes,
    dynamic_classes: tuple[str, ...],
    *,
    adjust_player_center: bool = False,
) -> tuple[EntityObservation, ...]:
    del dynamic_classes
    best: dict[tuple[str, tuple[int, int]], EntityObservation] = {}
    for box in dynamic_boxes:
        if box.kind not in {"player", "monster"} or not box.tiles:
            continue
        left, top, right, bottom = box.bbox_px
        center = ((left + right) * 0.5, (top + bottom) * 0.5)
        if box.kind == "player" and adjust_player_center:
            raw_tile = box.tiles[0]
            center = _adjust_player_center(center)
            tile = raw_tile if _is_boundary_tile(raw_tile) else _tile_from_center(center)
        else:
            tile = box.tiles[0]
        entity = EntityObservation(
            kind=box.kind,
            bbox=box.bbox_px,
            center_px=center,
            tile=tile,
            confidence=float(box.score),
        )
        key = (entity.kind, entity.tile)
        current = best.get(key)
        if current is None or entity.confidence > current.confidence:
            best[key] = entity

    entities = sorted(
        best.values(),
        key=lambda item: (item.kind != "player", item.tile[1], item.tile[0], -item.confidence),
    )
    player = next((entity for entity in entities if entity.kind == "player"), None)
    if player is not None:
        entities = [
            entity
            for entity in entities
            if entity.kind != "monster"
            or entity.confidence >= player.confidence
            or (entity.tile != player.tile and _bbox_iou(entity.bbox, player.bbox) < 0.35)
        ]
    return tuple(entities)


def _bbox_iou(
    left_box: tuple[int, int, int, int],
    right_box: tuple[int, int, int, int],
) -> float:
    left = max(left_box[0], right_box[0])
    top = max(left_box[1], right_box[1])
    right = min(left_box[2], right_box[2])
    bottom = min(left_box[3], right_box[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    left_area = max(0, left_box[2] - left_box[0]) * max(0, left_box[3] - left_box[1])
    right_area = max(0, right_box[2] - right_box[0]) * max(0, right_box[3] - right_box[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _adjust_player_center(center: tuple[float, float]) -> tuple[float, float]:
    x = min(GRID_WIDTH * TILE_SIZE - 1.0, max(0.0, center[0] + PLAYER_CENTER_ADJUST_PX[0]))
    y = min(GRID_HEIGHT * TILE_SIZE - 1.0, max(0.0, center[1] + PLAYER_CENTER_ADJUST_PX[1]))
    return (x, y)


def _tile_from_center(center: tuple[float, float]) -> tuple[int, int]:
    return (
        min(GRID_WIDTH - 1, max(0, int(center[0] // TILE_SIZE))),
        min(GRID_HEIGHT - 1, max(0, int(center[1] // TILE_SIZE))),
    )


def _is_boundary_tile(tile: tuple[int, int]) -> bool:
    x, y = tile
    return x in {0, GRID_WIDTH - 1} or y in {0, GRID_HEIGHT - 1}


def _map_only(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    min_height = GRID_HEIGHT * TILE_SIZE
    min_width = GRID_WIDTH * TILE_SIZE
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape (H, W, 3), got {arr.shape}")
    if arr.shape[0] < min_height or arr.shape[1] < min_width:
        raise ValueError(f"frame is too small for NesyLink map: {arr.shape}")
    return arr[:min_height, :min_width, :3]


def _float_setting(name: str, value: float | None, default: float) -> float:
    if value is not None:
        return float(value)
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _int_setting(name: str, value: int | None, default: int) -> int:
    if value is not None:
        return int(value)
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)
