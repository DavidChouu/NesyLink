from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from nesylink.core.constants import COLOR_NPC, GRID_HEIGHT, GRID_WIDTH, TILE_SIZE
from nesylink.core.rendering import sprites


Color = tuple[int, int, int]
Position = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class EntityObservation:
    kind: str
    bbox: BBox
    center_px: tuple[float, float]
    tile: Position
    confidence: float


@dataclass(frozen=True)
class TileObservation:
    kind: str
    tile: Position
    confidence: float
    scores: dict[str, float]


@dataclass(frozen=True)
class PixelObservation:
    grid: tuple[tuple[str, ...], ...]
    tiles: tuple[TileObservation, ...]
    player: EntityObservation | None
    monsters: tuple[EntityObservation, ...]
    entities: tuple[EntityObservation, ...]


KIND_PRIORITY = (
    "player",
    "monster",
    "npc",
    "wall",
    "gap",
    "abyss",
    "bridge",
    "trap",
    "chest",
    "button_pressed",
    "button",
    "switch",
    "exit_locked",
    "exit_conditional",
    "exit_normal",
    "floor",
)

KIND_COLORS: dict[str, tuple[Color, ...]] = {
    "floor": (sprites.FLOOR_LIGHT, sprites.FLOOR_DARK, sprites.FLOOR_DARKER),
    "wall": (sprites.WALL_MID, sprites.WALL_LIGHT, sprites.WALL_DARK, sprites.WALL_EDGE),
    "player": (
        sprites.PLAYER_TUNIC,
        sprites.PLAYER_TUNIC_LIGHT,
        sprites.PLAYER_FACE,
        sprites.PLAYER_HAIR,
    ),
    "monster": (
        (238, 126, 28),
        (255, 180, 48),
        (200, 78, 16),
        sprites.MONSTER_DARK,
        sprites.MONSTER_EYE,
    ),
    "npc": (COLOR_NPC,),
    "chest": (
        sprites.CHEST_WOOD,
        sprites.CHEST_BAND,
        sprites.CHEST_OPEN_INNER,
        sprites.LOCK_COLOR,
    ),
    "trap": (
        sprites.SPIKE_METAL,
        sprites.SPIKE_SHADE,
        sprites.SPIKE_HIGHLIGHT,
    ),
    "abyss": ((0, 0, 0),),
    "button": (sprites.BUTTON_UP,),
    "button_pressed": (sprites.BUTTON_DOWN, (86, 146, 104)),
    "switch": (sprites.SWITCH_BODY, sprites.SWITCH_DOWN),
    "gap": (sprites.GAP_DARK, sprites.GAP_MID),
    "bridge": (sprites.BRIDGE_WOOD, sprites.BRIDGE_EDGE),
    "exit_normal": (sprites.SHADOW, sprites.HIGHLIGHT, sprites.WALL_LIGHT),
    "exit_locked": (sprites.DOOR_WOOD, sprites.LOCK_COLOR),
    "exit_conditional": (sprites.CONDITIONAL_GLYPH, sprites.WALL_DARK),
}

ENTITY_COLORS: dict[str, tuple[Color, ...]] = {
    "player": (
        sprites.PLAYER_TUNIC,
        sprites.PLAYER_TUNIC_LIGHT,
    ),
    "monster": (
        (238, 126, 28),
        (255, 180, 48),
        (200, 78, 16),
    ),
}


def classify_frame(frame: np.ndarray, *, tolerance: int = 4) -> PixelObservation:
    """Extract a symbolic observation from a raw NesyLink pixel frame.

    The classifier is deliberately training-free. NesyLink uses fixed pixel-art
    palettes, so exact or near-exact color matching is a stronger first baseline
    than a CNN for the public simulator.
    """

    map_frame = _map_only(frame)
    entities = detect_entities(map_frame, tolerance=tolerance)
    grid_tiles = classify_tile_grid(map_frame, entities=entities, tolerance=tolerance)
    grid = _tiles_to_grid(grid_tiles)
    player = next((entity for entity in entities if entity.kind == "player"), None)
    monsters = tuple(entity for entity in entities if entity.kind == "monster")
    return PixelObservation(
        grid=grid,
        tiles=grid_tiles,
        player=player,
        monsters=monsters,
        entities=entities,
    )


def classify_tile_grid(
    frame: np.ndarray,
    *,
    entities: Iterable[EntityObservation] = (),
    tolerance: int = 4,
) -> tuple[TileObservation, ...]:
    map_frame = _map_only(frame)
    entity_by_tile: dict[Position, EntityObservation] = {}
    for entity in entities:
        existing = entity_by_tile.get(entity.tile)
        if existing is None or entity.confidence > existing.confidence:
            entity_by_tile[entity.tile] = entity

    observations: list[TileObservation] = []
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            tile = (col, row)
            if tile in entity_by_tile:
                entity = entity_by_tile[tile]
                observations.append(
                    TileObservation(
                        kind=entity.kind,
                        tile=tile,
                        confidence=entity.confidence,
                        scores={entity.kind: entity.confidence},
                    )
                )
                continue

            patch = _tile_patch(map_frame, col, row)
            scores = _tile_scores(patch, tolerance=tolerance)
            kind = _choose_tile_kind(scores)
            observations.append(
                TileObservation(
                    kind=kind,
                    tile=tile,
                    confidence=scores.get(kind, 0.0),
                    scores=scores,
                )
            )
    return tuple(observations)


def detect_entities(frame: np.ndarray, *, tolerance: int = 4) -> tuple[EntityObservation, ...]:
    map_frame = _map_only(frame)
    entities: list[EntityObservation] = []
    for kind, colors in ENTITY_COLORS.items():
        mask = _multi_color_mask(map_frame, colors, tolerance=tolerance)
        mask = _dilate_mask(mask, radius=1)
        min_area = 20 if kind == "player" else 12
        for bbox, area in _connected_components(mask, min_area=min_area):
            x0, y0, x1, y1 = bbox
            width = max(1, x1 - x0)
            height = max(1, y1 - y0)
            if width > TILE_SIZE * 2 or height > TILE_SIZE * 2:
                continue
            center = ((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0)
            tile = _pixel_to_tile(center)
            confidence = min(1.0, area / 70.0)
            entities.append(
                EntityObservation(
                    kind=kind,
                    bbox=bbox,
                    center_px=center,
                    tile=tile,
                    confidence=confidence,
                )
            )
    entities.sort(key=lambda item: (item.kind != "player", item.tile[1], item.tile[0], -item.confidence))
    return tuple(entities)


def _map_only(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape (H, W, 3), got {arr.shape}")
    min_height = GRID_HEIGHT * TILE_SIZE
    min_width = GRID_WIDTH * TILE_SIZE
    if arr.shape[0] < min_height or arr.shape[1] < min_width:
        raise ValueError(f"frame is too small for NesyLink map: {arr.shape}")
    return arr[:min_height, :min_width, :3]


def _tile_patch(frame: np.ndarray, col: int, row: int) -> np.ndarray:
    y0 = row * TILE_SIZE
    x0 = col * TILE_SIZE
    return frame[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]


def _tile_scores(patch: np.ndarray, *, tolerance: int) -> dict[str, float]:
    return {
        kind: _color_fraction(patch, colors, tolerance=tolerance)
        for kind, colors in KIND_COLORS.items()
    }


def _choose_tile_kind(scores: dict[str, float]) -> str:
    if scores["wall"] >= 0.35:
        return "wall"
    if scores["gap"] >= 0.60:
        return "gap"
    if scores["abyss"] >= 0.60:
        return "abyss"
    if scores["exit_normal"] >= 0.20:
        return "exit_normal"
    if scores["exit_locked"] >= 0.40 and scores["bridge"] < 0.55:
        return "exit_locked"
    if scores["exit_conditional"] >= 0.18:
        return "exit_conditional"
    if scores["bridge"] >= 0.45:
        return "bridge"
    if scores["chest"] >= 0.20:
        return "chest"
    if scores["trap"] >= 0.08:
        return "trap"
    if scores["npc"] >= 0.12:
        return "npc"
    if scores["button_pressed"] >= 0.08:
        return "button_pressed"
    if scores["button"] >= 0.08:
        return "button"
    if scores["exit_locked"] >= 0.18:
        return "exit_locked"
    if scores["switch"] >= 0.10:
        return "switch"

    candidates = {kind: scores.get(kind, 0.0) for kind in KIND_PRIORITY}
    kind = max(candidates, key=candidates.__getitem__)
    return kind if candidates[kind] > 0.02 else "unknown"


def _tiles_to_grid(tiles: tuple[TileObservation, ...]) -> tuple[tuple[str, ...], ...]:
    rows: list[list[str]] = [["unknown" for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for tile in tiles:
        col, row = tile.tile
        rows[row][col] = tile.kind
    return tuple(tuple(row) for row in rows)


def _color_fraction(patch: np.ndarray, colors: Iterable[Color], *, tolerance: int) -> float:
    mask = _multi_color_mask(patch, colors, tolerance=tolerance)
    return float(mask.mean())


def _multi_color_mask(frame: np.ndarray, colors: Iterable[Color], *, tolerance: int) -> np.ndarray:
    masks = [_color_mask(frame, color, tolerance=tolerance) for color in colors]
    if not masks:
        return np.zeros(frame.shape[:2], dtype=bool)
    out = masks[0].copy()
    for mask in masks[1:]:
        out |= mask
    return out


def _dilate_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return out


def _color_mask(frame: np.ndarray, color: Color, *, tolerance: int) -> np.ndarray:
    target = np.asarray(color, dtype=np.int16)
    diff = np.abs(frame.astype(np.int16) - target)
    return np.all(diff <= tolerance, axis=2)


def _connected_components(mask: np.ndarray, *, min_area: int) -> list[tuple[BBox, int]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[BBox, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []

            while queue:
                cx, cy = queue.popleft()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if seen[ny, nx] or not mask[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    queue.append((nx, ny))

            area = len(xs)
            if area >= min_area:
                components.append(((min(xs), min(ys), max(xs) + 1, max(ys) + 1), area))

    return components


def _pixel_to_tile(center_px: tuple[float, float]) -> Position:
    x, y = center_px
    col = min(GRID_WIDTH - 1, max(0, int(x // TILE_SIZE)))
    row = min(GRID_HEIGHT - 1, max(0, int(y // TILE_SIZE)))
    return (col, row)
