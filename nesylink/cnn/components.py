from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


TILE_SIZE = 16
GRID_WIDTH = 10
GRID_HEIGHT = 8

COMPONENT_CLASSES = (
    "floor",
    "wall",
    "player",
    "chest",
    "monster",
    "trap",
    "abyss",
    "button",
    "switch",
    "gap",
    "bridge",
    "exit_normal",
    "exit_locked",
    "exit_conditional",
    "npc",
    "unknown",
)
DYNAMIC_CLASSES = ("player", "monster")
DYNAMIC_CLASS_TO_ID = {name: idx for idx, name in enumerate(DYNAMIC_CLASSES)}
CLASS_TO_ID = {name: idx for idx, name in enumerate(COMPONENT_CLASSES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}

SPLIT_TILE_COMPONENT_KINDS = (
    "wall",
    "player",
    "chest",
    "monster",
    "trap",
    "abyss",
    "button",
    "switch",
    "gap",
    "bridge",
    "npc",
    "unknown",
)

COLORS = {
    "player": (30, 255, 80),
    "wall": (255, 40, 90),
    "chest": (180, 95, 35),
    "monster": (255, 145, 20),
    "trap": (170, 170, 190),
    "abyss": (30, 30, 42),
    "button": (40, 210, 90),
    "switch": (255, 225, 40),
    "gap": (20, 25, 70),
    "bridge": (190, 115, 45),
    "exit_normal": (255, 245, 80),
    "exit_locked": (96, 48, 26),
    "exit_conditional": (70, 220, 180),
    "npc": (240, 154, 52),
    "unknown": (255, 255, 255),
}

EXIT_TILES = {
    "north": [(4, 0), (5, 0)],
    "south": [(4, GRID_HEIGHT - 1), (5, GRID_HEIGHT - 1)],
    "west": [(0, 3), (0, 4)],
    "east": [(GRID_WIDTH - 1, 3), (GRID_WIDTH - 1, 4)],
}


@dataclass(frozen=True)
class ComponentBox:
    kind: str
    tiles: tuple[tuple[int, int], ...]
    bbox_px: tuple[int, int, int, int]
    score: float = 1.0


@dataclass(frozen=True)
class DynamicTarget:
    kind: str
    center_px: tuple[float, float]
    bbox_px: tuple[int, int, int, int]


def labels_from_room_json(payload: dict[str, Any]) -> np.ndarray:
    """Build an 8x10 class-id target grid from a room JSON payload."""
    labels = np.full((GRID_HEIGHT, GRID_WIDTH), CLASS_TO_ID["floor"], dtype=np.int64)

    for y, row in enumerate(payload.get("layout", [])):
        if not isinstance(row, str):
            continue
        for x, value in enumerate(row[:GRID_WIDTH]):
            if y < GRID_HEIGHT and value == "#":
                labels[y, x] = CLASS_TO_ID["wall"]

    for obj in payload.get("dynamic_objects", []):
        if not isinstance(obj, dict):
            continue
        write_dynamic_object_labels(labels, obj)

    for exit_cfg in payload.get("exits", []):
        if not isinstance(exit_cfg, dict):
            continue
        exit_class = exit_class_from_config(exit_cfg)
        for x, y in EXIT_TILES.get(str(exit_cfg.get("direction", "")), []):
            labels[y, x] = CLASS_TO_ID[exit_class]

    for obj in payload.get("objects", []):
        if not isinstance(obj, dict):
            continue
        write_object_label(labels, obj)

    spawns = payload.get("spawns", {})
    default_spawn = str(payload.get("default_spawn", "default"))
    if isinstance(spawns, dict) and valid_tile(spawns.get(default_spawn)):
        x, y = spawns[default_spawn]
        labels[int(y), int(x)] = CLASS_TO_ID["player"]

    return labels


def static_labels_from_room_json(payload: dict[str, Any]) -> np.ndarray:
    """Build tile labels while keeping pixel-moving entities out of the grid target."""
    labels = np.full((GRID_HEIGHT, GRID_WIDTH), CLASS_TO_ID["floor"], dtype=np.int64)

    for y, row in enumerate(payload.get("layout", [])):
        if not isinstance(row, str):
            continue
        for x, value in enumerate(row[:GRID_WIDTH]):
            if y < GRID_HEIGHT and value == "#":
                labels[y, x] = CLASS_TO_ID["wall"]

    for obj in payload.get("dynamic_objects", []):
        if not isinstance(obj, dict):
            continue
        write_dynamic_object_labels(labels, obj)

    for exit_cfg in payload.get("exits", []):
        if not isinstance(exit_cfg, dict):
            continue
        exit_class = exit_class_from_config(exit_cfg)
        for x, y in EXIT_TILES.get(str(exit_cfg.get("direction", "")), []):
            labels[y, x] = CLASS_TO_ID[exit_class]

    dynamic_objects = [obj for obj in payload.get("dynamic_objects", []) if isinstance(obj, dict)]
    for obj in payload.get("objects", []):
        if not isinstance(obj, dict) or str(obj.get("kind")) == "monster":
            continue
        write_object_label(labels, obj)

    # Dynamic bridge/gap tiles are drawn underneath exits but protect traps from rendering.
    # Reapply them last so full-map abyss traps do not erase active bridges in the target grid.
    for obj in dynamic_objects:
        write_dynamic_object_labels(labels, obj)

    return labels


def dynamic_targets_from_room_json(payload: dict[str, Any]) -> list[DynamicTarget]:
    """Build pixel-level targets for player and monsters from JSON placement.

    Generated datasets may add exact moving-entity boxes under
    annotations.pixel_boxes. Those boxes are preferred over spawn/tile fallbacks.
    """
    targets: list[DynamicTarget] = []
    annotated_kinds: set[str] = set()
    for target in pixel_box_targets_from_room_json(payload):
        targets.append(target)
        annotated_kinds.add(target.kind)

    spawns = payload.get("spawns", {})
    default_spawn = str(payload.get("default_spawn", "default"))
    if "player" not in annotated_kinds and isinstance(spawns, dict) and valid_tile(spawns.get(default_spawn)):
        x, y = spawns[default_spawn]
        targets.append(dynamic_target_from_tile("player", (int(x), int(y))))

    if "monster" not in annotated_kinds:
        for obj in payload.get("objects", []):
            if not isinstance(obj, dict) or str(obj.get("kind")) != "monster":
                continue
            pos = obj.get("pos")
            if valid_tile(pos):
                targets.append(dynamic_target_from_tile("monster", (int(pos[0]), int(pos[1]))))
    return targets


def pixel_box_targets_from_room_json(payload: dict[str, Any]) -> list[DynamicTarget]:
    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict):
        return []
    raw_boxes = annotations.get("pixel_boxes", [])
    if not isinstance(raw_boxes, list):
        return []

    targets: list[DynamicTarget] = []
    for raw_box in raw_boxes:
        if not isinstance(raw_box, dict):
            continue
        kind = str(raw_box.get("kind", "unknown"))
        if kind not in DYNAMIC_CLASS_TO_ID:
            continue
        bbox = raw_box.get("bbox_px")
        if not valid_bbox(bbox):
            continue
        left, top, right, bottom = tuple(int(round(value)) for value in bbox)
        targets.append(
            DynamicTarget(
                kind=kind,
                center_px=((left + right) * 0.5, (top + bottom) * 0.5),
                bbox_px=(left, top, right, bottom),
            )
        )
    return targets


def dynamic_target_from_tile(kind: str, tile: tuple[int, int]) -> DynamicTarget:
    left = tile[0] * TILE_SIZE
    top = tile[1] * TILE_SIZE
    right = left + TILE_SIZE
    bottom = top + TILE_SIZE
    return DynamicTarget(
        kind=kind,
        center_px=(left + TILE_SIZE * 0.5, top + TILE_SIZE * 0.5),
        bbox_px=(left, top, right, bottom),
    )


def exit_class_from_config(exit_cfg: dict[str, Any]) -> str:
    exit_type = str(exit_cfg.get("type", "normal"))
    if exit_type == "conditional":
        return "exit_conditional"
    if exit_type == "locked_key" or "key_count" in exit_cfg.get("requires", {}):
        return "exit_locked"
    return "exit_normal"


def dynamic_heatmap_targets(
    targets: Sequence[DynamicTarget],
    *,
    stride: int = 4,
    height: int = 32,
    width: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode dynamic targets as heatmaps and box regression maps.

    Returns:
        heatmap: (2, 32, 40), one positive cell per object center.
        box: (8, 32, 40), dx, dy, width, height per class.
        mask: (2, 32, 40), positive locations for box loss.
    """
    heatmap = np.zeros((len(DYNAMIC_CLASSES), height, width), dtype=np.float32)
    box = np.zeros((len(DYNAMIC_CLASSES) * 4, height, width), dtype=np.float32)
    mask = np.zeros((len(DYNAMIC_CLASSES), height, width), dtype=np.float32)

    for target in targets:
        class_id = DYNAMIC_CLASS_TO_ID.get(target.kind)
        if class_id is None:
            continue
        center_x, center_y = target.center_px
        cell_x = min(width - 1, max(0, int(center_x // stride)))
        cell_y = min(height - 1, max(0, int(center_y // stride)))
        left, top, right, bottom = target.bbox_px
        heatmap[class_id, cell_y, cell_x] = 1.0
        mask[class_id, cell_y, cell_x] = 1.0
        base = class_id * 4
        box[base + 0, cell_y, cell_x] = float(center_x - cell_x * stride)
        box[base + 1, cell_y, cell_x] = float(center_y - cell_y * stride)
        box[base + 2, cell_y, cell_x] = float(right - left)
        box[base + 3, cell_y, cell_x] = float(bottom - top)
    return heatmap, box, mask


def write_object_label(labels: np.ndarray, obj: dict[str, Any]) -> None:
    kind = str(obj.get("kind", "unknown"))
    if kind not in CLASS_TO_ID:
        return
    if kind == "chest" and bool(obj.get("hidden", False)):
        return
    if kind == "trap":
        trap_class = trap_class_from_object(obj)
        for x, y in trap_tiles_from_object(obj):
            if ID_TO_CLASS.get(int(labels[y, x])) == "bridge":
                continue
            labels[y, x] = CLASS_TO_ID[trap_class]
        return
    pos = obj.get("pos")
    if valid_tile(pos):
        labels[int(pos[1]), int(pos[0])] = CLASS_TO_ID[kind]



def trap_class_from_object(obj: dict[str, Any]) -> str:
    return "abyss" if str(obj.get("trap_type", obj.get("type", "spike"))).lower() == "abyss" else "trap"

def trap_tiles_from_object(obj: dict[str, Any]) -> list[tuple[int, int]]:
    tiles: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    pos = obj.get("pos")
    if valid_tile(pos):
        tile = (int(pos[0]), int(pos[1]))
        tiles.append(tile)
        seen.add(tile)

    raw_tiles = obj.get("tiles", [])
    if isinstance(raw_tiles, list):
        for raw_tile in raw_tiles:
            if valid_tile(raw_tile):
                tile = (int(raw_tile[0]), int(raw_tile[1]))
                if tile not in seen:
                    tiles.append(tile)
                    seen.add(tile)

    raw_rects = obj.get("rects", [])
    if isinstance(raw_rects, list):
        for raw_rect in raw_rects:
            if not isinstance(raw_rect, dict):
                continue
            start = raw_rect.get("from")
            end = raw_rect.get("to")
            if not valid_tile(start) or not valid_tile(end):
                continue
            min_x, max_x = sorted((int(start[0]), int(end[0])))
            min_y, max_y = sorted((int(start[1]), int(end[1])))
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    tile = (x, y)
                    if tile not in seen:
                        tiles.append(tile)
                        seen.add(tile)
    return tiles


def write_dynamic_object_labels(labels: np.ndarray, obj: dict[str, Any]) -> None:
    states = obj.get("states", {})
    if not isinstance(states, dict):
        return
    initial_state = str(obj.get("initial_state", ""))
    active_state = states.get(initial_state)
    active_tile = str(obj.get("active_tile", "bridge"))
    background_tile = str(obj.get("background_tile", "gap"))

    for state in states.values():
        if not isinstance(state, dict):
            continue
        for pos in state.get("tiles", []):
            if valid_tile(pos) and background_tile in CLASS_TO_ID:
                labels[int(pos[1]), int(pos[0])] = CLASS_TO_ID[background_tile]

    if not isinstance(active_state, dict):
        return
    for pos in active_state.get("tiles", []):
        if valid_tile(pos) and active_tile in CLASS_TO_ID:
            labels[int(pos[1]), int(pos[0])] = CLASS_TO_ID[active_tile]


def component_boxes_from_class_grid(
    class_grid: Sequence[Sequence[int | str]] | np.ndarray,
    *,
    score_grid: Sequence[Sequence[float]] | np.ndarray | None = None,
    ignored: Iterable[str] = ("floor",),
    split_tile_kinds: Iterable[str] = SPLIT_TILE_COMPONENT_KINDS,
) -> list[ComponentBox]:
    """Convert a class grid into boxes.

    Static map objects are kept as one box per grid cell so navigation can use
    exact tile occupancy. The only static classes merged by default are the
    two-tile door classes: exit_normal and exit_locked.
    """
    grid = normalize_class_grid(class_grid)
    scores = None if score_grid is None else np.asarray(score_grid, dtype=np.float32)
    ignored_set = set(ignored)
    split_set = set(split_tile_kinds)
    seen = np.zeros(grid.shape, dtype=bool)
    boxes: list[ComponentBox] = []

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if seen[y, x]:
                continue
            kind = ID_TO_CLASS.get(int(grid[y, x]), "unknown")
            if kind in ignored_set:
                seen[y, x] = True
                continue
            if kind in split_set:
                seen[y, x] = True
                boxes.append(make_component_box(kind, ((x, y),), scores))
                continue
            tiles = flood_fill_same_kind(grid, seen, (x, y), int(grid[y, x]))
            boxes.append(make_component_box(kind, tiles, scores))
    return boxes


def flood_fill_same_kind(
    grid: np.ndarray,
    seen: np.ndarray,
    start: tuple[int, int],
    class_id: int,
) -> tuple[tuple[int, int], ...]:
    queue: deque[tuple[int, int]] = deque([start])
    seen[start[1], start[0]] = True
    tiles: list[tuple[int, int]] = []
    while queue:
        x, y = queue.popleft()
        tiles.append((x, y))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
                continue
            if seen[ny, nx] or int(grid[ny, nx]) != class_id:
                continue
            seen[ny, nx] = True
            queue.append((nx, ny))
    return tuple(tiles)


def make_component_box(
    kind: str,
    tiles: tuple[tuple[int, int], ...],
    scores: np.ndarray | None,
) -> ComponentBox:
    xs = [tile[0] for tile in tiles]
    ys = [tile[1] for tile in tiles]
    bbox = (
        min(xs) * TILE_SIZE,
        min(ys) * TILE_SIZE,
        (max(xs) + 1) * TILE_SIZE,
        (max(ys) + 1) * TILE_SIZE,
    )
    if scores is None:
        score = 1.0
    else:
        score = float(np.mean([scores[y, x] for x, y in tiles]))
    return ComponentBox(kind=kind, tiles=tiles, bbox_px=bbox, score=score)


def draw_component_boxes(
    image: Image.Image,
    boxes: Sequence[ComponentBox],
    *,
    labels: bool = True,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        color = COLORS.get(box.kind, (255, 255, 255))
        left, top, right, bottom = box.bbox_px
        draw.rectangle([left + 1, top + 1, right - 2, bottom - 2], outline=color, width=2)
        if labels:
            text = f"{box.kind[:3]} {box.score:.2f}"
            draw.text((left + 2, top + 2), text, fill=color)
    return out


def save_component_overlay(
    image_path: Path,
    boxes: Sequence[ComponentBox],
    out_path: Path,
    *,
    labels: bool = True,
) -> None:
    image = Image.open(image_path)
    out = draw_component_boxes(image, boxes, labels=labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def normalize_class_grid(class_grid: Sequence[Sequence[int | str]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(class_grid)
    if arr.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(f"class grid must have shape {(GRID_HEIGHT, GRID_WIDTH)}, got {arr.shape}")
    if arr.dtype.kind in {"U", "S", "O"}:
        converted = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int64)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                converted[y, x] = CLASS_TO_ID.get(str(arr[y, x]), CLASS_TO_ID["unknown"])
        return converted
    return arr.astype(np.int64, copy=False)


def valid_tile(pos: Any) -> bool:
    return (
        isinstance(pos, list)
        and len(pos) == 2
        and isinstance(pos[0], int)
        and isinstance(pos[1], int)
        and 0 <= pos[0] < GRID_WIDTH
        and 0 <= pos[1] < GRID_HEIGHT
    )


def valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if not all(isinstance(value, int | float) for value in bbox):
        return False
    left, top, right, bottom = [float(value) for value in bbox]
    return 0 <= left < right <= GRID_WIDTH * TILE_SIZE and 0 <= top < bottom <= GRID_HEIGHT * TILE_SIZE
