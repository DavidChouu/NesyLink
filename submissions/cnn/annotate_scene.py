from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from nesylink.cnn.components import (
    ComponentBox,
    component_boxes_from_class_grid,
    draw_component_boxes,
    dynamic_targets_from_room_json,
    static_labels_from_room_json,
)


TILE_SIZE = 16
GRID_WIDTH = 10
GRID_HEIGHT = 8

EXIT_TILES = {
    "north": [(4, 0), (5, 0)],
    "south": [(4, GRID_HEIGHT - 1), (5, GRID_HEIGHT - 1)],
    "west": [(0, 3), (0, 4)],
    "east": [(GRID_WIDTH - 1, 3), (GRID_WIDTH - 1, 4)],
}

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
    "exit": (255, 245, 80),
    "exit_normal": (255, 245, 80),
    "exit_locked": (96, 48, 26),
    "exit_conditional": (70, 220, 180),
    "npc": (240, 154, 52),
}


@dataclass(frozen=True)
class BoxLabel:
    kind: str
    tile: tuple[int, int]
    label: str


@dataclass(frozen=True)
class PixelBoxLabel:
    kind: str
    bbox_px: tuple[int, int, int, int]
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw colored tile annotations over a generated NesyLink scene.")
    parser.add_argument("--image", type=Path, required=True, help="Input PNG, usually 160x128")
    parser.add_argument("--json", type=Path, required=True, help="Room JSON used to generate the PNG")
    parser.add_argument("--out", type=Path, default=None, help="Output annotated PNG")
    parser.add_argument("--labels", action="store_true", help="Draw short text labels inside boxes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.image.with_name(f"{args.image.stem}_annotated.png")
    payload = json.loads(args.json.read_text(encoding="utf-8"))

    image = Image.open(args.image).convert("RGB")
    annotated = draw_component_boxes(image, collect_component_boxes(payload), labels=args.labels)

    out.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out)
    print(f"saved {out}")



def collect_component_boxes(payload: dict[str, Any]) -> list[ComponentBox]:
    boxes = component_boxes_from_class_grid(static_labels_from_room_json(payload))
    for target in dynamic_targets_from_room_json(payload):
        boxes.append(
            ComponentBox(
                kind=target.kind,
                tiles=(pixel_to_tile(target.center_px),),
                bbox_px=target.bbox_px,
                score=1.0,
            )
        )
    return boxes


def pixel_to_tile(center_px: tuple[float, float]) -> tuple[int, int]:
    x, y = center_px
    return min(GRID_WIDTH - 1, max(0, int(x // TILE_SIZE))), min(GRID_HEIGHT - 1, max(0, int(y // TILE_SIZE)))

def collect_labels(payload: dict[str, Any]) -> list[BoxLabel]:
    labels: list[BoxLabel] = []
    labels.extend(layout_wall_labels(payload))
    labels.extend(player_labels(payload))
    labels.extend(object_labels(payload))
    labels.extend(dynamic_tile_labels(payload))
    labels.extend(exit_labels(payload))
    return dedupe_labels(labels)


def layout_wall_labels(payload: dict[str, Any]) -> list[BoxLabel]:
    labels: list[BoxLabel] = []
    for y, row in enumerate(payload.get("layout", [])):
        if not isinstance(row, str):
            continue
        for x, char in enumerate(row):
            if char == "#":
                labels.append(BoxLabel(kind="wall", tile=(x, y), label="wall"))
    return labels


def player_labels(payload: dict[str, Any]) -> list[BoxLabel]:
    if has_pixel_annotation(payload, "player"):
        return []
    spawns = payload.get("spawns", {})
    if not isinstance(spawns, dict):
        return []
    default_spawn = str(payload.get("default_spawn", "default"))
    pos = spawns.get(default_spawn)
    if not valid_pos(pos):
        return []
    return [BoxLabel(kind="player", tile=(int(pos[0]), int(pos[1])), label="player")]


def object_labels(payload: dict[str, Any]) -> list[BoxLabel]:
    labels: list[BoxLabel] = []
    for obj in payload.get("objects", []):
        if not isinstance(obj, dict):
            continue
        kind = str(obj.get("kind", "unknown"))
        pos = obj.get("pos")
        if kind == "trap":
            kind = trap_class_from_object(obj)
        if kind not in COLORS or not valid_pos(pos):
            continue
        label = kind
        if kind == "monster":
            label = str(obj.get("monster_type", "monster"))
        labels.append(BoxLabel(kind=kind, tile=(int(pos[0]), int(pos[1])), label=label))
    return labels


def dynamic_tile_labels(payload: dict[str, Any]) -> list[BoxLabel]:
    labels: list[BoxLabel] = []
    for obj in payload.get("dynamic_objects", []):
        if not isinstance(obj, dict):
            continue
        initial_state = str(obj.get("initial_state", ""))
        states = obj.get("states", {})
        if not isinstance(states, dict):
            continue
        state = states.get(initial_state)
        if not isinstance(state, dict):
            continue
        active_tile = str(obj.get("active_tile", "bridge"))
        background_tile = str(obj.get("background_tile", "gap"))
        all_tiles = set()
        for raw_state in states.values():
            if not isinstance(raw_state, dict):
                continue
            for pos in raw_state.get("tiles", []):
                if valid_pos(pos):
                    all_tiles.add((int(pos[0]), int(pos[1])))
        active_tiles = set()
        for pos in state.get("tiles", []):
            if valid_pos(pos):
                active_tiles.add((int(pos[0]), int(pos[1])))
        for tile in sorted(all_tiles - active_tiles):
            if background_tile in COLORS:
                labels.append(BoxLabel(kind=background_tile, tile=tile, label=background_tile))
        for tile in sorted(active_tiles):
            if active_tile in COLORS:
                labels.append(BoxLabel(kind=active_tile, tile=tile, label=active_tile))
    return labels


def exit_labels(payload: dict[str, Any]) -> list[BoxLabel]:
    labels: list[BoxLabel] = []
    for exit_cfg in payload.get("exits", []):
        if not isinstance(exit_cfg, dict):
            continue
        direction = str(exit_cfg.get("direction", ""))
        kind = exit_class_from_config(exit_cfg)
        label = {
            "exit_locked": "locked",
            "exit_conditional": "cond",
        }.get(kind, "normal")
        for tile in EXIT_TILES.get(direction, []):
            labels.append(BoxLabel(kind=kind, tile=tile, label=label))
    return labels


def exit_class_from_config(exit_cfg: dict[str, Any]) -> str:
    exit_type = str(exit_cfg.get("type", "normal"))
    if exit_type == "conditional":
        return "exit_conditional"
    if exit_type == "locked_key" or "key_count" in exit_cfg.get("requires", {}):
        return "exit_locked"
    return "exit_normal"


def trap_class_from_object(obj: dict[str, Any]) -> str:
    return "abyss" if str(obj.get("trap_type", obj.get("type", "spike"))).lower() == "abyss" else "trap"


def dedupe_labels(labels: list[BoxLabel]) -> list[BoxLabel]:
    # Later labels are more specific and should appear on top of layout labels.
    by_key: dict[tuple[str, tuple[int, int], str], BoxLabel] = {}
    for label in labels:
        by_key[(label.kind, label.tile, label.label)] = label
    return list(by_key.values())


def draw_box(draw: ImageDraw.ImageDraw, item: BoxLabel, *, labels: bool) -> None:
    x, y = item.tile
    color = COLORS.get(item.kind, (255, 255, 255))
    rect = [x * TILE_SIZE + 1, y * TILE_SIZE + 1, (x + 1) * TILE_SIZE - 2, (y + 1) * TILE_SIZE - 2]
    draw.rectangle(rect, outline=color, width=2)
    if labels:
        draw.text((x * TILE_SIZE + 2, y * TILE_SIZE + 2), item.label[:3], fill=color)


def collect_pixel_labels(payload: dict[str, Any]) -> list[PixelBoxLabel]:
    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict):
        return []
    raw_boxes = annotations.get("pixel_boxes", [])
    if not isinstance(raw_boxes, list):
        return []

    labels: list[PixelBoxLabel] = []
    for raw_box in raw_boxes:
        if not isinstance(raw_box, dict):
            continue
        kind = str(raw_box.get("kind", "unknown"))
        if kind not in COLORS:
            continue
        bbox = raw_box.get("bbox_px")
        if not valid_bbox(bbox):
            continue
        labels.append(
            PixelBoxLabel(
                kind=kind,
                bbox_px=tuple(int(round(value)) for value in bbox),
                label=str(raw_box.get("label", kind)),
            )
        )
    return labels


def draw_pixel_box(draw: ImageDraw.ImageDraw, item: PixelBoxLabel, *, labels: bool) -> None:
    color = COLORS.get(item.kind, (255, 255, 255))
    left, top, right, bottom = item.bbox_px
    rect = [left, top, right - 1, bottom - 1]
    draw.rectangle(rect, outline=color, width=3)
    if labels:
        draw.text((left + 1, top + 1), item.label[:3], fill=color)


def has_pixel_annotation(payload: dict[str, Any], kind: str) -> bool:
    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict):
        return False
    raw_boxes = annotations.get("pixel_boxes", [])
    if not isinstance(raw_boxes, list):
        return False
    return any(isinstance(raw_box, dict) and raw_box.get("kind") == kind for raw_box in raw_boxes)


def valid_pos(pos: Any) -> bool:
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


if __name__ == "__main__":
    main()
