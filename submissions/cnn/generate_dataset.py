from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CNN_DIR = Path(__file__).resolve().parent

from nesylink.cnn.annotate_scene import collect_component_boxes
from nesylink.cnn.components import draw_component_boxes
from nesylink.cnn.generate_synthetic_scene import (
    apply_monster_offsets,
    apply_player_offset,
    build_synthetic_room,
    player_context_for_index,
    player_facing_for_index,
    player_facing_from_payload,
    scene_style_for_index,
    write_runtime_pixel_annotations,
)
from nesylink.core.constants import MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH
from nesylink.core.rendering import render_frame
from nesylink.core.state import PlayerState, tile_to_top_left_px
from nesylink.core.world.rooms import RoomManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/test PNG+JSON datasets for CNN perception.")
    parser.add_argument("--train-dir", type=Path, default=CNN_DIR / "generated" / "train")
    parser.add_argument("--test-dir", type=Path, default=CNN_DIR / "generated" / "test")
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=30)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--test-seed-start", type=int, default=10000)
    parser.add_argument("--annotate-test", action="store_true")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--sheet-out", type=Path, default=CNN_DIR / "generated" / "test_annotated_sheet.png")
    parser.add_argument("--sheet-cols", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_paths = generate_split(
        args.train_dir,
        count=args.train_count,
        seed_start=args.train_seed_start,
        prefix="train",
        annotate=False,
        labels=False,
    )
    test_paths = generate_split(
        args.test_dir,
        count=args.test_count,
        seed_start=args.test_seed_start,
        prefix="test",
        annotate=args.annotate_test,
        labels=args.labels,
    )
    print(f"generated train images={len(train_paths)} dir={args.train_dir}")
    print(f"generated test images={len(test_paths)} dir={args.test_dir}")

    if args.annotate_test:
        make_sheet(
            [path.with_name(f"{path.stem}_annotated.png") for path in test_paths],
            args.sheet_out,
            cols=args.sheet_cols,
        )
        print(f"saved test sheet {args.sheet_out}")


def generate_split(
    out_dir: Path,
    *,
    count: int,
    seed_start: int,
    prefix: str,
    annotate: bool,
    labels: bool,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for index in range(count):
        seed = seed_start + index
        image_path = out_dir / f"{prefix}_{index:04d}.png"
        json_path = image_path.with_suffix(".json")
        offset = player_offset_for_index(index)
        context = player_context_for_index(index)
        facing = player_facing_for_index(index)
        scene_style = scene_style_for_index(index)
        generate_one(
            seed,
            offset,
            image_path,
            json_path,
            player_context=context,
            player_facing=facing,
            scene_style=scene_style,
        )
        image_paths.append(image_path)
        if annotate:
            annotated_path = image_path.with_name(f"{image_path.stem}_annotated.png")
            annotate_one(image_path, json_path, annotated_path, labels=labels)
        if (index + 1) % 50 == 0 or index + 1 == count:
            print(f"{prefix}: {index + 1}/{count}")
    return image_paths


def generate_one(
    seed: int,
    player_offset_px: tuple[int, int],
    image_path: Path,
    json_path: Path,
    *,
    player_context: str | None = None,
    player_facing: str | None = None,
    scene_style: str = "mixed",
) -> None:
    rng = random.Random(seed)
    payload = build_synthetic_room(
        rng,
        player_context=player_context,
        player_facing=player_facing,
        scene_style=scene_style,
    )

    with tempfile.TemporaryDirectory(prefix="nesylink_cnn_dataset_") as tmpdir:
        room_path = Path(tmpdir) / "synthetic_room.json"
        room_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manager = RoomManager(room_path)
        room = manager.get_room(manager.start_room)
        player = PlayerState(position_px=tile_to_top_left_px(room.spawns[room.default_spawn_name]))
        player.facing = player_facing_from_payload(payload)
        apply_player_offset(player, player_offset_px)
        apply_monster_offsets(room, seed)
        write_runtime_pixel_annotations(payload, player, room)
        frame = render_frame(room, player)

    image = frame[:MAP_PIXEL_HEIGHT, :MAP_PIXEL_WIDTH]
    Image.fromarray(image).save(image_path)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def annotate_one(image_path: Path, json_path: Path, out_path: Path, *, labels: bool) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    image = Image.open(image_path).convert("RGB")
    annotated = draw_component_boxes(image, collect_component_boxes(payload), labels=labels)
    annotated.save(out_path)


def player_offset_for_index(index: int) -> tuple[int, int]:
    # Cover almost the full in-tile movement range while keeping the center in the original tile.
    offsets = [(dx, dy) for dy in range(-7, 8) for dx in range(-7, 8)]
    return offsets[index % len(offsets)]


def make_sheet(image_paths: list[Path], out_path: Path, *, cols: int) -> None:
    if not image_paths:
        return
    first = Image.open(image_paths[0]).convert("RGB")
    thumb_w, thumb_h = first.size
    label_h = 14
    rows = (len(image_paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 3, y + 2), path.stem.replace("_annotated", ""), fill=(235, 235, 235))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


if __name__ == "__main__":
    main()
