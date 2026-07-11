from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CNN_DIR = Path(__file__).resolve().parent

from nesylink.core.constants import MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH
from nesylink.core.rendering import render_frame
from nesylink.core.state import PlayerState, tile_to_top_left_px
from nesylink.core.world.loader import load_map
from nesylink.core.world.rooms import RoomManager
from nesylink.cnn.generate_synthetic_scene import (
    apply_monster_offsets,
    apply_player_offset,
    write_runtime_pixel_annotations,
)
from utils.evaluate_policy import SPATIAL_MAP_VARIANTS, materialize_spatial_map_variant


TASK_IDS = tuple(f"mathematical_logic/task_{index}" for index in range(1, 6))
MAP_VARIANTS = ("default", *SPATIAL_MAP_VARIANTS)
COLOR_VARIANTS = ("default", "grayscale", "dark", "bright", "high_contrast", "inverted")
PLAYER_FACINGS = ("down", "up", "left", "right")
PLAYER_OFFSETS = (
    (0, 0),
    (-6, 0),
    (6, 0),
    (0, -6),
    (0, 6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CNN PNG/JSON samples from official evaluation maps and variants."
    )
    parser.add_argument("--out-dir", type=Path, default=CNN_DIR / "generated" / "eval_robust")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_IDS), choices=list(TASK_IDS))
    parser.add_argument("--map-variants", nargs="+", default=list(MAP_VARIANTS), choices=list(MAP_VARIANTS))
    parser.add_argument("--color-variants", nargs="+", default=list(COLOR_VARIANTS), choices=list(COLOR_VARIANTS))
    parser.add_argument("--test-every", type=int, default=10, help="Send every Nth sample to test split.")
    parser.add_argument("--max-spawns-per-room", type=int, default=0, help="0 means use all spawns.")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_every < 2:
        raise ValueError("--test-every must be >= 2")

    train_dir = args.out_dir / "train"
    test_dir = args.out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    train_count = 0
    test_count = 0
    for sample in iter_samples(args):
        if args.limit and total >= args.limit:
            break
        split_dir = test_dir if total % args.test_every == 0 else train_dir
        prefix = "test" if split_dir == test_dir else "train"
        stem = f"{prefix}_{total:05d}_{sample['task_slug']}_{sample['room_id']}_{sample['map_variant']}_{sample['obs_variant']}"
        image_path = split_dir / f"{stem}.png"
        json_path = split_dir / f"{stem}.json"
        Image.fromarray(sample["image"]).save(image_path)
        json_path.write_text(json.dumps(sample["payload"], indent=2) + "\n", encoding="utf-8")
        total += 1
        if split_dir == test_dir:
            test_count += 1
        else:
            train_count += 1
        if total % 500 == 0:
            print(f"generated {total} samples")

    summary = {
        "total": total,
        "train": train_count,
        "test": test_count,
        "tasks": args.tasks,
        "map_variants": args.map_variants,
        "color_variants": args.color_variants,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"generated total={total} train={train_count} test={test_count} out={args.out_dir}")


def iter_samples(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    sample_index = 0
    for task_id in args.tasks:
        task_slug = task_id.replace("/", "_")
        for map_variant in args.map_variants:
            for room_payload in iter_room_payloads(task_id, map_variant):
                room_id = str(room_payload.get("id", "room"))
                for payload in iter_runtime_payloads(room_payload):
                    spawn_names = sorted(payload.get("spawns", {}).keys())
                    if args.max_spawns_per_room > 0:
                        spawn_names = spawn_names[: args.max_spawns_per_room]
                    for rendered in iter_rendered_payloads(
                        payload,
                        spawn_names=spawn_names,
                        sample_index_start=sample_index,
                    ):
                        sample_index = int(rendered["sample_index"]) + 1
                        for obs_variant in args.color_variants:
                            variant_image = apply_obs_variant(rendered["image"], obs_variant)
                            variant_payload = copy.deepcopy(rendered["payload"])
                            annotations = dict(variant_payload.get("annotations", {}))
                            annotations.update(
                                {
                                    "task_id": task_id,
                                    "map_variant": map_variant,
                                    "obs_variant": obs_variant,
                                    "spawn_name": rendered["spawn_name"],
                                    "player_facing": rendered["player_facing"],
                                    "player_offset_px": list(rendered["player_offset_px"]),
                                }
                            )
                            variant_payload["annotations"] = annotations
                            yield {
                                "task_slug": task_slug,
                                "room_id": safe_name(room_id),
                                "map_variant": map_variant,
                                "obs_variant": obs_variant,
                                "payload": variant_payload,
                                "image": variant_image,
                            }


def iter_room_payloads(task_id: str, map_variant: str) -> Iterable[dict[str, Any]]:
    if map_variant == "default":
        map_path = load_map(map_id=task_id)
    else:
        map_path = materialize_spatial_map_variant(task_id, map_variant, seed=0)

    if map_path.name == "dungeon.json":
        dungeon = json.loads(map_path.read_text(encoding="utf-8-sig"))
        for room_file in dungeon.get("room_files", []):
            room_path = map_path.parent / str(room_file)
            yield json.loads(room_path.read_text(encoding="utf-8-sig"))
    else:
        yield json.loads(map_path.read_text(encoding="utf-8-sig"))


def iter_runtime_payloads(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for dynamic_payload in iter_dynamic_state_payloads(payload):
        yield dynamic_payload
        if has_hidden_chest(dynamic_payload):
            visible_payload = copy.deepcopy(dynamic_payload)
            for obj in visible_payload.get("objects", []):
                if isinstance(obj, dict) and str(obj.get("kind")) == "chest" and bool(obj.get("hidden", False)):
                    obj["hidden"] = False
            annotations = dict(visible_payload.get("annotations", {}))
            annotations["hidden_chests_forced_visible"] = True
            visible_payload["annotations"] = annotations
            yield visible_payload


def iter_dynamic_state_payloads(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    dynamic_objects = [obj for obj in payload.get("dynamic_objects", []) if isinstance(obj, dict)]
    if not dynamic_objects:
        yield copy.deepcopy(payload)
        return

    state_options: list[list[str]] = []
    for obj in dynamic_objects:
        states = obj.get("states", {})
        if isinstance(states, dict) and states:
            state_options.append(sorted(str(state_id) for state_id in states))
        else:
            state_options.append([str(obj.get("initial_state", ""))])

    for state_ids in itertools.product(*state_options):
        variant = copy.deepcopy(payload)
        dynamic_state_meta: dict[str, str] = {}
        variant_objects = [obj for obj in variant.get("dynamic_objects", []) if isinstance(obj, dict)]
        for obj, state_id in zip(variant_objects, state_ids, strict=True):
            obj["initial_state"] = state_id
            dynamic_state_meta[str(obj.get("id", "dynamic"))] = state_id
        annotations = dict(variant.get("annotations", {}))
        annotations["dynamic_states"] = dynamic_state_meta
        variant["annotations"] = annotations
        yield variant


def iter_rendered_payloads(
    payload: dict[str, Any],
    *,
    spawn_names: list[str],
    sample_index_start: int,
) -> Iterable[dict[str, Any]]:
    if not spawn_names:
        return

    base_payload = copy.deepcopy(payload)
    make_room_standalone(base_payload, spawn_names[0])

    with tempfile.TemporaryDirectory(prefix="nesylink_eval_cnn_") as tmpdir:
        room_path = Path(tmpdir) / "room.json"
        room_path.write_text(json.dumps(base_payload, indent=2) + "\n", encoding="utf-8")
        manager = RoomManager(room_path)
        room = manager.get_room(manager.start_room)
        monster_positions = {
            monster_id: monster.position_px
            for monster_id, monster in room.monsters.items()
        }

        sample_index = sample_index_start
        for spawn_name in spawn_names:
            for facing in PLAYER_FACINGS:
                for offset in PLAYER_OFFSETS:
                    for monster_id, position_px in monster_positions.items():
                        room.monsters[monster_id].position_px = position_px
                    rendered_payload = copy.deepcopy(base_payload)
                    rendered_payload["default_spawn"] = spawn_name
                    make_room_standalone(rendered_payload, spawn_name)

                    player = PlayerState(position_px=tile_to_top_left_px(room.spawns[spawn_name]))
                    player.facing = facing
                    apply_player_offset(player, offset)
                    apply_monster_offsets(room, sample_index)
                    write_runtime_pixel_annotations(rendered_payload, player, room)
                    frame = render_frame(room, player)

                    yield {
                        "payload": rendered_payload,
                        "image": frame[:MAP_PIXEL_HEIGHT, :MAP_PIXEL_WIDTH],
                        "spawn_name": spawn_name,
                        "player_facing": facing,
                        "player_offset_px": offset,
                        "sample_index": sample_index,
                    }
                    sample_index += 1


def make_room_standalone(payload: dict[str, Any], spawn_name: str) -> None:
    room_id = str(payload.get("id", "room"))
    for exit_cfg in payload.get("exits", []):
        if not isinstance(exit_cfg, dict):
            continue
        exit_cfg["target_room"] = room_id
        exit_cfg["target_entry"] = spawn_name

    dynamic_objects = payload.setdefault("dynamic_objects", [])
    dynamic_ids = {
        str(obj.get("id"))
        for obj in dynamic_objects
        if isinstance(obj, dict) and obj.get("id") is not None
    }
    for obj in payload.get("objects", []):
        if not isinstance(obj, dict) or str(obj.get("kind")) != "switch":
            continue
        effect = obj.get("effect")
        if not isinstance(effect, dict):
            continue
        target = str(effect.get("target", ""))
        if not target or target in dynamic_ids:
            continue
        order = effect.get("order")
        if not isinstance(order, list) or not order:
            continue
        pos = obj.get("pos") if isinstance(obj.get("pos"), list) else [0, 0]
        states = {str(state_id): {"tiles": [pos]} for state_id in order}
        dynamic_objects.append(
            {
                "id": target,
                "kind": "rotating_bridge",
                "initial_state": str(order[0]),
                "background_tile": "none",
                "active_tile": "none",
                "states": states,
            }
        )
        dynamic_ids.add(target)


def apply_obs_variant(image: np.ndarray, variant: str) -> np.ndarray:
    image = np.asarray(image)
    if variant == "default":
        return image
    if variant == "grayscale":
        gray = image.mean(axis=2, keepdims=True).astype(np.uint8)
        return np.repeat(gray, 3, axis=2)
    if variant == "dark":
        return (image.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)
    if variant == "bright":
        return (image.astype(np.float32) * 1.35).clip(0, 255).astype(np.uint8)
    if variant == "high_contrast":
        return np.where(image > 127, 255, 0).astype(np.uint8)
    if variant == "inverted":
        return 255 - image
    raise ValueError(f"unknown color variant: {variant}")


def has_hidden_chest(payload: dict[str, Any]) -> bool:
    return any(
        isinstance(obj, dict) and str(obj.get("kind")) == "chest" and bool(obj.get("hidden", False))
        for obj in payload.get("objects", [])
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


if __name__ == "__main__":
    main()
