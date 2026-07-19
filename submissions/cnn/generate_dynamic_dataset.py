from __future__ import annotations

import argparse
import copy
import json
import random
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

from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH, MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH
from nesylink.core.rendering import render_frame
from nesylink.core.state import PlayerState, tile_to_top_left_px
from nesylink.core.world.rooms import RoomManager
from nesylink.cnn.generate_eval_dataset import (
    COLOR_VARIANTS,
    MAP_VARIANTS,
    TASK_IDS,
    apply_runtime_state_annotations,
    apply_obs_variant,
    iter_room_payloads,
    iter_runtime_payloads,
    make_room_standalone,
    safe_annotations,
    safe_name,
)
from nesylink.cnn.generate_synthetic_scene import write_runtime_pixel_annotations


PLAYER_FACINGS = ("down", "up", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hard dynamic CNN samples with random player/monster pixel positions."
    )
    parser.add_argument("--out-dir", type=Path, default=CNN_DIR / "generated" / "eval_robust_v1" / "train")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_IDS), choices=list(TASK_IDS))
    parser.add_argument("--map-variants", nargs="+", default=list(MAP_VARIANTS), choices=list(MAP_VARIANTS))
    parser.add_argument("--color-variants", nargs="+", default=list(COLOR_VARIANTS), choices=list(COLOR_VARIANTS))
    parser.add_argument("--samples-per-room-state", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for sample in iter_samples(args, rng):
        if args.limit and total >= args.limit:
            break
        stem = (
            f"dynamic_{total:06d}_{sample['task_slug']}_{sample['room_id']}_"
            f"{sample['map_variant']}_{sample['obs_variant']}"
        )
        Image.fromarray(sample["image"]).save(args.out_dir / f"{stem}.png")
        (args.out_dir / f"{stem}.json").write_text(
            json.dumps(sample["payload"], indent=2) + "\n",
            encoding="utf-8",
        )
        total += 1
        if total % 500 == 0:
            print(f"generated {total} dynamic samples")

    print(f"generated dynamic total={total} out={args.out_dir}")


def iter_samples(args: argparse.Namespace, rng: random.Random) -> Iterable[dict[str, Any]]:
    for task_id in args.tasks:
        task_slug = task_id.replace("/", "_")
        for map_variant in args.map_variants:
            for room_payload in iter_room_payloads(task_id, map_variant):
                room_id = str(room_payload.get("id", "room"))
                for payload in iter_runtime_payloads(room_payload):
                    for rendered in iter_rendered_dynamic_payloads(
                        payload,
                        rng=rng,
                        samples_per_room_state=args.samples_per_room_state,
                    ):
                        for obs_variant in args.color_variants:
                            variant_payload = copy.deepcopy(rendered["payload"])
                            annotations = safe_annotations(variant_payload)
                            annotations.update(
                                {
                                    "task_id": task_id,
                                    "map_variant": map_variant,
                                    "obs_variant": obs_variant,
                                    "player_facing": rendered["player_facing"],
                                    "dynamic_sample": True,
                                }
                            )
                            variant_payload["annotations"] = annotations
                            yield {
                                "task_slug": task_slug,
                                "room_id": safe_name(room_id),
                                "map_variant": map_variant,
                                "obs_variant": obs_variant,
                                "payload": variant_payload,
                                "image": apply_obs_variant(rendered["image"], obs_variant),
                            }


def iter_rendered_dynamic_payloads(
    payload: dict[str, Any],
    *,
    rng: random.Random,
    samples_per_room_state: int,
) -> Iterable[dict[str, Any]]:
    base_payload = copy.deepcopy(payload)
    spawn_names = sorted(base_payload.get("spawns", {}).keys())
    spawn_name = spawn_names[0] if spawn_names else "default"
    make_room_standalone(base_payload, spawn_name)

    with tempfile.TemporaryDirectory(prefix="nesylink_dynamic_cnn_") as tmpdir:
        room_path = Path(tmpdir) / "room.json"
        room_path.write_text(json.dumps(base_payload, indent=2) + "\n", encoding="utf-8")
        manager = RoomManager(room_path)
        room = manager.get_room(manager.start_room)
        apply_runtime_state_annotations(room, base_payload)

        walkable_tiles = sorted(runtime_walkable_tiles(room))
        if not walkable_tiles:
            return

        monster_ids = sorted(room.monsters)
        for sample_index in range(samples_per_room_state):
            player_tile = rng.choice(walkable_tiles)
            player = PlayerState(position_px=random_position_in_tile(rng, player_tile, 16))
            player.facing = rng.choice(PLAYER_FACINGS)

            for monster_index, monster_id in enumerate(monster_ids):
                monster = room.monsters[monster_id]
                if rng.random() < 0.70:
                    monster.position_px = random_position_near(
                        rng,
                        player.position_px,
                        int(monster.size_px),
                        radius_px=22 + monster_index * 4,
                    )
                else:
                    monster_tile = rng.choice(walkable_tiles)
                    monster.position_px = random_position_in_tile(rng, monster_tile, int(monster.size_px))

            rendered_payload = copy.deepcopy(base_payload)
            annotations = safe_annotations(rendered_payload)
            annotations["tile_labels"] = tile_labels_from_runtime_room(room)
            rendered_payload["annotations"] = annotations
            write_runtime_pixel_annotations(rendered_payload, player, room)
            frame = render_frame(room, player)

            yield {
                "payload": rendered_payload,
                "image": frame[:MAP_PIXEL_HEIGHT, :MAP_PIXEL_WIDTH],
                "player_facing": player.facing,
                "sample_index": sample_index,
            }


def runtime_walkable_tiles(room: Any) -> set[tuple[int, int]]:
    blocked = set(room.walls)
    blocked.update(pos for pos, kind in room.dynamic_tiles.items() if kind == "gap")
    blocked.update(chest.pos for chest in room.chests.values() if chest.is_visible)
    blocked.update(npc.pos for npc in room.npcs.values())
    return {
        (x, y)
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
        if (x, y) not in blocked
    }


def random_position_in_tile(rng: random.Random, tile: tuple[int, int], size_px: int) -> tuple[float, float]:
    left, top = tile_to_top_left_px(tile)
    jitter_x = rng.randint(-7, 7)
    jitter_y = rng.randint(-7, 7)
    return clamp_position((left + jitter_x, top + jitter_y), size_px)


def random_position_near(
    rng: random.Random,
    position: tuple[float, float],
    size_px: int,
    *,
    radius_px: int,
) -> tuple[float, float]:
    dx = rng.randint(-radius_px, radius_px)
    dy = rng.randint(-radius_px, radius_px)
    return clamp_position((position[0] + dx, position[1] + dy), size_px)


def clamp_position(position: tuple[float, float], size_px: int) -> tuple[float, float]:
    x = min(max(0.0, float(position[0])), float(MAP_PIXEL_WIDTH - size_px))
    y = min(max(0.0, float(position[1])), float(MAP_PIXEL_HEIGHT - size_px))
    return (x, y)


def tile_labels_from_runtime_room(room: Any) -> list[list[str]]:
    labels = [["floor" for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    for (x, y), tile_kind in room.dynamic_tiles.items():
        if tile_kind in {"gap", "bridge"}:
            labels[y][x] = tile_kind

    for exit_config in room.exits:
        exit_kind = exit_class_from_runtime_exit(exit_config)
        for x, y in exit_config.tiles:
            labels[y][x] = exit_kind

    for x, y in room.walls:
        labels[y][x] = "wall"

    for trap in room.traps.values():
        if not trap.is_active:
            continue
        x, y = trap.pos
        if room.dynamic_tiles.get(trap.pos) == "bridge":
            continue
        labels[y][x] = "abyss" if str(trap.trap_type).lower() == "abyss" else "trap"

    for chest in room.chests.values():
        if chest.is_visible:
            labels[chest.pos[1]][chest.pos[0]] = "chest"

    for npc in room.npcs.values():
        labels[npc.pos[1]][npc.pos[0]] = "npc"

    for button in room.buttons.values():
        labels[button.pos[1]][button.pos[0]] = "button_pressed" if button.is_pressed else "button"

    for switch in room.switches.values():
        labels[switch.pos[1]][switch.pos[0]] = "switch"

    return labels


def exit_class_from_runtime_exit(exit_config: Any) -> str:
    if exit_config.exit_type == "conditional":
        return "exit_conditional"
    if exit_config.exit_type == "locked_key" or "key_count" in exit_config.requires:
        return "exit_locked"
    return "exit_normal"


if __name__ == "__main__":
    main()
