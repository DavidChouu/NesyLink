from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CNN_DIR = Path(__file__).resolve().parent

from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH, MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH
from nesylink.core.rendering import render_frame
from nesylink.core.state import PlayerState, tile_to_top_left_px
from nesylink.core.world.rooms import RoomManager


EDGE_EXIT_TILES = {
    "north": {(4, 0), (5, 0)},
    "south": {(4, GRID_HEIGHT - 1), (5, GRID_HEIGHT - 1)},
    "west": {(0, 3), (0, 4)},
    "east": {(GRID_WIDTH - 1, 3), (GRID_WIDTH - 1, 4)},
}

PLAYER_CONTEXTS = (
    "floor",
    "bridge",
    "gap",
    "spike_trap",
    "abyss_trap",
    "exit_tile",
    "exit_adjacent",
)
PLAYER_FACINGS = ("down", "up", "left", "right")
SCENE_STYLES = ("mixed", "abyss_bridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one synthetic NesyLink scene PNG for CNN experiments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=CNN_DIR / "generated" / "synthetic_seed0.png")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--full-frame", action="store_true")
    parser.add_argument(
        "--player-offset-px",
        nargs=2,
        type=int,
        metavar=("DX", "DY"),
        default=(0, 0),
        help="Move the rendered player by pixel offsets after room construction.",
    )
    parser.add_argument("--player-context", choices=PLAYER_CONTEXTS, default=None)
    parser.add_argument("--player-facing", choices=PLAYER_FACINGS, default=None)
    parser.add_argument("--scene-style", choices=SCENE_STYLES, default="mixed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    payload = build_synthetic_room(
        rng,
        player_context=args.player_context,
        player_facing=args.player_facing,
        scene_style=args.scene_style,
    )

    with tempfile.TemporaryDirectory(prefix="nesylink_cnn_scene_") as tmpdir:
        room_path = Path(tmpdir) / "synthetic_room.json"
        room_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manager = RoomManager(room_path)
        room = manager.get_room(manager.start_room)
        player = PlayerState(position_px=tile_to_top_left_px(room.spawns[room.default_spawn_name]))
        player.facing = player_facing_from_payload(payload)
        apply_player_offset(player, tuple(args.player_offset_px))
        apply_monster_offsets(room, args.seed)
        write_runtime_pixel_annotations(payload, player, room)
        frame = render_frame(room, player)

    image = frame if args.full_frame else frame[:MAP_PIXEL_HEIGHT, :MAP_PIXEL_WIDTH]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.out)

    json_out = args.json_out
    if json_out is None:
        json_out = args.out.with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"saved {args.out} shape={image.shape}")
    print(f"saved {json_out}")


def apply_player_offset(player: PlayerState, offset_px: tuple[int, int]) -> None:
    dx, dy = offset_px
    x, y = player.position_px
    max_x = MAP_PIXEL_WIDTH - player.size_px
    max_y = MAP_PIXEL_HEIGHT - player.size_px
    player.position_px = (
        float(max(0, min(max_x, x + dx))),
        float(max(0, min(max_y, y + dy))),
    )


def apply_monster_offsets(room: Any, index: int) -> None:
    for monster_index, monster in enumerate(room.monsters.values()):
        dx, dy = monster_offset_for_index(index + monster_index * 13)
        x, y = monster.position_px
        max_x = MAP_PIXEL_WIDTH - monster.size_px
        max_y = MAP_PIXEL_HEIGHT - monster.size_px
        monster.position_px = (
            float(max(0, min(max_x, x + dx))),
            float(max(0, min(max_y, y + dy))),
        )


def monster_offset_for_index(index: int) -> tuple[int, int]:
    offsets = (
        (0, 0),
        (-6, 0),
        (6, 0),
        (0, -6),
        (0, 6),
        (-4, -4),
        (4, -4),
        (-4, 4),
        (4, 4),
        (-7, 2),
        (7, -2),
        (2, 7),
        (-2, -7),
    )
    return offsets[index % len(offsets)]


def write_player_pixel_annotation(payload: dict[str, Any], player: PlayerState) -> None:
    write_runtime_pixel_annotations(payload, player)


def write_runtime_pixel_annotations(payload: dict[str, Any], player: PlayerState, room: Any | None = None) -> None:
    boxes = [dynamic_pixel_box("player", "player_px", player.position_px, int(player.size_px))]
    if room is not None:
        for monster in room.monsters.values():
            boxes.append(
                dynamic_pixel_box(
                    "monster",
                    f"monster_{monster.monster_type}",
                    monster.position_px,
                    int(monster.size_px),
                )
            )

    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict):
        annotations = {}
    else:
        annotations = dict(annotations)
    annotations["pixel_boxes"] = boxes
    payload["annotations"] = annotations


def dynamic_pixel_box(kind: str, label: str, position_px: tuple[float, float], size_px: int) -> dict[str, Any]:
    left = int(round(position_px[0]))
    top = int(round(position_px[1]))
    right = left + int(size_px)
    bottom = top + int(size_px)
    return {
        "kind": kind,
        "label": label,
        "bbox_px": [left, top, right, bottom],
        "center_px": [(left + right) * 0.5, (top + bottom) * 0.5],
    }


def build_synthetic_room(
    rng: random.Random,
    *,
    player_context: str | None = None,
    player_facing: str | None = None,
    scene_style: str = "mixed",
) -> dict[str, Any]:
    if scene_style == "abyss_bridge":
        return build_abyss_bridge_room(rng, player_context=player_context, player_facing=player_facing)

    blocked: set[tuple[int, int]] = set()
    layout = [["." for _x in range(GRID_WIDTH)] for _y in range(GRID_HEIGHT)]

    exit_direction = rng.choice(["north", "south", "west", "east"])
    reserved = set(EDGE_EXIT_TILES[exit_direction])

    dynamic_objects, dynamic_tiles = build_bridge_if_possible(rng, reserved)
    blocked.update(dynamic_tiles)

    context = choose_player_context(rng, player_context, dynamic_objects)
    player_pos = choose_player_tile(rng, context, blocked, reserved, exit_direction, dynamic_objects)
    blocked.add(player_pos)
    facing = choose_player_facing(rng, player_facing)

    wall_count = rng.randint(6, 12)
    for _ in range(wall_count):
        pos = choose_free_tile(rng, blocked | reserved, margin=False)
        layout[pos[1]][pos[0]] = "#"
        blocked.add(pos)

    objects: list[dict[str, Any]] = []
    add_player_background_object(objects, context, player_pos)
    occupied = set(blocked) | reserved
    add_chests(rng, objects, occupied, count=rng.randint(1, 3))
    add_npcs(rng, objects, occupied, count=rng.randint(1, 2))
    add_monsters(rng, objects, occupied, count=rng.randint(1, 3))
    add_traps(rng, objects, occupied, count=rng.randint(2, 5))
    add_buttons_and_switches(rng, objects, occupied, enabled=bool(dynamic_objects))

    exit_type = rng.choice(["normal", "locked_key", "conditional"])
    exit_config: dict[str, Any] = {
        "id": f"{exit_direction}_exit",
        "direction": exit_direction,
        "target_room": "synthetic_room",
        "target_entry": "default",
        "type": exit_type,
        "blocked_message": "NEED KEY",
        "success_message": "SYNTHETIC EXIT",
        "complete_task": True,
    }
    if exit_type == "locked_key":
        exit_config["requires"] = {"key_count": 1, "consume_key": False}
    elif exit_type == "conditional":
        exit_config["requires"] = {"all_monsters_defeated": True}
        exit_config["blocked_message"] = "CONDITION LOCKED"

    return {
        "id": "synthetic_room",
        "coord": [0, 0],
        "layout": ["".join(row) for row in layout],
        "spawns": {"default": [player_pos[0], player_pos[1]]},
        "default_spawn": "default",
        "objects": objects,
        "dynamic_objects": dynamic_objects,
        "exits": [exit_config],
        "annotations": {
            "scene_style": "mixed",
            "player_context": context,
            "player_facing": facing,
            "player_tile": [player_pos[0], player_pos[1]],
        },
    }


def player_context_for_index(index: int) -> str:
    return PLAYER_CONTEXTS[(index // len(PLAYER_FACINGS)) % len(PLAYER_CONTEXTS)]


def player_facing_for_index(index: int) -> str:
    return PLAYER_FACINGS[index % len(PLAYER_FACINGS)]


def scene_style_for_index(index: int) -> str:
    # Half of the data uses Task4-center-like abyss rooms to teach long bridges over abyss.
    return "abyss_bridge" if index % 2 == 0 else "mixed"


def player_facing_from_payload(payload: dict[str, Any]) -> str:
    annotations = payload.get("annotations", {})
    if isinstance(annotations, dict):
        facing = annotations.get("player_facing")
        if facing in PLAYER_FACINGS:
            return str(facing)
    return "down"


def choose_player_context(
    rng: random.Random,
    requested: str | None,
    dynamic_objects: list[dict[str, Any]],
) -> str:
    if requested in PLAYER_CONTEXTS:
        context = str(requested)
    else:
        context = rng.choice(PLAYER_CONTEXTS)
    if context in {"bridge", "gap"} and not dynamic_objects:
        return "floor"
    return context


def choose_player_facing(rng: random.Random, requested: str | None) -> str:
    if requested in PLAYER_FACINGS:
        return str(requested)
    return rng.choice(PLAYER_FACINGS)


def choose_player_tile(
    rng: random.Random,
    context: str,
    blocked: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    exit_direction: str,
    dynamic_objects: list[dict[str, Any]],
) -> tuple[int, int]:
    if context == "bridge":
        candidates = sorted(active_bridge_tiles(dynamic_objects) - reserved)
        if candidates:
            return rng.choice(candidates)
    if context == "gap":
        candidates = sorted(all_dynamic_tiles(dynamic_objects) - active_bridge_tiles(dynamic_objects) - reserved)
        if candidates:
            return rng.choice(candidates)
    if context == "exit_tile":
        return rng.choice(sorted(reserved))
    if context == "exit_adjacent":
        candidates = sorted(exit_approach_tiles(exit_direction) - blocked - reserved)
        if candidates:
            return rng.choice(candidates)
    return choose_free_tile(rng, blocked | reserved, margin=True)


def active_bridge_tiles(dynamic_objects: list[dict[str, Any]]) -> set[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    for dynamic_object in dynamic_objects:
        initial_state = str(dynamic_object.get("initial_state", ""))
        state = dynamic_object.get("states", {}).get(initial_state, {})
        for x, y in state.get("tiles", []):
            tiles.add((int(x), int(y)))
    return tiles


def all_dynamic_tiles(dynamic_objects: list[dict[str, Any]]) -> set[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    for dynamic_object in dynamic_objects:
        for state in dynamic_object.get("states", {}).values():
            for x, y in state.get("tiles", []):
                tiles.add((int(x), int(y)))
    return tiles


def exit_approach_tiles(direction: str) -> set[tuple[int, int]]:
    if direction == "north":
        return {(4, 1), (5, 1)}
    if direction == "south":
        return {(4, GRID_HEIGHT - 2), (5, GRID_HEIGHT - 2)}
    if direction == "west":
        return {(1, 3), (1, 4)}
    if direction == "east":
        return {(GRID_WIDTH - 2, 3), (GRID_WIDTH - 2, 4)}
    return set()


def add_player_background_object(
    objects: list[dict[str, Any]],
    context: str,
    player_pos: tuple[int, int],
) -> None:
    if context not in {"spike_trap", "abyss_trap"}:
        return
    objects.append(
        {
            "id": "player_background_trap",
            "kind": "trap",
            "pos": [player_pos[0], player_pos[1]],
            "trap_type": "spike" if context == "spike_trap" else "abyss",
            "damage": 1,
            "respawn_to": "default",
        }
    )


def build_abyss_bridge_room(
    rng: random.Random,
    *,
    player_context: str | None = None,
    player_facing: str | None = None,
) -> dict[str, Any]:
    states = abyss_bridge_states()
    initial_state = rng.choice(list(states))
    active_tiles = set(states[initial_state])
    context = choose_abyss_player_context(rng, player_context)
    player_pos = choose_abyss_player_tile(rng, context, active_tiles)
    facing = choose_player_facing(rng, player_facing)

    return {
        "id": "synthetic_abyss_bridge",
        "coord": [0, 0],
        "layout": ["." * GRID_WIDTH for _ in range(GRID_HEIGHT)],
        "spawns": {
            "default": [player_pos[0], player_pos[1]],
            "west_door": [1, 4],
            "east_door": [8, 4],
            "from_north": [4, 1],
            "from_south": [4, 6],
        },
        "default_spawn": "default",
        "objects": [
            {
                "id": "full_abyss",
                "kind": "trap",
                "trap_type": "abyss",
                "damage": 1,
                "respawn_delay_steps": 2,
                "rects": [{"from": [0, 0], "to": [GRID_WIDTH - 1, GRID_HEIGHT - 1]}],
            },
            {
                "id": "hidden_bridge_chest",
                "kind": "chest",
                "pos": [4, 4],
                "hidden": True,
                "loot": {"kind": "gold", "amount": 1},
            },
        ],
        "dynamic_objects": [
            {
                "id": "center_bridge",
                "kind": "rotating_bridge",
                "initial_state": initial_state,
                "background_tile": "none",
                "active_tile": "bridge",
                "states": {
                    state_id: {"tiles": [[x, y] for x, y in tiles]}
                    for state_id, tiles in states.items()
                },
            }
        ],
        "exits": [
            {
                "id": "west_exit",
                "direction": "west",
                "target_room": "synthetic_abyss_bridge",
                "target_entry": "east_door",
                "type": "normal",
                "success_message": "WEST",
            },
            {
                "id": "east_exit",
                "direction": "east",
                "target_room": "synthetic_abyss_bridge",
                "target_entry": "west_door",
                "type": "locked_key",
                "requires": {"key_count": 1, "consume_key": False},
                "blocked_message": "NEED KEY",
                "success_message": "SWORD ROOM",
            },
            {
                "id": "north_exit",
                "direction": "north",
                "target_room": "synthetic_abyss_bridge",
                "target_entry": "from_south",
                "type": "normal",
                "success_message": "KEY ROOM",
            },
            {
                "id": "south_exit",
                "direction": "south",
                "target_room": "synthetic_abyss_bridge",
                "target_entry": "from_north",
                "type": "normal",
                "success_message": "MONSTER ROOM",
            },
        ],
        "annotations": {
            "scene_style": "abyss_bridge",
            "player_context": context,
            "player_facing": facing,
            "player_tile": [player_pos[0], player_pos[1]],
            "active_bridge_state": initial_state,
        },
    }


def abyss_bridge_states() -> dict[str, tuple[tuple[int, int], ...]]:
    def hrow(x0: int, x1: int) -> set[tuple[int, int]]:
        return {(x, y) for x in range(x0, x1 + 1) for y in (3, 4)}

    def vcol(y0: int, y1: int) -> set[tuple[int, int]]:
        return {(x, y) for y in range(y0, y1 + 1) for x in (4, 5)}

    raw_states = {
        "west_to_north": hrow(0, 5) | vcol(0, 4),
        "west_to_east": hrow(0, GRID_WIDTH - 1),
        "west_to_south": hrow(0, 5) | vcol(3, GRID_HEIGHT - 1),
        "east_to_north": hrow(4, GRID_WIDTH - 1) | vcol(0, 4),
        "east_to_south": hrow(4, GRID_WIDTH - 1) | vcol(3, GRID_HEIGHT - 1),
        "north_to_south": vcol(0, GRID_HEIGHT - 1),
    }
    return {name: tuple(sorted(tiles)) for name, tiles in raw_states.items()}


def choose_abyss_player_context(rng: random.Random, requested: str | None) -> str:
    if requested in {"bridge", "exit_tile", "exit_adjacent", "abyss_trap"}:
        return str(requested)
    if requested in {"gap", "spike_trap"}:
        return "abyss_trap"
    if requested == "floor":
        return "bridge"
    return rng.choice(["bridge", "bridge", "bridge", "exit_tile", "exit_adjacent", "abyss_trap"])


def choose_abyss_player_tile(
    rng: random.Random,
    context: str,
    active_tiles: set[tuple[int, int]],
) -> tuple[int, int]:
    all_tiles = {(x, y) for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH)}
    if context == "exit_tile":
        candidates = sorted(set().union(*EDGE_EXIT_TILES.values()) & active_tiles)
        if candidates:
            return rng.choice(candidates)
    if context == "exit_adjacent":
        candidates = sorted(set().union(*(exit_approach_tiles(direction) for direction in EDGE_EXIT_TILES)) & active_tiles)
        if candidates:
            return rng.choice(candidates)
    if context == "abyss_trap":
        candidates = sorted(all_tiles - active_tiles - set().union(*EDGE_EXIT_TILES.values()))
        if candidates:
            return rng.choice(candidates)
    candidates = sorted(active_tiles)
    if not candidates:
        raise RuntimeError("abyss bridge room has no active bridge tiles")
    return rng.choice(candidates)


def add_chests(
    rng: random.Random,
    objects: list[dict[str, Any]],
    blocked: set[tuple[int, int]],
    *,
    count: int,
) -> None:
    loot_kinds = [
        {"kind": "key", "key_id": "synthetic_key"},
        {"kind": "gold", "amount": 1},
        {"kind": "heal", "amount": 1},
    ]
    for index in range(count):
        pos = choose_free_tile(rng, blocked, margin=False)
        blocked.add(pos)
        objects.append(
            {
                "id": f"chest_{index}",
                "kind": "chest",
                "pos": [pos[0], pos[1]],
                "loot": rng.choice(loot_kinds),
            }
        )



def add_npcs(
    rng: random.Random,
    objects: list[dict[str, Any]],
    blocked: set[tuple[int, int]],
    *,
    count: int,
) -> None:
    for index in range(count):
        pos = choose_free_tile(rng, blocked, margin=True)
        blocked.add(pos)
        objects.append(
            {
                "id": f"npc_{index}",
                "kind": "npc",
                "pos": [pos[0], pos[1]],
                "text": rng.choice(["HELLO", "GUIDE", "TRADE", "CLUE"]),
            }
        )

def add_monsters(
    rng: random.Random,
    objects: list[dict[str, Any]],
    blocked: set[tuple[int, int]],
    *,
    count: int,
) -> None:
    monster_types = ["chaser", "ambusher", "patroller"]
    for index in range(count):
        pos = choose_free_tile(rng, blocked, margin=True)
        blocked.add(pos)
        objects.append(
            {
                "id": f"monster_{index}",
                "kind": "monster",
                "pos": [pos[0], pos[1]],
                "monster_type": rng.choice(monster_types),
                "hp": rng.randint(1, 3),
                "damage": 1,
            }
        )


def add_traps(
    rng: random.Random,
    objects: list[dict[str, Any]],
    blocked: set[tuple[int, int]],
    *,
    count: int,
) -> None:
    for index in range(count):
        pos = choose_free_tile(rng, blocked, margin=False)
        blocked.add(pos)
        trap_type = rng.choice(["spike", "abyss"])
        objects.append(
            {
                "id": f"trap_{index}",
                "kind": "trap",
                "pos": [pos[0], pos[1]],
                "trap_type": trap_type,
                "damage": 1,
                "respawn_to": "default",
            }
        )


def add_buttons_and_switches(
    rng: random.Random,
    objects: list[dict[str, Any]],
    blocked: set[tuple[int, int]],
    *,
    enabled: bool,
) -> None:
    button_pos = choose_free_tile(rng, blocked, margin=True)
    blocked.add(button_pos)
    objects.append(
        {
            "id": "button_0",
            "kind": "button",
            "pos": [button_pos[0], button_pos[1]],
            "message": "SYNTH BUTTON",
        }
    )

    if not enabled:
        return

    switch_pos = choose_free_tile(rng, blocked, margin=True)
    blocked.add(switch_pos)
    objects.append(
        {
            "id": "switch_0",
            "kind": "switch",
            "pos": [switch_pos[0], switch_pos[1]],
            "message": "SYNTH SWITCH",
            "effect": {
                "type": "cycle_state",
                "target": "bridge_0",
                "order": ["horizontal", "vertical"],
            },
        }
    )


def build_bridge_if_possible(
    rng: random.Random,
    blocked: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    candidates = [
        (((3, 3), (4, 3), (5, 3)), ((4, 2), (4, 3), (4, 4))),
        (((2, 4), (3, 4), (4, 4)), ((3, 3), (3, 4), (3, 5))),
        (((5, 3), (6, 3), (7, 3)), ((6, 2), (6, 3), (6, 4))),
    ]
    rng.shuffle(candidates)
    for horizontal_tiles, vertical_tiles in candidates:
        all_tiles = set(horizontal_tiles) | set(vertical_tiles)
        if all(tile not in blocked for tile in all_tiles):
            return (
                [
                    {
                        "id": "bridge_0",
                        "kind": "rotating_bridge",
                        "initial_state": "horizontal",
                        "background_tile": "gap",
                        "active_tile": "bridge",
                        "states": {
                            "horizontal": {"tiles": [[x, y] for x, y in horizontal_tiles]},
                            "vertical": {"tiles": [[x, y] for x, y in vertical_tiles]},
                        },
                    }
                ],
                all_tiles,
            )
    return [], set()


def choose_free_tile(
    rng: random.Random,
    blocked: set[tuple[int, int]],
    *,
    margin: bool,
) -> tuple[int, int]:
    xs = range(1, GRID_WIDTH - 1) if margin else range(GRID_WIDTH)
    ys = range(1, GRID_HEIGHT - 1) if margin else range(GRID_HEIGHT)
    candidates = [(x, y) for y in ys for x in xs if (x, y) not in blocked]
    if not candidates:
        raise RuntimeError("no free tile available")
    return rng.choice(candidates)


if __name__ == "__main__":
    main()
