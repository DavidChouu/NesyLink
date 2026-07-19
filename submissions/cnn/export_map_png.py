from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH
from nesylink.core.rendering import render_frame
from nesylink.core.state import PlayerState, tile_to_top_left_px
from nesylink.core.world.loader import load_map
from nesylink.core.world.rooms import RoomManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a NesyLink map start room as a 160x128 pixel PNG.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--map-id", help="Built-in map id, e.g. mathematical_logic/task_1")
    source.add_argument("--map-path", type=Path, help="Path to a standalone room JSON or dungeon JSON")
    parser.add_argument("--room-id", default=None, help="Optional room id to render instead of the start room")
    parser.add_argument("--spawn", default=None, help="Optional spawn name; defaults to the room default spawn")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--full-frame", action="store_true", help="Save full 160x160 frame including HUD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_path = load_map(map_id=args.map_id, map_path=args.map_path)
    manager = RoomManager(map_path)

    if args.room_id is None:
        room = manager.get_room(manager.start_room)
    else:
        room = manager.get_room(manager.coord_for_room_id(args.room_id))

    spawn_name = args.spawn or room.default_spawn_name
    player = PlayerState(position_px=tile_to_top_left_px(room.spawns[spawn_name]))
    frame = render_frame(room, player)
    image = frame if args.full_frame else frame[:MAP_PIXEL_HEIGHT, :MAP_PIXEL_WIDTH]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.out)
    print(f"saved {args.out} shape={image.shape} map={map_path} room={room.room_id} spawn={spawn_name}")


if __name__ == "__main__":
    main()
