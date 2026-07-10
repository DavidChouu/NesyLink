from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.env import make_env
from nesylink.vision import classify_frame


SYMBOLS = {
    "floor": "..",
    "wall": "##",
    "player": "PP",
    "monster": "MM",
    "chest": "CC",
    "trap": "^^",
    "abyss": "OO",
    "button": "BT",
    "switch": "SW",
    "gap": "GG",
    "bridge": "BR",
    "exit_normal": "EN",
    "exit_locked": "EL",
    "exit_conditional": "EC",
    "unknown": "??",
}


def print_symbol_grid(grid: tuple[tuple[str, ...], ...]) -> None:
    for row in grid:
        print(" ".join(SYMBOLS.get(kind, "??") for kind in row))


def main() -> None:
    for index in range(1, 6):
        task_id = f"mathematical_logic/task_{index}"
        env = make_env(task_id=task_id, observation_mode="pixels")
        obs, _info = env.reset(seed=0)
        vision = classify_frame(obs)
        env.close()

        print(f"\n{task_id}")
        print(f"player: {None if vision.player is None else vision.player.tile}")
        print(f"monsters: {[monster.tile for monster in vision.monsters]}")
        print_symbol_grid(vision.grid)


if __name__ == "__main__":
    main()
