from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import (
    ACTION_A,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
)
from nesylink.env import make_env


def repeat(action: int, count: int) -> list[int]:
    return [action] * count


def build_plan() -> list[int]:
    """Reference pixel-action plan for built-in task_1."""
    plan: list[int] = []

    # Start at tile (4, 6). Move around the wall barrier to the key chest.
    plan += repeat(ACTION_RIGHT, 48)  # tile (7, 6)
    plan += repeat(ACTION_UP, 48)  # tile (7, 3)
    plan += repeat(ACTION_LEFT, 96)  # tile (1, 3), adjacent to chest
    plan.append(ACTION_A)

    # Move to the north locked exit after collecting the key.
    plan += repeat(ACTION_RIGHT, 32)  # tile (3, 3)
    plan += repeat(ACTION_UP, 48)  # tile (3, 0)
    plan += repeat(ACTION_RIGHT, 16)  # tile (4, 0)
    plan += repeat(ACTION_UP, 20)  # cross the north exit boundary

    return plan


def run(seed: int = 0) -> dict:
    env = make_env(task_id="mathematical_logic/task_1", observation_mode="pixels")
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    terminated = False
    truncated = False

    try:
        for step_index, action in enumerate(build_plan(), start=1):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "task_id": "task_1",
        "steps": step_index,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "terminal_reason": info.get("terminal_reason"),
        "world_completed": info.get("game", {}).get("world_completed"),
        "final_tile": info.get("agent", {}).get("tile"),
        "obs_shape": tuple(obs.shape),
        "events": info.get("events", {}).get("records", []),
    }


if __name__ == "__main__":
    print(run())
