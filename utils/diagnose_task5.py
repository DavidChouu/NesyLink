"""Run one CNN-only Task5 episode and print room-level failure diagnostics.

The policy receives only pixels and safe_info.  Raw events are inspected only
after each environment step by this local diagnostic runner.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.env import make_env
from docs.Mathematical_logic.examples.task5_fsm_bfs_agent import make_policy
from utils.evaluate_policy import build_policy_info, materialize_spatial_map_variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("default", "spatial_a", "spatial_b", "spatial_c"), default="default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.variant == "default":
        env = make_env(task_id="mathematical_logic/task_5", observation_mode="pixels")
    else:
        map_path = materialize_spatial_map_variant("mathematical_logic/task_5", args.variant, seed=args.seed)
        env = make_env(task_id="mathematical_logic/task_5", map_path=map_path, observation_mode="pixels")

    policy = make_policy()
    policy.reset(seed=args.seed, task_id="mathematical_logic/task_5")
    obs, raw_info = env.reset(seed=args.seed)
    last_reward = 0.0
    rooms: Counter[str] = Counter()
    goals: Counter[str] = Counter()
    events: Counter[str] = Counter()
    terminated = truncated = False
    steps = 0
    try:
        while not (terminated or truncated):
            safe_info = build_policy_info(
                info_mode="safe",
                raw_info=raw_info,
                last_reward=last_reward,
                task_id="mathematical_logic/task_5",
            )
            action = policy.act(obs, safe_info)
            goal = policy.current_goal
            goals["none" if goal is None else goal.kind] += 1
            obs, reward, terminated, truncated, raw_info = env.step(action)
            rooms[str(policy.current_room)] += 1
            for record in raw_info.get("events", {}).get("records", []):
                if isinstance(record, dict) and record.get("name"):
                    events[str(record["name"])] += 1
            last_reward = float(reward)
            steps += 1
    finally:
        env.close()

    print(f"variant={args.variant} seed={args.seed} steps={steps} reason={raw_info.get('terminal_reason')}")
    print("room_steps:", dict(sorted(rooms.items())))
    print("goal_steps:", dict(sorted(goals.items())))
    print("events:", dict(sorted(events.items())))
    for room, memory in sorted(policy.rooms.items()):
        print(
            f"room={room} opened={sorted(memory.opened_chests)} "
            f"seen_chests={sorted(memory.remembered_chests)} "
            f"blocked={memory.blocked_move_count} risk={memory.damage_risk_count} "
            f"edges={{{', '.join(f'{d}:{s.estimated_cost():.1f}' for d, s in memory.exit_stats.items())}}}"
        )


if __name__ == "__main__":
    main()
