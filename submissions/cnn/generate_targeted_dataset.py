from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CNN_DIR = Path(__file__).resolve().parent

from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH, MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH, TILE_SIZE
from nesylink.cnn.generate_dynamic_dataset import tile_labels_from_runtime_room
from nesylink.cnn.generate_eval_dataset import apply_obs_variant
from nesylink.cnn.generate_synthetic_scene import write_runtime_pixel_annotations
from nesylink.env import make_env
from utils.evaluate_policy import (
    build_policy_info,
    event_names,
    materialize_spatial_map_variant,
)


COLOR_VARIANTS = ("default", "grayscale", "dark", "bright")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate targeted CNN samples from runtime states that broke Task2/Task4 robustness."
    )
    parser.add_argument("--out-dir", type=Path, default=CNN_DIR / "generated" / "targeted_v1" / "train")
    parser.add_argument("--max-per-case", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--color-variants", nargs="+", default=list(COLOR_VARIANTS), choices=list(COLOR_VARIANTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        TargetCase(
            name="task2_default_after_kill_open_chest",
            task_id="mathematical_logic/task_2",
            policy_path=PROJECT_ROOT / "submissions/examples/task2_agent.py",
            seed=args.seed,
            map_variant="default",
            max_steps=500,
            selector=select_task2_after_progress,
            color_variants=tuple(args.color_variants),
        ),
        TargetCase(
            name="task2_spatial_a_exit_alignment",
            task_id="mathematical_logic/task_2",
            policy_path=PROJECT_ROOT / "submissions/examples/task2_agent.py",
            seed=args.seed,
            map_variant="spatial_a",
            max_steps=500,
            selector=select_task2_after_progress,
            color_variants=("default",),
        ),
        TargetCase(
            name="task4_spatial_c_final_chest",
            task_id="mathematical_logic/task_4",
            policy_path=PROJECT_ROOT / "submissions/examples/task4_agent.py",
            seed=args.seed + 2,
            map_variant="spatial_c",
            max_steps=2000,
            selector=select_task4_final_chest,
            color_variants=tuple(args.color_variants),
        ),
    ]

    total = 0
    summary: dict[str, Any] = {"cases": {}, "total": 0, "out_dir": str(args.out_dir)}
    for case in cases:
        written = generate_case(case, args.out_dir, args.max_per_case, start_index=total)
        total += written
        summary["cases"][case.name] = written
        print(f"{case.name}: wrote {written}")

    summary["total"] = total
    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"generated targeted total={total} out={args.out_dir}")


class TargetCase:
    def __init__(
        self,
        *,
        name: str,
        task_id: str,
        policy_path: Path,
        seed: int,
        map_variant: str,
        max_steps: int,
        selector: Callable[[RuntimeSample], bool],
        color_variants: tuple[str, ...],
    ) -> None:
        self.name = name
        self.task_id = task_id
        self.policy_path = policy_path
        self.seed = seed
        self.map_variant = map_variant
        self.max_steps = max_steps
        self.selector = selector
        self.color_variants = color_variants


class RuntimeSample:
    def __init__(
        self,
        *,
        env: Any,
        raw_obs: Any,
        step: int,
        event_counter: Counter[str],
        latest_events: list[str],
        policy: Any,
    ) -> None:
        self.env = env
        self.raw_obs = raw_obs
        self.step = step
        self.event_counter = event_counter
        self.latest_events = latest_events
        self.policy = policy


def generate_case(case: TargetCase, out_dir: Path, max_per_case: int, *, start_index: int) -> int:
    env_kwargs = {
        "observation_mode": "pixels",
        "max_steps": case.max_steps,
    }
    if case.map_variant == "default":
        env = make_env(task_id=case.task_id, **env_kwargs)
    else:
        map_path = materialize_spatial_map_variant(case.task_id, case.map_variant, seed=case.seed)
        env = make_env(task_id=case.task_id, map_path=map_path, **env_kwargs)

    policy = load_policy(case.policy_path)
    if hasattr(policy, "reset"):
        policy.reset(seed=case.seed, task_id=case.task_id)

    raw_obs, raw_info = env.reset(seed=case.seed)
    obs = raw_obs
    policy_info = build_policy_info(
        info_mode="safe",
        raw_info=raw_info,
        last_reward=0.0,
        task_id=case.task_id,
    )
    event_counter: Counter[str] = Counter()
    written = 0
    step = 0
    terminated = False
    truncated = False

    try:
        while not (terminated or truncated) and written < max_per_case:
            sample = RuntimeSample(
                env=env,
                raw_obs=raw_obs,
                step=step,
                event_counter=event_counter,
                latest_events=[],
                policy=policy,
            )
            if case.selector(sample):
                written += write_sample_variants(
                    case,
                    sample,
                    out_dir,
                    sample_index=start_index + written,
                )

            action = int(policy.act(obs, policy_info))
            raw_obs, reward, terminated, truncated, raw_info = env.step(action)
            latest_events = event_names(raw_info)
            event_counter.update(latest_events)
            obs = raw_obs
            policy_info = build_policy_info(
                info_mode="safe",
                raw_info=raw_info,
                last_reward=float(reward),
                task_id=case.task_id,
            )
            step += 1
    finally:
        env.close()
    return written


def write_sample_variants(case: TargetCase, sample: RuntimeSample, out_dir: Path, *, sample_index: int) -> int:
    raw_image = sample.raw_obs[:MAP_PIXEL_HEIGHT, :MAP_PIXEL_WIDTH]
    base_payload = runtime_payload(case, sample)
    written = 0
    for variant in case.color_variants:
        payload = json.loads(json.dumps(base_payload))
        payload["annotations"]["obs_variant"] = variant
        image = apply_obs_variant(raw_image, variant)
        stem = f"targeted_{sample_index:06d}_{safe_name(case.name)}_{variant}"
        Image.fromarray(image).save(out_dir / f"{stem}.png")
        (out_dir / f"{stem}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        written += 1
    return written


def runtime_payload(case: TargetCase, sample: RuntimeSample) -> dict[str, Any]:
    runtime = sample.env.engine.runtime
    room = runtime.room
    player = runtime.player
    payload: dict[str, Any] = {
        "id": "targeted_runtime_state",
        "annotations": {
            "task_id": case.task_id,
            "map_variant": case.map_variant,
            "target_case": case.name,
            "step": sample.step,
            "event_counts": dict(sorted(sample.event_counter.items())),
            "tile_labels": tile_labels_from_runtime_room(room),
        },
    }
    write_runtime_pixel_annotations(payload, player, room)
    return payload


def select_task2_after_progress(sample: RuntimeSample) -> bool:
    counts = sample.event_counter
    if counts.get("monster_killed", 0) <= 0 or counts.get("chest_opened", 0) <= 0:
        return False
    runtime = sample.env.engine.runtime
    if getattr(runtime.room, "monsters", {}):
        return False
    player_tile = px_to_tile(runtime.player.position_px)
    if player_tile[0] <= 2:
        return sample.step % 3 == 0
    return sample.step % 10 == 0


def select_task4_final_chest(sample: RuntimeSample) -> bool:
    counts = sample.event_counter
    if counts.get("chest_revealed", 0) <= 0:
        return False
    runtime = sample.env.engine.runtime
    visible_chests = [
        chest
        for chest in getattr(runtime.room, "chests", {}).values()
        if getattr(chest, "is_visible", False)
    ]
    if not visible_chests:
        return False
    return sample.step % 4 == 0


def px_to_tile(position_px: tuple[float, float]) -> tuple[int, int]:
    return (
        min(GRID_WIDTH - 1, max(0, int(position_px[0] // TILE_SIZE))),
        min(GRID_HEIGHT - 1, max(0, int(position_px[1] // TILE_SIZE))),
    )


def load_policy(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_targeted_policy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.make_policy()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    main()
