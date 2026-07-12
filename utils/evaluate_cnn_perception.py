"""Quantitatively evaluate CNN perception against simulator runtime truth.

This tool is for offline validation only.  It may inspect the environment's
runtime state to construct labels; submitted policies must not use that state.
Both the raw static tile head and the final CNN+postprocessing observation are
reported so deterministic refinements are not mistaken for model accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.cnn.components import CLASS_TO_ID, COMPONENT_CLASSES, ID_TO_CLASS
from nesylink.cnn.model import DYNAMIC_CLASSES, suppress_tile_classes
from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH
from nesylink.core.state import tile_from_position_px
from nesylink.env import make_env
from nesylink.vision import classify_frame_cnn
from nesylink.vision.cnn_classifier import DEFAULT_CHECKPOINT, _load_model
from utils.evaluate_policy import (
    apply_obs_variant,
    build_policy_info,
    call_policy,
    load_policy,
    materialize_spatial_map_variant,
    reset_policy,
)


STATIC_CLASSES = tuple(name for name in COMPONENT_CLASSES if name not in DYNAMIC_CLASSES)


def _exit_kind(exit_config: Any) -> str:
    if exit_config.exit_type == "locked_key":
        return "exit_locked"
    if exit_config.exit_type == "conditional":
        return "exit_conditional"
    return "exit_normal"


def oracle_static_grid(env: Any) -> list[list[str]]:
    """Reproduce renderer layer order without player/monster overlays."""

    room = env.engine.runtime.room
    grid = [["floor" for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    for (x, y), kind in room.dynamic_tiles.items():
        grid[y][x] = kind
    for exit_config in room.exits:
        for x, y in exit_config.tiles:
            grid[y][x] = _exit_kind(exit_config)
    for x, y in room.walls:
        grid[y][x] = "wall"
    for trap in room.traps.values():
        if trap.is_active and room.dynamic_tiles.get(trap.pos) != "bridge":
            x, y = trap.pos
            grid[y][x] = "abyss" if trap.trap_type == "abyss" else "trap"
    for button in room.buttons.values():
        x, y = button.pos
        grid[y][x] = "button"
    for switch in room.switches.values():
        x, y = switch.pos
        grid[y][x] = "switch"
    for chest in room.chests.values():
        if chest.is_visible:
            x, y = chest.pos
            grid[y][x] = "chest"
    for npc in room.npcs.values():
        x, y = npc.pos
        grid[y][x] = "npc"
    return grid


def oracle_final_grid(env: Any) -> tuple[list[list[str]], tuple[int, int], set[tuple[int, int]]]:
    grid = oracle_static_grid(env)
    room = env.engine.runtime.room
    monster_tiles = {tuple(monster.tile_pos) for monster in room.monsters.values()}
    for x, y in monster_tiles:
        grid[y][x] = "monster"
    player = env.engine.runtime.player
    player_tile = tuple(tile_from_position_px(player.position_px, player.size_px))
    grid[player_tile[1]][player_tile[0]] = "player"
    return grid, player_tile, monster_tiles


def update_confusion(counter: Counter[tuple[str, str]], truth, predicted) -> None:
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            counter[(truth[y][x], predicted[y][x])] += 1


def class_metrics(confusion: Counter[tuple[str, str]], class_name: str) -> dict[str, float | int]:
    tp = confusion[(class_name, class_name)]
    truth_total = sum(value for (truth, _), value in confusion.items() if truth == class_name)
    pred_total = sum(value for (_, pred), value in confusion.items() if pred == class_name)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / truth_total if truth_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": truth_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def accuracy(confusion: Counter[tuple[str, str]]) -> float:
    total = sum(confusion.values())
    correct = sum(value for (truth, pred), value in confusion.items() if truth == pred)
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--obs-variant", default="default")
    parser.add_argument("--map-variant", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import torch

    model = _load_model(str(args.checkpoint), args.device)
    policy = load_policy(args.policy)
    raw_confusion: Counter[tuple[str, str]] = Counter()
    final_confusion: Counter[tuple[str, str]] = Counter()
    frames = 0
    failures = 0
    player_exact = 0
    player_manhattan = 0
    player_center_error = 0.0
    monster_tp = monster_fp = monster_fn = 0

    for episode in range(args.episodes):
        seed = args.seed + episode
        env_kwargs: dict[str, Any] = {"observation_mode": "pixels"}
        if args.map_variant == "default":
            env = make_env(task_id=args.task, **env_kwargs)
        else:
            map_path = materialize_spatial_map_variant(args.task, args.map_variant, seed=seed)
            env = make_env(task_id=args.task, map_path=map_path, **env_kwargs)
        reset_policy(policy)
        raw_obs, raw_info = env.reset(seed=seed)
        last_reward = 0.0
        step = 0
        terminated = truncated = False
        try:
            while not (terminated or truncated):
                policy_obs = apply_obs_variant(raw_obs, args.obs_variant, info=raw_info, env=env)
                if step % args.sample_every == 0:
                    static_truth = oracle_static_grid(env)
                    final_truth, true_player, true_monsters = oracle_final_grid(env)
                    array = np.asarray(policy_obs, dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        output = model(tensor)
                        logits = suppress_tile_classes(output["tile_logits"], DYNAMIC_CLASSES)
                        raw_ids = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy()
                    raw_grid = [[ID_TO_CLASS[int(raw_ids[y, x])] for x in range(GRID_WIDTH)]
                                for y in range(GRID_HEIGHT)]
                    update_confusion(raw_confusion, static_truth, raw_grid)
                    try:
                        observation = classify_frame_cnn(
                            policy_obs,
                            checkpoint=args.checkpoint,
                            device=args.device,
                            fallback=False,
                        )
                    except Exception:
                        failures += 1
                    else:
                        predicted_grid = [list(row) for row in observation.grid]
                        update_confusion(final_confusion, final_truth, predicted_grid)
                        if observation.player is not None:
                            pred_player = observation.player.tile
                            player_exact += int(pred_player == true_player)
                            player_manhattan += abs(pred_player[0] - true_player[0]) + abs(pred_player[1] - true_player[1])
                            true_center = (
                                env.engine.runtime.player.position_px[0] + env.engine.runtime.player.size_px / 2,
                                env.engine.runtime.player.position_px[1] + env.engine.runtime.player.size_px / 2,
                            )
                            player_center_error += abs(observation.player.center_px[0] - true_center[0]) + abs(
                                observation.player.center_px[1] - true_center[1]
                            )
                        predicted_monsters = {monster.tile for monster in observation.monsters}
                        monster_tp += len(predicted_monsters & true_monsters)
                        monster_fp += len(predicted_monsters - true_monsters)
                        monster_fn += len(true_monsters - predicted_monsters)
                    frames += 1

                info = build_policy_info(
                    info_mode="safe",
                    raw_info=raw_info,
                    last_reward=last_reward,
                    task_id=args.task,
                )
                action = call_policy(policy, policy_obs, info)
                raw_obs, reward, terminated, truncated, raw_info = env.step(action)
                last_reward = float(reward)
                step += 1
        finally:
            env.close()

    successful_frames = frames - failures
    monster_precision = monster_tp / (monster_tp + monster_fp) if monster_tp + monster_fp else 0.0
    monster_recall = monster_tp / (monster_tp + monster_fn) if monster_tp + monster_fn else 0.0
    report = {
        "task": args.task,
        "obs_variant": args.obs_variant,
        "map_variant": args.map_variant,
        "frames": frames,
        "cnn_failures": failures,
        "raw_static_tile_accuracy": accuracy(raw_confusion),
        "final_pipeline_tile_accuracy": accuracy(final_confusion),
        "player_exact_rate": player_exact / successful_frames if successful_frames else 0.0,
        "player_mean_manhattan_tiles": player_manhattan / successful_frames if successful_frames else 0.0,
        "player_mean_center_l1_px": player_center_error / successful_frames if successful_frames else 0.0,
        "monster_precision": monster_precision,
        "monster_recall": monster_recall,
        "classes": {
            name: class_metrics(final_confusion, name)
            for name in COMPONENT_CLASSES
            if any(truth == name for truth, _ in final_confusion)
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
