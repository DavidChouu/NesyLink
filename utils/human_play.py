"""
Human-play debugging script for NesyLink environments.

Play the game with keyboard controls and inspect obs/info by pressing Tab.

Usage:
    # 本次数理逻辑任务只有 pixels 输出，所以无法使用 Tab 来查看 obs/info 的变化细节，但你仍然可以通过观察游戏画面来理解环境状态。
    python utils/human_play.py --task mathematical_logic/task_1
    python utils/human_play.py --task mathematical_logic/task_4
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pygame

# Ensure project root is on sys.path so `import nesylink` works when running directly.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import nesylink  
from nesylink.core.constants import (
    ACTION_LABELS,
    TARGET_FPS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from nesylink.core.input import HumanInputState 
from nesylink.tasks import list_tasks 

HISTORY_SIZE = 5 # set the number of past steps to keep in history for inspection


# ─── Diff helpers ────────────────────────────────────────────────────────────


def _is_leaf(v) -> bool:
    return not isinstance(v, (dict, list, tuple)) or (
        isinstance(v, (list, tuple)) and len(v) == 0
    )


def _format_value(v) -> str:
    if isinstance(v, np.ndarray):
        if v.size <= 16:
            return f"ndarray({v.shape}) {v.dtype} {v.tolist()}"
        return f"ndarray({v.shape}) {v.dtype}"
    if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.ndarray):
        return f"[ndarray({v[0].shape}) ... x{len(v)}]"
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return f"{type(v).__name__}[{len(v)} items]"
    return repr(v)


def _equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, Sequence) and not isinstance(a, str) and isinstance(b, Sequence) and not isinstance(b, str):
        if len(a) != len(b):
            return False
        return all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def diff_info(old: dict, new: dict, prefix: str = "") -> list[tuple[str, object, object]]:
    """Recursively diff two info dicts. Returns [(dotted_path, old_val, new_val), ...]."""
    changes: list[tuple[str, object, object]] = []
    all_keys = set(old.keys()) | set(new.keys())
    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        if key not in old:
            changes.append((path, "<absent>", new[key]))
            continue
        if key not in new:
            changes.append((path, old[key], "<removed>"))
            continue
        ov, nv = old[key], new[key]
        if isinstance(ov, Mapping) and isinstance(nv, Mapping):
            changes.extend(diff_info(ov, nv, path))
        elif _is_leaf(ov) or _is_leaf(nv):
            if not _equal(ov, nv):
                changes.append((path, ov, nv))
        else:
            if not _equal(ov, nv):
                changes.append((path, ov, nv))
    return changes


def _changed_top_sections(old: dict, new: dict) -> set[str]:
    """Return set of top-level keys in info that contain changes."""
    sections: set[str] = set()
    for path, _, _ in diff_info(old, new):
        sections.add(path.split(".")[0])
    return sections


# ─── Print helpers ───────────────────────────────────────────────────────────


def _print_obs(obs: dict) -> None:
    print("  --- obs ---")
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray):
            if val.size <= 16:
                print(f"    {key}: {val.tolist()}")
            else:
                print(f"    {key}: ndarray{val.shape} {val.dtype}")
        else:
            print(f"    {key}: {val!r}")


def _print_info_full(info: dict, indent: int = 4) -> None:
    """Print a full info dict in a compact but readable way."""
    _print_dict_recursive(info, indent)


def _print_dict_recursive(d: dict, indent: int) -> None:
    for key in sorted(d.keys()):
        val = d[key]
        prefix = " " * indent
        if isinstance(val, Mapping):
            print(f"{prefix}{key}:")
            _print_dict_recursive(val, indent + 2)
        elif isinstance(val, np.ndarray):
            if val.size <= 16:
                print(f"{prefix}{key}: {val.tolist()}")
            else:
                print(f"{prefix}{key}: ndarray{val.shape} {val.dtype}")
        elif isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], dict):
            print(f"{prefix}{key}: [{len(val)} items]")
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    print(f"{prefix}  [{i}]:")
                    _print_dict_recursive(item, indent + 4)
                else:
                    print(f"{prefix}  [{i}]: {_format_value(item)}")
        else:
            print(f"{prefix}{key}: {_format_value(val)}")


def dump_history(history: deque) -> None:
    """Print the last N steps with diffs highlighting changed info signals."""
    if not history:
        print("\n[No steps recorded yet]\n")
        return

    entries = list(history)
    print(f"\n{'═' * 60}")
    print(f"  Last {len(entries)} steps (oldest → newest)")
    print(f"{'═' * 60}")

    for i, (step_count, action, obs, info) in enumerate(entries):
        action_label = ACTION_LABELS.get(action, str(action))
        print(f"\n{'─' * 60}")
        print(f"  Step {step_count} | Action: {action_label}")
        print(f"{'─' * 60}")

        _print_obs(obs)

        if i > 0:
            _, _, _, prev_info = entries[i - 1]
            changes = diff_info(prev_info, info)
            if changes:
                changed_sections = _changed_top_sections(prev_info, info)
                all_sections = set(info.keys())
                unchanged = all_sections - changed_sections

                print("  --- info (changes) ---")
                for path, ov, nv in changes:
                    print(f"    >>> {path}: {_format_value(ov)} -> {_format_value(nv)}")
                if unchanged:
                    print(f"    (unchanged: {', '.join(sorted(unchanged))})")
            else:
                print("  --- info --- (no changes)")
        else:
            print("  --- info (full, first entry) ---")
            _print_info_full(info, indent=4)

    print(f"\n{'═' * 60}\n")


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NesyLink human-play debugger with obs/info inspection"
    )

    task_ids = [t.task_id for t in list_tasks()]
    parser.add_argument(
        "--task",
        type=str,
        default="mathematical_logic/task_1",
        choices=task_ids,
        help=f"Task ID to play (default: mathematical_logic/task_1). Available: {task_ids}",
    )
    parser.add_argument(
        "--rooms",
        type=str,
        default=None,
        help="Direct map JSON path (overrides --task)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Environment seed for reproducibility",
    )
    args = parser.parse_args()

    # Build environment
    kwargs = dict(api="gym", render_mode="rgb_array", auto_reset_on_step=True, observation_mode="pixels") # NOTE:  你可以修改 observation_mode ，可选项有 “full” “pixels” “grid”
    if args.rooms:
        env = nesylink.make_env(args.rooms, **kwargs)
    else:
        env = nesylink.make_env(task_id=args.task, **kwargs)

    obs, info = env.reset(seed=args.seed)
    print(f"\n[Loaded task: {args.task}]")
    print(f"[Mission: {info.get('env', {}).get('map_id', 'N/A')}]")
    print(f"[Controls: Arrows=move, Z=A(sword/interact), X=B(shield), Tab=dump last {HISTORY_SIZE} steps, Esc=quit]\n")

    # Pygame setup
    pygame.init()
    pygame.display.set_caption(f"NesyLink Debugger — {args.task}")
    display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    input_state = HumanInputState()

    # History buffer: deque of (step_count, action, obs, info)
    history: deque = deque(maxlen=HISTORY_SIZE)
    game_over = False
    victory = False
    running = True

    def reset_episode() -> None:
        nonlocal obs, info, game_over, victory
        obs, info = env.reset(seed=args.seed)
        history.clear()
        game_over = False
        victory = False
        print("[Episode reset]\n")

    while running:
        clock.tick(TARGET_FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_TAB:
                    dump_history(history)

                elif game_over or victory:
                    reset_episode()

                else:
                    input_state.handle_keydown(event.key)

            elif event.type == pygame.KEYUP:
                input_state.handle_keyup(event.key)

        if running and not game_over and not victory:
            action = input_state.resolve_action()
            step_count = info.get("episode", {}).get("step_count", len(history))
            obs, reward, terminated, truncated, info = env.step(action)
            history.append((step_count, action, obs, info))

            if terminated:
                reason = info.get("terminal_reason")
                if reason == "agent_dead":
                    game_over = True
                elif reason == "world_completed":
                    victory = True

        # Render
        frame = env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        display_surface.blit(scaled, (0, 0))

        if game_over or victory:
            font = pygame.font.SysFont(None, 28)
            text = "GAME OVER - Press any key" if game_over else "VICTORY - Press any key"
            text_surface = font.render(text, True, (255, 255, 255))
            rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            display_surface.blit(text_surface, rect)

        pygame.display.flip()

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
