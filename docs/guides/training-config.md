# Training Configuration Guide

This guide explains how to use NesyLink as a reinforcement learning
environment, with emphasis on the parameters that control task selection,
observations, actions, rewards, and episode behavior.

## Quick Start

Use the Gymnasium interface for most training code:

```python
from nesylink.env import make_env

env = make_env(
    task_id="mathematical_logic/task_1",
    observation_mode="pixels",
    max_steps=500,
)

obs, info = env.reset(seed=0)
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

The action space is always `gymnasium.spaces.Discrete(7)`. The observation
space depends on `observation_mode`.

## Environment Creation

There are two supported ways to create a training env.

Use a registered Gymnasium ID for a stable built-in task:

```python
import gymnasium as gym
import nesylink

env = gym.make("NesyLink-MathematicalLogic-Task1-v0")
```

Use `make_env(...)` when you want to override maps, rewards, observations, or
episode settings:

```python
from nesylink.env import make_env

env = make_env(
    task_id="mathematical_logic/task_5",
    observation_mode="pixels",
    reward_kwargs={"step": -0.01, "world_completed": 100.0},
    max_steps=1000,
)
```

Explicit `make_env(...)` arguments override task defaults.

## Built-in Tasks

Use `task_id` to load a predefined task spec. A task spec provides the default
map, reward, horizon, mission text, and optional environment defaults.

Available built-in task IDs in the current task specs:

| `task_id` | Gym ID | Default observation | Default max steps |
|---|---|---|---:|
| `mathematical_logic/task_1` | `NesyLink-MathematicalLogic-Task1-v0` | `pixels` | 500 |
| `mathematical_logic/task_2` | `NesyLink-MathematicalLogic-Task2-v0` | `pixels` | 500 |
| `mathematical_logic/task_3` | `NesyLink-MathematicalLogic-Task3-v0` | `pixels` | 1000 |
| `mathematical_logic/task_4` | `NesyLink-MathematicalLogic-Task4-v0` | `pixels` | 1000 |
| `mathematical_logic/task_5` | `NesyLink-MathematicalLogic-Task5-v0` | `pixels` | 1000 |

For normal experiments, prefer `task_id`. Use `map_id` or `map_path` only when
you are training on a custom map or intentionally separating map selection from
reward selection.

## `make_env` Parameters

| Parameter | Default | Options / type | Purpose |
|---|---:|---|---|
| `task_id` | `None` | built-in task ID | Loads a predefined task spec. |
| `map_id` | task default | built-in map ID | Selects a packaged map. Overrides the task map when set. |
| `map_path` | task default | path-like | Loads a map or dungeon JSON file from disk. |
| `config_path` | `None` | path-like | Backward-compatible alias used when `map_path` is not set. |
| `api` | `"gym"` | `"gym"` | Selects the env wrapper interface. |
| `reward_id` | task default | built-in reward ID | Selects a packaged reward function. |
| `reward_module` | `None` | Python module path or module | Loads a custom reward module with `make_reward(**kwargs)`. |
| `reward_kwargs` | task default or `{}` | `dict[str, float]` | Overrides reward weights. |
| `max_steps` | task default | positive int or `None` | Truncates the episode after this many outer `env.step()` calls. |
| `action_repeat` | task default or `1` | positive int | Repeats each action for multiple engine ticks in pixel control mode. |
| `control_mode` | task default or `"pixel"` | `"pixel"`, `"grid"` | Chooses pixel-level or tile-level movement semantics. |
| `observation_mode` | task default or `"full"` | `"full"`, `"grid"`, `"pixels"` | Chooses the observation returned by `reset()` and `step()`. |
| `monster_move_periods` | task default or `{}` | `dict[str, int]` | Controls how often each monster type moves in grid control mode. |
| `max_monsters` | task default or `None` | positive int or `None` | Fixes padded monster observation slots. |
| `max_inventory` | task default or `2` | positive int | Fixes padded inventory observation slots. |
| `mission` | task default or `""` | string | Stores mission text on the env; useful for logging. |

The Gym wrapper also accepts these keyword arguments through `make_env(...)`:

| Parameter | Default | Options / type | Purpose |
|---|---:|---|---|
| `render_mode` | `None` | `"rgb_array"` or `None` | Declares render output. In current NesyLink, `env.render()` returns an RGB array either way. |
| `auto_reset_on_step` | `False` for `GymDungeonEnv` | bool | If true, stepping after a terminal state auto-resets instead of raising. |
| `move_speed_px` | `1.0` | float | Pixel movement amount per engine tick in pixel control mode. |
| `player_config` | task default or `{}` | dict | Overrides initial player settings when constructing the engine. |

## Action Space

The action space is `Discrete(7)`.

| ID | Label | Meaning |
|---:|---|---|
| 0 | `WAIT` | No-op / wait. |
| 1 | `UP` | Move upward. |
| 2 | `DOWN` | Move downward. |
| 3 | `LEFT` | Move left. |
| 4 | `RIGHT` | Move right. |
| 5 | `BUTTON_A` | Interact or use the A-slot tool. |
| 6 | `BUTTON_B` | Use the B-slot tool. |

In `control_mode="pixel"`, movement actions move by `move_speed_px` pixels per
engine tick. A tile is `16 x 16` pixels, so moving one full tile usually takes
16 repeated movement ticks when `move_speed_px=1.0`.

In `control_mode="grid"`, movement is tile-level. `action_repeat` is ignored
by the wrapper in grid mode because each outer step already maps to one grid
transition.

## Observation Modes

Use `observation_mode` to choose the data returned by `reset()` and `step()`.

### `observation_mode="pixels"`

This is the recommended mode for raw-pixel RL experiments.

```python
import numpy as np

env = make_env(
    task_id="mathematical_logic/task_5",
    observation_mode="pixels",
)

obs, info = env.reset(seed=0)
assert obs.shape == (128, 160, 3)
assert obs.dtype == np.uint8
```

Observation space:

```python
gymnasium.spaces.Box(
    low=0,
    high=255,
    shape=(128, 160, 3),
    dtype=np.uint8,
)
```

This frame contains only the playable map area. It omits the HUD. Use
`env.render()` if you want the full `(160, 160, 3)` RGB frame with HUD for
debugging or video capture.

### `observation_mode="full"`

This is the default structured observation. It returns a dict with grid,
player, inventory, and monster fields.

Important keys:

| Key | dtype | Shape |
|---|---|---|
| `grid` | `uint8` | `(8, 10)` |
| `player_position_px` | `float32` | `(2,)` |
| `player_tile` | `int32` | `(2,)` |
| `health` | `int32` | `(1,)` |
| `gold` | `int32` | `(1,)` |
| `keys` | `int32` | `(1,)` |
| `inventory_ids` | `int32` | `(max_inventory,)` |
| `monsters_position_px` | `float32` | `(max_monsters, 2)` |
| `monsters_tile` | `int32` | `(max_monsters, 2)` |
| `monsters_active_mask` | `uint8` | `(max_monsters,)` |
| `monsters_hp` | `int32` | `(max_monsters,)` |

### `observation_mode="grid"`

This is a smaller structured observation for tile-level logic. It omits pixel
coordinates.

Keys:

| Key | dtype | Shape |
|---|---|---|
| `grid` | `uint8` | `(8, 10)` |
| `player_tile` | `int32` | `(2,)` |
| `health` | `int32` | `(1,)` |
| `gold` | `int32` | `(1,)` |
| `keys` | `int32` | `(1,)` |
| `inventory_ids` | `int32` | `(max_inventory,)` |
| `monsters_tile` | `int32` | `(max_monsters, 2)` |
| `monsters_active_mask` | `bool` | `(max_monsters,)` |
| `monsters_hp` | `int32` | `(max_monsters,)` |

The `grid` field uses tile codes for floor, wall, player, monsters, chests,
exits, traps, buttons, NPCs, gaps, bridges, and switches. See
`docs/reference/env-api.md` for the full code table.

## Rendering

`render_mode` is separate from `observation_mode`.

- `observation_mode` controls what `reset()` and `step()` return.
- `render_mode` declares how `env.render()` should return visual output.

Current NesyLink render output:

```python
import numpy as np

env = make_env(task_id="mathematical_logic/task_1", render_mode="rgb_array")
obs, info = env.reset(seed=0)
frame = env.render()

assert frame.shape == (160, 160, 3)
assert frame.dtype == np.uint8
```

The render frame includes the playable map plus HUD. Pixel observations return
only the playable map.

## Rewards

Use `reward_id` for packaged rewards:

```python
env = make_env(
    task_id="mathematical_logic/task_2",
    reward_id="mathematical_logic/task_2",
)
```

Built-in reward IDs:

- `mathematical_logic/task_1`
- `mathematical_logic/task_2`
- `mathematical_logic/task_3`
- `mathematical_logic/task_4`
- `mathematical_logic/task_5`
- `custom_reward`

Use `reward_kwargs` to override reward weights:

```python
env = make_env(
    task_id="mathematical_logic/task_5",
    reward_kwargs={
        "step": -0.01,
        "room_changed": 2.0,
        "key_collected": 5.0,
        "monster_kill": 5.0,
        "world_completed": 100.0,
        "death": -20.0,
    },
)
```

Common reward weight keys include `step`, `hp_loss`, `gold_delta`,
`keys_delta`, `monster_hit`, `monster_kill`, `key_collected`, `gold_collected`,
`item_collected`, `agent_healed`, `trap_triggered`, `shield_block`,
`door_opened`, `chest_opened`, `button_pressed`, `switch_activated`,
`room_changed`, `exit_reached`, `environment_completed`, `world_completed`,
`death`, and `invalid_action`.

Use `reward_module` for custom rewards. The module must define
`make_reward(**kwargs)` and return a `nesylink.rewards.base.BaseReward`
instance.

## Episode Length and Termination

`max_steps` controls Gym truncation. If `max_steps=500`, the wrapper returns
`truncated=True` when the episode reaches 500 outer `env.step()` calls without
terminating.

Task rewards can also terminate the episode. Built-in mathematical logic
rewards terminate on:

- `world_completed` or `environment_completed`
- `agent_dead`

Inspect these fields during training:

```python
print(info["terminal_reason"])
print(info["game"]["world_completed"])
print(info["episode"]["step_count"])
```

## Recommended Training Setups

For raw-pixel RL:

```python
env = make_env(
    task_id="mathematical_logic/task_1",
    observation_mode="pixels",
    control_mode="pixel",
    action_repeat=1,
    max_steps=500,
)
```

For symbolic or debugging agents:

```python
env = make_env(
    task_id="mathematical_logic/task_1",
    observation_mode="full",
    control_mode="pixel",
)
```

For tile-level planning experiments:

```python
env = make_env(
    task_id="mathematical_logic/task_1",
    observation_mode="grid",
    control_mode="grid",
)
```

Recommended first pass:

- Start with one small task and fixed evaluation seeds.
- Keep `action_repeat=1` until the agent can reliably move and interact.
- Use `observation_mode="pixels"` for final raw-pixel experiments.
- Use `observation_mode="full"` or `"grid"` for debugging policies and reward logic.
- Log `info["events"]["counts"]` and `info["reward"]["reward_signals"]`.

## Random Rollout Smoke Test

Run this before launching a long training job:

```python
from nesylink.env import make_env

env = make_env(
    task_id="mathematical_logic/task_1",
    observation_mode="pixels",
)

obs, info = env.reset(seed=0)
total_reward = 0.0

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(total_reward, info["terminal_reason"])
env.close()
```

## Dreamer-style Usage

The Dreamer-facing adapter lives in `nesylink.wrappers.dreamer_env`. It exposes
an `embodied.Env`-style interface, flattens structured fields into a vector,
and can include resized rendered images.

Use the Dreamer adapter only when your training stack requires that interface.
For new Gymnasium-based experiments, prefer `make_env(...)`.

## Debugging Reward Learning

When learning stalls, inspect:

- `info["events"]["counts"]`
- `info["events"]["records"]`
- `info["reward"]["reward_signals"]`
- `info["reward"]["reward_weights"]`
- `info["terminal_reason"]`
- `info["episode"]["step_count"]`
- `info["agent"]`
- `info["inventory"]`
- `info["entities"]`

If an event appears but reward remains zero, check the corresponding reward
weight. If the reward signal never appears, check the map object, exit
condition, or action sequence that should generate the event.
