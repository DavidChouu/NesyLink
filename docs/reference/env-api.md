# NesyLink API

## Environment Construction

Preferred entrypoint:

```python
from nesylink.env import make_env
```

Supported forms:

```python
env = make_env(task_id="mathematical_logic/task_1")
env = make_env(map_id="dungeon", reward_id="sparse_exit", max_steps=500)
env = make_env(
    map_path="nesylink/map_data/mathematical_logic/task_5/dungeon.json",
    reward_id="mathematical_logic/task_5",
)
env = make_env(map_id="dungeon", reward_module="nesylink.rewards.exploration")
```

Gymnasium registered form:

```python
import gymnasium as gym
import nesylink

env = gym.make("NesyLink-MathematicalLogic-Task1-v0")
```

Parameters:

- `task_id`
- `map_id`
- `map_path`
- `reward_id`
- `reward_module`
- `reward_kwargs`
- `max_steps`
- `render_mode`
- `action_repeat`
- `control_mode`: `"pixel"` or `"grid"`; defaults to `"pixel"`
- `observation_mode`: `"full"`, `"grid"`, or `"pixels"`; defaults to `"full"`
- `monster_move_periods`: monster type to environment-step period mapping for grid mode
- `max_monsters`: fixed monster slot count override
- `max_inventory`: fixed inventory slot count; defaults to `2`
- `api`

Explicit parameters take precedence over task defaults. `map_path` takes
precedence over `map_id`.

## reset / step

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

`reset()` synchronizes the reward object by calling `reward_fn.reset(obs, info)`.

`step()`:

- advances game mechanics
- computes reward from the configured reward object
- merges base termination with reward-driven termination
- truncates on `max_steps`
- stores reward metadata in `info["reward"]`

Action semantics:

- `0`: wait
- `1..4`: move up/down/left/right
- `5`: trigger slot `A`
- `6`: trigger slot `B`

Default slot behavior:

- `A` starts with `sword`
- `B` starts with `shield`
- `A` first tries adjacent chest, NPC, or switch interaction; if nothing is interactable, it uses the equipped `A` item
- `shield` blocks contact damage and never damages monsters
- `sword` handles melee damage with a one-tile forward hitbox
- action poses remain visible for multiple ticks in RGB renders, but damage/block resolution still happens on the triggering step only

Grid control mode:

- Movement actions move the player by exactly one tile.
- Failed movement leaves the player tile unchanged and emits `action_blocked`.
- Tile coordinates use `(x, y)`, where `x` is column `0..9` and `y` is row `0..7`.
- The `grid` array is indexed as `grid[y, x]`.
- Monster movement periods are measured in environment steps.

## Info Shape

Top-level `info` keys:

- `episode`
- `env`
- `agent`
- `inventory`
- `entities`
- `events`
- `game`
- `terminal_reason`
- `control`
- `debug`
- `reward`

`info["task"]` is deprecated and no longer part of the contract.

Additional fields exposed by this version:

- `info["agent"]["facing"]`
- `info["inventory"]["equipped"]`
- `info["debug"]["action_item"]`
- `info["debug"]["action_pose"]`
- `info["debug"]["action_ticks_remaining"]`
- `info["debug"]["control_lock_steps_remaining"]`
- `info["debug"]["pending_respawn_tile"]`
- `info["dynamic"]["objects"]`
- `info["dynamic"]["current_room_tiles"]`

Dynamic info is useful for bridge/switch tasks. `info["dynamic"]["objects"]`
maps dynamic object ids to their kind, owning room id, and current state.
`info["dynamic"]["current_room_tiles"]` lists runtime dynamic tiles in the
current room, such as `gap` and `bridge`.

## Observation Shape

The default observation is a `gymnasium.spaces.Dict`.

Common keys:

- `grid`: `uint8`, shape `(8, 10)`
- `player_position_px`: `float32`, shape `(2,)`
- `player_tile`: `int32`, shape `(2,)`
- `health`: `int32`, shape `(1,)`
- `gold`: `int32`, shape `(1,)`
- `keys`: `int32`, shape `(1,)`
- `inventory_ids`: `int32`, shape `(2,)`
- `monsters_position_px`: `float32`, shape `(max_monsters, 2)`
- `monsters_tile`: `int32`, shape `(max_monsters, 2)`
- `monsters_active_mask`: `uint8`, shape `(max_monsters,)`
- `monsters_hp`: `int32`, shape `(max_monsters,)`

Use `env.observation_space` for exact bounds in code.

The `grid` observation uses these tile codes:

| Code | Meaning |
|---:|---|
| 0 | Empty floor |
| 1 | Wall |
| 2 | Player |
| 3 | Monster |
| 4 | Closed chest |
| 5 | Exit tile |
| 6 | Active trap |
| 7 | Button |
| 8 | NPC |
| 9 | Gap |
| 10 | Bridge |
| 11 | Switch |

With `observation_mode="grid"`, pixel-coordinate keys are omitted. The grid
observation keys are:

- `grid`: `uint8`, shape `(8, 10)`
- `player_tile`: `int32`, shape `(2,)`
- `health`: `int32`, shape `(1,)`
- `gold`: `int32`, shape `(1,)`
- `keys`: `int32`, shape `(1,)`
- `inventory_ids`: `int32`, shape `(max_inventory,)`
- `monsters_tile`: `int32`, shape `(max_monsters, 2)`, padded with `[-1, -1]`
- `monsters_active_mask`: `bool`, shape `(max_monsters,)`
- `monsters_hp`: `int32`, shape `(max_monsters,)`, padded with `0`

With `observation_mode="pixels"`, observations are raw RGB map frames:

- dtype: `uint8`
- shape: `(128, 160, 3)`
- bounds: `[0, 255]`

This mode returns only the playable map area and omits the HUD.

## Rendering

```python
env = make_env(task_id="mathematical_logic/task_1", render_mode="rgb_array")
obs, info = env.reset(seed=0)
frame = env.render()
```

The RGB frame includes the dungeon area and HUD. The structured `grid`
observation covers only the playable 10 by 8 tile area.
