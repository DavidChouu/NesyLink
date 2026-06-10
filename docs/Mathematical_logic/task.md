# Task & Environment 说明

本文档是对 2026 春学期数理逻辑大作业 涉及到的5个 Task 的详细说明，以及 nesylink 环境使用方法的简单介绍

> 💡 如何对 Raw pixels 进行符号抽取、规划和执行，也是本次作业的重点之一
> 祝大家玩的开心 😉

## nesylink 环境介绍

### 安装与运行
从项目根目录安装本仓库：

```bash
git clone https://crazyjassbread/nesylink.git
cd nesylink
conda activate your_env_name # 激活你的 Python 环境
pip install -e .

# Human play: 需要安装 pygame 依赖：
pip install -e ".[pygame]"
```

之后可以直接在终端运行
```shell
python utils/human_play.py --task mathematical_logic/task_1 # 1~5 任选其一
```
Tips：可以在游玩过程中使用 `ESC` 键退出，使用 `Tab` 了解 nesylink 提供的 info 与 obs 结构。

### 任务列表
> 五个任务的默认配置在 [nesylink/tasks/builtin.py](/nesylink/tasks/builtin.py) 和 [nesylink/tasks/task_config/mathematical_logic.yaml](/nesylink/tasks/task_config/mathematical_logic.yaml) 中定义。可以根据实验需要调整 `max_steps`, `action_repeat` ,`reward` 等训练参数。
> ⚠️本次数理逻辑任务要求使用 `observation_mode="pixels"`，请不要把结构化 obs 作为 agent 输入。提交作业时请明确标注修改过的环境配置。

| task_id | 地图 | 奖励 | 最大步数 | 任务说明 |
|---|---|---|---:|---|
| `mathematical_logic/task_1` | [map_1](/nesylink/map_data/mathematical_logic/task_1/room_001.json) | [reward_1](/nesylink/rewards/mathematical_logic/task_1.py) | 500 | 收集钥匙并从北侧锁门离开 |
| `mathematical_logic/task_2` | [map_2](/nesylink/map_data/mathematical_logic/task_2/room_001.json) | [reward_2](/nesylink/rewards/mathematical_logic/task_2.py) | 500 | 击败怪物、拿钥匙、从西侧条件门离开 |
| `mathematical_logic/task_3` | [map_3](/nesylink/map_data/mathematical_logic/task_3/dungeon.json) | [reward_3](/nesylink/rewards/mathematical_logic/task_3.py) | 1500 | 穿过怪物房，去西侧房间拿钥匙，返回起点并打开东侧锁门 |
| `mathematical_logic/task_4` | [map_4](/nesylink/map_data/mathematical_logic/task_4/dungeon.json) | [reward_4](/nesylink/rewards/mathematical_logic/task_4.py) | 2000 | 旋转桥、拿钥匙和剑、击败怪物并打开最终宝箱 |
| `mathematical_logic/task_5` | [map_5](/nesylink/map_data/mathematical_logic/task_5/dungeon.json) | [reward_5](/nesylink/rewards/mathematical_logic/task_5.py) | 2000 | 探索多房间地牢并打开所有宝箱 |

## Examples 参考实现

仓库的 `examples/` 目录给出了两个 Python 参考实现，用于演示如何通过当前 `nesylink` 框架接口运行内置任务：

| 文件 | 对应任务 | 说明 |
|---|---|---|
| `examples/task1_reference.py` | `mathematical_logic/task_1` | 使用固定像素级动作序列完成“拿钥匙并通过北侧锁门”的任务。 |
| `examples/task2_reference.py` | `mathematical_logic/task_2` | 演示符号状态、邻接谓词和 BFS 子目标规划，完成当前 `mathematical_logic/task_2`。 |

运行方式：

```bash
python docs/Mathematical_logic/examples/task1_reference.py
python docs/Mathematical_logic/examples/task2_reference.py
```

它们是本作业给出的参考实现，重点是展示：

1. 如何从 raw pixels 中抽取离散符号状态，并利用 `info` 做状态抽象与调试。
2. 如何把 tile 级计划展开为像素级动作 replay。
3. 如何通过 `terminated/truncated`、`info["terminal_reason"]` 和 `info["game"]["world_completed"]` 检查真实环境执行结果。

### 动作空间、观测空间和信息结构

> 本次数理逻辑任务默认使用 `observation_mode="pixels"`。所以 agent 的输入 `obs` 只能是 raw pixels 🥲
> 先别灰心，环境提供了结构化 `info` 可以用于状态抽象、日志、调试和验证执行结果。

动作空间是离散动作，编号如下：

| 编号 | 名称 | 含义 |
|---:|---|---|
| 0 | `WAIT` | 等待 |
| 1 | `UP` | 向上移动 1 像素 |
| 2 | `DOWN` | 向下移动 1 像素 |
| 3 | `LEFT` | 向左移动 1 像素 |
| 4 | `RIGHT` | 向右移动 1 像素 |
| 5 | `BUTTON_A` | 交互；使用物品A（默认是剑） |
| 6 | `BUTTON_B` | 使用物品B（默认是盾） |

地图大小为 `10 x 8` 个 tile，每个 tile 是 `16 x 16` 像素。因此从一个 tile 的左上角移动到相邻 tile 的左上角，通常需要连续执行 16 次同方向动作。

#### `obs`: raw pixels

默认任务配置中，`reset()` 和 `step()` 返回的 `obs` 是当前地图区域的 RGB 图像：

```python
obs, info = env.reset(seed=0)
print(obs.shape, obs.dtype)  # (128, 160, 3), uint8
```

观测空间为：

```python
Box(low=0, high=255, shape=(128, 160, 3), dtype=uint8)
```

说明：

- `obs` 只包含可游玩的地图区域，不包含底部 HUD。
- 完整渲染画面可以通过 `env.render()` 获取，shape 为 `(160, 160, 3)`，包含地图与 HUD。
- 坐标约定：tile 坐标使用 `(x, y)`，其中 `x` 是列 `0..9`，`y` 是行 `0..7`；像素坐标的地图区域大小是 `160 x 128`。

#### `info`: 状态抽象与调试信息

`info` 是一个嵌套字典。它不是 agent 的视觉输入，但可以用于从 pixels 中抽取符号后的状态对齐、执行验证、debug、日志记录和 reward 分析。

顶层字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `info["episode"]` | `dict` | 当前 episode 的编号、步数、seed 和无进展步数 |
| `info["env"]` | `dict` | 当前地图、房间 id、房间坐标 |
| `info["agent"]` | `dict` | 玩家生命值、tile 坐标、朝向、像素坐标 |
| `info["inventory"]` | `dict` | 金币、钥匙、物品、工具和装备槽 |
| `info["entities"]` | `dict` | 当前房间实体数量统计，例如怪物、宝箱、陷阱、出口 |
| `info["dynamic"]` | `dict` | 动态机关状态，例如桥、缺口、可切换对象 |
| `info["events"]` | `dict` | 当前 step 产生的事件记录、计数和布尔标记 |
| `info["game"]` | `dict` | 死亡、换房、到达出口、完成世界等游戏级状态 |
| `info["terminal_reason"]` | `str | None` | episode 终止原因 |
| `info["control"]` | `dict` | 控制模式、观测模式、动作重复和移动像素等控制参数 |
| `info["debug"]` | `dict` | 当前动作姿态、控制锁、调试消息等内部状态 |
| `info["reward"]` | `dict` | reward 名称、信号、权重和 reward 触发的终止信息 |

##### `info["episode"]`

| 字段 | 含义 |
|---|---|
| `info["episode"]["id"]` | 当前 episode 编号 |
| `info["episode"]["step_count"]` | 当前 episode 已执行的环境步数 |
| `info["episode"]["seed"]` | reset 时使用的随机种子 |
| `info["episode"]["no_progress_steps"]` | 连续无进展步数 |

##### `info["env"]`

| 字段 | 含义 |
|---|---|
| `info["env"]["map_id"]` | 当前地图 id，例如 `mathematical_logic/task_1` |
| `info["env"]["room_id"]` | 当前房间 id |
| `info["env"]["room_coord"]` | 当前房间在 dungeon 中的坐标 `(x, y)` |

##### `info["agent"]`

| 字段 | 含义 |
|---|---|
| `info["agent"]["hp"]` | 玩家当前生命值 |
| `info["agent"]["tile"]` | 玩家当前 tile 坐标 `(x, y)` |
| `info["agent"]["facing"]` | 玩家朝向，通常为 `up/down/left/right` |
| `info["agent"]["position_px"]` | 玩家左上角像素坐标 `(x, y)`；pixel control 下提供 |

##### `info["inventory"]`

| 字段 | 含义 |
|---|---|
| `info["inventory"]["gold"]` | 当前金币数量 |
| `info["inventory"]["keys"]` | 当前钥匙数量 |
| `info["inventory"]["items"]` | 当前持有物品列表 |
| `info["inventory"]["tools"]` | 当前可用工具列表 |
| `info["inventory"]["equipped"]` | 装备槽映射，例如 `{"A": "sword", "B": "shield"}` |

##### `info["entities"]`

| 字段 | 含义 |
|---|---|
| `info["entities"]["monsters_remaining"]` | 当前房间剩余怪物数量 |
| `info["entities"]["monster_ids"]` | 当前房间怪物 id 列表 |
| `info["entities"]["chests_remaining"]` | 当前房间可见且未开启宝箱数量 |
| `info["entities"]["chests_hidden"]` | 当前房间隐藏宝箱数量 |
| `info["entities"]["traps_active"]` | 当前房间激活陷阱数量 |
| `info["entities"]["buttons_pressed"]` | 当前房间已按下按钮数量 |
| `info["entities"]["switches_total"]` | 当前房间 switch 总数 |
| `info["entities"]["exits_open"]` | 当前房间已打开出口数量 |
| `info["entities"]["exits_total"]` | 当前房间出口总数 |

##### `info["dynamic"]`

动态信息主要用于 Task 4 这类桥/机关任务。

| 字段 | 含义 |
|---|---|
| `info["dynamic"]["objects"]` | 全局动态对象状态，key 是 object id |
| `info["dynamic"]["objects"][object_id]["kind"]` | 动态对象类型 |
| `info["dynamic"]["objects"][object_id]["room_id"]` | 动态对象所属房间 |
| `info["dynamic"]["objects"][object_id]["state"]` | 动态对象当前状态 |
| `info["dynamic"]["current_room_tiles"]` | 当前房间运行时动态 tile 列表 |

`current_room_tiles` 中的元素形如：

```python
{"pos": [x, y], "tile": "gap"}      # 或 "bridge"
```

##### `info["events"]`

事件信息只描述“当前 step”中发生的事件，适合做 reward debug 和执行轨迹分析。

| 字段 | 含义 |
|---|---|
| `info["events"]["records"]` | 事件记录列表，每个元素至少包含 `name` |
| `info["events"]["counts"]` | 事件名到出现次数的映射 |
| `info["events"]["flags"]` | 事件名到布尔值的映射，表示该事件是否出现 |
| `info["events"]["details"]` | 原始事件 detail 列表 |

常见事件名包括：

| 事件 | 含义 |
|---|---|
| `chest_opened` | 打开宝箱 |
| `key_collected` | 获得钥匙 |
| `gold_collected` | 获得金币 |
| `item_collected` | 获得道具 |
| `agent_healed` | 玩家回血 |
| `agent_damaged` | 玩家受到伤害 |
| `trap_triggered` | 触发陷阱 |
| `abyss_fall` | 掉入深渊/缺口 |
| `monster_damaged` | 怪物受到伤害 |
| `monster_killed` | 怪物被击败 |
| `shield_block` | 盾牌格挡成功 |
| `door_opened` | 门被打开 |
| `button_pressed` | 按钮被按下 |
| `switch_activated` | switch 被触发 |
| `bridge_rotated` | 桥状态发生变化 |
| `dynamic_object_state_changed` | 动态对象状态变化 |
| `talked_npc` | 与 NPC 对话 |
| `room_changed` | 进入新房间 |
| `exit_reached` | 到达出口 |
| `environment_completed` | 当前环境目标完成 |
| `world_completed` | 整个任务完成 |
| `action_blocked` | 行动被阻挡 |

##### `info["game"]`

| 字段 | 含义 |
|---|---|
| `info["game"]["dead"]` | 玩家是否死亡 |
| `info["game"]["room_changed"]` | 当前 step 是否切换房间 |
| `info["game"]["exit_reached"]` | 当前 step 是否到达出口 |
| `info["game"]["world_completed"]` | 是否完成整个任务 |

##### `info["terminal_reason"]`

`terminal_reason` 表示 episode 的终止原因。常见值：

| 值 | 含义 |
|---|---|
| `None` | 尚未终止 |
| `world_completed` | 完成整个任务 |
| `environment_completed` | 完成当前环境目标 |
| `agent_dead` | 玩家死亡 |

如果达到 `max_steps`，Gymnasium 会返回 `truncated=True`；这种情况不一定有 `terminal_reason`。

##### `info["control"]`

在默认 pixel control 下：

| 字段 | 含义 |
|---|---|
| `info["control"]["action_repeat"]` | 外部动作重复的 engine tick 数 |
| `info["control"]["inner_steps"]` | 当前 `env.step()` 实际执行的内部步数 |
| `info["control"]["movement_pixels"]` | 每个内部步移动的像素数 |

在 grid control 下，还会提供：

| 字段 | 含义 |
|---|---|
| `info["control"]["control_mode"]` | 控制模式 |
| `info["control"]["observation_mode"]` | 观测模式 |
| `info["control"]["tile_size"]` | tile 大小，当前为 `16` |
| `info["control"]["monster_move_periods"]` | 不同怪物类型的移动周期 |

##### `info["debug"]`

| 字段 | 含义 |
|---|---|
| `info["debug"]["message"]` | 最近的环境调试消息 |
| `info["debug"]["engine_done"]` | engine 是否已经结束 |
| `info["debug"]["action_item"]` | 当前动作使用的物品 |
| `info["debug"]["action_pose"]` | 当前动作姿态 |
| `info["debug"]["action_ticks_remaining"]` | 动作姿态剩余 tick |
| `info["debug"]["control_lock_steps_remaining"]` | 控制锁剩余步数 |
| `info["debug"]["pending_respawn_tile"]` | 等待重生的 tile；没有则为 `None` |

##### `info["reward"]`

| 字段 | 含义 |
|---|---|
| `info["reward"]["reward_name"]` | 当前 reward 名称 |
| `info["reward"]["reward_signals"]` | 当前 step 提取出的 reward 信号 |
| `info["reward"]["reward_weights"]` | 当前 reward 权重 |
| `info["reward"]["terminated"]` | reward 是否要求终止 episode |
| `info["reward"]["terminated_reason"]` | reward 触发的终止原因 |
