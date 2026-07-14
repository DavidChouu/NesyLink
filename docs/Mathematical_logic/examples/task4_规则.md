# Task 4 规则说明书

## 1. 房间定义（共 7 个）

| 房间 | 视觉特征（CNN 可见） | 连接 |
|------|---------------------|------|
| **west** | 有 `switch` 或 `button_pressed` tile | 东出口 → center （初始状态下为center北，之后判断依靠桥转换规则）|
| **center北** | 桥向上方延伸 + 北边界 (row=0) 有 bridge tile | 北出口 → north 房间，西出口 → west |
| **center东** | 桥横向延伸 + 东边界 (col=9) 有 bridge tile | 东出口 → east 房间，西出口 → west |
| **center南** | 桥向下方延伸 + 南边界 (row=7) 有 bridge tile | 南出口 → south 房间，西出口 → west |
| **north** | 无 switch/bridge/abyss，有南边界 exit | 南出口 → center |
| **east** | 无 switch/bridge/abyss，有西边界 exit | 西出口 → center |
| **south** | 无 switch/bridge/abyss，有北边界 exit | 北出口 → center |

### 房间识别 `_detect_room(vision)`

```
1. CNN 扫到 "switch" 或 "button_pressed"  → "west"
2. CNN 扫到 "bridge" 或 abyss > 10 个      → "center"（具体哪个 center 按桥边界判断）
3. 都不满足 → "target"（north/east/south，具体哪个靠 mission 上下文）
```

**center 细分**：检查 bridge tile 在哪个边界——
- row=0 有 bridge → center北
- col=9 有 bridge → center东
- row=7 有 bridge → center南

---

## 2. 桥转换规则

west 房间有一个开关。**每按一次**，center 的桥循环切换：

```
west_to_north(center北) → west_to_east(center东) → west_to_south(center南) → (循环)
```

累计按开关次数记为 `switch_presses_done`，`switch_presses_done % 3` 对应：
- `0` → center北
- `1` → center东
- `2` → center南

---

## 3. 全局状态变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `mission` | int (0/1/2/3) | 当前任务编号 |
| `mission_phase` | str | `to_switch` / `to_target` / `at_target` / `return_to_west` / `go_final_chest` |
| `switch_presses_done` | int | 累计按开关次数 |
| `move_target_tile` | (int,int) | 像素移动目标 tile |
| `move_action` | int | 像素移动方向 |
| `pending_interact` | bool | 下一帧按 A |
| `exit_push_action` | int | 出口推出方向 |
| `target_exit_tile` | (int,int) | 出口目标 tile |
| `interacted_positions` | set | 已按 A 交互过的宝箱位置（打开后贴图残留，用行为记忆过滤） |
| `key_confirmed` | bool | 已拿到钥匙 |
| `has_sword` | bool | 已拿到剑 |
| `monster_was_visible` | bool | 之前看到过怪物 |
| `switch_pos` | (int,int) | CNN 第一次扫到的开关 tile 坐标，**全程不清零** |

**Mission 3 专用状态：**

| 变量 | 说明 |
|------|------|
| `fc_scans` | 已遍历了几座桥（0/1/2） |
| `fc_state` | `"arrive_center"` / `"go_west"` / `"press_switch"` / `"return_center"` |

---

## 4. 全流程

### Mission 0：北房间拿钥匙（presses=0，桥=center北）

```
to_switch:     在 west 房间，走到 switch 上按 A → switch_presses_done += 1
               switch_presses_done 够了(0次)→切 to_target
to_target:     west → 东出口 → center北 → 北出口 → north 房间 → 切 at_target
at_target:     CNN 扫 chest → BFS 走到邻格 → 面向按 A
               按 A 时标记 interacted_positions
               没宝箱了 → 切 return_to_west
return_to_west: north → 南出口 → center北 → 西出口 → west → mission=1, 切 to_switch
```

### Mission 1：东房间拿剑（presses=1，桥=center东）

```
to_switch:     在 west，按 1 次开关（累计 1 次）→ 切 to_target
to_target:     west → 东出口 → center东 → 东出口(需钥匙) → east 房间 → 切 at_target
at_target:     开宝箱拿剑 → 切 return_to_west
return_to_west: east → 西出口 → center东 → 西出口 → west → mission=2, 切 to_switch
```

### Mission 2：南房间杀怪（presses=2，桥=center南）

```
to_switch:     在 west，按 1 次开关（累计 2 次）→ 切 to_target
to_target:     west → 东出口 → center南 → 南出口 → south 房间 → 切 at_target
at_target:     CNN 扫 monsters → BFS 走到邻格 → 面向按 A
               怪物消失 → 切 go_final_chest (mission=3)
```

### Mission 3：遍历三座桥开最终宝箱

**规则**：宝箱一定在 center北、center东、center南 **三座桥之一** 上。从 center南 开始，依次去 west 按开关转桥，最多转 2 次覆盖全部三座桥。

```
fc_state = "arrive_center"
  → 从 south 北出口进入 center南
  → CNN 扫 bridge 走一圈看有没有 chest
  → 没宝箱 + fc_scans < 2 → fc_state = "go_west"

fc_state = "go_west"
  → 如果当前在 target 房间（north/east/south）→ 先走返回出口去 center
  → 如果在 center → 往左走到 col=0 → 推出西出口 → 进入 west
  → 如果在 west → 直接切 fc_state = "press_switch"

fc_state = "press_switch"
  → 用已记住的 switch_pos（mission 0 开局扫到的），不重新扫
  → BFS 走到 switch_pos 格子上 → ACTION_A
  → 按完：fc_scans += 1, fc_state = "return_center"

fc_state = "return_center"
  → 如果在 west → 往右走到 col=9 → 推出东出口 → 进入 center
  → 如果在 center → 直接切 fc_state = "arrive_center"

循环：arrive_center → go_west → press_switch → return_center → arrive_center
       每轮 fc_scans++，桥转一次。fc_scans >= 2 后不再去 west。
```

**注意**：任何时刻 CNN 扫到 `chest` tile，立即中断当前状态，走 `_act_open_chest_at` 开宝箱。

---

## 5. 按开关函数

```python
def _act_press_switch(vision, player):
    # switch_pos 在 mission 0 开局时 CNN 第一次扫到就记住，全程不清零
    # 1. 如果 player == switch_pos → 站在开关上了 → 按 A
    # 2. 如果 manhattan(player, switch_pos) == 1 → 直接朝开关走一格
    # 3. 否则 BFS 走到 switch_pos 格子
```

**关键**：
- `switch_pos` 开局记住后全程不清零，后续复用，不重新扫
- 走到开关**格子上**（不是邻格），按 A
- `switch` 和 `button_pressed` 是同一个东西（按前按后图标不同，本质一样）

---

## 6. BFS 路径规划

```python
def bfs_path(start, goals, vision):
    # 标准 BFS，只走 is_walkable 的 tile
    # is_walkable: tile 不在 BLOCKING 中 + 在 SAFE_WALKABLE 中
    # BLOCKING = {wall, chest, trap, abyss, gap, monster, unknown}
    # SAFE = {floor, player, bridge, button, switch, exit_*, npc}
```

---

## 7. 像素感知移动

```python
# 每帧检查 vision.player.tile == move_target_tile → 到达停止
# 超时 80 帧放弃
# 出口边界有像素对齐微调
```

---

## 8. 出口推进

```python
def _act_exit_directional(direction):
    # 1. CNN 扫指定方向边界上的 exit tile
    # 2. BFS 走过去
    # 3. 站在边界上时设 exit_push_action 推出去
    # Fallback: CNN 漏检时用该方向所有 walkable 边界 tile 当候选
```

---

## 9. Safety Shield

移动前安检：
- 出口推出 → 放行
- 目标格是 chest/switch/monster → 放行
- 目标格 walkable → 放行
- 否则 → 拦截，返回 NOOP

---

## 10. 硬编码清理

**必须删除的硬编码：**
- `_act_go_final_chest` 中的 `(4, 4)` 等待坐标
- 任何写死的位置数字

**合规约束：**
- 只读 `obs`（RGB 像素）和 `info["inventory"]`（keys/items/tools/equipped）
- 不读 `info["agent"]` / `info["env"]` / `info["events"]`
- 所有坐标从 CNN 像素分类获取，不写死

---

## 11. 宝箱交互标记

打开宝箱后，贴图可能残留（CNN 仍分类为 chest）。用 `interacted_positions` 集合记住已交互过的位置。

**标记时机**：必须在真正按 A 时标记，不能在面向（face_action）时标记。否则面向动作移动后若交互未命中，宝箱被永久跳过。

```python
# act() 中 pending_interact 触发时：
if phase in ("at_target", "go_final_chest"):
    chests = CNN扫到的chest
    if chests:
        interacted_positions.add(最近的chest)
```
