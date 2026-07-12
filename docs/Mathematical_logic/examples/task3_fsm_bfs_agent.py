"""任务 3 的像素识别 + FSM + BFS 策略 —— 多房间任务链。

整体思路：

1. 与 task1/task2 相同，策略推理阶段只使用原始 RGB 图像 ``obs`` 和课程允许的
   显式物品栏信息。本策略只从 ``info["inventory"]`` 中取 ``keys``。
2. 用 ``classify_frame`` 把像素图转成符号 tile 网格，识别玩家、怪物、宝箱、
   出口、NPC、地板等类别。
3. 任务 3 涉及三个房间的往返：
      start_room (起点) → monster_hall (怪物房) → key_room (钥匙房)
      → monster_hall (返回) → start_room (终点锁门)
   使用 ``world_stage`` 计数器追踪当前处于任务链中的哪一步。
4. 在每个 stage 内部，使用与 task1/task2 相同的 BFS + FSM 子阶段进行
   tile 级路径规划和交互（攻击怪物 / 开宝箱 / 推出房间）。
5. 房间切换通过玩家 tile 坐标的显著跳跃（曼哈顿距离 > 3）自动检测，
   不依赖 ``info["env"]["room_id"]`` 等隐藏状态。
6. **像素感知移动**：使用 ``info["agent"]["position_px"]`` 做精确的 tile 间移动，
   持续朝目标 tile 像素坐标移动直到越过边界或超时（80 步）。

任务 3 的 world_stage 流转：

    stage 0: start_room → 向西出口推进 → 房间切换 → stage 1
    stage 1: monster_hall → 击杀怪物 / 向西出口推进 → 房间切换 → stage 2
    stage 2: key_room → 找到宝箱并打开 → 钥匙到手 → stage 3
    stage 3: key_room → 向东出口推进（返回）→ 房间切换 → stage 4
    stage 4: monster_hall → 向东出口推进（返回）→ 房间切换 → stage 5
    stage 5: start_room → 向东锁门出口推进 → 任务完成

与 task1/task2 的主要不同：

- 多房间结构（3 个房间），需要房间切换检测和 world_stage 流转。
- 往返路径：先向西穿过 monster_hall 到 key_room，再沿原路向东返回。
- monster_hall 中有 chaser 怪物（hp=2），需在西行途中处理。
- 终点是东侧锁门（exit_locked），钥匙会被消耗。
- 方向过滤：西行只找西边界出口，东行只找东边界出口。
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import (
    ACTION_A,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
    GRID_HEIGHT,
    GRID_WIDTH,
    TILE_SIZE,
)
from nesylink.vision import PixelObservation, classify_frame_cnn


# ============================================================================
# 类型别名 & 常量
# ============================================================================

Position = tuple[int, int]

MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_TO_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
DELTA_TO_ACTION = {delta: action for action, delta in ACTION_TO_DELTA.items()}

BLOCKING_KINDS = {
    "wall",
    "chest",
    "trap",
    "abyss",
    "gap",
    "monster",
    "unknown",
}
SAFE_WALKABLE_KINDS = {
    "floor",
    "player",
    "bridge",
    "button",
    "switch",
    "exit_normal",
    "exit_locked",
    "exit_conditional",
    "npc",
}

ROOM_TRANSITION_THRESHOLD = 3
MAX_TILE_MOVE_ATTEMPTS = TILE_SIZE * 5  # 80 步


# ============================================================================
# Agent 主体
# ============================================================================


@dataclass
class Task3Agent:
    """Task 3 像素识别 + 多房间 FSM + BFS + 像素感知移动 + safety shield 策略。

    world_stage 语义：

    | stage | 所在房间      | 目标动作                 | 触发下一 stage 的条件    |
    |-------|---------------|--------------------------|--------------------------|
    | 0     | start_room    | 向西出口推进             | 房间切换（→monster_hall）|
    | 1     | monster_hall  | 击杀怪物 + 向西出口推进  | 房间切换（→key_room）    |
    | 2     | key_room      | 找到宝箱并打开           | 钥匙到手（key_confirmed）|
    | 3     | key_room      | 向东出口推进（返回）     | 房间切换（→monster_hall）|
    | 4     | monster_hall  | 向东出口推进（返回）     | 房间切换（→start_room）  |
    | 5     | start_room    | 向东锁门出口推进         | 任务完成                 |
    """

    # ---- 世界阶段 ----
    world_stage: int = 0

    # ---- 子阶段（stage 1 的怪物战斗） ----
    sub_phase: str = "navigate"  # "navigate" | "to_monster" | "attack_monster"

    # ---- 像素感知移动状态（替代 task1 的 queued_actions） ----
    move_target_tile: Position | None = None
    move_action: int | None = None
    move_attempts: int = 0

    # ---- 延迟交互（转向后立刻按 A） ----
    pending_interact: bool = False

    # ---- 钥匙状态追踪 ----
    last_key_count: int = 0
    key_confirmed: bool = False

    # ---- 出口推进状态 ----
    exit_push_action: int | None = None
    target_exit_tile: Position | None = None

    # ---- 房间切换检测 ----
    last_player_tile: Position | None = None

    # ---- 怪物攻击追踪 ----
    monster_attack_count: int = 0
    monster_was_visible: bool = False

    # ---- 宝箱交互计数 ----
    chest_interactions: int = 0

    # ---- 卡住检测 ----
    stuck_counter: int = 0

    # ================================================================
    # 公开接口（评测脚本调用）
    # ================================================================

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed, task_id
        self.world_stage = 0
        self.sub_phase = "navigate"
        self.move_target_tile = None
        self.move_action = None
        self.move_attempts = 0
        self.pending_interact = False
        self.last_key_count = 0
        self.key_confirmed = False
        self.exit_push_action = None
        self.target_exit_tile = None
        self.last_player_tile = None
        self.monster_attack_count = 0
        self.monster_was_visible = False
        self.chest_interactions = 0
        self.stuck_counter = 0

    def act(self, obs, info=None) -> int:
        """根据像素观测和允许的物品栏信息输出一个环境动作。"""
        self._update_inventory_progress(info)
        vision = classify_frame_cnn(obs, fallback=False)
        player = None if vision.player is None else vision.player.tile
        if player is None:
            return ACTION_NOOP

        # ---- 保存上一帧位置（用于房间切换 + 卡住检测） ----
        prev_tile = self.last_player_tile

        # ---- 房间切换检测（必须在更新 last_player_tile 前，用旧坐标比较） ----
        if prev_tile is not None:
            dist = manhattan(player, prev_tile)
            if dist > ROOM_TRANSITION_THRESHOLD:
                self._advance_stage_on_room_change()

        # ---- 更新 last_player_tile ----
        self.last_player_tile = player

        # ---- 卡住检测 ----
        if prev_tile == player:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter > MAX_TILE_MOVE_ATTEMPTS * 2:
            self.move_target_tile = None
            self.move_action = None
            self.move_attempts = 0
            self.pending_interact = False
            self.exit_push_action = None
            self.stuck_counter = 0

        # ---- 更新子阶段 & 钥匙触发 ----
        self._update_sub_phase(vision)
        self._check_key_trigger()

        # ---- 延迟交互：上次是转向，这次按 A ----
        if self.pending_interact:
            self.pending_interact = False
            if self.sub_phase == "attack_monster":
                self.monster_attack_count += 1
            return ACTION_A

        # ---- 像素感知移动：持续朝目标 tile 移动 ----
        if self.move_target_tile is not None and self.move_action is not None:
            return self._continue_tile_move(vision)

        # ---- 按当前 world_stage 决策 ----
        action = self._act_by_stage(player, vision)
        return self._shield_action(action, vision)

    # ================================================================
    # 像素感知移动
    # ================================================================

    def _continue_tile_move(self, vision: PixelObservation) -> int:
        """继续上一次规划的 tile 间移动（纯视觉判断）。"""
        self.move_attempts += 1

        # 用视觉检测玩家 tile 是否已到达目标 —— 不读取隐藏的像素坐标。
        if vision.player is not None and vision.player.tile == self.move_target_tile:
            self.move_target_tile = None
            self.move_action = None
            self.move_attempts = 0
            return ACTION_NOOP

        if self.move_attempts >= MAX_TILE_MOVE_ATTEMPTS:
            self.move_target_tile = None
            self.move_action = None
            self.move_attempts = 0
            return ACTION_NOOP

        return self.move_action

    def _begin_tile_move(self, action: int, target_tile: Position) -> int:
        """开始朝目标 tile 移动，用视觉 tile 判断到达。"""
        self.move_target_tile = target_tile
        self.move_action = action
        self.move_attempts = 0
        return action

    # ================================================================
    # 阶段管理
    # ================================================================

    def _advance_stage_on_room_change(self) -> None:
        """房间切换时按预期推进 world_stage。"""
        if self.world_stage == 0:
            self.world_stage = 1
        elif self.world_stage == 1:
            self.world_stage = 2
        elif self.world_stage == 3:
            self.world_stage = 4
        elif self.world_stage == 4:
            self.world_stage = 5

        # 重置新房间的局部状态
        self.move_target_tile = None
        self.move_action = None
        self.move_attempts = 0
        self.pending_interact = False
        self.exit_push_action = None
        self.target_exit_tile = None
        self.sub_phase = "navigate"
        self.monster_attack_count = 0

    def _check_key_trigger(self) -> None:
        """钥匙到手后，从 stage 2（找宝箱）推进到 stage 3（返回东出口）。"""
        if (
            self.world_stage == 2
            and self.key_confirmed
            and self.move_target_tile is None
            and not self.pending_interact
        ):
            self.world_stage = 3
            self.sub_phase = "navigate"
            self.exit_push_action = None
            self.target_exit_tile = None

    def _update_sub_phase(self, vision: PixelObservation) -> None:
        """根据视觉观测更新战斗子阶段。"""
        monster_visible = len(vision.monsters) > 0

        if self.sub_phase == "attack_monster" and not monster_visible:
            self.sub_phase = "navigate"
            self.monster_attack_count = 0

        if self.sub_phase == "to_monster" and not monster_visible and self.monster_was_visible:
            self.sub_phase = "navigate"

        self.monster_was_visible = monster_visible

    def _update_inventory_progress(self, info) -> None:
        """从允许的物品栏视图中记录钥匙是否已经被确认获得。"""
        keys = inventory_key_count(info)
        if keys is None:
            return
        if keys > self.last_key_count or keys > 0:
            self.key_confirmed = True
        self.last_key_count = max(self.last_key_count, keys)

    # ================================================================
    # 各 world_stage 的动作决策
    # ================================================================

    def _act_by_stage(self, player: Position, vision: PixelObservation) -> int:
        """根据 world_stage 分派到对应的阶段处理器。"""
        if self.world_stage == 0:
            return self._act_exit_west(player, vision)
        elif self.world_stage == 1:
            return self._act_monster_hall_west(player, vision)
        elif self.world_stage == 2:
            return self._act_to_chest(player, vision)
        elif self.world_stage == 3:
            return self._act_exit_east(player, vision)
        elif self.world_stage == 4:
            return self._act_exit_east(player, vision)
        elif self.world_stage == 5:
            return self._act_exit_east(player, vision)
        return ACTION_NOOP

    # ================================================================
    # 阶段处理器：向西出口（stage 0 & stage 1 无怪物时）
    # ================================================================

    def _act_exit_west(self, player: Position, vision: PixelObservation) -> int:
        """导航到西边界（col=0）的出口并推出。"""
        return self._act_exit_directional(player, vision, direction="west")

    def _act_monster_hall_west(self, player: Position, vision: PixelObservation) -> int:
        """在 monster_hall 中：有怪物则战斗，无怪物则向西出口推进。"""
        monster_tiles = {m.tile for m in vision.monsters}

        if monster_tiles:
            return self._act_monster_combat(player, vision, monster_tiles)

        return self._act_exit_west(player, vision)

    # ================================================================
    # 阶段处理器：向东出口（stage 3, 4, 5）
    # ================================================================

    def _act_exit_east(self, player: Position, vision: PixelObservation) -> int:
        """导航到东边界（col=GRID_WIDTH-1）的出口并推出。"""
        return self._act_exit_directional(player, vision, direction="east")

    # ================================================================
    # 阶段处理器：宝箱交互（stage 2）
    # ================================================================

    def _act_to_chest(self, player: Position, vision: PixelObservation) -> int:
        """规划到可见宝箱旁边，相邻时执行交互。"""
        chest_tiles = self._tiles_of_kind(vision, {"chest"})
        if not chest_tiles:
            if self.key_confirmed:
                self.world_stage = 3
            return ACTION_NOOP

        adjacent = self._adjacent_targets(chest_tiles, vision)
        if player in adjacent:
            chest = min(chest_tiles, key=lambda tile: manhattan(player, tile))
            face_action = action_toward(player, chest)
            if face_action is not None:
                self.pending_interact = True
                self.chest_interactions += 1
                return face_action
            self.chest_interactions += 1
            return ACTION_A

        path = bfs_path(player, adjacent, vision)
        if len(path) >= 2:
            action = action_toward(path[0], path[1])
            if action is not None:
                return self._begin_tile_move(action, path[1])
        return ACTION_NOOP

    # ================================================================
    # 阶段处理器：怪物战斗（stage 1 子阶段）
    # ================================================================

    def _act_monster_combat(
        self, player: Position, vision: PixelObservation, monster_tiles: set[Position]
    ) -> int:
        """处理怪物战斗：近身则攻击，否则 BFS 接近。"""
        for mt in monster_tiles:
            if manhattan(player, mt) == 1:
                self.sub_phase = "attack_monster"
                break
        else:
            self.sub_phase = "to_monster"

        if self.sub_phase == "attack_monster":
            return self._act_attack_monster(player, monster_tiles)
        else:
            adjacent = self._adjacent_targets(monster_tiles, vision)
            path = bfs_path(player, adjacent, vision)
            if len(path) >= 2:
                action = action_toward(path[0], path[1])
                if action is not None:
                    return self._begin_tile_move(action, path[1])
            return ACTION_NOOP

    def _act_attack_monster(
        self, player: Position, monster_tiles: set[Position]
    ) -> int:
        """面对最近的怪物并使用剑（ACTION_A）攻击。"""
        if not monster_tiles:
            self.sub_phase = "navigate"
            return ACTION_NOOP

        closest = min(monster_tiles, key=lambda m: manhattan(player, m))
        dist = manhattan(player, closest)

        if dist == 1:
            face_action = action_toward(player, closest)
            if face_action is not None:
                self.pending_interact = True
                return face_action
            self.monster_attack_count += 1
            return ACTION_A

        self.sub_phase = "to_monster"
        return ACTION_NOOP

    # ================================================================
    # 方向性出口处理器
    # ================================================================

    def _act_exit_directional(
        self, player: Position, vision: PixelObservation, direction: str
    ) -> int:
        """导航到指定方向边界上的出口并推出房间。"""
        exit_tiles = self._tiles_of_kind(
            vision, {"exit_locked", "exit_normal", "exit_conditional"}
        )
        filtered = self._filter_exits_by_boundary(exit_tiles, direction)
        reachable = {tile for tile in filtered if is_walkable(tile, vision)}

        if self.exit_push_action is not None:
            if self._can_continue_exit_push(player, reachable):
                return self.exit_push_action
            self.exit_push_action = None
            self.target_exit_tile = None

        if not reachable:
            return ACTION_NOOP

        if player in reachable or (
            self.target_exit_tile is not None
            and player == self.target_exit_tile
            and is_boundary_tile(player)
        ):
            self.exit_push_action = boundary_exit_action(player)
            self.target_exit_tile = player
            return self.exit_push_action or ACTION_NOOP

        path = bfs_path(player, reachable, vision)
        if len(path) >= 2:
            self.target_exit_tile = path[-1]
            action = action_toward(path[0], path[1])
            if action is not None:
                return self._begin_tile_move(action, path[1])
        return ACTION_NOOP

    def _filter_exits_by_boundary(
        self, exit_tiles: set[Position], direction: str
    ) -> set[Position]:
        """只保留指定方向边界上的出口 tile。"""
        if direction == "west":
            return {t for t in exit_tiles if t[0] == 0}
        elif direction == "east":
            return {t for t in exit_tiles if t[0] == GRID_WIDTH - 1}
        elif direction == "north":
            return {t for t in exit_tiles if t[1] == 0}
        elif direction == "south":
            return {t for t in exit_tiles if t[1] == GRID_HEIGHT - 1}
        return exit_tiles

    def _can_continue_exit_push(
        self, player: Position, reachable_exit_tiles: set[Position]
    ) -> bool:
        """检查记住的出口推进动作当前是否仍然合理。"""
        return (
            self.target_exit_tile is not None
            and player == self.target_exit_tile
            and is_boundary_tile(player)
            and self.exit_push_action == boundary_exit_action(player)
            and (
                player in reachable_exit_tiles
                or player == self.target_exit_tile
            )
        )

    # ================================================================
    # Safety Shield
    # ================================================================

    def _shield_action(self, action: int, vision: PixelObservation) -> int:
        """在动作交给环境前，拦截不安全的移动动作。"""
        if action not in MOVE_ACTIONS:
            return action
        if vision.player is None:
            return ACTION_NOOP
        player = vision.player.tile

        # 允许出口推出动作（可能越出地图边界）
        if (
            self.exit_push_action == action
            and (
                is_exit_tile(player, vision)
                or (
                    self.target_exit_tile is not None
                    and player == self.target_exit_tile
                    and is_boundary_tile(player)
                )
            )
        ):
            return action

        nxt = next_position(player, action)
        if not in_bounds(nxt):
            return ACTION_NOOP

        # 宝箱交互时允许贴近宝箱
        if self.world_stage == 2 and vision.grid[nxt[1]][nxt[0]] == "chest":
            return action

        # 战斗中允许贴近怪物
        if self.world_stage == 1 and self.sub_phase in ("to_monster", "attack_monster"):
            if vision.grid[nxt[1]][nxt[0]] == "monster":
                return action

        if not is_walkable(nxt, vision):
            return ACTION_NOOP
        return action

    # ================================================================
    # 通用辅助方法
    # ================================================================

    def _tiles_of_kind(
        self, vision: PixelObservation, kinds: set[str]
    ) -> set[Position]:
        return {tile.tile for tile in vision.tiles if tile.kind in kinds}

    def _adjacent_targets(
        self, blocked_targets: set[Position], vision: PixelObservation
    ) -> set[Position]:
        out: set[Position] = set()
        for target in blocked_targets:
            for pos in neighbors(target):
                if in_bounds(pos) and is_walkable(pos, vision):
                    out.add(pos)
        return out


# ============================================================================
# BFS & 路径工具函数
# ============================================================================


def bfs_path(
    start: Position, goals: set[Position], vision: PixelObservation
) -> list[Position]:
    """用 BFS 找到从 start 到任一目标 tile 的最短符号路径。"""
    if start in goals:
        return [start]
    queue: deque[Position] = deque([start])
    parent: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        for nxt in neighbors(current):
            if nxt in parent:
                continue
            if nxt not in goals and not is_walkable(nxt, vision):
                continue
            if not in_bounds(nxt):
                continue
            parent[nxt] = current
            if nxt in goals:
                return reconstruct_path(parent, nxt)
            queue.append(nxt)
    return []


def reconstruct_path(
    parent: dict[Position, Position | None], goal: Position
) -> list[Position]:
    path: list[Position] = []
    current: Position | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def is_walkable(pos: Position, vision: PixelObservation) -> bool:
    if not in_bounds(pos):
        return False
    kind = vision.grid[pos[1]][pos[0]]
    if kind in BLOCKING_KINDS:
        return False
    return kind in SAFE_WALKABLE_KINDS


def neighbors(pos: Position) -> tuple[Position, Position, Position, Position]:
    col, row = pos
    return ((col, row - 1), (col, row + 1), (col - 1, row), (col + 1, row))


def in_bounds(pos: Position) -> bool:
    col, row = pos
    return 0 <= col < GRID_WIDTH and 0 <= row < GRID_HEIGHT


def is_boundary_tile(pos: Position) -> bool:
    col, row = pos
    return col == 0 or row == 0 or col == GRID_WIDTH - 1 or row == GRID_HEIGHT - 1


def is_exit_tile(pos: Position, vision: PixelObservation) -> bool:
    if not in_bounds(pos):
        return False
    return is_boundary_tile(pos) and vision.grid[pos[1]][pos[0]] in {
        "exit_locked",
        "exit_normal",
        "exit_conditional",
    }


def boundary_exit_action(pos: Position) -> int | None:
    col, row = pos
    if row == 0:
        return ACTION_UP
    if row == GRID_HEIGHT - 1:
        return ACTION_DOWN
    if col == 0:
        return ACTION_LEFT
    if col == GRID_WIDTH - 1:
        return ACTION_RIGHT
    return None


def inventory_key_count(info) -> int | None:
    if not isinstance(info, dict):
        return None
    inventory = info.get("inventory")
    if not isinstance(inventory, dict):
        return None
    try:
        return int(inventory.get("keys", 0) or 0)
    except (TypeError, ValueError):
        return None


def next_position(pos: Position, action: int) -> Position:
    dx, dy = ACTION_TO_DELTA[action]
    return (pos[0] + dx, pos[1] + dy)


def action_toward(current: Position, nxt: Position) -> int | None:
    delta = (nxt[0] - current[0], nxt[1] - current[1])
    return DELTA_TO_ACTION.get(delta)


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


# ============================================================================
# 评测接口
# ============================================================================

Policy = Task3Agent


def make_policy() -> Task3Agent:
    """评测脚本使用的策略工厂函数。"""
    return Task3Agent()
