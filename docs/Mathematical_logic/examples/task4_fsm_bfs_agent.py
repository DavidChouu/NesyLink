"""任务 4 的像素识别 + 多 Mission FSM + BFS 策略 —— 旋转桥、拿钥匙剑、杀怪、开最终宝箱。

整体思路：

1. 与 task1/2/3 相同，策略推理阶段只使用原始 RGB 图像 obs 和课程允许的显式物品栏信息。
2. 任务 4 是 5 房间十字结构 + 旋转桥：
      west (起点, 有开关) → center (旋转桥 + 深渊 + 隐藏最终宝箱)
      → north (钥匙宝箱) / east (剑宝箱, 需钥匙) / south (怪物)
   玩家初始无剑（slot A = none），必须按顺序完成：
     ① 北房间拿钥匙 → ② 东房间拿剑 → ③ 南房间杀怪 → ④ center 开最终宝箱
3. 旋转桥有 3 个状态（west_to_north → west_to_east → west_to_south），
   按 west 房间的开关可循环切换。桥 tile 分类为 "bridge"（可通过），
   其余 center 房间全是 "abyss"（不可通过）。BFS 自然只沿桥走，
   桥状态不对时 BFS 找不到路 → agent 回 west 再按一次开关。
4. 房间切换检测与 task3 相同（tile 跳跃 > 3）。
5. 使用像素感知移动（与 task2/task3 一致）。

Mission 序列：
   mission 0: 北拿钥匙  桥=west_to_north (0 presses)  目标：north 宝箱 → 返回 west
   mission 1: 东拿剑    桥=west_to_east  (1 press)    目标：east 宝箱（需钥匙）→ 返回 west
   mission 2: 南杀怪    桥=west_to_south (2 presses)  目标：south 怪物(hp=1)
             → 杀完直接北出口回 center → 开最终宝箱 → 完成！
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
    "wall", "chest", "trap", "abyss", "gap", "monster", "unknown",
}
SAFE_WALKABLE_KINDS = {
    "floor", "player", "bridge", "button", "switch",
    "exit_normal", "exit_locked", "exit_conditional", "npc",
}

ROOM_TRANSITION_THRESHOLD = 3
MAX_TILE_MOVE_ATTEMPTS = TILE_SIZE * 5  # 80

# 每趟 mission 需要的桥状态 → 需要按几次开关（累计）
MISSION_SWITCH_PRESSES = [0, 1, 2]  # mission 0=north, 1=east, 2=south

# 每趟 mission 的目标出口方向（从 center 出发）
MISSION_EXIT_DIRECTION = ["north", "east", "south"]

# 每趟 mission 的返回方向（从目标房间回 center）
# mission 2 不返回 west——杀完怪直接去 center 开最终宝箱
MISSION_RETURN_DIRECTION = ["south", "west", None]


# ============================================================================
# Agent 主体
# ============================================================================


@dataclass
class Task4Agent:
    """Task 4 像素识别 + Mission FSM + BFS + 像素感知移动 + safety shield 策略。

    Mission 序列：
      0: 北拿钥匙 → 返回 west
      1: 东拿剑   → 返回 west
      2: 南杀怪   → 直接 center 开最终宝箱
    """

    # ---- Mission 追踪 ----
    mission: int = 0
    # 子阶段: "to_switch" | "to_target" | "at_target" | "return_to_west" | "go_final_chest"
    mission_phase: str = "to_switch"

    # ---- 开关按次数 ----
    switch_presses_done: int = 0

    # ---- 像素感知移动（纯视觉：用 vision.player.tile 判断到达） ----
    move_target_tile: Position | None = None
    move_action: int | None = None
    move_attempts: int = 0

    # ---- 延迟交互 ----
    pending_interact: bool = False

    # ---- 钥匙 / 物品状态 ----
    last_key_count: int = 0
    key_confirmed: bool = False
    has_sword: bool = False  # 初始 slot A = none

    # ---- 出口推进 ----
    exit_push_action: int | None = None
    target_exit_tile: Position | None = None

    # ---- 房间切换检测 ----
    last_player_tile: Position | None = None

    # ---- 怪物攻击 ----
    monster_attack_count: int = 0
    monster_was_visible: bool = False

    # ---- 卡住检测 ----
    stuck_counter: int = 0
    stuck_player_tile: Position | None = None

    # ---- 宝箱/开关交互计数 ----
    interact_count: int = 0

    # ---- 已交互过的宝箱位置（打开后贴图仍被分类为 chest，需行为记忆绕过） ----
    interacted_positions: set[Position] = field(default_factory=set)

    # ================================================================
    # 公开接口
    # ================================================================

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed, task_id
        self.mission = 0
        self.mission_phase = "to_switch"
        self.switch_presses_done = 0
        self.move_target_tile = None
        self.move_action = None
        self.move_attempts = 0
        self.pending_interact = False
        self.last_key_count = 0
        self.key_confirmed = False
        self.has_sword = False
        self.exit_push_action = None
        self.target_exit_tile = None
        self.last_player_tile = None
        self.monster_attack_count = 0
        self.monster_was_visible = False
        self.stuck_counter = 0
        self.stuck_player_tile = None
        self.interact_count = 0
        self.interacted_positions.clear()

    def act(self, obs, info=None) -> int:
        """根据像素观测和允许的物品栏信息输出一个环境动作。"""
        self._update_inventory_progress(info)
        vision = classify_frame_cnn(obs, fallback=False)
        player = None if vision.player is None else vision.player.tile
        if player is None:
            return ACTION_NOOP

        # ---- 房间切换检测（必须在 last_player_tile 更新前） ----
        prev_tile = self.last_player_tile
        if prev_tile is not None:
            dist = manhattan(player, prev_tile)
            if dist > ROOM_TRANSITION_THRESHOLD:
                self._on_room_change()
        self.last_player_tile = player

        # ---- 卡住检测 ----
        if prev_tile == player:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > MAX_TILE_MOVE_ATTEMPTS * 3:
            self._on_stuck()

        # ---- 延迟交互 ----
        if self.pending_interact:
            self.pending_interact = False
            # 根据当前 phase 判断交互类型并更新计数器
            if self.mission_phase == "to_switch":
                self.switch_presses_done += 1
            elif self.mission_phase == "at_target" and self.mission == 2:
                self.monster_attack_count += 1
            return ACTION_A

        # ---- 像素感知移动：检查是否已到达目标 tile（纯视觉，不读 info） ----
        if self.move_target_tile is not None and self.move_action is not None:
            self.move_attempts += 1
            if player == self.move_target_tile:
                # 视觉确认已到达目标 tile
                self.move_target_tile = None
                self.move_action = None
                self.move_attempts = 0
            elif self.move_attempts >= MAX_TILE_MOVE_ATTEMPTS:
                # 超时 → 放弃重规划
                self.move_target_tile = None
                self.move_action = None
                self.move_attempts = 0
            else:
                align_action = self._boundary_alignment_action(vision)
                if align_action is not None:
                    return self._shield_action(align_action, vision)
                return self._shield_action(self.move_action, vision)

        # ---- 按 mission + phase 决策 ----
        action = self._act_by_mission(player, vision)
        return self._shield_action(action, vision)

    # ================================================================
    # 像素感知移动（纯视觉：用 classify_frame 的 player.tile 判断到达）
    # ================================================================

    def _begin_tile_move(self, action: int, target_tile: Position) -> int:
        """开始朝目标 tile 移动，视觉确认到达时自动停止。"""
        self.move_target_tile = target_tile
        self.move_action = action
        self.move_attempts = 0
        return action

    def _boundary_alignment_action(self, vision: PixelObservation) -> int | None:
        """Before pushing through edge exits, align the perpendicular pixel axis.

        A player can be classified into the correct tile while its sprite still
        overlaps the neighboring wall row/column. Aligning the center prevents
        corner collision from blocking the exit push.
        """

        if self.move_target_tile is None or self.move_action is None or vision.player is None:
            return None
        target_x, target_y = self.move_target_tile
        center_x, center_y = vision.player.center_px
        tolerance = 0.5

        if target_x in {0, GRID_WIDTH - 1} and self.move_action in {ACTION_LEFT, ACTION_RIGHT}:
            desired_y = target_y * TILE_SIZE + TILE_SIZE * 0.5
            if center_y < desired_y - tolerance:
                return ACTION_DOWN
            if center_y > desired_y + tolerance:
                return ACTION_UP

        if target_y in {0, GRID_HEIGHT - 1} and self.move_action in {ACTION_UP, ACTION_DOWN}:
            desired_x = target_x * TILE_SIZE + TILE_SIZE * 0.5
            if center_x < desired_x - tolerance:
                return ACTION_RIGHT
            if center_x > desired_x + tolerance:
                return ACTION_LEFT
        return None

    # ================================================================
    # 房间切换 & 卡住处理
    # ================================================================

    def _on_room_change(self) -> None:
        """房间切换时只重置局部状态，不改变 phase。

        Phase 的推进由各 phase handler 根据当前房间的视觉特征自行判断。
        因为 to_target 和 return_to_west 都需要跨越两个房间（west→center→目标），
        不能在每次房间切换时盲目推进。
        """
        self.move_target_tile = None
        self.move_action = None
        self.move_attempts = 0
        self.pending_interact = False
        self.exit_push_action = None
        self.target_exit_tile = None
        self.stuck_counter = 0

    def _on_stuck(self) -> None:
        """卡住时重置所有移动状态。"""
        self.move_target_tile = None
        self.move_action = None
        self.move_attempts = 0
        self.pending_interact = False
        self.exit_push_action = None
        self.target_exit_tile = None
        self.stuck_counter = 0

    # ================================================================
    # 物品栏追踪
    # ================================================================

    def _update_inventory_progress(self, info) -> None:
        """追踪钥匙数量和是否获得了剑。"""
        keys = inventory_key_count(info)
        if keys is not None:
            if keys > self.last_key_count or keys > 0:
                self.key_confirmed = True
            self.last_key_count = max(self.last_key_count, keys)

        # 检测剑：检查 equipped slot A
        if not self.has_sword and isinstance(info, dict):
            inventory = info.get("inventory")
            if isinstance(inventory, dict):
                equipped = inventory.get("equipped")
                if isinstance(equipped, dict):
                    if equipped.get("A") == "sword":
                        self.has_sword = True

    # ================================================================
    # Mission 决策分发
    # ================================================================

    def _act_by_mission(self, player: Position, vision: PixelObservation) -> int:
        """根据当前 mission + phase 决策。"""
        if self.mission_phase == "to_switch":
            return self._act_press_switch(player, vision)
        elif self.mission_phase == "to_target":
            return self._act_go_to_target(player, vision)
        elif self.mission_phase == "at_target":
            return self._act_at_target(player, vision)
        elif self.mission_phase == "return_to_west":
            return self._act_return_to_west(player, vision)
        elif self.mission_phase == "go_final_chest":
            return self._act_go_final_chest(player, vision)
        return ACTION_NOOP

    # ================================================================
    # Phase: to_switch —— 在 west 房间按开关
    # ================================================================

    def _act_press_switch(self, player: Position, vision: PixelObservation) -> int:
        """在 west 房间找到开关并按到需要的次数。"""
        needed = MISSION_SWITCH_PRESSES[self.mission]

        if self.switch_presses_done >= needed:
            # 开关已按够 → 去东出口进 center
            self.mission_phase = "to_target"
            return ACTION_NOOP

        # 找开关 tile（分类为 "switch"）
        switch_tiles = self._tiles_of_kind(vision, {"switch"})
        if not switch_tiles:
            # 视觉暂时没检测到开关 → 尝试去东出口
            self.mission_phase = "to_target"
            return ACTION_NOOP

        switch_pos = next(iter(switch_tiles))

        # 检查是否在开关旁边
        if manhattan(player, switch_pos) == 1:
            face_action = action_toward(player, switch_pos)
            if face_action is not None:
                self.pending_interact = True
                self.interact_count += 1
                return face_action
            self.interact_count += 1
            return ACTION_A

        # BFS 到开关旁边
        adjacent = self._adjacent_positions({switch_pos}, vision)
        path = bfs_path(player, adjacent, vision)
        if len(path) >= 2:
            action = action_toward(path[0], path[1])
            if action is not None:
                return self._begin_tile_move(action, path[1])
        return ACTION_NOOP

    # ================================================================
    # Phase: to_target —— 从 west 穿越 center 去目标房间
    # ================================================================

    def _act_go_to_target(self, player: Position, vision: PixelObservation) -> int:
        """穿越 center 去目标房间。根据视觉判断当前在哪个房间。"""
        room = self._detect_room(vision)

        if room == "west":
            # 还在 west → 去东出口进入 center
            return self._act_exit_directional(player, vision, "east")
        elif room == "center":
            # 在 center → 导航到目标出口
            direction = MISSION_EXIT_DIRECTION[self.mission]
            return self._act_exit_directional(player, vision, direction)
        else:
            # 已到达目标房间（north/east/south）
            self.mission_phase = "at_target"
            return ACTION_NOOP

    # ================================================================
    # Phase: at_target —— 在目标房间完成任务
    # ================================================================

    def _act_at_target(self, player: Position, vision: PixelObservation) -> int:
        """在目标房间完成子任务（开宝箱 / 杀怪物）。"""
        if self.mission == 0:
            # 北房间：开钥匙宝箱
            return self._act_open_chest(player, vision)
        elif self.mission == 1:
            # 东房间：开剑宝箱
            return self._act_open_chest(player, vision)
        elif self.mission == 2:
            # 南房间：杀怪物
            return self._act_kill_monster(player, vision)
        return ACTION_NOOP

    # ================================================================
    # Phase: return_to_west —— 从目标房间穿越 center 回 west
    # ================================================================

    def _act_return_to_west(self, player: Position, vision: PixelObservation) -> int:
        """从目标房间穿越 center 回 west。根据视觉判断当前在哪个房间。"""
        room = self._detect_room(vision)

        if room == "west":
            # 已回到 west → 进入下一 mission（mission 2 不会走到这里）
            self.mission += 1
            self.mission_phase = "to_switch"
            return ACTION_NOOP
        elif room == "center":
            # 在 center → 找 west 出口回 west
            return self._act_exit_directional(player, vision, "west")
        else:
            # 在目标房间 → 找返回出口回 center
            ret_dir = MISSION_RETURN_DIRECTION[self.mission]
            return self._act_exit_directional(player, vision, ret_dir)

    def _detect_room(self, vision: PixelObservation) -> str:
        """根据视觉特征判断当前房间类型。"""
        if self._tiles_of_kind(vision, {"switch"}):
            return "west"
        if self._tiles_of_kind(vision, {"bridge"}) or len(self._tiles_of_kind(vision, {"abyss"})) > 10:
            return "center"
        return "target"

    # ================================================================
    # Phase: go_final_chest —— 杀完南怪物后直接去 center 开最终宝箱
    # ================================================================

    def _act_go_final_chest(self, player: Position, vision: PixelObservation) -> int:
        """杀完南房间怪物后：从南→center→找到并打开最终宝箱。"""
        room = self._detect_room(vision)

        if room == "target":
            # 还在南房间 → 去北出口进 center
            return self._act_exit_directional(player, vision, "north")

        # 在 center（或 west）→ 找宝箱
        chest_tiles = self._tiles_of_kind(vision, {"chest"})
        if chest_tiles:
            return self._act_open_chest_at(player, vision, chest_tiles)

        # 宝箱还没视觉出现 → 走向 center 中央 (4,4) 等待揭示
        if player != (4, 4):
            path = bfs_path(player, {(4, 4)}, vision)
            if len(path) >= 2:
                action = action_toward(path[0], path[1])
                if action is not None:
                    return self._begin_tile_move(action, path[1])
        return ACTION_NOOP

    # ================================================================
    # 共享行为：开宝箱
    # ================================================================

    def _act_open_chest(self, player: Position, vision: PixelObservation) -> int:
        """导航到宝箱旁边并打开。用行为记忆区分开/关宝箱。"""
        chest_tiles = self._tiles_of_kind(vision, {"chest"})

        # 过滤掉已经交互过的宝箱（打开后贴图仍在，但不应再尝试）
        unopened = chest_tiles - self.interacted_positions

        if not unopened:
            # 所有可见宝箱都已交互过 → 返回
            self.mission_phase = "return_to_west"
            return ACTION_NOOP
        return self._act_open_chest_at(player, vision, unopened)

    def _act_open_chest_at(
        self, player: Position, vision: PixelObservation, chest_tiles: set[Position]
    ) -> int:
        """在已知宝箱位置的情况下导航并打开。"""
        adjacent = self._adjacent_positions(chest_tiles, vision)
        if player in adjacent:
            chest = min(chest_tiles, key=lambda t: manhattan(player, t))
            # ★ 标记为已交互：即使宝箱贴图仍在，下次不再尝试
            self.interacted_positions.add(chest)
            face_action = action_toward(player, chest)
            if face_action is not None:
                self.pending_interact = True
                return face_action
            return ACTION_A

        path = bfs_path(player, adjacent, vision)
        if len(path) >= 2:
            action = action_toward(path[0], path[1])
            if action is not None:
                return self._begin_tile_move(action, path[1])
        return ACTION_NOOP

    # ================================================================
    # 共享行为：杀怪物
    # ================================================================

    def _act_kill_monster(self, player: Position, vision: PixelObservation) -> int:
        """在南房间找到并击杀怪物（hp=1）。击杀后自动切换到 return_to_west。"""
        monster_tiles = {m.tile for m in vision.monsters}

        if not monster_tiles:
            # 怪物已死 → mission 2 直接去找最终宝箱，不回 west
            if self.monster_was_visible:
                self.mission_phase = "go_final_chest"
                return ACTION_NOOP
            self.mission_phase = "go_final_chest"
            return ACTION_NOOP

        self.monster_was_visible = True

        # 检查是否相邻
        for mt in monster_tiles:
            if manhattan(player, mt) == 1:
                # 相邻 → 攻击
                face_action = action_toward(player, mt)
                if face_action is not None:
                    self.pending_interact = True
                    return face_action
                return ACTION_A

        # BFS 到怪物旁边
        adjacent = self._adjacent_positions(monster_tiles, vision)
        path = bfs_path(player, adjacent, vision)
        if len(path) >= 2:
            action = action_toward(path[0], path[1])
            if action is not None:
                return self._begin_tile_move(action, path[1])
        return ACTION_NOOP

    # ================================================================
    # 共享行为：方向性出口
    # ================================================================

    def _act_exit_directional(
        self, player: Position, vision: PixelObservation, direction: str
    ) -> int:
        """导航到指定方向边界出口并推出房间。"""
        exit_tiles = self._tiles_of_kind(
            vision, {"exit_locked", "exit_normal", "exit_conditional"}
        )
        filtered = self._filter_exits_by_boundary(exit_tiles, direction)

        # ★ Fallback：桥/深渊/墙等复杂背景下，exit 可能被视觉漏掉。
        # 此时把目标边界上所有 walkable tile 当作候选（如 bridge tile 上的锁门）。
        if not filtered:
            all_boundary = _boundary_tiles(direction)
            filtered = {t for t in all_boundary if is_walkable(t, vision)}

        reachable = {t for t in filtered if is_walkable(t, vision)}

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
        return (
            self.target_exit_tile is not None
            and player == self.target_exit_tile
            and is_boundary_tile(player)
            and self.exit_push_action == boundary_exit_action(player)
            and (player in reachable_exit_tiles or player == self.target_exit_tile)
        )

    # ================================================================
    # Safety Shield
    # ================================================================

    def _shield_action(self, action: int, vision: PixelObservation) -> int:
        if action not in MOVE_ACTIONS:
            return action
        if vision.player is None:
            return ACTION_NOOP
        player = vision.player.tile

        # 允许出口推出
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

        # 允许贴近交互目标（宝箱、开关、怪物）
        nxt_kind = vision.grid[nxt[1]][nxt[0]]
        if nxt_kind in ("chest", "switch", "monster"):
            return action

        if not is_walkable(nxt, vision):
            return ACTION_NOOP
        return action

    # ================================================================
    # 通用辅助
    # ================================================================

    def _tiles_of_kind(self, vision: PixelObservation, kinds: set[str]) -> set[Position]:
        return {tile.tile for tile in vision.tiles if tile.kind in kinds}

    def _adjacent_positions(
        self, positions: set[Position], vision: PixelObservation
    ) -> set[Position]:
        out: set[Position] = set()
        for pos in positions:
            for nb in neighbors(pos):
                if in_bounds(nb) and is_walkable(nb, vision):
                    out.add(nb)
        return out


# ============================================================================
# BFS & 路径工具函数
# ============================================================================

def bfs_path(start: Position, goals: set[Position], vision: PixelObservation) -> list[Position]:
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

def reconstruct_path(parent: dict[Position, Position | None], goal: Position) -> list[Position]:
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

def _boundary_tiles(direction: str) -> set[Position]:
    """返回指定方向的所有边界 tile 坐标（作为视觉漏检出口时的 fallback）。"""
    if direction == "west":
        return {(0, r) for r in range(GRID_HEIGHT)}
    elif direction == "east":
        return {(GRID_WIDTH - 1, r) for r in range(GRID_HEIGHT)}
    elif direction == "north":
        return {(c, 0) for c in range(GRID_WIDTH)}
    elif direction == "south":
        return {(c, GRID_HEIGHT - 1) for c in range(GRID_WIDTH)}
    return set()


def is_boundary_tile(pos: Position) -> bool:
    col, row = pos
    return col == 0 or row == 0 or col == GRID_WIDTH - 1 or row == GRID_HEIGHT - 1

def is_exit_tile(pos: Position, vision: PixelObservation) -> bool:
    if not in_bounds(pos):
        return False
    return is_boundary_tile(pos) and vision.grid[pos[1]][pos[0]] in {
        "exit_locked", "exit_normal", "exit_conditional",
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

Policy = Task4Agent


def make_policy() -> Task4Agent:
    return Task4Agent()
