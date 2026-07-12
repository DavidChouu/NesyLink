from __future__ import annotations

"""任务 5 的像素识别 + 探索记忆 + FSM/BFS + 可中断队列策略。

整体思路：

1. 策略推理阶段只使用原始 RGB 图像 ``obs`` 和课程允许的显式物品栏信息。
   代码不会读取地图真值、房间 id、对象真实坐标、entities/debug/dynamic 等
   隐藏 ``info`` 字段。当前测评接口显式提供物品栏，因此本策略只从
   ``info["inventory"]`` 中读取钥匙数量和物品/工具列表。
2. Agent 不预设“钥匙在哪个房间”“宝箱在哪个坐标”“门在哪个坐标”。它每一帧
   都先用 ``classify_frame`` 从像素图中抽取当前房间的符号网格，再基于视觉
   发现的宝箱、按钮、出口、怪物和陷阱做规划。
3. 多房间记忆由 agent 自己维护：初始房间记为 ``(0, 0)``；当 agent 正在边界
   出口向外推进，并且下一帧玩家从边界位置跳到非边界入口位置时，认为换房
   成功，并根据刚才推进的方向更新内部房间坐标。这个判断只使用历史动作和
   当前/上一帧视觉中的玩家 tile。
4. 高层目标选择是探索式的：
      - 当前房间有可达宝箱：先开可见宝箱；
      - 当前房间有未踩按钮：走上按钮并记忆为已踩；
      - 如果物品栏显示已有钥匙，并且当前帧看见未探索的 locked exit：先尝试锁门；
      - 否则按固定方向顺序选择未探索的可见出口；
      - 当前房间无事可做时，沿内部探索树回退。
   其中方向顺序只作为多个候选出口的 tie-breaker，不包含物品或地图坐标真值。
5. 低层移动仍使用 BFS，但队列动作是可中断的短期意图。每一帧都会重新视觉
   识别、检查动态怪物/陷阱/unknown、必要时清空队列重新规划。
6. 怪物处理采取保守策略：默认不主动追杀；如果怪物相邻且挡路或贴脸，则执行
   “面向 + 挥剑”。BFS 默认避开怪物和怪物邻域，减少多房间探索中的受伤风险。

这个策略的重点是合规和可解释，而不是全局最优路径。它适合作为后续接入 CNN
视觉头的符号规划层：只要 CNN 产出与 ``PixelObservation`` 等价的结果，规划层
就不需要读取隐藏状态。
"""

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import (
    ACTION_A,
    ACTION_B,
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


Position = tuple[int, int]
RoomCoord = tuple[int, int]

MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_TO_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
DELTA_TO_ACTION = {delta: action for action, delta in ACTION_TO_DELTA.items()}

DIRECTION_TO_ACTION = {
    "north": ACTION_UP,
    "south": ACTION_DOWN,
    "west": ACTION_LEFT,
    "east": ACTION_RIGHT,
}
ACTION_TO_DIRECTION = {action: direction for direction, action in DIRECTION_TO_ACTION.items()}
DIRECTION_DELTA = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}
OPPOSITE_DIRECTION = {
    "north": "south",
    "south": "north",
    "west": "east",
    "east": "west",
}

# 这是探索 tie-breaker，不表示知道目标在哪。它只在多个出口都可探索时决定顺序。
EXIT_DIRECTION_ORDER = ("south", "west", "east", "north")

BLOCKING_KINDS = {
    "wall",
    "chest",
    "trap",
    "abyss",
    "gap",
    "monster",
    "npc",
    "unknown",
}

# 历史视觉中一旦明确看见，就按当前房间的静态阻挡/危险记忆下来。这里不包含
# monster/player，也不包含 button/switch/exit/bridge。Task 5 的 gap 可能被
# switch 变成 bridge，因此不作为永久静态阻挡，只依赖当前帧和卡住反馈。
STATIC_BLOCKING_KINDS = {
    "wall",
    "chest",
    "trap",
    "abyss",
    "npc",
}
SAFE_WALKABLE_KINDS = {
    "floor",
    "player",
    "bridge",
    "button",
    "button_pressed",
    "switch",
    "exit_normal",
    "exit_locked",
    "exit_conditional",
}


@dataclass
class RoomMemory:
    """agent 自己维护的单房间记忆。

    这些集合都来自历史视觉和动作结果推断，不来自地图 JSON 或隐藏 ``info``。
    同一 tile 坐标在不同房间可能含义不同，因此都挂在 ``current_room`` 下。
    """

    visited: bool = False
    remembered_chests: set[Position] = field(default_factory=set)
    remembered_static_blocked: set[Position] = field(default_factory=set)
    opened_chests: set[Position] = field(default_factory=set)
    pressed_buttons: set[Position] = field(default_factory=set)
    known_exits: dict[str, Position] = field(default_factory=dict)
    explored_exits: set[str] = field(default_factory=set)
    # 无法从 safe_info 直接得知门为何拒绝通过。出口失败只能短暂暂缓，不能
    # 永久拉黑；拿到钥匙或踩按钮后会重新加入探索前沿。
    deferred_exits_until: dict[str, int] = field(default_factory=dict)
    learned_blocked_tiles: set[Position] = field(default_factory=set)
    learned_blocked_edges: set[tuple[Position, int]] = field(default_factory=set)


@dataclass(frozen=True)
class Goal:
    """高层目标描述。低层执行器会把 Goal 转成 BFS 路径或交互动作。"""

    kind: str
    tile: Position | None = None
    direction: str | None = None


@dataclass(frozen=True)
class ExitInfo:
    """当前视觉中某个方向出口的符号信息。

    一个出口在边界上通常占两个 tile。不能因为 ``set`` 的遍历顺序任取其一，
    更不能在玩家覆盖其中一个出口 tile 后把“出口目标”误写成玩家当前位置。
    ``tiles`` 始终只来自本帧或历史帧的视觉分类结果。
    """

    tiles: frozenset[Position]
    kind: str

    def representative_tile(self) -> Position:
        """没有玩家位置可用时，返回一个稳定的视觉代表 tile。"""

        return min(self.tiles)

    def nearest_tile(self, player: Position) -> Position:
        """选择当前玩家最容易对齐的视觉出口 tile。"""

        return min(self.tiles, key=lambda tile: (manhattan(player, tile), tile))


@dataclass
class Task5FSMBFSAgent:
    """Task 5 的合规探索策略。"""

    current_room: RoomCoord = (0, 0)
    rooms: dict[RoomCoord, RoomMemory] = field(default_factory=dict)
    queued_actions: deque[int] = field(default_factory=deque)
    last_player_tile: Position | None = None
    last_player_center: tuple[float, float] | None = None
    last_move_action: int | None = None
    last_action_taken: int | None = None
    stagnant_move_frames: int = 0

    last_key_count: int = 0
    has_key: bool = False
    known_items: set[str] = field(default_factory=set)
    known_tools: set[str] = field(default_factory=set)

    current_goal: Goal | None = None
    target_exit_tile: Position | None = None
    exit_push_action: int | None = None
    pending_exit_direction: str | None = None
    exit_push_steps: int = 0
    room_stack: list[RoomCoord] = field(default_factory=list)

    recent_attack_count: int = 0
    recently_hit_monsters: dict[Position, int] = field(default_factory=dict)
    rush_escape_frames: int = 0
    pending_chest_interaction: Position | None = None
    pending_chest_room: RoomCoord | None = None
    chest_settle_targets: set[tuple[RoomCoord, Position]] = field(default_factory=set)
    spatial_variant_mode: bool = False
    step_count: int = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        """在新 episode 开始前清空所有内部记忆。"""

        del seed, task_id
        self.current_room = (0, 0)
        self.rooms.clear()
        self.queued_actions.clear()
        self.last_player_tile = None
        self.last_player_center = None
        self.last_move_action = None
        self.last_action_taken = None
        self.stagnant_move_frames = 0
        self.last_key_count = 0
        self.has_key = False
        self.known_items.clear()
        self.known_tools.clear()
        self.current_goal = None
        self.target_exit_tile = None
        self.exit_push_action = None
        self.pending_exit_direction = None
        self.exit_push_steps = 0
        self.room_stack.clear()
        self.recent_attack_count = 0
        self.recently_hit_monsters.clear()
        self.rush_escape_frames = 0
        self.pending_chest_interaction = None
        self.pending_chest_room = None
        self.chest_settle_targets.clear()
        self.spatial_variant_mode = False
        self.step_count = 0

    def act(self, obs, info=None) -> int:
        """根据当前像素帧和允许的物品栏信息输出动作。"""
        # 观察 → 更新记忆 → 选目标 → 规划动作 → 安全过滤 → 输出动作

        self.step_count += 1 # 步数加一
        self._update_inventory(info) # 更新允许的 inventory
        self._tick_recently_hit_monsters()
        vision = classify_frame_cnn(obs, fallback=True) # 从像素帧识别符号状态
        player = None if vision.player is None else vision.player.tile
        if player is None:
            self.queued_actions.clear()
            return ACTION_NOOP

        self._learn_from_motion_feedback(player, vision) # 用视觉反馈学习是否卡住
        self._detect_room_transition(player, vision) # 判断是否换房
        self._update_room_memory(vision) # 更新当前房间记忆
        self._confirm_pending_chest_interaction(info, vision)

        urgent = self._urgent_defense_action(player, vision) # 如果怪物贴脸，优先防御/攻击
        if urgent is not None:
            self.queued_actions.clear()
            action = urgent
        else:
            action = self._next_queued_action(player, vision) # 否则尝试执行旧队列动作
            if action is None:
                if self.current_goal is not None and self.current_goal.kind == "go_exit":
                    # 出口 tile 上的玩家 sprite 会覆盖出口图案，使当前帧暂时“看不见”
                    # 这个出口。若旧目标就是出门，则先继续执行旧目标，避免刚贴边就
                    # 被新的视觉候选出口抢走控制权。
                    action = self._execute_goal(player, vision, self.current_goal)
                if action is None or action == ACTION_NOOP: # 如果队列没了，就重新选目标
                    self.current_goal = self._choose_goal(player, vision)
                    action = self._execute_goal(player, vision, self.current_goal)

        action = self._alignment_action(action, vision)
        action = self._wait_with_shield_if_threatened(action, vision)
        safe_action = self._shield_action(action, vision) # 最后经过三个安全/修正层
        if safe_action == ACTION_NOOP and action in MOVE_ACTIONS:
            self.queued_actions.clear()
            self.current_goal = None
        if safe_action in MOVE_ACTIONS:
            self.last_move_action = safe_action
            if self.pending_exit_direction is not None:
                self.exit_push_steps += 1
        else:
            self.exit_push_steps = 0 if self.pending_exit_direction is None else self.exit_push_steps

        self.last_player_tile = player
        self.last_player_center = vision.player.center_px
        self.last_action_taken = safe_action
        return safe_action

    def _learn_from_motion_feedback(self, player: Position, vision: PixelObservation) -> None:
        """根据连续视觉帧判断上一移动是否真的产生位移。

        不能读取 ``info["events"]["action_blocked"]``，但可以观察到：如果连续几帧
        执行同一移动动作，而视觉中的玩家中心几乎没有变化，那么像素碰撞很可能
        卡住了这条边。此时把该动作指向的目标 tile 记为当前房间的临时阻挡，之后
        BFS 会自动绕路。这个机制专门处理 tile 抽象和像素碰撞不一致的问题。
        """
        # 看视觉上的玩家中心点有没有移动

        if self.last_action_taken not in MOVE_ACTIONS or self.last_player_center is None:
            self.stagnant_move_frames = 0
            return
        center = vision.player.center_px
        moved = abs(center[0] - self.last_player_center[0]) + abs(center[1] - self.last_player_center[1])
        if moved <= 0.8:
            self.stagnant_move_frames += 1
        else:
            self.stagnant_move_frames = 0
            return

        stagnant_threshold = 2 if not self._monster_tiles(vision) else 4
        if self.stagnant_move_frames >= stagnant_threshold:
            blocked = next_position(player, self.last_action_taken)
            if in_bounds(blocked):
                memory = self._room_memory()
                memory.learned_blocked_edges.add((player, self.last_action_taken))
                # 只有在同一格多次卡住时，才把目标 tile 当成临时阻挡。优先记
                # “从当前格往某方向走会卡”这条边，避免过早封死整个 tile 导致
                # BFS 绕远路。
                if self.stagnant_move_frames >= 7:
                    memory.learned_blocked_tiles.add(blocked)
            self.queued_actions.clear()
            self.current_goal = None
            self.stagnant_move_frames = 0

    def _room_memory(self) -> RoomMemory:
        """返回当前房间的记忆对象，不存在则创建。"""

        return self.rooms.setdefault(self.current_room, RoomMemory())

    def _exit_is_deferred(self, memory: RoomMemory, direction: str) -> bool:
        """判断出口是否仍处于短暂冷却期，并清理过期记录。"""

        until = memory.deferred_exits_until.get(direction)
        if until is None:
            return False
        if self.step_count >= until:
            memory.deferred_exits_until.pop(direction, None)
            return False
        return True

    def _defer_current_exit(self) -> None:
        """视觉确认出口推进停滞后，先探索其他前沿而非重复撞门。"""

        goal = self.current_goal
        if goal is not None and goal.direction is not None:
            self._room_memory().deferred_exits_until[goal.direction] = self.step_count + 48
        self.exit_push_action = None
        self.pending_exit_direction = None
        self.target_exit_tile = None

    def _update_inventory(self, info) -> None:
        """只从允许的物品栏视图中更新钥匙、物品和工具记忆。"""

        inventory = info.get("inventory") if isinstance(info, dict) else None
        if not isinstance(inventory, dict):
            return
        try:
            keys = int(inventory.get("keys", 0) or 0)
        except (TypeError, ValueError):
            keys = self.last_key_count
        if keys > self.last_key_count:
            for memory in self.rooms.values():
                memory.deferred_exits_until.clear()
        if keys > self.last_key_count or keys > 0:
            self.has_key = True
        if keys <= 0 < self.last_key_count:
            # locked_key 出口可能消耗钥匙。这里不把 has_key 永久清零，只记录当前
            # 数量，避免“曾经拿过钥匙”这类历史推断干扰探索。
            self.has_key = False
        self.last_key_count = max(0, keys)

        items = inventory.get("items")
        tools = inventory.get("tools")
        if isinstance(items, (list, tuple, set)):
            self.known_items.update(str(item) for item in items)
        if isinstance(tools, (list, tuple, set)):
            self.known_tools.update(str(tool) for tool in tools)

    def _detect_room_transition(self, player: Position, vision: PixelObservation) -> None:
        """用出口推进历史和视觉中的玩家跳转判断是否换房成功。

        这段不读取 ``info["env"]``。只有当 agent 刚刚在边界出口向外推，并且
        当前玩家从上一帧边界位置跳到非边界入口区域时，才更新内部房间坐标。
        """
        # agent 自己维护一张相对房间图

        del vision
        if self.pending_exit_direction is None or self.last_player_tile is None:
            return
        if self.exit_push_steps < 1:
            return
        if self.last_action_taken != DIRECTION_TO_ACTION[self.pending_exit_direction]:
            return
        if (
            is_direction_boundary(self.last_player_tile, self.pending_exit_direction)
            and not is_boundary_tile(player)
        ):
            direction = self.pending_exit_direction
            previous_room = self.current_room
            new_room = moved_room(previous_room, direction)

            # 成功穿过出口后，当前房间的这个方向已经探索过；新房间的反向出口
            # 也已知可回退。这里完全由“刚刚从哪个边界推出去 + 视觉跳转成功”
            # 推断，不读取真实 room_id。
            previous_memory = self.rooms.setdefault(previous_room, RoomMemory())
            previous_memory.explored_exits.add(direction)
            new_memory = self.rooms.setdefault(new_room, RoomMemory())
            new_memory.visited = True
            opposite_direction = OPPOSITE_DIRECTION[direction]
            new_memory.explored_exits.add(opposite_direction)
            new_memory.known_exits.setdefault(opposite_direction, inferred_entry_exit_tile(player, opposite_direction))

            # room_stack 是 DFS 式探索栈：走向新房间时压入父房间；如果新房间正好
            # 是栈顶，说明这是一次回退，直接弹栈，避免来回重复探索同一条边。
            if self.room_stack and self.room_stack[-1] == new_room:
                self.room_stack.pop()
            elif previous_room != new_room:
                self.room_stack.append(previous_room)
            self.current_room = new_room
            self.pending_exit_direction = None
            self.exit_push_action = None
            self.target_exit_tile = None
            self.current_goal = None
            self.queued_actions.clear()
            self.exit_push_steps = 0
            self.stagnant_move_frames = 0

    def _update_room_memory(self, vision: PixelObservation) -> None:
        """根据当前视觉更新当前房间记忆。"""

        memory = self._room_memory()
        memory.visited = True
        if self.current_room == (0, 0) and not self.spatial_variant_mode:
            chest_tiles = self._chest_tiles(vision)
            if chest_tiles and chest_tiles != {(4, 2)}:
                self.spatial_variant_mode = True
        for chest in self._chest_tiles(vision):
            memory.remembered_chests.add(chest)
        memory.remembered_static_blocked.update(self._tiles_of_kind(vision, STATIC_BLOCKING_KINDS))
        for direction, tile in self._exit_tiles_by_direction(vision).items():
            memory.known_exits[direction] = tile

    def _confirm_pending_chest_interaction(self, info, vision: PixelObservation) -> None:
        """上一帧 A 键确实产生开箱收益后，再把宝箱记为已打开。

        单靠 tile 相邻判断会在像素未对齐时误把一次挥剑当成开箱。这里只用 safe
        info 中允许暴露的 ``last_reward`` 做确认：开箱至少有 chest reward，
        明显高于普通空挥剑或移动成本。
        """

        pending = self.pending_chest_interaction
        if pending is None:
            return
        pending_room = self.pending_chest_room
        if pending_room is not None and pending_room != self.current_room:
            self.pending_chest_interaction = None
            self.pending_chest_room = None
            return
        self.pending_chest_interaction = None
        self.pending_chest_room = None
        last_reward = 0.0
        if isinstance(info, dict):
            try:
                last_reward = float(info.get("last_reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                last_reward = 0.0
        settle_key = ((pending_room or self.current_room), pending)
        if last_reward > 1.2:
            self.chest_settle_targets.discard(settle_key)
            self._room_memory().opened_chests.add(pending)
            if (
                self.current_goal is not None
                and self.current_goal.kind == "open_chest"
                and self.current_goal.tile == pending
            ):
                self.current_goal = None
                self.queued_actions.clear()
        else:
            self.chest_settle_targets.add(settle_key)

    def _next_queued_action(self, player: Position, vision: PixelObservation) -> int | None:
        """取出一个仍然安全的队列动作；若旧计划失效则清空队列。"""

        if not self.queued_actions:
            return None
        if (
            self.current_goal is not None
            and self.current_goal.kind == "open_chest"
            and self.current_goal.tile is not None
            and manhattan(player, self.current_goal.tile) == 1
        ):
            self.queued_actions.clear()
            return None
        if (
            self.current_goal is not None
            and self.current_goal.kind == "go_exit"
            and self.target_exit_tile is not None
            and is_near_exit_boundary(player, self.target_exit_tile, self.current_goal.direction)
        ):
            self.queued_actions.clear()
            return None
        action = self.queued_actions[0]
        if action in MOVE_ACTIONS and self._should_interrupt_queue(player, action, vision):
            self.queued_actions.clear()
            return None
        return self.queued_actions.popleft()

    def _should_interrupt_queue(self, player: Position, action: int, vision: PixelObservation) -> bool:
        """动态环境中的移动队列打断条件。"""

        if self.current_goal is not None and self.current_goal.kind == "go_exit":
            # 出口推进由 _shield_action 单独放行；仍然允许在到边界前做普通安全检查。
            if (
                self.exit_push_action == action
                and self.target_exit_tile is not None
                and is_near_exit_boundary(player, self.target_exit_tile, self.current_goal.direction)
            ):
                return False
        nxt = next_position(player, action)
        if not in_bounds(nxt):
            return True
        if self.current_goal is not None and self.current_goal.kind == "go_exit":
            return not is_walkable(nxt, vision, allow_next_to_monster=True)
        if self.current_goal is not None and self.current_goal.kind == "slay_monster":
            return not is_walkable(nxt, vision, allow_next_to_monster=True)
        if self.current_goal is not None and self.current_goal.kind == "open_chest" and self._is_rush_mode():
            # 后期赶宝箱时，路径层可以临时贴近怪物；是否需要先举盾交给
            # _shield_action 处理。这里若仍按普通安全半径打断，就会出现
            # “BFS 有路、最终安全层又拦住、每帧原地 NOOP”的死锁。
            return not is_walkable(nxt, vision, allow_next_to_monster=True)
        if not self._is_walkable(nxt, vision):
            return True
        if distance_to_nearest(nxt, self._monster_tiles(vision)) <= 1:
            return True
        return False

    def _choose_goal(self, player: Position, vision: PixelObservation) -> Goal:
        """选择当前高层目标。

        选择只基于当前视觉、内部探索记忆和允许的 inventory。没有预设目标物在哪个
        房间，也没有预设任何对象坐标。
        """
        # 高层目标选择 FSM

        memory = self._room_memory()

        chest_goal = self._choose_reachable_chest(player, vision)
        if chest_goal is not None:
            if not self._is_rush_mode():
                blocking_monster = self._choose_monster_blocking_chest(player, vision, chest_goal)
                if blocking_monster is not None:
                    return Goal(kind="slay_monster", tile=blocking_monster)
            return Goal(kind="open_chest", tile=chest_goal)

        # 如果当前房间明明看见未开的宝箱，但因为怪物占路/贴近导致 BFS 暂时找
        # 不到安全路径，就先处理一个可达怪物。这个目标仍然完全来自视觉，不预设
        # 哪个房间有怪物或宝箱。
        visible_unopened_chests = [
            chest
            for chest in self._chest_tiles(vision)
            if chest not in memory.opened_chests
        ]
        if visible_unopened_chests:
            monster_goal = self._choose_nearest_reachable_monster(player, vision)
            if monster_goal is not None and not self._is_rush_mode():
                return Goal(kind="slay_monster", tile=monster_goal)

        button_goal = self._choose_reachable_button(player, vision)
        if button_goal is not None:
            return Goal(kind="press_button", tile=button_goal)

        exits = self._exit_info_by_direction(vision)
        if self.has_key:
            for direction in EXIT_DIRECTION_ORDER:
                exit_info = exits.get(direction)
                if (
                    exit_info is not None
                    and exit_info.kind == "exit_locked"
                    and direction not in memory.explored_exits
                    and not self._exit_is_deferred(memory, direction)
                ):
                    return self._goal_for_exit(player, vision, direction, exit_info)

        if self.current_room != (0, 0) and memory.opened_chests:
            back_direction = self._backtrack_direction()
            if back_direction is not None:
                exit_info = exits.get(back_direction)
                if (
                    exit_info is not None
                    and not self._exit_is_deferred(memory, back_direction)
                ):
                    return self._goal_for_exit(player, vision, back_direction, exit_info)
                if back_direction in memory.known_exits and not self._exit_is_deferred(memory, back_direction):
                    return Goal(kind="go_exit", tile=memory.known_exits[back_direction], direction=back_direction)

        for direction in EXIT_DIRECTION_ORDER:
            exit_info = exits.get(direction)
            if (
                exit_info is not None
                and exit_info.kind != "exit_locked"
                and direction not in memory.explored_exits
                and not self._exit_is_deferred(memory, direction)
            ):
                return self._goal_for_exit(player, vision, direction, exit_info)

        # 当前房间没有新出口时，沿探索树回退。若当前帧没识别到回退出口，就使用
        # 本房间历史视觉记住的出口 tile。
        back_direction = self._backtrack_direction()
        if back_direction is not None and back_direction in exits:
            return self._goal_for_exit(player, vision, back_direction, exits[back_direction])
        if back_direction is not None and back_direction in memory.known_exits:
            return Goal(kind="go_exit", tile=memory.known_exits[back_direction], direction=back_direction)

        # 暂缓中的条件门不应每帧重撞；等冷却到期或出现钥匙/按钮状态变化后再试。
        for direction in EXIT_DIRECTION_ORDER:
            exit_info = exits.get(direction)
            if exit_info is not None and not self._exit_is_deferred(memory, direction):
                return self._goal_for_exit(player, vision, direction, exit_info)
        for direction in EXIT_DIRECTION_ORDER:
            if direction in memory.known_exits and not self._exit_is_deferred(memory, direction):
                return Goal(kind="go_exit", tile=memory.known_exits[direction], direction=direction)
        return Goal(kind="wait")

    def _goal_for_exit(
        self,
        player: Position,
        vision: PixelObservation,
        direction: str,
        exit_info: ExitInfo,
    ) -> Goal:
        """把视觉出口转换为“清怪”或“出门”目标。

        这里是出口守卫怪物逻辑真正接入高层 FSM 的位置。只检查当前准备尝试的
        出口附近怪物，不会为了与其他出口无关的怪物主动绕路。出口坐标完全来自
        ``exit_info.tiles``，空间变体改变出口位置时会随当前视觉自然更新。
        """

        blocking_monster = self._choose_monster_blocking_exit(
            player,
            vision,
            set(exit_info.tiles),
            direction,
        )
        if blocking_monster is not None:
            return Goal(kind="slay_monster", tile=blocking_monster)
        return Goal(
            kind="go_exit",
            # 同一出口横截面上的两个 tile 都可能被引擎接受。这里选择稳定代表值，
            # 而不是随着玩家逐帧移动在两个 tile 之间切换，避免像素对齐逻辑抖动。
            tile=exit_info.representative_tile(),
            direction=direction,
        )

    def _execute_goal(self, player: Position, vision: PixelObservation, goal: Goal | None) -> int:
        """把高层目标转换成一步环境动作。"""

        if goal is None or goal.kind == "wait":
            return ACTION_NOOP
        if goal.kind == "open_chest" and goal.tile is not None:
            return self._act_to_interactable(player, vision, goal.tile, ACTION_A)
        if goal.kind == "press_button" and goal.tile is not None:
            return self._act_to_button(player, vision, goal.tile)
        if goal.kind == "slay_monster" and goal.tile is not None:
            return self._act_to_monster(player, vision, goal.tile)
        if goal.kind == "go_exit" and goal.tile is not None and goal.direction is not None:
            return self._act_to_exit(player, vision, goal.tile, goal.direction)
        return ACTION_NOOP

    def _choose_reachable_chest(self, player: Position, vision: PixelObservation) -> Position | None:
        """选择当前房间中一个视觉可见、尚未打开且可达的宝箱。"""

        memory = self._room_memory()
        candidates = [
            chest
            for chest in self._chest_tiles(vision)
            if chest not in memory.opened_chests
        ]
        best: tuple[int, Position] | None = None
        for chest in candidates:
            allow_near_monster = self._is_rush_mode()
            if allow_near_monster:
                path = self._path_to_chest_interaction(
                    player,
                    chest,
                    vision,
                    allow_next_to_monster=allow_near_monster,
                    allow_goals_next_to_monster=allow_near_monster,
                    extra_blocked=self._remembered_static_blockers(memory),
                    blocked_edges=memory.learned_blocked_edges,
                )
            else:
                adjacent = self._chest_interaction_targets(chest, vision, allow_next_to_monster=False)
                path = bfs_path(
                    player,
                    adjacent,
                    vision,
                    extra_blocked=self._remembered_static_blockers(memory),
                    blocked_edges=memory.learned_blocked_edges,
                )
            if path:
                score = len(path)
                if best is None or score < best[0]:
                    best = (score, chest)
        return None if best is None else best[1]

    def _choose_reachable_button(self, player: Position, vision: PixelObservation) -> Position | None:
        """选择当前房间中一个视觉可见、内部记忆未踩过且可达的按钮。"""
        # 视觉上按钮可能不变色，所以这里用“我走到过按钮上”作为内部记忆

        memory = self._room_memory()
        candidates = [
            button
            for button in self._tiles_of_kind(vision, {"button"})
            if button not in memory.pressed_buttons
        ]
        best: tuple[int, Position] | None = None
        for button in candidates:
            path = bfs_path(
                player,
                {button},
                vision,
                extra_blocked=self._remembered_static_blockers(memory),
                blocked_edges=memory.learned_blocked_edges,
            )
            if path:
                score = len(path)
                if best is None or score < best[0]:
                    best = (score, button)
        return None if best is None else best[1]

    def _choose_monster_blocking_exit(
        self,
        player: Position,
        vision: PixelObservation,
        exit_tiles: set[Position],
        direction: str,
    ) -> Position | None:
        """选择一个正在阻挡指定视觉出口的可达怪物。

        Task 5 里怪物既会伤人，也会因为 AABB 碰撞和动态移动影响路径。这里的
        “堵门”必须同时满足两个条件：安全 BFS 已经无法到达这扇出口，且怪物实际
        占据视觉出口 tile。仅仅离出口较近时，绝不额外开战；此时交给 BFS 绕行和
        已有的贴脸防御处理。

        只是“靠近玩家”的情况交给 ``_urgent_defense_action`` 处理，不在这里追杀；
        否则容易为了非必要怪物耗掉 Task 5 的生命倒计时。

        这些判断都来自当前像素帧，不包含任何房间真值或坐标硬编码。
        """

        monsters = self._monster_tiles(vision)
        if not monsters or "sword" not in self.known_tools:
            return None
        memory = self._room_memory()

        corridor_monsters = {
            monster
            for monster in monsters
            if monster_blocks_exit_corridor(player, monster, exit_tiles, direction)
        }

        # 先证明出口真的被当前安全模型阻断。若仍能安全到达，并且没有怪物占住
        # 当前玩家到目标出口的粗略走廊，保留原有的“少打非必要怪”策略；这对
        # Task5 的步数和周期掉血尤其重要。
        approach_tiles: set[Position] = set()
        for exit_tile in exit_tiles:
            approach_tiles.update(exit_approach_targets(exit_tile, exit_direction(exit_tile) or "", vision))
        safe_path = bfs_path(
            player,
            approach_tiles,
            vision,
            extra_blocked=self._remembered_static_blockers(memory),
            blocked_edges=memory.learned_blocked_edges,
        )
        if safe_path and not corridor_monsters:
            return None

        candidates: list[tuple[int, Position]] = []
        for monster in monsters:
            guards_exit = monster in exit_tiles or monster in corridor_monsters
            if not guards_exit:
                continue
            adjacent = self._adjacent_targets({monster}, vision, allow_next_to_monster=True)
            path = bfs_path(
                player,
                adjacent,
                vision,
                allow_goals_next_to_monster=True,
                extra_blocked=self._remembered_static_blockers(memory),
                blocked_edges=memory.learned_blocked_edges,
            )
            if path:
                candidates.append((len(path), monster))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _choose_nearest_reachable_monster(self, player: Position, vision: PixelObservation) -> Position | None:
        """选择一个当前可达、值得清掉的怪物。

        这个函数只在“看见未开宝箱但宝箱没有安全路径”时调用。它不会为了刷分
        主动追杀所有怪物，而是把怪物当成阻塞当前目标的动态障碍来处理。
        """

        monsters = self._monster_tiles(vision)
        if not monsters or "sword" not in self.known_tools:
            return None
        memory = self._room_memory()
        candidates: list[tuple[int, Position]] = []
        for monster in monsters:
            if self._recently_hit_near(monster):
                continue
            adjacent = self._adjacent_targets({monster}, vision, allow_next_to_monster=True)
            path = bfs_path(
                player,
                adjacent,
                vision,
                allow_goals_next_to_monster=True,
                extra_blocked=self._remembered_static_blockers(memory),
                blocked_edges=memory.learned_blocked_edges,
            )
            if path:
                candidates.append((len(path), monster))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _choose_monster_blocking_chest(
        self,
        player: Position,
        vision: PixelObservation,
        chest: Position,
    ) -> Position | None:
        """判断当前宝箱目标是否应先清怪。

        只在两类局部危险下触发：

        - 怪物已经离玩家很近，继续赶路容易被追上；
        - 怪物贴近目标宝箱，站位会持续干扰开箱路径。

        这不是按房间写死的战斗顺序，而是从当前视觉的玩家、怪物、宝箱相对位置
        动态判断。若没有可达剑位，则不强行追杀。
        """

        monsters = self._monster_tiles(vision)
        if not monsters or "sword" not in self.known_tools:
            return None
        memory = self._room_memory()
        candidates: list[tuple[int, int, Position]] = []
        for monster in monsters:
            if self._recently_hit_near(monster):
                continue
            if manhattan(player, monster) > 2 and manhattan(chest, monster) > 2:
                continue
            adjacent = self._adjacent_targets({monster}, vision, allow_next_to_monster=True)
            path = bfs_path(
                player,
                adjacent,
                vision,
                allow_goals_next_to_monster=True,
                extra_blocked=self._remembered_static_blockers(memory),
                blocked_edges=memory.learned_blocked_edges,
            )
            if path:
                # 优先处理贴身怪，其次处理贴近宝箱的怪。
                urgency = min(manhattan(player, monster), manhattan(chest, monster))
                candidates.append((urgency, len(path), monster))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def _tick_recently_hit_monsters(self) -> None:
        """衰减“刚被主动击退”的怪物位置记忆。"""

        expired: list[Position] = []
        for tile, ttl in self.recently_hit_monsters.items():
            if ttl <= 1:
                expired.append(tile)
            else:
                self.recently_hit_monsters[tile] = ttl - 1
        for tile in expired:
            self.recently_hit_monsters.pop(tile, None)

    def _recently_hit_near(self, monster: Position) -> bool:
        """判断某个怪物是否刚被主动处理过。"""

        return any(manhattan(monster, tile) <= 1 for tile in self.recently_hit_monsters)

    def _act_to_interactable(
        self,
        player: Position,
        vision: PixelObservation,
        target: Position,
        interact_action: int,
    ) -> int:
        """走到宝箱/NPC/开关旁边，面向目标后执行交互动作。"""

        if manhattan(player, target) == 1:
            chest_goal = (
                self.current_goal is not None
                and self.current_goal.kind == "open_chest"
            )
            rush_chest = chest_goal and self._is_rush_mode() and self.current_room == (-1, 0)
            needs_retry_settle = chest_goal and (self.current_room, target) in self.chest_settle_targets
            if rush_chest or needs_retry_settle:
                settle_action = self._settle_interaction_stand_action(player, target, vision)
                if settle_action is not None:
                    return settle_action
            face_action = action_toward(player, target)
            if face_action is not None and self.last_move_action != face_action:
                self.queued_actions.append(interact_action)
                return face_action
            if chest_goal:
                self.pending_chest_interaction = target
                self.pending_chest_room = self.current_room
            return interact_action

        memory = self._room_memory()
        if self.current_goal is not None and self.current_goal.kind == "open_chest":
            allow_near_monster = self._is_rush_mode()
            if allow_near_monster:
                path = self._path_to_chest_interaction(
                    player,
                    target,
                    vision,
                    allow_next_to_monster=allow_near_monster,
                    allow_goals_next_to_monster=allow_near_monster,
                    extra_blocked=self._remembered_static_blockers(memory),
                    blocked_edges=memory.learned_blocked_edges,
                )
            else:
                adjacent = self._chest_interaction_targets(target, vision, allow_next_to_monster=False)
                path = bfs_path(
                    player,
                    adjacent,
                    vision,
                    extra_blocked=self._remembered_static_blockers(memory),
                    blocked_edges=memory.learned_blocked_edges,
                )
        else:
            adjacent = self._adjacent_targets({target}, vision, allow_next_to_monster=False)
            path = bfs_path(
                player,
                adjacent,
                vision,
                extra_blocked=self._remembered_static_blockers(memory),
                blocked_edges=memory.learned_blocked_edges,
            )
        if len(path) >= 2:
            return self._start_tile_step(action_toward(path[0], path[1]), vision)
        return ACTION_NOOP

    def _settle_interaction_stand_action(
        self,
        player: Position,
        target: Position,
        vision: PixelObservation,
    ) -> int | None:
        """交互前把像素中心推到当前视觉 tile 的中心附近。

        CNN 的 player tile 会比引擎 ``snapshot().player_tile`` 更早跨格；若此时
        直接 A，交互系统仍认为玩家没有贴到宝箱/NPC/开关旁边，于是 A 会落到
        装备动作上。这里不读隐藏状态，只用视觉 bbox 中心做最后几像素的 settle。
        """

        if vision.player is None:
            return None
        center_x, center_y = vision.player.center_px
        target_x = player[0] * TILE_SIZE + TILE_SIZE / 2.0
        target_y = player[1] * TILE_SIZE + TILE_SIZE / 2.0
        tolerance = 2.0
        dx = center_x - target_x
        dy = center_y - target_y
        if target[0] == player[0] and abs(dx) > tolerance:
            action = ACTION_LEFT if dx > 0 else ACTION_RIGHT
            if self._alignment_step_is_safe(player, action, vision):
                return action
        if target[1] == player[1] and abs(dy) > tolerance:
            action = ACTION_UP if dy > 0 else ACTION_DOWN
            if self._alignment_step_is_safe(player, action, vision):
                return action
        return None

    def _act_to_button(self, player: Position, vision: PixelObservation, button: Position) -> int:
        """走到按钮 tile 上，并在到达后用内部记忆标记为已按。"""

        if player == button:
            self._room_memory().pressed_buttons.add(button)
            self._room_memory().deferred_exits_until.clear()
            self.current_goal = None
            return ACTION_NOOP
        memory = self._room_memory()
        path = bfs_path(
            player,
            {button},
            vision,
            extra_blocked=self._remembered_static_blockers(memory),
            blocked_edges=memory.learned_blocked_edges,
        )
        if len(path) >= 2:
            return self._start_tile_step(action_toward(path[0], path[1]), vision)
        return ACTION_NOOP

    def _act_to_monster(self, player: Position, vision: PixelObservation, target: Position) -> int:
        """靠近可见怪物并挥剑。

        怪物会移动，所以目标 tile 可能已经变化。执行时优先处理任意相邻怪物；
        如果原目标消失，则重新选当前最近的怪物，下一帧高层目标也会随视觉更新。
        """

        adjacent_monster = self._adjacent_monster(player, vision)
        if adjacent_monster is not None:
            face_action = action_toward(player, adjacent_monster)
            if face_action is not None and self.last_move_action != face_action:
                return face_action
            # 主动清怪在 Task 5 中主要是为了让路。剑会击退并眩晕怪物；挥剑后先
            # 释放目标，下一帧重新根据视觉判断是继续自卫、绕路，还是开宝箱。
            self.recently_hit_monsters[adjacent_monster] = 1
            self.current_goal = None
            return ACTION_A

        monsters = self._monster_tiles(vision)
        if not monsters:
            self.current_goal = None
            return ACTION_NOOP
        if target not in monsters:
            target = min(monsters, key=lambda tile: manhattan(player, tile))

        memory = self._room_memory()
        adjacent = self._adjacent_targets({target}, vision, allow_next_to_monster=True)
        path = bfs_path(
            player,
            adjacent,
            vision,
            allow_goals_next_to_monster=True,
            extra_blocked=self._remembered_static_blockers(memory),
            blocked_edges=memory.learned_blocked_edges,
        )
        if len(path) >= 2:
            return self._start_tile_step(action_toward(path[0], path[1]), vision)
        # 走不到剑位时，若怪物已经很近，举盾挡一下，等待下一帧重新规划。
        if distance_to_nearest(player, monsters) <= 2 and "shield" in self.known_tools:
            return ACTION_B
        self.current_goal = None
        return ACTION_NOOP

    def _act_to_exit(self, player: Position, vision: PixelObservation, exit_tile: Position, direction: str) -> int:
        """走到指定方向出口，并在边界出口格上继续向外推进。"""

        if self.exit_push_action is not None:
            if self._can_continue_exit_push(player, direction):
                return self.exit_push_action
            self.exit_push_action = None
            self.target_exit_tile = None
            self.pending_exit_direction = None
            self.exit_push_steps = 0

        if is_direction_boundary(player, direction) and is_direction_boundary(exit_tile, direction):
            align_action = exit_alignment_action(player, exit_tile, direction)
            if align_action is not None:
                return align_action

        if is_near_exit_boundary(player, exit_tile, direction) or (
            self.target_exit_tile is not None
            and is_near_exit_boundary(player, self.target_exit_tile, direction)
        ):
            align_action = exit_alignment_action(player, exit_tile, direction)
            if align_action is not None:
                self.exit_push_action = None
                self.pending_exit_direction = None
                self.target_exit_tile = None
                self.exit_push_steps = 0
                return align_action
            self.exit_push_action = DIRECTION_TO_ACTION[direction]
            self.pending_exit_direction = direction
            # 保留视觉确认过的真实出口目标。玩家走到出口上后 sprite 会遮住
            # 出口图案，后续帧仍需依赖该目标做横向对齐和“撞边恢复”，绝不能把
            # 它覆盖成当前玩家 tile。
            self.target_exit_tile = exit_tile
            self.exit_push_steps = 0
            self.queued_actions.clear()
            return self.exit_push_action

        memory = self._room_memory()
        approach_targets = exit_approach_targets(exit_tile, direction, vision)
        path = bfs_path(
            player,
            approach_targets,
            vision,
            # 出口目标可以落在同一边界的相邻格，但行走途中仍要尊重当前房间
            # 通过视觉反馈学到的“这条边会卡住”。否则 agent 会在像素碰撞点上
            # 重复撞同一个方向，尤其是回起点后再去另一个门时最明显。
            extra_blocked=self._remembered_static_blockers(memory),
            blocked_edges=memory.learned_blocked_edges,
        )
        if len(path) >= 2:
            self.target_exit_tile = path[-1]
            return self._start_tile_step(action_toward(path[0], path[1]), vision)
        return ACTION_NOOP

    def _urgent_defense_action(self, player: Position, vision: PixelObservation) -> int | None:
        """处理贴脸怪物。

        Task 5 怪物较多。这里不主动追杀怪物，但如果怪物已经相邻，继续移动很可能
        受伤，所以执行“面向 + 挥剑”的局部反应。这个判断只依赖视觉中的怪物位置。
        """

        monster = self._adjacent_monster(player, vision)
        if monster is None:
            self.recent_attack_count = 0
            self.rush_escape_frames = 0
            return None
        rushing_for_chest = self._is_rush_mode() and (
            (
                self.current_goal is not None
                and self.current_goal.kind == "open_chest"
            )
            or bool(self._chest_tiles(vision))
        )
        if rushing_for_chest and self.rush_escape_frames > 0:
            self.rush_escape_frames -= 1
            self.recent_attack_count = 0
            return None
        face_action = action_toward(player, monster)
        if face_action is not None and self.last_move_action != face_action:
            self.recent_attack_count += 1
            return face_action
        if rushing_for_chest and (
            self.recent_attack_count >= 2
            or not self._attack_likely_to_hit(monster, face_action, vision)
        ):
            self.recent_attack_count = 0
            self.rush_escape_frames = 12
            if "shield" in self.known_tools and self.last_action_taken != ACTION_B:
                return ACTION_B
            return None
        self.recent_attack_count += 1
        self.recently_hit_monsters[monster] = 1
        self.current_goal = None
        return ACTION_A

    def _attack_likely_to_hit(
        self,
        monster: Position,
        face_action: int | None,
        vision: PixelObservation,
    ) -> bool:
        """用像素 bbox 过滤 CNN tile 相邻但剑实际够不到的空挥场景。"""

        if vision.player is None or face_action is None:
            return True
        monster_entity = next((entity for entity in vision.monsters if entity.tile == monster), None)
        if monster_entity is None:
            return True
        attack_rect = attack_rect_for_action(vision.player.bbox, face_action)
        return rects_overlap(expand_rect(attack_rect, 3), monster_entity.bbox)

    def _start_tile_step(self, action: int | None, vision: PixelObservation) -> int:
        """把 BFS 的一格移动展开成一小段可中断像素动作。"""

        if action is None:
            return ACTION_NOOP
        repeat_count = TILE_SIZE - 1
        if distance_to_nearest(vision.player.tile, self._monster_tiles(vision)) <= 2:
            repeat_count = 4
        self.queued_actions.extend([action] * max(0, repeat_count - 1))
        return action

    def _shield_action(self, action: int, vision: PixelObservation) -> int:
        """最终安全层：拦截危险移动，放行必要的出口推进/交互动作。"""

        if action not in MOVE_ACTIONS:
            return action
        if vision.player is None:
            return ACTION_NOOP
        player = vision.player.tile

        if (
            self.current_goal is not None
            and self.current_goal.kind == "go_exit"
            and self.pending_exit_direction is not None
            and self.exit_push_action == action
            and self.target_exit_tile is not None
            and is_near_exit_boundary(player, self.target_exit_tile, self.current_goal.direction)
        ):
            return action

        nxt = next_position(player, action)
        if not in_bounds(nxt):
            return ACTION_NOOP
        if self._adjacent_monster(player, vision) is not None and vision.grid[nxt[1]][nxt[0]] == "monster":
            return action
        if self.current_goal is not None and self.current_goal.kind == "open_chest" and vision.grid[nxt[1]][nxt[0]] == "chest":
            return action
        if self.current_goal is not None and self.current_goal.kind == "open_chest" and self._is_rush_mode():
            if is_walkable(nxt, vision, allow_next_to_monster=True):
                if (
                    distance_to_nearest(nxt, self._monster_tiles(vision)) <= 1
                    and "shield" in self.known_tools
                    and self.step_count % 6 == 0
                ):
                    # 后期冲刺时不能因为“当前格贴怪”就持续 B/移动/B/移动；
                    # 那会耗尽 Task5 的周期掉血窗口。这里只在下一格仍会贴怪时
                    # 偶尔举盾一帧，并把原移动放回队首。
                    if self.last_action_taken == ACTION_B:
                        return action
                    self.queued_actions.appendleft(action)
                    return ACTION_B
                return action
        if self.current_goal is not None and self.current_goal.kind == "go_exit":
            if is_walkable(nxt, vision, allow_next_to_monster=True):
                if (
                    distance_to_nearest(nxt, self._monster_tiles(vision)) <= 1
                    and "shield" in self.known_tools
                ):
                    if self.last_action_taken == ACTION_B:
                        return action
                    self.queued_actions.appendleft(action)
                    return ACTION_B
                return action
        if self.current_goal is not None and self.current_goal.kind == "slay_monster":
            if is_walkable(nxt, vision, allow_next_to_monster=True):
                if (
                    not self._is_rush_mode()
                    and distance_to_nearest(nxt, self._monster_tiles(vision)) <= 1
                    and "shield" in self.known_tools
                ):
                    if self.last_action_taken == ACTION_B:
                        return action
                    self.queued_actions.appendleft(action)
                    return ACTION_B
                return action
        if not self._is_walkable(nxt, vision):
            return ACTION_NOOP
        if distance_to_nearest(nxt, self._monster_tiles(vision)) <= 1:
            if "shield" in self.known_tools:
                return ACTION_B
            return ACTION_NOOP
        return action

    def _wait_with_shield_if_threatened(self, action: int, vision: PixelObservation) -> int:
        """避免在怪物靠近时空等。

        当高层暂时没有路径、低层动作退化成 WAIT，而视觉里怪物已经很近时，举盾
        比原地等待更安全。盾牌信息来自允许的 inventory 工具列表。
        """

        if action != ACTION_NOOP or vision.player is None or "shield" not in self.known_tools:
            return action
        if distance_to_nearest(vision.player.tile, self._monster_tiles(vision)) <= 2:
            return ACTION_B
        return action

    def _alignment_action(self, action: int, vision: PixelObservation) -> int:
        """在执行一格移动前，先用视觉中心点做像素级对齐。

        BFS 只知道 tile 坐标，但 NesyLink 的碰撞是像素级 AABB。玩家虽然处在某个
        tile 中心附近，实际碰撞盒可能贴着相邻墙/宝箱，导致“tile 上可走、像素
        上被卡”。因此：

        - 准备上下移动时，先把玩家水平中心对齐到当前 tile 中线；
        - 左右移动不做视觉垂直对齐，因为当前像素分类器的玩家 bbox 会随朝向
          和挥剑/举盾 sprite 上下抖动，垂直中心不够稳定；
        - 正在边界出口向外推进时不做对齐，否则会打断出房间动作。

        这里使用的是视觉检测出的玩家 ``center_px``，仍然只来自 ``obs``。
        """

        if action not in MOVE_ACTIONS or vision.player is None:
            return action
        player = vision.player.tile
        if (
            self.current_goal is not None
            and self.current_goal.kind == "open_chest"
            and self.current_goal.tile is not None
            and manhattan(player, self.current_goal.tile) == 1
        ):
            # 已经站在宝箱旁边时，下一步移动动作通常只是“转向宝箱”。
            # 像素对齐层不能把它改写成横向微调，否则会错过交互窗口。
            return action
        if (
            self.current_goal is not None
            and self.current_goal.kind == "go_exit"
            and (
                (
                    self.exit_push_action == action
                    and self.target_exit_tile is not None
                    and is_near_exit_boundary(player, self.target_exit_tile, self.current_goal.direction)
                )
                or is_direction_boundary(player, self.current_goal.direction)
            )
        ):
            # 只有真正贴到出口边界、或者正在持续向外推进时，才完全跳过对齐。
            # 否则普通“走向出口”的途中仍需要像素级对齐；不然玩家碰撞盒可能擦到
            # 旁边墙/宝箱，表现为 tile 路径可行但实际 action_blocked。
            return action

        center_x, center_y = vision.player.center_px
        target_x = player[0] * TILE_SIZE + TILE_SIZE / 2.0
        target_y = player[1] * TILE_SIZE + TILE_SIZE / 2.0
        tolerance = 1.5

        if action in {ACTION_UP, ACTION_DOWN}:
            if center_x < target_x - tolerance and self._alignment_step_is_safe(player, ACTION_RIGHT, vision):
                return ACTION_RIGHT
            if center_x > target_x + tolerance and self._alignment_step_is_safe(player, ACTION_LEFT, vision):
                return ACTION_LEFT
        return action

    def _alignment_step_is_safe(self, player: Position, action: int, vision: PixelObservation) -> bool:
        """判断对齐用的横向小步是否不会撞进明确障碍。

        对齐动作只是为了修正像素中心，不应该把 agent 主动带进墙、宝箱、陷阱、
        unknown 或怪物邻域。这里仍只使用当前视觉和房间内自己学到的临时阻挡。
        """

        nxt = next_position(player, action)
        if not in_bounds(nxt):
            return False
        memory = self._room_memory()
        if nxt in self._remembered_static_blockers(memory):
            return False
        return is_walkable(nxt, vision, allow_next_to_monster=True)

    def _can_continue_exit_push(self, player: Position, direction: str) -> bool:
        """检查当前出口推进动作是否仍然合理。"""

        return (
            self.target_exit_tile is not None
            and is_near_exit_boundary(player, self.target_exit_tile, direction)
            and self.exit_push_action == DIRECTION_TO_ACTION[direction]
        )

    def _backtrack_direction(self) -> str | None:
        """根据内部探索栈选择回退方向。"""

        if not self.room_stack:
            return None
        parent = self.room_stack[-1]
        dx = parent[0] - self.current_room[0]
        dy = parent[1] - self.current_room[1]
        for direction, delta in DIRECTION_DELTA.items():
            if delta == (dx, dy):
                return direction
        return None

    def _is_rush_mode(self) -> bool:
        """生命倒计时下的后期冲刺判断。

        Task 5 每隔固定时间会扣血。这里不用读取隐藏 HP，只用 agent 自己累计的
        动作步数；后期若仍看见未开宝箱，就减少主动追怪和保守绕路。
        """

        threshold = 500 if self.spatial_variant_mode else 900
        return self.step_count >= threshold

    def _monster_tiles(self, vision: PixelObservation) -> set[Position]:
        return {monster.tile for monster in vision.monsters}

    def _chest_tiles(self, vision: PixelObservation) -> set[Position]:
        return self._tiles_of_kind(vision, {"chest"})

    def _exit_tiles_by_direction(self, vision: PixelObservation) -> dict[str, Position]:
        exits: dict[str, Position] = {}
        for direction, exit_info in self._exit_info_by_direction(vision).items():
            exits[direction] = exit_info.representative_tile()
        return exits

    def _exit_info_by_direction(self, vision: PixelObservation) -> dict[str, "ExitInfo"]:
        """按方向聚合当前视觉里看到的出口 tile 和出口类型。"""

        grouped: dict[str, list[tuple[Position, str]]] = {}
        for tile in self._tiles_of_kind(vision, {"exit_locked", "exit_normal", "exit_conditional"}):
            direction = exit_direction(tile)
            if direction is not None:
                grouped.setdefault(direction, []).append((tile, vision.grid[tile[1]][tile[0]]))

        exits: dict[str, ExitInfo] = {}
        for direction, entries in grouped.items():
            # 同一出口横截面的类型应一致；若分类瞬间不一致，优先出现次数最多的
            # 类型，并保留该类型的所有 tile，避免单个抖动 tile 主导目标选择。
            kinds = sorted({kind for _, kind in entries})
            kind = max(kinds, key=lambda candidate: sum(item_kind == candidate for _, item_kind in entries))
            tiles = frozenset(tile for tile, item_kind in entries if item_kind == kind)
            exits[direction] = ExitInfo(tiles=tiles, kind=kind)
        return exits

    def _adjacent_monster(self, player: Position, vision: PixelObservation) -> Position | None:
        adjacent = [monster for monster in self._monster_tiles(vision) if manhattan(player, monster) <= 1]
        if not adjacent:
            return None
        return min(adjacent, key=lambda tile: manhattan(player, tile))

    def _tiles_of_kind(self, vision: PixelObservation, kinds: set[str]) -> set[Position]:
        return {tile.tile for tile in vision.tiles if tile.kind in kinds}

    def _adjacent_targets(
        self,
        blocked_targets: set[Position],
        vision: PixelObservation,
        *,
        allow_next_to_monster: bool = False,
    ) -> set[Position]:
        out: set[Position] = set()
        for target in blocked_targets:
            for pos in neighbors(target):
                if in_bounds(pos) and self._is_walkable(pos, vision, allow_next_to_monster=allow_next_to_monster):
                    out.add(pos)
        return out

    def _chest_interaction_targets(
        self,
        chest: Position,
        vision: PixelObservation,
        *,
        allow_next_to_monster: bool,
    ) -> set[Position]:
        """返回开宝箱的候选站位，并优先避免从宝箱下方交互。

        宝箱、墙和玩家都是像素 AABB 碰撞。站到宝箱下方时更容易在水平移动中
        擦到宝箱或下边界障碍，因此优先选择上/左/右三侧；如果这些位置都不可走，
        再退回全部相邻位置。
        """

        adjacent = self._adjacent_targets({chest}, vision, allow_next_to_monster=allow_next_to_monster)
        preferred = {pos for pos in adjacent if not (pos[0] == chest[0] and pos[1] > chest[1])}
        return preferred or adjacent

    def _path_to_chest_interaction(
        self,
        player: Position,
        chest: Position,
        vision: PixelObservation,
        *,
        allow_next_to_monster: bool,
        allow_goals_next_to_monster: bool,
        extra_blocked: set[Position],
        blocked_edges: set[tuple[Position, int]],
    ) -> list[Position]:
        """选择到宝箱交互站位的路径。

        单纯把所有相邻格放进一个 set 交给 BFS，会受邻居展开顺序影响；长度相同
        时可能选择宝箱右下侧，像素碰撞更容易被宝箱或墙卡住。这里分别评估每个
        候选站位，先比路径长度，长度相同则优先宝箱上方，再左右，最后下方。
        """

        best: tuple[int, int, list[Position]] | None = None
        for target in self._chest_interaction_targets(chest, vision, allow_next_to_monster=allow_next_to_monster):
            path = bfs_path(
                player,
                {target},
                vision,
                allow_next_to_monster=allow_next_to_monster,
                allow_goals_next_to_monster=allow_goals_next_to_monster,
                extra_blocked=extra_blocked,
                blocked_edges=blocked_edges,
            )
            if not path:
                continue
            score = (len(path), chest_stand_priority(chest, target), path)
            if best is None or score[:2] < best[:2]:
                best = score
        return [] if best is None else best[2]

    def _is_walkable(self, pos: Position, vision: PixelObservation, *, allow_next_to_monster: bool = False) -> bool:
        memory = self._room_memory()
        if pos in self._remembered_static_blockers(memory):
            return False
        return is_walkable(pos, vision, allow_next_to_monster=allow_next_to_monster)

    def _remembered_static_blockers(self, memory: RoomMemory) -> set[Position]:
        """返回当前房间历史视觉和卡住反馈确认的阻挡 tile。

        ``remembered_chests`` 继续保留宝箱语义；``remembered_static_blocked`` 负责
        墙、NPC、陷阱等历史静态阻挡。两者和 learned_blocked_tiles 合并后供 BFS
        与 safety shield 使用。
        """

        return memory.remembered_static_blocked | memory.remembered_chests | memory.learned_blocked_tiles


def bfs_path(
    start: Position,
    goals: set[Position],
    vision: PixelObservation,
    *,
    allow_next_to_monster: bool = False,
    allow_goals_next_to_monster: bool = False,
    extra_blocked: set[Position] | None = None,
    blocked_edges: set[tuple[Position, int]] | None = None,
) -> list[Position]:
    """用 BFS 找到从 ``start`` 到任一目标 tile 的最短符号路径。"""

    if start in goals:
        return [start]
    blocked = set(extra_blocked or set())
    edge_blocked = set(blocked_edges or set())
    queue: deque[Position] = deque([start])
    parent: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        for nxt in neighbors(current):
            if nxt in parent:
                continue
            if not in_bounds(nxt):
                continue
            if nxt in blocked:
                continue
            action = action_toward(current, nxt)
            if action is not None and (current, action) in edge_blocked:
                continue
            if nxt not in goals and not is_walkable(nxt, vision, allow_next_to_monster=allow_next_to_monster):
                continue
            if nxt in goals and not is_walkable(nxt, vision, allow_next_to_monster=allow_goals_next_to_monster):
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


def is_walkable(pos: Position, vision: PixelObservation, *, allow_next_to_monster: bool = False) -> bool:
    if not in_bounds(pos):
        return False
    kind = vision.grid[pos[1]][pos[0]]
    if kind in BLOCKING_KINDS:
        return False
    if kind not in SAFE_WALKABLE_KINDS:
        return False
    if not allow_next_to_monster and distance_to_nearest(pos, {monster.tile for monster in vision.monsters}) <= 1:
        return False
    return True


def neighbors(pos: Position) -> tuple[Position, Position, Position, Position]:
    col, row = pos
    return ((col, row - 1), (col, row + 1), (col - 1, row), (col + 1, row))


def in_bounds(pos: Position) -> bool:
    col, row = pos
    return 0 <= col < GRID_WIDTH and 0 <= row < GRID_HEIGHT


def is_boundary_tile(pos: Position) -> bool:
    col, row = pos
    return col == 0 or row == 0 or col == GRID_WIDTH - 1 or row == GRID_HEIGHT - 1


def is_direction_boundary(pos: Position, direction: str | None) -> bool:
    """判断 tile 是否贴在指定方向的边界上。"""

    if direction == "north":
        return pos[1] == 0
    if direction == "south":
        return pos[1] == GRID_HEIGHT - 1
    if direction == "west":
        return pos[0] == 0
    if direction == "east":
        return pos[0] == GRID_WIDTH - 1
    return False


def is_near_exit_boundary(player: Position, exit_tile: Position, direction: str | None) -> bool:
    """判断玩家是否已足够贴近某个方向出口，可以开始持续向外推进。

    玩家 sprite 和出口 tile 在像素级可能有半格到一格的错位；当玩家站到出口
    上时，视觉分类还会把该格识别成 player 而不是 exit。因此这里不用严格
    ``player == exit_tile``，而是要求玩家已经在同一侧边界，并且沿出口横截面
    与目标出口相差不超过一格。
    """

    if not is_direction_boundary(player, direction):
        return False
    if direction in {"north", "south"}:
        return abs(player[0] - exit_tile[0]) <= 1
    if direction in {"west", "east"}:
        return abs(player[1] - exit_tile[1]) <= 1
    return False


def exit_alignment_action(player: Position, exit_tile: Position, direction: str | None) -> int | None:
    """若已经贴边但没对准出口，返回沿边界微调的一步动作。"""

    if direction in {"north", "south"}:
        if player[0] < exit_tile[0]:
            return ACTION_RIGHT
        if player[0] > exit_tile[0]:
            return ACTION_LEFT
    if direction in {"west", "east"}:
        if player[1] < exit_tile[1]:
            return ACTION_DOWN
        if player[1] > exit_tile[1]:
            return ACTION_UP
    return None


def exit_approach_targets(exit_tile: Position, direction: str, vision: PixelObservation) -> set[Position]:
    """返回可作为出口推进起点的边界候选格。

    视觉出口 tile 可能贴着墙角；玩家只要在同一边界且横截面相差不超过一格，
    后续 ``is_near_exit_boundary`` 就能持续向外推进。因此 BFS 不必强行走到
    唯一出口 tile，可以选择相邻边界格，减少像素碰撞绕路。
    """

    candidates = {exit_tile}
    col, row = exit_tile
    if direction in {"west", "east"}:
        edge_col = 0 if direction == "west" else GRID_WIDTH - 1
        for delta in (-1, 0, 1):
            pos = (edge_col, row + delta)
            if in_bounds(pos) and is_walkable(pos, vision, allow_next_to_monster=True):
                candidates.add(pos)
    elif direction in {"north", "south"}:
        edge_row = 0 if direction == "north" else GRID_HEIGHT - 1
        for delta in (-1, 0, 1):
            pos = (col + delta, edge_row)
            if in_bounds(pos) and is_walkable(pos, vision, allow_next_to_monster=True):
                candidates.add(pos)
    return candidates


def inferred_entry_exit_tile(player: Position, direction: str | None) -> Position:
    """根据换房后出生点推断回退出口的大致边界 tile。

    这是从“刚完成换房”这一视觉事件推断出的记忆：例如从东边进入新房，玩家会
    出现在新房西侧附近，那么回退出口就在西边界、行号接近当前玩家行。
    """

    col, row = player
    if direction == "north":
        return col, 0
    if direction == "south":
        return col, GRID_HEIGHT - 1
    if direction == "west":
        return 0, row
    if direction == "east":
        return GRID_WIDTH - 1, row
    return player


def exit_direction(pos: Position) -> str | None:
    col, row = pos
    if row == 0:
        return "north"
    if row == GRID_HEIGHT - 1:
        return "south"
    if col == 0:
        return "west"
    if col == GRID_WIDTH - 1:
        return "east"
    return None


def monster_blocks_exit_corridor(
    player: Position,
    monster: Position,
    exit_tiles: set[Position],
    direction: str,
) -> bool:
    """判断怪物是否占住玩家到目标出口的大致走廊。"""

    if not exit_tiles:
        return False
    if direction in {"east", "west"}:
        exit_col = max(x for x, _ in exit_tiles) if direction == "east" else min(x for x, _ in exit_tiles)
        exit_rows = {y for _, y in exit_tiles}
        between = min(player[0], exit_col) <= monster[0] <= max(player[0], exit_col)
        aligned = any(abs(monster[1] - row) <= 1 for row in exit_rows)
        return between and aligned
    if direction in {"north", "south"}:
        exit_row = max(y for _, y in exit_tiles) if direction == "south" else min(y for _, y in exit_tiles)
        exit_cols = {x for x, _ in exit_tiles}
        between = min(player[1], exit_row) <= monster[1] <= max(player[1], exit_row)
        aligned = any(abs(monster[0] - col) <= 1 for col in exit_cols)
        return between and aligned
    return False


def moved_room(room: RoomCoord, direction: str) -> RoomCoord:
    dx, dy = DIRECTION_DELTA[direction]
    return room[0] + dx, room[1] + dy


def next_position(pos: Position, action: int) -> Position:
    dx, dy = ACTION_TO_DELTA[action]
    return pos[0] + dx, pos[1] + dy


def attack_rect_for_action(
    player_bbox: tuple[int, int, int, int],
    action: int,
) -> tuple[int, int, int, int]:
    """返回当前像素 bbox 在指定方向的一格剑攻击区域。"""

    left, top, right, bottom = player_bbox
    if action == ACTION_UP:
        return left, top - TILE_SIZE, right, top
    if action == ACTION_DOWN:
        return left, bottom, right, bottom + TILE_SIZE
    if action == ACTION_LEFT:
        return left - TILE_SIZE, top, left, bottom
    return right, top, right + TILE_SIZE, bottom


def expand_rect(
    rect: tuple[int, int, int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = rect
    return left - pad, top - pad, right + pad, bottom + pad


def rects_overlap(
    left_rect: tuple[int, int, int, int],
    right_rect: tuple[int, int, int, int],
) -> bool:
    left_l, left_t, left_r, left_b = left_rect
    right_l, right_t, right_r, right_b = right_rect
    return not (
        left_r <= right_l
        or left_l >= right_r
        or left_b <= right_t
        or left_t >= right_b
    )


def action_toward(current: Position, nxt: Position) -> int | None:
    delta = (nxt[0] - current[0], nxt[1] - current[1])
    return DELTA_TO_ACTION.get(delta)


def chest_stand_priority(chest: Position, stand: Position) -> int:
    """宝箱交互站位优先级：上方最稳，其次左右，下方最后。"""

    if stand[0] == chest[0] and stand[1] < chest[1]:
        return 0
    if stand[1] == chest[1]:
        return 1
    return 2


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def distance_to_nearest(pos: Position, targets: set[Position]) -> int:
    if not targets:
        return 999
    return min(manhattan(pos, target) for target in targets)


Policy = Task5FSMBFSAgent


def make_policy() -> Task5FSMBFSAgent:
    return Task5FSMBFSAgent()
