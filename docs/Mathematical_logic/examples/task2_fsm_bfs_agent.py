from __future__ import annotations

"""任务 2 的像素识别 + FSM + BFS + 可中断移动队列策略。

整体思路：

1. 策略推理阶段只使用原始 RGB 图像 ``obs`` 和课程允许的显式物品栏信息。
   代码不会读取地图真值、对象真实坐标、房间编号、debug 状态、entities 等
   隐藏环境信息。当前评测接口会把允许使用的物品栏放在 ``info["inventory"]``
   中，因此本策略只从里面读取 ``inventory["keys"]``。
2. 先用 ``classify_frame`` 把像素图转成符号 tile 网格，识别玩家、怪物、宝箱、
   陷阱、出口等类别。之后所有规划都基于这个视觉抽取得到的符号网格。
3. 使用有限状态机描述 Task 2 的子目标顺序：
      ``to_monster`` -> 先靠近并击败可见怪物；
      ``to_chest``   -> 怪物消失后，去宝箱旁边交互拿钥匙；
      ``to_exit``    -> 确认钥匙后，走到西侧条件出口并推出房间。
4. tile 级路径规划使用 BFS。BFS 只在当前识别出的可走符号格上搜索，不依赖
   task2 的固定坐标，也不使用助教 reference 中的地图真值。
5. 与 task1 不同，task2 有 chaser 怪物。为了避免“连续 16 帧盲走一格”导致
   怪物靠近时来不及反应，本策略把队列动作改成可中断：
      - 每一帧都会重新做视觉识别；
      - 执行队列动作前会检查是否需要打断；
      - 队列动作也必须经过 safety shield；
      - 怪物相邻、目标消失、下一格变危险时会清空队列并重新规划。
6. 攻击逻辑保持简单可解释：如果玩家与怪物相邻，则先尝试面向怪物，再使用
   ``ACTION_A`` 挥剑。这个“面向 + 攻击”的动作也通过短队列表达，方便评测器
   按普通 policy 接口逐步调用。

这个文件是为了后续扩展到 Task 5 做准备：FSM、BFS、可中断队列、safety shield
都尽量写成通用的小函数。将来接入 CNN 头时，只需要让 ``classify_frame`` 的输出
被替换为同样结构的 ``PixelObservation`` 即可。
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
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
    GRID_HEIGHT,
    GRID_WIDTH,
    TILE_SIZE,
)
from nesylink.vision import PixelObservation, classify_frame


Position = tuple[int, int]

MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_TO_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
DELTA_TO_ACTION = {delta: action for action, delta in ACTION_TO_DELTA.items()}

# 普通移动不能主动进入这些 tile。monster 是动态危险源；unknown 在最终测评中
# 也按危险处理，因为视觉不确定时宁可停下来重新规划。
BLOCKING_KINDS = {
    "wall",
    "chest",
    "trap",
    "abyss",
    "gap",
    "monster",
    "unknown",
}

# 这些是“看见过一次就应当保守记住”的静态阻挡/危险 tile。它们只来自历史
# 像素观测，不来自地图真值。monster/player 不在这里，因为它们会移动。
STATIC_BLOCKING_KINDS = {
    "wall",
    "chest",
    "trap",
    "abyss",
    "gap",
    "npc",
}

# 这些 tile 可以作为 BFS 的普通通行区域。exit_conditional 在拿到钥匙且击杀
# 怪物后才真正能完成任务，但从视觉/路径角度它仍是边界出口 tile。
SAFE_WALKABLE_KINDS = {
    "floor",
    "player",
    "bridge",
    "button",
    "switch",
    "exit_normal",
    "exit_locked",
    "exit_conditional",
}


@dataclass
class Task2FSMBFSAgent:
    """Task 2 的合规策略：视觉抽象、FSM、BFS、可中断队列和安全层。

    评测器为了兼容会传入 ``info``。本策略只读取其中课程允许的显式物品栏视图，
    不读取对象坐标、地图真值、debug 字段、entities 字段或 room id。
    """

    phase: str = "to_monster"
    queued_actions: deque[int] = field(default_factory=deque)
    last_key_count: int = 0
    key_confirmed: bool = False
    monster_missing_frames: int = 0
    monster_cleared: bool = False
    remembered_chest_tiles: set[Position] = field(default_factory=set)
    remembered_static_blocked_tiles: set[Position] = field(default_factory=set)
    target_exit_tile: Position | None = None
    exit_push_action: int | None = None
    last_move_action: int | None = None

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        """在新 episode 开始前清空策略内部记忆。"""

        del seed, task_id
        self.phase = "to_monster"
        self.queued_actions.clear()
        self.last_key_count = 0
        self.key_confirmed = False
        self.monster_missing_frames = 0
        self.monster_cleared = False
        self.remembered_chest_tiles.clear()
        self.remembered_static_blocked_tiles.clear()
        self.target_exit_tile = None
        self.exit_push_action = None
        self.last_move_action = None

    def act(self, obs, info=None) -> int:
        """根据像素观测和允许的物品栏信息输出一个环境动作。"""

        self._update_inventory_progress(info)
        vision = classify_frame(obs)
        player = None if vision.player is None else vision.player.tile
        if player is None:
            self.queued_actions.clear()
            return ACTION_NOOP

        self._remember_visible_objects(vision)
        self._update_monster_progress(vision)
        self._update_phase(vision)

        # 与 task1 不同：队列动作不能直接弹出执行。每一帧都要重新确认当前视觉
        # 状态是否仍然支持继续执行之前的微动作计划。
        action: int | None = None
        if self.queued_actions:
            if self._should_interrupt_queue(player, vision):
                self.queued_actions.clear()
            else:
                action = self.queued_actions.popleft()

        if action is None:
            if self.phase == "to_monster":
                action = self._act_to_monster(player, vision)
            elif self.phase == "to_chest":
                action = self._act_to_chest(player, vision)
            elif self.phase == "to_exit":
                action = self._act_to_exit(player, vision)
            else:
                action = ACTION_NOOP

        safe_action = self._shield_action(action, vision)
        if safe_action == ACTION_NOOP and action in MOVE_ACTIONS:
            # 如果安全层拦截了移动，说明旧计划已经不可靠，下帧重新规划。
            self.queued_actions.clear()
        if safe_action in MOVE_ACTIONS:
            self.last_move_action = safe_action
        return safe_action

    def _update_inventory_progress(self, info) -> None:
        """从允许的物品栏视图中记录钥匙是否已经被确认获得。"""

        keys = inventory_key_count(info)
        if keys is None:
            return
        if keys > self.last_key_count or keys > 0:
            self.key_confirmed = True
        self.last_key_count = max(self.last_key_count, keys)

    def _update_monster_progress(self, vision: PixelObservation) -> None:
        """用连续视觉帧判断怪物是否已经消失。

        Task 2 的怪物会移动，也可能被玩家挥剑击退。单帧看不到怪物可能是视觉
        遮挡或分类误差，所以这里要求连续若干帧都看不到怪物，才确认进入开箱
        阶段。这样比“某一帧看不到就切阶段”更稳。
        """

        if self._monster_tiles(vision):
            self.monster_missing_frames = 0
            self.monster_cleared = False
            return
        self.monster_missing_frames += 1
        if self.monster_missing_frames >= 3:
            self.monster_cleared = True

    def _update_phase(self, vision: PixelObservation) -> None:
        """根据视觉和物品栏进展切换任务阶段。"""

        if self.phase == "to_monster" and self.monster_cleared and not self.queued_actions:
            self.phase = "to_chest"
        if self.phase == "to_chest" and (self.key_confirmed or not self._chest_tiles(vision)) and not self.queued_actions:
            self.phase = "to_exit"

    def _should_interrupt_queue(self, player: Position, vision: PixelObservation) -> bool:
        """判断当前排队的像素动作是否应该被打断。

        队列动作只是“短期意图”，不是不可打断的宏动作。只要发现动态危险、阶段
        目标变化或下一步不再安全，就清空队列重新规划。
        """

        if not self.queued_actions:
            return False
        next_action = self.queued_actions[0]
        if next_action not in MOVE_ACTIONS:
            return False

        if self.phase == "to_monster" and self._adjacent_monster(player, vision) is not None:
            return True
        if self.phase == "to_monster" and self.monster_cleared:
            return True
        if self.phase == "to_chest" and self.key_confirmed:
            return True

        nxt = next_position(player, next_action)
        if not in_bounds(nxt):
            return True
        if not self._is_walkable(nxt, vision):
            return True

        # 对动态怪物额外保守：如果下一格会进入怪物一格邻域，停下来重新评估。
        if self.phase != "to_monster" and distance_to_nearest(nxt, self._monster_tiles(vision)) <= 1:
            return True
        return False

    def _act_to_monster(self, player: Position, vision: PixelObservation) -> int:
        """靠近怪物，并在相邻时执行“面向 + 挥剑”。"""

        monster_tiles = self._monster_tiles(vision)
        if not monster_tiles:
            return ACTION_NOOP

        adjacent_monster = self._adjacent_monster(player, vision)
        if adjacent_monster is not None:
            face_action = action_toward(player, adjacent_monster)
            if face_action is not None and self.last_move_action != face_action:
                # 先用一个极短移动尝试修正朝向，下一帧再挥剑。由于怪物相邻时
                # 接触风险存在，这里只排一个 ACTION_A，不继续排移动。
                self.queued_actions.append(ACTION_A)
                return face_action
            return ACTION_A

        # 目标不是怪物所在格，而是怪物旁边的安全站位。
        target_tiles = self._adjacent_targets(monster_tiles, vision, allow_next_to_monster=True)
        path = bfs_path(
            player,
            target_tiles,
            vision,
            allow_goals_next_to_monster=True,
            extra_blocked=self._remembered_static_blockers(),
        )
        if len(path) >= 2:
            return self._start_tile_step(action_toward(path[0], path[1]), vision)
        return ACTION_NOOP

    def _act_to_chest(self, player: Position, vision: PixelObservation) -> int:
        """怪物消失后，规划到可见宝箱旁边，并在相邻时交互拿钥匙。"""

        chest_tiles = self._chest_tiles(vision)
        if not chest_tiles:
            self.phase = "to_exit"
            return ACTION_NOOP

        adjacent = self._adjacent_targets(chest_tiles, vision)
        if player in adjacent:
            chest = min(chest_tiles, key=lambda tile: manhattan(player, tile))
            face_action = action_toward(player, chest)
            if face_action is not None:
                self.queued_actions.append(ACTION_A)
                return face_action
            return ACTION_A

        path = bfs_path(player, adjacent, vision, extra_blocked=self._remembered_static_blockers())
        if len(path) >= 2:
            return self._start_tile_step(action_toward(path[0], path[1]), vision)
        return ACTION_NOOP

    def _act_to_exit(self, player: Position, vision: PixelObservation) -> int:
        """拿到钥匙后，规划到条件出口，并在边界出口格上继续向外推进。"""

        exit_tiles = self._exit_tiles(vision)
        reachable_exit_tiles = {tile for tile in exit_tiles if self._is_walkable(tile, vision)}

        if self.exit_push_action is not None:
            if self._can_continue_exit_push(player, reachable_exit_tiles):
                return self.exit_push_action
            self.exit_push_action = None
            self.target_exit_tile = None

        if not reachable_exit_tiles:
            return ACTION_NOOP

        if player in reachable_exit_tiles or (
            self.target_exit_tile is not None
            and player == self.target_exit_tile
            and is_boundary_tile(player)
        ):
            self.exit_push_action = boundary_exit_action(player)
            self.target_exit_tile = player
            return self.exit_push_action or ACTION_NOOP

        path = bfs_path(player, reachable_exit_tiles, vision, extra_blocked=self._remembered_static_blockers())
        if len(path) >= 2:
            self.target_exit_tile = path[-1]
            return self._start_tile_step(action_toward(path[0], path[1]), vision)
        return ACTION_NOOP

    def _start_tile_step(self, action: int | None, vision: PixelObservation) -> int:
        """把 BFS 的一格移动展开成可中断的像素动作。

        Task 2 仍然可以缓存接下来的一小段同向移动，但这些动作每一帧都会重新
        经过视觉检查和 safety shield，因此不会像 task1 那样盲走完整一格。
        """

        if action is None:
            return ACTION_NOOP
        repeat_count = TILE_SIZE - 1
        if self._monster_tiles(vision):
            # 有动态怪物时缩短缓存，让 agent 更频繁重新规划。
            repeat_count = min(repeat_count, 4)
        self.queued_actions.extend([action] * repeat_count)
        return action

    def _shield_action(self, action: int, vision: PixelObservation) -> int:
        """在动作交给环境前，拦截不安全的移动动作。"""

        if action not in MOVE_ACTIONS:
            return action
        if vision.player is None:
            return ACTION_NOOP
        player = vision.player.tile

        # 出口推进是唯一允许离开 10x8 tile 边界的情况。
        if (
            self.phase == "to_exit"
            and self.exit_push_action == action
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

        # 面向相邻怪物时，允许短促地朝怪物方向发出一次移动命令来修正朝向。
        # 这一步不是路径规划中的普通移动，下一帧会立即 ACTION_A。
        if self.phase == "to_monster" and vision.grid[nxt[1]][nxt[0]] == "monster":
            return action

        # 面向宝箱时同理：宝箱阻挡移动，但移动命令可以更新朝向，随后 ACTION_A。
        if self.phase == "to_chest" and vision.grid[nxt[1]][nxt[0]] == "chest":
            return action

        if not self._is_walkable(nxt, vision):
            return ACTION_NOOP

        # 非打怪阶段不主动走到怪物邻域，避免 chaser 靠近时被队列带进去。
        if self.phase != "to_monster" and distance_to_nearest(nxt, self._monster_tiles(vision)) <= 1:
            return ACTION_NOOP
        return action

    def _can_continue_exit_push(self, player: Position, reachable_exit_tiles: set[Position]) -> bool:
        """检查记住的出口推进动作当前是否仍然合理。"""

        return (
            self.target_exit_tile is not None
            and player == self.target_exit_tile
            and is_boundary_tile(player)
            and self.exit_push_action == boundary_exit_action(player)
            and (player in reachable_exit_tiles or player == self.target_exit_tile)
        )

    def _remember_visible_objects(self, vision: PixelObservation) -> None:
        """记住曾经从视觉中看到过的静态阻挡位置。

        Task 2 的宝箱打开后，视觉分类器可能仍把打开后的图案识别成 chest，
        也可能在未来的 CNN 版本中把它识别成 floor。环境里宝箱附近仍可能产生
        像素级碰撞/贴边堵塞；墙、陷阱、gap 等静态障碍也不应因为某一帧误判
        成 floor 就被 BFS 当成通路。因此这里把历史视觉中明确见过的静态阻挡
        记下来。这个记忆只来自历史 obs，不来自地图真值或隐藏 info。
        """

        self.remembered_chest_tiles.update(self._chest_tiles(vision))
        self.remembered_static_blocked_tiles.update(self._tiles_of_kind(vision, STATIC_BLOCKING_KINDS))

    def _is_walkable(self, pos: Position, vision: PixelObservation, *, allow_next_to_monster: bool = False) -> bool:
        """结合当前视觉和历史静态阻挡记忆判断 tile 是否可走。"""

        if pos in self._remembered_static_blockers():
            return False
        return is_walkable(pos, vision, allow_next_to_monster=allow_next_to_monster)

    def _remembered_static_blockers(self) -> set[Position]:
        """返回当前单房间任务中历史记住的所有静态阻挡。"""

        return self.remembered_static_blocked_tiles | self.remembered_chest_tiles

    def _monster_tiles(self, vision: PixelObservation) -> set[Position]:
        """返回当前视觉帧中所有怪物 tile。"""

        return {monster.tile for monster in vision.monsters}

    def _chest_tiles(self, vision: PixelObservation) -> set[Position]:
        """返回当前视觉帧中所有未开启宝箱 tile。"""

        return self._tiles_of_kind(vision, {"chest"})

    def _exit_tiles(self, vision: PixelObservation) -> set[Position]:
        """返回当前视觉帧中所有出口 tile。"""

        return self._tiles_of_kind(vision, {"exit_locked", "exit_normal", "exit_conditional"})

    def _adjacent_monster(self, player: Position, vision: PixelObservation) -> Position | None:
        """如果有怪物与玩家曼哈顿距离为 1，返回最近的那只怪物。"""

        adjacent = [monster for monster in self._monster_tiles(vision) if manhattan(player, monster) == 1]
        if not adjacent:
            return None
        return min(adjacent, key=lambda tile: manhattan(player, tile))

    def _tiles_of_kind(self, vision: PixelObservation, kinds: set[str]) -> set[Position]:
        """返回所有分类类别属于 ``kinds`` 的 tile 坐标。"""

        return {tile.tile for tile in vision.tiles if tile.kind in kinds}

    def _adjacent_targets(
        self,
        blocked_targets: set[Position],
        vision: PixelObservation,
        *,
        allow_next_to_monster: bool = False,
    ) -> set[Position]:
        """返回阻挡型交互目标旁边所有可站立的 tile。"""

        out: set[Position] = set()
        for target in blocked_targets:
            for pos in neighbors(target):
                if in_bounds(pos) and self._is_walkable(pos, vision, allow_next_to_monster=allow_next_to_monster):
                    out.add(pos)
        return out


def bfs_path(
    start: Position,
    goals: set[Position],
    vision: PixelObservation,
    *,
    allow_goals_next_to_monster: bool = False,
    extra_blocked: set[Position] | None = None,
) -> list[Position]:
    """用 BFS 找到从 ``start`` 到任一目标 tile 的最短符号路径。"""

    if start in goals:
        return [start]
    blocked = set(extra_blocked or set())
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
            if nxt not in goals and not is_walkable(nxt, vision):
                continue
            if nxt in goals and not is_walkable(nxt, vision, allow_next_to_monster=allow_goals_next_to_monster):
                continue
            parent[nxt] = current
            if nxt in goals:
                return reconstruct_path(parent, nxt)
            queue.append(nxt)
    return []


def reconstruct_path(parent: dict[Position, Position | None], goal: Position) -> list[Position]:
    """沿 BFS 的 parent 指针回溯出完整路径。"""

    path: list[Position] = []
    current: Position | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def is_walkable(pos: Position, vision: PixelObservation, *, allow_next_to_monster: bool = False) -> bool:
    """判断某个符号 tile 是否适合普通移动进入。"""

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
    """返回上下左右四个相邻 tile 坐标。"""

    col, row = pos
    return ((col, row - 1), (col, row + 1), (col - 1, row), (col + 1, row))


def in_bounds(pos: Position) -> bool:
    """检查 tile 坐标是否位于 10x8 地图范围内。"""

    col, row = pos
    return 0 <= col < GRID_WIDTH and 0 <= row < GRID_HEIGHT


def is_boundary_tile(pos: Position) -> bool:
    """检查 tile 是否位于房间外边界上。"""

    col, row = pos
    return col == 0 or row == 0 or col == GRID_WIDTH - 1 or row == GRID_HEIGHT - 1


def is_exit_tile(pos: Position, vision: PixelObservation) -> bool:
    """检查当前可见 tile 是否是边界出口。"""

    if not in_bounds(pos):
        return False
    return is_boundary_tile(pos) and vision.grid[pos[1]][pos[0]] in {
        "exit_locked",
        "exit_normal",
        "exit_conditional",
    }


def boundary_exit_action(pos: Position) -> int | None:
    """根据边界 tile 位置，返回离开当前房间所需的移动动作。"""

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
    """只从显式允许的物品栏视图中提取钥匙数量。"""

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
    """把一个移动动作应用到 tile 坐标上。"""

    dx, dy = ACTION_TO_DELTA[action]
    return (pos[0] + dx, pos[1] + dy)


def action_toward(current: Position, nxt: Position) -> int | None:
    """返回从当前 tile 走向相邻 tile 所需的方向动作。"""

    delta = (nxt[0] - current[0], nxt[1] - current[1])
    return DELTA_TO_ACTION.get(delta)


def manhattan(left: Position, right: Position) -> int:
    """计算两个 tile 坐标之间的曼哈顿距离。"""

    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def distance_to_nearest(pos: Position, targets: set[Position]) -> int:
    """返回 ``pos`` 到一组目标 tile 的最近曼哈顿距离；目标为空时返回大数。"""

    if not targets:
        return 999
    return min(manhattan(pos, target) for target in targets)


Policy = Task2FSMBFSAgent


def make_policy() -> Task2FSMBFSAgent:
    """评测脚本使用的策略工厂函数。"""

    return Task2FSMBFSAgent()
