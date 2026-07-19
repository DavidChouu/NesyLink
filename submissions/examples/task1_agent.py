from __future__ import annotations

"""任务 1 的像素识别 + FSM + BFS 策略。

整体思路：

1. 策略推理阶段只使用原始 RGB 图像 ``obs`` 和课程允许的显式物品栏信息。
   代码不会读取地图真值、对象真实坐标、房间编号、debug 状态、entities 等
   隐藏环境信息。当前评测接口把允许使用的物品栏放在 ``info["inventory"]``
   中，因此本策略只从里面取 ``inventory["keys"]``，其余 ``info`` 字段一律不用。
2. 先用 ``classify_frame_cnn`` 把像素图转成符号 tile 网格，识别玩家、宝箱、
   墙、出口、陷阱、地板等类别。本文件现在使用 CNN-only perception，不回退
   传统颜色分类器。
3. 使用一个很小的有限状态机：
      ``to_chest`` -> 先走到可见宝箱旁边并交互；
      ``to_exit``  -> 确认钥匙数量增加后，再走到可见边界出口并推出房间。
4. tile 级路径规划使用 BFS。BFS 只在当前识别出的可走符号格上搜索，不依赖
   task1 的固定坐标，也不使用固定动作脚本。
5. NesyLink 默认是像素级移动，所以 BFS 的“一格移动”会被展开成一小段重复的
   像素移动动作。
6. 新规划出的移动动作会经过 safety shield。它会阻止主动走进墙、未打开宝箱、
   陷阱、gap、怪物或 unknown tile。唯一允许越出地图边界的情况，是已经确认
   玩家位于边界出口格，需要继续向外推门。
7. 策略会把图像中见过的墙和宝箱记为静态阻挡物。这样即使宝箱打开后外观变化、
   CNN 不再把它判成 ``chest``，BFS 也不会尝试从宝箱格穿过去。这个记忆仍然只
   来自图像识别结果，不来自环境地图真值。

针对前面检查出的几个问题，这里做了约束：

- 早期版本只要尝试开一次宝箱就进入 ``to_exit``，这不严谨，因为“尝试交互”
  不等于“钥匙已经拿到”。当前代码会等显式物品栏中的 ``keys`` 增加或大于 0，
  才确认可以进入出口阶段。
- 早期版本一旦设置了出口推进动作，就会一直重复该动作。当前代码会重新确认
  玩家仍在记住的边界出口格，并且推进方向仍与该边界一致，避免因为识别误差
  或位置偏移而盲目走下去。
- 当玩家站在出口格上时，玩家 sprite 可能遮住出口图案，所以代码会记住 BFS
  选中的目标出口格；只要玩家仍在这个记住的边界格上，就允许最后向外推进。
- 打开后的宝箱可能被 CNN 识别成 floor 或 exit，但环境中宝箱仍是阻挡物。当前
  代码会记住曾经看见或交互过的宝箱格，把它持续作为 BFS 阻挡。
- 这个策略仍然是面向 task1 的简单策略，默认环境是单房间、静态的“拿钥匙后
  出门”任务。如果用于有动态怪物或会变化陷阱的任务，需要更频繁地重检队列中
  的像素移动动作。
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
from nesylink.vision import PixelObservation, classify_frame_cnn


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
}


@dataclass
class FSMBFSAgent:
    """基于像素识别、FSM、BFS、action mask 和 safety shield 的策略。

    评测器为了兼容会传入 ``info``。本策略只读取其中课程允许的显式物品栏视图，
    不读取对象坐标、地图真值、debug 字段等隐藏环境状态。
    """

    phase: str = "to_chest"
    queued_actions: deque[int] = field(default_factory=deque)
    chest_interactions: int = 0
    last_key_count: int = 0
    key_confirmed: bool = False
    exit_push_action: int | None = None
    target_exit_tile: Position | None = None
    queued_target_tile: Position | None = None
    remembered_blockers: set[Position] = field(default_factory=set)

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        """在新 episode 开始前清空策略内部记忆。"""

        del seed, task_id
        self.phase = "to_chest"
        self.queued_actions.clear()
        self.chest_interactions = 0
        self.last_key_count = 0
        self.key_confirmed = False
        self.exit_push_action = None
        self.target_exit_tile = None
        self.queued_target_tile = None
        self.remembered_blockers.clear()

    def act(self, obs, info=None) -> int:
        """根据像素观测和允许的物品栏信息输出一个环境动作。"""

        self._update_inventory_progress(info)
        vision = classify_frame_cnn(obs, fallback=False)
        player = None if vision.player is None else vision.player.tile
        if player is None:
            return ACTION_NOOP

        self._remember_static_blockers(vision)
        self._update_phase(vision)

        if self.queued_actions:
            queued_action = self.queued_actions.popleft()
            shielded_action = self._shield_queued_action(queued_action, vision)
            if shielded_action != queued_action:
                self.queued_actions.clear()
                self.queued_target_tile = None
            elif not self.queued_actions:
                self.queued_target_tile = None
            return shielded_action

        if self.phase == "to_chest":
            action = self._act_to_chest(player, vision)
        elif self.phase == "to_exit":
            action = self._act_to_exit(player, vision)
        else:
            action = ACTION_NOOP

        return self._shield_action(action, vision)

    def _update_phase(self, vision: PixelObservation) -> None:
        """确认拿到钥匙后，从找宝箱阶段切换到找出口阶段。"""

        del vision
        if self.phase == "to_chest" and self.key_confirmed and not self.queued_actions:
            self.phase = "to_exit"

    def _update_inventory_progress(self, info) -> None:
        """从允许的物品栏视图中记录钥匙是否已经被确认获得。"""

        keys = inventory_key_count(info)
        if keys is None:
            return
        if keys > self.last_key_count or keys > 0:
            self.key_confirmed = True
        self.last_key_count = max(self.last_key_count, keys)

    def _act_to_chest(self, player: Position, vision: PixelObservation) -> int:
        """规划到可见宝箱旁边，并在相邻时执行交互。"""

        chest_tiles = self._tiles_of_kind(vision, {"chest"})
        if not chest_tiles:
            self.phase = "to_exit"
            return ACTION_NOOP

        adjacent = self._adjacent_targets(chest_tiles, vision)
        if player in adjacent:
            chest = min(chest_tiles, key=lambda tile: manhattan(player, tile))
            self.remembered_blockers.add(chest)
            face_action = action_toward(player, chest)
            if face_action is not None:
                self.queued_actions.append(ACTION_A)
                self.chest_interactions += 1
                return face_action
            self.chest_interactions += 1
            return ACTION_A

        path = bfs_path(player, adjacent, vision, self.remembered_blockers)
        if len(path) >= 2:
            return self._start_tile_step(action_toward(path[0], path[1]), path[1])
        return ACTION_NOOP

    def _act_to_exit(self, player: Position, vision: PixelObservation) -> int:
        """规划到可见出口，并在边界出口格上继续向外推进。"""

        exit_tiles = {
            tile
            for tile in self._tiles_of_kind(vision, {"exit_locked", "exit_normal"})
            if is_structural_exit_tile(tile, vision)
        }
        reachable_exit_tiles = {tile for tile in exit_tiles if self._is_walkable(tile, vision)}

        if self.exit_push_action is not None:
            if self._can_continue_exit_push(player, reachable_exit_tiles):
                return self.exit_push_action
            self.exit_push_action = None
            self.target_exit_tile = None

        if player in reachable_exit_tiles or (
            self.target_exit_tile is not None
            and player == self.target_exit_tile
            and is_boundary_tile(player)
        ):
            self.exit_push_action = boundary_exit_action(player)
            self.target_exit_tile = player
            return self.exit_push_action or ACTION_NOOP

        if not reachable_exit_tiles:
            return ACTION_NOOP

        path = bfs_path(player, reachable_exit_tiles, vision, self.remembered_blockers)
        if len(path) >= 2:
            self.target_exit_tile = path[-1]
            return self._start_tile_step(action_toward(path[0], path[1]), path[1])
        return ACTION_NOOP

    def _can_continue_exit_push(self, player: Position, reachable_exit_tiles: set[Position]) -> bool:
        """检查记住的出口推进动作当前是否仍然合理。"""

        return (
            self.target_exit_tile is not None
            and player == self.target_exit_tile
            and is_boundary_tile(player)
            and self.exit_push_action == boundary_exit_action(player)
            and (player in reachable_exit_tiles or player == self.target_exit_tile)
        )

    def _start_tile_step(self, action: int | None, target_tile: Position | None) -> int:
        """把 BFS 的一格移动展开成若干个像素级重复动作。"""

        if action is None:
            return ACTION_NOOP
        self.queued_target_tile = target_tile
        self.queued_actions.extend([action] * (TILE_SIZE - 1))
        return action

    def _shield_queued_action(self, action: int, vision: PixelObservation) -> int:
        """检查像素级连续移动是否仍然朝着原本的 BFS 目标格前进。

        队列中的动作是“一格 tile 移动”拆成的后续像素帧。这里不能直接调用
        ``_shield_action``，因为玩家还没完全走完整格时，tile 识别可能已经切到
        目标格；如果此时再检查 ``next_position(player, action)``，就会误以为
        策略要继续走到下一格，从而提前打断正常移动。
        """

        if action not in MOVE_ACTIONS:
            return action
        if vision.player is None:
            return ACTION_NOOP
        if self.queued_target_tile is None:
            return self._shield_action(action, vision)

        player = vision.player.tile
        if manhattan(player, self.queued_target_tile) > 1:
            return ACTION_NOOP
        if player == self.queued_target_tile:
            return action
        if not self._is_walkable(self.queued_target_tile, vision):
            return ACTION_NOOP
        return action

    def _shield_action(self, action: int, vision: PixelObservation) -> int:
        """在动作交给环境前，拦截不安全的移动动作。"""

        if action not in MOVE_ACTIONS:
            return action
        if vision.player is None:
            return ACTION_NOOP
        player = vision.player.tile

        # Crossing an exit intentionally steps outside the tile grid. This is a
        # generic boundary rule derived from the visible exit tile, not a fixed
        # task coordinate.
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
        if self.phase == "to_chest" and vision.grid[nxt[1]][nxt[0]] == "chest":
            return action
        if not self._is_walkable(nxt, vision):
            return ACTION_NOOP
        return action

    def _tiles_of_kind(self, vision: PixelObservation, kinds: set[str]) -> set[Position]:
        """返回所有分类类别属于 ``kinds`` 的 tile 坐标。"""

        return {tile.tile for tile in vision.tiles if tile.kind in kinds}

    def _remember_static_blockers(self, vision: PixelObservation) -> None:
        """记住从图像中看到的稳定阻挡物。"""

        self.remembered_blockers.update(self._tiles_of_kind(vision, {"wall", "chest"}))

    def _is_walkable(self, pos: Position, vision: PixelObservation) -> bool:
        """结合当前视觉和静态阻挡记忆判断某格是否可走。"""

        return is_walkable(pos, vision, self.remembered_blockers)

    def _adjacent_targets(self, blocked_targets: set[Position], vision: PixelObservation) -> set[Position]:
        """返回阻挡型交互目标旁边所有可站立的 tile。"""

        out: set[Position] = set()
        for target in blocked_targets:
            for pos in neighbors(target):
                if in_bounds(pos) and self._is_walkable(pos, vision):
                    out.add(pos)
        return out


def bfs_path(
    start: Position,
    goals: set[Position],
    vision: PixelObservation,
    remembered_blockers: set[Position] | frozenset[Position] = frozenset(),
) -> list[Position]:
    """用 BFS 找到从 ``start`` 到任一目标 tile 的最短符号路径。"""

    if start in goals:
        return [start]
    queue: deque[Position] = deque([start])
    parent: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        for nxt in neighbors(current):
            if nxt in parent:
                continue
            if nxt not in goals and not is_walkable(nxt, vision, remembered_blockers):
                continue
            if not in_bounds(nxt):
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


def is_walkable(
    pos: Position,
    vision: PixelObservation,
    remembered_blockers: set[Position] | frozenset[Position] = frozenset(),
) -> bool:
    """判断某个符号 tile 是否适合普通移动进入。"""

    if not in_bounds(pos):
        return False
    if pos in remembered_blockers:
        return False
    kind = vision.grid[pos[1]][pos[0]]
    if kind in BLOCKING_KINDS:
        return False
    return kind in SAFE_WALKABLE_KINDS


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


def is_structural_exit_tile(pos: Position, vision: PixelObservation) -> bool:
    """检查出口是否符合 NesyLink 门的两格结构，过滤单格误报。"""

    if not is_exit_tile(pos, vision):
        return False
    col, row = pos
    if row == 0 or row == GRID_HEIGHT - 1:
        return (
            is_exit_tile((col - 1, row), vision)
            or is_exit_tile((col + 1, row), vision)
        )
    if col == 0 or col == GRID_WIDTH - 1:
        return (
            is_exit_tile((col, row - 1), vision)
            or is_exit_tile((col, row + 1), vision)
        )
    return False


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


Policy = FSMBFSAgent
Agent = FSMBFSAgent
FinalAgent = FSMBFSAgent


def make_policy() -> FSMBFSAgent:
    """评测脚本使用的策略工厂函数。"""

    return FSMBFSAgent()
