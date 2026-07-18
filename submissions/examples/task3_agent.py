"""Task 3 的观测驱动像素策略。

策略推理时只接收当前 RGB、上一动作的标量奖励和安全物品栏。它不会
预设钥匙、怪物或锁门位于哪个方向，也不知道地牢有几个房间。每次实际
换房后，策略把新区域记为匿名节点，并用亲自走过的出口建立双向房间图。

高层调度只依赖当前可见事实和已学习连接：服务宝箱、处理可见怪物、
持钥匙寻找锁门、探索未知普通出口，最后沿房间图回访仍有目标的节点。
房间内移动由可中断 BFS 和像素中心对齐控制，交互必须在真正输出 A 后
再由视觉、物品栏差分或标量奖励确认。
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import (  # noqa: E402
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
from nesylink.vision import PixelObservation, classify_frame_cnn  # noqa: E402


Position = tuple[int, int]
RoomId = int

MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_TO_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
DELTA_TO_ACTION = {delta: action for action, delta in ACTION_TO_DELTA.items()}
DIRECTION_ACTION = {
    "north": ACTION_UP,
    "south": ACTION_DOWN,
    "west": ACTION_LEFT,
    "east": ACTION_RIGHT,
}
OPPOSITE = {"north": "south", "south": "north", "west": "east", "east": "west"}
EXIT_KINDS = {"exit_normal", "exit_locked", "exit_conditional"}
BLOCKING_KINDS = {"wall", "chest", "trap", "abyss", "gap", "monster", "unknown"}
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
    "npc",
}
ROOM_TRANSITION_THRESHOLD = 3
MAX_TILE_MOVE_ATTEMPTS = TILE_SIZE * 5
MAX_COMBAT_MISSES = 4
MAX_EXIT_PERCEPTION_MISSES = TILE_SIZE * 2


@dataclass
class RoomMemory:
    """一个只由实际观测和换房结果建立的匿名房间节点。"""

    entry_direction: str | None = None
    connections: dict[str, RoomId] = field(default_factory=dict)
    exits: set[str] = field(default_factory=set)
    exit_kinds: dict[str, str] = field(default_factory=dict)
    exit_tiles: dict[str, set[Position]] = field(default_factory=dict)
    exit_visits: dict[str, int] = field(default_factory=dict)
    chests: set[Position] = field(default_factory=set)
    opened_chests: set[Position] = field(default_factory=set)
    monster_seen: bool = False
    monster_cleared: bool = False
    visits: int = 0


@dataclass
class Task3Agent:
    """结合 CNN、匿名房间图、BFS 和有限战斗状态机的 Task 3 策略。"""

    current_room: RoomId = 0
    next_room_id: RoomId = 1
    rooms: dict[RoomId, RoomMemory] = field(default_factory=lambda: {0: RoomMemory()})

    keys: int = 0
    last_key_delta: int = 0
    has_sword: bool = False
    last_reward: float = 0.0

    last_player_tile: Position | None = None
    move_target_tile: Position | None = None
    move_action: int | None = None
    move_attempts: int = 0
    exit_push_action: int | None = None
    target_exit_tile: Position | None = None
    pending_exit_direction: str | None = None

    pending_face_kind: str | None = None
    pending_face_target: Position | None = None
    awaiting_kind: str | None = None
    awaiting_target: Position | None = None
    combat_misses: int = 0
    combat_attacks: int = 0

    stagnant_frames: int = 0
    perception_misses: int = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        """清空全部单局状态；seed 和 task_id 不参与策略选择。"""
        del seed, task_id
        self.current_room = 0
        self.next_room_id = 1
        self.rooms = {0: RoomMemory()}
        self.keys = 0
        self.last_key_delta = 0
        self.has_sword = False
        self.last_reward = 0.0
        self.last_player_tile = None
        self._clear_motion()
        self.pending_face_kind = None
        self.pending_face_target = None
        self.awaiting_kind = None
        self.awaiting_target = None
        self.combat_misses = 0
        self.combat_attacks = 0
        self.stagnant_frames = 0
        self.perception_misses = 0

    def act(self, obs, info=None) -> int:
        """根据当前 RGB 和 safe_info 返回合法动作 0..6。"""
        self._read_safe_feedback(info)
        try:
            vision = classify_frame_cnn(obs, fallback=False)
        except Exception:
            self.perception_misses += 1
            return self._action_during_perception_miss()
        if vision.player is None:
            self.perception_misses += 1
            return self._action_during_perception_miss()
        self.perception_misses = 0
        player = vision.player.tile

        previous = self.last_player_tile
        if (
            previous is not None
            and self.pending_exit_direction is not None
            and manhattan(previous, player) > ROOM_TRANSITION_THRESHOLD
        ):
            self._confirm_room_change()
        self.last_player_tile = player
        self._observe_room(vision)
        self._confirm_interaction(vision)

        if previous == player:
            self.stagnant_frames += 1
        else:
            self.stagnant_frames = 0
        if self.stagnant_frames >= MAX_TILE_MOVE_ATTEMPTS * 2:
            self._clear_motion()
            self.pending_face_kind = None
            self.pending_face_target = None
            self.stagnant_frames = 0

        # 上一帧只负责面向目标；这一帧才真正交互，并等待下一帧确认。
        if self.pending_face_kind is not None:
            self.awaiting_kind = self.pending_face_kind
            self.awaiting_target = self.pending_face_target
            self.pending_face_kind = None
            self.pending_face_target = None
            if self.awaiting_kind == "monster":
                self.combat_attacks += 1
            return ACTION_A

        continued = self._continue_motion(player, vision)
        if continued is not None:
            return self._safe_action(continued, vision)

        action = self._choose_action(player, vision)
        action = self._safe_action(action, vision)
        return int(action if action in range(7) else ACTION_NOOP)

    # ------------------------------------------------------------------
    # 安全反馈、感知记忆与执行确认

    def _read_safe_feedback(self, info) -> None:
        """仅读取允许的 last_reward 和 inventory，并计算物品栏差分。"""
        reward = 0.0
        inventory = None
        if isinstance(info, dict):
            try:
                reward = float(info.get("last_reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                reward = 0.0
            inventory = info.get("inventory")
        old_keys = self.keys
        if isinstance(inventory, dict):
            try:
                self.keys = max(0, int(inventory.get("keys", self.keys) or 0))
            except (TypeError, ValueError):
                pass
            tools = inventory.get("tools", ())
            items = inventory.get("items", ())
            equipped = inventory.get("equipped", {})
            self.has_sword = (
                isinstance(tools, (list, tuple, set))
                and "sword" in tools
            ) or (
                isinstance(items, (list, tuple, set))
                and "sword" in items
            ) or (
                isinstance(equipped, dict) and equipped.get("A") == "sword"
            )
        self.last_key_delta = self.keys - old_keys
        self.last_reward = reward
        # 小额负奖励表示上一移动被挡；立即丢弃旧的局部移动意图。
        if -0.20 <= reward <= -0.04:
            self.move_target_tile = None
            self.move_action = None
            self.move_attempts = 0

    def _memory(self) -> RoomMemory:
        """返回当前匿名房间的记忆。"""
        return self.rooms[self.current_room]

    def _action_during_perception_miss(self) -> int:
        """在动态玩家短暂漏检时有限推算动作，随后安全停止。

        玩家逐像素穿过边界时精灵会先离开当前画面，再出现在目标房间。
        若出口及推进方向已由此前画面确认，可在有限窗口继续向外推进；
        普通房间内移动最多复用两帧，避免把旧路径盲目执行到障碍中。
        """
        if (
            self.pending_exit_direction is not None
            and self.exit_push_action is not None
            and self.perception_misses <= MAX_EXIT_PERCEPTION_MISSES
        ):
            return self.exit_push_action
        if self.move_action is not None and self.perception_misses <= 2:
            return self.move_action
        if self.perception_misses >= 3:
            self._clear_motion()
        return ACTION_NOOP

    def _observe_room(self, vision: PixelObservation) -> None:
        """把当前帧可见出口、宝箱和怪物合并到当前房间记忆。"""
        memory = self._memory()
        visible_chests = self._tiles_of_kind(vision, {"chest"})
        memory.chests.update(visible_chests)
        if vision.monsters:
            memory.monster_seen = True
            memory.monster_cleared = False

        for tile in vision.tiles:
            if tile.kind not in EXIT_KINDS or not is_boundary_tile(tile.tile):
                continue
            for direction in boundary_directions(tile.tile):
                memory.exits.add(direction)
                memory.exit_kinds[direction] = tile.kind
                memory.exit_tiles.setdefault(direction, set()).add(tile.tile)

    def _confirm_interaction(self, vision: PixelObservation) -> None:
        """用当前视觉、奖励和钥匙差分确认上一帧真正执行的 A。"""
        kind = self.awaiting_kind
        target = self.awaiting_target
        if kind is None:
            return
        self.awaiting_kind = None
        self.awaiting_target = None
        memory = self._memory()

        if kind == "chest" and target is not None:
            visible = self._tiles_of_kind(vision, {"chest"})
            if target not in visible or self.last_key_delta > 0 or self.last_reward > 1.0:
                memory.opened_chests.add(target)
            return

        if kind == "monster":
            if not vision.monsters:
                memory.monster_cleared = True
                self.combat_misses = 0
                self.combat_attacks = 0
            elif self.last_reward > 0.2:
                self.combat_misses = 0
            else:
                self.combat_misses += 1

    def _confirm_room_change(self) -> None:
        """边界推进成功后创建或复用节点，并建立实际走过的双向边。"""
        old_id = self.current_room
        direction = self.pending_exit_direction
        if direction is None:
            return
        old = self.rooms[old_id]
        if direction in old.connections:
            new_id = old.connections[direction]
        else:
            new_id = self.next_room_id
            self.next_room_id += 1
            self.rooms[new_id] = RoomMemory(entry_direction=OPPOSITE[direction])
            old.connections[direction] = new_id
            self.rooms[new_id].connections[OPPOSITE[direction]] = old_id
        old.exit_visits[direction] = old.exit_visits.get(direction, 0) + 1
        self.current_room = new_id
        self.rooms[new_id].visits += 1
        self.pending_face_kind = None
        self.pending_face_target = None
        self.awaiting_kind = None
        self.awaiting_target = None
        self.combat_misses = 0
        self.combat_attacks = 0
        self._clear_motion()

    # ------------------------------------------------------------------
    # 观测驱动目标调度与房间图搜索

    def _choose_action(self, player: Position, vision: PixelObservation) -> int:
        """按当前目标、物品前置条件和已学习房间图选择下一动作。"""
        memory = self._memory()
        visible_chests = self._tiles_of_kind(vision, {"chest"})
        unopened = visible_chests - memory.opened_chests
        if unopened:
            return self._act_open_chest(player, vision, unopened)

        if vision.monsters:
            if self.has_sword:
                return self._act_monster(player, vision)
            return self._act_return(player, vision)

        # 钥匙到手后优先服务已经观察到的锁门，不假定它位于哪个方向。
        if self.keys > 0:
            locked_here = [
                direction
                for direction in self._reachable_directions(player, vision)
                if memory.exit_kinds.get(direction) == "exit_locked"
            ]
            if locked_here:
                return self._act_best_exit(player, vision, locked_here)
            route = self._route_to_room(
                lambda room: any(kind == "exit_locked" for kind in room.exit_kinds.values())
            )
            if route is not None:
                return self._act_exit_directional(player, vision, route)

        unknown = [
            direction
            for direction in self._reachable_directions(player, vision)
            if direction not in memory.connections
            and self._exit_prerequisite_met(memory.exit_kinds.get(direction, "exit_normal"))
        ]
        if unknown:
            return self._act_best_exit(player, vision, unknown)

        route = self._route_to_room(self._room_serviceable)
        if route is not None:
            return self._act_exit_directional(player, vision, route)
        return self._act_return(player, vision)

    def _exit_prerequisite_met(self, kind: str) -> bool:
        """根据公开物品条件判断出口当前是否值得尝试。"""
        if kind == "exit_locked":
            return self.keys > 0
        return kind == "exit_normal"

    def _room_serviceable(self, memory: RoomMemory) -> bool:
        """判断一个已知房间在当前物品条件下是否仍有观测目标。"""
        if memory.chests - memory.opened_chests:
            return True
        if memory.monster_seen and not memory.monster_cleared and self.has_sword:
            return True
        if self.keys > 0 and any(kind == "exit_locked" for kind in memory.exit_kinds.values()):
            return True
        return any(
            direction not in memory.connections and self._exit_prerequisite_met(kind)
            for direction, kind in memory.exit_kinds.items()
        )

    def _route_to_room(self, predicate) -> str | None:
        """在已学习房间图上 BFS，返回前往任一目标节点的第一条边。"""
        queue: deque[tuple[RoomId, str | None]] = deque([(self.current_room, None)])
        visited = {self.current_room}
        while queue:
            room_id, first_direction = queue.popleft()
            if room_id != self.current_room and predicate(self.rooms[room_id]):
                return first_direction
            memory = self.rooms[room_id]
            for direction, neighbor in sorted(memory.connections.items()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, first_direction or direction))
        return None

    def _act_return(self, player: Position, vision: PixelObservation) -> int:
        """当前节点无目标时沿入口回退；无入口则选最少使用的已知边。"""
        memory = self._memory()
        reachable = self._reachable_directions(player, vision)
        if memory.entry_direction in reachable:
            return self._act_exit_directional(player, vision, memory.entry_direction)
        connected = [direction for direction in reachable if direction in memory.connections]
        if connected:
            connected.sort(key=lambda direction: (memory.exit_visits.get(direction, 0), direction))
            return self._act_exit_directional(player, vision, connected[0])
        return ACTION_NOOP

    # ------------------------------------------------------------------
    # 宝箱、战斗、出口与局部移动

    def _act_open_chest(
        self, player: Position, vision: PixelObservation, chests: set[Position]
    ) -> int:
        """BFS 到任一宝箱相邻格，面向后在下一帧真正按 A。"""
        adjacent = self._adjacent_positions(chests, vision)
        if player in adjacent:
            target = min(chests, key=lambda pos: (manhattan(player, pos), pos))
            face = action_toward(player, target)
            if face is not None:
                self.pending_face_kind = "chest"
                self.pending_face_target = target
                return face
            return ACTION_NOOP
        return self._start_path_step(bfs_path(player, adjacent, vision))

    def _act_monster(self, player: Position, vision: PixelObservation) -> int:
        """只在确认有剑时接近当前可见怪物，并限制连续无效攻击。"""
        targets = {monster.tile for monster in vision.monsters}
        target = min(targets, key=lambda pos: (manhattan(player, pos), pos))
        if self.combat_misses >= MAX_COMBAT_MISSES:
            self.combat_misses = 0
            retreat = [
                pos
                for pos in neighbors(player)
                if in_bounds(pos)
                and is_walkable(pos, vision)
                and min(manhattan(pos, monster) for monster in targets) > 1
            ]
            if retreat:
                action = action_toward(player, retreat[0])
                if action is not None:
                    return self._begin_tile_move(action, retreat[0])
        if manhattan(player, target) == 1:
            face = action_toward(player, target)
            if face is not None:
                self.pending_face_kind = "monster"
                self.pending_face_target = target
                return face
            return ACTION_NOOP
        adjacent = self._adjacent_positions(targets, vision)
        return self._start_path_step(bfs_path(player, adjacent, vision))

    def _reachable_directions(self, player: Position, vision: PixelObservation) -> list[str]:
        """返回当前画面中有符号路径可达的已观察出口方向。"""
        memory = self._memory()
        ranked: list[tuple[int, int, str]] = []
        for direction in memory.exits:
            path = bfs_path(player, self._exit_targets(vision, direction), vision)
            if path:
                ranked.append((memory.exit_visits.get(direction, 0), len(path), direction))
        return [direction for _, _, direction in sorted(ranked)]

    def _act_best_exit(
        self, player: Position, vision: PixelObservation, directions: list[str]
    ) -> int:
        """按使用次数和当前真实 BFS 距离选择出口，不采用固定方向顺序。"""
        memory = self._memory()
        ranked: list[tuple[int, int, str]] = []
        for direction in directions:
            path = bfs_path(player, self._exit_targets(vision, direction), vision)
            if path:
                ranked.append((memory.exit_visits.get(direction, 0), len(path), direction))
        if not ranked:
            return ACTION_NOOP
        _, _, direction = min(ranked)
        return self._act_exit_directional(player, vision, direction)

    def _act_exit_directional(
        self, player: Position, vision: PixelObservation, direction: str
    ) -> int:
        """导航到观测出口并持续向边界外推进，直到换房得到视觉确认。"""
        targets = self._exit_targets(vision, direction)
        if self.exit_push_action is not None:
            if (
                self.pending_exit_direction == direction
                and self.target_exit_tile == player
                and direction in boundary_directions(player)
            ):
                return self.exit_push_action
            self._clear_motion()
        if not targets:
            return ACTION_NOOP
        remembered = self.target_exit_tile == player and direction in boundary_directions(player)
        if player in targets or remembered:
            action = DIRECTION_ACTION[direction]
            self.exit_push_action = action
            self.target_exit_tile = player
            self.pending_exit_direction = direction
            return action
        path = bfs_path(player, targets, vision)
        if len(path) >= 2:
            self.target_exit_tile = path[-1]
            action = action_toward(path[0], path[1])
            if action is not None:
                return self._begin_tile_move(action, path[1])
        return ACTION_NOOP

    def _exit_targets(self, vision: PixelObservation, direction: str) -> set[Position]:
        """返回该方向已观察且当前可走的出口格，保留被玩家遮挡的旧位置。"""
        memory = self._memory()
        visible = {
            tile.tile
            for tile in vision.tiles
            if tile.kind in EXIT_KINDS and direction in boundary_directions(tile.tile)
        }
        if visible:
            memory.exit_tiles.setdefault(direction, set()).update(visible)
        return {
            pos
            for pos in memory.exit_tiles.get(direction, set())
            if is_walkable(pos, vision)
        }

    def _start_path_step(self, path: list[Position]) -> int:
        """开始执行 BFS 路径的第一个 tile 步。"""
        if len(path) < 2:
            return ACTION_NOOP
        action = action_toward(path[0], path[1])
        if action is None:
            return ACTION_NOOP
        return self._begin_tile_move(action, path[1])

    def _begin_tile_move(self, action: int, target: Position) -> int:
        """建立一个短移动意图，并在之后每帧复核目标。"""
        self.move_action = action
        self.move_target_tile = target
        self.move_attempts = 0
        return action

    def _continue_motion(self, player: Position, vision: PixelObservation) -> int | None:
        """持续短移动，到格、超时或发现像素轴偏移时立即重新处理。"""
        if self.move_target_tile is None or self.move_action is None:
            return None
        self.move_attempts += 1
        if player == self.move_target_tile or self.move_attempts >= MAX_TILE_MOVE_ATTEMPTS:
            self.move_target_tile = None
            self.move_action = None
            self.move_attempts = 0
            return None
        alignment = self._boundary_alignment_action(vision)
        return self.move_action if alignment is None else alignment

    def _boundary_alignment_action(self, vision: PixelObservation) -> int | None:
        """进入边界出口前按玩家 bbox 中心对齐另一像素轴。"""
        if self.move_target_tile is None or self.move_action is None or vision.player is None:
            return None
        x, y = self.move_target_tile
        center_x, center_y = vision.player.center_px
        if x in {0, GRID_WIDTH - 1} and self.move_action in {ACTION_LEFT, ACTION_RIGHT}:
            desired = y * TILE_SIZE + TILE_SIZE * 0.5
            if center_y < desired - 0.5:
                return ACTION_DOWN
            if center_y > desired + 0.5:
                return ACTION_UP
        if y in {0, GRID_HEIGHT - 1} and self.move_action in {ACTION_UP, ACTION_DOWN}:
            desired = x * TILE_SIZE + TILE_SIZE * 0.5
            if center_x < desired - 0.5:
                return ACTION_RIGHT
            if center_x > desired + 0.5:
                return ACTION_LEFT
        return None

    def _clear_motion(self) -> None:
        """清除局部移动及出口推进状态。"""
        self.move_target_tile = None
        self.move_action = None
        self.move_attempts = 0
        self.exit_push_action = None
        self.target_exit_tile = None
        self.pending_exit_direction = None

    def _safe_action(self, action: int, vision: PixelObservation) -> int:
        """拦截越界和明确阻挡；已确认出口推进、面向交互除外。"""
        if action not in MOVE_ACTIONS or vision.player is None:
            return action
        player = vision.player.tile
        if (
            self.exit_push_action == action
            and self.pending_exit_direction in boundary_directions(player)
            and self.target_exit_tile == player
        ):
            return action
        nxt = next_position(player, action)
        if not in_bounds(nxt):
            return ACTION_NOOP
        if vision.grid[nxt[1]][nxt[0]] in {"chest", "monster"}:
            return action
        return action if is_walkable(nxt, vision) else ACTION_NOOP

    def _tiles_of_kind(
        self, vision: PixelObservation, kinds: set[str]
    ) -> set[Position]:
        """提取当前 CNN 观测中属于指定类别的 tile。"""
        return {tile.tile for tile in vision.tiles if tile.kind in kinds}

    def _adjacent_positions(
        self, positions: set[Position], vision: PixelObservation
    ) -> set[Position]:
        """返回一组阻挡目标周围可供 BFS 到达的四邻格。"""
        return {
            neighbor
            for position in positions
            for neighbor in neighbors(position)
            if in_bounds(neighbor) and is_walkable(neighbor, vision)
        }


def bfs_path(
    start: Position, goals: set[Position], vision: PixelObservation
) -> list[Position]:
    """在当前视觉符号网格上执行四邻接 BFS；不可达时返回空列表。"""
    if start in goals:
        return [start]
    queue: deque[Position] = deque([start])
    parent: dict[Position, Position | None] = {start: None}
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current):
            if nxt in parent or not in_bounds(nxt):
                continue
            if nxt not in goals and not is_walkable(nxt, vision):
                continue
            parent[nxt] = current
            if nxt in goals:
                return reconstruct_path(parent, nxt)
            queue.append(nxt)
    return []


def reconstruct_path(
    parent: dict[Position, Position | None], goal: Position
) -> list[Position]:
    """由 BFS 前驱表回溯并返回从起点到终点的正向路径。"""
    path: list[Position] = []
    current: Position | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def is_walkable(pos: Position, vision: PixelObservation) -> bool:
    """判断当前视觉网格中的位置是否可安全通行。"""
    if not in_bounds(pos):
        return False
    kind = vision.grid[pos[1]][pos[0]]
    return kind not in BLOCKING_KINDS and kind in SAFE_WALKABLE_KINDS


def neighbors(pos: Position) -> tuple[Position, Position, Position, Position]:
    """返回位置的上、下、左、右四个邻居。"""
    x, y = pos
    return ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))


def in_bounds(pos: Position) -> bool:
    """判断 tile 是否位于 10×8 地图区域。"""
    return 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT


def boundary_directions(pos: Position) -> tuple[str, ...]:
    """返回一个边界 tile 所属的全部方向；内部 tile 返回空元组。"""
    directions: list[str] = []
    if pos[1] == 0:
        directions.append("north")
    if pos[1] == GRID_HEIGHT - 1:
        directions.append("south")
    if pos[0] == 0:
        directions.append("west")
    if pos[0] == GRID_WIDTH - 1:
        directions.append("east")
    return tuple(directions)


def is_boundary_tile(pos: Position) -> bool:
    """判断位置是否处于任一房间边界。"""
    return bool(boundary_directions(pos))


def next_position(pos: Position, action: int) -> Position:
    """把一个移动动作应用到 tile 坐标。"""
    dx, dy = ACTION_TO_DELTA[action]
    return (pos[0] + dx, pos[1] + dy)


def action_toward(current: Position, nxt: Position) -> int | None:
    """若 nxt 是 current 的四邻格，返回对应移动动作。"""
    return DELTA_TO_ACTION.get((nxt[0] - current[0], nxt[1] - current[1]))


def manhattan(left: Position, right: Position) -> int:
    """返回两个 tile 的曼哈顿距离。"""
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


Policy = Task3Agent


def make_policy() -> Task3Agent:
    """为每个 episode 创建全新的观测驱动 Task3 策略。"""
    return Task3Agent()
