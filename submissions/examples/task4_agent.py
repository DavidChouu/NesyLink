"""Task 4 的观测驱动像素策略。

策略不预设钥匙、剑、怪物或最终宝箱所在方向，也不记忆固定
物体坐标。它仅根据当前 RGB 的 CNN 分类、允许的物品栏以及已经
亲自验证的换房结果，动态建立匿名房间图并服务当前可达目标。

旋转桥状态由桥 tile 抵达的边界方向集合作为视觉指纹。当前桥没有
可服务分支时，Agent 回到已发现的机关房转桥，然后继续探索；代码
不知道桥有几种状态或它们的循环顺序。
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
RoomId = int
BridgeFingerprint = frozenset[str]

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


@dataclass
class RoomMemory:
    """一个由实际换房创建的匿名房间节点。"""

    kind: str = "unknown"
    entry_direction: str | None = None
    connections: dict[str, RoomId] = field(default_factory=dict)
    exits: set[str] = field(default_factory=set)
    exit_kinds: dict[str, str] = field(default_factory=dict)
    chests: set[Position] = field(default_factory=set)
    opened_chests: set[Position] = field(default_factory=set)
    switch_pos: Position | None = None
    monster_seen: bool = False
    monster_cleared: bool = False
    bridge_modes: dict[BridgeFingerprint, int] = field(default_factory=dict)
    current_bridge_mode: BridgeFingerprint = frozenset()


@dataclass
class Task4Agent:
    """RGB + 匿名房间图 + 动态桥指纹 + BFS 策略。"""

    current_room: RoomId = 0
    next_room_id: RoomId = 1
    rooms: dict[RoomId, RoomMemory] = field(default_factory=lambda: {0: RoomMemory()})
    world_revision: int = 0

    keys: int = 0
    has_sword: bool = False
    inventory_revision: int = 0

    last_player_tile: Position | None = None
    move_target_tile: Position | None = None
    move_action: int | None = None
    move_attempts: int = 0
    exit_push_action: int | None = None
    target_exit_tile: Position | None = None
    pending_exit_direction: str | None = None

    pending_interaction_kind: str | None = None
    pending_interaction_target: Position | None = None
    awaiting_monster_result: bool = False
    depart_after_switch: bool = False
    stuck_counter: int = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        """彻底清空局内学习结果；seed 和 task_id 不参与决策。"""
        del seed, task_id
        self.current_room = 0
        self.next_room_id = 1
        self.rooms = {0: RoomMemory()}
        self.world_revision = 0
        self.keys = 0
        self.has_sword = False
        self.inventory_revision = 0
        self.last_player_tile = None
        self._clear_motion()
        self.pending_interaction_kind = None
        self.pending_interaction_target = None
        self.awaiting_monster_result = False
        self.depart_after_switch = False
        self.stuck_counter = 0

    def act(self, obs, info=None) -> int:
        """根据当前 RGB 与 safe inventory 返回 0..6 的动作。"""
        self._update_inventory(info)
        try:
            vision = classify_frame_cnn(obs, fallback=False)
        except Exception:
            return ACTION_NOOP
        if vision.player is None:
            return ACTION_NOOP
        player = vision.player.tile

        previous = self.last_player_tile
        if previous is not None and manhattan(player, previous) > ROOM_TRANSITION_THRESHOLD:
            self._confirm_room_change()
        self.last_player_tile = player

        self._observe_room(vision)

        if previous == player:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > MAX_TILE_MOVE_ATTEMPTS * 3:
            self._clear_motion()
            self.stuck_counter = 0

        if self.pending_interaction_kind is not None:
            kind = self.pending_interaction_kind
            target = self.pending_interaction_target
            self.pending_interaction_kind = None
            self.pending_interaction_target = None
            memory = self._memory()
            if kind == "chest" and target is not None:
                memory.opened_chests.add(target)
                self.world_revision += 1
            elif kind == "switch":
                self.depart_after_switch = True
            elif kind == "monster":
                self.awaiting_monster_result = True
            return ACTION_A

        continued = self._continue_motion(player, vision)
        if continued is not None:
            return self._safe_action(continued, vision)

        action = self._choose_action(player, vision)
        return self._safe_action(action, vision)

    def _memory(self) -> RoomMemory:
        """取得当前匿名房间记忆。"""
        return self.rooms[self.current_room]

    def _update_inventory(self, info) -> None:
        """仅从允许的 inventory 更新钥匙和剑状态。"""
        if not isinstance(info, dict) or not isinstance(info.get("inventory"), dict):
            return
        inventory = info["inventory"]
        old = (self.keys, self.has_sword)
        try:
            self.keys = max(0, int(inventory.get("keys", self.keys) or 0))
        except (TypeError, ValueError):
            pass
        tools = inventory.get("tools", ())
        items = inventory.get("items", ())
        equipped = inventory.get("equipped", {})
        self.has_sword = (
            "sword" in tools
            or "sword" in items
            or (isinstance(equipped, dict) and equipped.get("A") == "sword")
        )
        if old != (self.keys, self.has_sword):
            self.inventory_revision += 1
            self.world_revision += 1

    def _observe_room(self, vision: PixelObservation) -> None:
        """将当前画面合并到房间记忆，不使用房间真值。"""
        memory = self._memory()
        raw_switches = self._tiles_of_kind(vision, {"switch"})
        pressed = self._tiles_of_kind(vision, {"button_pressed"})
        if raw_switches:
            memory.switch_pos = min(raw_switches)
            memory.kind = "switch"
        elif memory.switch_pos is not None and memory.switch_pos in pressed:
            memory.kind = "switch"

        bridges = self._tiles_of_kind(vision, {"bridge"})
        abyss_count = len(self._tiles_of_kind(vision, {"abyss"}))
        if bridges or abyss_count > GRID_WIDTH:
            memory.kind = "hub"
        elif memory.kind == "unknown":
            memory.kind = "leaf"

        memory.chests.update(self._tiles_of_kind(vision, {"chest"}))
        monsters_visible = bool(vision.monsters)
        if monsters_visible:
            memory.monster_seen = True
            memory.monster_cleared = False
        elif memory.monster_seen and self.awaiting_monster_result:
            memory.monster_cleared = True
            self.awaiting_monster_result = False
            self.world_revision += 1

        for tile in vision.tiles:
            if tile.kind not in EXIT_KINDS or not is_boundary_tile(tile.tile):
                continue
            for direction in boundary_directions(tile.tile):
                memory.exits.add(direction)
                memory.exit_kinds[direction] = tile.kind
        for pos in bridges:
            for direction in boundary_directions(pos):
                memory.exits.add(direction)

        if memory.kind == "hub":
            # CNN 对桥面本身的类别置信度可低于出口，因此桥状态不仅
            # 依赖 "bridge" 标签，而是用边界上确实可通行的方向集合建立
            # 视觉指纹。深渊方向没有可通行边界格，不会进入指纹。
            fingerprint = frozenset(
                direction
                for direction in DIRECTION_ACTION
                if any(is_walkable(pos, vision) for pos in boundary_tiles(direction))
            )
            memory.current_bridge_mode = fingerprint
            memory.bridge_modes[fingerprint] = self.world_revision

    def _confirm_room_change(self) -> None:
        """仅在边界推进后玩家 tile 跨越时建立双向房间连接。"""
        old_id = self.current_room
        direction = self.pending_exit_direction
        old = self.rooms[old_id]
        if direction is None:
            new_id = self.next_room_id
            self.next_room_id += 1
            self.rooms[new_id] = RoomMemory()
        elif direction in old.connections:
            new_id = old.connections[direction]
        else:
            new_id = self.next_room_id
            self.next_room_id += 1
            self.rooms[new_id] = RoomMemory(entry_direction=OPPOSITE[direction])
            old.connections[direction] = new_id
            self.rooms[new_id].connections[OPPOSITE[direction]] = old_id
        self.current_room = new_id
        self.depart_after_switch = False
        self.pending_interaction_kind = None
        self.pending_interaction_target = None
        self.awaiting_monster_result = False
        self._clear_motion()

    def _choose_action(self, player: Position, vision: PixelObservation) -> int:
        """按当前可观测目标和已学习房间图选择动作。"""
        memory = self._memory()
        unopened = self._tiles_of_kind(vision, {"chest"}) - memory.opened_chests
        if unopened:
            return self._act_open_chest(player, vision, unopened)

        if vision.monsters:
            if self.has_sword:
                return self._act_monster(player, vision)
            return self._return_from_room(player, vision)

        if memory.kind == "switch":
            return self._act_switch_room(player, vision)
        if memory.kind == "hub":
            return self._act_hub_room(player, vision)
        return self._return_from_room(player, vision)

    def _act_switch_room(self, player: Position, vision: PixelObservation) -> int:
        """机关房先探索未知出口；已连接桥房后才按需转桥。"""
        memory = self._memory()
        unknown = [direction for direction in self._reachable_directions(player, vision) if direction not in memory.connections]
        if unknown:
            return self._act_best_exit(player, vision, unknown)
        if self.depart_after_switch:
            directions = self._reachable_directions(player, vision)
            if directions:
                return self._act_best_exit(player, vision, directions)
            return ACTION_NOOP
        return self._act_switch(player, vision)

    def _act_hub_room(self, player: Position, vision: PixelObservation) -> int:
        """优先服务当前桥可达的未知或未完成分支，否则回机关房。"""
        memory = self._memory()
        reachable = self._reachable_directions(player, vision)
        candidates: list[str] = []
        switch_returns: list[str] = []
        for direction in reachable:
            if memory.exit_kinds.get(direction) == "exit_locked" and self.keys <= 0:
                continue
            neighbor_id = memory.connections.get(direction)
            if neighbor_id is None:
                candidates.append(direction)
                continue
            neighbor = self.rooms[neighbor_id]
            if neighbor.kind == "switch":
                switch_returns.append(direction)
            elif self._room_serviceable(neighbor):
                candidates.append(direction)
        if candidates:
            return self._act_best_exit(player, vision, candidates)
        if switch_returns:
            return self._act_best_exit(player, vision, switch_returns)
        if memory.entry_direction in reachable:
            return self._act_exit_directional(player, vision, memory.entry_direction)
        return ACTION_NOOP

    def _room_serviceable(self, memory: RoomMemory) -> bool:
        """判断已知房间在当前物品条件下是否仍有可执行目标。"""
        if memory.chests - memory.opened_chests:
            return True
        return memory.monster_seen and not memory.monster_cleared and self.has_sword

    def _return_from_room(self, player: Position, vision: PixelObservation) -> int:
        """叶子房完成或前置条件不足时，沿实际入口返回。"""
        memory = self._memory()
        if memory.entry_direction is not None:
            return self._act_exit_directional(player, vision, memory.entry_direction)
        directions = self._reachable_directions(player, vision)
        if directions:
            return self._act_best_exit(player, vision, directions)
        return ACTION_NOOP

    def _act_switch(self, player: Position, vision: PixelObservation) -> int:
        """导航到视觉开关相邻格，面向后在下一帧按 A。"""
        memory = self._memory()
        raw = self._tiles_of_kind(vision, {"switch"})
        if raw:
            memory.switch_pos = min(raw)
        if memory.switch_pos is None:
            return ACTION_NOOP
        target = memory.switch_pos
        if manhattan(player, target) == 1:
            face = action_toward(player, target)
            if face is not None:
                self.pending_interaction_kind = "switch"
                self.pending_interaction_target = target
                return face
            return ACTION_NOOP
        path = bfs_path(player, self._adjacent_positions({target}, vision), vision)
        return self._start_path_step(path)

    def _act_open_chest(self, player: Position, vision: PixelObservation, chests: set[Position]) -> int:
        """靠近任一未交互宝箱，真正按 A 时才记录为已开。"""
        adjacent = self._adjacent_positions(chests, vision)
        if player in adjacent:
            target = min(chests, key=lambda pos: manhattan(player, pos))
            face = action_toward(player, target)
            if face is not None:
                self.pending_interaction_kind = "chest"
                self.pending_interaction_target = target
                return face
            return ACTION_NOOP
        return self._start_path_step(bfs_path(player, adjacent, vision))

    def _act_monster(self, player: Position, vision: PixelObservation) -> int:
        """仅在物品栏确认有剑时靠近并攻击当前可见怪物。"""
        targets = {monster.tile for monster in vision.monsters}
        target = min(targets, key=lambda pos: manhattan(player, pos))
        if manhattan(player, target) == 1:
            face = action_toward(player, target)
            if face is not None:
                self.pending_interaction_kind = "monster"
                self.pending_interaction_target = target
                return face
            return ACTION_NOOP
        return self._start_path_step(bfs_path(player, self._adjacent_positions(targets, vision), vision))

    def _reachable_directions(self, player: Position, vision: PixelObservation) -> list[str]:
        """返回当前画面中确实有像素路径可达的边界方向。"""
        memory = self._memory()
        out: list[tuple[int, str]] = []
        for direction in sorted(memory.exits):
            targets = self._exit_targets(vision, direction)
            path = bfs_path(player, targets, vision)
            if path:
                out.append((len(path), direction))
        return [direction for _, direction in sorted(out)]

    def _act_best_exit(self, player: Position, vision: PixelObservation, directions: list[str]) -> int:
        """按当前 BFS 实际距离选择出口，不使用固定方向优先级。"""
        ranked: list[tuple[int, str]] = []
        for direction in directions:
            path = bfs_path(player, self._exit_targets(vision, direction), vision)
            if path:
                ranked.append((len(path), direction))
        if not ranked:
            return ACTION_NOOP
        _, direction = min(ranked)
        return self._act_exit_directional(player, vision, direction)

    def _act_exit_directional(self, player: Position, vision: PixelObservation, direction: str) -> int:
        """导航至指定边界的可达出口并持续向外推进。"""
        targets = self._exit_targets(vision, direction)
        if self.exit_push_action is not None:
            if (
                self.pending_exit_direction == direction
                and self.target_exit_tile == player
                and is_boundary_tile(player)
            ):
                return self.exit_push_action
            self.exit_push_action = None
            self.target_exit_tile = None
            self.pending_exit_direction = None
        if not targets:
            return ACTION_NOOP
        # 玩家覆盖出口格后，CNN 会把该格改分为 player，因而它会从
        # 当前帧的 exit targets 中消失。保留上一次 BFS 已验证的边界
        # 目标，避免在同一出口的两个格之间来回振荡。
        reached_remembered_target = (
            self.target_exit_tile == player and direction in boundary_directions(player)
        )
        if player in targets or reached_remembered_target:
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
        """使用边界出口分类，并在漏检时退化到可通行边界格。"""
        visible = {
            tile.tile
            for tile in vision.tiles
            if tile.kind in EXIT_KINDS and direction in boundary_directions(tile.tile)
        }
        reachable = {pos for pos in visible if is_walkable(pos, vision)}
        if reachable:
            return reachable
        return {pos for pos in boundary_tiles(direction) if is_walkable(pos, vision)}

    def _start_path_step(self, path: list[Position]) -> int:
        """执行 BFS 路径的第一个 tile 移动。"""
        if len(path) < 2:
            return ACTION_NOOP
        action = action_toward(path[0], path[1])
        if action is None:
            return ACTION_NOOP
        return self._begin_tile_move(action, path[1])

    def _begin_tile_move(self, action: int, target: Position) -> int:
        self.move_action = action
        self.move_target_tile = target
        self.move_attempts = 0
        return action

    def _continue_motion(self, player: Position, vision: PixelObservation) -> int | None:
        """持续上一个短移动，到格、超时或需边界对齐时立即处理。"""
        if self.move_target_tile is None or self.move_action is None:
            return None
        self.move_attempts += 1
        if player == self.move_target_tile or self.move_attempts >= MAX_TILE_MOVE_ATTEMPTS:
            self.move_target_tile = None
            self.move_action = None
            self.move_attempts = 0
            return None
        align = self._boundary_alignment_action(vision)
        return self.move_action if align is None else align

    def _boundary_alignment_action(self, vision: PixelObservation) -> int | None:
        """到达边界出口格前对齐垂直像素轴，避免角落墙体碰撞。"""
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
        self.move_target_tile = None
        self.move_action = None
        self.move_attempts = 0
        self.exit_push_action = None
        self.target_exit_tile = None
        self.pending_exit_direction = None

    def _safe_action(self, action: int, vision: PixelObservation) -> int:
        """拦截越界或明确不可通行的移动，已确认出口推进除外。"""
        if action not in MOVE_ACTIONS or vision.player is None:
            return action
        player = vision.player.tile
        if self.exit_push_action == action and self.target_exit_tile == player and is_boundary_tile(player):
            return action
        nxt = next_position(player, action)
        if not in_bounds(nxt):
            return ACTION_NOOP
        if vision.grid[nxt[1]][nxt[0]] in {"chest", "switch", "button_pressed", "monster"}:
            return action
        return action if is_walkable(nxt, vision) else ACTION_NOOP

    def _tiles_of_kind(self, vision: PixelObservation, kinds: set[str]) -> set[Position]:
        return {tile.tile for tile in vision.tiles if tile.kind in kinds}

    def _adjacent_positions(self, positions: set[Position], vision: PixelObservation) -> set[Position]:
        return {
            neighbor
            for position in positions
            for neighbor in neighbors(position)
            if in_bounds(neighbor) and is_walkable(neighbor, vision)
        }


def bfs_path(start: Position, goals: set[Position], vision: PixelObservation) -> list[Position]:
    """在当前视觉网格上执行标准 BFS。"""
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


def reconstruct_path(parent: dict[Position, Position | None], goal: Position) -> list[Position]:
    path: list[Position] = []
    current: Position | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def is_walkable(pos: Position, vision: PixelObservation) -> bool:
    if not in_bounds(pos):
        return False
    kind = vision.grid[pos[1]][pos[0]]
    return kind not in BLOCKING_KINDS and kind in SAFE_WALKABLE_KINDS


def neighbors(pos: Position) -> tuple[Position, Position, Position, Position]:
    x, y = pos
    return ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))


def in_bounds(pos: Position) -> bool:
    return 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT


def boundary_tiles(direction: str) -> set[Position]:
    if direction == "west":
        return {(0, row) for row in range(GRID_HEIGHT)}
    if direction == "east":
        return {(GRID_WIDTH - 1, row) for row in range(GRID_HEIGHT)}
    if direction == "north":
        return {(column, 0) for column in range(GRID_WIDTH)}
    return {(column, GRID_HEIGHT - 1) for column in range(GRID_WIDTH)}


def boundary_directions(pos: Position) -> tuple[str, ...]:
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
    return bool(boundary_directions(pos))


def next_position(pos: Position, action: int) -> Position:
    dx, dy = ACTION_TO_DELTA[action]
    return (pos[0] + dx, pos[1] + dy)


def action_toward(current: Position, nxt: Position) -> int | None:
    return DELTA_TO_ACTION.get((nxt[0] - current[0], nxt[1] - current[1]))


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


Policy = Task4Agent


def make_policy() -> Task4Agent:
    """为每个 episode 创建全新的观测驱动 Task4 策略。"""
    return Task4Agent()
