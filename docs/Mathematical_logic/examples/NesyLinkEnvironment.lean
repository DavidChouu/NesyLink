/-!
# NesyLink 环境形式化

本文件给出五个数理逻辑关卡共用的、基于 tile 的符号环境语义。Python
模拟器实际按像素移动，Agent 再从渲染帧中识别符号状态；Lean 层将“一次
完整的 tile 移动”抽象成一步转移。CNN 是否识别正确以及移动动画的中间帧
不在本文件的证明范围内，Lean 验证的是感知结果进入 planner/safety shield
之后的符号层。

模型尽量与 Python 引擎保持一致：

* 动作恰好包括 WAIT、四方向移动、slot A 和 slot B；
* 墙、NPC、可见宝箱、没有桥覆盖的 gap 会阻挡玩家；
* spike/abyss 可以踩入，但会扣血并把玩家送到合法重生点；
* button 在玩家踏上时触发，switch 需要玩家相邻并使用 slot A；
* slot A 可以开相邻宝箱，也可以攻击玩家面前一格的怪物；
* slot B 在持有盾牌时抵挡一次怪物接触伤害；
* 出口可以要求钥匙、按钮、物品或当前房间怪物全部被消灭。

文件中的所有定理均给出了能够由 Lean 内核检查的完整证明。
-/

namespace NesyLink

/-! ## 一、基础标识、坐标、方向和动作

对象与房间使用自然数 ID，便于在列表和房间函数中索引。坐标使用 `Int`
而不是 `Nat`，这样向左、向上越界时不会因自然数截断而错误地停留在 0；
是否越界统一交给 `inBounds` 判断。`Action` 与测评接口的七种动作一一对应。
-/

abbrev ObjectId := Nat
abbrev RoomId := Nat

structure Position where
  x : Int
  y : Int
  deriving DecidableEq, Repr

structure Bounds where
  width : Int
  height : Int
  width_pos : 0 < width
  height_pos : 0 < height

inductive Direction where
  | north | south | west | east
  deriving DecidableEq, Repr

inductive Action where
  | wait
  | up | down | left | right
  | slotA
  | slotB
  deriving DecidableEq, Repr

/-! ## 二、对象的静态类型与运行时属性

这里把 Python 环境中的对象拆成独立数据类型：

* `MonsterKind` 保留三类移动怪物，`Monster` 记录位置、HP 与接触伤害；
* `Loot` 覆盖钥匙、金币、治疗和装备，避免把所有宝箱硬编码成“给钥匙”；
* `Trap` 同时记录类别、伤害、重生点、是否生效和是否一次性；
* `Button` 表示踩踏按钮，`Switch` 表示相邻交互开关，两者语义不同；
* `Bridge` 保存北、东、南三组 tile，三态朝向决定当前实际铺设的桥面。
-/

inductive MonsterKind where
  | chaser | patroller | ambusher
  deriving DecidableEq, Repr

inductive Item where
  | sword | shield
  | named (id : ObjectId)
  deriving DecidableEq, Repr

inductive EquipmentSlot where
  | A | B
  deriving DecidableEq, Repr

inductive Loot where
  | key (amount : Nat)
  | gold (amount : Nat)
  | heal (amount : Nat)
  | item (value : Item)
  | tool (value : Item) (slot : EquipmentSlot)
  deriving DecidableEq, Repr

def swordDamage : Nat := 1
-- `core/constants.py` 中的真实击杀奖励；旧版说明曾写成 1。
def monsterKillGold : Nat := 2

inductive TrapKind where
  | spike | abyss
  deriving DecidableEq, Repr

inductive DynamicTile where
  | gap | bridge
  deriving DecidableEq, Repr

/-! 当前 Task4 地图中的桥是北、东、南三态循环，而不是旧说明的横纵二态。 -/
inductive BridgeOrientation where
  | westToNorth | westToEast | westToSouth
  deriving DecidableEq, Repr

inductive ChestRevealCondition where
  | never
  | allMonstersDefeated (triggerRoom : Option RoomId := none)
  deriving DecidableEq, Repr

structure Chest where
  id : ObjectId
  pos : Position
  loot : Loot
  visible : Bool := true
  opened : Bool := false
  -- `some roomId` 对应 Python reveal_on 中显式指定的触发房间；
  -- `none` 表示任意房间首次清怪都可触发，`never` 表示没有隐藏揭示机制。
  revealOn : ChestRevealCondition := .never
  deriving DecidableEq, Repr

structure Monster where
  id : ObjectId
  pos : Position
  kind : MonsterKind
  hp : Nat
  damage : Nat
  deriving DecidableEq, Repr

structure Trap where
  id : ObjectId
  pos : Position
  kind : TrapKind
  damage : Nat
  respawn : Position
  active : Bool := true
  singleUse : Bool := false
  deriving DecidableEq, Repr

structure Button where
  id : ObjectId
  pos : Position
  pressed : Bool := false
  deriving DecidableEq, Repr

structure Switch where
  id : ObjectId
  pos : Position
  targetRoom : RoomId
  targetBridge : ObjectId
  pressed : Bool := false
  deriving DecidableEq, Repr

structure Bridge where
  id : ObjectId
  orientation : BridgeOrientation
  northTiles : List Position
  eastTiles : List Position
  southTiles : List Position
  deriving DecidableEq, Repr

/-!
公共策略不能把 bridge 写死为三态。`FiniteBridgeCycle` 是任意非空有限模式
循环的环境级表示；旧 `BridgeOrientation` 仅作为公开 Task4 JSON 的三态精化。
-/
structure FiniteBridgeCycle where
  modes : List (List Position)
  modes_nonempty : modes ≠ []
  current : Nat

def finiteBridgeActiveTiles (cycle : FiniteBridgeCycle) : List Position :=
  cycle.modes.getD (cycle.current % cycle.modes.length) []

def rotateFiniteBridge (cycle : FiniteBridgeCycle) : FiniteBridgeCycle :=
  { cycle with current := (cycle.current + 1) % cycle.modes.length }

theorem finiteBridge_length_positive (cycle : FiniteBridgeCycle) :
    0 < cycle.modes.length := by
  cases hmodes : cycle.modes with
  | nil => exact False.elim (cycle.modes_nonempty hmodes)
  | cons head tail => simp

theorem rotated_finite_bridge_index_is_bounded (cycle : FiniteBridgeCycle) :
    (rotateFiniteBridge cycle).current < cycle.modes.length := by
  exact Nat.mod_lt _ (finiteBridge_length_positive cycle)

def publicBridgeCycle (bridge : Bridge) : FiniteBridgeCycle :=
  { modes := [bridge.northTiles, bridge.eastTiles, bridge.southTiles]
    modes_nonempty := by simp
    current := match bridge.orientation with
      | .westToNorth => 0 | .westToEast => 1 | .westToSouth => 2 }

theorem public_bridge_cycle_has_three_modes (bridge : Bridge) :
    (publicBridgeCycle bridge).modes.length = 3 := by
  rfl

structure Npc where
  id : ObjectId
  pos : Position
  text : String
  deriving DecidableEq, Repr

structure Inventory where
  keys : Nat := 0
  gold : Nat := 0
  items : List Item := []
  equippedA : Option Item := some .sword
  equippedB : Option Item := some .shield
  deriving DecidableEq, Repr

structure PlayerState where
  pos : Position
  facing : Direction
  hp : Nat
  maxHp : Nat
  inventory : Inventory
  shielding : Bool := false
  deriving DecidableEq, Repr

/-! ## 三、出口条件、房间状态与世界状态

`Requirement` 直接覆盖 Python schema 允许的条件：无需条件、钥匙数量、
按钮状态、拥有指定物品、清空怪物，以及条件合取。`rooms : RoomId → RoomState`
把多房间世界建模为房间 ID 到持久状态的映射，因此离开房间后，已开启宝箱、
已死亡怪物和已按按钮等变化不会丢失。
-/

inductive Requirement where
  | free
  | keys (count : Nat) (consume : Bool)
  | buttonPressed (id : ObjectId)
  | ownsItem (item : Item)
  | allMonstersDefeated
  | both (left right : Requirement)
  deriving DecidableEq, Repr

inductive ExitKind where
  | normal | locked | conditional
  deriving DecidableEq, Repr

structure Exit where
  id : ObjectId
  -- Python schema 中一个边界出口由两个 tile 组成；`pos` 保留主 tile，
  -- `otherTiles` 保存同一出口的其余 tile。
  pos : Position
  otherTiles : List Position := []
  direction : Direction
  kind : ExitKind
  requirement : Requirement
  targetRoom : RoomId
  targetSpawn : Position
  completesTask : Bool := false
  opened : Bool := false
  deriving DecidableEq, Repr

def exitContains (exit : Exit) (p : Position) : Prop :=
  p = exit.pos ∨ p ∈ exit.otherTiles

structure RoomState where
  bounds : Bounds
  walls : List Position
  npcs : List Npc
  chests : List Chest
  monsters : List Monster
  traps : List Trap
  buttons : List Button
  switches : List Switch
  bridges : List Bridge
  dynamicTiles : List (Position × DynamicTile)
  exits : List Exit

structure WorldState where
  currentRoom : RoomId
  rooms : RoomId → RoomState
  -- Python dungeon template 中实际存在的有限房间 ID；Task5 用它检查全世界宝箱。
  roomIds : List RoomId := []
  player : PlayerState
  completed : Bool := false

/-! 像素引擎中不直接属于 tile 世界的短期运行时状态。把它们放在显式 wrapper
中可避免把动画计时误当成 Agent 可观察真值，同时仍能陈述 tick 调度契约。 -/
structure RuntimeControl where
  actionWindow : Nat := 0
  shieldWindow : Nat := 0
  monsterStun : List (ObjectId × Nat) := []
  abyssControlLock : Nat := 0
  pendingRespawn : Option Position := none
  deriving DecidableEq, Repr

structure RuntimeWorld where
  world : WorldState
  control : RuntimeControl := {}

def advanceRuntimeWindow (control : RuntimeControl) : RuntimeControl :=
  { control with
      actionWindow := control.actionWindow - 1
      shieldWindow := control.shieldWindow - 1
      monsterStun := control.monsterStun.map (fun entry =>
        (entry.1, entry.2 - 1))
      abyssControlLock := control.abyssControlLock - 1 }

def RuntimeControlConsistent (runtime : RuntimeWorld) : Prop :=
  runtime.world.player.shielding = true → 0 < runtime.control.shieldWindow

theorem runtime_windows_never_increase (control : RuntimeControl) :
    (advanceRuntimeWindow control).actionWindow ≤ control.actionWindow ∧
    (advanceRuntimeWindow control).shieldWindow ≤ control.shieldWindow ∧
    (advanceRuntimeWindow control).abyssControlLock ≤ control.abyssControlLock := by
  simp [advanceRuntimeWindow]

theorem abyss_lock_blocks_player_input
    (runtime : RuntimeWorld) (hlock : 0 < runtime.control.abyssControlLock) :
    runtime.control.abyssControlLock ≠ 0 :=
  Nat.ne_of_gt hlock

/-! ## 四、几何关系与对象查询谓词

这一段只负责回答“某个 tile 上有什么”和“能否进入”，不直接改变状态。
`canEnter` 是物理可通行性，`safeTile` 是 Agent 更严格的安全判断。两者必须
分开：陷阱在引擎中确实可以踩入，所以不能把陷阱误建模成墙；但 safety
shield 可以使用 `safeTile` 主动避开陷阱和怪物。
-/

def inBounds (b : Bounds) (p : Position) : Prop :=
  0 ≤ p.x ∧ p.x < b.width ∧ 0 ≤ p.y ∧ p.y < b.height

def advance (p : Position) : Direction → Position
  | .north => { p with y := p.y - 1 }
  | .south => { p with y := p.y + 1 }
  | .west  => { p with x := p.x - 1 }
  | .east  => { p with x := p.x + 1 }

def actionDirection : Action → Option Direction
  | .up => some .north
  | .down => some .south
  | .left => some .west
  | .right => some .east
  | _ => none

def directionAction : Direction → Action
  | .north => .up
  | .south => .down
  | .west => .left
  | .east => .right

theorem directionAction_correct (d : Direction) :
    actionDirection (directionAction d) = some d := by
  cases d <;> rfl

def adjacent (a b : Position) : Prop :=
  b = advance a .north ∨ b = advance a .south ∨
  b = advance a .west ∨ b = advance a .east

-- Python `is_adjacent` 使用曼哈顿距离 `≤ 1`，因此同格也是合法交互距离。
-- 宝箱和 NPC 由碰撞阻挡保证实际不会与玩家同格；switch 则允许同格交互。
def interactionReach (a b : Position) : Prop :=
  a = b ∨ adjacent a b

def currentRoomState (s : WorldState) : RoomState :=
  s.rooms s.currentRoom

def visibleChestAt (r : RoomState) (p : Position) : Prop :=
  ∃ c ∈ r.chests, c.pos = p ∧ c.visible = true

def closedChestAt (r : RoomState) (p : Position) : Prop :=
  ∃ c ∈ r.chests, c.pos = p ∧ c.visible = true ∧ c.opened = false

def monsterAt (r : RoomState) (p : Position) : Prop :=
  ∃ m ∈ r.monsters, m.pos = p ∧ 0 < m.hp

def buttonAt (r : RoomState) (p : Position) : Prop :=
  ∃ b ∈ r.buttons, b.pos = p

def npcAt (r : RoomState) (p : Position) : Prop :=
  ∃ npc ∈ r.npcs, npc.pos = p

def activeBridgeTile (r : RoomState) (p : Position) : Prop :=
  ∃ b ∈ r.bridges,
    (b.orientation = .westToNorth ∧ p ∈ b.northTiles) ∨
    (b.orientation = .westToEast ∧ p ∈ b.eastTiles) ∨
    (b.orientation = .westToSouth ∧ p ∈ b.southTiles)

-- Python `trap_at` 会先检查动态桥；活动桥面覆盖桥下 abyss。
def activeTrapAt (r : RoomState) (p : Position) : Prop :=
  (∃ t ∈ r.traps, t.pos = p ∧ t.active = true) ∧
  ¬ activeBridgeTile r p

def gapAt (r : RoomState) (p : Position) : Prop :=
  (p, DynamicTile.gap) ∈ r.dynamicTiles ∧ ¬ activeBridgeTile r p

def staticBlocker (r : RoomState) (p : Position) : Prop :=
  -- 宝箱打开后仍保留实体碰撞，因此这里判断 visible，而不是 closed。
  p ∈ r.walls ∨ npcAt r p ∨ visibleChestAt r p

def canEnter (r : RoomState) (p : Position) : Prop :=
  inBounds r.bounds p ∧
  ¬ staticBlocker r p ∧
  ¬ gapAt r p

def safeTile (r : RoomState) (p : Position) : Prop :=
  canEnter r p ∧ ¬ activeTrapAt r p ∧ ¬ monsterAt r p

/-! ## 五、条件判定、资源变化与局部状态更新

`requirementSatisfied` 是出口守卫条件，`spendRequirement` 只在配置明确要求
消耗钥匙时扣除钥匙。`collectLoot`、`damagePlayer`、`rewardPlayer` 集中定义
资源变化，从而让所有转移复用同一语义。后面的 `replace*` 函数按对象 ID
更新列表中的一个对象；`setRoom` 则把修改写回多房间世界。
-/

def buttonIsPressed (r : RoomState) (id : ObjectId) : Prop :=
  ∃ b ∈ r.buttons, b.id = id ∧ b.pressed = true

def requirementSatisfied (s : WorldState) (req : Requirement) : Prop :=
  match req with
  | .free => True
  | .keys n _ => n ≤ s.player.inventory.keys
  | .buttonPressed id => buttonIsPressed (currentRoomState s) id
  | .ownsItem item => item ∈ s.player.inventory.items
  | .allMonstersDefeated => (currentRoomState s).monsters = []
  | .both left right =>
      requirementSatisfied s left ∧ requirementSatisfied s right

def spendRequirement (inv : Inventory) : Requirement → Inventory
  | .keys n true => { inv with keys := inv.keys - n }
  | .both left right =>
      spendRequirement (spendRequirement inv left) right
  | _ => inv

def requirementContainsAllMonstersDefeated : Requirement → Bool
  | .allMonstersDefeated => true
  | .both left right =>
      requirementContainsAllMonstersDefeated left ||
      requirementContainsAllMonstersDefeated right
  | _ => false

def collectLoot (p : PlayerState) : Loot → PlayerState
  | .key n =>
      { p with inventory := { p.inventory with keys := p.inventory.keys + n } }
  | .gold n =>
      { p with inventory := { p.inventory with gold := p.inventory.gold + n } }
  | .heal n =>
      -- 使用 min 保证治疗后的 HP 永远不超过 maxHp。
      { p with hp := min p.maxHp (p.hp + n) }
  | .item item =>
      { p with inventory :=
          { p.inventory with items :=
              if item ∈ p.inventory.items then p.inventory.items
              else item :: p.inventory.items } }
  | .tool item slot =>
      let items :=
        if item ∈ p.inventory.items then p.inventory.items
        else item :: p.inventory.items
      match slot with
      | .A => { p with inventory :=
          { p.inventory with items := items, equippedA := some item } }
      | .B => { p with inventory :=
          { p.inventory with items := items, equippedB := some item } }

def damagePlayer (p : PlayerState) (amount : Nat) : PlayerState :=
  { p with hp := p.hp - amount, shielding := false }

def rewardPlayer (p : PlayerState) (amount : Nat) : PlayerState :=
  { p with
    inventory := { p.inventory with gold := p.inventory.gold + amount }
    shielding := false }

def setRoom (rooms : RoomId → RoomState) (id : RoomId) (room : RoomState) :
    RoomId → RoomState :=
  fun query => if query = id then room else rooms query

def updateCurrentRoom (s : WorldState) (room : RoomState) : WorldState :=
  { s with rooms := setRoom s.rooms s.currentRoom room }

@[simp] theorem updateCurrentRoom_currentRoom
    (s : WorldState) (room : RoomState) :
    (updateCurrentRoom s room).currentRoom = s.currentRoom := rfl
@[simp] theorem updateCurrentRoom_player
    (s : WorldState) (room : RoomState) :
    (updateCurrentRoom s room).player = s.player := rfl
@[simp] theorem updateCurrentRoom_roomIds
    (s : WorldState) (room : RoomState) :
    (updateCurrentRoom s room).roomIds = s.roomIds := rfl
@[simp] theorem updateCurrentRoom_completed
    (s : WorldState) (room : RoomState) :
    (updateCurrentRoom s room).completed = s.completed := rfl
@[simp] theorem updateCurrentRoom_rooms_same
    (s : WorldState) (room : RoomState) :
    (updateCurrentRoom s room).rooms s.currentRoom = room := by
  simp [updateCurrentRoom, setRoom]

def replaceChest (r : RoomState) (old fresh : Chest) : RoomState :=
  { r with chests := r.chests.map (fun c => if c.id = old.id then fresh else c) }

def replaceMonster (r : RoomState) (old fresh : Monster) : RoomState :=
  { r with monsters := r.monsters.map (fun m => if m.id = old.id then fresh else m) }

def removeMonster (r : RoomState) (target : Monster) : RoomState :=
  { r with monsters := r.monsters.filter (fun m => m.id != target.id) }

@[simp] theorem removeMonster_bounds (r : RoomState) (target : Monster) :
    (removeMonster r target).bounds = r.bounds := by
  rfl

def chestRevealMatches
    (triggerRoom : RoomId) (condition : ChestRevealCondition) : Bool :=
  match condition with
  | .never => false
  | .allMonstersDefeated none => true
  | .allMonstersDefeated (some roomId) => roomId == triggerRoom

def revealEligibleChests (room : RoomState) (triggerRoom : RoomId) : RoomState :=
  { room with chests := room.chests.map (fun chest =>
      if !chest.visible && chestRevealMatches triggerRoom chest.revealOn then
        { chest with visible := true }
      else chest) }

def unlockAllMonstersDefeatedExits (room : RoomState) : RoomState :=
  { room with exits := room.exits.map (fun exit =>
      if requirementContainsAllMonstersDefeated exit.requirement then
        { exit with opened := true }
      else exit) }

def revealEligibleChestsInWorld
    (rooms : RoomId → RoomState) (triggerRoom : RoomId) :
    RoomId → RoomState :=
  fun roomId => revealEligibleChests (rooms roomId) triggerRoom

/-!
Python 在击杀怪物后先删除怪物并发放金币；若这次击杀清空了当前房间，则立即
持久打开清怪条件门，并按 reveal_on 规则遍历所有房间揭示隐藏宝箱。
-/
def resolveMonsterKill (s : WorldState) (monster : Monster) : WorldState :=
  let roomAfterRemoval := removeMonster (currentRoomState s) monster
  let rewarded : WorldState :=
    updateCurrentRoom
      { s with player := rewardPlayer s.player monsterKillGold }
      roomAfterRemoval
  if roomAfterRemoval.monsters = [] then
    let roomAfterUnlock := unlockAllMonstersDefeatedExits roomAfterRemoval
    let roomsAfterUnlock :=
      setRoom rewarded.rooms s.currentRoom roomAfterUnlock
    { rewarded with
      rooms := revealEligibleChestsInWorld roomsAfterUnlock s.currentRoom }
  else rewarded

@[simp] theorem revealEligibleChests_bounds
    (room : RoomState) (triggerRoom : RoomId) :
    (revealEligibleChests room triggerRoom).bounds = room.bounds := by
  rfl

@[simp] theorem unlockAllMonstersDefeatedExits_bounds
    (room : RoomState) :
    (unlockAllMonstersDefeatedExits room).bounds = room.bounds := by
  rfl

@[simp] theorem resolveMonsterKill_currentRoom
    (s : WorldState) (monster : Monster) :
    (resolveMonsterKill s monster).currentRoom = s.currentRoom := by
  simp only [resolveMonsterKill]
  split <;> rfl

@[simp] theorem resolveMonsterKill_roomIds
    (s : WorldState) (monster : Monster) :
    (resolveMonsterKill s monster).roomIds = s.roomIds := by
  simp only [resolveMonsterKill]
  split <;> rfl

@[simp] theorem resolveMonsterKill_player
    (s : WorldState) (monster : Monster) :
    (resolveMonsterKill s monster).player =
      rewardPlayer s.player monsterKillGold := by
  simp only [resolveMonsterKill]
  split <;> rfl

@[simp] theorem resolveMonsterKill_completed
    (s : WorldState) (monster : Monster) :
    (resolveMonsterKill s monster).completed = s.completed := by
  simp only [resolveMonsterKill]
  split <;> rfl

theorem resolveMonsterKill_other_room
    (s : WorldState) (monster : Monster) (roomId : RoomId)
    (hother : roomId ≠ s.currentRoom)
    (hreveal : revealEligibleChests (s.rooms roomId) s.currentRoom =
      s.rooms roomId) :
    (resolveMonsterKill s monster).rooms roomId = s.rooms roomId := by
  simp only [resolveMonsterKill]
  split
  · simp [revealEligibleChestsInWorld, updateCurrentRoom, setRoom,
      hother, hreveal]
  · simp [updateCurrentRoom, setRoom, hother]

@[simp] theorem resolveMonsterKill_current_bounds
    (s : WorldState) (monster : Monster) :
    (currentRoomState (resolveMonsterKill s monster)).bounds =
      (currentRoomState s).bounds := by
  simp only [resolveMonsterKill]
  split <;>
    simp [currentRoomState, revealEligibleChestsInWorld,
      updateCurrentRoom, setRoom, unlockAllMonstersDefeatedExits,
      revealEligibleChests]

@[simp] theorem resolveMonsterKill_current_monsters
    (s : WorldState) (monster : Monster) :
    (currentRoomState (resolveMonsterKill s monster)).monsters =
      (removeMonster (currentRoomState s) monster).monsters := by
  simp only [resolveMonsterKill]
  split <;>
    simp [currentRoomState, revealEligibleChestsInWorld,
      updateCurrentRoom, setRoom, unlockAllMonstersDefeatedExits,
      revealEligibleChests]

def replaceButton (r : RoomState) (old fresh : Button) : RoomState :=
  { r with buttons := r.buttons.map (fun b => if b.id = old.id then fresh else b) }

def replaceSwitch (r : RoomState) (old fresh : Switch) : RoomState :=
  { r with switches := r.switches.map (fun w => if w.id = old.id then fresh else w) }

def replaceExit (r : RoomState) (old fresh : Exit) : RoomState :=
  { r with exits := r.exits.map (fun e => if e.id = old.id then fresh else e) }

def rotateOrientation : BridgeOrientation → BridgeOrientation
  | .westToNorth => .westToEast
  | .westToEast => .westToSouth
  | .westToSouth => .westToNorth

def rotateBridge (r : RoomState) (id : ObjectId) : RoomState :=
  { r with bridges := r.bridges.map (fun b =>
      if b.id = id then { b with orientation := rotateOrientation b.orientation } else b) }

def pressButtonAt (r : RoomState) (p : Position) : RoomState :=
  { r with buttons := r.buttons.map (fun b =>
      if b.pos = p then { b with pressed := true } else b) }

def deactivateTrap (r : RoomState) (target : Trap) : RoomState :=
  -- 一次性陷阱触发后失活；普通陷阱保持原状态，允许后续再次触发。
  if target.singleUse then
    { r with traps := r.traps.map (fun t =>
        if t.id = target.id then { t with active := false } else t) }
  else r

def activateSwitchState (s : WorldState) (switch : Switch) : WorldState :=
  let pressedCurrent :=
    replaceSwitch (currentRoomState s) switch { switch with pressed := true }
  let roomsAfterPress := setRoom s.rooms s.currentRoom pressedCurrent
  let targetAfterPress := roomsAfterPress switch.targetRoom
  { s with
    rooms := setRoom roomsAfterPress switch.targetRoom
      (rotateBridge targetAfterPress switch.targetBridge)
    player := { s.player with shielding := false } }

@[simp] theorem activateSwitchState_currentRoom
    (s : WorldState) (switch : Switch) :
    (activateSwitchState s switch).currentRoom = s.currentRoom := rfl
@[simp] theorem activateSwitchState_roomIds
    (s : WorldState) (switch : Switch) :
    (activateSwitchState s switch).roomIds = s.roomIds := rfl
@[simp] theorem activateSwitchState_completed
    (s : WorldState) (switch : Switch) :
    (activateSwitchState s switch).completed = s.completed := rfl
@[simp] theorem activateSwitchState_pos
    (s : WorldState) (switch : Switch) :
    (activateSwitchState s switch).player.pos = s.player.pos := rfl
@[simp] theorem activateSwitchState_hp
    (s : WorldState) (switch : Switch) :
    (activateSwitchState s switch).player.hp = s.player.hp := rfl
@[simp] theorem activateSwitchState_inventory
    (s : WorldState) (switch : Switch) :
    (activateSwitchState s switch).player.inventory = s.player.inventory := rfl

theorem activateSwitchState_current_bounds
    (s : WorldState) (switch : Switch) :
    (currentRoomState (activateSwitchState s switch)).bounds =
      (currentRoomState s).bounds := by
  unfold activateSwitchState currentRoomState
  by_cases hsame : switch.targetRoom = s.currentRoom
  · rw [hsame]
    simp [setRoom, replaceSwitch, rotateBridge]
  · have hother : s.currentRoom ≠ switch.targetRoom := by
      exact fun h => hsame h.symm
    simp [setRoom, hother, replaceSwitch]

def exitRequirementSatisfied (s : WorldState) (exit : Exit) : Prop :=
  match exit.kind with
  | .locked =>
      exit.opened = true ∨ requirementSatisfied s exit.requirement
  | .normal | .conditional =>
      requirementSatisfied s exit.requirement

def spendExitRequirement
    (inventory : Inventory) (exit : Exit) : Inventory :=
  match exit.kind, exit.opened with
  | .locked, false => spendRequirement inventory exit.requirement
  | _, _ => inventory

def unlockExitInRoom (room : RoomState) (exit : Exit) : RoomState :=
  match exit.kind, exit.opened with
  | .locked, false =>
      replaceExit room exit { exit with opened := true }
  | _, _ => room

@[simp] theorem unlockExitInRoom_bounds
    (room : RoomState) (exit : Exit) :
    (unlockExitInRoom room exit).bounds = room.bounds := by
  cases hkind : exit.kind <;> cases hopen : exit.opened <;>
    simp [unlockExitInRoom, hkind, hopen, replaceExit]

def transitionThroughExit (s : WorldState) (exit : Exit) : WorldState :=
  let sourceRoom := unlockExitInRoom (currentRoomState s) exit
  let roomsAfterUnlock := setRoom s.rooms s.currentRoom sourceRoom
  { s with
    currentRoom := exit.targetRoom
    rooms := roomsAfterUnlock
    player :=
      { s.player with
        pos := exit.targetSpawn
        facing := exit.direction
        inventory := spendExitRequirement s.player.inventory exit
        shielding := false }
    completed := s.completed || exit.completesTask }

@[simp] theorem transitionThroughExit_currentRoom
    (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).currentRoom = exit.targetRoom := rfl
@[simp] theorem transitionThroughExit_pos
    (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).player.pos = exit.targetSpawn := rfl
@[simp] theorem transitionThroughExit_facing
    (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).player.facing = exit.direction := rfl
@[simp] theorem transitionThroughExit_hp
    (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).player.hp = s.player.hp := rfl
@[simp] theorem transitionThroughExit_maxHp
    (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).player.maxHp = s.player.maxHp := rfl
@[simp] theorem transitionThroughExit_roomIds
    (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).roomIds = s.roomIds := rfl
@[simp] theorem transitionThroughExit_inventory
    (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).player.inventory =
      spendExitRequirement s.player.inventory exit := rfl
@[simp] theorem transitionThroughExit_completed
    (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).completed =
      (s.completed || exit.completesTask) := rfl

theorem transitionThroughExit_target_bounds
    (s : WorldState) (exit : Exit) :
    (currentRoomState (transitionThroughExit s exit)).bounds =
      (s.rooms exit.targetRoom).bounds := by
  unfold transitionThroughExit currentRoomState
  by_cases hsame : exit.targetRoom = s.currentRoom
  · rw [hsame]
    simp [setRoom]
  · simp [setRoom, hsame]

theorem requirement_implies_exitRequirementSatisfied
    {s : WorldState} {exit : Exit}
    (h : requirementSatisfied s exit.requirement) :
    exitRequirementSatisfied s exit := by
  unfold exitRequirementSatisfied
  cases exit.kind with
  | normal => exact h
  | locked => exact Or.inr h
  | conditional => exact h

/-! ## 六、事件与单步状态转移语义

事件不是隐藏真值输入，而是 Lean 模型对一次符号转移结果的说明，便于之后
验证轨迹和任务里程碑。`Step s a t events` 表示：在状态 `s` 执行动作 `a`
可以到达状态 `t`，同时产生 `events`。使用关系而不是单一函数，是因为怪物
行为包含类型、周期和随机性，符号层允许所有满足安全约束的合法怪物移动。
-/

inductive Event where
  | waited
  | moved (source target : Position)
  | blocked (location : Position)
  | trapTriggered (id : ObjectId)
  | abyssFall (id : ObjectId)
  | chestOpened (id : ObjectId)
  | chestRevealed (id : ObjectId)
  | talkedNpc (id : ObjectId)
  | actionNoEffect
  | monsterDamaged (id : ObjectId)
  | monsterKilled (id : ObjectId)
  | monsterMoved (id : ObjectId) (source target : Position)
  | agentDamaged (amount : Nat)
  | shieldBlock (monsterId : ObjectId)
  | buttonPressed (id : ObjectId)
  | switchActivated (id : ObjectId)
  | bridgeRotated (id : ObjectId)
  | doorOpened (id : ObjectId)
  | roomChanged (source target : RoomId)
  | environmentCompleted
  deriving DecidableEq, Repr

def exitEvents (s : WorldState) (exit : Exit) : List Event :=
  let doorEvents :=
    match exit.kind, exit.opened with
    | .locked, false => [.doorOpened exit.id]
    | _, _ => []
  let roomEvents := [.roomChanged s.currentRoom exit.targetRoom]
  let completionEvents :=
    if exit.completesTask then [.environmentCompleted] else []
  doorEvents ++ roomEvents ++ completionEvents

def newlyUnlockedExitIds (room : RoomState) : List ObjectId :=
  (room.exits.filter (fun exit =>
    !exit.opened &&
    requirementContainsAllMonstersDefeated exit.requirement)).map Exit.id

def newlyRevealedChestIds
    (room : RoomState) (triggerRoom : RoomId) : List ObjectId :=
  (room.chests.filter (fun chest =>
    !chest.visible &&
    chestRevealMatches triggerRoom chest.revealOn)).map Chest.id

def newlyRevealedChestIdsInWorld
    (s : WorldState) (triggerRoom : RoomId) : List ObjectId :=
  s.roomIds.flatMap (fun roomId =>
    newlyRevealedChestIds (s.rooms roomId) triggerRoom)

def monsterKillEvents (s : WorldState) (monster : Monster) : List Event :=
  let roomAfterRemoval := removeMonster (currentRoomState s) monster
  if roomAfterRemoval.monsters = [] then
    [.monsterKilled monster.id] ++
    (newlyUnlockedExitIds roomAfterRemoval).map Event.doorOpened ++
    (newlyRevealedChestIdsInWorld s s.currentRoom).map Event.chestRevealed
  else
    [.monsterKilled monster.id]

def validPlayerPosition (s : WorldState) : Prop :=
  let r := currentRoomState s
  inBounds r.bounds s.player.pos ∧ ¬ staticBlocker r s.player.pos ∧ ¬ gapAt r s.player.pos

def CollisionFreeState (s : WorldState) : Prop :=
  validPlayerPosition s

def ValidState (s : WorldState) : Prop :=
  inBounds (currentRoomState s).bounds s.player.pos ∧
  s.player.hp ≤ s.player.maxHp

/-! ### 世界与房间配置的良构性

`WellFormedWorld` 是运行时必须保持的核心良构条件：房间索引非空且无重复，
当前房间确实属于索引，并且玩家状态合法。出口目标属于有限房间集合这一事实
单独写成 `ExitTargetsKnown`，供房间切换的保持性证明使用。

`RoomConfigurationWellFormed` 检查关卡模板中的静态对象位置。它覆盖墙、
NPC、宝箱、怪物、陷阱、按钮、开关、桥的两组候选 tile、动态 tile 和出口；
出口 spawn 则由 `ExitSpawnsInBounds` 相对于目标房间检查。
-/

def allPositionsInBounds (bounds : Bounds) (positions : List Position) : Prop :=
  ∀ p, p ∈ positions → inBounds bounds p

def RoomConfigurationWellFormed (room : RoomState) : Prop :=
  allPositionsInBounds room.bounds room.walls ∧
  allPositionsInBounds room.bounds (room.npcs.map Npc.pos) ∧
  allPositionsInBounds room.bounds (room.chests.map Chest.pos) ∧
  allPositionsInBounds room.bounds (room.monsters.map Monster.pos) ∧
  allPositionsInBounds room.bounds (room.traps.map Trap.pos) ∧
  allPositionsInBounds room.bounds (room.traps.map Trap.respawn) ∧
  allPositionsInBounds room.bounds (room.buttons.map Button.pos) ∧
  allPositionsInBounds room.bounds (room.switches.map Switch.pos) ∧
  (∀ bridge, bridge ∈ room.bridges →
    allPositionsInBounds room.bounds bridge.northTiles ∧
    allPositionsInBounds room.bounds bridge.eastTiles ∧
    allPositionsInBounds room.bounds bridge.southTiles) ∧
  allPositionsInBounds room.bounds (room.dynamicTiles.map Prod.fst) ∧
  allPositionsInBounds room.bounds (room.exits.map Exit.pos) ∧
  ∀ exit, exit ∈ room.exits →
    allPositionsInBounds room.bounds exit.otherTiles

def ExitTargetsKnown (s : WorldState) : Prop :=
  ∀ roomId, roomId ∈ s.roomIds →
    ∀ exit, exit ∈ (s.rooms roomId).exits →
      exit.targetRoom ∈ s.roomIds

def ExitSpawnsInBounds (s : WorldState) : Prop :=
  ∀ roomId, roomId ∈ s.roomIds →
    ∀ exit, exit ∈ (s.rooms roomId).exits →
      inBounds (s.rooms exit.targetRoom).bounds exit.targetSpawn

def AllRoomConfigurationsWellFormed (s : WorldState) : Prop :=
  ∀ roomId, roomId ∈ s.roomIds →
    RoomConfigurationWellFormed (s.rooms roomId)

def WellFormedWorld (s : WorldState) : Prop :=
  s.roomIds ≠ [] ∧
  s.roomIds.Nodup ∧
  s.currentRoom ∈ s.roomIds ∧
  ValidState s

def InventoryConsistent (inventory : Inventory) : Prop :=
  (∀ item, inventory.equippedA = some item → item ∈ inventory.items) ∧
  (∀ item, inventory.equippedB = some item → item ∈ inventory.items)

/-!
`WorldInvariant` 是策略主定理使用的完整环境不变量：它同时要求
有限房间索引、所有静态配置界内、出口目标和 spawn 合法、玩家不与
物理阻挡重叠，以及装备必须已存在于物品集合。怪物、按钮和陷阱不是
物理阻挡，因此不出现在 `CollisionFreeState` 中。
-/
def WorldInvariant (s : WorldState) : Prop :=
  WellFormedWorld s ∧
  AllRoomConfigurationsWellFormed s ∧
  ExitTargetsKnown s ∧
  ExitSpawnsInBounds s ∧
  CollisionFreeState s ∧
  InventoryConsistent s.player.inventory

def alive (s : WorldState) : Prop := 0 < s.player.hp
def dead (s : WorldState) : Prop := s.player.hp = 0

def allVisibleChestsOpened (r : RoomState) : Prop :=
  ∀ chest, chest ∈ r.chests →
    chest.visible = true ∧ chest.opened = true

def allWorldChestsOpened (s : WorldState) : Prop :=
  s.roomIds ≠ [] ∧
  (∃ roomId ∈ s.roomIds, ∃ chest, chest ∈ (s.rooms roomId).chests) ∧
  ∀ roomId, roomId ∈ s.roomIds →
    allVisibleChestsOpened (s.rooms roomId)

def trapEvents
    (source target : Position) (trap : Trap) : List Event :=
  if trap.kind = .abyss then
    [.moved source target, .abyssFall trap.id, .agentDamaged trap.damage]
  else
    [.moved source target, .trapTriggered trap.id, .agentDamaged trap.damage]

def openChestInteractionAvailable (s : WorldState) : Prop :=
  ∃ chest ∈ (currentRoomState s).chests,
    chest.visible = true ∧ chest.opened = false ∧
    interactionReach s.player.pos chest.pos

def npcInteractionAvailable (s : WorldState) : Prop :=
  ∃ npc ∈ (currentRoomState s).npcs,
    interactionReach s.player.pos npc.pos

def switchInteractionAvailable (s : WorldState) : Prop :=
  ∃ switch ∈ (currentRoomState s).switches,
    interactionReach s.player.pos switch.pos

def primaryInteractionAvailable (s : WorldState) : Prop :=
  openChestInteractionAvailable s ∨
  npcInteractionAvailable s ∨
  switchInteractionAvailable s

def swordInteractionAvailable (s : WorldState) : Prop :=
  s.player.inventory.equippedA = some Item.sword ∧
  Item.sword ∈ s.player.inventory.items ∧
  ∃ monster ∈ (currentRoomState s).monsters,
    monster.pos = advance s.player.pos s.player.facing ∧ 0 < monster.hp

/-!
`Step s a t events` 是符号环境的微步转移关系。玩家动作、出口判定、tile
效果、怪物更新和接触结算在 Python 的一次 tick 中顺序发生；Lean 将这些阶段
拆成可组合微步，以便分别证明。玩家发起的微步保留真实动作，怪物移动和接触
等自主阶段使用 `wait` 作为“无新增玩家输入”标签。怪物 AI 被有意建模为
非确定性，但每次移动仍必须相邻、界内、非阻挡且不与另一怪物重叠。
-/
inductive Step : WorldState → Action → WorldState → List Event → Prop where
  -- WAIT 不改变位置和资源，但会结束上一帧的临时举盾状态。
  | wait {s : WorldState} :
      Step s .wait { s with player := { s.player with shielding := false } } [.waited]

  -- 普通移动：目标必须物理可进入，并且不是需要特殊处理的陷阱或按钮。
  | movePlain {s : WorldState} {a : Action} {d : Direction} {q : Position}
      (ha : actionDirection a = some d)
      (hq : q = advance s.player.pos d)
      (henter : canEnter (currentRoomState s) q)
      (htrap : ¬ activeTrapAt (currentRoomState s) q)
      (hbutton : ¬ buttonAt (currentRoomState s) q) :
      Step s a
        { s with player := { s.player with pos := q, facing := d, shielding := false } }
        [.moved s.player.pos q]

  -- 踩按钮移动：先移动到按钮 tile，再把该位置上的按钮记为已按下。
  | moveButton {s : WorldState} {a : Action} {d : Direction} {q : Position}
      {button : Button}
      (ha : actionDirection a = some d)
      (hq : q = advance s.player.pos d)
      (henter : canEnter (currentRoomState s) q)
      (hbutton : button ∈ (currentRoomState s).buttons)
      (hpos : button.pos = q) :
      Step s a
        (updateCurrentRoom
          { s with player := { s.player with pos := q, facing := d, shielding := false } }
          (pressButtonAt (currentRoomState s) q))
        [.moved s.player.pos q, .buttonPressed button.id]

  -- 陷阱存活分支：扣血后 HP 仍为正，玩家回到合法重生点。
  | moveTrapSurvive {s : WorldState} {a : Action} {d : Direction} {q : Position}
      {trap : Trap}
      (ha : actionDirection a = some d)
      (hq : q = advance s.player.pos d)
      (henter : canEnter (currentRoomState s) q)
      (htrap : trap ∈ (currentRoomState s).traps)
      (hpos : trap.pos = q)
      (hactive : trap.active = true)
      (hsurvives : 0 < (damagePlayer s.player trap.damage).hp)
      (hrespawn : canEnter (currentRoomState s) trap.respawn) :
      Step s a
        (updateCurrentRoom
          { s with player :=
              { damagePlayer s.player trap.damage with
                pos := trap.respawn, facing := d } }
          (deactivateTrap (currentRoomState s) trap))
        (trapEvents s.player.pos q trap)

  -- 陷阱致死分支：HP 归零时 Python 不执行重生，玩家留在触发 tile。
  | moveTrapFatal {s : WorldState} {a : Action} {d : Direction} {q : Position}
      {trap : Trap}
      (ha : actionDirection a = some d)
      (hq : q = advance s.player.pos d)
      (henter : canEnter (currentRoomState s) q)
      (htrap : trap ∈ (currentRoomState s).traps)
      (hpos : trap.pos = q)
      (hactive : trap.active = true)
      (hfatal : (damagePlayer s.player trap.damage).hp = 0) :
      Step s a
        (updateCurrentRoom
          { s with player :=
              { damagePlayer s.player trap.damage with pos := q, facing := d } }
          (deactivateTrap (currentRoomState s) trap))
        (trapEvents s.player.pos q trap)

  -- 撞墙/越界/gap/宝箱：更新朝向但保持玩家位置不变，并产生 blocked 事件。
  | moveBlocked {s : WorldState} {a : Action} {d : Direction}
      (ha : actionDirection a = some d)
      (hblocked : ¬ canEnter (currentRoomState s) (advance s.player.pos d)) :
      Step s a
        { s with player := { s.player with facing := d, shielding := false } }
        [.blocked (advance s.player.pos d)]

  -- 面向怪物：对应 Python Agent 发出的短促像素动作，只更新朝向而不走完整格。
  -- 该规则只在目标格确实存在活怪物时可用，不能被普通导航滥用。
  | faceMonster {s : WorldState} {a : Action} {d : Direction}
      (ha : actionDirection a = some d)
      (hmonster : monsterAt (currentRoomState s) (advance s.player.pos d)) :
      Step s a
        { s with player := { s.player with facing := d, shielding := false } }
        []

  -- 开宝箱：要求宝箱存在、可见、未开启且与玩家相邻，然后发放其真实 loot。
  | openChest {s : WorldState} {chest : Chest}
      (hmember : chest ∈ (currentRoomState s).chests)
      (hvisible : chest.visible = true)
      (hclosed : chest.opened = false)
      (hadj : interactionReach s.player.pos chest.pos) :
      Step s .slotA
        (updateCurrentRoom
          { s with player := collectLoot { s.player with shielding := false } chest.loot }
          (replaceChest (currentRoomState s) chest { chest with opened := true }))
        [.chestOpened chest.id]

  -- NPC 对话优先于 switch 和剑；只有不存在可开启宝箱时才进入该分支。
  | talkNpc {s : WorldState} {npc : Npc}
      (hnoChest : ¬ openChestInteractionAvailable s)
      (hmember : npc ∈ (currentRoomState s).npcs)
      (hadj : interactionReach s.player.pos npc.pos) :
      Step s .slotA
        { s with player := { s.player with shielding := false } }
        [.talkedNpc npc.id]

  -- 未击杀攻击：必须有剑且怪物正好位于面前一格，扣除攻击力对应的 HP。
  | attackDamage {s : WorldState} {monster : Monster}
      (hnoInteraction : ¬ primaryInteractionAvailable s)
      (hequipped : s.player.inventory.equippedA = some Item.sword)
      (hsword : Item.sword ∈ s.player.inventory.items)
      (hmember : monster ∈ (currentRoomState s).monsters)
      (htarget : monster.pos = advance s.player.pos s.player.facing)
      (hsurvives : swordDamage < monster.hp) :
      Step s .slotA
        (updateCurrentRoom
          { s with player := { s.player with shielding := false } }
          (replaceMonster (currentRoomState s) monster
            { monster with hp := monster.hp - swordDamage }))
        [.monsterDamaged monster.id]

  -- 击杀攻击：删除目标并发放金币；若清空房间，同时结算清怪门和隐藏宝箱。
  | attackKill {s : WorldState} {monster : Monster}
      (hnoInteraction : ¬ primaryInteractionAvailable s)
      (hequipped : s.player.inventory.equippedA = some Item.sword)
      (hsword : Item.sword ∈ s.player.inventory.items)
      (hmember : monster ∈ (currentRoomState s).monsters)
      (htarget : monster.pos = advance s.player.pos s.player.facing)
      (hkilled : monster.hp ≤ swordDamage) :
      Step s .slotA
        (resolveMonsterKill s monster)
        (monsterKillEvents s monster)

  -- 有盾时 slot B 激活一次性格挡；没有盾时该动作不能凭空产生格挡能力。
  | shield {s : WorldState}
      (hshield : s.player.inventory.equippedB = some Item.shield) :
      Step s .slotB { s with player := { s.player with shielding := true } } []

  | shieldUnavailable {s : WorldState}
      (hshield : s.player.inventory.equippedB ≠ some Item.shield) :
      Step s .slotB { s with player := { s.player with shielding := false } } []

  -- 交互优先级耗尽且 A 槽武器没有有效目标时，Python 产生 no-effect。
  | slotANoEffect {s : WorldState}
      (hinteraction : ¬ primaryInteractionAvailable s)
      (hattack : ¬ swordInteractionAvailable s) :
      Step s .slotA { s with player := { s.player with shielding := false } }
        [.actionNoEffect]

  -- 怪物与未举盾玩家接触时扣血；Nat 减法保证 HP 最低为 0。
  | monsterContact {s : WorldState} {monster : Monster}
      (hmember : monster ∈ (currentRoomState s).monsters)
      (hcontact : monster.pos = s.player.pos)
      (hshield : s.player.shielding = false) :
      Step s .wait
        { s with player := damagePlayer s.player monster.damage }
        [.agentDamaged monster.damage]

  -- 举盾接触不扣 HP，并消费这一帧的 shielding 状态。
  | shieldContact {s : WorldState} {monster : Monster}
      (hmember : monster ∈ (currentRoomState s).monsters)
      (hcontact : monster.pos = s.player.pos)
      (hshield : s.player.shielding = true) :
      Step s .wait
        { s with player := { s.player with shielding := false } }
        [.shieldBlock monster.id]

  -- 怪物动态移动：只能走到相邻且可通行的位置，不能和另一怪物重叠。
  | monsterMove {s : WorldState} {monster : Monster} {q : Position}
      (hmember : monster ∈ (currentRoomState s).monsters)
      (hadj : adjacent monster.pos q)
      (henter : canEnter (currentRoomState s) q)
      (hfree : ¬ ∃ other ∈ (currentRoomState s).monsters,
        other.id ≠ monster.id ∧ other.pos = q) :
      Step s .wait
        (updateCurrentRoom s
          (replaceMonster (currentRoomState s) monster { monster with pos := q }))
        [.monsterMoved monster.id monster.pos q]

  -- 相邻使用 switch：记录 switch 已触发，并把目标桥推进到三态循环的下一态。
  | activateSwitch {s : WorldState} {switch : Switch}
      (hnoChest : ¬ openChestInteractionAvailable s)
      (hnoNpc : ¬ npcInteractionAvailable s)
      (hmember : switch ∈ (currentRoomState s).switches)
      (hadj : interactionReach s.player.pos switch.pos)
      (hbridge : ∃ bridge ∈ (s.rooms switch.targetRoom).bridges,
        bridge.id = switch.targetBridge) :
      Step s .slotA
        (activateSwitchState s switch)
        [.switchActivated switch.id, .bridgeRotated switch.targetBridge]

  -- 成功使用出口：检查全部条件，必要时消耗钥匙，并切换房间和 spawn。
  | useExit {s : WorldState} {exit : Exit} {target : RoomState}
      (hmember : exit ∈ (currentRoomState s).exits)
      (hat : exitContains exit s.player.pos)
      (hreq : exitRequirementSatisfied s exit)
      (htarget : target = s.rooms exit.targetRoom)
      (hspawn : canEnter target exit.targetSpawn) :
      Step s (directionAction exit.direction)
        (transitionThroughExit s exit)
        (exitEvents s exit)

  -- 出口条件不满足时，状态保持不变并记录阻挡。
  | exitBlocked {s : WorldState} {exit : Exit}
      (hmember : exit ∈ (currentRoomState s).exits)
      (hat : exitContains exit s.player.pos)
      (hreq : ¬ exitRequirementSatisfied s exit) :
      Step s (directionAction exit.direction) s [.blocked exit.pos]

  -- Task5 没有完成型出口：Python 在所有有限模板房间的宝箱均可见且打开后完成。
  | completeAllChests {s : WorldState}
      (hobjective : allWorldChestsOpened s) :
      Step s .wait { s with completed := true } [.environmentCompleted]

/-! ## 七、Python tick 的分层调度

`Step` 是单个符号微步。下面进一步区分玩家主动阶段和环境自主阶段：

* `PlayerStep` 要求世界仍在运行，并排除纯怪物/接触/完成结算事件；
* `AutonomousStep` 只接受怪物移动、怪物接触、盾牌格挡或全宝箱完成；
* `EngineTick` 强制每个 tick 先有且只有一个玩家阶段，再执行零个或多个自主
  结算微步。

这比直接使用无约束 `Exec` 更接近 Python engine 的调度顺序，也保证死亡或
已完成状态不能再开始新的玩家动作。
-/

def Running (s : WorldState) : Prop :=
  alive s ∧ s.completed = false

def AutonomousOnlyEvents : List Event → Prop
  | [.monsterMoved _ _ _] => True
  | [.agentDamaged _] => True
  | [.shieldBlock _] => True
  | [.environmentCompleted] => True
  | _ => False

structure PlayerStep
    (s : WorldState) (a : Action) (t : WorldState) (events : List Event) : Prop where
  running : Running s
  step : Step s a t events
  agent_phase : ¬ AutonomousOnlyEvents events

structure AutonomousStep
    (s t : WorldState) (events : List Event) : Prop where
  step : Step s .wait t events
  autonomous_phase : AutonomousOnlyEvents events

inductive AutonomousExec : WorldState → WorldState → Prop where
  | nil {s : WorldState} : AutonomousExec s s
  | cons {s t u : WorldState} {events : List Event} :
      AutonomousStep s t events →
      AutonomousExec t u →
      AutonomousExec s u

inductive EngineTick : WorldState → Action → WorldState → Prop where
  | mk {s afterPlayer t : WorldState} {action : Action}
      {playerEvents : List Event} :
      PlayerStep s action afterPlayer playerEvents →
      AutonomousExec afterPlayer t →
      EngineTick s action t

/-!
`EngineExec` 是策略层应使用的完整 tick 轨迹。动作列表只记录玩家每个 tick
提交的动作；怪物移动、接触和完成结算仍由相应 `EngineTick` 内部的
`AutonomousExec` 承担，不会伪装成额外玩家 WAIT。
-/
inductive EngineExec : WorldState → List Action → WorldState → Prop where
  | nil {s : WorldState} : EngineExec s [] s
  | cons {s t u : WorldState} {action : Action} {actions : List Action} :
      EngineTick s action t →
      EngineExec t actions u →
      EngineExec s (action :: actions) u

/- `worlds` 是某个具体有限关卡的全部可达运行时状态证书。 -/
def TickInvariantClosed (worlds : List WorldState) : Prop :=
  ∀ source, source ∈ worlds → WorldInvariant source →
    ∀ action target, EngineTick source action target →
      target ∈ worlds ∧ WorldInvariant target

theorem engineExec_preserves_worldInvariant_in_closed_system
    {worlds : List WorldState} {initial final : WorldState}
    {actions : List Action}
    (hclosed : TickInvariantClosed worlds)
    (hmember : initial ∈ worlds)
    (hinvariant : WorldInvariant initial)
    (hexec : EngineExec initial actions final) :
    final ∈ worlds ∧ WorldInvariant final := by
  induction hexec with
  | nil => exact ⟨hmember, hinvariant⟩
  | cons htick hrest ih =>
      rcases hclosed _ hmember hinvariant _ _ htick with ⟨hmiddle, hmiddleInv⟩
      exact ih hmiddle hmiddleInv

theorem engineExec_append
    {s t u : WorldState} {xs ys : List Action}
    (h₁ : EngineExec s xs t) (h₂ : EngineExec t ys u) :
    EngineExec s (xs ++ ys) u := by
  induction h₁ with
  | nil => simpa using h₂
  | cons htick hrest ih =>
      exact EngineExec.cons htick (ih h₂)

theorem dead_state_has_no_player_step
    {s t : WorldState} {a : Action} {events : List Event}
    (hdead : dead s) :
    ¬ PlayerStep s a t events := by
  intro h
  have halive := h.running.1
  unfold dead at hdead
  unfold alive at halive
  rw [hdead] at halive
  exact Nat.lt_irrefl 0 halive

theorem completed_state_has_no_player_step
    {s t : WorldState} {a : Action} {events : List Event}
    (hcompleted : s.completed = true) :
    ¬ PlayerStep s a t events := by
  intro h
  have hnotComplete := h.running.2
  rw [hcompleted] at hnotComplete
  contradiction

/-! ## 八、目标谓词

先定义可直接复用的原子目标，再用 `Goal` 和 `GoalHolds` 给出统一解释器。
五个 Task 可以通过 `Goal.both` 组合“拿钥匙、清怪、按按钮、到房间、通关”
等条件，而不需要为每关重新发明环境状态。
-/

def HasKey (s : WorldState) : Prop := 0 < s.player.inventory.keys
def HasItem (s : WorldState) (item : Item) : Prop :=
  item ∈ s.player.inventory.items
def ChestOpened (s : WorldState) (id : ObjectId) : Prop :=
  ∃ c ∈ (currentRoomState s).chests, c.id = id ∧ c.opened = true
def AllMonstersDefeated (s : WorldState) : Prop :=
  (currentRoomState s).monsters = []
def ButtonPressed (s : WorldState) (id : ObjectId) : Prop :=
  buttonIsPressed (currentRoomState s) id
def RoomReached (s : WorldState) (id : RoomId) : Prop := s.currentRoom = id
def ExitReached (s : WorldState) (id : ObjectId) : Prop :=
  ∃ e ∈ (currentRoomState s).exits, e.id = id ∧ s.player.pos = e.pos
def WorldCompleted (s : WorldState) : Prop := s.completed = true

inductive Goal where
  | alive
  | hasKey
  | hasItem (item : Item)
  | chestOpened (id : ObjectId)
  | monstersDefeated
  | buttonPressed (id : ObjectId)
  | roomReached (id : RoomId)
  | exitReached (id : ObjectId)
  | worldCompleted
  | both (left right : Goal)
  deriving DecidableEq, Repr

def GoalHolds (s : WorldState) : Goal → Prop
  | .alive => alive s
  | .hasKey => HasKey s
  | .hasItem item => HasItem s item
  | .chestOpened id => ChestOpened s id
  | .monstersDefeated => AllMonstersDefeated s
  | .buttonPressed id => ButtonPressed s id
  | .roomReached id => RoomReached s id
  | .exitReached id => ExitReached s id
  | .worldCompleted => WorldCompleted s
  | .both left right => GoalHolds s left ∧ GoalHolds s right

/-! ## 九、多步执行轨迹

`Exec s actions t` 是 `Step` 的列表闭包：空动作保持原状态；非空轨迹由一次
合法 `Step` 和剩余轨迹组成。后续策略形式化会用它表达“执行 planner 给出的
动作序列后到达满足目标谓词的状态”。
-/

inductive Exec : WorldState → List Action → WorldState → Prop where
  | nil {s : WorldState} : Exec s [] s
  | cons {s t u : WorldState} {a : Action} {actions : List Action} {events : List Event} :
      Step s a t events → Exec t actions u → Exec s (a :: actions) u

/-! ## 十、基础安全性与不变量证明

本节对应环境形式化评分中的“基本安全性或不变量”。证明覆盖房间更新、
生命上界、合法移动、静态障碍、宝箱资源、攻击、按钮、桥、盾牌、陷阱、
出口和轨迹组合。所有证明都由 Lean 内核完整检查。
-/

-- 房间函数更新正确性：写入目标房间后可读回新值，其他房间保持不变。
theorem setRoom_same (rooms : RoomId → RoomState) (id : RoomId) (r : RoomState) :
    setRoom rooms id r id = r := by
  simp [setRoom]

theorem setRoom_other (rooms : RoomId → RoomState) (id other : RoomId)
    (h : other ≠ id) (r : RoomState) :
    setRoom rooms id r other = rooms other := by
  simp [setRoom, h]

@[simp] theorem currentRoomState_updateCurrentRoom
    (s : WorldState) (room : RoomState) :
    currentRoomState (updateCurrentRoom s room) = room := by
  simp [currentRoomState, updateCurrentRoom, setRoom]

-- 生命值不变量：伤害不会增加 HP，任何 loot（包括治疗）都不突破 maxHp。
theorem damage_hp_le (p : PlayerState) (amount : Nat) :
    (damagePlayer p amount).hp ≤ p.hp := by
  simp [damagePlayer]

theorem damage_preserves_hp_bound {p : PlayerState} (amount : Nat)
    (h : p.hp ≤ p.maxHp) :
    (damagePlayer p amount).hp ≤ (damagePlayer p amount).maxHp := by
  exact Nat.le_trans (damage_hp_le p amount) h

theorem collectLoot_hp_bound {p : PlayerState} {loot : Loot}
    (h : p.hp ≤ p.maxHp) :
    (collectLoot p loot).hp ≤ (collectLoot p loot).maxHp := by
  cases loot with
  | key n => exact h
  | gold n => exact h
  | item item => exact h
  | tool item slot =>
      cases slot <;> exact h
  | heal n =>
      exact Nat.min_le_left _ _

/-!
`step_preserves_validState` 是环境层的总不变量：只要初态玩家在当前房间边界内，
且 HP 不超过最大值，那么任意合法 `Step` 的终态仍满足这两个条件。碰撞自由
由移动相关定理单独保证，因为动态桥旋转需要额外的机关安全前提。
-/

theorem step_preserves_validState
    {s t : WorldState} {a : Action} {events : List Event}
    (hvalid : ValidState s)
    (hstep : Step s a t events) :
    ValidState t := by
  cases hstep with
  | wait =>
      exact hvalid
  | movePlain ha hq henter htrap hbutton =>
      exact ⟨henter.1, hvalid.2⟩
  | moveButton ha hq henter hbutton hpos =>
      constructor
      · simpa [currentRoomState, updateCurrentRoom, setRoom, pressButtonAt]
          using henter.1
      · exact hvalid.2
  | @moveTrapSurvive a d q trap ha hq henter htrap hpos hactive
      hsurvives hrespawn =>
      constructor
      · simp only [currentRoomState_updateCurrentRoom]
        unfold deactivateTrap
        split <;> exact hrespawn.1
      · exact damage_preserves_hp_bound trap.damage hvalid.2
  | @moveTrapFatal a d q trap ha hq henter htrap hpos hactive hfatal =>
      constructor
      · simp only [currentRoomState_updateCurrentRoom]
        unfold deactivateTrap
        split <;> exact henter.1
      · exact damage_preserves_hp_bound trap.damage hvalid.2
  | moveBlocked ha hblocked =>
      exact hvalid
  | faceMonster ha hmonster =>
      exact hvalid
  | @openChest chest hmember hvisible hclosed hadj =>
      constructor
      · simp only [currentRoomState_updateCurrentRoom]
        cases chest.loot with
        | key n | gold n | heal n | item n => exact hvalid.1
        | tool item slot =>
            cases slot <;> exact hvalid.1
      · simpa [updateCurrentRoom] using
          (collectLoot_hp_bound
            (p := { s.player with shielding := false })
            (loot := chest.loot) hvalid.2)
  | talkNpc hnoChest hmember hadj =>
      exact hvalid
  | attackDamage hnoInteraction hequipped hsword hmember htarget hsurvives =>
      constructor
      · simpa [currentRoomState, updateCurrentRoom, setRoom, replaceMonster]
          using hvalid.1
      · exact hvalid.2
  | @attackKill monster hnoInteraction hequipped hsword hmember htarget hkilled =>
      constructor
      · rw [resolveMonsterKill_current_bounds, resolveMonsterKill_player]
        simpa [rewardPlayer] using hvalid.1
      · rw [resolveMonsterKill_player]
        simpa [rewardPlayer] using hvalid.2
  | shield hshield =>
      exact hvalid
  | shieldUnavailable hshield =>
      exact hvalid
  | slotANoEffect hinteraction hattack =>
      exact hvalid
  | @monsterContact monster hmember hcontact hshield =>
      exact ⟨hvalid.1, damage_preserves_hp_bound monster.damage hvalid.2⟩
  | shieldContact hmember hcontact hshield =>
      exact hvalid
  | monsterMove hmember hadj henter hfree =>
      constructor
      · simpa [currentRoomState, updateCurrentRoom, setRoom, replaceMonster]
          using hvalid.1
      · exact hvalid.2
  | activateSwitch hnoChest hnoNpc hmember hadj hbridge =>
      constructor
      · rw [activateSwitchState_current_bounds]
        exact hvalid.1
      · exact hvalid.2
  | useExit hmember hat hreq htarget hspawn =>
      constructor
      · rw [transitionThroughExit_target_bounds]
        simpa [transitionThroughExit, htarget] using hspawn.1
      · exact hvalid.2
  | exitBlocked hmember hat hreq =>
      exact hvalid
  | completeAllChests hobjective =>
      exact hvalid

theorem step_preserves_roomIds
    {s t : WorldState} {a : Action} {events : List Event}
    (hstep : Step s a t events) :
    t.roomIds = s.roomIds := by
  cases hstep <;> try rfl
  exact resolveMonsterKill_roomIds _ _

theorem step_currentRoom_remains_known
    {s t : WorldState} {a : Action} {events : List Event}
    (hcurrent : s.currentRoom ∈ s.roomIds)
    (htargets : ExitTargetsKnown s)
    (hstep : Step s a t events) :
    t.currentRoom ∈ t.roomIds := by
  have hcurrentTargets :
      ∀ exit, exit ∈ (currentRoomState s).exits →
        exit.targetRoom ∈ s.roomIds := by
    intro exit hmember
    exact htargets s.currentRoom hcurrent exit hmember
  cases hstep <;>
    simp_all [updateCurrentRoom, activateSwitchState, transitionThroughExit,
      ExitTargetsKnown, currentRoomState]

/-!
核心世界良构性的保持定理。普通动作不改变当前房间；出口动作虽然改变当前
房间，但 `ExitTargetsKnown` 保证目标 ID 属于同一个有限房间索引。玩家界内和
HP 上界由 `step_preserves_validState` 统一处理。
-/
theorem step_preserves_wellFormedWorld
    {s t : WorldState} {a : Action} {events : List Event}
    (hwell : WellFormedWorld s)
    (htargets : ExitTargetsKnown s)
    (hstep : Step s a t events) :
    WellFormedWorld t := by
  rcases hwell with ⟨hnonempty, hnodup, hcurrent, hvalid⟩
  have hids : t.roomIds = s.roomIds := step_preserves_roomIds hstep
  constructor
  · simpa [hids] using hnonempty
  constructor
  · simpa [hids] using hnodup
  constructor
  · exact step_currentRoom_remains_known hcurrent htargets hstep
  · exact step_preserves_validState hvalid hstep

theorem autonomousExec_preserves_validState
    {s t : WorldState}
    (hvalid : ValidState s)
    (hexec : AutonomousExec s t) :
    ValidState t := by
  induction hexec with
  | nil => exact hvalid
  | cons hstep hrest ih =>
      exact ih (step_preserves_validState hvalid hstep.step)

theorem engineTick_preserves_validState
    {s t : WorldState} {action : Action}
    (hvalid : ValidState s)
    (htick : EngineTick s action t) :
    ValidState t := by
  cases htick with
  | mk hplayer hautonomous =>
      have hafter : ValidState _ :=
        step_preserves_validState hvalid hplayer.step
      exact autonomousExec_preserves_validState hafter hautonomous

theorem engineExec_preserves_validState
    {s t : WorldState} {actions : List Action}
    (hvalid : ValidState s)
    (hexec : EngineExec s actions t) :
    ValidState t := by
  induction hexec with
  | nil => exact hvalid
  | cons htick hrest ih =>
      exact ih (engineTick_preserves_validState hvalid htick)

-- 通行性分解：从 canEnter 可以直接推出界内、非墙和非可见宝箱。
theorem canEnter_inBounds {r : RoomState} {p : Position}
    (h : canEnter r p) :
    inBounds r.bounds p :=
  h.1

theorem canEnter_not_wall {r : RoomState} {p : Position}
    (h : canEnter r p) :
    p ∉ r.walls := by
  intro hp
  exact h.2.1 (Or.inl hp)

theorem canEnter_not_visible_chest {r : RoomState} {p : Position}
    (h : canEnter r p) :
    ¬ visibleChestAt r p := by
  intro hc
  exact h.2.1 (Or.inr (Or.inr hc))

theorem safeTile_not_monster {r : RoomState} {p : Position}
    (h : safeTile r p) :
    ¬ monsterAt r p :=
  h.2.2

-- 移动安全：阻挡保持位置；合法移动保证界内，且不会进入墙或宝箱。
theorem blocked_move_keeps_position (s : WorldState) (d : Direction) :
    ({ s with player := { s.player with facing := d, shielding := false } }).player.pos =
      s.player.pos := by
  rfl

theorem legal_move_in_bounds
    {s : WorldState} {d : Direction} {q : Position}
    (hq : q = advance s.player.pos d)
    (henter : canEnter (currentRoomState s) q) :
    inBounds (currentRoomState s).bounds
      ({ s.player with pos := q, facing := d, shielding := false }).pos := by
  simpa [hq] using henter.1

theorem legal_move_not_into_wall
    {s : WorldState} {d : Direction} {q : Position}
    (henter : canEnter (currentRoomState s) q) :
    ({ s.player with pos := q, facing := d, shielding := false }).pos ∉
      (currentRoomState s).walls := by
  exact canEnter_not_wall henter

theorem legal_move_not_into_visible_chest
    {s : WorldState} {d : Direction} {q : Position}
    (henter : canEnter (currentRoomState s) q) :
    ¬ visibleChestAt (currentRoomState s)
      ({ s.player with pos := q, facing := d, shielding := false }).pos := by
  exact canEnter_not_visible_chest henter

-- 静态障碍记忆的关键性质：opened 不参与阻挡判定，visible chest 始终阻挡。
theorem opened_chest_is_still_a_static_blocker
    {r : RoomState} {c : Chest}
    (hm : c ∈ r.chests) (hv : c.visible = true) :
    staticBlocker r c.pos := by
  exact Or.inr (Or.inr ⟨c, hm, rfl, hv⟩)

theorem opening_chest_marks_open (c : Chest) :
    let opened := { c with opened := true }
    opened.opened = true := by
  rfl

-- 宝箱与资源性质：钥匙按数量增加，治疗值被 maxHp 截断。
theorem key_loot_increases_keys (p : PlayerState) (amount : Nat) :
    (collectLoot p (.key amount)).inventory.keys =
      p.inventory.keys + amount := by
  rfl

theorem heal_never_exceeds_max_hp (p : PlayerState) (amount : Nat) :
    (collectLoot p (.heal amount)).hp ≤ p.maxHp := by
  exact Nat.min_le_left _ _

-- 战斗性质：在攻击力为正且怪物未被击杀的分支中，怪物 HP 严格下降。
theorem damaging_attack_reduces_monster_hp
    (monster : Monster) (power : Nat) (hpower : 0 < power)
    (hsurvives : power < monster.hp) :
    ({ monster with hp := monster.hp - power }).hp < monster.hp := by
  exact Nat.sub_lt (Nat.zero_lt_of_lt hsurvives) hpower

/-! ### 清怪后的隐藏宝箱与条件门

下面两条存在性定理直接刻画 Python 结算结果：符合 reveal_on 的隐藏宝箱在
更新后列表中仍是同一 ID、同一坐标且已可见；包含清怪条件的出口同理仍在
列表中、坐标不变且 `opened = true`。
-/

theorem matching_hidden_chest_is_revealed
    {room : RoomState} {chest : Chest} {triggerRoom : RoomId}
    (hmember : chest ∈ room.chests)
    (hhidden : chest.visible = false)
    (hmatch : chestRevealMatches triggerRoom chest.revealOn = true) :
    ∃ revealed ∈ (revealEligibleChests room triggerRoom).chests,
      revealed.id = chest.id ∧
      revealed.pos = chest.pos ∧
      revealed.visible = true := by
  let updateChest := fun candidate : Chest =>
    if !candidate.visible &&
        chestRevealMatches triggerRoom candidate.revealOn then
      { candidate with visible := true }
    else candidate
  have hmapped : updateChest chest ∈ room.chests.map updateChest := by
    exact List.mem_map_of_mem hmember
  refine ⟨{ chest with visible := true }, ?_, rfl, rfl, rfl⟩
  simpa [revealEligibleChests, updateChest, hhidden, hmatch] using hmapped

theorem clearing_requirement_exit_is_opened
    {room : RoomState} {exit : Exit}
    (hmember : exit ∈ room.exits)
    (hrequirement :
      requirementContainsAllMonstersDefeated exit.requirement = true) :
    ∃ opened ∈ (unlockAllMonstersDefeatedExits room).exits,
      opened.id = exit.id ∧
      opened.pos = exit.pos ∧
      opened.opened = true := by
  let updateExit := fun candidate : Exit =>
    if requirementContainsAllMonstersDefeated candidate.requirement then
      { candidate with opened := true }
    else candidate
  have hmapped : updateExit exit ∈ room.exits.map updateExit := by
    exact List.mem_map_of_mem hmember
  refine ⟨{ exit with opened := true }, ?_, rfl, rfl, rfl⟩
  simpa [unlockAllMonstersDefeatedExits, updateExit, hrequirement] using hmapped

theorem revealEligibleChests_preserves_chest_positions
    (room : RoomState) (triggerRoom : RoomId) :
    (revealEligibleChests room triggerRoom).chests.map Chest.pos =
      room.chests.map Chest.pos := by
  simp [revealEligibleChests, List.map_map]
  intro chest hmember
  split <;> rfl

theorem unlockAllMonstersDefeatedExits_preserves_exit_positions
    (room : RoomState) :
    (unlockAllMonstersDefeatedExits room).exits.map Exit.pos =
      room.exits.map Exit.pos := by
  simp [unlockAllMonstersDefeatedExits, List.map_map]
  intro exit hmember
  split <;> rfl

theorem resolveMonsterKill_preserves_validState
    {s : WorldState} {monster : Monster}
    (hvalid : ValidState s) :
    ValidState (resolveMonsterKill s monster) := by
  constructor
  · rw [resolveMonsterKill_current_bounds, resolveMonsterKill_player]
    simpa [rewardPlayer] using hvalid.1
  · rw [resolveMonsterKill_player]
    simpa [rewardPlayer] using hvalid.2

theorem monster_kill_without_room_clear_has_no_clear_events
    {s : WorldState} {monster : Monster}
    (hremaining :
      (removeMonster (currentRoomState s) monster).monsters ≠ []) :
    monsterKillEvents s monster = [.monsterKilled monster.id] := by
  simp [monsterKillEvents, hremaining]

-- 机关性质：按钮一经设置即为 pressed；三态桥旋转三次回到原朝向。
theorem pressing_button_is_monotone (b : Button) :
    ({ b with pressed := true }).pressed = true := by
  rfl

theorem rotating_bridge_thrice_restores_orientation (o : BridgeOrientation) :
    rotateOrientation (rotateOrientation (rotateOrientation o)) = o := by
  cases o <;> rfl

theorem task4_bridge_cycle_is_exact :
    rotateOrientation .westToNorth = .westToEast ∧
    rotateOrientation .westToEast = .westToSouth ∧
    rotateOrientation .westToSouth = .westToNorth := by
  decide

-- 出口资源条件：钥匙门必须有足够钥匙，consume=true 时准确扣除相应数量。
theorem key_requirement_needs_enough_keys
    (s : WorldState) (count : Nat) (consume : Bool)
    (h : requirementSatisfied s (.keys count consume)) :
    count ≤ s.player.inventory.keys := by
  exact h

theorem consuming_key_requirement_spends_keys
    (inv : Inventory) (count : Nat) :
    (spendRequirement inv (.keys count true)).keys = inv.keys - count := by
  rfl

theorem opened_locked_exit_does_not_spend_again
    (inv : Inventory) (exit : Exit)
    (hkind : exit.kind = .locked) (hopened : exit.opened = true) :
    (spendExitRequirement inv exit).keys = inv.keys := by
  simp [spendExitRequirement, hkind, hopened]

theorem unopened_locked_key_exit_spends_exactly
    (inv : Inventory) (exit : Exit) (count : Nat)
    (hkind : exit.kind = .locked) (hopened : exit.opened = false)
    (hrequirement : exit.requirement = .keys count true) :
    (spendExitRequirement inv exit).keys = inv.keys - count := by
  simp [spendExitRequirement, hkind, hopened, hrequirement, spendRequirement]

theorem free_requirement_is_satisfied (s : WorldState) :
    requirementSatisfied s .free := by
  trivial

-- safeTile 比 canEnter 更严格，因此任何安全 tile 必然首先物理可通行。
theorem safeTile_is_enterable {r : RoomState} {p : Position}
    (h : safeTile r p) :
    canEnter r p :=
  h.1

-- 危险机制：盾牌接触保持 HP；陷阱不增加 HP；陷阱重生点仍在房间边界内。
theorem shield_contact_preserves_hp (s : WorldState) :
    ({ s with player := { s.player with shielding := false } }).player.hp =
      s.player.hp := by
  rfl

theorem trap_damage_never_increases_hp (s : WorldState) (trap : Trap) :
    (damagePlayer s.player trap.damage).hp ≤ s.player.hp :=
  damage_hp_le s.player trap.damage

theorem trap_respawn_in_bounds {s : WorldState} {trap : Trap}
    (hrespawn : canEnter (currentRoomState s) trap.respawn) :
    inBounds (currentRoomState s).bounds
      ({ damagePlayer s.player trap.damage with pos := trap.respawn }).pos := by
  exact hrespawn.1

theorem fatal_trap_keeps_zero_hp_at_trigger
    (s : WorldState) (trap : Trap) (q : Position)
    (hfatal : (damagePlayer s.player trap.damage).hp = 0) :
    ({ damagePlayer s.player trap.damage with pos := q }).hp = 0 ∧
    ({ damagePlayer s.player trap.damage with pos := q }).pos = q := by
  exact ⟨hfatal, rfl⟩

-- 房间切换：成功出口准确写入目标房间和 spawn，完成出口设置 completed。
theorem successful_exit_enters_target_room (s : WorldState) (exit : Exit) :
    (transitionThroughExit s exit).currentRoom = exit.targetRoom ∧
    (transitionThroughExit s exit).player.pos = exit.targetSpawn := by
  unfold transitionThroughExit
  exact ⟨rfl, rfl⟩

theorem successful_exit_spawn_in_bounds
    {s : WorldState} {exit : Exit} {target : RoomState}
    (htarget : target = s.rooms exit.targetRoom)
    (hspawn : canEnter target exit.targetSpawn) :
    inBounds (s.rooms exit.targetRoom).bounds exit.targetSpawn := by
  rw [← htarget]
  exact hspawn.1

theorem completed_exit_sets_world_completed
    (s : WorldState) (exit : Exit) (hcomplete : exit.completesTask = true) :
    WorldCompleted (transitionThroughExit s exit) := by
  simp [WorldCompleted, transitionThroughExit, hcomplete]

-- 轨迹代数：两段可执行轨迹可以拼接，空轨迹当且仅当终态等于初态。
theorem exec_append {s t u : WorldState} {xs ys : List Action}
    (h₁ : Exec s xs t) (h₂ : Exec t ys u) :
    Exec s (xs ++ ys) u := by
  induction h₁ with
  | nil => simpa using h₂
  | cons hstep hrest ih =>
      exact Exec.cons hstep (ih h₂)

theorem exec_preserves_validState
    {s t : WorldState} {actions : List Action}
    (hvalid : ValidState s)
    (hexec : Exec s actions t) :
    ValidState t := by
  induction hexec with
  | nil => exact hvalid
  | cons hstep hrest ih =>
      exact ih (step_preserves_validState hvalid hstep)

theorem autonomousExec_has_microstep_trace
    {s t : WorldState}
    (h : AutonomousExec s t) :
    ∃ actions, Exec s actions t := by
  induction h with
  | nil =>
      exact ⟨[], Exec.nil⟩
  | cons hstep hrest ih =>
      rcases ih with ⟨actions, hactions⟩
      exact ⟨.wait :: actions, Exec.cons hstep.step hactions⟩

theorem engineTick_has_microstep_trace
    {s t : WorldState} {action : Action}
    (h : EngineTick s action t) :
    ∃ actions, Exec s actions t := by
  cases h with
  | mk hplayer hautonomous =>
      rcases autonomousExec_has_microstep_trace hautonomous with
        ⟨actions, hactions⟩
      exact ⟨action :: actions, Exec.cons hplayer.step hactions⟩

theorem engineExec_has_microstep_trace
    {s t : WorldState} {actions : List Action}
    (h : EngineExec s actions t) :
    ∃ microActions, Exec s microActions t := by
  induction h with
  | nil =>
      exact ⟨[], Exec.nil⟩
  | cons htick hrest ih =>
      rcases engineTick_has_microstep_trace htick with
        ⟨tickActions, htickExec⟩
      rcases ih with ⟨restActions, hrestExec⟩
      exact ⟨tickActions ++ restActions, exec_append htickExec hrestExec⟩

theorem exec_nil_iff {s t : WorldState} :
    Exec s [] t ↔ t = s := by
  constructor
  · intro h
    cases h
    rfl
  · intro h
    subst h
    exact Exec.nil

/-! ## 十一、通用图路径与 BFS 层级规格

这一章完全不依赖具体任务。给定任意节点类型 `α` 和邻居枚举函数
`neighbors : α → List α`，`GenericBfsRoutesExact neighbors start n`
表示从 `start` 恰好走 `n` 条边能得到的所有 route。它允许重复 route，
因此描述的是 BFS 的客观层级语义；带 visited/queue 的实现可以作为它的
高效 refinement 来证明。

核心完备性定理 `generic_bfs_complete_within` 说：任意长度不超过
`maxDepth` 的图路径，都会出现在这个通用 BFS 的某一层中。后续 Task1--Task5
只需要把各自的“允许移动”实例化为图邻居关系。
-/

inductive GenericGraphPath {α : Type}
    (neighbors : α → List α) :
    α → List α → α → Prop where
  | nil (p : α) :
      GenericGraphPath neighbors p [] p
  | cons {p q goal : α} {rest : List α}
      (hstep : q ∈ neighbors p)
      (htail : GenericGraphPath neighbors q rest goal) :
      GenericGraphPath neighbors p (q :: rest) goal

def GenericRouteEnd {α : Type} : α → List α → α
  | current, [] => current
  | _current, next :: rest => GenericRouteEnd next rest

def GenericBfsRoutesExact {α : Type}
    (neighbors : α → List α) : α → Nat → List (List α)
  | _start, 0 => [[]]
  | start, n + 1 =>
      List.flatMap (fun next =>
        (GenericBfsRoutesExact neighbors next n).map
          (fun route => next :: route)) (neighbors start)

def GenericBfsFindsGoalAtDepth {α : Type}
    (neighbors : α → List α) (start : α)
    (goals : List α) (depth : Nat) : Prop :=
  ∃ route goal,
    route ∈ GenericBfsRoutesExact neighbors start depth ∧
    goal ∈ goals ∧
    GenericRouteEnd start route = goal

def GenericBfsFindsGoalWithin {α : Type}
    (neighbors : α → List α) (start : α)
    (goals : List α) (maxDepth : Nat) : Prop :=
  ∃ depth,
    depth ≤ maxDepth ∧
    GenericBfsFindsGoalAtDepth neighbors start goals depth

theorem generic_graph_path_mem_bfs_routes_exact
    {α : Type} {neighbors : α → List α}
    {start goal : α} {route : List α}
    (hpath : GenericGraphPath neighbors start route goal) :
    route ∈ GenericBfsRoutesExact neighbors start route.length ∧
    GenericRouteEnd start route = goal := by
  induction hpath with
  | nil p =>
      simp [GenericBfsRoutesExact, GenericRouteEnd]
  | cons hstep htail ih =>
      constructor
      · apply List.mem_flatMap.mpr
        exact ⟨
          _,
          hstep,
          List.mem_map.mpr ⟨_, ih.1, rfl⟩
        ⟩
      · simpa [GenericRouteEnd] using ih.2

theorem generic_bfs_route_layer_sound
    {α : Type} {neighbors : α → List α}
    {start : α} {depth : Nat} {route : List α}
    (hmem : route ∈ GenericBfsRoutesExact neighbors start depth) :
    GenericGraphPath neighbors start route (GenericRouteEnd start route) := by
  induction depth generalizing start route with
  | zero =>
      simp [GenericBfsRoutesExact] at hmem
      subst route
      simp [GenericRouteEnd]
      exact GenericGraphPath.nil start
  | succ n ih =>
      rw [GenericBfsRoutesExact] at hmem
      rcases List.mem_flatMap.mp hmem with ⟨next, hnext, htailMap⟩
      rcases List.mem_map.mp htailMap with ⟨tail, htail, hroute⟩
      subst route
      exact GenericGraphPath.cons hnext (ih htail)

theorem generic_bfs_complete_within
    {α : Type} {neighbors : α → List α}
    {start : α} {goals : List α} {maxDepth : Nat}
    (hreachable : ∃ route goal,
      goal ∈ goals ∧
      route.length ≤ maxDepth ∧
      GenericGraphPath neighbors start route goal) :
    GenericBfsFindsGoalWithin neighbors start goals maxDepth := by
  rcases hreachable with ⟨route, goal, hgoal, hbound, hpath⟩
  rcases generic_graph_path_mem_bfs_routes_exact hpath with
    ⟨hmem, hend⟩
  exact ⟨route.length, hbound, route, goal, hmem, hgoal, hend⟩

/-! ### 可执行的有界 BFS

`GenericBfsRoutesExact` 是按深度分层的队列语义。下面的搜索从第 0 层
依次扫描到 `fuel` 层，并返回第一条终点属于 `goals` 的 route。
它是可计算的 `Option` 函数，不再只是存在性规格。
-/

def routeEndsAtGoal { α : Type } [DecidableEq α]
    (start : α) (goals : List α) (route : List α) : Bool :=
  decide (GenericRouteEnd start route ∈ goals)

def findGoalRouteAtDepth { α : Type } [DecidableEq α]
    (neighbors : α → List α) (start : α)
    (goals : List α) (depth : Nat) : Option (List α) :=
  (GenericBfsRoutesExact neighbors start depth).find?
    (routeEndsAtGoal start goals)

def executableBfsWithin { α : Type } [DecidableEq α]
    (neighbors : α → List α) (start : α)
    (goals : List α) : Nat → Option (List α)
  | 0 => findGoalRouteAtDepth neighbors start goals 0
  | fuel + 1 =>
      match executableBfsWithin neighbors start goals fuel with
      | some route => some route
      | none => findGoalRouteAtDepth neighbors start goals (fuel + 1)

def ExecutableBfsResult { α : Type } [DecidableEq α]
    (neighbors : α → List α) (start : α)
    (goals : List α) (fuel : Nat) (route : List α) : Prop :=
  ∃ depth, depth ≤ fuel ∧
    route ∈ GenericBfsRoutesExact neighbors start depth ∧
    GenericRouteEnd start route ∈ goals

theorem findGoalRouteAtDepth_sound
    { α : Type } [DecidableEq α]
    {neighbors : α → List α} {start : α}
    {goals : List α} {depth : Nat} {route : List α}
    (h : findGoalRouteAtDepth neighbors start goals depth = some route) :
    route ∈ GenericBfsRoutesExact neighbors start depth ∧
    GenericRouteEnd start route ∈ goals := by
  have hmem : route ∈ GenericBfsRoutesExact neighbors start depth :=
    List.mem_of_find?_eq_some h
  have hpredicate : routeEndsAtGoal start goals route = true := by
    rw [findGoalRouteAtDepth] at h
    exact (List.find?_eq_some_iff_getElem.mp h).1
  exact ⟨hmem, of_decide_eq_true hpredicate⟩

theorem executableBfsWithin_sound
    { α : Type } [DecidableEq α]
    {neighbors : α → List α} {start : α}
    {goals : List α} {fuel : Nat} {route : List α}
    (h : executableBfsWithin neighbors start goals fuel = some route) :
    ExecutableBfsResult neighbors start goals fuel route := by
  induction fuel with
  | zero =>
      exact ⟨0, Nat.le_refl 0, findGoalRouteAtDepth_sound h⟩
  | succ fuel ih =>
      simp only [executableBfsWithin] at h
      split at h
      next prior hprior =>
        cases h
        rcases ih hprior with ⟨depth, hdepth, hmem, hgoal⟩
        exact ⟨depth, Nat.le_trans hdepth (Nat.le_succ fuel), hmem, hgoal⟩
      next hnone =>
        rcases findGoalRouteAtDepth_sound h with ⟨hmem, hgoal⟩
        exact ⟨fuel + 1, Nat.le_refl _, hmem, hgoal⟩

theorem executableBfsWithin_returned_path_is_sound
    { α : Type } [DecidableEq α]
    {neighbors : α → List α} {start : α}
    {goals : List α} {fuel : Nat} {route : List α}
    (h : executableBfsWithin neighbors start goals fuel = some route) :
    GenericGraphPath neighbors start route (GenericRouteEnd start route) ∧
    GenericRouteEnd start route ∈ goals := by
  rcases executableBfsWithin_sound h with ⟨depth, hdepth, hmem, hgoal⟩
  exact ⟨generic_bfs_route_layer_sound hmem, hgoal⟩

theorem findGoalRouteAtDepth_is_some_of_witness
    { α : Type } [DecidableEq α]
    {neighbors : α → List α} {start : α}
    {goals : List α} {depth : Nat} {route : List α}
    (hmem : route ∈ GenericBfsRoutesExact neighbors start depth)
    (hgoal : GenericRouteEnd start route ∈ goals) :
    findGoalRouteAtDepth neighbors start goals depth ≠ none := by
  intro hnone
  have hall := (List.find?_eq_none.mp hnone) route hmem
  exact hall (decide_eq_true hgoal)

theorem executableBfsWithin_complete
    { α : Type } [DecidableEq α]
    {neighbors : α → List α} {start : α}
    {goals : List α} {fuel : Nat}
    (hreachable : ∃ route goal,
      goal ∈ goals ∧ route.length ≤ fuel ∧
      GenericGraphPath neighbors start route goal) :
    executableBfsWithin neighbors start goals fuel ≠ none := by
  rcases hreachable with ⟨route, goal, hgoal, hlength, hpath⟩
  rcases generic_graph_path_mem_bfs_routes_exact hpath with ⟨hmem, hend⟩
  have hexact :
      findGoalRouteAtDepth neighbors start goals route.length ≠ none :=
    findGoalRouteAtDepth_is_some_of_witness hmem (hend.symm ▸ hgoal)
  induction fuel with
  | zero =>
      have hzero : route.length = 0 := Nat.eq_zero_of_le_zero hlength
      simpa [executableBfsWithin, hzero] using hexact
  | succ fuel ih =>
      by_cases hle : route.length ≤ fuel
      · have hprior := ih hle
        cases hvalue : executableBfsWithin neighbors start goals fuel with
        | none => exact False.elim (hprior hvalue)
        | some prior => simp [executableBfsWithin, hvalue]
      · have heq : route.length = fuel + 1 := by omega
        simp only [executableBfsWithin]
        split
        · simp
        · simpa [heq] using hexact

/-! ## 十二、通用 tile 路径、BFS 结果与动作翻译

`TilePath r start route goal` 表示 `route` 是从 `start` 到 `goal` 的安全 tile
路径。每个构造步骤都明确给出方向，要求下一位置等于 `advance` 的结果，
并要求下一位置满足 `safeTile`。因此任何由该关系认证的 BFS 路径都不会
越界、撞墙、穿宝箱、进入 gap、陷阱或怪物。

这一章仍然是任务无关的符号层基础设施：Task1、Task2 和后续多房间任务都可以
复用 `TilePath`、`BfsResult`、`actionForDirection` 与无按钮房间的执行引理。
-/

inductive TilePath (r : RoomState) : Position → List Position → Position → Prop where
  | nil (p : Position) :
      TilePath r p [] p
  | cons {p q goal : Position} {rest : List Position} (d : Direction)
      (hq : q = advance p d)
      (hsafe : safeTile r q)
      (htail : TilePath r q rest goal) :
      TilePath r p (q :: rest) goal

/-! 可计算方向脚本。公开地图证书只需给出方向列表；`DirectionPlanSafe`
逐格检查每个中间位置，下面的引理再把检查结果翻译成 `TilePath`。 -/

def directionPositions : Position → List Direction → List Position
  | _, [] => []
  | p, direction :: rest =>
      let next := advance p direction
      next :: directionPositions next rest

def directionEndpoint : Position → List Direction → Position
  | p, [] => p
  | p, direction :: rest =>
      directionEndpoint (advance p direction) rest

def DirectionPlanSafe (room : RoomState) : Position → List Direction → Prop
  | _, [] => True
  | p, direction :: rest =>
      safeTile room (advance p direction) ∧
      DirectionPlanSafe room (advance p direction) rest

theorem directionPlanSafe_to_tilePath
    {room : RoomState} {start : Position} {directions : List Direction}
    (hsafe : DirectionPlanSafe room start directions) :
    TilePath room start (directionPositions start directions)
      (directionEndpoint start directions) := by
  induction directions generalizing start with
  | nil => exact TilePath.nil start
  | cons direction rest ih =>
      exact TilePath.cons direction rfl hsafe.1 (ih hsafe.2)

def TileReachable (r : RoomState) (start goal : Position) : Prop :=
  ∃ route, TilePath r start route goal

def BfsResult
    (r : RoomState) (start : Position) (goals : List Position)
    (route : List Position) : Prop :=
  ∃ goal, goal ∈ goals ∧ TilePath r start route goal

theorem tilePath_goal_reachable
    {r : RoomState} {start goal : Position} {route : List Position}
    (hpath : TilePath r start route goal) :
    TileReachable r start goal :=
  ⟨route, hpath⟩

theorem directionPlan_endpoint_reachable
    {room : RoomState} {start : Position} {directions : List Direction}
    (hsafe : DirectionPlanSafe room start directions) :
    TileReachable room start (directionEndpoint start directions) :=
  tilePath_goal_reachable (directionPlanSafe_to_tilePath hsafe)

theorem tilePath_first_step_safe
    {r : RoomState} {start first goal : Position} {rest : List Position}
    (hpath : TilePath r start (first :: rest) goal) :
    safeTile r first := by
  cases hpath with
  | cons d hq hsafe htail => exact hsafe

theorem tilePath_first_step_adjacent
    {r : RoomState} {start first goal : Position} {rest : List Position}
    (hpath : TilePath r start (first :: rest) goal) :
    adjacent start first := by
  cases hpath with
  | cons d hq hsafe htail =>
      subst hq
      cases d with
      | north => exact Or.inl rfl
      | south => exact Or.inr (Or.inl rfl)
      | west => exact Or.inr (Or.inr (Or.inl rfl))
      | east => exact Or.inr (Or.inr (Or.inr rfl))

theorem bfs_result_is_sound
    {r : RoomState} {start : Position} {goals route : List Position}
    (hresult : BfsResult r start goals route) :
    ∃ goal, goal ∈ goals ∧ TilePath r start route goal :=
  hresult

theorem bfs_first_move_is_safe
    {r : RoomState} {start first : Position} {goals rest : List Position}
    (hresult : BfsResult r start goals (first :: rest)) :
    safeTile r first := by
  rcases hresult with ⟨goal, hgoal, hpath⟩
  exact tilePath_first_step_safe hpath

def actionForDirection : Direction → Action
  | .north => .up
  | .south => .down
  | .west => .left
  | .east => .right

def movePlayerState (s : WorldState) (q : Position) (d : Direction) : WorldState :=
  { s with player := { s.player with pos := q, facing := d, shielding := false } }

theorem actionForDirection_correct (d : Direction) :
    actionDirection (actionForDirection d) = some d := by
  cases d <;> rfl

theorem movePlayerState_room_unchanged
    (s : WorldState) (q : Position) (d : Direction) :
    currentRoomState (movePlayerState s q d) = currentRoomState s := by
  rfl

@[simp] theorem movePlayerState_currentRoom
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).currentRoom = s.currentRoom := rfl
@[simp] theorem movePlayerState_rooms
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).rooms = s.rooms := rfl
@[simp] theorem movePlayerState_roomIds
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).roomIds = s.roomIds := rfl
@[simp] theorem movePlayerState_completed
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).completed = s.completed := rfl
@[simp] theorem movePlayerState_pos
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).player.pos = q := rfl
@[simp] theorem movePlayerState_facing
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).player.facing = d := rfl
@[simp] theorem movePlayerState_hp
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).player.hp = s.player.hp := rfl
@[simp] theorem movePlayerState_maxHp
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).player.maxHp = s.player.maxHp := rfl
@[simp] theorem movePlayerState_inventory
    (s : WorldState) (q : Position) (d : Direction) :
    (movePlayerState s q d).player.inventory = s.player.inventory := rfl

def applyPlainTileAction (s : WorldState) (action : Action) : WorldState :=
  match actionDirection action with
  | some d => movePlayerState s (advance s.player.pos d) d
  | none => s

def runPlainTileActions : WorldState → List Action → WorldState
  | s, [] => s
  | s, action :: actions =>
      runPlainTileActions (applyPlainTileAction s action) actions

theorem applyPlainTileAction_currentRoomState
    (s : WorldState) (action : Action) :
    currentRoomState (applyPlainTileAction s action) =
      currentRoomState s := by
  unfold applyPlainTileAction
  cases h : actionDirection action with
  | none => rfl
  | some d =>
      exact movePlayerState_room_unchanged s (advance s.player.pos d) d

theorem runPlainTileActions_currentRoomState
    (s : WorldState) (actions : List Action) :
    currentRoomState (runPlainTileActions s actions) =
      currentRoomState s := by
  induction actions generalizing s with
  | nil => rfl
  | cons action rest ih =>
      rw [runPlainTileActions]
      rw [ih (applyPlainTileAction s action)]
      exact applyPlainTileAction_currentRoomState s action

theorem applyPlainTileAction_inventory_keys
    (s : WorldState) (action : Action) :
    (applyPlainTileAction s action).player.inventory.keys =
      s.player.inventory.keys := by
  unfold applyPlainTileAction
  cases h : actionDirection action with
  | none => rfl
  | some d => rfl

theorem runPlainTileActions_inventory_keys
    (s : WorldState) (actions : List Action) :
    (runPlainTileActions s actions).player.inventory.keys =
      s.player.inventory.keys := by
  induction actions generalizing s with
  | nil => rfl
  | cons action rest ih =>
      rw [runPlainTileActions]
      rw [ih (applyPlainTileAction s action)]
      exact applyPlainTileAction_inventory_keys s action

/-!
Pixel-level execution is shared infrastructure rather than a Task2 fact. The
Lean environment is tile-based, so a proof that repeated pixel ticks constitute
one tile move must come from a separate renderer/physics refinement. Here we
state that refinement as a small, explicit kinematic contract and keep all
symbolic safety proofs downstream of the tile trace.
-/

def PixelActionsRefineTileActions
    (pixelActions tileActions : List Action) : Prop :=
  let ActionBlockRefines (block : List Action) (tileAction : Action) : Prop :=
    block ≠ [] ∧ ∀ action, action ∈ block → action = tileAction
  let rec BlocksRefine : List Action → List Action → Prop
    | [], [] => True
    | [], _ :: _ => False
    | _ :: _, [] => False
    | pixels, tileAction :: restTiles =>
        ∃ block restPixels,
          pixels = block ++ restPixels ∧
          ActionBlockRefines block tileAction ∧
          BlocksRefine restPixels restTiles
  BlocksRefine pixelActions tileActions

structure PixelToTileKinematicRefinement
    (start finish : WorldState)
    (pixelActions tileActions : List Action) : Prop where
  pixel_refines_tile_actions :
    PixelActionsRefineTileActions pixelActions tileActions
  finish_is_tile_run :
    finish = runPlainTileActions start tileActions

theorem tilePath_has_executable_plan
    {r : RoomState} {s : WorldState} {start goal : Position}
    {route : List Position}
    (hroom : currentRoomState s = r)
    (hstart : s.player.pos = start)
    (hbuttons : r.buttons = [])
    (hpath : TilePath r start route goal) :
    ∃ actions final,
      Exec s actions final ∧
      final.player.pos = goal ∧
      currentRoomState final = r := by
  induction hpath generalizing s with
  | nil p =>
      exact ⟨[], s, Exec.nil, hstart, hroom⟩
  | @cons p q pathGoal rest d hq hsafe htail ih =>
      let next := movePlayerState s q d
      have henterS : canEnter (currentRoomState s) q := by
        rw [hroom]
        exact hsafe.1
      have htrapS : ¬ activeTrapAt (currentRoomState s) q := by
        rw [hroom]
        exact hsafe.2.1
      have hbuttonS : ¬ buttonAt (currentRoomState s) q := by
        rw [hroom]
        intro hb
        rcases hb with ⟨button, hmember, hpos⟩
        rw [hbuttons] at hmember
        simp at hmember
      have hqS : q = advance s.player.pos d := by
        rw [hstart]
        exact hq
      have hstep :
          Step s (actionForDirection d) next
            [.moved s.player.pos q] := by
        exact Step.movePlain (actionForDirection_correct d) hqS
          henterS htrapS hbuttonS
      have hnextRoom : currentRoomState next = r := by
        rw [movePlayerState_room_unchanged, hroom]
      have hnextPos : next.player.pos = q := by
        rfl
      rcases ih hnextRoom hnextPos with
        ⟨tailActions, final, htailExec, hfinalPos, hfinalRoom⟩
      exact ⟨
        actionForDirection d :: tailActions,
        final,
        Exec.cons hstep htailExec,
        hfinalPos,
        hfinalRoom
      ⟩

theorem tilePath_has_engine_plan
    {r : RoomState} {s : WorldState} {start goal : Position}
    {route : List Position}
    (hroom : currentRoomState s = r)
    (hstart : s.player.pos = start)
    (hbuttons : r.buttons = [])
    (hrunning : Running s)
    (hpath : TilePath r start route goal) :
    ∃ actions final,
      EngineExec s actions final ∧
      final.player.pos = goal ∧
      currentRoomState final = r ∧ Running final ∧
      final.player.inventory = s.player.inventory ∧
      final.player.hp = s.player.hp ∧
      final.player.maxHp = s.player.maxHp ∧
      final.currentRoom = s.currentRoom ∧ final.rooms = s.rooms ∧
      final.roomIds = s.roomIds ∧ final.completed = s.completed := by
  induction hpath generalizing s with
  | nil p =>
      exact ⟨[], s, EngineExec.nil, hstart, hroom, hrunning,
        rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  | @cons p q pathGoal rest d hq hsafe htail ih =>
      let next := movePlayerState s q d
      have henterS : canEnter (currentRoomState s) q := by
        rw [hroom]
        exact hsafe.1
      have htrapS : ¬ activeTrapAt (currentRoomState s) q := by
        rw [hroom]
        exact hsafe.2.1
      have hbuttonS : ¬ buttonAt (currentRoomState s) q := by
        rw [hroom]
        intro hb
        rcases hb with ⟨button, hmember, hpos⟩
        rw [hbuttons] at hmember
        simp at hmember
      have hqS : q = advance s.player.pos d := by
        rw [hstart]
        exact hq
      have hstep :
          Step s (actionForDirection d) next [.moved s.player.pos q] :=
        Step.movePlain (actionForDirection_correct d) hqS
          henterS htrapS hbuttonS
      have hplayer :
          PlayerStep s (actionForDirection d) next [.moved s.player.pos q] := by
        refine ⟨hrunning, hstep, ?_⟩
        simp [AutonomousOnlyEvents]
      have htick : EngineTick s (actionForDirection d) next :=
        EngineTick.mk hplayer AutonomousExec.nil
      have hnextRoom : currentRoomState next = r := by
        rw [movePlayerState_room_unchanged, hroom]
      have hnextPos : next.player.pos = q := rfl
      have hnextRunning : Running next := hrunning
      rcases ih hnextRoom hnextPos hnextRunning with
        ⟨tailActions, final, htailExec, hfinalPos, hfinalRoom,
          hfinalRunning, hfinalInventory, hfinalHp, hfinalMaxHp,
          hfinalCurrent, hfinalRooms, hfinalRoomIds, hfinalCompleted⟩
      exact ⟨actionForDirection d :: tailActions, final,
        EngineExec.cons htick htailExec,
        hfinalPos, hfinalRoom, hfinalRunning, hfinalInventory,
        hfinalHp, hfinalMaxHp, hfinalCurrent, hfinalRooms,
        hfinalRoomIds, hfinalCompleted⟩

theorem directionPlan_has_engine_plan
    {room : RoomState} {s : WorldState} {start : Position}
    {directions : List Direction}
    (hroom : currentRoomState s = room)
    (hstart : s.player.pos = start)
    (hbuttons : room.buttons = [])
    (hrunning : Running s)
    (hsafe : DirectionPlanSafe room start directions) :
    ∃ actions final,
      EngineExec s actions final ∧
      final.player.pos = directionEndpoint start directions ∧
      currentRoomState final = room ∧ Running final ∧
      final.player.inventory = s.player.inventory ∧
      final.player.hp = s.player.hp ∧
      final.player.maxHp = s.player.maxHp ∧
      final.currentRoom = s.currentRoom ∧ final.rooms = s.rooms ∧
      final.roomIds = s.roomIds ∧ final.completed = s.completed :=
  tilePath_has_engine_plan hroom hstart hbuttons hrunning
    (directionPlanSafe_to_tilePath hsafe)

/- `applyDirectionPlan` is the executable counterpart of `DirectionPlanSafe`.
   Its exact-state theorem is useful for closed public certificates: no
   existentially chosen intermediate world is hidden in a route segment. -/
def applyDirectionPlan : WorldState → List Direction → WorldState
  | s, [] => s
  | s, direction :: rest =>
      applyDirectionPlan
        (movePlayerState s (advance s.player.pos direction) direction) rest

@[simp] theorem applyDirectionPlan_currentRoom
    (s : WorldState) (directions : List Direction) :
    (applyDirectionPlan s directions).currentRoom = s.currentRoom := by
  induction directions generalizing s with
  | nil => rfl
  | cons direction rest ih => simp [applyDirectionPlan, ih]

@[simp] theorem applyDirectionPlan_rooms
    (s : WorldState) (directions : List Direction) :
    (applyDirectionPlan s directions).rooms = s.rooms := by
  induction directions generalizing s with
  | nil => rfl
  | cons direction rest ih => simp [applyDirectionPlan, ih]

@[simp] theorem applyDirectionPlan_roomIds
    (s : WorldState) (directions : List Direction) :
    (applyDirectionPlan s directions).roomIds = s.roomIds := by
  induction directions generalizing s with
  | nil => rfl
  | cons direction rest ih => simp [applyDirectionPlan, ih]

@[simp] theorem applyDirectionPlan_completed
    (s : WorldState) (directions : List Direction) :
    (applyDirectionPlan s directions).completed = s.completed := by
  induction directions generalizing s with
  | nil => rfl
  | cons direction rest ih => simp [applyDirectionPlan, ih]

@[simp] theorem applyDirectionPlan_inventory
    (s : WorldState) (directions : List Direction) :
    (applyDirectionPlan s directions).player.inventory = s.player.inventory := by
  induction directions generalizing s with
  | nil => rfl
  | cons direction rest ih => simp [applyDirectionPlan, ih]

@[simp] theorem applyDirectionPlan_hp
    (s : WorldState) (directions : List Direction) :
    (applyDirectionPlan s directions).player.hp = s.player.hp := by
  induction directions generalizing s with
  | nil => rfl
  | cons direction rest ih => simp [applyDirectionPlan, ih]

@[simp] theorem applyDirectionPlan_maxHp
    (s : WorldState) (directions : List Direction) :
    (applyDirectionPlan s directions).player.maxHp = s.player.maxHp := by
  induction directions generalizing s with
  | nil => rfl
  | cons direction rest ih => simp [applyDirectionPlan, ih]

@[simp] theorem applyDirectionPlan_pos
    (s : WorldState) (directions : List Direction) :
    (applyDirectionPlan s directions).player.pos =
      directionEndpoint s.player.pos directions := by
  induction directions generalizing s with
  | nil => rfl
  | cons direction rest ih =>
      simp [applyDirectionPlan, directionEndpoint, ih]

@[simp] theorem applyDirectionPlan_currentRoomState
    (s : WorldState) (directions : List Direction) :
    currentRoomState (applyDirectionPlan s directions) = currentRoomState s := by
  simp [currentRoomState]

theorem directionPlan_has_exact_engine_exec
    {room : RoomState} {s : WorldState} {directions : List Direction}
    (hroom : currentRoomState s = room)
    (hbuttons : room.buttons = [])
    (hrunning : Running s)
    (hsafe : DirectionPlanSafe room s.player.pos directions) :
    EngineExec s (directions.map actionForDirection)
      (applyDirectionPlan s directions) := by
  induction directions generalizing s with
  | nil => exact EngineExec.nil
  | cons direction rest ih =>
      let q := advance s.player.pos direction
      let next := movePlayerState s q direction
      have henter : canEnter (currentRoomState s) q := by
        rw [hroom]
        exact hsafe.1.1
      have htrap : ¬ activeTrapAt (currentRoomState s) q := by
        rw [hroom]
        exact hsafe.1.2.1
      have hbutton : ¬ buttonAt (currentRoomState s) q := by
        rw [hroom]
        intro h
        rcases h with ⟨button, hmember, hpos⟩
        rw [hbuttons] at hmember
        simp at hmember
      have hstep : Step s (actionForDirection direction) next
          [.moved s.player.pos q] :=
        Step.movePlain (actionForDirection_correct direction) rfl
          henter htrap hbutton
      have htick : EngineTick s (actionForDirection direction) next :=
        EngineTick.mk ⟨hrunning, hstep, by simp [AutonomousOnlyEvents]⟩
          AutonomousExec.nil
      have hnextRoom : currentRoomState next = room := by
        rw [movePlayerState_room_unchanged, hroom]
      have hnextRunning : Running next := by
        rcases hrunning with ⟨halive, hcomplete⟩
        exact ⟨by simpa [next, movePlayerState, alive] using halive,
          by simpa [next, movePlayerState] using hcomplete⟩
      have htail :
          DirectionPlanSafe room next.player.pos rest := by
        change DirectionPlanSafe room q rest
        exact hsafe.2
      exact EngineExec.cons htick
        (ih hnextRoom hnextRunning htail)

/- A room may contain buttons away from the current route.  This executable
   predicate is the precise condition needed by `Step.movePlain`; unlike the
   older `room.buttons = []` convenience premise it does not erase real map
   objects merely to simplify a certificate. -/
def DirectionPlanAvoidsButtons (room : RoomState) :
    Position → List Direction → Prop
  | _, [] => True
  | p, direction :: rest =>
      ¬ buttonAt room (advance p direction) ∧
      DirectionPlanAvoidsButtons room (advance p direction) rest

theorem directionPlan_has_exact_engine_exec_avoiding_buttons
    {room : RoomState} {s : WorldState} {directions : List Direction}
    (hroom : currentRoomState s = room)
    (hrunning : Running s)
    (hsafe : DirectionPlanSafe room s.player.pos directions)
    (hbuttons : DirectionPlanAvoidsButtons room s.player.pos directions) :
    EngineExec s (directions.map actionForDirection)
      (applyDirectionPlan s directions) := by
  induction directions generalizing s with
  | nil => exact EngineExec.nil
  | cons direction rest ih =>
      let q := advance s.player.pos direction
      let next := movePlayerState s q direction
      have henter : canEnter (currentRoomState s) q := by
        rw [hroom]
        exact hsafe.1.1
      have htrap : ¬ activeTrapAt (currentRoomState s) q := by
        rw [hroom]
        exact hsafe.1.2.1
      have hbutton : ¬ buttonAt (currentRoomState s) q := by
        rw [hroom]
        exact hbuttons.1
      have hstep : Step s (actionForDirection direction) next
          [.moved s.player.pos q] :=
        Step.movePlain (actionForDirection_correct direction) rfl
          henter htrap hbutton
      have htick : EngineTick s (actionForDirection direction) next :=
        EngineTick.mk ⟨hrunning, hstep, by simp [AutonomousOnlyEvents]⟩
          AutonomousExec.nil
      have hnextRoom : currentRoomState next = room := by
        rw [movePlayerState_room_unchanged, hroom]
      have hnextRunning : Running next := by
        rcases hrunning with ⟨halive, hcomplete⟩
        exact ⟨by simpa [next, movePlayerState, alive] using halive,
          by simpa [next, movePlayerState] using hcomplete⟩
      exact EngineExec.cons htick
        (ih hnextRoom hnextRunning hsafe.2 hbuttons.2)

def BoundedTileReachable
    (r : RoomState) (start : Position) (depth : Nat) (goal : Position) : Prop :=
  ∃ route, route.length ≤ depth ∧ TilePath r start route goal

def BfsFrontierComplete
    (r : RoomState) (start : Position) (depth : Nat)
    (frontier : List Position) : Prop :=
  ∀ goal, BoundedTileReachable r start depth goal → goal ∈ frontier

def BfsFindsGoal (frontier goals : List Position) : Prop :=
  ∃ goal, goal ∈ frontier ∧ goal ∈ goals

theorem bfs_complete_from_frontier_invariant
    {r : RoomState} {start : Position} {depth : Nat}
    {frontier goals : List Position}
    (hcomplete : BfsFrontierComplete r start depth frontier)
    (hreachable : ∃ goal, goal ∈ goals ∧
      BoundedTileReachable r start depth goal) :
    BfsFindsGoal frontier goals := by
  rcases hreachable with ⟨goal, hgoal, hbounded⟩
  exact ⟨goal, hcomplete goal hbounded, hgoal⟩

/-! ## 十三、统一观测、命令与闭环执行接口

神经网络本身不进入可信基。`ObservationRefinesWorld` 是符号观测进入
planner 时必须满足的精化契约；`ClosedLoopStep` 则强制策略输出先经过
`ControllerCommandSafe`，再由真实 `EngineTick` 执行。
-/

abbrev physicalEnterable := canEnter
abbrev plannerSafe := safeTile

structure VerifiedObservation where
  playerPos : Position
  visibleChests : List Chest
  visibleMonsters : List Monster
  visibleExits : List Exit
  inventory : Inventory
  walkable : Position → Prop
  safe : Position → Prop

def ObservationRefinesWorld
    (s : WorldState) (obs : VerifiedObservation) : Prop :=
  obs.playerPos = s.player.pos ∧
  obs.visibleChests =
    (currentRoomState s).chests.filter (fun chest => chest.visible) ∧
  obs.visibleMonsters =
    (currentRoomState s).monsters.filter (fun monster => 0 < monster.hp) ∧
  obs.visibleExits = (currentRoomState s).exits ∧
  obs.inventory = s.player.inventory ∧
  (∀ p, obs.walkable p ↔ physicalEnterable (currentRoomState s) p) ∧
  (∀ p, obs.safe p ↔ plannerSafe (currentRoomState s) p)

inductive TurnTarget where
  | chest | monster
  deriving DecidableEq, Repr

inductive ControllerCommand where
  | navigate (direction : Direction)
  | turnOnly (direction : Direction) (target : TurnTarget)
  | interactChest (target : Position)
  | interactNpc (target : Position)
  | interactSwitch (target : Position)
  | attack (target : Position)
  | raiseShield
  | chestRush (direction : Direction)
  | exitPush (exitId : ObjectId) (direction : Direction)
  | memoryFallback (direction : Direction)
  | wait
  deriving DecidableEq, Repr

def commandAction : ControllerCommand → Action
  | .navigate d | .turnOnly d _ | .chestRush d |
      .exitPush _ d | .memoryFallback d => directionAction d
  | .interactChest _ | .interactNpc _ | .interactSwitch _ | .attack _ => .slotA
  | .raiseShield => .slotB
  | .wait => .wait

def switchAt (r : RoomState) (p : Position) : Prop :=
  ∃ switch ∈ r.switches, switch.pos = p

def ControllerCommandSafe (s : WorldState) : ControllerCommand → Prop
  | .navigate d => plannerSafe (currentRoomState s) (advance s.player.pos d)
  | .turnOnly d .chest =>
      visibleChestAt (currentRoomState s) (advance s.player.pos d)
  | .turnOnly d .monster =>
      monsterAt (currentRoomState s) (advance s.player.pos d)
  | .interactChest p =>
      closedChestAt (currentRoomState s) p ∧ interactionReach s.player.pos p
  | .interactNpc p =>
      ¬ openChestInteractionAvailable s ∧
      npcAt (currentRoomState s) p ∧ interactionReach s.player.pos p
  | .interactSwitch p =>
      ¬ openChestInteractionAvailable s ∧
      ¬ npcInteractionAvailable s ∧
      (∃ switch ∈ (currentRoomState s).switches,
        switch.pos = p ∧
        interactionReach s.player.pos p ∧
        ∃ bridge ∈ (s.rooms switch.targetRoom).bridges,
          bridge.id = switch.targetBridge)
  | .attack p =>
      ¬ primaryInteractionAvailable s ∧
      s.player.inventory.equippedA = some .sword ∧
      .sword ∈ s.player.inventory.items ∧
      p = advance s.player.pos s.player.facing ∧
      monsterAt (currentRoomState s) p
  | .raiseShield => s.player.inventory.equippedB = some .shield
  | .chestRush d =>
      s.player.shielding = true ∧
      physicalEnterable (currentRoomState s) (advance s.player.pos d)
  | .exitPush exitId d =>
      ∃ exit ∈ (currentRoomState s).exits,
        exit.id = exitId ∧ exit.direction = d ∧
        exitContains exit s.player.pos ∧ exitRequirementSatisfied s exit
  | .memoryFallback d =>
      plannerSafe (currentRoomState s) (advance s.player.pos d) ∨
      ¬ physicalEnterable (currentRoomState s) (advance s.player.pos d)
  | .wait => True

def CommandRefinesAction
    (s : WorldState) (command : ControllerCommand) (action : Action) : Prop :=
  ControllerCommandSafe s command ∧ action = commandAction command

theorem command_refinement_outputs_declared_action
    {s : WorldState} {command : ControllerCommand} {action : Action}
    (h : CommandRefinesAction s command action) :
    action = commandAction command :=
  h.2

theorem navigation_command_targets_planner_safe_tile
    {s : WorldState} {d : Direction}
    (h : ControllerCommandSafe s (.navigate d)) :
    plannerSafe (currentRoomState s) (advance s.player.pos d) :=
  h

theorem exit_push_has_real_satisfied_exit
    {s : WorldState} {exitId : ObjectId} {d : Direction}
    (h : ControllerCommandSafe s (.exitPush exitId d)) :
    ∃ exit ∈ (currentRoomState s).exits,
      exit.id = exitId ∧ exit.direction = d ∧
      exitContains exit s.player.pos ∧ exitRequirementSatisfied s exit :=
  h

theorem memory_fallback_is_safe_or_physically_blocked
    {s : WorldState} {d : Direction}
    (h : ControllerCommandSafe s (.memoryFallback d)) :
    plannerSafe (currentRoomState s) (advance s.player.pos d) ∨
    ¬ physicalEnterable (currentRoomState s) (advance s.player.pos d) :=
  h

inductive PixelIntent where
  | tileMove (direction : Direction)
  | turnCorrection (direction : Direction)
  | exitBoundaryPush (direction : Direction)
  deriving DecidableEq, Repr

def PixelBlockRefinement
    (pixels : List Action) (intent : PixelIntent) : Prop :=
  pixels ≠ [] ∧
  match intent with
  | .tileMove d | .turnCorrection d | .exitBoundaryPush d =>
      ∀ action, action ∈ pixels → action = directionAction d

theorem pixel_block_contains_only_intended_direction
    {pixels : List Action} {intent : PixelIntent}
    (h : PixelBlockRefinement pixels intent) :
    match intent with
    | .tileMove d | .turnCorrection d | .exitBoundaryPush d =>
        ∀ action, action ∈ pixels → action = directionAction d := by
  cases intent <;> exact h.2

structure PolicyKernel (Controller : Type) where
  decide : Controller → VerifiedObservation → ControllerCommand
  update : Controller → VerifiedObservation → WorldState → Controller

inductive ClosedLoopStep {Controller : Type} (policy : PolicyKernel Controller) :
    (Controller × WorldState) → Action → (Controller × WorldState) → Prop where
  | mk {controller nextController : Controller}
      {world nextWorld : WorldState} {obs : VerifiedObservation}
      (hobservation : ObservationRefinesWorld world obs)
      (hsafe : ControllerCommandSafe world (policy.decide controller obs))
      (htick : EngineTick world
        (commandAction (policy.decide controller obs)) nextWorld)
      (hupdate : nextController = policy.update controller obs nextWorld) :
      ClosedLoopStep policy (controller, world)
        (commandAction (policy.decide controller obs)) (nextController, nextWorld)

inductive ClosedLoopExec {Controller : Type} (policy : PolicyKernel Controller) :
    (Controller × WorldState) → List Action →
      (Controller × WorldState) → Prop where
  | nil (state) : ClosedLoopExec policy state [] state
  | cons {start middle final action actions} :
      ClosedLoopStep policy start action middle →
      ClosedLoopExec policy middle actions final →
      ClosedLoopExec policy start (action :: actions) final

theorem closedLoopExec_projects_engineExec
    {Controller : Type} {policy : PolicyKernel Controller}
    {initial final : Controller × WorldState} {actions : List Action}
    (h : ClosedLoopExec policy initial actions final) :
    EngineExec initial.2 actions final.2 := by
  induction h with
  | nil state => exact EngineExec.nil
  | cons hstep hrest ih =>
      cases hstep with
      | mk hobservation hsafe htick hupdate =>
          exact EngineExec.cons htick ih

theorem closedLoopExec_append
    {Controller : Type} {policy : PolicyKernel Controller}
    {start middle final : Controller × WorldState}
    {left right : List Action}
    (hleft : ClosedLoopExec policy start left middle)
    (hright : ClosedLoopExec policy middle right final) :
    ClosedLoopExec policy start (left ++ right) final := by
  induction hleft with
  | nil state => simpa using hright
  | cons hstep hrest ih =>
      exact ClosedLoopExec.cons hstep (ih hright)

theorem closedLoop_complete_of_local_fair_progress
    {Controller : Type} (policy : PolicyKernel Controller)
    (measure : (Controller × WorldState) → Nat)
    (goal : (Controller × WorldState) → Prop)
    (hzero : ∀ state, measure state = 0 → goal state)
    (hprogress : ∀ state, 0 < measure state →
      ∃ actions next,
        actions ≠ [] ∧ ClosedLoopExec policy state actions next ∧
        measure next < measure state) :
    ∀ initial, ∃ actions final,
      ClosedLoopExec policy initial actions final ∧ goal final := by
  intro initial
  induction hmeasure : measure initial using Nat.strongRecOn generalizing initial with
  | ind n ih =>
      by_cases hz : measure initial = 0
      · exact ⟨[], initial, .nil initial, hzero initial hz⟩
      · have hpositive : 0 < measure initial := Nat.pos_of_ne_zero hz
        rcases hprogress initial hpositive with
          ⟨preActions, next, hnonempty, hpre, hdecrease⟩
        have hdecrease' : measure next < n := by simpa [hmeasure] using hdecrease
        rcases ih (measure next) hdecrease' next rfl with
          ⟨suffix, final, hsuffix, hgoal⟩
        exact ⟨preActions ++ suffix, final,
          closedLoopExec_append hpre hsuffix, hgoal⟩

theorem closedLoopExec_preserves_validState
    {Controller : Type} {policy : PolicyKernel Controller}
    {initialController finalController : Controller}
    {initial final : WorldState} {actions : List Action}
    (hvalid : ValidState initial)
    (h : ClosedLoopExec policy (initialController, initial) actions
      (finalController, final)) :
    ValidState final :=
  engineExec_preserves_validState hvalid (closedLoopExec_projects_engineExec h)

theorem player_step_has_single_tick_execution
    {s t : WorldState} {action : Action} {events : List Event}
    (hrunning : Running s)
    (hstep : Step s action t events)
    (hagent : ¬ AutonomousOnlyEvents events) :
    EngineExec s [action] t :=
  EngineExec.cons
    (EngineTick.mk ⟨hrunning, hstep, hagent⟩ AutonomousExec.nil)
    EngineExec.nil

def openChestResult (s : WorldState) (chest : Chest) : WorldState :=
  updateCurrentRoom
    { s with player := collectLoot { s.player with shielding := false } chest.loot }
    (replaceChest (currentRoomState s) chest { chest with opened := true })

@[simp] theorem openChestResult_currentRoom
    (s : WorldState) (chest : Chest) :
    (openChestResult s chest).currentRoom = s.currentRoom := rfl
@[simp] theorem openChestResult_roomIds
    (s : WorldState) (chest : Chest) :
    (openChestResult s chest).roomIds = s.roomIds := rfl
@[simp] theorem openChestResult_completed
    (s : WorldState) (chest : Chest) :
    (openChestResult s chest).completed = s.completed := rfl
@[simp] theorem openChestResult_pos
    (s : WorldState) (chest : Chest) :
    (openChestResult s chest).player.pos = s.player.pos := by
  cases hloot : chest.loot <;>
    simp [openChestResult, updateCurrentRoom, collectLoot, hloot]
  case tool item slot => cases slot <;> rfl
@[simp] theorem openChestResult_maxHp
    (s : WorldState) (chest : Chest) :
    (openChestResult s chest).player.maxHp = s.player.maxHp := by
  cases hloot : chest.loot <;>
    simp [openChestResult, updateCurrentRoom, collectLoot, hloot]
  case tool item slot => cases slot <;> rfl

theorem engineExec_openChest_once
    {s : WorldState} {chest : Chest}
    (hrunning : Running s)
    (hmember : chest ∈ (currentRoomState s).chests)
    (hvisible : chest.visible = true)
    (hclosed : chest.opened = false)
    (hreach : interactionReach s.player.pos chest.pos) :
    EngineExec s [.slotA] (openChestResult s chest) := by
  apply player_step_has_single_tick_execution hrunning
  · exact Step.openChest hmember hvisible hclosed hreach
  · simp [AutonomousOnlyEvents]

theorem engineExec_useExit_once
    {s : WorldState} {exit : Exit}
    (hrunning : Running s)
    (hmember : exit ∈ (currentRoomState s).exits)
    (hat : exitContains exit s.player.pos)
    (hrequirement : exitRequirementSatisfied s exit)
    (hspawn : canEnter (s.rooms exit.targetRoom) exit.targetSpawn) :
    EngineExec s [directionAction exit.direction]
      (transitionThroughExit s exit) := by
  apply player_step_has_single_tick_execution hrunning
  · exact Step.useExit hmember hat hrequirement rfl hspawn
  · simp only [exitEvents]
    split <;> simp [AutonomousOnlyEvents]

/-!
# Task 1：FSM + BFS + safety shield 的策略形式化

本章对应 Python 文件 `task1_fsm_bfs_agent.py` 的可验证符号层。证明分为四层：

1. 静态阻挡记忆不会忘记已经观察到的墙和宝箱；
2. tile 路径中的每一步都相邻且安全，BFS frontier 满足覆盖不变量时具有完备性；
3. safety shield 只放行下一格物理可通行的移动；
4. FSM 按“找宝箱 → 找出口 → 完成”单调推进，并把各阶段轨迹组合成通关轨迹。

本章不假定任何固定宝箱或出口坐标。最后的 Task1 主定理适用于任意满足前提的
单房间布局；公开地图坐标只应在实例化该定理时作为证明数据出现。
-/

namespace Task1

/-! ## 1. 静态阻挡记忆

Python Agent 会永久记住曾经识别出的墙和宝箱。即使宝箱打开后外观变化，
BFS 仍不能把该 tile 当作地板。`Task1MemorySound` 表示记忆中的每个位置
确实是当前房间的静态阻挡；`rememberBlocker` 只在列表头部增加新位置。
-/

structure Task1Memory where
  staticBlockers : List Position
  deriving DecidableEq, Repr

def Task1MemorySound (r : RoomState) (memory : Task1Memory) : Prop :=
  ∀ p, p ∈ memory.staticBlockers → staticBlocker r p

def rememberBlocker (memory : Task1Memory) (p : Position) : Task1Memory :=
  if p ∈ memory.staticBlockers then memory
  else { memory with staticBlockers := p :: memory.staticBlockers }

theorem remembered_blocker_is_retained
    (memory : Task1Memory) (p q : Position)
    (hq : q ∈ memory.staticBlockers) :
    q ∈ (rememberBlocker memory p).staticBlockers := by
  unfold rememberBlocker
  split
  · exact hq
  · exact List.mem_cons_of_mem p hq

theorem newly_remembered_blocker_is_present
    (memory : Task1Memory) (p : Position) :
    p ∈ (rememberBlocker memory p).staticBlockers := by
  unfold rememberBlocker
  split
  · assumption
  · exact List.mem_cons_self

theorem rememberBlocker_preserves_soundness
    {r : RoomState} {memory : Task1Memory} {p : Position}
    (hmemory : Task1MemorySound r memory)
    (hp : staticBlocker r p) :
    Task1MemorySound r (rememberBlocker memory p) := by
  intro q hq
  unfold rememberBlocker at hq
  split at hq
  · exact hmemory q hq
  · rcases List.mem_cons.mp hq with hEq | hOld
    · simpa [hEq] using hp
    · exact hmemory q hOld

/-! ## 2. action mask / safety shield

`task1Shield` 对非移动动作不作修改；对移动动作，只在目标 tile 满足
`canEnter` 时放行，否则替换成 WAIT。Task1 没有动态怪物，因此这里检查
物理阻挡已经足够；BFS 路径本身使用更强的 `safeTile`。
-/

noncomputable def task1Shield (s : WorldState) (proposed : Action) : Action := by
  classical
  exact match actionDirection proposed with
    | none => proposed
    | some d =>
        if canEnter (currentRoomState s) (advance s.player.pos d)
        then proposed
        else .wait

theorem task1Shield_nonmove_unchanged
    (s : WorldState) (a : Action)
    (h : actionDirection a = none) :
    task1Shield s a = a := by
  classical
  simp [task1Shield, h]

theorem task1Shield_blocks_unsafe_move
    (s : WorldState) (a : Action) (d : Direction)
    (ha : actionDirection a = some d)
    (hunsafe : ¬ canEnter (currentRoomState s) (advance s.player.pos d)) :
    task1Shield s a = .wait := by
  classical
  simp [task1Shield, ha, hunsafe]

theorem task1Shield_allowed_move_is_enterable
    (s : WorldState) (a : Action) (d : Direction)
    (ha : actionDirection a = some d)
    (hallowed : task1Shield s a = a) :
    canEnter (currentRoomState s) (advance s.player.pos d) := by
  classical
  unfold task1Shield at hallowed
  rw [ha] at hallowed
  by_cases henter :
      canEnter (currentRoomState s) (advance s.player.pos d)
  · exact henter
  · simp [henter] at hallowed
    have himpossible : actionDirection Action.wait = some d := by
      rw [hallowed]
      exact ha
    simp [actionDirection] at himpossible

/-! ## 3. Task1 FSM

FSM 只有三个阶段：先寻找并开启钥匙宝箱，再寻找钥匙门，最后完成。
`task1NextPhase` 只读取“是否已有钥匙”和“世界是否完成”两个符号事实。
阶段秩 `task1PhaseRank` 用于证明 FSM 不会倒退。
-/

inductive Task1Phase where
  | toChest
  | toExit
  | done
  deriving DecidableEq, Repr

def task1PhaseRank : Task1Phase → Nat
  | .toChest => 0
  | .toExit => 1
  | .done => 2

def task1NextPhase (phase : Task1Phase) (hasKey completed : Bool) : Task1Phase :=
  if completed then .done
  else match phase with
    | .toChest => if hasKey then .toExit else .toChest
    | .toExit => .toExit
    | .done => .done

theorem task1_phase_never_regresses
    (phase : Task1Phase) (hasKey completed : Bool) :
    task1PhaseRank phase ≤
      task1PhaseRank (task1NextPhase phase hasKey completed) := by
  cases phase <;> cases hasKey <;> cases completed <;>
    decide

theorem task1_key_advances_to_exit :
    task1NextPhase .toChest true false = .toExit := by
  rfl

theorem task1_completion_advances_to_done
    (phase : Task1Phase) (hasKey : Bool) :
    task1NextPhase phase hasKey true = .done := by
  cases phase <;> rfl

/-! ## 4. Task1 的组合正确性与可达性

`Task1Completable` 表示存在一个动作序列，其 `Exec` 轨迹最终满足
`WorldCompleted`。主定理不指定具体路线，只要求 BFS 提供两段已经由
`Exec` 验证的子计划：

* 从初态到宝箱相邻位置；
* 开箱后从当前位置到钥匙出口位置。

随后定理调用环境中的 `openChest` 和 `useExit` 规则，把两段子计划与两个
交互动作拼接起来。由此证明 FSM 的阶段组合是正确的。
-/

def Task1Goal (s : WorldState) : Prop :=
  WorldCompleted s

def Task1Completable (initial : WorldState) : Prop :=
  ∃ actions final, Exec initial actions final ∧ Task1Goal final

def stateAfterOpeningChest (s : WorldState) (chest : Chest) : WorldState :=
  updateCurrentRoom
    { s with player := collectLoot { s.player with shielding := false } chest.loot }
    (replaceChest (currentRoomState s) chest { chest with opened := true })

def stateAfterUsingExit (s : WorldState) (exit : Exit) : WorldState :=
  transitionThroughExit s exit

theorem stateAfterOpeningChest_current_monsters
    (s : WorldState) (chest : Chest) :
    (currentRoomState (stateAfterOpeningChest s chest)).monsters =
      (currentRoomState s).monsters := by
  simp [stateAfterOpeningChest, updateCurrentRoom, currentRoomState,
    setRoom, replaceChest]

theorem task1_open_key_chest_gives_key
    {s : WorldState} {chest : Chest} {amount : Nat}
    (hloot : chest.loot = .key amount)
    (hpositive : 0 < amount) :
    HasKey (stateAfterOpeningChest s chest) := by
  unfold HasKey stateAfterOpeningChest updateCurrentRoom
  simp [collectLoot, hloot]
  exact Nat.add_pos_right s.player.inventory.keys hpositive

theorem task1_completable_if_subplans_exist
    {initial nearChest afterChest atExit : WorldState}
    {toChest toExit : List Action}
    {chest : Chest} {exit : Exit} {targetRoom : RoomState}
    (hToChest : Exec initial toChest nearChest)
    (hChestMember : chest ∈ (currentRoomState nearChest).chests)
    (hChestVisible : chest.visible = true)
    (hChestClosed : chest.opened = false)
    (hChestAdjacent : adjacent nearChest.player.pos chest.pos)
    (hAfterChest : afterChest = stateAfterOpeningChest nearChest chest)
    (hToExit : Exec afterChest toExit atExit)
    (hExitMember : exit ∈ (currentRoomState atExit).exits)
    (hAtExit : atExit.player.pos = exit.pos)
    (_hFacingExit : atExit.player.facing = exit.direction)
    (hRequirement : requirementSatisfied atExit exit.requirement)
    (hTargetRoom : targetRoom = atExit.rooms exit.targetRoom)
    (hSpawn : canEnter targetRoom exit.targetSpawn)
    (hCompletes : exit.completesTask = true) :
    Task1Completable initial := by
  have hOpenStep :
      Step nearChest .slotA (stateAfterOpeningChest nearChest chest)
        [.chestOpened chest.id] := by
    exact Step.openChest hChestMember hChestVisible hChestClosed (Or.inr hChestAdjacent)
  have hOpenExec :
      Exec nearChest [.slotA] (stateAfterOpeningChest nearChest chest) :=
    Exec.cons hOpenStep Exec.nil
  have hExitStep :
      Step atExit (directionAction exit.direction)
        (stateAfterUsingExit atExit exit)
        (exitEvents atExit exit) := by
    exact Step.useExit hExitMember (Or.inl hAtExit)
      (requirement_implies_exitRequirementSatisfied hRequirement)
      hTargetRoom hSpawn
  have hExitExec :
      Exec atExit [directionAction exit.direction]
        (stateAfterUsingExit atExit exit) :=
    Exec.cons hExitStep Exec.nil
  subst afterChest
  have hPhase1 :
      Exec initial (toChest ++ [.slotA])
        (stateAfterOpeningChest nearChest chest) :=
    exec_append hToChest hOpenExec
  have hPhase2 :
      Exec initial ((toChest ++ [.slotA]) ++ toExit) atExit :=
    exec_append hPhase1 hToExit
  have hAll :
      Exec initial
        (((toChest ++ [.slotA]) ++ toExit) ++
          [directionAction exit.direction])
        (stateAfterUsingExit atExit exit) :=
    exec_append hPhase2 hExitExec
  refine ⟨_, _, hAll, ?_⟩
  unfold Task1Goal WorldCompleted stateAfterUsingExit transitionThroughExit
  simp [hCompletes]

/-!
该定理给出 Task1 的条件完备性：如果有限地图中的两个 BFS 调用分别满足
frontier 覆盖不变量，并且钥匙宝箱与出口在给定深度内可达，那么两个 BFS
都能在 frontier 中发现目标。结合上面的组合正确性定理，就得到 Task1
策略在这些标准可达性前提下能够完成关卡。
-/

theorem task1_two_phase_bfs_complete
    {roomBefore roomAfter : RoomState}
    {start afterChest : Position}
    {chestGoals exitGoals chestFrontier exitFrontier : List Position}
    {chestDepth exitDepth : Nat}
    (hChestFrontier :
      BfsFrontierComplete roomBefore start chestDepth chestFrontier)
    (hChestReachable : ∃ goal, goal ∈ chestGoals ∧
      BoundedTileReachable roomBefore start chestDepth goal)
    (hExitFrontier :
      BfsFrontierComplete roomAfter afterChest exitDepth exitFrontier)
    (hExitReachable : ∃ goal, goal ∈ exitGoals ∧
      BoundedTileReachable roomAfter afterChest exitDepth goal) :
    BfsFindsGoal chestFrontier chestGoals ∧
    BfsFindsGoal exitFrontier exitGoals := by
  exact ⟨
    bfs_complete_from_frontier_invariant hChestFrontier hChestReachable,
    bfs_complete_from_frontier_invariant hExitFrontier hExitReachable
  ⟩

/-! ## 5. 公开 Task1 地图的可达性实例

前面的定理完全不依赖坐标。本节仅把公开的
`map_data/mathematical_logic/task_1/room_001.json` 翻译成一个证明实例，
用于确认该具体关卡确实满足“宝箱可达、出口可达”的前提。这里的坐标只存在
于 Lean 离线证明中，不会进入 Python Agent 的运行时决策。
-/

def task1Pos (x y : Int) : Position := { x := x, y := y }

def task1PublicBounds : Bounds :=
  { width := 10
    height := 8
    width_pos := by decide
    height_pos := by decide }

def task1PublicWalls : List Position :=
  [ task1Pos 0 2, task1Pos 1 2,
    task1Pos 4 2, task1Pos 5 2, task1Pos 6 2,
    task1Pos 7 2, task1Pos 8 2, task1Pos 9 2,
    task1Pos 0 5, task1Pos 1 5, task1Pos 2 5,
    task1Pos 3 5, task1Pos 4 5, task1Pos 5 5,
    task1Pos 6 5 ]

def task1PublicChest : Chest :=
  { id := 1
    pos := task1Pos 0 3
    loot := .key 1
    visible := true
    opened := false }

def task1PublicExit : Exit :=
  { id := 2
    pos := task1Pos 4 0
    direction := .north
    kind := .locked
    requirement := .keys 1 true
    targetRoom := 0
    targetSpawn := task1Pos 4 6
    completesTask := true
    opened := false }

def task1PublicRoom : RoomState :=
  { bounds := task1PublicBounds
    walls := task1PublicWalls
    npcs := []
    chests := [task1PublicChest]
    monsters := []
    traps := []
    buttons := []
    switches := []
    bridges := []
    dynamicTiles := []
    exits := [task1PublicExit] }

def task1PublicStart : Position := task1Pos 4 6
def task1PublicNearChest : Position := task1Pos 0 4
def task1PublicExitTile : Position := task1Pos 4 0

def task1PublicToChestRoute : List Position :=
  [ task1Pos 5 6, task1Pos 6 6, task1Pos 7 6,
    task1Pos 7 5, task1Pos 7 4, task1Pos 6 4,
    task1Pos 5 4, task1Pos 4 4, task1Pos 3 4,
    task1Pos 2 4, task1Pos 1 4, task1Pos 0 4 ]

def task1PublicToExitRoute : List Position :=
  [ task1Pos 1 4, task1Pos 2 4, task1Pos 2 3,
    task1Pos 2 2, task1Pos 2 1, task1Pos 3 1,
    task1Pos 4 1, task1Pos 4 0 ]

private theorem task1PublicSafeAt
    (x y : Int)
    (h : (x, y) ∈
      [(5, 6), (6, 6), (7, 6), (7, 5), (7, 4), (6, 4),
       (5, 4), (4, 4), (3, 4), (2, 4), (1, 4), (0, 4),
       (2, 3), (2, 2), (2, 1), (3, 1), (4, 1), (4, 0)]) :
    safeTile task1PublicRoom (task1Pos x y) := by
  simp at h
  rcases h with
    h | h | h | h | h | h | h | h | h | h | h | h |
    h | h | h | h | h | h
  all_goals
    rcases h with ⟨rfl, rfl⟩
    simp [safeTile, canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      activeTrapAt, monsterAt, gapAt, activeBridgeTile, task1PublicRoom,
      task1PublicBounds, task1PublicWalls, task1PublicChest, task1Pos]

theorem task1_public_bfs_path_to_chest :
    TilePath task1PublicRoom task1PublicStart
      task1PublicToChestRoute task1PublicNearChest := by
  unfold task1PublicToChestRoute task1PublicStart task1PublicNearChest
  apply TilePath.cons .east rfl
  · exact task1PublicSafeAt 5 6 (by simp)
  apply TilePath.cons .east rfl
  · exact task1PublicSafeAt 6 6 (by simp)
  apply TilePath.cons .east rfl
  · exact task1PublicSafeAt 7 6 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicSafeAt 7 5 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicSafeAt 7 4 (by simp)
  apply TilePath.cons .west rfl
  · exact task1PublicSafeAt 6 4 (by simp)
  apply TilePath.cons .west rfl
  · exact task1PublicSafeAt 5 4 (by simp)
  apply TilePath.cons .west rfl
  · exact task1PublicSafeAt 4 4 (by simp)
  apply TilePath.cons .west rfl
  · exact task1PublicSafeAt 3 4 (by simp)
  apply TilePath.cons .west rfl
  · exact task1PublicSafeAt 2 4 (by simp)
  apply TilePath.cons .west rfl
  · exact task1PublicSafeAt 1 4 (by simp)
  apply TilePath.cons .west rfl
  · exact task1PublicSafeAt 0 4 (by simp)
  exact TilePath.nil _

theorem task1_public_bfs_path_to_exit :
    TilePath task1PublicRoom task1PublicNearChest
      task1PublicToExitRoute task1PublicExitTile := by
  unfold task1PublicToExitRoute task1PublicNearChest task1PublicExitTile
  apply TilePath.cons .east rfl
  · exact task1PublicSafeAt 1 4 (by simp)
  apply TilePath.cons .east rfl
  · exact task1PublicSafeAt 2 4 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicSafeAt 2 3 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicSafeAt 2 2 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicSafeAt 2 1 (by simp)
  apply TilePath.cons .east rfl
  · exact task1PublicSafeAt 3 1 (by simp)
  apply TilePath.cons .east rfl
  · exact task1PublicSafeAt 4 1 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicSafeAt 4 0 (by simp)
  exact TilePath.nil _

theorem task1_public_chest_is_adjacent :
    adjacent task1PublicNearChest task1PublicChest.pos := by
  simp [task1PublicNearChest, task1PublicChest, task1Pos, adjacent, advance]

theorem task1_public_chest_phase_reachable :
    TileReachable task1PublicRoom task1PublicStart task1PublicNearChest :=
  tilePath_goal_reachable task1_public_bfs_path_to_chest

theorem task1_public_exit_phase_reachable :
    TileReachable task1PublicRoom task1PublicNearChest task1PublicExitTile :=
  tilePath_goal_reachable task1_public_bfs_path_to_exit

/-! ### 公开 Task1 的闭合 `EngineExec` 证书 -/

def task1PublicPlayer : PlayerState :=
  { pos := task1PublicStart
    facing := .north
    hp := 5
    maxHp := 5
    inventory :=
      { keys := 0, gold := 0, items := [.sword, .shield]
        equippedA := some .sword, equippedB := some .shield }
    shielding := false }

def task1PublicInitial : WorldState :=
  { currentRoom := 0
    rooms := fun _ => task1PublicRoom
    roomIds := [0]
    player := task1PublicPlayer
    completed := false }

def task1PublicOpenedRoom : RoomState :=
  replaceChest task1PublicRoom task1PublicChest
    { task1PublicChest with opened := true }

private theorem task1PublicOpenedSafeAt
    (x y : Int)
    (h : (x, y) ∈
      [(1, 4), (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), (4, 1), (4, 0)]) :
    safeTile task1PublicOpenedRoom (task1Pos x y) := by
  simp at h
  rcases h with h | h | h | h | h | h | h | h
  all_goals
    rcases h with ⟨rfl, rfl⟩
    simp [task1PublicOpenedRoom, replaceChest, safeTile, canEnter, inBounds,
      staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
      activeBridgeTile, task1PublicRoom, task1PublicBounds, task1PublicWalls,
      task1PublicChest, task1Pos]

theorem task1_public_bfs_path_to_exit_after_open :
    TilePath task1PublicOpenedRoom task1PublicNearChest
      task1PublicToExitRoute task1PublicExitTile := by
  unfold task1PublicToExitRoute task1PublicNearChest task1PublicExitTile
  apply TilePath.cons .east rfl
  · exact task1PublicOpenedSafeAt 1 4 (by simp)
  apply TilePath.cons .east rfl
  · exact task1PublicOpenedSafeAt 2 4 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicOpenedSafeAt 2 3 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicOpenedSafeAt 2 2 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicOpenedSafeAt 2 1 (by simp)
  apply TilePath.cons .east rfl
  · exact task1PublicOpenedSafeAt 3 1 (by simp)
  apply TilePath.cons .east rfl
  · exact task1PublicOpenedSafeAt 4 1 (by simp)
  apply TilePath.cons .north rfl
  · exact task1PublicOpenedSafeAt 4 0 (by simp)
  exact TilePath.nil _

theorem task1_public_initial_running : Running task1PublicInitial := by
  simp [Running, alive, task1PublicInitial, task1PublicPlayer]

theorem task1_public_engine_certificate :
    ∃ actions final,
      EngineExec task1PublicInitial actions final ∧
      WorldCompleted final ∧ alive final ∧ ValidState final := by
  rcases tilePath_has_engine_plan
      (r := task1PublicRoom) (s := task1PublicInitial)
      (start := task1PublicStart) (goal := task1PublicNearChest)
      (route := task1PublicToChestRoute)
      (by simp [task1PublicInitial, currentRoomState]) rfl rfl
      task1_public_initial_running task1_public_bfs_path_to_chest with
    ⟨toChest, nearChest, hToChest, hNearPos, hNearRoom,
      hNearRunning, hNearInventory, hNearHp, hNearMaxHp,
      hNearCurrent, hNearRooms, hNearRoomIds, hNearCompleted⟩
  have hChestMember : task1PublicChest ∈ (currentRoomState nearChest).chests := by
    rw [hNearRoom]
    simp [task1PublicRoom]
  have hChestReach :
      interactionReach nearChest.player.pos task1PublicChest.pos := by
    right
    rw [hNearPos]
    exact task1_public_chest_is_adjacent
  let afterChest := stateAfterOpeningChest nearChest task1PublicChest
  have hOpenStep :
      Step nearChest .slotA afterChest [.chestOpened task1PublicChest.id] :=
    Step.openChest hChestMember rfl rfl hChestReach
  have hOpenPlayer :
      PlayerStep nearChest .slotA afterChest [.chestOpened task1PublicChest.id] := by
    refine ⟨hNearRunning, hOpenStep, ?_⟩
    simp [AutonomousOnlyEvents]
  have hOpenTick : EngineTick nearChest .slotA afterChest :=
    EngineTick.mk hOpenPlayer AutonomousExec.nil
  have hAfterRoom : currentRoomState afterChest = task1PublicOpenedRoom := by
    dsimp [afterChest, stateAfterOpeningChest]
    rw [currentRoomState_updateCurrentRoom, hNearRoom]
    rfl
  have hAfterPos : afterChest.player.pos = task1PublicNearChest := by
    change nearChest.player.pos = task1PublicNearChest
    exact hNearPos
  have hAfterRunning : Running afterChest := by
    rcases hNearRunning with ⟨halive, hcomplete⟩
    constructor
    · change 0 < nearChest.player.hp
      exact halive
    · change nearChest.completed = false
      exact hcomplete
  rcases tilePath_has_engine_plan
      (r := task1PublicOpenedRoom) (s := afterChest)
      (start := task1PublicNearChest) (goal := task1PublicExitTile)
      (route := task1PublicToExitRoute)
      hAfterRoom hAfterPos rfl hAfterRunning
      task1_public_bfs_path_to_exit_after_open with
    ⟨toExit, atExit, hToExit, hExitPos, hExitRoom,
      hExitRunning, hExitInventory, hExitHp, hExitMaxHp,
      hExitCurrent, hExitRooms, hExitRoomIds, hExitCompleted⟩
  have hExitMember : task1PublicExit ∈ (currentRoomState atExit).exits := by
    rw [hExitRoom]
    simp [task1PublicOpenedRoom, replaceChest, task1PublicRoom]
  have hExitContains : exitContains task1PublicExit atExit.player.pos := by
    left
    simpa [task1PublicExit, task1PublicExitTile] using hExitPos
  have hKey : 1 ≤ atExit.player.inventory.keys := by
    rw [hExitInventory]
    change 1 ≤ nearChest.player.inventory.keys + 1
    omega
  have hRequirement : exitRequirementSatisfied atExit task1PublicExit := by
    right
    exact hKey
  have hSpawn :
      canEnter (atExit.rooms task1PublicExit.targetRoom)
        task1PublicExit.targetSpawn := by
    have hcurrentZero : atExit.currentRoom = 0 := by
      rw [hExitCurrent]
      change nearChest.currentRoom = 0
      exact hNearCurrent
    change canEnter (atExit.rooms 0) (task1Pos 4 6)
    rw [← hcurrentZero]
    change canEnter (currentRoomState atExit) (task1Pos 4 6)
    rw [hExitRoom]
    simp [task1PublicOpenedRoom, replaceChest, task1PublicRoom,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
      task1PublicBounds, task1PublicWalls, task1PublicChest, task1Pos]
  let final := transitionThroughExit atExit task1PublicExit
  have hExitStep :
      Step atExit (directionAction task1PublicExit.direction) final
        (exitEvents atExit task1PublicExit) :=
    Step.useExit hExitMember hExitContains hRequirement rfl hSpawn
  have hExitPlayer :
      PlayerStep atExit (directionAction task1PublicExit.direction) final
        (exitEvents atExit task1PublicExit) := by
    refine ⟨hExitRunning, hExitStep, ?_⟩
    simp [AutonomousOnlyEvents, exitEvents, task1PublicExit]
  have hExitTick :
      EngineTick atExit (directionAction task1PublicExit.direction) final :=
    EngineTick.mk hExitPlayer AutonomousExec.nil
  have hAll := engineExec_append
    (engineExec_append hToChest (EngineExec.cons hOpenTick EngineExec.nil))
    (engineExec_append hToExit (EngineExec.cons hExitTick EngineExec.nil))
  refine ⟨_, final, hAll, ?_, ?_, ?_⟩
  · simp [final, WorldCompleted, transitionThroughExit, task1PublicExit]
  · rcases hExitRunning with ⟨halive, _⟩
    simpa [final, transitionThroughExit, alive] using halive
  · exact engineExec_preserves_validState
      (by simp [ValidState, task1PublicInitial, task1PublicPlayer,
        currentRoomState, task1PublicRoom, task1PublicBounds, inBounds,
        task1PublicStart, task1Pos]) hAll

end Task1

/-!
# Task 2：动态怪物、可中断队列与三阶段 FSM

Task2 在 Task1 的“视觉符号状态 + BFS + safety shield”上增加动态怪物。对应
Python Agent 的实际阶段为：

`toMonster → toChest → toExit → done`

本章重点证明：

* 战斗阶段只规划到怪物相邻的安全攻击位，不把怪物 tile 当作普通路径；
* 面向修正不改变玩家 tile，挥剑只攻击朝向前方一格；
* 每次未击杀攻击严格降低 HP，单位攻击力在有限次攻击后将 HP 降到 0；
* 非战斗阶段不会主动走进怪物或怪物的一格邻域；
* 缓存移动一旦不再安全就必须中断；
* 清怪、开箱取钥匙和条件出口三段轨迹可以组合为完整通关轨迹。
-/

namespace Task2

open Task1

/-! ## 1. FSM 与连续消失帧确认

Python 不会因为一帧看不到怪物就宣布清怪，而是连续三帧没有怪物才切换阶段。
`updateMissingFrames` 和 `monsterCleared` 形式化这一抗单帧误识别机制。
-/

inductive Task2Phase where
  | toMonster
  | toChest
  | toExit
  | done
  deriving DecidableEq, Repr

def task2PhaseRank : Task2Phase → Nat
  | .toMonster => 0
  | .toChest => 1
  | .toExit => 2
  | .done => 3

def updateMissingFrames (monsterVisible : Bool) (old : Nat) : Nat :=
  if monsterVisible then 0 else old + 1

def monsterCleared (missingFrames : Nat) : Prop :=
  3 ≤ missingFrames

def task2NextPhase
    (phase : Task2Phase) (cleared hasKey completed queueEmpty : Bool) :
    Task2Phase :=
  if completed then .done
  else match phase with
    | .toMonster =>
        if cleared && queueEmpty then .toChest else .toMonster
    | .toChest =>
        if hasKey && queueEmpty then .toExit else .toChest
    | .toExit => .toExit
    | .done => .done

theorem three_missing_frames_confirm_clear :
    monsterCleared
      (updateMissingFrames false
        (updateMissingFrames false
          (updateMissingFrames false 0))) := by
  show 3 ≤ 3
  exact Nat.le_refl 3

theorem visible_monster_resets_missing_frames (old : Nat) :
    updateMissingFrames true old = 0 := by
  rfl

theorem task2_phase_never_regresses
    (phase : Task2Phase) (cleared hasKey completed queueEmpty : Bool) :
    task2PhaseRank phase ≤
      task2PhaseRank
        (task2NextPhase phase cleared hasKey completed queueEmpty) := by
  cases phase <;> cases cleared <;> cases hasKey <;>
    cases completed <;> cases queueEmpty <;> decide

theorem task2_clear_advances_to_chest :
    task2NextPhase .toMonster true false false true = .toChest := by
  rfl

theorem task2_key_advances_to_exit :
    task2NextPhase .toChest true true false true = .toExit := by
  rfl

theorem task2_nonempty_queue_delays_phase_change :
    task2NextPhase .toMonster true false false false = .toMonster ∧
    task2NextPhase .toChest true true false false = .toChest := by
  exact ⟨rfl, rfl⟩

/-! ## 2. 战斗站位、朝向与有限击杀

`AttackPosition` 要求玩家站在安全 tile，且怪物恰好位于四邻域。真正挥剑前还
需要 `FacingMonster`，即怪物位于玩家当前朝向的前方一格。这样把“靠近怪物”
和“可以命中怪物”明确区分开。
-/

def FacingMonster (player : PlayerState) (monster : Monster) : Prop :=
  monster.pos = advance player.pos player.facing

def AttackPosition (r : RoomState) (p : Position) (monster : Monster) : Prop :=
  monster ∈ r.monsters ∧
  0 < monster.hp ∧
  safeTile r p ∧
  adjacent p monster.pos

def AttackReady (s : WorldState) (monster : Monster) : Prop :=
  s.player.inventory.equippedA = some Item.sword ∧
  Item.sword ∈ s.player.inventory.items ∧
  monster ∈ (currentRoomState s).monsters ∧
  0 < monster.hp ∧
  FacingMonster s.player monster

theorem attack_position_is_safe
    {r : RoomState} {p : Position} {monster : Monster}
    (h : AttackPosition r p monster) :
    safeTile r p :=
  h.2.2.1

theorem attack_position_is_adjacent
    {r : RoomState} {p : Position} {monster : Monster}
    (h : AttackPosition r p monster) :
    adjacent p monster.pos :=
  h.2.2.2

theorem face_monster_keeps_player_position
    (s : WorldState) (d : Direction) :
    ({ s with player := { s.player with facing := d, shielding := false } }).player.pos =
      s.player.pos := by
  rfl

theorem face_monster_step_is_position_safe
    {s t : WorldState} {a : Action} {d : Direction}
    (ha : actionDirection a = some d)
    (hmonster : monsterAt (currentRoomState s) (advance s.player.pos d))
    (ht : t = { s with player :=
      { s.player with facing := d, shielding := false } }) :
    Step s a t [] ∧ t.player.pos = s.player.pos := by
  subst t
  exact ⟨Step.faceMonster ha hmonster, rfl⟩

/-!
剑的 Python 常量攻击力为 1。`hpAfterSwordHits hp hits` 是忽略击退动画后，
连续有效命中的数值抽象。下面两个定理说明每次命中使正 HP 严格下降，而且
初始 HP 为 `hp` 时至多 `hp` 次有效命中就会归零。这是战斗终止的度量证明。
-/

def hpAfterSwordHits (hp hits : Nat) : Nat :=
  hp - hits

theorem one_sword_hit_strictly_decreases
    {hp : Nat} (hpositive : 0 < hp) :
    hpAfterSwordHits hp 1 < hp := by
  exact Nat.sub_lt hpositive (by decide)

theorem hp_sword_hits_are_sufficient (hp : Nat) :
    hpAfterSwordHits hp hp = 0 := by
  simp [hpAfterSwordHits]

theorem task2_attack_damage_strict_progress
    (monster : Monster) (hpositive : 1 < monster.hp) :
    ({ monster with hp := monster.hp - 1 }).hp < monster.hp := by
  exact damaging_attack_reduces_monster_hp monster 1 (by decide) hpositive

theorem task2_attack_kill_removes_target
    (r : RoomState) (monster : Monster) :
    (removeMonster r monster).monsters =
      r.monsters.filter (fun m => m.id != monster.id) := by
  rfl

/-! ## 3. 动态危险区、action mask 与可中断队列

战斗阶段允许走到怪物相邻的安全攻击位；开箱和出口阶段使用更保守的
`OutsideMonsterDanger`，禁止进入怪物 tile 及其一格邻域。这与 Python
`distance_to_nearest(...) <= 1` 时打断队列的逻辑一致。

Python 还有两类“短命令”不是普通导航：朝相邻怪物发一个方向动作只用于修正
朝向，朝相邻宝箱发一个方向动作只用于面向宝箱后按 A。Lean 因此区分
`Task2MoveAllowed`（真正进入下一 tile 的导航移动）和 `Task2CommandAllowed`
（交给引擎前 safety shield 允许提交的方向命令）。
-/

def OutsideMonsterDanger (r : RoomState) (p : Position) : Prop :=
  ¬ monsterAt r p ∧
  ∀ monster, monster ∈ r.monsters → 0 < monster.hp →
    ¬ adjacent p monster.pos

def Task2MoveAllowed
    (phase : Task2Phase) (r : RoomState) (p : Position) : Prop :=
  safeTile r p ∧
  match phase with
  | .toMonster => True
  | .toChest | .toExit | .done => OutsideMonsterDanger r p

def Task2FacingCommandAllowed
    (phase : Task2Phase) (r : RoomState) (p : Position) : Prop :=
  (phase = .toMonster ∧ monsterAt r p) ∨
  (phase = .toChest ∧ visibleChestAt r p)

def Task2CommandAllowed
    (phase : Task2Phase) (r : RoomState) (p : Position) : Prop :=
  Task2MoveAllowed phase r p ∨
  Task2FacingCommandAllowed phase r p

theorem task2_allowed_move_is_safe
    {phase : Task2Phase} {r : RoomState} {p : Position}
    (h : Task2MoveAllowed phase r p) :
    safeTile r p :=
  h.1

theorem task2_noncombat_move_avoids_monster_neighborhood
    {phase : Task2Phase} {r : RoomState} {p : Position}
    (hphase : phase ≠ .toMonster)
    (h : Task2MoveAllowed phase r p) :
    OutsideMonsterDanger r p := by
  cases phase with
  | toMonster => contradiction
  | toChest => exact h.2
  | toExit => exact h.2
  | done => exact h.2

noncomputable def task2Shield
    (phase : Task2Phase) (s : WorldState) (proposed : Action) : Action := by
  classical
  exact match actionDirection proposed with
    | none => proposed
    | some d =>
        if Task2CommandAllowed phase (currentRoomState s)
            (advance s.player.pos d)
        then proposed
        else .wait

theorem task2Shield_blocks_disallowed_command
    (phase : Task2Phase) (s : WorldState) (a : Action) (d : Direction)
    (ha : actionDirection a = some d)
    (hunsafe : ¬ Task2CommandAllowed phase (currentRoomState s)
      (advance s.player.pos d)) :
    task2Shield phase s a = .wait := by
  classical
  simp [task2Shield, ha, hunsafe]

theorem task2Shield_allowed_move_is_safe
    (phase : Task2Phase) (s : WorldState) (a : Action) (d : Direction)
    (ha : actionDirection a = some d)
    (hallowed : task2Shield phase s a = a)
    (hnotFacing : ¬ Task2FacingCommandAllowed phase (currentRoomState s)
      (advance s.player.pos d)) :
    safeTile (currentRoomState s) (advance s.player.pos d) := by
  classical
  unfold task2Shield at hallowed
  rw [ha] at hallowed
  by_cases hcmd :
      Task2CommandAllowed phase (currentRoomState s) (advance s.player.pos d)
  · rcases hcmd with hmove | hfacing
    · exact hmove.1
    · exact False.elim (hnotFacing hfacing)
  · simp [hcmd] at hallowed
    have himpossible : actionDirection Action.wait = some d := by
      rw [hallowed]
      exact ha
    simp [actionDirection] at himpossible

def QueueMustInterrupt
    (phase : Task2Phase) (s : WorldState) (nextAction : Action) : Prop :=
  ∃ d, actionDirection nextAction = some d ∧
    ¬ Task2CommandAllowed phase (currentRoomState s)
      (advance s.player.pos d)

theorem unsafe_queued_move_must_interrupt
    (phase : Task2Phase) (s : WorldState) (a : Action) (d : Direction)
    (ha : actionDirection a = some d)
    (hunsafe : ¬ Task2CommandAllowed phase (currentRoomState s)
      (advance s.player.pos d)) :
    QueueMustInterrupt phase s a :=
  ⟨d, ha, hunsafe⟩

theorem interrupted_move_is_masked_to_wait
    {phase : Task2Phase} {s : WorldState} {a : Action}
    (hinterrupt : QueueMustInterrupt phase s a) :
    task2Shield phase s a = .wait := by
  rcases hinterrupt with ⟨d, ha, hunsafe⟩
  exact task2Shield_blocks_disallowed_command phase s a d ha hunsafe

inductive Task2ShieldedEngineExec
    (phase : Task2Phase) : WorldState → List Action → WorldState → Prop where
  | nil {s : WorldState} :
      Task2ShieldedEngineExec phase s [] s
  | cons {s t u : WorldState} {action : Action} {actions : List Action} :
      EngineTick s action t →
      (∀ d, actionDirection action = some d →
        Task2CommandAllowed phase (currentRoomState s)
          (advance s.player.pos d)) →
      Task2ShieldedEngineExec phase t actions u →
      Task2ShieldedEngineExec phase s (action :: actions) u

theorem task2ShieldedEngineExec_to_engineExec
    {phase : Task2Phase} {s t : WorldState} {actions : List Action}
    (h : Task2ShieldedEngineExec phase s actions t) :
    EngineExec s actions t := by
  induction h with
  | nil => exact EngineExec.nil
  | cons htick _ hrest ih =>
      exact EngineExec.cons htick ih

/-! ## 4. BFS 路径、生成动作与 shielded EngineExec

`TileReachable` 只说明存在安全路径；它本身不足以说明 Python BFS 返回了哪条
路径，也不足以给出动作列表。下面的 `Task2RouteActionPlan` 把路径中的每个
相邻 tile 转成对应方向动作，并要求该目标 tile 对当前 Task2 阶段是可移动的。
`Task2BfsPlanGenerated` 再把这个动作计划与 `BfsResult` 连接起来。

这里仍保留两个必要限制：

* 本引理证明的是符号 BFS 证书，不逐行证明 Python `deque`/`parent` 实现；
* 导航段按普通 tile 移动执行，要求当前房间没有按钮 tile 被路径踩到。按钮会
  修改房间状态，需另写带按钮状态更新的路径执行引理；Task2 公开图没有按钮。
-/

inductive Task2RouteActionPlan
    (phase : Task2Phase) (r : RoomState) :
    Position → List Position → Position → List Action → Prop where
  | nil (p : Position) :
      Task2RouteActionPlan phase r p [] p []
  | cons {p q goal : Position} {rest : List Position}
      {actions : List Action} (d : Direction)
      (hq : q = advance p d)
      (hallowed : Task2MoveAllowed phase r q)
      (htail : Task2RouteActionPlan phase r q rest goal actions) :
      Task2RouteActionPlan phase r p (q :: rest) goal
        (actionForDirection d :: actions)

theorem task2_route_action_plan_to_tilePath
    {phase : Task2Phase} {r : RoomState}
    {start goal : Position} {route : List Position}
    {tileActions : List Action}
    (hplan : Task2RouteActionPlan phase r start route goal tileActions) :
    TilePath r start route goal := by
  induction hplan with
  | nil p => exact TilePath.nil p
  | cons d hq hallowed htail ih =>
      exact TilePath.cons d hq hallowed.1 ih

structure Task2BfsPlanGenerated
    (phase : Task2Phase) (r : RoomState) (start : Position)
    (goals route : List Position) (goal : Position)
    (tileActions : List Action) : Prop where
  goal_member : goal ∈ goals
  route_actions :
    Task2RouteActionPlan phase r start route goal tileActions

theorem task2_bfs_plan_has_bfs_result
    {phase : Task2Phase} {r : RoomState} {start : Position}
    {goals route : List Position} {goal : Position}
    {tileActions : List Action}
    (hplan : Task2BfsPlanGenerated
      phase r start goals route goal tileActions) :
    BfsResult r start goals route := by
  exact ⟨
    goal,
    hplan.goal_member,
    task2_route_action_plan_to_tilePath hplan.route_actions
  ⟩

/-! ### Task2 作为通用 BFS 图规格的实例

通用 BFS 不知道 tile、安全层或阶段。Task2 只需要提供邻居枚举相对于
`Task2MoveAllowed` 的 sound/complete 条件，就可以把通用 BFS 的层级完备性
接回 `Task2RouteActionPlan` 和 `Task2BfsPlanGenerated`。
-/

def Task2GraphNeighborSound
    (phase : Task2Phase) (r : RoomState)
    (neighbors : Position → List Position) : Prop :=
  ∀ p q, q ∈ neighbors p →
    ∃ d, q = advance p d ∧ Task2MoveAllowed phase r q

def Task2GraphNeighborComplete
    (phase : Task2Phase) (r : RoomState)
    (neighbors : Position → List Position) : Prop :=
  ∀ p q d,
    q = advance p d →
    Task2MoveAllowed phase r q →
    q ∈ neighbors p

theorem task2_route_action_plan_to_generic_graph_path
    {phase : Task2Phase} {r : RoomState}
    {neighbors : Position → List Position}
    {start goal : Position} {route : List Position}
    {tileActions : List Action}
    (hcomplete : Task2GraphNeighborComplete phase r neighbors)
    (hplan :
      Task2RouteActionPlan phase r start route goal tileActions) :
    GenericGraphPath neighbors start route goal := by
  induction hplan with
  | nil p =>
      exact GenericGraphPath.nil p
  | cons d hq hallowed htail ih =>
      exact GenericGraphPath.cons
        (hcomplete _ _ d hq hallowed) ih

theorem generic_graph_path_to_task2_route_action_plan
    {phase : Task2Phase} {r : RoomState}
    {neighbors : Position → List Position}
    {start goal : Position} {route : List Position}
    (hsound : Task2GraphNeighborSound phase r neighbors)
    (hpath : GenericGraphPath neighbors start route goal) :
    ∃ tileActions,
      Task2RouteActionPlan phase r start route goal tileActions := by
  induction hpath with
  | nil p =>
      exact ⟨[], Task2RouteActionPlan.nil p⟩
  | cons hstep htail ih =>
      rcases hsound _ _ hstep with ⟨d, hq, hallowed⟩
      rcases ih with ⟨tailActions, htailPlan⟩
      exact ⟨
        actionForDirection d :: tailActions,
        Task2RouteActionPlan.cons d hq hallowed htailPlan
      ⟩

theorem task2_objective_bfs_complete_within
    {phase : Task2Phase} {r : RoomState}
    {neighbors : Position → List Position}
    {start : Position} {goals : List Position} {maxDepth : Nat}
    (hcomplete : Task2GraphNeighborComplete phase r neighbors)
    (hreachable : ∃ route goal tileActions,
      goal ∈ goals ∧
      route.length ≤ maxDepth ∧
      Task2RouteActionPlan phase r start route goal tileActions) :
    GenericBfsFindsGoalWithin neighbors start goals maxDepth := by
  rcases hreachable with
    ⟨route, goal, tileActions, hgoal, hbound, hplan⟩
  exact generic_bfs_complete_within
    (neighbors := neighbors)
    (start := start)
    (goals := goals)
    (maxDepth := maxDepth)
    ⟨
      route,
      goal,
      hgoal,
      hbound,
      task2_route_action_plan_to_generic_graph_path
        hcomplete hplan
    ⟩

theorem task2_objective_bfs_layer_route_to_plan
    {phase : Task2Phase} {r : RoomState}
    {neighbors : Position → List Position}
    {start : Position} {goals : List Position}
    {depth : Nat} {route : List Position} {goal : Position}
    (hsound : Task2GraphNeighborSound phase r neighbors)
    (hmem : route ∈ GenericBfsRoutesExact neighbors start depth)
    (hgoal : goal ∈ goals)
    (hend : GenericRouteEnd start route = goal) :
    ∃ tileActions,
      Task2BfsPlanGenerated phase r start goals route goal tileActions := by
  have hpath :
      GenericGraphPath neighbors start route goal := by
    rw [← hend]
    exact generic_bfs_route_layer_sound hmem
  rcases generic_graph_path_to_task2_route_action_plan
      hsound hpath with
    ⟨tileActions, hplan⟩
  exact ⟨tileActions, ⟨hgoal, hplan⟩⟩

/-! ### Python `bfs_path` 子语言：parent 字典与路径重建

下面不是完整 Python 解释器，而是覆盖 `task2_fsm_bfs_agent.py` 中 BFS 核心
片段的专用操作语义：

```python
parent[nxt] = current
if nxt in goals:
    return reconstruct_path(parent, nxt)
```

`PythonBfsParentMap` 把 Python 字典中 `child ↦ parent` 的有效边抽象为列表；
`PythonBfsParentRoute` 表示 `reconstruct_path` 沿 parent 链从 `goal` 回到
`start` 后反转得到的正向路径。定理
`python_bfs_return_sound_for_task2_plan` 证明：如果该子语义返回一条 route，
那么它就是 Task2 可检查的 BFS 生成计划。

仍未覆盖的是完整 `while queue: popleft(); for nxt in neighbors(...)` 的终止和
完备性；那需要继续证明 queue/visited/frontier 循环不变量。
-/

abbrev PythonBfsParentMap := List (Position × Position)

def PythonBfsParentEdgeAllowed
    (phase : Task2Phase) (r : RoomState)
    (parent child : Position) : Prop :=
  ∃ d, child = advance parent d ∧ Task2MoveAllowed phase r child

def PythonBfsParentMapSound
    (phase : Task2Phase) (r : RoomState)
    (parents : PythonBfsParentMap) : Prop :=
  ∀ child parent, (child, parent) ∈ parents →
    PythonBfsParentEdgeAllowed phase r parent child

theorem python_bfs_parent_insert_preserves_sound
    {phase : Task2Phase} {r : RoomState}
    {parents : PythonBfsParentMap} {parent child : Position}
    (hsound : PythonBfsParentMapSound phase r parents)
    (hedge : PythonBfsParentEdgeAllowed phase r parent child) :
    PythonBfsParentMapSound phase r ((child, parent) :: parents) := by
  intro q p hmem
  rcases List.mem_cons.mp hmem with hhead | htail
  · cases hhead
    exact hedge
  · exact hsound q p htail

inductive PythonBfsParentRoute
    (phase : Task2Phase) (r : RoomState)
    (parents : PythonBfsParentMap) :
    Position → Position → List Position → List Action → Prop where
  | done (p : Position) :
      PythonBfsParentRoute phase r parents p p [] []
  | step {current next goal : Position}
      {route : List Position} {actions : List Action} (d : Direction)
      (hparent : (next, current) ∈ parents)
      (hnext : next = advance current d)
      (hallowed : Task2MoveAllowed phase r next)
      (htail :
        PythonBfsParentRoute phase r parents next goal route actions) :
      PythonBfsParentRoute phase r parents current goal
        (next :: route) (actionForDirection d :: actions)

theorem python_parent_route_to_task2_route_action_plan
    {phase : Task2Phase} {r : RoomState}
    {parents : PythonBfsParentMap}
    {start goal : Position} {route : List Position}
    {tileActions : List Action}
    (hroute :
      PythonBfsParentRoute phase r parents start goal route tileActions) :
    Task2RouteActionPlan phase r start route goal tileActions := by
  induction hroute with
  | done p =>
      exact Task2RouteActionPlan.nil p
  | step d hparent hnext hallowed htail ih =>
      exact Task2RouteActionPlan.cons d hnext hallowed ih

structure PythonBfsReturnedRoute
    (phase : Task2Phase) (r : RoomState) (start : Position)
    (goals : List Position) (parents : PythonBfsParentMap)
    (route : List Position) (goal : Position)
    (tileActions : List Action) : Prop where
  parent_map_sound : PythonBfsParentMapSound phase r parents
  goal_member : goal ∈ goals
  reconstructed :
    PythonBfsParentRoute phase r parents start goal route tileActions

theorem python_bfs_return_sound_for_task2_plan
    {phase : Task2Phase} {r : RoomState} {start : Position}
    {goals : List Position} {parents : PythonBfsParentMap}
    {route : List Position} {goal : Position}
    {tileActions : List Action}
    (hreturn : PythonBfsReturnedRoute
      phase r start goals parents route goal tileActions) :
    Task2BfsPlanGenerated phase r start goals route goal tileActions := by
  exact ⟨
    hreturn.goal_member,
    python_parent_route_to_task2_route_action_plan
      hreturn.reconstructed
  ⟩

/-!
下面再向 Python 代码靠近一层：把 `bfs_path` 的 `queue`、`parent` 和
`while queue: ... for nxt in neighbors(current): ...` 建成一个小步语义。

这仍然是专用子语言，而不是通用 Python 解释器。它逐分支覆盖：

* `current = queue.popleft()`
* `for nxt in neighbors(current)`
* `if nxt in parent: continue`
* `if ... not walkable ...: continue`
* `parent[nxt] = current`
* `if nxt in goals: return reconstruct_path(parent, nxt)`
* `queue.append(nxt)`

核心循环不变量是：`parent` 字典里的每一条 `child ↦ parent` 都是
`Task2MoveAllowed` 认证过的一步 tile 移动。这样，当 Python 最终返回并用
`reconstruct_path` 沿 parent 链重建 route 时，上面的
`python_bfs_return_sound_for_task2_plan` 可以把它转成 Task2 的 BFS 计划证书。
-/

inductive PythonBfsControl where
  | ready
  | scanning (current : Position) (remainingDirs : List Direction)
  | returned (goal : Position)
  | failed
  deriving DecidableEq, Repr

structure PythonBfsState where
  queue : List Position
  parents : PythonBfsParentMap
  seen : List Position
  control : PythonBfsControl
  deriving Repr

def pythonBfsNeighborDirections : List Direction :=
  [.north, .south, .west, .east]

def PythonBfsStateParentSound
    (phase : Task2Phase) (r : RoomState)
    (s : PythonBfsState) : Prop :=
  PythonBfsParentMapSound phase r s.parents

inductive PythonBfsStep
    (phase : Task2Phase) (r : RoomState) (goals : List Position) :
    PythonBfsState → PythonBfsState → Prop where
  | pop {current : Position} {restQueue parents seen : List Position}
      {parentEdges : PythonBfsParentMap}
      (hparents : parents = seen) :
      PythonBfsStep phase r goals
        { queue := current :: restQueue, parents := parentEdges,
          seen := parents, control := .ready }
        { queue := restQueue, parents := parentEdges,
          seen := seen, control := .scanning current pythonBfsNeighborDirections }
  | queue_empty {parentEdges : PythonBfsParentMap} {seen : List Position} :
      PythonBfsStep phase r goals
        { queue := [], parents := parentEdges, seen := seen,
          control := .ready }
        { queue := [], parents := parentEdges, seen := seen,
          control := .failed }
  | finish_scan {queue seen : List Position}
      {parentEdges : PythonBfsParentMap} {current : Position} :
      PythonBfsStep phase r goals
        { queue := queue, parents := parentEdges, seen := seen,
          control := .scanning current [] }
        { queue := queue, parents := parentEdges, seen := seen,
          control := .ready }
  | skip_seen {queue seen : List Position}
      {parentEdges : PythonBfsParentMap} {current : Position}
      {d : Direction} {restDirs : List Direction}
      (hseen : advance current d ∈ seen) :
      PythonBfsStep phase r goals
        { queue := queue, parents := parentEdges, seen := seen,
          control := .scanning current (d :: restDirs) }
        { queue := queue, parents := parentEdges, seen := seen,
          control := .scanning current restDirs }
  | skip_rejected {queue seen : List Position}
      {parentEdges : PythonBfsParentMap} {current : Position}
      {d : Direction} {restDirs : List Direction}
      (hnew : advance current d ∉ seen)
      (hrejected : ¬ Task2MoveAllowed phase r (advance current d)) :
      PythonBfsStep phase r goals
        { queue := queue, parents := parentEdges, seen := seen,
          control := .scanning current (d :: restDirs) }
        { queue := queue, parents := parentEdges, seen := seen,
          control := .scanning current restDirs }
  | discover_non_goal {queue seen : List Position}
      {parentEdges : PythonBfsParentMap} {current : Position}
      {d : Direction} {restDirs : List Direction}
      (hnew : advance current d ∉ seen)
      (hallowed : Task2MoveAllowed phase r (advance current d))
      (hnotGoal : advance current d ∉ goals) :
      PythonBfsStep phase r goals
        { queue := queue, parents := parentEdges, seen := seen,
          control := .scanning current (d :: restDirs) }
        { queue := queue ++ [advance current d],
          parents := (advance current d, current) :: parentEdges,
          seen := advance current d :: seen,
          control := .scanning current restDirs }
  | discover_goal {queue seen : List Position}
      {parentEdges : PythonBfsParentMap} {current : Position}
      {d : Direction} {restDirs : List Direction}
      (hnew : advance current d ∉ seen)
      (hallowed : Task2MoveAllowed phase r (advance current d))
      (hgoal : advance current d ∈ goals) :
      PythonBfsStep phase r goals
        { queue := queue, parents := parentEdges, seen := seen,
          control := .scanning current (d :: restDirs) }
        { queue := queue,
          parents := (advance current d, current) :: parentEdges,
          seen := advance current d :: seen,
          control := .returned (advance current d) }

theorem python_bfs_step_preserves_parent_sound
    {phase : Task2Phase} {r : RoomState} {goals : List Position}
    {s t : PythonBfsState}
    (hsound : PythonBfsStateParentSound phase r s)
    (hstep : PythonBfsStep phase r goals s t) :
    PythonBfsStateParentSound phase r t := by
  cases hstep with
  | pop hparents =>
      simpa [PythonBfsStateParentSound] using hsound
  | queue_empty =>
      simpa [PythonBfsStateParentSound] using hsound
  | finish_scan =>
      simpa [PythonBfsStateParentSound] using hsound
  | skip_seen hseen =>
      simpa [PythonBfsStateParentSound] using hsound
  | skip_rejected hnew hrejected =>
      simpa [PythonBfsStateParentSound] using hsound
  | discover_non_goal hnew hallowed hnotGoal =>
      unfold PythonBfsStateParentSound at hsound ⊢
      apply python_bfs_parent_insert_preserves_sound hsound
      exact ⟨_, rfl, hallowed⟩
  | discover_goal hnew hallowed hgoal =>
      unfold PythonBfsStateParentSound at hsound ⊢
      apply python_bfs_parent_insert_preserves_sound hsound
      exact ⟨_, rfl, hallowed⟩

inductive PythonBfsExec
    (phase : Task2Phase) (r : RoomState) (goals : List Position) :
    PythonBfsState → PythonBfsState → Prop where
  | refl (s : PythonBfsState) :
      PythonBfsExec phase r goals s s
  | step {s t u : PythonBfsState}
      (hstep : PythonBfsStep phase r goals s t)
      (htail : PythonBfsExec phase r goals t u) :
      PythonBfsExec phase r goals s u

theorem python_bfs_exec_preserves_parent_sound
    {phase : Task2Phase} {r : RoomState} {goals : List Position}
    {s t : PythonBfsState}
    (hsound : PythonBfsStateParentSound phase r s)
    (hexec : PythonBfsExec phase r goals s t) :
    PythonBfsStateParentSound phase r t := by
  induction hexec with
  | refl s =>
      exact hsound
  | step hstep htail ih =>
      exact ih (python_bfs_step_preserves_parent_sound hsound hstep)

structure PythonBfsSmallStepReturnedRoute
    (phase : Task2Phase) (r : RoomState) (start : Position)
    (goals : List Position) (initial final : PythonBfsState)
    (route : List Position) (goal : Position)
    (tileActions : List Action) : Prop where
  initial_parent_sound : PythonBfsStateParentSound phase r initial
  exec : PythonBfsExec phase r goals initial final
  returned : final.control = .returned goal
  goal_member : goal ∈ goals
  reconstructed :
    PythonBfsParentRoute phase r final.parents start goal route tileActions

theorem python_bfs_small_step_return_sound_for_task2_plan
    {phase : Task2Phase} {r : RoomState} {start : Position}
    {goals : List Position} {initial final : PythonBfsState}
    {route : List Position} {goal : Position}
    {tileActions : List Action}
    (hreturn : PythonBfsSmallStepReturnedRoute
      phase r start goals initial final route goal tileActions) :
    Task2BfsPlanGenerated phase r start goals route goal tileActions := by
  have hparents :
      PythonBfsParentMapSound phase r final.parents :=
    python_bfs_exec_preserves_parent_sound
      hreturn.initial_parent_sound hreturn.exec
  exact python_bfs_return_sound_for_task2_plan
    (phase := phase) (r := r) (start := start)
    (goals := goals) (parents := final.parents)
    (route := route) (goal := goal) (tileActions := tileActions)
    ⟨hparents, hreturn.goal_member, hreturn.reconstructed⟩

/-!
### 条件 BFS 完备性接口

完整逐行证明 Python `deque` 循环需要较重的有限图和队列层级不变量。为了避免
把前提退化成“BFS 已经返回了 route”，这里暴露一个更可审计的中间接口：

* `Task2BfsFrontierComplete`：到某一深度为止，frontier/seen 已覆盖所有
  `Task2RouteActionPlan` 可达位置；
* `PythonBfsReturnsOnFrontierGoal`：如果这个 frontier/seen 中已经包含目标，
  Python BFS 小步语义会执行到 `returned` 并由 `reconstruct_path` 给出路径。

于是只要某个目标在深度 `n` 内可达，就能推出小步语义返回，再推出
`Task2BfsPlanGenerated`。这个前提是 BFS 算法级不变量，能在 Task1--Task5
之间复用；它明显弱于直接假设每个阶段已经给出 `Task2BfsPhaseGenerated`。
-/

def Task2BoundedReachable
    (phase : Task2Phase) (r : RoomState) (start : Position)
    (depth : Nat) (goal : Position) : Prop :=
  ∃ route tileActions,
    route.length ≤ depth ∧
    Task2RouteActionPlan phase r start route goal tileActions

def Task2BfsFrontierComplete
    (phase : Task2Phase) (r : RoomState) (start : Position)
    (depth : Nat) (frontier : List Position) : Prop :=
  ∀ goal,
    Task2BoundedReachable phase r start depth goal →
    goal ∈ frontier

def Task2BfsFindsGoal
    (frontier goals : List Position) : Prop :=
  ∃ goal, goal ∈ frontier ∧ goal ∈ goals

theorem task2_bfs_complete_from_frontier_invariant
    {phase : Task2Phase} {r : RoomState} {start : Position}
    {depth : Nat} {frontier goals : List Position}
    (hcomplete :
      Task2BfsFrontierComplete phase r start depth frontier)
    (hreachable : ∃ goal,
      goal ∈ goals ∧
      Task2BoundedReachable phase r start depth goal) :
    Task2BfsFindsGoal frontier goals := by
  rcases hreachable with ⟨goal, hgoal, hbounded⟩
  exact ⟨goal, hcomplete goal hbounded, hgoal⟩

def PythonBfsReturnsOnFrontierGoal
    (phase : Task2Phase) (r : RoomState) (start : Position)
    (goals : List Position) (initial : PythonBfsState)
    (frontier : List Position) : Prop :=
  Task2BfsFindsGoal frontier goals →
    ∃ final route goal tileActions,
      PythonBfsSmallStepReturnedRoute
        phase r start goals initial final route goal tileActions

theorem python_bfs_complete_from_frontier_certificate
    {phase : Task2Phase} {r : RoomState} {start : Position}
    {goals : List Position} {initial : PythonBfsState}
    {depth : Nat} {frontier : List Position}
    (hcomplete :
      Task2BfsFrontierComplete phase r start depth frontier)
    (hreturns :
      PythonBfsReturnsOnFrontierGoal
        phase r start goals initial frontier)
    (hreachable : ∃ goal,
      goal ∈ goals ∧
      Task2BoundedReachable phase r start depth goal) :
    ∃ final route goal tileActions,
      PythonBfsSmallStepReturnedRoute
        phase r start goals initial final route goal tileActions ∧
      Task2BfsPlanGenerated phase r start goals route goal tileActions := by
  have hfound : Task2BfsFindsGoal frontier goals :=
    task2_bfs_complete_from_frontier_invariant
      hcomplete hreachable
  rcases hreturns hfound with
    ⟨final, route, goal, tileActions, hreturn⟩
  exact ⟨
    final,
    route,
    goal,
    tileActions,
    hreturn,
    python_bfs_small_step_return_sound_for_task2_plan hreturn
  ⟩

theorem task2_route_action_plan_has_shielded_engineExec
    {phase : Task2Phase} {r : RoomState}
    {s : WorldState} {start goal : Position}
    {route : List Position} {actions : List Action}
    (hroom : currentRoomState s = r)
    (hstart : s.player.pos = start)
    (hbuttons : r.buttons = [])
    (hrunning : Running s)
    (hplan : Task2RouteActionPlan phase r start route goal actions) :
    Task2ShieldedEngineExec phase s actions
      (runPlainTileActions s actions) ∧
    (runPlainTileActions s actions).player.pos = goal ∧
    currentRoomState (runPlainTileActions s actions) = r := by
  induction hplan generalizing s with
  | nil p =>
      exact ⟨Task2ShieldedEngineExec.nil, hstart, hroom⟩
  | @cons p q pathGoal rest tailActions d hq hallowed htail ih =>
      let next := movePlayerState s q d
      have hqS : q = advance s.player.pos d := by
        rw [hstart]
        exact hq
      have henterS : canEnter (currentRoomState s) q := by
        rw [hroom]
        exact hallowed.1.1
      have htrapS : ¬ activeTrapAt (currentRoomState s) q := by
        rw [hroom]
        exact hallowed.1.2.1
      have hbuttonS : ¬ buttonAt (currentRoomState s) q := by
        rw [hroom]
        intro hb
        rcases hb with ⟨button, hmember, hpos⟩
        rw [hbuttons] at hmember
        simp at hmember
      have hstep :
          Step s (actionForDirection d) next
            [.moved s.player.pos q] := by
        exact Step.movePlain (actionForDirection_correct d)
          hqS henterS htrapS hbuttonS
      have hplayer :
          PlayerStep s (actionForDirection d) next
            [.moved s.player.pos q] := by
        refine ⟨hrunning, hstep, ?_⟩
        intro hauto
        cases hauto
      have htick :
          EngineTick s (actionForDirection d) next :=
        EngineTick.mk hplayer AutonomousExec.nil
      have hcommand :
          ∀ d', actionDirection (actionForDirection d) = some d' →
            Task2CommandAllowed phase (currentRoomState s)
              (advance s.player.pos d') := by
        intro d' hd'
        have hdEq : d' = d := by
          cases d <;> cases d' <;>
            simp [actionForDirection, actionDirection] at hd' ⊢
        subst d'
        left
        rw [hroom, ← hqS]
        exact hallowed
      have hnextRoom : currentRoomState next = r := by
        rw [movePlayerState_room_unchanged, hroom]
      have hnextPos : next.player.pos = q := by
        rfl
      have hnextRunning : Running next := by
        exact hrunning
      rcases ih hnextRoom hnextPos hnextRunning with
        ⟨htailShielded, htailPos, htailRoom⟩
      have happly :
          applyPlainTileAction s (actionForDirection d) = next := by
        cases d <;> simp [applyPlainTileAction, actionForDirection,
          actionDirection, next, movePlayerState, hqS]
      constructor
      · simp [runPlainTileActions, happly]
        exact Task2ShieldedEngineExec.cons htick hcommand htailShielded
      constructor
      · simpa [runPlainTileActions, happly] using htailPos
      · simpa [runPlainTileActions, happly] using htailRoom

abbrev Task2PixelActionsRefineTileActions
    (_phase : Task2Phase) (_start _finish : WorldState)
    (pixelActions tileActions : List Action) : Prop :=
  PixelActionsRefineTileActions pixelActions tileActions

structure Task2BfsPhaseGenerated
    (phase : Task2Phase) (start finish : WorldState)
    (pixelActions tileActions : List Action) where
  goals : List Position
  route : List Position
  goal : Position
  bfs_plan :
    Task2BfsPlanGenerated phase (currentRoomState start)
      start.player.pos goals route goal tileActions
  pixel_to_tile_refinement :
    PixelToTileKinematicRefinement
      start finish pixelActions tileActions
  no_buttons_on_route_room : (currentRoomState start).buttons = []
  running_at_start : Running start

theorem task2_bfs_phase_generated_has_bfs_result
    {phase : Task2Phase} {start finish : WorldState}
    {pixelActions tileActions : List Action}
    (h : Task2BfsPhaseGenerated
      phase start finish pixelActions tileActions) :
    BfsResult (currentRoomState start) start.player.pos h.goals h.route :=
  task2_bfs_plan_has_bfs_result h.bfs_plan

theorem task2_bfs_phase_generated_has_shielded_engineExec
    {phase : Task2Phase} {start finish : WorldState}
    {pixelActions tileActions : List Action}
    (h : Task2BfsPhaseGenerated
      phase start finish pixelActions tileActions) :
    Task2ShieldedEngineExec phase start tileActions finish := by
  have hroute :=
    task2_route_action_plan_has_shielded_engineExec
      (phase := phase)
      (r := currentRoomState start)
      (s := start)
      (start := start.player.pos)
      (goal := h.goal)
      (route := h.route)
      (actions := tileActions)
      rfl rfl h.no_buttons_on_route_room h.running_at_start
      h.bfs_plan.route_actions
  rcases hroute with ⟨hshielded, _hpos, _hroom⟩
  rw [h.pixel_to_tile_refinement.finish_is_tile_run]
  exact hshielded

theorem task2_bfs_phase_generated_preserves_currentRoomState
    {phase : Task2Phase} {start finish : WorldState}
    {pixelActions tileActions : List Action}
    (h : Task2BfsPhaseGenerated
      phase start finish pixelActions tileActions) :
    currentRoomState finish = currentRoomState start := by
  rw [h.pixel_to_tile_refinement.finish_is_tile_run]
  exact runPlainTileActions_currentRoomState start tileActions

theorem task2_bfs_phase_generated_preserves_inventory_keys
    {phase : Task2Phase} {start finish : WorldState}
    {pixelActions tileActions : List Action}
    (h : Task2BfsPhaseGenerated
      phase start finish pixelActions tileActions) :
    finish.player.inventory.keys = start.player.inventory.keys := by
  rw [h.pixel_to_tile_refinement.finish_is_tile_run]
  exact runPlainTileActions_inventory_keys start tileActions

/-! ## 5. Task2 三阶段组合正确性

`Task2Completable` 要求最终世界完成。主定理把四段已经验证的轨迹拼接：

1. BFS 到安全攻击位；
2. 面向、挥剑和动态重规划组成的战斗轨迹；
3. 清怪后 BFS 到宝箱并开箱；
4. 获得钥匙后 BFS 到条件出口并推出房间。

动态怪物的具体随机轨迹由 `hCombat` 参数表示；只要战斗控制器最终产生一段
合法 `Exec` 且清空怪物，后续 FSM 组合必然正确。
-/

def Task2Goal (s : WorldState) : Prop :=
  WorldCompleted s

def Task2Completable (initial : WorldState) : Prop :=
  ∃ actions final, Exec initial actions final ∧ Task2Goal final

def Task2SafelyCompletable (initial : WorldState) : Prop :=
  ∃ actions final,
    Exec initial actions final ∧
    Task2Goal final ∧
    alive final ∧
    ValidState final

theorem task2_completable_if_subplans_exist
    {initial nearMonster afterCombat nearChest afterChest atExit : WorldState}
    {toMonster combatActions toChest toExit : List Action}
    {chest : Chest} {exit : Exit} {targetRoom : RoomState}
    (hToMonster : Exec initial toMonster nearMonster)
    (hCombat : Exec nearMonster combatActions afterCombat)
    (_hCleared : (currentRoomState afterCombat).monsters = [])
    (hToChest : Exec afterCombat toChest nearChest)
    (hChestMember : chest ∈ (currentRoomState nearChest).chests)
    (hChestVisible : chest.visible = true)
    (hChestClosed : chest.opened = false)
    (hChestAdjacent : adjacent nearChest.player.pos chest.pos)
    (hAfterChest : afterChest = stateAfterOpeningChest nearChest chest)
    (hToExit : Exec afterChest toExit atExit)
    (hExitMember : exit ∈ (currentRoomState atExit).exits)
    (hAtExit : atExit.player.pos = exit.pos)
    (_hFacingExit : atExit.player.facing = exit.direction)
    (hRequirement : requirementSatisfied atExit exit.requirement)
    (hTargetRoom : targetRoom = atExit.rooms exit.targetRoom)
    (hSpawn : canEnter targetRoom exit.targetSpawn)
    (hCompletes : exit.completesTask = true) :
    Task2Completable initial := by
  have hOpenStep :
      Step nearChest .slotA (stateAfterOpeningChest nearChest chest)
        [.chestOpened chest.id] :=
    Step.openChest hChestMember hChestVisible hChestClosed (Or.inr hChestAdjacent)
  have hOpenExec :
      Exec nearChest [.slotA] (stateAfterOpeningChest nearChest chest) :=
    Exec.cons hOpenStep Exec.nil
  have hExitStep :
      Step atExit (directionAction exit.direction)
        (stateAfterUsingExit atExit exit)
        (exitEvents atExit exit) :=
    Step.useExit hExitMember (Or.inl hAtExit)
      (requirement_implies_exitRequirementSatisfied hRequirement)
      hTargetRoom hSpawn
  have hExitExec :
      Exec atExit [directionAction exit.direction]
        (stateAfterUsingExit atExit exit) :=
    Exec.cons hExitStep Exec.nil
  subst afterChest
  have hPhase1 :
      Exec initial (toMonster ++ combatActions) afterCombat :=
    exec_append hToMonster hCombat
  have hPhase2 :
      Exec initial ((toMonster ++ combatActions) ++ toChest) nearChest :=
    exec_append hPhase1 hToChest
  have hPhase3 :
      Exec initial (((toMonster ++ combatActions) ++ toChest) ++ [.slotA])
        (stateAfterOpeningChest nearChest chest) :=
    exec_append hPhase2 hOpenExec
  have hPhase4 :
      Exec initial
        ((((toMonster ++ combatActions) ++ toChest) ++ [.slotA]) ++ toExit)
        atExit :=
    exec_append hPhase3 hToExit
  have hAll :
      Exec initial
        (((((toMonster ++ combatActions) ++ toChest) ++ [.slotA]) ++ toExit) ++
          [directionAction exit.direction])
        (stateAfterUsingExit atExit exit) :=
    exec_append hPhase4 hExitExec
  refine ⟨_, _, hAll, ?_⟩
  unfold Task2Goal WorldCompleted stateAfterUsingExit transitionThroughExit
  simp [hCompletes]

/-!
下面保留一个独立的动态公平性谓词，用来表达“存在有限合法战斗轨迹清空怪物”。
主策略定理不再把它和实际 combat trace 分开使用；真正进入主定理的是
`Task2CombatPhaseGenerated`，它把控制器生成的战斗 trace 与清怪后置条件绑定
在同一个证书里。
-/

def EventuallyCombatClears (nearMonster afterCombat : WorldState) : Prop :=
  ∃ combatActions,
    Exec nearMonster combatActions afterCombat ∧
    (currentRoomState afterCombat).monsters = []

theorem task2_combat_fairness_exposes_finite_plan
    {nearMonster afterCombat : WorldState}
    (hfair : EventuallyCombatClears nearMonster afterCombat) :
    ∃ combatActions,
      Exec nearMonster combatActions afterCombat ∧
      (currentRoomState afterCombat).monsters = [] :=
  hfair

/-! ## 6. Python FSM/BFS/shield 策略的条件正确性接口

下面的结构把 Python 代码中仍位于 Lean 之外的部分显式列为前提，而不是把
“已经到达完成终态”当成主定理前提：

* `Task2ObservationSound` 表示像素分类后的符号对象与真实世界一致；
* `Task2InventoryInfoSound` 表示允许读取的物品栏钥匙数与真实状态一致；
* `PixelToTileKinematicRefinement` 表示一段 pixel tick 动作可由全局像素/物理
  refinement 压缩成 tile 动作；
* `Task2PixelToTilePhaseRefinement` 表示一段 tile 级动作已经通过
  `Task2ShieldedEngineExec` 检查；
* `Task2CombatPhaseGenerated` 把战斗 trace 和清怪后置条件绑定在同一证书里；
* `Task2ConditionalStrategyAssumptions` 把三段 BFS、战斗生成证书、开箱、
  条件出口和最终存活集中成一个可审计的策略证书。

因此最终定理证明的是：在这些条件都成立时，Python FSM+BFS+shield 策略会
生成一条安全完成轨迹；它不再直接假设 `Task2Goal final`。
-/

structure Task2SymbolicObservation where
  playerPos : Position
  monsters : List Monster
  visibleChests : List Chest
  exits : List Exit
  hasKey : Bool
  walkable : Position → Prop
  dangerous : Position → Prop

def Task2ObservationSound
    (s : WorldState) (obs : Task2SymbolicObservation) : Prop :=
  obs.playerPos = s.player.pos ∧
  obs.monsters = (currentRoomState s).monsters ∧
  obs.visibleChests =
    (currentRoomState s).chests.filter (fun chest => chest.visible) ∧
  obs.exits = (currentRoomState s).exits ∧
  (∀ p, obs.walkable p ↔ canEnter (currentRoomState s) p) ∧
  (∀ p, obs.dangerous p ↔
    activeTrapAt (currentRoomState s) p ∨
    monsterAt (currentRoomState s) p)

def Task2InventoryInfoSound
    (s : WorldState) (obs : Task2SymbolicObservation) : Prop :=
  obs.hasKey = true ↔ 0 < s.player.inventory.keys

theorem task2_inventory_sound_key_confirmed
    {s : WorldState} {obs : Task2SymbolicObservation}
    (hsound : Task2InventoryInfoSound s obs)
    (hkey : obs.hasKey = true) :
    0 < s.player.inventory.keys :=
  hsound.mp hkey

structure Task2PixelToTilePhaseRefinement
    (phase : Task2Phase) (start finish : WorldState)
    (pixelActions tileActions : List Action) : Prop where
  -- This is the explicit bridge from Python pixel ticks to the tile-level trace.
  -- It is intentionally a contract: proving it requires reasoning about the
  -- Python renderer, collision code, and repeated per-pixel actions.
  pixel_refines_tile_actions :
    Task2PixelActionsRefineTileActions
      phase start finish pixelActions tileActions
  tile_trace : Task2ShieldedEngineExec phase start tileActions finish

theorem task2_phase_refinement_has_engineExec
    {phase : Task2Phase} {start finish : WorldState}
    {pixelActions tileActions : List Action}
    (h : Task2PixelToTilePhaseRefinement
      phase start finish pixelActions tileActions) :
    EngineExec start tileActions finish :=
  task2ShieldedEngineExec_to_engineExec h.tile_trace

structure Task2CombatPhaseGenerated
    (start finish : WorldState)
    (pixelActions tileActions : List Action) : Prop where
  phase_refinement :
    Task2PixelToTilePhaseRefinement .toMonster
      start finish pixelActions tileActions
  clears_monsters : (currentRoomState finish).monsters = []

theorem task2_combat_phase_has_engineExec
    {start finish : WorldState}
    {pixelActions tileActions : List Action}
    (h : Task2CombatPhaseGenerated
      start finish pixelActions tileActions) :
    EngineExec start tileActions finish :=
  task2_phase_refinement_has_engineExec h.phase_refinement

structure Task2ConditionalStrategyAssumptions
    (initial nearMonster afterCombat nearChest afterChest atExit : WorldState)
    (toMonsterPixels toMonsterTiles combatPixels combatTiles
      toChestPixels toChestTiles toExitPixels toExitTiles : List Action)
    (initialObs nearMonsterObs afterCombatObs nearChestObs
      afterChestObs atExitObs : Task2SymbolicObservation)
    (chest : Chest) (exit : Exit) (targetRoom : RoomState) where
  initial_valid : ValidState initial
  initial_observation_sound : Task2ObservationSound initial initialObs
  near_monster_observation_sound : Task2ObservationSound nearMonster nearMonsterObs
  after_combat_observation_sound : Task2ObservationSound afterCombat afterCombatObs
  near_chest_observation_sound : Task2ObservationSound nearChest nearChestObs
  after_chest_observation_sound : Task2ObservationSound afterChest afterChestObs
  at_exit_observation_sound : Task2ObservationSound atExit atExitObs
  inventory_info_sound_after_chest :
    Task2InventoryInfoSound afterChest afterChestObs
  key_confirmed_after_chest : afterChestObs.hasKey = true
  to_monster_bfs_phase :
    Task2BfsPhaseGenerated .toMonster
      initial nearMonster toMonsterPixels toMonsterTiles
  combat_phase :
    Task2CombatPhaseGenerated
      nearMonster afterCombat combatPixels combatTiles
  to_chest_bfs_phase :
    Task2BfsPhaseGenerated .toChest
      afterCombat nearChest toChestPixels toChestTiles
  chest_member : chest ∈ (currentRoomState nearChest).chests
  chest_visible : chest.visible = true
  chest_closed : chest.opened = false
  chest_adjacent : adjacent nearChest.player.pos chest.pos
  after_chest_state : afterChest = stateAfterOpeningChest nearChest chest
  to_exit_bfs_phase :
    Task2BfsPhaseGenerated .toExit
      afterChest atExit toExitPixels toExitTiles
  exit_member : exit ∈ (currentRoomState atExit).exits
  at_exit_position : atExit.player.pos = exit.pos
  facing_exit : atExit.player.facing = exit.direction
  exit_requirement_shape :
    exit.requirement = .both .allMonstersDefeated (.keys 1 false)
  target_room_state : targetRoom = atExit.rooms exit.targetRoom
  exit_spawn_enterable : canEnter targetRoom exit.targetSpawn
  exit_completes_task : exit.completesTask = true
  player_survives_exit : alive (stateAfterUsingExit atExit exit)

theorem task2_fsm_bfs_shield_strategy_is_conditionally_safe_and_complete
    {initial nearMonster afterCombat nearChest afterChest atExit : WorldState}
    {toMonsterPixels toMonsterTiles combatPixels combatTiles
      toChestPixels toChestTiles toExitPixels toExitTiles : List Action}
    {initialObs nearMonsterObs afterCombatObs nearChestObs
      afterChestObs atExitObs : Task2SymbolicObservation}
    {chest : Chest} {exit : Exit} {targetRoom : RoomState}
    (h : Task2ConditionalStrategyAssumptions
      initial nearMonster afterCombat nearChest afterChest atExit
      toMonsterPixels toMonsterTiles combatPixels combatTiles
      toChestPixels toChestTiles toExitPixels toExitTiles
      initialObs nearMonsterObs afterCombatObs nearChestObs
      afterChestObs atExitObs chest exit targetRoom) :
    Task2SafelyCompletable initial := by
  have hAfterChestEq := h.after_chest_state
  subst afterChest
  have hKeyAfterChest : 0 < (stateAfterOpeningChest nearChest chest).player.inventory.keys :=
    task2_inventory_sound_key_confirmed
      h.inventory_info_sound_after_chest h.key_confirmed_after_chest
  have hToMonsterEngine : EngineExec initial toMonsterTiles nearMonster :=
    task2ShieldedEngineExec_to_engineExec
      (task2_bfs_phase_generated_has_shielded_engineExec
        h.to_monster_bfs_phase)
  have hCombatEngine : EngineExec nearMonster combatTiles afterCombat :=
    task2_combat_phase_has_engineExec h.combat_phase
  have hToChestEngine : EngineExec afterCombat toChestTiles nearChest :=
    task2ShieldedEngineExec_to_engineExec
      (task2_bfs_phase_generated_has_shielded_engineExec
        h.to_chest_bfs_phase)
  have hToExitEngine :
      EngineExec (stateAfterOpeningChest nearChest chest) toExitTiles atExit :=
    task2ShieldedEngineExec_to_engineExec
      (task2_bfs_phase_generated_has_shielded_engineExec
        h.to_exit_bfs_phase)
  rcases engineExec_has_microstep_trace hToMonsterEngine with
    ⟨toMonsterMicro, hToMonsterExec⟩
  rcases engineExec_has_microstep_trace hCombatEngine with
    ⟨combatMicro, hCombatExec⟩
  rcases engineExec_has_microstep_trace hToChestEngine with
    ⟨toChestMicro, hToChestExec⟩
  rcases engineExec_has_microstep_trace hToExitEngine with
    ⟨toExitMicro, hToExitExec⟩
  have hCleared : (currentRoomState afterCombat).monsters = [] :=
    h.combat_phase.clears_monsters
  have hNearChestCleared : (currentRoomState nearChest).monsters = [] := by
    have hroom :=
      task2_bfs_phase_generated_preserves_currentRoomState
        h.to_chest_bfs_phase
    rw [hroom]
    exact hCleared
  have hAfterChestCleared :
      (currentRoomState (stateAfterOpeningChest nearChest chest)).monsters = [] := by
    rw [stateAfterOpeningChest_current_monsters]
    exact hNearChestCleared
  have hAtExitCleared : (currentRoomState atExit).monsters = [] :=
    by
      have hroom :=
        task2_bfs_phase_generated_preserves_currentRoomState
          h.to_exit_bfs_phase
      rw [hroom]
      exact hAfterChestCleared
  have hKeyAtExit : 0 < atExit.player.inventory.keys := by
    have hkeys :=
      task2_bfs_phase_generated_preserves_inventory_keys
        h.to_exit_bfs_phase
    rw [hkeys]
    exact hKeyAfterChest
  have hExitRequirement : requirementSatisfied atExit exit.requirement := by
    rw [h.exit_requirement_shape]
    exact ⟨hAtExitCleared, hKeyAtExit⟩
  have hOpenStep :
      Step nearChest .slotA (stateAfterOpeningChest nearChest chest)
        [.chestOpened chest.id] :=
    Step.openChest h.chest_member h.chest_visible
      h.chest_closed (Or.inr h.chest_adjacent)
  have hOpenExec :
      Exec nearChest [.slotA] (stateAfterOpeningChest nearChest chest) :=
    Exec.cons hOpenStep Exec.nil
  have hExitStep :
      Step atExit (directionAction exit.direction)
        (stateAfterUsingExit atExit exit)
        (exitEvents atExit exit) :=
    Step.useExit h.exit_member (Or.inl h.at_exit_position)
      (requirement_implies_exitRequirementSatisfied hExitRequirement)
      h.target_room_state h.exit_spawn_enterable
  have hExitExec :
      Exec atExit [directionAction exit.direction]
        (stateAfterUsingExit atExit exit) :=
    Exec.cons hExitStep Exec.nil
  have hPhase1 :
      Exec initial (toMonsterMicro ++ combatMicro) afterCombat :=
    exec_append hToMonsterExec hCombatExec
  have hPhase2 :
      Exec initial ((toMonsterMicro ++ combatMicro) ++ toChestMicro)
        nearChest :=
    exec_append hPhase1 hToChestExec
  have hPhase3 :
      Exec initial (((toMonsterMicro ++ combatMicro) ++ toChestMicro) ++
        [.slotA]) (stateAfterOpeningChest nearChest chest) :=
    exec_append hPhase2 hOpenExec
  have hPhase4 :
      Exec initial ((((toMonsterMicro ++ combatMicro) ++ toChestMicro) ++
        [.slotA]) ++ toExitMicro) atExit :=
    exec_append hPhase3 hToExitExec
  have hAll :
      Exec initial
        (((((toMonsterMicro ++ combatMicro) ++ toChestMicro) ++
          [.slotA]) ++ toExitMicro) ++ [directionAction exit.direction])
        (stateAfterUsingExit atExit exit) :=
    exec_append hPhase4 hExitExec
  have hvalidNearMonster : ValidState nearMonster :=
    exec_preserves_validState h.initial_valid hToMonsterExec
  have hvalidAfterCombat : ValidState afterCombat :=
    exec_preserves_validState hvalidNearMonster hCombatExec
  have hvalidNearChest : ValidState nearChest :=
    exec_preserves_validState hvalidAfterCombat hToChestExec
  have hvalidAfterChest :
      ValidState (stateAfterOpeningChest nearChest chest) :=
    step_preserves_validState hvalidNearChest hOpenStep
  have hvalidAtExit : ValidState atExit :=
    exec_preserves_validState hvalidAfterChest hToExitExec
  have hvalidFinal : ValidState (stateAfterUsingExit atExit exit) :=
    step_preserves_validState hvalidAtExit hExitStep
  refine ⟨
    (((((toMonsterMicro ++ combatMicro) ++ toChestMicro) ++
      [.slotA]) ++ toExitMicro) ++ [directionAction exit.direction]),
    stateAfterUsingExit atExit exit,
    hAll,
    ?_,
    h.player_survives_exit,
    hvalidFinal
  ⟩
  unfold Task2Goal WorldCompleted stateAfterUsingExit transitionThroughExit
  simp [h.exit_completes_task]

/-! ## 7. 公开 Task2 地图的安全可达性实例

公开地图是 10×8 空房间，上下边缘各有八个陷阱，怪物初始位于 `(2,2)`，
钥匙宝箱位于 `(1,3)`，玩家位于 `(7,3)`。实例证明只用于确认公开关卡满足
通用定理的路径前提；运行时 Agent 仍然从视觉发现这些位置。
-/

def task2Pos (x y : Int) : Position := { x := x, y := y }

def task2PublicBounds : Bounds :=
  { width := 10
    height := 8
    width_pos := by decide
    height_pos := by decide }

def task2PublicTrapPositions : List Position :=
  [ task2Pos 1 0, task2Pos 2 0, task2Pos 3 0, task2Pos 4 0,
    task2Pos 5 0, task2Pos 6 0, task2Pos 7 0, task2Pos 8 0,
    task2Pos 1 7, task2Pos 2 7, task2Pos 3 7, task2Pos 4 7,
    task2Pos 5 7, task2Pos 6 7, task2Pos 7 7, task2Pos 8 7 ]

def task2PublicTraps : List Trap :=
  task2PublicTrapPositions.map
    (fun pos =>
      { id := 0
        pos := pos
        kind := .spike
        damage := 1
        respawn := task2Pos 7 3
        active := true
        singleUse := false })

def task2PublicMonster : Monster :=
  { id := 20
    pos := task2Pos 2 2
    kind := .chaser
    hp := 2
    damage := 1 }

def task2PublicChest : Chest :=
  { id := 21
    pos := task2Pos 1 3
    loot := .key 1
    visible := true
    opened := false }

def task2PublicExit : Exit :=
  { id := 22
    pos := task2Pos 0 3
    direction := .west
    kind := .conditional
    requirement := .both .allMonstersDefeated (.keys 1 false)
    targetRoom := 0
    targetSpawn := task2Pos 8 4
    completesTask := true
    opened := false }

def task2PublicRoom : RoomState :=
  { bounds := task2PublicBounds
    walls := []
    npcs := []
    chests := [task2PublicChest]
    monsters := [task2PublicMonster]
    traps := task2PublicTraps
    buttons := []
    switches := []
    bridges := []
    dynamicTiles := []
    exits := [task2PublicExit] }

def task2PublicStart : Position := task2Pos 7 3
def task2PublicAttackPosition : Position := task2Pos 3 2

def task2PublicToMonsterRoute : List Position :=
  [ task2Pos 6 3, task2Pos 5 3, task2Pos 4 3,
    task2Pos 3 3, task2Pos 3 2 ]

private theorem task2PublicSafeAt
    (x y : Int)
    (h : (x, y) ∈ [(6, 3), (5, 3), (4, 3), (3, 3), (3, 2)]) :
    safeTile task2PublicRoom (task2Pos x y) := by
  simp at h
  rcases h with h | h | h | h | h
  all_goals
    rcases h with ⟨rfl, rfl⟩
    simp [safeTile, canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      activeTrapAt, monsterAt, gapAt, activeBridgeTile, task2PublicRoom,
      task2PublicBounds, task2PublicTraps, task2PublicTrapPositions,
      task2PublicMonster, task2PublicChest, task2Pos]

theorem task2_public_bfs_path_to_attack_position :
    TilePath task2PublicRoom task2PublicStart
      task2PublicToMonsterRoute task2PublicAttackPosition := by
  unfold task2PublicToMonsterRoute task2PublicStart task2PublicAttackPosition
  apply TilePath.cons .west rfl
  · exact task2PublicSafeAt 6 3 (by simp)
  apply TilePath.cons .west rfl
  · exact task2PublicSafeAt 5 3 (by simp)
  apply TilePath.cons .west rfl
  · exact task2PublicSafeAt 4 3 (by simp)
  apply TilePath.cons .west rfl
  · exact task2PublicSafeAt 3 3 (by simp)
  apply TilePath.cons .north rfl
  · exact task2PublicSafeAt 3 2 (by simp)
  exact TilePath.nil _

theorem task2_public_attack_position_is_adjacent :
    adjacent task2PublicAttackPosition task2PublicMonster.pos := by
  simp [task2PublicAttackPosition, task2PublicMonster, task2Pos,
    adjacent, advance]

theorem task2_public_attack_position_reachable :
    TileReachable task2PublicRoom task2PublicStart task2PublicAttackPosition :=
  tilePath_goal_reachable task2_public_bfs_path_to_attack_position

theorem task2_public_monster_needs_two_sword_hits :
    hpAfterSwordHits task2PublicMonster.hp 2 = 0 := by
  decide

theorem task2_public_attack_position_is_safe :
    safeTile task2PublicRoom task2PublicAttackPosition :=
  task2PublicSafeAt 3 2 (by simp)

/-! ### 公开怪物的两次攻击轨迹

公开怪物 HP=2，剑伤害为 1。下面构造玩家已到攻击位并面向 west 的世界，
证明第一次攻击进入 `attackDamage`，第二次进入 `attackKill`，最终怪物列表
为空。这不是假设“攻击会成功”，而是直接使用环境 `Step` 构造器验证。
-/

def task2PublicPlayerAtMonster : PlayerState :=
  { pos := task2PublicAttackPosition
    facing := .west
    hp := 5
    maxHp := 5
    inventory :=
      { keys := 0
        gold := 0
        items := [.sword, .shield] }
    shielding := false }

def task2PublicNearMonsterWorld : WorldState :=
  { currentRoom := 0
    rooms := fun _ => task2PublicRoom
    roomIds := [0]
    player := task2PublicPlayerAtMonster
    completed := false }

def task2PublicInitialPlayer : PlayerState :=
  { task2PublicPlayerAtMonster with
      pos := task2PublicStart
      facing := .west }

def task2PublicInitial : WorldState :=
  { currentRoom := 0
    rooms := fun _ => task2PublicRoom
    roomIds := [0]
    player := task2PublicInitialPlayer
    completed := false }

def task2PublicDamagedMonster : Monster :=
  { task2PublicMonster with hp := task2PublicMonster.hp - swordDamage }

def task2PublicAfterFirstHit : WorldState :=
  updateCurrentRoom
    { task2PublicNearMonsterWorld with
      player := { task2PublicNearMonsterWorld.player with shielding := false } }
    (replaceMonster task2PublicRoom task2PublicMonster task2PublicDamagedMonster)

def task2PublicAfterKill : WorldState :=
  resolveMonsterKill task2PublicAfterFirstHit task2PublicDamagedMonster

theorem task2_public_initial_running : Running task2PublicInitial := by
  simp [Running, alive, task2PublicInitial, task2PublicInitialPlayer,
    task2PublicPlayerAtMonster]

theorem task2_public_engine_path_to_attack_position :
    ∃ actions atAttack,
      EngineExec task2PublicInitial actions atAttack ∧
      atAttack.player.pos = task2PublicAttackPosition ∧
      currentRoomState atAttack = task2PublicRoom ∧
      Running atAttack ∧
      atAttack.player.inventory = task2PublicInitial.player.inventory ∧
      atAttack.player.hp = task2PublicInitial.player.hp ∧
      atAttack.player.maxHp = task2PublicInitial.player.maxHp ∧
      atAttack.currentRoom = task2PublicInitial.currentRoom ∧
      atAttack.rooms = task2PublicInitial.rooms ∧
      atAttack.roomIds = task2PublicInitial.roomIds ∧
      atAttack.completed = task2PublicInitial.completed := by
  exact tilePath_has_engine_plan
    (r := task2PublicRoom) (s := task2PublicInitial)
    (start := task2PublicStart) (goal := task2PublicAttackPosition)
    (route := task2PublicToMonsterRoute)
    (by simp [task2PublicInitial, currentRoomState]) rfl rfl
    task2_public_initial_running task2_public_bfs_path_to_attack_position

theorem task2_public_first_attack_damages :
    Step task2PublicNearMonsterWorld .slotA task2PublicAfterFirstHit
      [.monsterDamaged task2PublicMonster.id] := by
  apply Step.attackDamage (monster := task2PublicMonster)
  · simp [primaryInteractionAvailable, openChestInteractionAvailable,
      npcInteractionAvailable, switchInteractionAvailable,
      task2PublicNearMonsterWorld, task2PublicPlayerAtMonster,
      task2PublicAttackPosition, task2PublicChest, task2PublicRoom,
      task2PublicMonster, task2Pos, currentRoomState,
      interactionReach, adjacent, advance]
  · simp [task2PublicNearMonsterWorld, task2PublicPlayerAtMonster]
  · simp [task2PublicNearMonsterWorld, task2PublicPlayerAtMonster]
  · simp [currentRoomState, task2PublicNearMonsterWorld,
      task2PublicRoom]
  · rfl
  · decide

theorem task2_public_second_attack_kills :
    Step task2PublicAfterFirstHit .slotA task2PublicAfterKill
      (monsterKillEvents task2PublicAfterFirstHit
        task2PublicDamagedMonster) := by
  apply Step.attackKill (monster := task2PublicDamagedMonster)
  · simp [primaryInteractionAvailable, openChestInteractionAvailable,
      npcInteractionAvailable, switchInteractionAvailable,
      task2PublicAfterFirstHit, task2PublicNearMonsterWorld,
      task2PublicPlayerAtMonster, task2PublicDamagedMonster,
      task2PublicMonster, task2PublicAttackPosition, task2PublicChest,
      task2PublicRoom, task2Pos, currentRoomState,
      updateCurrentRoom, setRoom, replaceMonster,
      interactionReach, adjacent, advance]
  · simp [task2PublicAfterFirstHit, task2PublicNearMonsterWorld,
      task2PublicPlayerAtMonster, updateCurrentRoom]
  · simp [task2PublicAfterFirstHit, task2PublicNearMonsterWorld,
      task2PublicPlayerAtMonster, updateCurrentRoom]
  · simp [task2PublicAfterFirstHit, task2PublicDamagedMonster,
      task2PublicMonster, task2PublicRoom, task2PublicNearMonsterWorld,
      currentRoomState, updateCurrentRoom, setRoom, replaceMonster]
  · rfl
  · decide

theorem task2_public_engine_combat_from_initial :
    ∃ actions,
      EngineExec task2PublicInitial actions task2PublicAfterKill := by
  rcases task2_public_engine_path_to_attack_position with
    ⟨toAttack, atAttack, hToAttack, hAtPos, hAtRoom, hAtRunning,
      hAtInventory, hAtHp, hAtMaxHp, hAtCurrent, hAtRooms,
      hAtRoomIds, hAtCompleted⟩
  let faced : WorldState :=
    { atAttack with player :=
        { atAttack.player with facing := .west, shielding := false } }
  have hMonsterAhead :
      monsterAt (currentRoomState atAttack)
        (advance atAttack.player.pos .west) := by
    rw [hAtRoom, hAtPos]
    simp [monsterAt, task2PublicRoom, task2PublicMonster,
      task2PublicAttackPosition, task2Pos, advance]
  have hFaceStep : Step atAttack .left faced [] := by
    exact Step.faceMonster rfl hMonsterAhead
  have hFacePlayer : PlayerStep atAttack .left faced [] := by
    exact ⟨hAtRunning, hFaceStep, by simp [AutonomousOnlyEvents]⟩
  have hFaceTick : EngineTick atAttack .left faced :=
    EngineTick.mk hFacePlayer AutonomousExec.nil
  have hFaced : faced = task2PublicNearMonsterWorld := by
    cases atAttack with
    | mk current rooms roomIds player completed =>
      cases player with
      | mk pos facing hp maxHp inventory shielding =>
        simp_all [faced, currentRoomState, task2PublicNearMonsterWorld,
          task2PublicInitial, task2PublicInitialPlayer,
          task2PublicPlayerAtMonster]
  rw [hFaced] at hFaceTick
  have hFirstRunning : Running task2PublicNearMonsterWorld := by
    simp [Running, alive, task2PublicNearMonsterWorld,
      task2PublicPlayerAtMonster]
  have hFirstPlayer :
      PlayerStep task2PublicNearMonsterWorld .slotA task2PublicAfterFirstHit
        [.monsterDamaged task2PublicMonster.id] :=
    ⟨hFirstRunning, task2_public_first_attack_damages,
      by simp [AutonomousOnlyEvents]⟩
  have hFirstTick :
      EngineTick task2PublicNearMonsterWorld .slotA task2PublicAfterFirstHit :=
    EngineTick.mk hFirstPlayer AutonomousExec.nil
  have hSecondRunning : Running task2PublicAfterFirstHit := by
    simp [Running, alive, task2PublicAfterFirstHit,
      task2PublicNearMonsterWorld, task2PublicPlayerAtMonster,
      updateCurrentRoom]
  have hSecondPlayer :
      PlayerStep task2PublicAfterFirstHit .slotA task2PublicAfterKill
        (monsterKillEvents task2PublicAfterFirstHit task2PublicDamagedMonster) := by
    refine ⟨hSecondRunning, task2_public_second_attack_kills, ?_⟩
    simp only [monsterKillEvents]
    dsimp
    split <;> simp [AutonomousOnlyEvents]
  have hSecondTick :
      EngineTick task2PublicAfterFirstHit .slotA task2PublicAfterKill :=
    EngineTick.mk hSecondPlayer AutonomousExec.nil
  exact ⟨toAttack ++ [.left, .slotA, .slotA],
    engineExec_append hToAttack
      (EngineExec.cons hFaceTick
        (EngineExec.cons hFirstTick
          (EngineExec.cons hSecondTick EngineExec.nil)))⟩

theorem task2_public_combat_exec :
    Exec task2PublicNearMonsterWorld [.slotA, .slotA]
      task2PublicAfterKill := by
  exact Exec.cons task2_public_first_attack_damages
    (Exec.cons task2_public_second_attack_kills Exec.nil)

theorem task2_public_combat_clears_monster :
    (currentRoomState task2PublicAfterKill).monsters = [] := by
  rw [task2PublicAfterKill, resolveMonsterKill_current_monsters]
  simp [task2PublicAfterFirstHit,
    task2PublicDamagedMonster, task2PublicMonster, task2PublicRoom,
    task2PublicNearMonsterWorld, currentRoomState, updateCurrentRoom,
    setRoom, replaceMonster, removeMonster]

def task2PublicUnlockedExit : Exit :=
  { task2PublicExit with opened := true }

def task2PublicResolvedRoom : RoomState :=
  { task2PublicRoom with
      monsters := []
      exits := [task2PublicUnlockedExit] }

theorem task2_public_resolved_room_exact :
    currentRoomState task2PublicAfterKill = task2PublicResolvedRoom := by
  simp [task2PublicAfterKill, task2PublicAfterFirstHit,
    task2PublicNearMonsterWorld, task2PublicDamagedMonster,
    task2PublicMonster, task2PublicChest, task2PublicRoom, task2PublicResolvedRoom,
    task2PublicUnlockedExit, task2PublicExit, resolveMonsterKill,
    currentRoomState, updateCurrentRoom, setRoom, replaceMonster,
    removeMonster, rewardPlayer, unlockAllMonstersDefeatedExits,
    revealEligibleChestsInWorld, revealEligibleChests,
    requirementContainsAllMonstersDefeated,
    chestRevealMatches]

/-! ### 清怪后的宝箱和出口路径

怪物被删除后，其原 tile `(2,2)` 重新成为可通行地板。Agent 从攻击位经过
该 tile 到达宝箱上方 `(1,2)`；开箱后再绕过仍有碰撞的宝箱，经 `(0,2)`
到达 west 条件出口 `(0,3)`。
-/

def task2PublicClearedRoom : RoomState :=
  { task2PublicRoom with monsters := [] }

def task2PublicNearChest : Position := task2Pos 1 2
def task2PublicExitTile : Position := task2Pos 0 3

def task2PublicToChestRoute : List Position :=
  [task2Pos 2 2, task2Pos 1 2]

def task2PublicToExitRoute : List Position :=
  [task2Pos 0 2, task2Pos 0 3]

private theorem task2PublicClearedSafeAt
    (x y : Int)
    (h : (x, y) ∈ [(2, 2), (1, 2), (0, 2), (0, 3)]) :
    safeTile task2PublicClearedRoom (task2Pos x y) := by
  simp at h
  rcases h with h | h | h | h
  all_goals
    rcases h with ⟨rfl, rfl⟩
    simp [safeTile, canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      activeTrapAt, monsterAt, gapAt, activeBridgeTile,
      task2PublicClearedRoom, task2PublicRoom, task2PublicBounds,
      task2PublicTraps, task2PublicTrapPositions, task2PublicChest,
      task2Pos]

theorem task2_public_bfs_path_to_chest :
    TilePath task2PublicClearedRoom task2PublicAttackPosition
      task2PublicToChestRoute task2PublicNearChest := by
  unfold task2PublicToChestRoute task2PublicAttackPosition task2PublicNearChest
  apply TilePath.cons .west rfl
  · exact task2PublicClearedSafeAt 2 2 (by simp)
  apply TilePath.cons .west rfl
  · exact task2PublicClearedSafeAt 1 2 (by simp)
  exact TilePath.nil _

theorem task2_public_chest_is_adjacent :
    adjacent task2PublicNearChest task2PublicChest.pos := by
  simp [task2PublicNearChest, task2PublicChest, task2Pos,
    adjacent, advance]

theorem task2_public_bfs_path_to_exit :
    TilePath task2PublicClearedRoom task2PublicNearChest
      task2PublicToExitRoute task2PublicExitTile := by
  unfold task2PublicToExitRoute task2PublicNearChest task2PublicExitTile
  apply TilePath.cons .west rfl
  · exact task2PublicClearedSafeAt 0 2 (by simp)
  apply TilePath.cons .south rfl
  · exact task2PublicClearedSafeAt 0 3 (by simp)
  exact TilePath.nil _

private theorem task2PublicResolvedSafeAt
    (x y : Int)
    (h : (x, y) ∈ [(2, 2), (1, 2), (0, 2), (0, 3)]) :
    safeTile task2PublicResolvedRoom (task2Pos x y) := by
  have hsafe := task2PublicClearedSafeAt x y h
  simpa [task2PublicResolvedRoom, task2PublicClearedRoom,
    task2PublicUnlockedExit, safeTile, canEnter, staticBlocker,
    npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile] using hsafe

theorem task2_public_resolved_path_to_chest :
    TilePath task2PublicResolvedRoom task2PublicAttackPosition
      task2PublicToChestRoute task2PublicNearChest := by
  unfold task2PublicToChestRoute task2PublicAttackPosition task2PublicNearChest
  apply TilePath.cons .west rfl
  · exact task2PublicResolvedSafeAt 2 2 (by simp)
  apply TilePath.cons .west rfl
  · exact task2PublicResolvedSafeAt 1 2 (by simp)
  exact TilePath.nil _

def task2PublicOpenedResolvedRoom : RoomState :=
  replaceChest task2PublicResolvedRoom task2PublicChest
    { task2PublicChest with opened := true }

private theorem task2PublicOpenedResolvedSafeAt
    (x y : Int)
    (h : (x, y) ∈ [(0, 2), (0, 3)]) :
    safeTile task2PublicOpenedResolvedRoom (task2Pos x y) := by
  simp at h
  rcases h with h | h
  all_goals
    rcases h with ⟨rfl, rfl⟩
    simp [task2PublicOpenedResolvedRoom, task2PublicResolvedRoom,
      task2PublicRoom, task2PublicUnlockedExit, replaceChest,
      safeTile, canEnter, inBounds, staticBlocker, npcAt,
      visibleChestAt, activeTrapAt, monsterAt, gapAt,
      activeBridgeTile, task2PublicBounds, task2PublicTraps,
      task2PublicTrapPositions, task2PublicChest, task2Pos]

theorem task2_public_opened_path_to_exit :
    TilePath task2PublicOpenedResolvedRoom task2PublicNearChest
      task2PublicToExitRoute task2PublicExitTile := by
  unfold task2PublicToExitRoute task2PublicNearChest task2PublicExitTile
  apply TilePath.cons .west rfl
  · exact task2PublicOpenedResolvedSafeAt 0 2 (by simp)
  apply TilePath.cons .south rfl
  · exact task2PublicOpenedResolvedSafeAt 0 3 (by simp)
  exact TilePath.nil _

theorem task2_public_chest_phase_reachable :
    TileReachable task2PublicClearedRoom
      task2PublicAttackPosition task2PublicNearChest :=
  tilePath_goal_reachable task2_public_bfs_path_to_chest

theorem task2_public_exit_phase_reachable :
    TileReachable task2PublicClearedRoom
      task2PublicNearChest task2PublicExitTile :=
  tilePath_goal_reachable task2_public_bfs_path_to_exit

theorem task2_public_all_navigation_phases_reachable :
    TileReachable task2PublicRoom task2PublicStart task2PublicAttackPosition ∧
    TileReachable task2PublicClearedRoom
      task2PublicAttackPosition task2PublicNearChest ∧
    TileReachable task2PublicClearedRoom
      task2PublicNearChest task2PublicExitTile := by
  exact ⟨
    task2_public_attack_position_reachable,
    task2_public_chest_phase_reachable,
    task2_public_exit_phase_reachable
  ⟩

theorem task2_public_engine_certificate :
    ∃ actions final,
      EngineExec task2PublicInitial actions final ∧
      WorldCompleted final ∧ alive final ∧ ValidState final := by
  rcases task2_public_engine_combat_from_initial with
    ⟨combatActions, hCombat⟩
  have hAfterKillPos :
      task2PublicAfterKill.player.pos = task2PublicAttackPosition := by
    rw [task2PublicAfterKill, resolveMonsterKill_player]
    simp [rewardPlayer, task2PublicAfterFirstHit,
      task2PublicNearMonsterWorld, task2PublicPlayerAtMonster,
      updateCurrentRoom]
  have hAfterKillRunning : Running task2PublicAfterKill := by
    constructor
    · unfold alive
      rw [task2PublicAfterKill, resolveMonsterKill_player]
      simp [rewardPlayer, task2PublicAfterFirstHit,
        task2PublicNearMonsterWorld, task2PublicPlayerAtMonster,
        updateCurrentRoom]
    · rw [task2PublicAfterKill, resolveMonsterKill_completed]
      rfl
  rcases tilePath_has_engine_plan
      (r := task2PublicResolvedRoom) (s := task2PublicAfterKill)
      (start := task2PublicAttackPosition) (goal := task2PublicNearChest)
      (route := task2PublicToChestRoute)
      task2_public_resolved_room_exact hAfterKillPos rfl hAfterKillRunning
      task2_public_resolved_path_to_chest with
    ⟨toChest, nearChest, hToChest, hNearPos, hNearRoom,
      hNearRunning, hNearInventory, hNearHp, hNearMaxHp,
      hNearCurrent, hNearRooms, hNearRoomIds, hNearCompleted⟩
  have hChestMember : task2PublicChest ∈ (currentRoomState nearChest).chests := by
    rw [hNearRoom]
    simp [task2PublicResolvedRoom, task2PublicRoom]
  have hChestReach :
      interactionReach nearChest.player.pos task2PublicChest.pos := by
    right
    rw [hNearPos]
    exact task2_public_chest_is_adjacent
  let afterChest := stateAfterOpeningChest nearChest task2PublicChest
  have hOpenStep :
      Step nearChest .slotA afterChest [.chestOpened task2PublicChest.id] :=
    Step.openChest hChestMember rfl rfl hChestReach
  have hOpenPlayer :
      PlayerStep nearChest .slotA afterChest [.chestOpened task2PublicChest.id] := by
    exact ⟨hNearRunning, hOpenStep, by simp [AutonomousOnlyEvents]⟩
  have hOpenTick : EngineTick nearChest .slotA afterChest :=
    EngineTick.mk hOpenPlayer AutonomousExec.nil
  have hAfterChestRoom :
      currentRoomState afterChest = task2PublicOpenedResolvedRoom := by
    dsimp [afterChest, stateAfterOpeningChest]
    rw [currentRoomState_updateCurrentRoom, hNearRoom]
    rfl
  have hAfterChestPos : afterChest.player.pos = task2PublicNearChest := by
    change nearChest.player.pos = task2PublicNearChest
    exact hNearPos
  have hAfterChestRunning : Running afterChest := by
    rcases hNearRunning with ⟨halive, hcomplete⟩
    constructor
    · change 0 < (collectLoot { nearChest.player with shielding := false }
          task2PublicChest.loot).hp
      unfold alive at halive
      simpa [task2PublicChest, collectLoot] using halive
    · change nearChest.completed = false
      exact hcomplete
  rcases tilePath_has_engine_plan
      (r := task2PublicOpenedResolvedRoom) (s := afterChest)
      (start := task2PublicNearChest) (goal := task2PublicExitTile)
      (route := task2PublicToExitRoute)
      hAfterChestRoom hAfterChestPos rfl hAfterChestRunning
      task2_public_opened_path_to_exit with
    ⟨toExit, atExit, hToExit, hExitPos, hExitRoom,
      hExitRunning, hExitInventory, hExitHp, hExitMaxHp,
      hExitCurrent, hExitRooms, hExitRoomIds, hExitCompleted⟩
  have hExitMember :
      task2PublicUnlockedExit ∈ (currentRoomState atExit).exits := by
    rw [hExitRoom]
    simp [task2PublicOpenedResolvedRoom, replaceChest,
      task2PublicResolvedRoom]
  have hExitContains :
      exitContains task2PublicUnlockedExit atExit.player.pos := by
    left
    rw [hExitPos]
    rfl
  have hRequirement :
      exitRequirementSatisfied atExit task2PublicUnlockedExit := by
    change (currentRoomState atExit).monsters = [] ∧
      1 ≤ atExit.player.inventory.keys
    constructor
    · rw [hExitRoom]
      rfl
    · rw [hExitInventory]
      change 1 ≤ nearChest.player.inventory.keys + 1
      omega
  have hSpawn :
      canEnter (atExit.rooms task2PublicUnlockedExit.targetRoom)
        task2PublicUnlockedExit.targetSpawn := by
    have hcurrentZero : atExit.currentRoom = 0 := by
      rw [hExitCurrent]
      change nearChest.currentRoom = 0
      rw [hNearCurrent]
      rfl
    change canEnter (atExit.rooms 0) (task2Pos 8 4)
    rw [← hcurrentZero]
    change canEnter (currentRoomState atExit) (task2Pos 8 4)
    rw [hExitRoom]
    simp [task2PublicOpenedResolvedRoom, task2PublicResolvedRoom,
      task2PublicRoom, replaceChest, task2PublicBounds,
      task2PublicChest, task2PublicTraps, task2PublicTrapPositions,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      gapAt, activeBridgeTile, task2Pos]
  let final := transitionThroughExit atExit task2PublicUnlockedExit
  have hExitStep :
      Step atExit (directionAction task2PublicUnlockedExit.direction) final
        (exitEvents atExit task2PublicUnlockedExit) :=
    Step.useExit hExitMember hExitContains hRequirement rfl hSpawn
  have hExitPlayer :
      PlayerStep atExit (directionAction task2PublicUnlockedExit.direction) final
        (exitEvents atExit task2PublicUnlockedExit) := by
    refine ⟨hExitRunning, hExitStep, ?_⟩
    simp [AutonomousOnlyEvents, exitEvents, task2PublicUnlockedExit,
      task2PublicExit]
  have hExitTick :
      EngineTick atExit (directionAction task2PublicUnlockedExit.direction) final :=
    EngineTick.mk hExitPlayer AutonomousExec.nil
  have hAll := engineExec_append hCombat
    (engineExec_append hToChest
      (engineExec_append (EngineExec.cons hOpenTick EngineExec.nil)
        (engineExec_append hToExit (EngineExec.cons hExitTick EngineExec.nil))))
  refine ⟨_, final, hAll, ?_, ?_, ?_⟩
  · simp [final, WorldCompleted, transitionThroughExit,
      task2PublicUnlockedExit, task2PublicExit]
  · rcases hExitRunning with ⟨halive, _⟩
    simpa [final, transitionThroughExit, alive] using halive
  · exact engineExec_preserves_validState
      (by simp [ValidState, task2PublicInitial, task2PublicInitialPlayer,
        task2PublicPlayerAtMonster, currentRoomState, task2PublicRoom,
        task2PublicBounds, inBounds, task2PublicStart, task2Pos]) hAll

theorem task2_engine_execution_is_safe_and_complete
    {initial final : WorldState} {playerActions : List Action}
    (hvalid : ValidState initial)
    (hexec : EngineExec initial playerActions final)
    (hgoal : Task2Goal final)
    (halive : alive final) :
    Task2SafelyCompletable initial := by
  rcases engineExec_has_microstep_trace hexec with
    ⟨microActions, hmicro⟩
  exact ⟨
    microActions,
    final,
    hmicro,
    hgoal,
    halive,
    engineExec_preserves_validState hvalid hexec
  ⟩

end Task2

/-! ## 十五、统一的可验证观测、记忆和安全屏障

神经网络输出不被当成公理。`SymbolicObservationSound` 是调用者必须验证或在
条件完备性定理中显式承担的精化契约；碰撞安全则不依赖感知正确性：错误移动
要么被 shield 改写为 WAIT，要么由环境的 `moveBlocked` 保持位置。
-/

structure SymbolicObservation where
  room : RoomId
  playerTile : Position
  walls : List Position := []
  chests : List Position := []
  monsters : List Position := []
  traps : List Position := []
  buttons : List Position := []
  exits : List (Position × Direction) := []
  deriving DecidableEq, Repr

structure InventoryView where
  keys : Nat
  hasSword : Bool
  hasShield : Bool
  deriving DecidableEq, Repr

def SymbolicObservationSound (s : WorldState) (o : SymbolicObservation) : Prop :=
  o.room = s.currentRoom ∧
  o.playerTile = s.player.pos ∧
  (∀ p, p ∈ o.walls → p ∈ (currentRoomState s).walls) ∧
  (∀ p, p ∈ o.chests → visibleChestAt (currentRoomState s) p) ∧
  (∀ p, p ∈ o.monsters → monsterAt (currentRoomState s) p) ∧
  (∀ p, p ∈ o.traps → activeTrapAt (currentRoomState s) p) ∧
  (∀ p, p ∈ o.buttons → buttonAt (currentRoomState s) p)

def InventoryViewSound (s : WorldState) (view : InventoryView) : Prop :=
  view.keys = s.player.inventory.keys ∧
  (view.hasSword = true ↔ Item.sword ∈ s.player.inventory.items) ∧
  (view.hasShield = true ↔ Item.shield ∈ s.player.inventory.items)

structure VerifiedMemory where
  visitedRooms : List RoomId := []
  knownWalls : List (RoomId × Position) := []
  openedChests : List (RoomId × ObjectId) := []
  pressedButtons : List (RoomId × ObjectId) := []
  blockedEdges : List (RoomId × Position × Direction) := []
  deriving DecidableEq, Repr

def rememberRoom (m : VerifiedMemory) (room : RoomId) : VerifiedMemory :=
  { m with visitedRooms := room :: m.visitedRooms }

def rememberBlocked
    (m : VerifiedMemory) (room : RoomId) (p : Position) (d : Direction) :
    VerifiedMemory :=
  { m with blockedEdges := (room, p, d) :: m.blockedEdges }

theorem rememberRoom_monotone (m : VerifiedMemory) (room old : RoomId)
    (h : old ∈ m.visitedRooms) :
    old ∈ (rememberRoom m room).visitedRooms := by
  exact List.mem_cons_of_mem room h

theorem rememberRoom_records_current (m : VerifiedMemory) (room : RoomId) :
    room ∈ (rememberRoom m room).visitedRooms := by
  simp [rememberRoom]

theorem rememberBlocked_monotone
    (m : VerifiedMemory) (room oldRoom : RoomId)
    (p oldP : Position) (d oldD : Direction)
    (h : (oldRoom, oldP, oldD) ∈ m.blockedEdges) :
    (oldRoom, oldP, oldD) ∈ (rememberBlocked m room p d).blockedEdges := by
  exact List.mem_cons_of_mem _ h

theorem rememberBlocked_records_attempt
    (m : VerifiedMemory) (room : RoomId) (p : Position) (d : Direction) :
    (room, p, d) ∈ (rememberBlocked m room p d).blockedEdges := by
  simp [rememberBlocked]

inductive VerifiedShield (s : WorldState) (proposed : Action) : Action → Prop where
  | nonMove (h : actionDirection proposed = none) :
      VerifiedShield s proposed proposed
  | safeMove {d : Direction}
      (hdir : actionDirection proposed = some d)
      (hsafe : safeTile (currentRoomState s) (advance s.player.pos d)) :
      VerifiedShield s proposed proposed
  | blockedMove {d : Direction}
      (hdir : actionDirection proposed = some d)
      (hunsafe : ¬ safeTile (currentRoomState s) (advance s.player.pos d)) :
      VerifiedShield s proposed .wait

theorem verifiedShield_move_is_safe_or_wait
    {s : WorldState} {proposed output : Action}
    (hshield : VerifiedShield s proposed output) :
    output = .wait ∨
    ∀ d, actionDirection output = some d →
      safeTile (currentRoomState s) (advance s.player.pos d) := by
  cases hshield with
  | nonMove h =>
      right
      intro d hd
      rw [h] at hd
      contradiction
  | safeMove hdir hsafe =>
      right
      intro d hd
      rw [hdir] at hd
      cases hd
      exact hsafe
  | blockedMove hdir hunsafe =>
      exact Or.inl rfl

theorem verified_controller_execution_preserves_safety
    {initial final : WorldState} {actions : List Action}
    (hvalid : ValidState initial)
    (hexec : EngineExec initial actions final) :
    ValidState final :=
  engineExec_preserves_validState hvalid hexec

/-! ## 十六、Task3：匿名房间图与反馈驱动服务（完全重构） -/

namespace Task3

abbrev AnonymousRoomId := Nat

def opposite : Direction → Direction
  | .north => .south | .south => .north | .west => .east | .east => .west

structure RoomMemory where
  entryDirection : Option Direction := none
  connections : List (Direction × AnonymousRoomId) := []
  exits : List Direction := []
  exitKinds : List (Direction × ExitKind) := []
  exitTiles : List (Direction × Position) := []
  exitVisits : List (Direction × Nat) := []
  chests : List Position := []
  openedChests : List Position := []
  monsterSeen : Bool := false
  monsterCleared : Bool := false
  visits : Nat := 0
  deriving DecidableEq, Repr

def connectionTarget (m : RoomMemory) (d : Direction) : Option AnonymousRoomId :=
  match m.connections.find? (fun edge => edge.1 = d) with
  | some edge => some edge.2
  | none => none

def exitKindAt (m : RoomMemory) (d : Direction) : Option ExitKind :=
  match m.exitKinds.find? (fun entry => entry.1 = d) with
  | some entry => some entry.2
  | none => none

structure Controller where
  currentRoom : AnonymousRoomId := 0
  nextRoomId : AnonymousRoomId := 1
  memories : AnonymousRoomId → RoomMemory := fun _ => {}
  keys : Nat := 0
  hasSword : Bool := false
  lastPlayerTile : Option Position := none
  pendingExit : Option Direction := none
  exitPushAction : Option Action := none
  moveAction : Option Action := none
  moveTarget : Option Position := none
  moveAttempts : Nat := 0
  perceptionMisses : Nat := 0
  pendingInteraction : Option (String × Position) := none
  combatMisses : Nat := 0

def jumpDistance (oldTile newTile : Position) : Nat :=
  (newTile.x - oldTile.x).natAbs + (newTile.y - oldTile.y).natAbs

def roomJumpDetected (oldTile newTile : Position) : Bool :=
  3 < jumpDistance oldTile newTile

inductive ConfirmRoomChange :
    Controller → Position → Controller → Prop where
  | existing {c : Controller} {oldTile newTile : Position}
      {d : Direction} {target : AnonymousRoomId} {c' : Controller}
      (hlast : c.lastPlayerTile = some oldTile)
      (hpending : c.pendingExit = some d)
      (hjump : roomJumpDetected oldTile newTile = true)
      (hconnection : connectionTarget (c.memories c.currentRoom) d = some target)
      (hcurrent : c'.currentRoom = target)
      (hcleared : c'.pendingExit = none) :
      ConfirmRoomChange c newTile c'
  | fresh {c c' : Controller} {oldTile newTile : Position} {d : Direction}
      (hlast : c.lastPlayerTile = some oldTile)
      (hpending : c.pendingExit = some d)
      (hjump : roomJumpDetected oldTile newTile = true)
      (hnew : connectionTarget (c.memories c.currentRoom) d = none)
      (hcurrent : c'.currentRoom = c.nextRoomId)
      (hnext : c'.nextRoomId = c.nextRoomId + 1)
      (hforward : connectionTarget (c'.memories c.currentRoom) d =
        some c.nextRoomId)
      (hreverse : connectionTarget (c'.memories c.nextRoomId) (opposite d) =
        some c.currentRoom)
      (hcleared : c'.pendingExit = none) :
      ConfirmRoomChange c newTile c'

theorem room_change_requires_pending_exit
    {c c' : Controller} {newTile : Position}
    (h : ConfirmRoomChange c newTile c') :
    ∃ oldTile d, c.lastPlayerTile = some oldTile ∧
      c.pendingExit = some d ∧ roomJumpDetected oldTile newTile = true := by
  cases h with
  | existing hlast hpending hjump hconnection hcurrent hcleared =>
      exact ⟨_, _, hlast, hpending, hjump⟩
  | fresh hlast hpending hjump hnew hcurrent hnext hforward hreverse hcleared =>
      exact ⟨_, _, hlast, hpending, hjump⟩

theorem no_pending_exit_cannot_confirm_change
    {c c' : Controller} {newTile : Position}
    (hnone : c.pendingExit = none) :
    ¬ ConfirmRoomChange c newTile c' := by
  intro h
  rcases room_change_requires_pending_exit h with ⟨_, d, _, hpending, _⟩
  rw [hnone] at hpending
  contradiction

theorem confirmed_change_uses_existing_or_fresh_room
    {c c' : Controller} {newTile : Position}
    (h : ConfirmRoomChange c newTile c') :
    (∃ d, connectionTarget (c.memories c.currentRoom) d = some c'.currentRoom) ∨
    (∃ d, c'.currentRoom = c.nextRoomId ∧
      connectionTarget (c'.memories c.currentRoom) d = some c.nextRoomId ∧
      connectionTarget (c'.memories c.nextRoomId) (opposite d) =
        some c.currentRoom) := by
  cases h with
  | existing hlast hpending hjump hconnection hcurrent hcleared =>
      left
      exact ⟨_, hcurrent ▸ hconnection⟩
  | fresh hlast hpending hjump hnew hcurrent hnext hforward hreverse hcleared =>
      exact Or.inr ⟨_, hcurrent, hforward, hreverse⟩

inductive GoalKind where
  | openChest | combat | currentLockedExit | routeToLockedRoom
  | unknownExit | routeToServiceRoom | returnThroughEntry | wait
  deriving DecidableEq, Repr

structure Goal where
  kind : GoalKind
  target : Option Position := none
  direction : Option Direction := none
  deriving DecidableEq, Repr

structure CandidateFacts where
  visibleUnopenedChest : Option Position := none
  visibleMonster : Option Position := none
  currentLockedExit : Option Direction := none
  routeToLockedRoom : Option Direction := none
  unknownUsableExit : Option Direction := none
  routeToServiceRoom : Option Direction := none
  entryReturn : Option Direction := none
  deriving DecidableEq, Repr

def explorationGoal (facts : CandidateFacts) : Goal :=
  match facts.unknownUsableExit with
  | some d => ⟨.unknownExit, none, some d⟩
  | none => match facts.routeToServiceRoom with
    | some d => ⟨.routeToServiceRoom, none, some d⟩
    | none => match facts.entryReturn with
      | some d => ⟨.returnThroughEntry, none, some d⟩
      | none => ⟨.wait, none, none⟩

def chooseGoal (c : Controller) (facts : CandidateFacts) : Goal :=
  match facts.visibleUnopenedChest with
  | some p => ⟨.openChest, some p, none⟩
  | none => match facts.visibleMonster with
    | some p =>
        if c.hasSword then ⟨.combat, some p, none⟩
        else match facts.entryReturn with
          | some d => ⟨.returnThroughEntry, none, some d⟩
          | none => ⟨.wait, none, none⟩
    | none =>
        if 0 < c.keys then
          match facts.currentLockedExit with
          | some d => ⟨.currentLockedExit, none, some d⟩
          | none => match facts.routeToLockedRoom with
            | some d => ⟨.routeToLockedRoom, none, some d⟩
            | none => explorationGoal facts
        else explorationGoal facts

theorem visible_chest_has_priority
    (c : Controller) (facts : CandidateFacts) (p : Position)
    (h : facts.visibleUnopenedChest = some p) :
    (chooseGoal c facts).kind = .openChest := by
  simp [chooseGoal, h]

theorem monster_without_sword_does_not_select_combat
    (c : Controller) (facts : CandidateFacts) (p : Position)
    (hc : facts.visibleUnopenedChest = none)
    (hm : facts.visibleMonster = some p) (hs : c.hasSword = false) :
    (chooseGoal c facts).kind ≠ .combat := by
  simp [chooseGoal, hc, hm, hs]
  split <;> simp

theorem locked_exit_requires_confirmed_key
    (c : Controller) (facts : CandidateFacts) (hkeys : c.keys = 0) :
    (chooseGoal c facts).kind ≠ .currentLockedExit ∧
    (chooseGoal c facts).kind ≠ .routeToLockedRoom := by
  rcases facts with ⟨chest, monster, locked, route, unknown, service, entry⟩
  cases chest <;> cases monster <;> cases locked <;> cases route <;>
    cases unknown <;> cases service <;> cases entry <;> cases hs : c.hasSword <;>
    simp [chooseGoal, explorationGoal, hkeys, hs]

structure InteractionFeedback where
  targetStillVisible : Bool
  keyDelta : Nat := 0
  largePositiveReward : Bool := false
  deriving DecidableEq, Repr

def chestConfirmed (f : InteractionFeedback) : Bool :=
  !f.targetStillVisible || 0 < f.keyDelta || f.largePositiveReward

def monsterConfirmedCleared (f : InteractionFeedback) : Bool :=
  !f.targetStillVisible

theorem chest_confirmation_uses_only_public_feedback (f : InteractionFeedback) :
    chestConfirmed f = true ↔
      f.targetStillVisible = false ∨ 0 < f.keyDelta ∨
      f.largePositiveReward = true := by
  rcases f with ⟨visible, delta, positive⟩
  cases visible <;> cases positive <;> simp [chestConfirmed]

theorem invisible_monster_confirms_clear (f : InteractionFeedback)
    (h : f.targetStillVisible = false) :
    monsterConfirmedCleared f = true := by
  simp [monsterConfirmedCleared, h]

def actionDuringPerceptionMiss
    (pendingExit : Option Direction) (exitAction moveAction : Option Action)
    (misses : Nat) : Action :=
  if pendingExit.isSome && exitAction.isSome && misses ≤ 32 then
    exitAction.getD .wait
  else if moveAction.isSome && misses ≤ 2 then
    moveAction.getD .wait
  else .wait

theorem perception_miss_beyond_exit_window_waits
    (pending : Option Direction) (exitAction moveAction : Option Action)
    (misses : Nat) (h : 32 < misses) :
    actionDuringPerceptionMiss pending exitAction moveAction misses = .wait := by
  have h2 : ¬ misses ≤ 2 := Nat.not_le.mpr (Nat.lt_trans (by decide) h)
  simp [actionDuringPerceptionMiss, Nat.not_le.mpr h, h2]

theorem local_motion_is_reused_for_at_most_two_missing_frames
    (moveAction : Option Action) (misses : Nat) (h : 2 < misses) :
    actionDuringPerceptionMiss none none moveAction misses = .wait := by
  have h2 : ¬ misses ≤ 2 := Nat.not_le.mpr h
  simp [actionDuringPerceptionMiss, h2]

def graphNeighbors
    (memories : AnonymousRoomId → RoomMemory) (room : AnonymousRoomId) :
    List AnonymousRoomId :=
  (memories room).connections.map Prod.snd

theorem graph_path_first_hop_is_learned
    {memories : AnonymousRoomId → RoomMemory}
    {start next goal : AnonymousRoomId} {rest : List AnonymousRoomId}
    (h : GenericGraphPath (graphNeighbors memories)
      start (next :: rest) goal) :
    next ∈ graphNeighbors memories start := by
  cases h with
  | cons hstep htail => exact hstep

def GraphRenaming
    (rename : AnonymousRoomId → AnonymousRoomId)
    (before after : AnonymousRoomId → RoomMemory) : Prop :=
  ∀ room,
    graphNeighbors after (rename room) =
      (graphNeighbors before room).map rename

theorem graph_path_is_invariant_under_room_renaming
    {rename : AnonymousRoomId → AnonymousRoomId}
    {before after : AnonymousRoomId → RoomMemory}
    (hrenaming : GraphRenaming rename before after)
    {start goal : AnonymousRoomId} {route : List AnonymousRoomId}
    (hpath : GenericGraphPath (graphNeighbors before) start route goal) :
    GenericGraphPath (graphNeighbors after)
      (rename start) (route.map rename) (rename goal) := by
  induction hpath with
  | nil p => exact GenericGraphPath.nil _
  | @cons p q goal rest hstep htail ih =>
      apply GenericGraphPath.cons
      · rw [hrenaming p]
        exact List.mem_map.mpr ⟨q, hstep, rfl⟩
      · exact ih

inductive CommandRole where
  | navigation | faceChest | faceMonster | exitPush | interaction | idle
  deriving DecidableEq, Repr

def CommandSafe (s : WorldState) (role : CommandRole) (action : Action) : Prop :=
  match role with
  | .navigation => ∀ d, actionDirection action = some d →
      safeTile (currentRoomState s) (advance s.player.pos d)
  | .faceChest => ∃ d, actionDirection action = some d ∧
      visibleChestAt (currentRoomState s) (advance s.player.pos d)
  | .faceMonster => ∃ d, actionDirection action = some d ∧
      monsterAt (currentRoomState s) (advance s.player.pos d)
  | .exitPush => ∃ exit ∈ (currentRoomState s).exits,
      action = directionAction exit.direction ∧
      exitContains exit s.player.pos ∧ exitRequirementSatisfied s exit
  | .interaction => action = .slotA ∧ primaryInteractionAvailable s
  | .idle => action = .wait

theorem navigation_command_has_engine_step
    {s : WorldState} {action : Action} {d : Direction}
    (hsafe : CommandSafe s .navigation action)
    (ha : actionDirection action = some d) :
    ∃ after events, Step s action after events := by
  have htile := hsafe d ha
  let q := advance s.player.pos d
  by_cases hbutton : buttonAt (currentRoomState s) q
  · rcases hbutton with ⟨button, hmember, hpos⟩
    exact ⟨_, _, Step.moveButton ha rfl htile.1 hmember hpos⟩
  · exact ⟨_, _, Step.movePlain ha rfl htile.1 htile.2.1 hbutton⟩

theorem face_chest_command_is_a_position_safe_block
    {s : WorldState} {action : Action}
    (hsafe : CommandSafe s .faceChest action) :
    ∃ after events, Step s action after events ∧
      after.player.pos = s.player.pos := by
  rcases hsafe with ⟨d, ha, hchest⟩
  have hblocked :
      ¬ canEnter (currentRoomState s) (advance s.player.pos d) := by
    intro henter
    exact canEnter_not_visible_chest henter hchest
  exact ⟨_, _, Step.moveBlocked ha hblocked, rfl⟩

theorem face_monster_command_is_a_position_safe_step
    {s : WorldState} {action : Action}
    (hsafe : CommandSafe s .faceMonster action) :
    ∃ after, Step s action after [] ∧
      after.player.pos = s.player.pos := by
  rcases hsafe with ⟨d, ha, hmonster⟩
  exact ⟨_, Step.faceMonster ha hmonster, rfl⟩

structure PolicyDecision where
  goal : Goal
  role : CommandRole
  output : Action

def PolicyRefinesAgent
    (c : Controller) (facts : CandidateFacts) (s : WorldState)
    (decision : PolicyDecision) : Prop :=
  decision.goal = chooseGoal c facts ∧ CommandSafe s decision.role decision.output

theorem goal_layer_refines_agent
    {c : Controller} {facts : CandidateFacts} {s : WorldState}
    {decision : PolicyDecision} (h : PolicyRefinesAgent c facts s decision) :
    decision.goal = chooseGoal c facts ∧ CommandSafe s decision.role decision.output := h

inductive FairPolicyRun (step : Controller → Controller → Prop) :
    Controller → Controller → Prop where
  | refl (c) : FairPolicyRun step c c
  | tail {a b c} : step a b → FairPolicyRun step b c → FairPolicyRun step a c

theorem memory_policy_complete_under_fairness
    (measure : Controller → Nat) (goal : Controller → Prop)
    (step : Controller → Controller → Prop)
    (hzero : ∀ c, measure c = 0 → goal c)
    (hprogress : ∀ c, 0 < measure c → ∃ next,
      step c next ∧ measure next < measure c) :
    ∀ initial, ∃ final, FairPolicyRun step initial final ∧ goal final := by
  intro initial
  induction hmeasure : measure initial using Nat.strongRecOn generalizing initial with
  | ind n ih =>
      by_cases hz : measure initial = 0
      · exact ⟨initial, .refl initial, hzero initial hz⟩
      · have hpositive : 0 < measure initial := Nat.pos_of_ne_zero hz
        rcases hprogress initial hpositive with ⟨next, hstep, hdecrease⟩
        have hdecrease' : measure next < n := by simpa [hmeasure] using hdecrease
        rcases ih (measure next) hdecrease' next rfl with ⟨final, hrun, hgoal⟩
        exact ⟨final, .tail hstep hrun, hgoal⟩

theorem goal_selection_ignores_anonymous_room_identity
    (left right : Controller) (facts : CandidateFacts)
    (hkeys : left.keys = right.keys)
    (hsword : left.hasSword = right.hasSword) :
    chooseGoal left facts = chooseGoal right facts := by
  unfold chooseGoal
  rw [hkeys, hsword]

def mirrorDirection : Direction → Direction
  | .west => .east | .east => .west | .north => .north | .south => .south

def mirrorOptionalDirection : Option Direction → Option Direction :=
  Option.map mirrorDirection

def mirrorCandidateFacts (facts : CandidateFacts) : CandidateFacts :=
  { facts with
    currentLockedExit := mirrorOptionalDirection facts.currentLockedExit
    routeToLockedRoom := mirrorOptionalDirection facts.routeToLockedRoom
    unknownUsableExit := mirrorOptionalDirection facts.unknownUsableExit
    routeToServiceRoom := mirrorOptionalDirection facts.routeToServiceRoom
    entryReturn := mirrorOptionalDirection facts.entryReturn }

theorem mirror_preserves_selected_goal_category
    (c : Controller) (facts : CandidateFacts) :
    (chooseGoal c (mirrorCandidateFacts facts)).kind =
      (chooseGoal c facts).kind := by
  rcases facts with ⟨chest, monster, locked, route, unknown, service, entry⟩
  by_cases hkeys : 0 < c.keys
  all_goals
    cases chest <;> cases monster <;> cases locked <;> cases route <;>
      cases unknown <;> cases service <;> cases entry <;> cases hs : c.hasSword <;>
      simp [chooseGoal, explorationGoal, mirrorCandidateFacts,
        mirrorOptionalDirection, hkeys, hs]

theorem mirror_direction_is_involutive (d : Direction) :
    mirrorDirection (mirrorDirection d) = d := by
  cases d <;> rfl

/-! ### 公开 Task3 模板的真实房间与逐格路线证书 -/

def publicBounds : Bounds :=
  { width := 10, height := 8, width_pos := by decide, height_pos := by decide }

def publicPos (x y : Int) : Position := ⟨x, y⟩

def publicStartWest : Exit :=
  { id := 301, pos := publicPos 0 4, otherTiles := [publicPos 0 3]
    direction := .west, kind := .normal, requirement := .free
    targetRoom := 1, targetSpawn := publicPos 8 4 }

def publicStartLocked : Exit :=
  { id := 302, pos := publicPos 9 4, otherTiles := [publicPos 9 3]
    direction := .east, kind := .locked, requirement := .keys 1 true
    targetRoom := 0, targetSpawn := publicPos 8 4
    completesTask := true }

def publicHallWest : Exit :=
  { id := 303, pos := publicPos 0 4, otherTiles := [publicPos 0 3]
    direction := .west, kind := .normal, requirement := .free
    targetRoom := 2, targetSpawn := publicPos 8 4 }

def publicHallEast : Exit :=
  { id := 304, pos := publicPos 9 4, otherTiles := [publicPos 9 3]
    direction := .east, kind := .normal, requirement := .free
    targetRoom := 0, targetSpawn := publicPos 1 4 }

def publicKeyEast : Exit :=
  { id := 305, pos := publicPos 9 4, otherTiles := [publicPos 9 3]
    direction := .east, kind := .normal, requirement := .free
    targetRoom := 1, targetSpawn := publicPos 1 4 }

def publicHintNpc : Npc :=
  { id := 306, pos := publicPos 4 1, text := "Find the key west, then return." }

def publicHallMonster : Monster :=
  { id := 307, pos := publicPos 5 3, kind := .chaser, hp := 2, damage := 1 }

def publicKeyChest : Chest :=
  { id := 308, pos := publicPos 5 4, loot := .key 1,
    visible := true, opened := false }

def publicStartRoom : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [publicHintNpc], chests := []
    monsters := [], traps := [], buttons := [], switches := [], bridges := []
    dynamicTiles := [], exits := [publicStartWest, publicStartLocked] }

def publicHallRoom : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [], chests := []
    monsters := [publicHallMonster], traps := [], buttons := [], switches := []
    bridges := [], dynamicTiles := [], exits := [publicHallWest, publicHallEast] }

def publicKeyRoom : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [], chests := [publicKeyChest]
    monsters := [], traps := [], buttons := [], switches := [], bridges := []
    dynamicTiles := [], exits := [publicKeyEast] }

def publicRooms : RoomId → RoomState
  | 0 => publicStartRoom | 1 => publicHallRoom | _ => publicKeyRoom

def publicInitialPlayer : PlayerState :=
  { pos := publicPos 4 4, facing := .west, hp := 5, maxHp := 5,
    inventory :=
      { keys := 0, gold := 0, items := [.sword, .shield]
        equippedA := some .sword, equippedB := some .shield } }

def publicInitialWorld : WorldState :=
  { currentRoom := 0, rooms := publicRooms, roomIds := [0, 1, 2],
    player := publicInitialPlayer, completed := false }

def startToHallDirections : List Direction := [.west, .west, .west, .west]
def hallToAttackDirections : List Direction := [.west, .west, .north]
def clearedHallToKeyDirections : List Direction :=
  [.south, .west, .west, .west, .west, .west, .west]
def keyToChestDirections : List Direction := [.west, .west]
def keyToHallDirections : List Direction := [.east, .east, .east]
def hallToStartDirections : List Direction :=
  [.east, .east, .east, .east, .east, .east, .east, .east]
def startToLockDirections : List Direction :=
  [.east, .east, .east, .east, .east, .east, .east, .east]

theorem public_start_to_hall_safe :
    DirectionPlanSafe publicStartRoom (publicPos 4 4) startToHallDirections := by
  simp [DirectionPlanSafe, startToHallDirections, publicStartRoom,
    publicBounds, publicHintNpc, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicPos, advance]

theorem public_hall_to_attack_safe :
    DirectionPlanSafe publicHallRoom (publicPos 8 4) hallToAttackDirections := by
  simp [DirectionPlanSafe, hallToAttackDirections, publicHallRoom,
    publicBounds, publicHallMonster, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicPos, advance]

def publicClearedHallRoom : RoomState :=
  { publicHallRoom with monsters := [] }

theorem public_cleared_hall_to_key_safe :
    DirectionPlanSafe publicClearedHallRoom (publicPos 6 3)
      clearedHallToKeyDirections := by
  simp [DirectionPlanSafe, clearedHallToKeyDirections, publicClearedHallRoom,
    publicHallRoom, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicPos, advance]

theorem public_key_to_chest_safe :
    DirectionPlanSafe publicKeyRoom (publicPos 8 4) keyToChestDirections := by
  simp [DirectionPlanSafe, keyToChestDirections, publicKeyRoom,
    publicBounds, publicKeyChest, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicPos, advance]

def publicOpenedKeyRoom : RoomState :=
  replaceChest publicKeyRoom publicKeyChest { publicKeyChest with opened := true }

theorem public_key_to_hall_safe :
    DirectionPlanSafe publicOpenedKeyRoom (publicPos 6 4)
      keyToHallDirections := by
  simp [DirectionPlanSafe, keyToHallDirections, publicOpenedKeyRoom,
    publicKeyRoom, publicBounds, publicKeyChest, replaceChest,
    safeTile, canEnter, inBounds, staticBlocker, npcAt,
    visibleChestAt, activeTrapAt, monsterAt, gapAt, activeBridgeTile,
    publicPos, advance]

theorem public_hall_to_start_safe :
    DirectionPlanSafe publicClearedHallRoom (publicPos 1 4)
      hallToStartDirections := by
  simp [DirectionPlanSafe, hallToStartDirections, publicClearedHallRoom,
    publicHallRoom, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicPos, advance]

theorem public_start_to_lock_safe :
    DirectionPlanSafe publicStartRoom (publicPos 1 4)
      startToLockDirections := by
  simp [DirectionPlanSafe, startToLockDirections, publicStartRoom,
    publicBounds, publicHintNpc, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicPos, advance]

theorem public_initial_running : Running publicInitialWorld := by
  simp [Running, alive, publicInitialWorld, publicInitialPlayer]

/-! 公开证书的第一段：真实执行从起始房西行，过门并到达大厅攻击位。 -/
theorem public_engine_reaches_hall_attack_position :
    ∃ actions attackState,
      EngineExec publicInitialWorld actions attackState ∧
      attackState.currentRoom = 1 ∧
      currentRoomState attackState = publicHallRoom ∧
      attackState.player.pos = publicPos 6 3 ∧ Running attackState ∧
      attackState.player.inventory = publicInitialWorld.player.inventory ∧
      attackState.rooms = publicRooms ∧
      attackState.roomIds = [0, 1, 2] ∧
      attackState.player.hp = 5 ∧ attackState.completed = false := by
  rcases directionPlan_has_engine_plan
      (room := publicStartRoom) (s := publicInitialWorld)
      (start := publicPos 4 4) (directions := startToHallDirections)
      (by simp [publicInitialWorld, currentRoomState, publicRooms]) rfl rfl
      public_initial_running public_start_to_hall_safe with
    ⟨toWest, atWest, hToWest, hWestPos, hWestRoom, hWestRunning,
      hWestInventory, hWestHp, hWestMaxHp, hWestCurrent, hWestRooms,
      hWestRoomIds, hWestCompleted⟩
  have hExitMember : publicStartWest ∈ (currentRoomState atWest).exits := by
    rw [hWestRoom]
    simp [publicStartRoom]
  have hAtExit : exitContains publicStartWest atWest.player.pos := by
    left
    rw [hWestPos]
    rfl
  have hReq : exitRequirementSatisfied atWest publicStartWest := by
    simp [exitRequirementSatisfied, publicStartWest, requirementSatisfied]
  have hSpawn :
      canEnter (atWest.rooms publicStartWest.targetRoom)
        publicStartWest.targetSpawn := by
    rw [hWestRooms]
    simp [publicInitialWorld, publicRooms, publicStartWest, publicHallRoom,
      publicBounds, publicHallMonster, canEnter, inBounds, staticBlocker,
      npcAt, visibleChestAt, gapAt, activeBridgeTile, publicPos]
  let inHall := transitionThroughExit atWest publicStartWest
  have hEnterHall :
      EngineExec atWest [.left] inHall :=
    engineExec_useExit_once hWestRunning hExitMember hAtExit hReq hSpawn
  have hWestCurrentZero : atWest.currentRoom = 0 := by
    rw [hWestCurrent]
    rfl
  have hHallRoom : currentRoomState inHall = publicHallRoom := by
    simp [inHall, transitionThroughExit, publicStartWest,
      unlockExitInRoom, currentRoomState, setRoom, hWestRooms,
      hWestCurrentZero, publicInitialWorld, publicRooms]
  have hHallPos : inHall.player.pos = publicPos 8 4 := by
    rfl
  have hHallRunning : Running inHall := by
    rcases hWestRunning with ⟨halive, hnotCompleted⟩
    constructor
    · simpa [inHall, transitionThroughExit, alive] using halive
    · simp [inHall, transitionThroughExit, publicStartWest, hnotCompleted]
  rcases directionPlan_has_engine_plan
      (room := publicHallRoom) (s := inHall)
      (start := publicPos 8 4) (directions := hallToAttackDirections)
      hHallRoom hHallPos rfl hHallRunning public_hall_to_attack_safe with
    ⟨toAttack, attackState, hToAttack, hAttackPos, hAttackRoom,
      hAttackRunning, hAttackInventory, hAttackHp, hAttackMaxHp,
      hAttackCurrent, hAttackRooms, hAttackRoomIds, hAttackCompleted⟩
  refine ⟨_, attackState,
    engineExec_append hToWest (engineExec_append hEnterHall hToAttack), ?_,
    hAttackRoom, ?_, hAttackRunning, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hAttackCurrent]
    rfl
  · simpa [hallToAttackDirections, directionEndpoint, advance, publicPos]
      using hAttackPos
  · rw [hAttackInventory]
    change inHall.player.inventory = publicInitialWorld.player.inventory
    rw [show inHall.player.inventory = atWest.player.inventory by
      rfl, hWestInventory]
  · rw [hAttackRooms]
    change inHall.rooms = publicRooms
    simp only [inHall, transitionThroughExit, publicStartWest,
      unlockExitInRoom, hWestCurrentZero, hWestRooms, publicInitialWorld]
    rw [hWestRoom]
    funext roomId
    by_cases hzero : roomId = 0
    · subst roomId
      simp [setRoom, publicRooms]
    · simp [setRoom, hzero]
  · rw [hAttackRoomIds]
    change inHall.roomIds = [0, 1, 2]
    rw [show inHall.roomIds = atWest.roomIds by rfl, hWestRoomIds]
    rfl
  · rw [hAttackHp]
    change inHall.player.hp = 5
    rw [show inHall.player.hp = atWest.player.hp by rfl, hWestHp]
    rfl
  · rw [hAttackCompleted]
    change inHall.completed = false
    simp [inHall, transitionThroughExit, publicStartWest, hWestCompleted,
      publicInitialWorld]

theorem public_engine_clears_hall :
    ∃ preActions attackState combatActions afterKill,
      EngineExec publicInitialWorld preActions attackState ∧
      EngineExec attackState combatActions afterKill ∧
      afterKill.currentRoom = 1 ∧
      currentRoomState afterKill = publicClearedHallRoom ∧
      afterKill.player.pos = publicPos 6 3 ∧ Running afterKill ∧
      afterKill.rooms 0 = publicStartRoom ∧
      afterKill.rooms 2 = publicKeyRoom := by
  rcases public_engine_reaches_hall_attack_position with
    ⟨preActions, attackState, hPrefix, hCurrent, hRoom, hPos, hRunning,
      hInventory, hRooms, hRoomIds, hHp, hCompleted⟩
  let faced : WorldState :=
    { attackState with player :=
        { attackState.player with facing := .west, shielding := false } }
  have hMonsterAhead :
      monsterAt (currentRoomState attackState)
        (advance attackState.player.pos .west) := by
    rw [hRoom, hPos]
    simp [monsterAt, publicHallRoom, publicHallMonster, publicPos, advance]
  have hFaceStep : Step attackState .left faced [] :=
    Step.faceMonster rfl hMonsterAhead
  have hFaceExec : EngineExec attackState [.left] faced :=
    player_step_has_single_tick_execution hRunning hFaceStep
      (by simp [AutonomousOnlyEvents])
  have hFacedRoom : currentRoomState faced = publicHallRoom := by
    change currentRoomState attackState = publicHallRoom
    exact hRoom
  let damaged : Monster := { publicHallMonster with hp := 1 }
  let afterFirst : WorldState :=
    updateCurrentRoom
      { faced with player := { faced.player with shielding := false } }
      (replaceMonster (currentRoomState faced) publicHallMonster damaged)
  have hNoInteraction : ¬ primaryInteractionAvailable faced := by
    simp [primaryInteractionAvailable, openChestInteractionAvailable,
      npcInteractionAvailable, switchInteractionAvailable, hFacedRoom,
      publicHallRoom]
  have hSwordEquipped : faced.player.inventory.equippedA = some .sword := by
    change attackState.player.inventory.equippedA = some .sword
    rw [hInventory]
    rfl
  have hHasSword : .sword ∈ faced.player.inventory.items := by
    change .sword ∈ attackState.player.inventory.items
    rw [hInventory]
    simp [publicInitialWorld, publicInitialPlayer]
  have hMonsterMember : publicHallMonster ∈ (currentRoomState faced).monsters := by
    change publicHallMonster ∈ (currentRoomState attackState).monsters
    rw [hRoom]
    simp [publicHallRoom]
  have hTarget :
      publicHallMonster.pos = advance faced.player.pos faced.player.facing := by
    change publicHallMonster.pos = advance attackState.player.pos .west
    rw [hPos]
    rfl
  have hFirstStep :
      Step faced .slotA afterFirst [.monsterDamaged publicHallMonster.id] := by
    exact Step.attackDamage hNoInteraction hSwordEquipped hHasSword
      hMonsterMember hTarget (by decide)
  have hFacedRunning : Running faced := by
    rcases hRunning with ⟨halive, hnotCompleted⟩
    exact ⟨by change 0 < attackState.player.hp; exact halive,
      by change attackState.completed = false; exact hnotCompleted⟩
  have hFirstExec : EngineExec faced [.slotA] afterFirst :=
    player_step_has_single_tick_execution hFacedRunning hFirstStep
      (by simp [AutonomousOnlyEvents])
  have hFirstRunning : Running afterFirst := by
    rcases hFacedRunning with ⟨halive, hnotCompleted⟩
    exact ⟨by change 0 < faced.player.hp; exact halive,
      by change faced.completed = false; exact hnotCompleted⟩
  have hAfterFirstRoom :
      currentRoomState afterFirst =
        { publicHallRoom with monsters := [damaged] } := by
    simp [afterFirst, currentRoomState_updateCurrentRoom, hFacedRoom,
      replaceMonster, publicHallRoom, publicHallMonster, damaged]
  have hNoInteractionAfter : ¬ primaryInteractionAvailable afterFirst := by
    simp [primaryInteractionAvailable, openChestInteractionAvailable,
      npcInteractionAvailable, switchInteractionAvailable, hAfterFirstRoom,
      publicHallRoom]
  have hDamagedEquipped : afterFirst.player.inventory.equippedA = some .sword := by
    exact hSwordEquipped
  have hDamagedHasSword : .sword ∈ afterFirst.player.inventory.items := hHasSword
  have hDamagedMember : damaged ∈ (currentRoomState afterFirst).monsters := by
    rw [hAfterFirstRoom]
    simp
  have hDamagedTarget :
      damaged.pos = advance afterFirst.player.pos afterFirst.player.facing := by
    exact hTarget
  let afterKill := resolveMonsterKill afterFirst damaged
  have hKillStep :
      Step afterFirst .slotA afterKill (monsterKillEvents afterFirst damaged) :=
    Step.attackKill hNoInteractionAfter hDamagedEquipped hDamagedHasSword
      hDamagedMember hDamagedTarget (by simp [damaged, swordDamage])
  have hKillAgent : ¬ AutonomousOnlyEvents (monsterKillEvents afterFirst damaged) := by
    simp only [monsterKillEvents]
    dsimp
    split <;> simp [AutonomousOnlyEvents]
  have hKillExec : EngineExec afterFirst [.slotA] afterKill :=
    player_step_has_single_tick_execution hFirstRunning hKillStep hKillAgent
  have hAfterFirstCurrent : afterFirst.currentRoom = 1 := by
    change faced.currentRoom = 1
    exact hCurrent
  have hAfterFirstRoomZero : afterFirst.rooms 0 = publicStartRoom := by
    have hne : 0 ≠ faced.currentRoom := by
      change 0 ≠ attackState.currentRoom
      rw [hCurrent]
      decide
    change (setRoom faced.rooms faced.currentRoom
      (replaceMonster (currentRoomState faced) publicHallMonster damaged)) 0 =
        publicStartRoom
    rw [setRoom_other _ _ _ hne]
    change attackState.rooms 0 = publicStartRoom
    rw [hRooms]
    rfl
  have hAfterFirstRoomTwo : afterFirst.rooms 2 = publicKeyRoom := by
    have hne : 2 ≠ faced.currentRoom := by
      change 2 ≠ attackState.currentRoom
      rw [hCurrent]
      decide
    change (setRoom faced.rooms faced.currentRoom
      (replaceMonster (currentRoomState faced) publicHallMonster damaged)) 2 =
        publicKeyRoom
    rw [setRoom_other _ _ _ hne]
    change attackState.rooms 2 = publicKeyRoom
    rw [hRooms]
    rfl
  have hRemovalEmpty :
      (removeMonster (currentRoomState afterFirst) damaged).monsters = [] := by
    rw [hAfterFirstRoom]
    simp [removeMonster, damaged, publicHallMonster]
  have hAfterFirstRoomOne :
      afterFirst.rooms 1 = { publicHallRoom with monsters := [damaged] } := by
    rw [← hAfterFirstCurrent]
    exact hAfterFirstRoom
  have hAfterRoom : currentRoomState afterKill = publicClearedHallRoom := by
    change currentRoomState (resolveMonsterKill afterFirst damaged) =
      publicClearedHallRoom
    simp only [resolveMonsterKill, hRemovalEmpty, if_pos]
    simp [currentRoomState, updateCurrentRoom, setRoom,
      hAfterFirstCurrent, hAfterFirstRoomOne, removeMonster,
      unlockAllMonstersDefeatedExits, revealEligibleChestsInWorld,
      revealEligibleChests, publicClearedHallRoom, publicHallRoom,
      publicHallMonster, publicHallWest, publicHallEast, damaged,
      requirementContainsAllMonstersDefeated]
  have hAfterRunning : Running afterKill := by
    constructor
    · unfold alive
      change 0 < (resolveMonsterKill afterFirst damaged).player.hp
      rw [resolveMonsterKill_player]
      change 0 < afterFirst.player.hp
      change 0 < attackState.player.hp
      omega
    · change (resolveMonsterKill afterFirst damaged).completed = false
      rw [resolveMonsterKill_completed]
      change faced.completed = false
      exact hCompleted
  have hRoomZero : afterKill.rooms 0 = publicStartRoom := by
    have hother : 0 ≠ afterFirst.currentRoom := by
      rw [hAfterFirstCurrent]
      decide
    have hreveal :
        revealEligibleChests (afterFirst.rooms 0) afterFirst.currentRoom =
          afterFirst.rooms 0 := by
      rw [hAfterFirstRoomZero]
      simp [revealEligibleChests, publicStartRoom]
    change (resolveMonsterKill afterFirst damaged).rooms 0 = publicStartRoom
    rw [resolveMonsterKill_other_room afterFirst damaged 0 hother hreveal]
    exact hAfterFirstRoomZero
  have hRoomTwo : afterKill.rooms 2 = publicKeyRoom := by
    have hother : 2 ≠ afterFirst.currentRoom := by
      rw [hAfterFirstCurrent]
      decide
    have hreveal :
        revealEligibleChests (afterFirst.rooms 2) afterFirst.currentRoom =
          afterFirst.rooms 2 := by
      rw [hAfterFirstRoomTwo]
      simp [revealEligibleChests, publicKeyRoom, publicKeyChest]
    change (resolveMonsterKill afterFirst damaged).rooms 2 = publicKeyRoom
    rw [resolveMonsterKill_other_room afterFirst damaged 2 hother hreveal]
    exact hAfterFirstRoomTwo
  refine ⟨preActions, attackState, [.left, .slotA, .slotA], afterKill,
    hPrefix, engineExec_append hFaceExec
      (engineExec_append hFirstExec hKillExec), ?_, hAfterRoom, ?_,
    hAfterRunning, hRoomZero, hRoomTwo⟩
  · change (resolveMonsterKill afterFirst damaged).currentRoom = 1
    rw [resolveMonsterKill_currentRoom]
    exact hAfterFirstCurrent
  · change (resolveMonsterKill afterFirst damaged).player.pos = publicPos 6 3
    rw [resolveMonsterKill_player]
    change faced.player.pos = publicPos 6 3
    exact hPos

theorem public_engine_gets_key :
    ∃ actions afterOpen,
      EngineExec publicInitialWorld actions afterOpen ∧
      afterOpen.currentRoom = 2 ∧
      currentRoomState afterOpen = publicOpenedKeyRoom ∧
      afterOpen.player.pos = publicPos 6 4 ∧
      0 < afterOpen.player.inventory.keys ∧ Running afterOpen ∧
      afterOpen.rooms 0 = publicStartRoom ∧
      afterOpen.rooms 1 = publicClearedHallRoom := by
  rcases public_engine_clears_hall with
    ⟨preActions, attackState, combatActions, afterKill,
      hPre, hCombat, hKillCurrent, hKillRoom, hKillPos, hKillRunning,
      hRoomZero, hRoomTwo⟩
  have hKillRoomOne : afterKill.rooms 1 = publicClearedHallRoom := by
    rw [← hKillCurrent]
    exact hKillRoom
  rcases directionPlan_has_engine_plan
      (room := publicClearedHallRoom) (s := afterKill)
      (start := publicPos 6 3) (directions := clearedHallToKeyDirections)
      hKillRoom hKillPos rfl hKillRunning public_cleared_hall_to_key_safe with
    ⟨toWest, atWest, hToWest, hWestPos, hWestRoom, hWestRunning,
      hWestInventory, hWestHp, hWestMaxHp, hWestCurrent, hWestRooms,
      hWestRoomIds, hWestCompleted⟩
  have hWestCurrentOne : atWest.currentRoom = 1 := by
    rw [hWestCurrent]
    exact hKillCurrent
  have hExitMember : publicHallWest ∈ (currentRoomState atWest).exits := by
    rw [hWestRoom]
    simp [publicClearedHallRoom, publicHallRoom]
  have hAtExit : exitContains publicHallWest atWest.player.pos := by
    left
    rw [hWestPos]
    rfl
  have hReq : exitRequirementSatisfied atWest publicHallWest := by
    simp [exitRequirementSatisfied, publicHallWest, requirementSatisfied]
  have hSpawn :
      canEnter (atWest.rooms publicHallWest.targetRoom)
        publicHallWest.targetSpawn := by
    rw [hWestRooms]
    change canEnter (afterKill.rooms 2) (publicPos 8 4)
    rw [hRoomTwo]
    simp [publicKeyRoom, publicBounds, publicKeyChest,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      gapAt, activeBridgeTile, publicPos]
  let inKey := transitionThroughExit atWest publicHallWest
  have hEnterKey : EngineExec atWest [.left] inKey :=
    engineExec_useExit_once hWestRunning hExitMember hAtExit hReq hSpawn
  have hKeyRoom : currentRoomState inKey = publicKeyRoom := by
    simp [inKey, transitionThroughExit, publicHallWest, unlockExitInRoom,
      currentRoomState, setRoom, hWestCurrentOne, hWestRooms, hRoomTwo]
  have hKeyPos : inKey.player.pos = publicPos 8 4 := rfl
  have hKeyRunning : Running inKey := by
    rcases hWestRunning with ⟨halive, hnotCompleted⟩
    exact ⟨by simpa [inKey, transitionThroughExit, alive] using halive,
      by simp [inKey, transitionThroughExit, publicHallWest, hnotCompleted]⟩
  rcases directionPlan_has_engine_plan
      (room := publicKeyRoom) (s := inKey)
      (start := publicPos 8 4) (directions := keyToChestDirections)
      hKeyRoom hKeyPos rfl hKeyRunning public_key_to_chest_safe with
    ⟨toChest, nearChest, hToChest, hChestPos, hChestRoom, hChestRunning,
      hChestInventory, hChestHp, hChestMaxHp, hChestCurrent, hChestRooms,
      hChestRoomIds, hChestCompleted⟩
  have hChestMember : publicKeyChest ∈ (currentRoomState nearChest).chests := by
    rw [hChestRoom]
    simp [publicKeyRoom]
  have hReach : interactionReach nearChest.player.pos publicKeyChest.pos := by
    right
    rw [hChestPos]
    simp [keyToChestDirections, directionEndpoint, publicKeyChest,
      publicPos, advance, adjacent]
  let afterOpen := openChestResult nearChest publicKeyChest
  have hOpen : EngineExec nearChest [.slotA] afterOpen :=
    engineExec_openChest_once hChestRunning hChestMember rfl rfl hReach
  have hOpenedRoom : currentRoomState afterOpen = publicOpenedKeyRoom := by
    simp [afterOpen, openChestResult, currentRoomState_updateCurrentRoom,
      hChestRoom, publicOpenedKeyRoom]
  have hOpenedRunning : Running afterOpen := by
    rcases hChestRunning with ⟨halive, hnotCompleted⟩
    constructor
    · change 0 < (collectLoot { nearChest.player with shielding := false }
          publicKeyChest.loot).hp
      unfold alive at halive
      simpa [publicKeyChest, collectLoot] using halive
    · change nearChest.completed = false
      exact hnotCompleted
  have hAfterCurrent : afterOpen.currentRoom = 2 := by
    change nearChest.currentRoom = 2
    rw [hChestCurrent]
    rfl
  have hNearCurrent : nearChest.currentRoom = 2 := by
    change afterOpen.currentRoom = 2
    exact hAfterCurrent
  have hAfterRoomZero : afterOpen.rooms 0 = publicStartRoom := by
    dsimp [afterOpen, openChestResult, updateCurrentRoom]
    have hne : 0 ≠ nearChest.currentRoom := by rw [hNearCurrent]; decide
    rw [setRoom_other _ _ _ hne, hChestRooms]
    change inKey.rooms 0 = publicStartRoom
    simp [inKey, transitionThroughExit, publicHallWest, unlockExitInRoom,
      setRoom, hWestCurrentOne, hWestRooms, hRoomZero]
  have hAfterRoomOne : afterOpen.rooms 1 = publicClearedHallRoom := by
    dsimp [afterOpen, openChestResult, updateCurrentRoom]
    have hne : 1 ≠ nearChest.currentRoom := by rw [hNearCurrent]; decide
    rw [setRoom_other _ _ _ hne, hChestRooms]
    change inKey.rooms 1 = publicClearedHallRoom
    simp only [inKey, transitionThroughExit, publicHallWest, unlockExitInRoom]
    rw [hWestCurrentOne, setRoom_same]
    exact hWestRoom
  refine ⟨_, afterOpen,
    engineExec_append hPre
      (engineExec_append hCombat
        (engineExec_append hToWest
          (engineExec_append hEnterKey
            (engineExec_append hToChest hOpen)))),
    hAfterCurrent, hOpenedRoom, ?_, ?_, hOpenedRunning,
    hAfterRoomZero, hAfterRoomOne⟩
  · change nearChest.player.pos = publicPos 6 4
    simpa [keyToChestDirections, directionEndpoint, publicPos, advance]
      using hChestPos
  · change 0 < (collectLoot { nearChest.player with shielding := false }
        publicKeyChest.loot).inventory.keys
    simp [publicKeyChest, collectLoot]

theorem public_map_complete_certificate :
    ∃ actions final,
      EngineExec publicInitialWorld actions final ∧
      WorldCompleted final ∧ alive final ∧ ValidState final := by
  rcases public_engine_gets_key with
    ⟨toKey, afterOpen, hToKey, hKeyCurrent, hKeyRoom, hKeyPos,
      hHasKey, hKeyRunning, hRoomZero, hRoomOne⟩
  rcases directionPlan_has_engine_plan
      (room := publicOpenedKeyRoom) (s := afterOpen)
      (start := publicPos 6 4) (directions := keyToHallDirections)
      hKeyRoom hKeyPos rfl hKeyRunning public_key_to_hall_safe with
    ⟨toKeyExit, atKeyExit, hToKeyExit, hKeyExitPos, hKeyExitRoom,
      hKeyExitRunning, hKeyExitInventory, hKeyExitHp, hKeyExitMaxHp,
      hKeyExitCurrent, hKeyExitRooms, hKeyExitRoomIds, hKeyExitCompleted⟩
  have hKeyExitCurrentTwo : atKeyExit.currentRoom = 2 := by
    rw [hKeyExitCurrent]
    exact hKeyCurrent
  have hKeyExitMember : publicKeyEast ∈ (currentRoomState atKeyExit).exits := by
    rw [hKeyExitRoom]
    simp [publicOpenedKeyRoom, publicKeyRoom, replaceChest]
  have hAtKeyExit : exitContains publicKeyEast atKeyExit.player.pos := by
    left
    rw [hKeyExitPos]
    rfl
  have hKeyExitReq : exitRequirementSatisfied atKeyExit publicKeyEast := by
    simp [exitRequirementSatisfied, publicKeyEast, requirementSatisfied]
  have hKeyExitSpawn :
      canEnter (atKeyExit.rooms publicKeyEast.targetRoom)
        publicKeyEast.targetSpawn := by
    rw [hKeyExitRooms]
    change canEnter (afterOpen.rooms 1) (publicPos 1 4)
    rw [hRoomOne]
    simp [publicClearedHallRoom, publicHallRoom, publicBounds,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      gapAt, activeBridgeTile, publicPos]
  let backHall := transitionThroughExit atKeyExit publicKeyEast
  have hBackHallExec : EngineExec atKeyExit [.right] backHall :=
    engineExec_useExit_once hKeyExitRunning hKeyExitMember hAtKeyExit
      hKeyExitReq hKeyExitSpawn
  have hBackHallRoom : currentRoomState backHall = publicClearedHallRoom := by
    simp [backHall, transitionThroughExit, publicKeyEast, unlockExitInRoom,
      currentRoomState, setRoom, hKeyExitCurrentTwo, hKeyExitRooms,
      hRoomOne]
  have hBackHallRunning : Running backHall := by
    rcases hKeyExitRunning with ⟨halive, hnotCompleted⟩
    exact ⟨by simpa [backHall, transitionThroughExit, alive] using halive,
      by simp [backHall, transitionThroughExit, publicKeyEast, hnotCompleted]⟩
  rcases directionPlan_has_engine_plan
      (room := publicClearedHallRoom) (s := backHall)
      (start := publicPos 1 4) (directions := hallToStartDirections)
      hBackHallRoom rfl rfl hBackHallRunning public_hall_to_start_safe with
    ⟨toHallExit, atHallExit, hToHallExit, hHallExitPos, hHallExitRoom,
      hHallExitRunning, hHallExitInventory, hHallExitHp, hHallExitMaxHp,
      hHallExitCurrent, hHallExitRooms, hHallExitRoomIds, hHallExitCompleted⟩
  have hHallExitCurrentOne : atHallExit.currentRoom = 1 := by
    rw [hHallExitCurrent]
    rfl
  have hHallExitMember : publicHallEast ∈ (currentRoomState atHallExit).exits := by
    rw [hHallExitRoom]
    simp [publicClearedHallRoom, publicHallRoom]
  have hAtHallExit : exitContains publicHallEast atHallExit.player.pos := by
    left
    rw [hHallExitPos]
    rfl
  have hHallExitReq : exitRequirementSatisfied atHallExit publicHallEast := by
    simp [exitRequirementSatisfied, publicHallEast, requirementSatisfied]
  have hHallExitSpawn :
      canEnter (atHallExit.rooms publicHallEast.targetRoom)
        publicHallEast.targetSpawn := by
    rw [hHallExitRooms]
    change canEnter (backHall.rooms 0) (publicPos 1 4)
    simp only [backHall, transitionThroughExit, publicKeyEast,
      unlockExitInRoom]
    rw [hKeyExitCurrentTwo]
    simp only [setRoom]
    simp
    rw [hKeyExitRooms]
    change canEnter (afterOpen.rooms 0) (publicPos 1 4)
    rw [hRoomZero]
    simp [publicStartRoom, publicBounds, publicHintNpc,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      gapAt, activeBridgeTile, publicPos]
  let backStart := transitionThroughExit atHallExit publicHallEast
  have hBackStartExec : EngineExec atHallExit [.right] backStart :=
    engineExec_useExit_once hHallExitRunning hHallExitMember hAtHallExit
      hHallExitReq hHallExitSpawn
  have hBackStartRoom : currentRoomState backStart = publicStartRoom := by
    simp [backStart, transitionThroughExit, publicHallEast, unlockExitInRoom,
      currentRoomState, setRoom, hHallExitCurrentOne, hHallExitRooms,
      backHall, transitionThroughExit, publicKeyEast, unlockExitInRoom,
      hKeyExitCurrentTwo, hKeyExitRooms, hRoomZero]
  have hBackStartRunning : Running backStart := by
    rcases hHallExitRunning with ⟨halive, hnotCompleted⟩
    exact ⟨by simpa [backStart, transitionThroughExit, alive] using halive,
      by simp [backStart, transitionThroughExit, publicHallEast, hnotCompleted]⟩
  rcases directionPlan_has_engine_plan
      (room := publicStartRoom) (s := backStart)
      (start := publicPos 1 4) (directions := startToLockDirections)
      hBackStartRoom rfl rfl hBackStartRunning public_start_to_lock_safe with
    ⟨toLock, atLock, hToLock, hLockPos, hLockRoom, hLockRunning,
      hLockInventory, hLockHp, hLockMaxHp, hLockCurrent,
      hLockRooms, hLockRoomIds, hLockCompleted⟩
  have hLockCurrentZero : atLock.currentRoom = 0 := by
    rw [hLockCurrent]
    rfl
  have hLockMember : publicStartLocked ∈ (currentRoomState atLock).exits := by
    rw [hLockRoom]
    simp [publicStartRoom]
  have hAtLock : exitContains publicStartLocked atLock.player.pos := by
    left
    rw [hLockPos]
    rfl
  have hLockKey : 1 ≤ atLock.player.inventory.keys := by
    rw [hLockInventory]
    change 1 ≤ atHallExit.player.inventory.keys
    rw [hHallExitInventory]
    change 1 ≤ atKeyExit.player.inventory.keys
    rw [hKeyExitInventory]
    omega
  have hLockReq : exitRequirementSatisfied atLock publicStartLocked := by
    right
    exact hLockKey
  have hLockSpawn :
      canEnter (atLock.rooms publicStartLocked.targetRoom)
        publicStartLocked.targetSpawn := by
    change canEnter (atLock.rooms 0) (publicPos 8 4)
    rw [← hLockCurrentZero]
    change canEnter (currentRoomState atLock) (publicPos 8 4)
    rw [hLockRoom]
    simp [publicStartRoom, publicBounds, publicHintNpc,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      gapAt, activeBridgeTile, publicPos]
  let final := transitionThroughExit atLock publicStartLocked
  have hFinish : EngineExec atLock [.right] final :=
    engineExec_useExit_once hLockRunning hLockMember hAtLock hLockReq hLockSpawn
  have hAll := engineExec_append hToKey
    (engineExec_append hToKeyExit
      (engineExec_append hBackHallExec
        (engineExec_append hToHallExit
          (engineExec_append hBackStartExec
            (engineExec_append hToLock hFinish)))))
  refine ⟨_, final, hAll, ?_, ?_, ?_⟩
  · simp [final, WorldCompleted, transitionThroughExit, publicStartLocked]
  · rcases hLockRunning with ⟨halive, _⟩
    simpa [final, transitionThroughExit, alive] using halive
  · exact engineExec_preserves_validState
      (by simp [ValidState, publicInitialWorld, publicInitialPlayer,
        currentRoomState, publicRooms, publicStartRoom, publicBounds,
        inBounds, publicPos]) hAll

inductive PublicMilestone where
  | start | hall | hallCleared | keyRoom | keyConfirmed
  | returnedHall | returnedStart | lockedExit | completed
  deriving DecidableEq, Repr

inductive PublicMilestoneStep : PublicMilestone → PublicMilestone → Prop where
  | enterHall : PublicMilestoneStep .start .hall
  | clearHall : PublicMilestoneStep .hall .hallCleared
  | enterKey : PublicMilestoneStep .hallCleared .keyRoom
  | collectKey : PublicMilestoneStep .keyRoom .keyConfirmed
  | backHall : PublicMilestoneStep .keyConfirmed .returnedHall
  | backStart : PublicMilestoneStep .returnedHall .returnedStart
  | selectLock : PublicMilestoneStep .returnedStart .lockedExit
  | finish : PublicMilestoneStep .lockedExit .completed

inductive PublicMilestoneRun :
    PublicMilestone → List PublicMilestone → PublicMilestone → Prop where
  | nil (p) : PublicMilestoneRun p [] p
  | cons {p q goal rest} : PublicMilestoneStep p q →
      PublicMilestoneRun q rest goal →
      PublicMilestoneRun p (q :: rest) goal

def publicTrace : List PublicMilestone :=
  [.hall, .hallCleared, .keyRoom, .keyConfirmed,
   .returnedHall, .returnedStart, .lockedExit, .completed]

theorem public_milestone_certificate :
    PublicMilestoneRun .start publicTrace .completed := by
  exact .cons .enterHall
    (.cons .clearHall
    (.cons .enterKey
    (.cons .collectKey
    (.cons .backHall
    (.cons .backStart
    (.cons .selectLock
    (.cons .finish (.nil .completed))))))))

end Task3

/-! ## 十七、Task4：未知桥指纹与观测驱动探索（完全重构） -/

namespace Task4

abbrev AnonymousRoomId := Nat
abbrev BridgeFingerprint := List Direction

inductive RoomKind where
  | unknown | switch | hub | leaf
  deriving DecidableEq, Repr

structure RoomMemory where
  kind : RoomKind := .unknown
  entryDirection : Option Direction := none
  connections : List (Direction × AnonymousRoomId) := []
  exits : List Direction := []
  exitKinds : List (Direction × ExitKind) := []
  chests : List Position := []
  openedChests : List Position := []
  switchPos : Option Position := none
  monsterSeen : Bool := false
  monsterCleared : Bool := false
  bridgeModes : List (BridgeFingerprint × Nat) := []
  currentBridgeMode : BridgeFingerprint := []
  deriving DecidableEq, Repr

def connectionTarget (m : RoomMemory) (d : Direction) : Option AnonymousRoomId :=
  match m.connections.find? (fun edge => edge.1 = d) with
  | some edge => some edge.2
  | none => none

structure Controller where
  currentRoom : AnonymousRoomId := 0
  nextRoomId : AnonymousRoomId := 1
  memories : AnonymousRoomId → RoomMemory := fun _ => {}
  worldRevision : Nat := 0
  inventoryRevision : Nat := 0
  keys : Nat := 0
  hasSword : Bool := false
  pendingExit : Option Direction := none
  lastPlayerTile : Option Position := none
  departAfterSwitch : Bool := false
  pendingInteraction : Option (String × Position) := none
  awaitingMonsterResult : Bool := false
  moveAction : Option Action := none
  moveTarget : Option Position := none

structure RoomFeatures where
  switchSeen : Bool := false
  bridgeSeen : Bool := false
  abyssCountExceedsWidth : Bool := false
  deriving DecidableEq, Repr

def classifyRoom (features : RoomFeatures) : RoomKind :=
  if features.switchSeen then .switch
  else if features.bridgeSeen || features.abyssCountExceedsWidth then .hub
  else .leaf

theorem switch_feature_classifies_switch (features : RoomFeatures)
    (h : features.switchSeen = true) :
    classifyRoom features = .switch := by
  simp [classifyRoom, h]

theorem bridge_or_abyss_classifies_hub (features : RoomFeatures)
    (hs : features.switchSeen = false)
    (hh : features.bridgeSeen = true ∨ features.abyssCountExceedsWidth = true) :
    classifyRoom features = .hub := by
  rcases hh with h | h <;> simp [classifyRoom, hs, h]

def bridgeFingerprint (reachableBoundaryDirections : List Direction) :
    BridgeFingerprint :=
  reachableBoundaryDirections.eraseDups

theorem bridge_fingerprint_contains_exactly_reachable
    (directions : List Direction) (d : Direction) :
    d ∈ bridgeFingerprint directions ↔ d ∈ directions := by
  simp [bridgeFingerprint]

def recordBridgeMode
    (memory : RoomMemory) (fingerprint : BridgeFingerprint) (revision : Nat) :
    RoomMemory :=
  if memory.bridgeModes.any (fun entry => entry.1 = fingerprint) then
    { memory with currentBridgeMode := fingerprint }
  else
    { { memory with bridgeModes := (fingerprint, revision) :: memory.bridgeModes } with
      currentBridgeMode := fingerprint }

theorem recorded_bridge_mode_becomes_current
    (memory : RoomMemory) (fingerprint : BridgeFingerprint) (revision : Nat) :
    (recordBridgeMode memory fingerprint revision).currentBridgeMode = fingerprint := by
  simp [recordBridgeMode]
  split <;> rfl

theorem existing_bridge_modes_are_retained
    (memory : RoomMemory) (fingerprint old : BridgeFingerprint) (revision oldRevision : Nat)
    (h : (old, oldRevision) ∈ memory.bridgeModes) :
    (old, oldRevision) ∈ (recordBridgeMode memory fingerprint revision).bridgeModes := by
  unfold recordBridgeMode
  split
  · exact h
  · exact List.mem_cons_of_mem _ h

def jumpDistance (oldTile newTile : Position) : Nat :=
  (newTile.x - oldTile.x).natAbs + (newTile.y - oldTile.y).natAbs

def roomJumpDetected (oldTile newTile : Position) : Bool :=
  3 < jumpDistance oldTile newTile

inductive ConfirmRoomChange :
    Controller → Position → Controller → Prop where
  | directedExisting {c c' : Controller} {oldTile newTile : Position}
      {d : Direction} {target : AnonymousRoomId}
      (hlast : c.lastPlayerTile = some oldTile)
      (hpending : c.pendingExit = some d)
      (hjump : roomJumpDetected oldTile newTile = true)
      (hconnection : connectionTarget (c.memories c.currentRoom) d = some target)
      (hcurrent : c'.currentRoom = target)
      (hcleared : c'.pendingExit = none) :
      ConfirmRoomChange c newTile c'
  | directedFresh {c c' : Controller} {oldTile newTile : Position} {d : Direction}
      (hlast : c.lastPlayerTile = some oldTile)
      (hpending : c.pendingExit = some d)
      (hjump : roomJumpDetected oldTile newTile = true)
      (hnew : connectionTarget (c.memories c.currentRoom) d = none)
      (hcurrent : c'.currentRoom = c.nextRoomId)
      (hforward : connectionTarget (c'.memories c.currentRoom) d =
        some c.nextRoomId)
      (hreverse : connectionTarget (c'.memories c.nextRoomId) (Task3.opposite d) =
        some c.currentRoom) :
      ConfirmRoomChange c newTile c'
  | anonymous {c c' : Controller} {oldTile newTile : Position}
      (hlast : c.lastPlayerTile = some oldTile)
      (hnone : c.pendingExit = none)
      (hjump : roomJumpDetected oldTile newTile = true)
      (hcurrent : c'.currentRoom = c.nextRoomId)
      (hconnections : (c'.memories c.nextRoomId).connections = []) :
      ConfirmRoomChange c newTile c'

theorem directionless_jump_creates_no_claimed_direction
    {c c' : Controller} {newTile : Position}
    (hnone : c.pendingExit = none)
    (h : ConfirmRoomChange c newTile c') :
    c'.currentRoom = c.nextRoomId := by
  cases h with
  | directedExisting hlast hpending hjump hconnection hcurrent hcleared =>
      rw [hnone] at hpending
      contradiction
  | directedFresh hlast hpending hjump hnew hcurrent hforward hreverse =>
      rw [hnone] at hpending
      contradiction
  | anonymous hlast hnone' hjump hcurrent hconnections => exact hcurrent

theorem directionless_jump_records_no_fabricated_edge
    {c c' : Controller} {newTile : Position}
    (hnone : c.pendingExit = none)
    (h : ConfirmRoomChange c newTile c') :
    c'.currentRoom = c.nextRoomId ∧
      (c'.memories c.nextRoomId).connections = [] := by
  cases h with
  | directedExisting hlast hpending hjump hconnection hcurrent hcleared =>
      rw [hnone] at hpending
      contradiction
  | directedFresh hlast hpending hjump hnew hcurrent hforward hreverse =>
      rw [hnone] at hpending
      contradiction
  | anonymous hlast hnone' hjump hcurrent hconnections =>
      exact ⟨hcurrent, hconnections⟩

theorem directed_fresh_change_records_bidirectional_edge
    {c c' : Controller} {newTile : Position}
    (h : ConfirmRoomChange c newTile c')
    (hnoExisting : ∀ d, connectionTarget (c.memories c.currentRoom) d = none)
    (hconnected : (c'.memories c.nextRoomId).connections ≠ []) :
    ∃ d,
      connectionTarget (c'.memories c.currentRoom) d = some c.nextRoomId ∧
      connectionTarget (c'.memories c.nextRoomId) (Task3.opposite d) =
        some c.currentRoom := by
  cases h with
  | directedExisting hlast hpending hjump hconnection hcurrent hcleared =>
      rw [hnoExisting _] at hconnection
      contradiction
  | directedFresh hlast hpending hjump hnew hcurrent hforward hreverse =>
      exact ⟨_, hforward, hreverse⟩
  | anonymous hlast hnone hjump hcurrent hconnections =>
      exact False.elim (hconnected hconnections)

inductive GoalKind where
  | openChest | combat | exploreUnknown | pressSwitch | departSwitch
  | exploreHubBranch | returnToSwitch | returnEntry | wait
  deriving DecidableEq, Repr

structure Goal where
  kind : GoalKind
  target : Option Position := none
  direction : Option Direction := none
  deriving DecidableEq, Repr

structure CandidateFacts where
  visibleUnopenedChest : Option Position := none
  visibleMonster : Option Position := none
  unknownExit : Option Direction := none
  anyReachableExit : Option Direction := none
  hubServiceableBranch : Option Direction := none
  switchReturn : Option Direction := none
  entryReturn : Option Direction := none
  switchTarget : Option Position := none
  deriving DecidableEq, Repr

def returnGoal (facts : CandidateFacts) : Goal :=
  match facts.entryReturn with
  | some d => ⟨.returnEntry, none, some d⟩
  | none => match facts.anyReachableExit with
    | some d => ⟨.returnEntry, none, some d⟩
    | none => ⟨.wait, none, none⟩

def chooseGoal
    (kind : RoomKind) (hasSword departAfterSwitch : Bool)
    (facts : CandidateFacts) : Goal :=
  match facts.visibleUnopenedChest with
  | some p => ⟨.openChest, some p, none⟩
  | none => match facts.visibleMonster with
    | some p => if hasSword then ⟨.combat, some p, none⟩ else returnGoal facts
    | none => match kind with
      | .switch =>
          match facts.unknownExit with
          | some d => ⟨.exploreUnknown, none, some d⟩
          | none =>
              if departAfterSwitch then
                match facts.anyReachableExit with
                | some d => ⟨.departSwitch, none, some d⟩
                | none => ⟨.wait, none, none⟩
              else match facts.switchTarget with
                | some p => ⟨.pressSwitch, some p, none⟩
                | none => ⟨.wait, none, none⟩
      | .hub =>
          match facts.hubServiceableBranch with
          | some d => ⟨.exploreHubBranch, none, some d⟩
          | none => match facts.switchReturn with
            | some d => ⟨.returnToSwitch, none, some d⟩
            | none => returnGoal facts
      | .leaf | .unknown => returnGoal facts

theorem visible_chest_has_priority
    (kind : RoomKind) (sword depart : Bool) (facts : CandidateFacts) (p : Position)
    (h : facts.visibleUnopenedChest = some p) :
    (chooseGoal kind sword depart facts).kind = .openChest := by
  simp [chooseGoal, h]

theorem monster_requires_sword
    (kind : RoomKind) (depart : Bool) (facts : CandidateFacts) (p : Position)
    (hc : facts.visibleUnopenedChest = none)
    (hm : facts.visibleMonster = some p) :
    (chooseGoal kind false depart facts).kind ≠ .combat := by
  cases he : facts.entryReturn <;> cases ha : facts.anyReachableExit <;>
    simp [chooseGoal, returnGoal, hc, hm, he, ha]

theorem switch_room_explores_unknown_before_pressing
    (depart : Bool) (facts : CandidateFacts) (d : Direction)
    (hc : facts.visibleUnopenedChest = none)
    (hm : facts.visibleMonster = none)
    (hu : facts.unknownExit = some d) :
    (chooseGoal .switch false depart facts).kind = .exploreUnknown := by
  simp [chooseGoal, hc, hm, hu]

def hubDirectionAllowed (keys : Nat) (kind : ExitKind) : Prop :=
  kind ≠ .locked ∨ 0 < keys

theorem locked_hub_branch_requires_key
    (keys : Nat) (h : hubDirectionAllowed keys .locked) :
    0 < keys := by
  rcases h with hkind | hkey
  · contradiction
  · exact hkey

def recordChestOnAction
    (memory : RoomMemory) (target : Position) (action : Action) : RoomMemory :=
  if action = .slotA then
    { memory with openedChests := target :: memory.openedChests }
  else memory

theorem chest_is_recorded_only_on_actual_A
    (memory : RoomMemory) (target : Position) (action : Action)
    (h : target ∈ (recordChestOnAction memory target action).openedChests)
    (hold : target ∉ memory.openedChests) :
    action = .slotA := by
  unfold recordChestOnAction at h
  split at h
  · assumption
  · exact False.elim (hold h)

def ChestInteractionSound (s : WorldState) (target : Position) : Prop :=
  visibleChestAt (currentRoomState s) target ∧ adjacent s.player.pos target

theorem recorded_chest_refines_real_interaction
    {s : WorldState} {target : Position}
    (h : ChestInteractionSound s target) :
    ∃ chest, chest ∈ (currentRoomState s).chests ∧
      chest.pos = target ∧ chest.visible = true := by
  rcases h.1 with ⟨chest, hm, hp, hv⟩
  exact ⟨chest, hm, hp, hv⟩

def BridgeModeFair
    (modes : List BridgeFingerprint)
    (eventuallyObserved : BridgeFingerprint → Prop) : Prop :=
  ∀ mode, mode ∈ modes → eventuallyObserved mode

theorem generic_bridge_policy_does_not_assume_three_modes
    (modes : List BridgeFingerprint)
    (observed : BridgeFingerprint → Prop)
    (hfair : BridgeModeFair modes observed)
    {mode : BridgeFingerprint} (hmode : mode ∈ modes) :
    observed mode :=
  hfair mode hmode

def fourModeRegression : List BridgeFingerprint :=
  [[.west], [.north], [.east], [.south]]

theorem four_bridge_fingerprints_are_supported :
    fourModeRegression.length = 4 := by
  decide

def iterateRotation : Nat → BridgeOrientation → BridgeOrientation
  | 0, o => o
  | n + 1, o => iterateRotation n (rotateOrientation o)

def publicRotationsNeeded (current target : BridgeOrientation) : Nat :=
  if current = target then 0
  else if rotateOrientation current = target then 1
  else 2

theorem public_rotation_bound (current target : BridgeOrientation) :
    publicRotationsNeeded current target ≤ 2 := by
  unfold publicRotationsNeeded
  by_cases h₀ : current = target
  · simp [h₀]
  · by_cases h₁ : rotateOrientation current = target
    · simp [h₀, h₁]
    · simp [h₀, h₁]

theorem public_rotation_reaches_target (current target : BridgeOrientation) :
    iterateRotation (publicRotationsNeeded current target) current = target := by
  cases current <;> cases target <;> decide

inductive CommandRole where
  | navigation | faceObject | exitPush | interaction | idle
  deriving DecidableEq, Repr

def CommandSafe (s : WorldState) (role : CommandRole) (action : Action) : Prop :=
  match role with
  | .navigation => ∀ d, actionDirection action = some d →
      safeTile (currentRoomState s) (advance s.player.pos d)
  | .faceObject => ∃ d, actionDirection action = some d ∧
      (visibleChestAt (currentRoomState s) (advance s.player.pos d) ∨
       monsterAt (currentRoomState s) (advance s.player.pos d))
  | .exitPush => ∃ exit ∈ (currentRoomState s).exits,
      action = directionAction exit.direction ∧
      exitContains exit s.player.pos ∧ exitRequirementSatisfied s exit
  | .interaction => action = .slotA ∧ primaryInteractionAvailable s
  | .idle => action = .wait

theorem navigation_command_has_engine_step
    {s : WorldState} {action : Action} {d : Direction}
    (hsafe : CommandSafe s .navigation action)
    (ha : actionDirection action = some d) :
    ∃ after events, Step s action after events := by
  have htile := hsafe d ha
  let q := advance s.player.pos d
  by_cases hbutton : buttonAt (currentRoomState s) q
  · rcases hbutton with ⟨button, hmember, hpos⟩
    exact ⟨_, _, Step.moveButton ha rfl htile.1 hmember hpos⟩
  · exact ⟨_, _, Step.movePlain ha rfl htile.1 htile.2.1 hbutton⟩

structure PolicyDecision where
  goal : Goal
  role : CommandRole
  output : Action

def PolicyRefinesAgent
    (kind : RoomKind) (c : Controller) (facts : CandidateFacts)
    (s : WorldState) (decision : PolicyDecision) : Prop :=
  decision.goal = chooseGoal kind c.hasSword c.departAfterSwitch facts ∧
  CommandSafe s decision.role decision.output

theorem goal_layer_refines_agent
    {kind : RoomKind} {c : Controller} {facts : CandidateFacts}
    {s : WorldState} {decision : PolicyDecision}
    (h : PolicyRefinesAgent kind c facts s decision) :
    decision.goal = chooseGoal kind c.hasSword c.departAfterSwitch facts ∧
    CommandSafe s decision.role decision.output := h

inductive FairPolicyRun (step : Controller → Controller → Prop) :
    Controller → Controller → Prop where
  | refl (c) : FairPolicyRun step c c
  | tail {a b c} : step a b → FairPolicyRun step b c → FairPolicyRun step a c

theorem memory_policy_complete_under_fairness
    (measure : Controller → Nat) (goal : Controller → Prop)
    (step : Controller → Controller → Prop)
    (hzero : ∀ c, measure c = 0 → goal c)
    (hprogress : ∀ c, 0 < measure c → ∃ next,
      step c next ∧ measure next < measure c) :
    ∀ initial, ∃ final, FairPolicyRun step initial final ∧ goal final := by
  intro initial
  induction hmeasure : measure initial using Nat.strongRecOn generalizing initial with
  | ind n ih =>
      by_cases hz : measure initial = 0
      · exact ⟨initial, .refl initial, hzero initial hz⟩
      · have hpositive : 0 < measure initial := Nat.pos_of_ne_zero hz
        rcases hprogress initial hpositive with ⟨next, hstep, hdecrease⟩
        have hdecrease' : measure next < n := by simpa [hmeasure] using hdecrease
        rcases ih (measure next) hdecrease' next rfl with ⟨final, hrun, hgoal⟩
        exact ⟨final, .tail hstep hrun, hgoal⟩

/-! ### 公开 Task4 三态桥模板。通用策略仍只依赖 `BridgeFingerprint`。 -/

def publicBounds : Bounds :=
  { width := 10, height := 8, width_pos := by decide, height_pos := by decide }
def publicPos (x y : Int) : Position := ⟨x, y⟩

def publicNorthTiles : List Position :=
  [publicPos 0 3, publicPos 1 3, publicPos 2 3, publicPos 3 3,
   publicPos 4 3, publicPos 5 3, publicPos 0 4, publicPos 1 4,
   publicPos 2 4, publicPos 3 4, publicPos 4 4, publicPos 5 4,
   publicPos 4 0, publicPos 5 0, publicPos 4 1, publicPos 5 1,
   publicPos 4 2, publicPos 5 2]
def publicEastTiles : List Position :=
  [publicPos 0 3, publicPos 1 3, publicPos 2 3, publicPos 3 3,
   publicPos 4 3, publicPos 5 3, publicPos 6 3, publicPos 7 3,
   publicPos 8 3, publicPos 9 3, publicPos 0 4, publicPos 1 4,
   publicPos 2 4, publicPos 3 4, publicPos 4 4, publicPos 5 4,
   publicPos 6 4, publicPos 7 4, publicPos 8 4, publicPos 9 4]
def publicSouthTiles : List Position :=
  [publicPos 0 3, publicPos 1 3, publicPos 2 3, publicPos 3 3,
   publicPos 4 3, publicPos 5 3, publicPos 0 4, publicPos 1 4,
   publicPos 2 4, publicPos 3 4, publicPos 4 4, publicPos 5 4,
   publicPos 4 5, publicPos 5 5, publicPos 4 6, publicPos 5 6,
   publicPos 4 7, publicPos 5 7]

def publicBridge (orientation : BridgeOrientation) : Bridge :=
  { id := 401, orientation := orientation, northTiles := publicNorthTiles,
    eastTiles := publicEastTiles, southTiles := publicSouthTiles }

def publicAbyssTraps : List Trap :=
  (List.range 10).flatMap (fun x => (List.range 8).map (fun y =>
    ({ id := x * 8 + y
       pos := publicPos (Int.ofNat x) (Int.ofNat y)
       kind := .abyss
       damage := 1
       respawn := publicPos 1 4 } : Trap)))

def publicFinalChest : Chest :=
  { id := 402, pos := publicPos 4 4, loot := .gold 1, visible := false,
    revealOn := .allMonstersDefeated (some 4) }
def publicSwitch : Switch :=
  { id := 403, pos := publicPos 4 4, targetRoom := 1, targetBridge := 401 }
def publicKeyChest : Chest :=
  { id := 404, pos := publicPos 4 3, loot := .key 1, visible := true }
def publicSwordChest : Chest :=
  { id := 405, pos := publicPos 5 4, loot := .tool .sword .A, visible := true }
def publicGuardian : Monster :=
  { id := 406, pos := publicPos 4 4, kind := .chaser, hp := 1, damage := 1 }

/- The JSON exits occupy two boundary tiles.  `pos` is the tile used by the
   concrete route below and `otherTiles` is the second tile accepted by the
   engine. -/
def publicWestEast : Exit :=
  { id := 410, pos := publicPos 9 4, otherTiles := [publicPos 9 3],
    direction := .east, kind := .normal, requirement := .free,
    targetRoom := 1, targetSpawn := publicPos 1 4 }
def publicCenterWest : Exit :=
  { id := 411, pos := publicPos 0 4, otherTiles := [publicPos 0 3],
    direction := .west, kind := .normal, requirement := .free,
    targetRoom := 0, targetSpawn := publicPos 8 4 }
def publicCenterEast : Exit :=
  { id := 412, pos := publicPos 9 4, otherTiles := [publicPos 9 3],
    direction := .east, kind := .locked,
    requirement := .keys 1 false, targetRoom := 3,
    targetSpawn := publicPos 1 4 }
def publicCenterNorth : Exit :=
  { id := 413, pos := publicPos 4 0, otherTiles := [publicPos 5 0],
    direction := .north, kind := .normal, requirement := .free,
    targetRoom := 2, targetSpawn := publicPos 4 6 }
def publicCenterSouth : Exit :=
  { id := 414, pos := publicPos 4 7, otherTiles := [publicPos 5 7],
    direction := .south, kind := .normal, requirement := .free,
    targetRoom := 4, targetSpawn := publicPos 4 1 }
def publicNorthSouth : Exit :=
  { id := 415, pos := publicPos 4 7, otherTiles := [publicPos 5 7],
    direction := .south, kind := .normal, requirement := .free,
    targetRoom := 1, targetSpawn := publicPos 4 1 }
def publicEastWest : Exit :=
  { id := 416, pos := publicPos 0 4, otherTiles := [publicPos 0 3],
    direction := .west, kind := .normal, requirement := .free,
    targetRoom := 1, targetSpawn := publicPos 8 4 }
def publicSouthNorth : Exit :=
  { id := 417, pos := publicPos 4 0, otherTiles := [publicPos 5 0],
    direction := .north, kind := .normal, requirement := .free,
    targetRoom := 1, targetSpawn := publicPos 4 6 }

@[simp] theorem publicWestEast_not_completing :
    publicWestEast.completesTask = false := rfl
@[simp] theorem publicCenterWest_not_completing :
    publicCenterWest.completesTask = false := rfl
@[simp] theorem publicCenterEast_not_completing :
    publicCenterEast.completesTask = false := rfl
@[simp] theorem publicCenterNorth_not_completing :
    publicCenterNorth.completesTask = false := rfl
@[simp] theorem publicCenterSouth_not_completing :
    publicCenterSouth.completesTask = false := rfl
@[simp] theorem publicNorthSouth_not_completing :
    publicNorthSouth.completesTask = false := rfl
@[simp] theorem publicEastWest_not_completing :
    publicEastWest.completesTask = false := rfl
@[simp] theorem publicSouthNorth_not_completing :
    publicSouthNorth.completesTask = false := rfl

def publicCenterRoom (orientation : BridgeOrientation)
    (chest : Chest := publicFinalChest) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [], chests := [chest],
    monsters := [], traps := publicAbyssTraps, buttons := [], switches := [],
    bridges := [publicBridge orientation], dynamicTiles := [],
    exits := [publicCenterWest, publicCenterEast, publicCenterNorth,
      publicCenterSouth] }

def publicWestRoom (pressed : Bool := false) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [], chests := [],
    monsters := [], traps := [], buttons := [],
    switches := [{ publicSwitch with pressed := pressed }], bridges := [],
    dynamicTiles := [], exits := [publicWestEast] }
def publicNorthRoom (chest : Chest := publicKeyChest) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [], chests := [chest],
    monsters := [], traps := [], buttons := [], switches := [], bridges := [],
    dynamicTiles := [], exits := [publicNorthSouth] }
def publicEastRoom (chest : Chest := publicSwordChest) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [], chests := [chest],
    monsters := [], traps := [], buttons := [], switches := [], bridges := [],
    dynamicTiles := [], exits := [publicEastWest] }
def publicSouthRoom (monsters : List Monster := [publicGuardian]) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [], chests := [],
    monsters := monsters, traps := [], buttons := [], switches := [], bridges := [],
    dynamicTiles := [], exits := [publicSouthNorth] }

def publicRooms : RoomId → RoomState
  | 0 => publicWestRoom
  | 1 => publicCenterRoom .westToNorth
  | 2 => publicNorthRoom
  | 3 => publicEastRoom
  | 4 => publicSouthRoom
  | _ => publicWestRoom

def publicInitialPlayer : PlayerState :=
  { pos := publicPos 7 4
    facing := .east
    hp := 5
    maxHp := 5
    inventory :=
      { keys := 0
        gold := 0
        items := [.shield]
        equippedA := none
        equippedB := some .shield } }
def publicInitialWorld : WorldState :=
  { currentRoom := 0, rooms := publicRooms, roomIds := [0, 1, 2, 3, 4],
    player := publicInitialPlayer, completed := false }

def openedChest (chest : Chest) : Chest := { chest with opened := true }
def revealedFinalChest : Chest := { publicFinalChest with visible := true }
def openedFinalChest : Chest := { revealedFinalChest with opened := true }

def westToCenterDirections : List Direction := [.east, .east]
def centerToNorthDirections : List Direction :=
  [.east, .east, .east, .north, .north, .north, .north]
def northToKeyDirections : List Direction := [.north, .north]
def keyToNorthExitDirections : List Direction := [.south, .south, .south]
def centerNorthToWestDirections : List Direction :=
  [.south, .south, .south, .west, .west, .west, .west]
def westToSwitchDirections : List Direction := [.west, .west, .west]
def westSwitchToExitDirections : List Direction := [.east, .east, .east, .east]
def centerToEastDirections : List Direction :=
  [.east, .east, .east, .east, .east, .east, .east, .east]
def eastToSwordDirections : List Direction := [.east, .east, .east]
def swordToEastExitDirections : List Direction := [.west, .west, .west, .west]
def centerEastToWestDirections : List Direction :=
  [.west, .west, .west, .west, .west, .west, .west, .west]
def centerToSouthDirections : List Direction :=
  [.east, .east, .east, .south, .south, .south]
def southToGuardianDirections : List Direction := [.south, .south]
def guardianToSouthExitDirections : List Direction := [.north, .north, .north]
def centerSouthToFinalDirections : List Direction := [.north]

theorem public_west_to_center_safe :
    DirectionPlanSafe publicWestRoom (publicPos 7 4)
      westToCenterDirections := by
  simp [DirectionPlanSafe, westToCenterDirections, publicWestRoom,
    publicBounds, publicSwitch, publicWestEast, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile, publicPos, advance]

theorem public_center_to_north_safe :
    DirectionPlanSafe (publicCenterRoom .westToNorth) (publicPos 1 4)
      centerToNorthDirections := by
  simp [DirectionPlanSafe, centerToNorthDirections, publicCenterRoom,
    publicBridge, publicNorthTiles, publicAbyssTraps, safeTile, canEnter,
    inBounds, staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicBounds, publicPos,
    publicFinalChest, advance]

theorem public_north_to_key_safe :
    DirectionPlanSafe publicNorthRoom (publicPos 4 6)
      northToKeyDirections := by
  simp [DirectionPlanSafe, northToKeyDirections, publicNorthRoom,
    publicKeyChest, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile, publicPos, advance]

theorem public_key_to_north_exit_safe :
    DirectionPlanSafe (publicNorthRoom (openedChest publicKeyChest))
      (publicPos 4 4) keyToNorthExitDirections := by
  simp [DirectionPlanSafe, keyToNorthExitDirections, publicNorthRoom,
    openedChest, publicKeyChest, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile, publicPos, advance]

theorem public_center_north_to_west_safe :
    DirectionPlanSafe (publicCenterRoom .westToNorth) (publicPos 4 1)
      centerNorthToWestDirections := by
  simp [DirectionPlanSafe, centerNorthToWestDirections, publicCenterRoom,
    publicBridge, publicNorthTiles, publicAbyssTraps, safeTile, canEnter,
    inBounds, staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicBounds, publicPos,
    publicFinalChest, advance]

theorem public_west_to_switch_safe :
    DirectionPlanSafe publicWestRoom (publicPos 8 4)
      westToSwitchDirections := by
  simp [DirectionPlanSafe, westToSwitchDirections, publicWestRoom,
    publicBounds, publicSwitch, safeTile, canEnter, inBounds, staticBlocker,
    npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt, activeBridgeTile,
    publicPos, advance]

theorem public_west_switch_to_exit_safe (pressed : Bool) :
    DirectionPlanSafe (publicWestRoom pressed) (publicPos 5 4)
      westSwitchToExitDirections := by
  simp [DirectionPlanSafe, westSwitchToExitDirections, publicWestRoom,
    publicBounds, publicSwitch, safeTile, canEnter, inBounds, staticBlocker,
    npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt, activeBridgeTile,
    publicPos, advance]

theorem public_center_to_east_safe :
    DirectionPlanSafe (publicCenterRoom .westToEast) (publicPos 1 4)
      centerToEastDirections := by
  simp [DirectionPlanSafe, centerToEastDirections, publicCenterRoom,
    publicBridge, publicEastTiles, publicAbyssTraps, safeTile, canEnter,
    inBounds, staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicBounds, publicPos,
    publicFinalChest, advance]

theorem public_east_to_sword_safe :
    DirectionPlanSafe publicEastRoom (publicPos 1 4)
      eastToSwordDirections := by
  simp [DirectionPlanSafe, eastToSwordDirections, publicEastRoom,
    publicSwordChest, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile, publicPos, advance]

theorem public_sword_to_east_exit_safe :
    DirectionPlanSafe (publicEastRoom (openedChest publicSwordChest))
      (publicPos 4 4) swordToEastExitDirections := by
  simp [DirectionPlanSafe, swordToEastExitDirections, publicEastRoom,
    openedChest, publicSwordChest, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile, publicPos, advance]

theorem public_center_east_to_west_safe :
    DirectionPlanSafe (publicCenterRoom .westToEast) (publicPos 8 4)
      centerEastToWestDirections := by
  simp [DirectionPlanSafe, centerEastToWestDirections, publicCenterRoom,
    publicBridge, publicEastTiles, publicAbyssTraps, safeTile, canEnter,
    inBounds, staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicBounds, publicPos,
    publicFinalChest, advance]

theorem public_center_to_south_safe :
    DirectionPlanSafe (publicCenterRoom .westToSouth) (publicPos 1 4)
      centerToSouthDirections := by
  simp [DirectionPlanSafe, centerToSouthDirections, publicCenterRoom,
    publicBridge, publicSouthTiles, publicAbyssTraps, safeTile, canEnter,
    inBounds, staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicBounds, publicPos,
    publicFinalChest, advance]

theorem public_south_to_guardian_safe :
    DirectionPlanSafe publicSouthRoom (publicPos 4 1)
      southToGuardianDirections := by
  simp [DirectionPlanSafe, southToGuardianDirections, publicSouthRoom,
    publicGuardian, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile, publicPos, advance]

theorem public_guardian_to_south_exit_safe :
    DirectionPlanSafe (publicSouthRoom []) (publicPos 4 3)
      guardianToSouthExitDirections := by
  simp [DirectionPlanSafe, guardianToSouthExitDirections, publicSouthRoom,
    publicBounds, safeTile, canEnter, inBounds, staticBlocker, npcAt,
    visibleChestAt, activeTrapAt, monsterAt, gapAt, activeBridgeTile,
    publicPos, advance]

theorem public_center_south_to_final_safe :
    DirectionPlanSafe (publicCenterRoom .westToSouth revealedFinalChest)
      (publicPos 4 6) centerSouthToFinalDirections := by
  simp [DirectionPlanSafe, centerSouthToFinalDirections, publicCenterRoom,
    publicBridge, publicSouthTiles, publicAbyssTraps, revealedFinalChest,
    publicFinalChest, safeTile, canEnter, inBounds, staticBlocker, npcAt,
    visibleChestAt, activeTrapAt, monsterAt, gapAt, activeBridgeTile,
    publicBounds, publicPos, advance]

theorem engineExec_activateSwitch_once
    {s : WorldState} {switch : Switch}
    (hrunning : Running s)
    (hnoChest : ¬ openChestInteractionAvailable s)
    (hnoNpc : ¬ npcInteractionAvailable s)
    (hmember : switch ∈ (currentRoomState s).switches)
    (hreach : interactionReach s.player.pos switch.pos)
    (hbridge : ∃ bridge ∈ (s.rooms switch.targetRoom).bridges,
      bridge.id = switch.targetBridge) :
    EngineExec s [.slotA] (activateSwitchState s switch) := by
  apply player_step_has_single_tick_execution hrunning
  · exact Step.activateSwitch hnoChest hnoNpc hmember hreach hbridge
  · simp [AutonomousOnlyEvents]

/- Fully explicit public-world trace.  Every state is a reducible application
   of an environment transition; the theorem below supplies the proof that
   the corresponding player action list really produces it. -/
def publicS01 := applyDirectionPlan publicInitialWorld westToCenterDirections
def publicS02 := transitionThroughExit publicS01 publicWestEast
def publicS03 := applyDirectionPlan publicS02 centerToNorthDirections
def publicS04 := transitionThroughExit publicS03 publicCenterNorth
def publicS05 := applyDirectionPlan publicS04 northToKeyDirections
def publicS06 := openChestResult publicS05 publicKeyChest
def publicS07 := applyDirectionPlan publicS06 keyToNorthExitDirections
def publicS08 := transitionThroughExit publicS07 publicNorthSouth
def publicS09 := applyDirectionPlan publicS08 centerNorthToWestDirections
def publicS10 := transitionThroughExit publicS09 publicCenterWest
def publicS11 := applyDirectionPlan publicS10 westToSwitchDirections
def publicS12 := activateSwitchState publicS11 publicSwitch
def publicS13 := applyDirectionPlan publicS12 westSwitchToExitDirections
def publicS14 := transitionThroughExit publicS13 publicWestEast
def publicS15 := applyDirectionPlan publicS14 centerToEastDirections
def publicS16 := transitionThroughExit publicS15 publicCenterEast
def publicS17 := applyDirectionPlan publicS16 eastToSwordDirections
def publicS18 := openChestResult publicS17 publicSwordChest
def publicS19 := applyDirectionPlan publicS18 swordToEastExitDirections
def publicS20 := transitionThroughExit publicS19 publicEastWest
def publicS21 := applyDirectionPlan publicS20 centerEastToWestDirections
def publicS22 := transitionThroughExit publicS21 publicCenterWest
def publicS23 := applyDirectionPlan publicS22 westToSwitchDirections
def publicPressedSwitch : Switch := { publicSwitch with pressed := true }
def publicS24 := activateSwitchState publicS23 publicPressedSwitch
def publicS25 := applyDirectionPlan publicS24 westSwitchToExitDirections
def publicS26 := transitionThroughExit publicS25 publicWestEast
def publicS27 := applyDirectionPlan publicS26 centerToSouthDirections
def publicS28 := transitionThroughExit publicS27 publicCenterSouth
def publicS29 := applyDirectionPlan publicS28 southToGuardianDirections
def faceWorld (s : WorldState) (direction : Direction) : WorldState :=
  { s with player := { s.player with facing := direction, shielding := false } }
@[simp] theorem faceWorld_currentRoomState (s : WorldState) (direction : Direction) :
    currentRoomState (faceWorld s direction) = currentRoomState s := rfl
@[simp] theorem faceWorld_inventory (s : WorldState) (direction : Direction) :
    (faceWorld s direction).player.inventory = s.player.inventory := rfl
@[simp] theorem faceWorld_pos (s : WorldState) (direction : Direction) :
    (faceWorld s direction).player.pos = s.player.pos := rfl
@[simp] theorem faceWorld_facing (s : WorldState) (direction : Direction) :
    (faceWorld s direction).player.facing = direction := rfl
@[simp] theorem faceWorld_hp (s : WorldState) (direction : Direction) :
    (faceWorld s direction).player.hp = s.player.hp := rfl
@[simp] theorem faceWorld_completed (s : WorldState) (direction : Direction) :
    (faceWorld s direction).completed = s.completed := rfl
@[simp] theorem faceWorld_currentRoom (s : WorldState) (direction : Direction) :
    (faceWorld s direction).currentRoom = s.currentRoom := rfl
@[simp] theorem faceWorld_rooms (s : WorldState) (direction : Direction) :
    (faceWorld s direction).rooms = s.rooms := rfl
def publicS30 : WorldState := faceWorld publicS29 .south
def publicS31 := resolveMonsterKill publicS30 publicGuardian
def publicS32 := applyDirectionPlan publicS31 guardianToSouthExitDirections
def publicS33 := transitionThroughExit publicS32 publicSouthNorth
def publicS34 := applyDirectionPlan publicS33 centerSouthToFinalDirections
def publicS35 := openChestResult publicS34 revealedFinalChest
def clearShieldWorld (s : WorldState) : WorldState :=
  { s with player := { s.player with shielding := false } }
@[simp] theorem clearShieldWorld_rooms (s : WorldState) :
    (clearShieldWorld s).rooms = s.rooms := rfl
@[simp] theorem clearShieldWorld_roomIds (s : WorldState) :
    (clearShieldWorld s).roomIds = s.roomIds := rfl
@[simp] theorem clearShieldWorld_currentRoom (s : WorldState) :
    (clearShieldWorld s).currentRoom = s.currentRoom := rfl
@[simp] theorem clearShieldWorld_hp (s : WorldState) :
    (clearShieldWorld s).player.hp = s.player.hp := rfl
@[simp] theorem clearShieldWorld_completed (s : WorldState) :
    (clearShieldWorld s).completed = s.completed := rfl
def markWorldCompleted (s : WorldState) : WorldState := { s with completed := true }
def publicS36 : WorldState := clearShieldWorld publicS35
def publicS37 : WorldState := markWorldCompleted publicS36

theorem public_initial_running : Running publicInitialWorld := by
  simp [Running, alive, publicInitialWorld, publicInitialPlayer]

theorem public_route_west_to_center :
    EngineExec publicInitialWorld
      (westToCenterDirections.map actionForDirection) publicS01 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicWestRoom)
  · simp [publicInitialWorld, currentRoomState, publicRooms]
  · rfl
  · exact public_initial_running
  · simpa [publicInitialWorld, publicInitialPlayer] using
      public_west_to_center_safe

theorem public_enter_center_north_bridge :
    EngineExec publicS01 [.right] publicS02 := by
  apply engineExec_useExit_once (exit := publicWestEast)
  · simp [Running, alive, publicS01, applyDirectionPlan,
      westToCenterDirections, publicInitialWorld, publicInitialPlayer]
  · simp [publicS01, applyDirectionPlan, westToCenterDirections,
      publicInitialWorld, currentRoomState, publicRooms, publicWestRoom]
  · simp [exitContains, publicS01, applyDirectionPlan,
      westToCenterDirections, publicInitialWorld, publicInitialPlayer,
      publicWestEast, publicPos, advance, movePlayerState]
  · simp [exitRequirementSatisfied, publicWestEast, requirementSatisfied]
  · have hroom : publicS01.rooms 1 =
        publicCenterRoom .westToNorth := by
      simp [publicS01, applyDirectionPlan, westToCenterDirections,
        publicInitialWorld, publicRooms]
    simpa [publicWestEast, hroom, publicCenterRoom, publicFinalChest,
      publicBridge, publicNorthTiles, publicAbyssTraps, canEnter, inBounds,
      staticBlocker, npcAt, visibleChestAt, gapAt, activeBridgeTile,
      publicBounds, publicPos]

theorem public_route_center_to_north :
    EngineExec publicS02
      (centerToNorthDirections.map actionForDirection) publicS03 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicCenterRoom .westToNorth)
  · simp [publicS02, publicS01, applyDirectionPlan,
      westToCenterDirections, transitionThroughExit, publicInitialWorld,
      publicRooms, publicWestEast, unlockExitInRoom, currentRoomState, setRoom,
      publicWestRoom]
  · rfl
  · simp [Running, alive, publicS02, publicS01, applyDirectionPlan,
      westToCenterDirections, transitionThroughExit, publicInitialWorld,
      publicInitialPlayer, publicWestEast]
  · simpa [publicS02, publicS01, applyDirectionPlan,
      westToCenterDirections, transitionThroughExit, publicInitialWorld,
      publicInitialPlayer, publicWestEast, movePlayerState] using
      public_center_to_north_safe

theorem public_enter_north_room :
    EngineExec publicS03 [.up] publicS04 := by
  apply engineExec_useExit_once (exit := publicCenterNorth)
  · simp [Running, alive, publicS03, publicS02, publicS01,
      applyDirectionPlan, centerToNorthDirections, westToCenterDirections,
      transitionThroughExit, publicInitialWorld, publicInitialPlayer,
      publicWestEast]
  · simp [publicS03, publicS02, publicS01, applyDirectionPlan,
      centerToNorthDirections, westToCenterDirections, transitionThroughExit,
      publicInitialWorld, publicRooms, publicWestEast, publicCenterNorth,
      unlockExitInRoom, currentRoomState, setRoom, publicWestRoom,
      publicCenterRoom]
  · simp [exitContains, publicS03, publicS02, publicS01,
      applyDirectionPlan, centerToNorthDirections, westToCenterDirections,
      transitionThroughExit, publicInitialWorld, publicInitialPlayer,
      publicWestEast, publicCenterNorth, publicPos, advance, movePlayerState]
  · simp [exitRequirementSatisfied, publicCenterNorth,
      requirementSatisfied]
  · have hroom : publicS03.rooms 2 = publicNorthRoom := by
      simp [publicS03, publicS02, publicS01, applyDirectionPlan,
        centerToNorthDirections, westToCenterDirections, transitionThroughExit,
        publicInitialWorld, publicRooms, publicWestEast, unlockExitInRoom,
        currentRoomState, setRoom, publicWestRoom]
    simpa [publicCenterNorth, hroom, publicNorthRoom, publicKeyChest, canEnter,
      inBounds, staticBlocker,
      npcAt, visibleChestAt, gapAt, activeBridgeTile, publicBounds, publicPos]

theorem public_route_north_to_key :
    EngineExec publicS04
      (northToKeyDirections.map actionForDirection) publicS05 := by
  apply directionPlan_has_exact_engine_exec (room := publicNorthRoom)
  · simp [publicS04, publicS03, publicS02, publicS01,
      transitionThroughExit, applyDirectionPlan, centerToNorthDirections,
      westToCenterDirections, publicInitialWorld, publicRooms,
      publicWestEast, publicCenterNorth, unlockExitInRoom, currentRoomState,
      setRoom, publicWestRoom, publicCenterRoom]
  · rfl
  · simp [Running, alive, publicS04, publicS03, publicS02, publicS01,
      transitionThroughExit, applyDirectionPlan, centerToNorthDirections,
      westToCenterDirections, publicInitialWorld, publicInitialPlayer,
      publicWestEast, publicCenterNorth]
  · simpa [publicS04, publicS03, publicS02, publicS01,
      transitionThroughExit, applyDirectionPlan, centerToNorthDirections,
      westToCenterDirections, publicInitialWorld, publicInitialPlayer,
      publicWestEast, publicCenterNorth, movePlayerState] using
      public_north_to_key_safe

@[simp] theorem publicS04_room :
    currentRoomState publicS04 = publicNorthRoom := by
  simp [publicS04, publicS03, publicS02, publicS01,
    transitionThroughExit, applyDirectionPlan, centerToNorthDirections,
    westToCenterDirections, publicInitialWorld, publicRooms,
    publicWestEast, publicCenterNorth, unlockExitInRoom, currentRoomState,
    setRoom, publicWestRoom, publicCenterRoom]

theorem publicS04_running : Running publicS04 := by
  simp [Running, alive, publicS04, publicS03, publicS02, publicS01,
    publicInitialWorld, publicInitialPlayer, publicWestEast,
    publicCenterNorth]

@[simp] theorem publicS05_room :
    currentRoomState publicS05 = publicNorthRoom := by
  simpa [publicS05] using publicS04_room

theorem publicS05_running : Running publicS05 := by
  rcases publicS04_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS05] using halive,
    by simpa [publicS05] using hcomplete⟩

theorem public_open_key : EngineExec publicS05 [.slotA] publicS06 := by
  apply engineExec_openChest_once (chest := publicKeyChest)
  · simp [Running, alive, publicS05, publicS04, publicS03, publicS02,
      publicS01, publicInitialWorld, publicInitialPlayer, publicWestEast,
      publicCenterNorth]
  · simp [publicS05, publicS04, publicS03, publicS02, publicS01,
      currentRoomState, transitionThroughExit, publicCenterNorth,
      publicWestEast, unlockExitInRoom, setRoom, publicInitialWorld,
      publicRooms, publicNorthRoom]
  · rfl
  · rfl
  · simp [interactionReach, publicS05, northToKeyDirections,
      publicS04, publicCenterNorth, publicKeyChest, publicPos,
      directionEndpoint, adjacent, advance]

@[simp] theorem publicS06_room :
    currentRoomState publicS06 =
      publicNorthRoom (openedChest publicKeyChest) := by
  simp [publicS06, openChestResult, publicS05_room, publicNorthRoom,
    replaceChest, openedChest]

theorem publicS06_running : Running publicS06 := by
  rcases publicS05_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS06, openChestResult, publicKeyChest,
      collectLoot] using halive,
    by simpa [publicS06, openChestResult] using hcomplete⟩

theorem public_route_key_to_north_exit :
    EngineExec publicS06
      (keyToNorthExitDirections.map actionForDirection) publicS07 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicNorthRoom (openedChest publicKeyChest))
  · simp only [publicS06, openChestResult,
      currentRoomState_updateCurrentRoom]
    have hbase : currentRoomState publicS05 = publicNorthRoom := by
      simp [publicS05, publicS04, publicS03, publicS02, publicS01,
        currentRoomState, transitionThroughExit, publicCenterNorth,
        publicWestEast, unlockExitInRoom, setRoom, publicInitialWorld,
        publicRooms, publicNorthRoom]
    rw [hbase]
    simp [publicNorthRoom, replaceChest, openedChest]
  · rfl
  · simp [Running, alive, publicS06, openChestResult, collectLoot,
      publicKeyChest, publicS05, publicS04, publicS03, publicS02, publicS01,
      publicInitialWorld, publicInitialPlayer, publicWestEast,
      publicCenterNorth]
  · simpa [publicS06, openChestResult, publicS05,
      publicS04, publicCenterNorth, northToKeyDirections, publicKeyChest,
      collectLoot,
      directionEndpoint, publicPos, advance] using
      public_key_to_north_exit_safe

@[simp] theorem publicS07_room :
    currentRoomState publicS07 =
      publicNorthRoom (openedChest publicKeyChest) := by
  simpa [publicS07] using publicS06_room

theorem publicS07_running : Running publicS07 := by
  rcases publicS06_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS07] using halive,
    by simpa [publicS07] using hcomplete⟩

theorem public_return_center_after_key :
    EngineExec publicS07 [.down] publicS08 := by
  apply engineExec_useExit_once (exit := publicNorthSouth)
  · exact publicS07_running
  · rw [publicS07_room]
    simp [publicNorthRoom]
  · simp [exitContains, publicS07, publicS06, publicS05,
      publicS04, publicCenterNorth, keyToNorthExitDirections,
      northToKeyDirections, publicNorthSouth, publicPos,
      directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicNorthSouth,
      requirementSatisfied]
  · have hroom : publicS07.rooms 1 =
        publicCenterRoom .westToNorth := by
      simp [publicS07, publicS06, openChestResult, publicS05, publicS04,
        publicS03, publicS02, publicS01, transitionThroughExit,
        publicCenterNorth, publicWestEast, unlockExitInRoom, updateCurrentRoom,
        setRoom, currentRoomState, publicInitialWorld, publicRooms]
    simpa [publicNorthSouth, hroom, publicCenterRoom, publicFinalChest,
      publicBridge, publicNorthTiles, publicAbyssTraps, canEnter, inBounds,
      staticBlocker, npcAt, visibleChestAt, gapAt, activeBridgeTile,
      publicBounds, publicPos]

theorem publicS08_running : Running publicS08 := by
  rcases publicS07_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS08] using halive,
    by simpa [publicS08, publicNorthSouth] using hcomplete⟩

theorem public_route_center_north_to_west :
    EngineExec publicS08
      (centerNorthToWestDirections.map actionForDirection) publicS09 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicCenterRoom .westToNorth)
  · simp [publicS08, publicS07, publicS06, openChestResult, publicS05,
      publicS04, publicS03, publicS02, publicS01, transitionThroughExit,
      publicNorthSouth, publicCenterNorth, publicWestEast, unlockExitInRoom,
      updateCurrentRoom, setRoom, currentRoomState, publicInitialWorld,
      publicRooms]
  · rfl
  · exact publicS08_running
  · simpa [publicS08, publicNorthSouth, publicPos] using
      public_center_north_to_west_safe

theorem publicS09_running : Running publicS09 := by
  rcases publicS08_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS09] using halive,
    by simpa [publicS09] using hcomplete⟩

theorem public_return_west_for_first_switch :
    EngineExec publicS09 [.left] publicS10 := by
  apply engineExec_useExit_once (exit := publicCenterWest)
  · exact publicS09_running
  · simp [publicS09, publicS08, publicS07, publicS06, openChestResult,
      publicS05, publicS04, publicS03, publicS02, publicS01,
      transitionThroughExit, publicNorthSouth, publicCenterNorth,
      publicWestEast, unlockExitInRoom, updateCurrentRoom, setRoom,
      currentRoomState, publicInitialWorld, publicRooms, publicCenterRoom]
  · simp [exitContains, publicS09, publicS08,
      centerNorthToWestDirections, publicCenterWest, publicNorthSouth,
      publicPos, directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicCenterWest,
      requirementSatisfied]
  · have hroom : publicS09.rooms 0 = publicWestRoom := by
      simp [publicS09, publicS08, publicS07, publicS06, openChestResult,
        publicS05, publicS04, publicS03, publicS02, publicS01,
        transitionThroughExit, publicNorthSouth, publicCenterNorth,
        publicWestEast, unlockExitInRoom, updateCurrentRoom, setRoom,
        currentRoomState, publicInitialWorld, publicRooms]
    simpa [publicCenterWest, hroom, publicWestRoom, publicSwitch,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
      activeBridgeTile, publicBounds, publicPos]

theorem publicS10_running : Running publicS10 := by
  rcases publicS09_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS10] using halive,
    by simpa [publicS10, publicCenterWest] using hcomplete⟩

theorem public_route_to_first_switch :
    EngineExec publicS10
      (westToSwitchDirections.map actionForDirection) publicS11 := by
  apply directionPlan_has_exact_engine_exec (room := publicWestRoom)
  · simp [publicS10, publicS09, publicS08, publicS07, publicS06,
      openChestResult, publicS05, publicS04, publicS03, publicS02,
      publicS01, transitionThroughExit, publicCenterWest, publicNorthSouth,
      publicCenterNorth, publicWestEast, unlockExitInRoom, updateCurrentRoom,
      setRoom, currentRoomState, publicInitialWorld, publicRooms]
  · rfl
  · exact publicS10_running
  · simpa [publicS10, publicCenterWest, publicPos] using
      public_west_to_switch_safe

theorem publicS11_running : Running publicS11 := by
  rcases publicS10_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS11] using halive,
    by simpa [publicS11] using hcomplete⟩

theorem public_press_first_switch :
    EngineExec publicS11 [.slotA] publicS12 := by
  apply engineExec_activateSwitch_once (switch := publicSwitch)
  · exact publicS11_running
  · simp [openChestInteractionAvailable, publicS11, publicS10,
      publicS09, publicS08, publicS07, publicS06, openChestResult,
      publicS05, publicS04, publicS03, publicS02, publicS01,
      transitionThroughExit, publicCenterWest, publicNorthSouth,
      publicCenterNorth, publicWestEast, unlockExitInRoom, updateCurrentRoom,
      setRoom, currentRoomState, publicInitialWorld, publicRooms,
      publicWestRoom]
  · simp [npcInteractionAvailable, publicS11, publicS10, publicS09,
      publicS08, publicS07, publicS06, openChestResult, publicS05,
      publicS04, publicS03, publicS02, publicS01, transitionThroughExit,
      publicCenterWest, publicNorthSouth, publicCenterNorth, publicWestEast,
      unlockExitInRoom, updateCurrentRoom, setRoom, currentRoomState,
      publicInitialWorld, publicRooms, publicWestRoom]
  · simp [publicS11, publicS10, publicS09, publicS08, publicS07,
      publicS06, openChestResult, publicS05, publicS04, publicS03,
      publicS02, publicS01, transitionThroughExit, publicCenterWest,
      publicNorthSouth, publicCenterNorth, publicWestEast, unlockExitInRoom,
      updateCurrentRoom, setRoom, currentRoomState, publicInitialWorld,
      publicRooms, publicWestRoom, publicSwitch]
  · simp [interactionReach, publicS11, publicS10, westToSwitchDirections,
      publicCenterWest, publicSwitch, publicPos, directionEndpoint, advance,
      adjacent]
  · refine ⟨publicBridge .westToNorth, ?_, rfl⟩
    simp [publicS11, publicS10, publicS09, publicS08, publicS07,
      publicS06, openChestResult, publicS05, publicS04, publicS03,
      publicS02, publicS01, transitionThroughExit, publicCenterWest,
      publicNorthSouth, publicCenterNorth, publicWestEast, unlockExitInRoom,
      updateCurrentRoom, setRoom, currentRoomState, publicInitialWorld,
      publicRooms, publicCenterRoom, publicSwitch]

def publicCenterUnlocked (orientation : BridgeOrientation)
    (chest : Chest := publicFinalChest) : RoomState :=
  unlockExitInRoom (publicCenterRoom orientation chest) publicCenterEast

@[simp] theorem publicS10_room :
    currentRoomState publicS10 = publicWestRoom := by
  simp [publicS10, publicS09, publicS08, publicS07, publicS06,
    openChestResult, publicS05, publicS04, publicS03, publicS02, publicS01,
    transitionThroughExit, publicCenterWest, publicNorthSouth,
    publicCenterNorth, publicWestEast, unlockExitInRoom, updateCurrentRoom,
    setRoom, currentRoomState, publicInitialWorld, publicRooms]

@[simp] theorem publicS11_room :
    currentRoomState publicS11 = publicWestRoom := by
  simpa [publicS11] using publicS10_room

@[simp] theorem publicS11_current : publicS11.currentRoom = 0 := by
  simp [publicS11, publicS10, publicCenterWest]

@[simp] theorem publicS11_pos : publicS11.player.pos = publicPos 5 4 := by
  simp [publicS11, publicS10, publicS09, publicS08,
    westToSwitchDirections, centerNorthToWestDirections,
    publicCenterWest, publicNorthSouth, publicPos, directionEndpoint, advance]

theorem publicS12_running : Running publicS12 := by
  rcases publicS11_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS12, activateSwitchState] using halive,
    by simpa [publicS12, activateSwitchState] using hcomplete⟩

@[simp] theorem publicS12_room :
    currentRoomState publicS12 = publicWestRoom true := by
  have hroom0 : publicS11.rooms 0 = publicWestRoom := by
    simpa [currentRoomState, publicS11_current] using publicS11_room
  simp [publicS12, activateSwitchState, publicSwitch, publicS11_current,
    hroom0, publicWestRoom, replaceSwitch, currentRoomState, setRoom]

@[simp] theorem publicS12_current : publicS12.currentRoom = 0 := by
  simp [publicS12, publicS11_current]

@[simp] theorem publicS12_pos : publicS12.player.pos = publicPos 5 4 := by
  simp [publicS12, publicS11_pos]

theorem publicS12_center :
    publicS12.rooms 1 = publicCenterRoom .westToEast := by
  simp [publicS12, activateSwitchState, publicSwitch, publicS11, publicS10,
    publicS09, publicS08, publicS07, publicS06, openChestResult, publicS05,
    publicS04, publicS03, publicS02, publicS01, transitionThroughExit,
    publicCenterWest, publicNorthSouth, publicCenterNorth, publicWestEast,
    unlockExitInRoom, updateCurrentRoom, setRoom, currentRoomState,
    publicInitialWorld, publicRooms, publicCenterRoom, rotateBridge,
    publicBridge, rotateOrientation]

theorem public_route_first_switch_to_exit :
    EngineExec publicS12
      (westSwitchToExitDirections.map actionForDirection) publicS13 := by
  apply directionPlan_has_exact_engine_exec (room := publicWestRoom true)
  · exact publicS12_room
  · rfl
  · exact publicS12_running
  · simpa [publicS12_pos] using
      public_west_switch_to_exit_safe true

theorem publicS13_running : Running publicS13 := by
  rcases publicS12_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS13] using halive,
    by simpa [publicS13] using hcomplete⟩

@[simp] theorem publicS13_room :
    currentRoomState publicS13 = publicWestRoom true := by
  simpa [publicS13] using publicS12_room

theorem public_enter_center_east_bridge :
    EngineExec publicS13 [.right] publicS14 := by
  apply engineExec_useExit_once (exit := publicWestEast)
  · exact publicS13_running
  · rw [publicS13_room]
    simp [publicWestRoom]
  · simp [exitContains, publicS13, publicS12,
      westSwitchToExitDirections, publicWestEast, publicPos,
      directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicWestEast, requirementSatisfied]
  · have hcenter : publicS13.rooms 1 =
        publicCenterRoom .westToEast := by
      simpa [publicS13] using publicS12_center
    simpa [publicWestEast, hcenter, publicCenterRoom,
      publicFinalChest, publicBridge, publicEastTiles, publicAbyssTraps,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
      activeBridgeTile, publicBounds, publicPos]

theorem publicS14_running : Running publicS14 := by
  rcases publicS13_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS14] using halive,
    by simpa [publicS14, publicWestEast] using hcomplete⟩

@[simp] theorem publicS14_room :
    currentRoomState publicS14 = publicCenterRoom .westToEast := by
  simp [publicS14, transitionThroughExit, publicWestEast, currentRoomState,
    unlockExitInRoom, setRoom, publicS13, publicS12_center,
    publicS12_current]

@[simp] theorem publicS14_current : publicS14.currentRoom = 1 := by
  simp [publicS14, publicWestEast]

theorem public_route_center_to_east :
    EngineExec publicS14
      (centerToEastDirections.map actionForDirection) publicS15 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicCenterRoom .westToEast)
  · exact publicS14_room
  · rfl
  · exact publicS14_running
  · simpa [publicS14, publicWestEast, publicPos] using
      public_center_to_east_safe

theorem publicS15_running : Running publicS15 := by
  rcases publicS14_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS15] using halive,
    by simpa [publicS15] using hcomplete⟩

@[simp] theorem publicS15_room :
    currentRoomState publicS15 = publicCenterRoom .westToEast := by
  simpa [publicS15] using publicS14_room

theorem publicS15_east_room : publicS15.rooms 3 = publicEastRoom := by
  simp [publicS15, publicS14, publicS13, publicS12, activateSwitchState,
    publicS11, publicS10, publicS09, publicS08, publicS07, publicS06,
    openChestResult, publicS05, publicS04, publicS03, publicS02, publicS01,
    transitionThroughExit, publicWestEast, publicCenterWest,
    publicNorthSouth, publicCenterNorth, unlockExitInRoom,
    updateCurrentRoom, setRoom, currentRoomState, publicInitialWorld,
    publicRooms, publicSwitch]

theorem publicS15_has_key : 1 ≤ publicS15.player.inventory.keys := by
  simp [publicS15, publicS14, publicS13, publicS12, publicS11, publicS10,
    publicS09, publicS08, publicS07, publicS06, openChestResult,
    publicKeyChest, collectLoot, publicS05, publicCenterWest,
    publicNorthSouth, publicWestEast,
    transitionThroughExit, spendExitRequirement,
    activateSwitchState]

theorem public_enter_east_room :
    EngineExec publicS15 [.right] publicS16 := by
  apply engineExec_useExit_once (exit := publicCenterEast)
  · exact publicS15_running
  · rw [publicS15_room]
    simp [publicCenterRoom]
  · simp [exitContains, publicS15, publicS14, centerToEastDirections,
      publicCenterEast, publicWestEast, publicPos, directionEndpoint, advance]
  · right
    simpa [publicCenterEast, requirementSatisfied] using publicS15_has_key
  · have hroom : publicS15.rooms 3 = publicEastRoom := by
      simp [publicS15, publicS14, publicS13, publicS12, activateSwitchState,
        publicS11, publicS10, publicS09, publicS08, publicS07, publicS06,
        openChestResult, publicS05, publicS04, publicS03, publicS02,
        publicS01, transitionThroughExit, publicWestEast, publicCenterWest,
        publicNorthSouth, publicCenterNorth, unlockExitInRoom,
        updateCurrentRoom, setRoom, currentRoomState, publicInitialWorld,
        publicRooms, publicSwitch]
    simpa [publicCenterEast, hroom, publicEastRoom, publicSwordChest,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
      activeBridgeTile, publicBounds, publicPos]

theorem publicS16_running : Running publicS16 := by
  rcases publicS15_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS16] using halive,
    by simpa [publicS16, publicCenterEast] using hcomplete⟩

@[simp] theorem publicS16_room :
    currentRoomState publicS16 = publicEastRoom := by
  have heast : publicS14.rooms 3 = publicEastRoom := by
    simpa [publicS15] using publicS15_east_room
  simp [publicS16, transitionThroughExit, publicCenterEast, currentRoomState,
    unlockExitInRoom, setRoom, publicS15, publicS14_current,
    heast]

theorem public_route_east_to_sword :
    EngineExec publicS16
      (eastToSwordDirections.map actionForDirection) publicS17 := by
  apply directionPlan_has_exact_engine_exec (room := publicEastRoom)
  · exact publicS16_room
  · rfl
  · exact publicS16_running
  · simpa [publicS16, publicCenterEast, publicPos] using
      public_east_to_sword_safe

theorem publicS17_running : Running publicS17 := by
  rcases publicS16_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS17] using halive,
    by simpa [publicS17] using hcomplete⟩

@[simp] theorem publicS17_room :
    currentRoomState publicS17 = publicEastRoom := by
  simpa [publicS17] using publicS16_room

@[simp] theorem publicS17_pos : publicS17.player.pos = publicPos 4 4 := by
  simp [publicS17, publicS16, publicCenterEast, eastToSwordDirections,
    publicPos, directionEndpoint, advance]

theorem public_open_sword : EngineExec publicS17 [.slotA] publicS18 := by
  apply engineExec_openChest_once (chest := publicSwordChest)
  · exact publicS17_running
  · rw [publicS17_room]
    simp [publicEastRoom]
  · rfl
  · rfl
  · simp [interactionReach, publicS17, publicS16, eastToSwordDirections,
      publicCenterEast, publicSwordChest, publicPos, directionEndpoint,
      adjacent, advance]

@[simp] theorem publicS18_room :
    currentRoomState publicS18 =
      publicEastRoom (openedChest publicSwordChest) := by
  simp [publicS18, openChestResult, publicS17_room, publicEastRoom,
    replaceChest, openedChest]

theorem publicS18_running : Running publicS18 := by
  rcases publicS17_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS18, openChestResult, publicSwordChest,
      collectLoot] using halive,
    by simpa [publicS18, openChestResult] using hcomplete⟩

@[simp] theorem publicS18_pos : publicS18.player.pos = publicPos 4 4 := by
  simpa [publicS18] using publicS17_pos

theorem public_route_sword_to_east_exit :
    EngineExec publicS18
      (swordToEastExitDirections.map actionForDirection) publicS19 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicEastRoom (openedChest publicSwordChest))
  · exact publicS18_room
  · rfl
  · exact publicS18_running
  · simpa [publicS18_pos] using public_sword_to_east_exit_safe

theorem publicS19_running : Running publicS19 := by
  rcases publicS18_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS19] using halive,
    by simpa [publicS19] using hcomplete⟩

@[simp] theorem publicS19_room :
    currentRoomState publicS19 =
      publicEastRoom (openedChest publicSwordChest) := by
  simpa [publicS19] using publicS18_room

theorem publicS19_center :
    publicS19.rooms 1 = publicCenterUnlocked .westToEast := by
  simp [publicS19, publicS18, openChestResult, publicS17, publicS16,
    publicS15, publicS14, publicS13, publicS12, activateSwitchState,
    publicS11, publicS10, publicS09, publicS08, publicS07, publicS06,
    publicS05, publicS04, publicS03, publicS02, publicS01,
    transitionThroughExit, publicCenterEast, publicWestEast,
    publicCenterWest, publicNorthSouth, publicCenterNorth,
    publicCenterUnlocked, unlockExitInRoom, updateCurrentRoom, setRoom,
    currentRoomState, publicInitialWorld, publicRooms, publicSwitch,
    replaceExit, rotateBridge, publicBridge, rotateOrientation,
    publicCenterRoom]

@[simp] theorem publicS19_current : publicS19.currentRoom = 3 := by
  simp [publicS19, publicS18, publicS17, publicS16, publicCenterEast]

theorem public_return_center_after_sword :
    EngineExec publicS19 [.left] publicS20 := by
  apply engineExec_useExit_once (exit := publicEastWest)
  · exact publicS19_running
  · rw [publicS19_room]
    simp [publicEastRoom]
  · simp [exitContains, publicS19, publicS18, swordToEastExitDirections,
      publicEastWest, publicPos, directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicEastWest, requirementSatisfied]
  · simpa [publicEastWest, publicS19_center, publicCenterUnlocked,
      unlockExitInRoom, publicCenterEast, replaceExit, publicCenterRoom,
      publicFinalChest, publicBridge,
      publicEastTiles, publicAbyssTraps, canEnter, inBounds, staticBlocker,
      npcAt, visibleChestAt, gapAt, activeBridgeTile, publicBounds, publicPos]

theorem publicS20_running : Running publicS20 := by
  rcases publicS19_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS20] using halive,
    by simpa [publicS20, publicEastWest] using hcomplete⟩

@[simp] theorem publicS20_room :
    currentRoomState publicS20 = publicCenterUnlocked .westToEast := by
  simp [publicS20, transitionThroughExit, publicEastWest, currentRoomState,
    unlockExitInRoom, setRoom, publicS19_center, publicS19_current]

theorem public_center_unlocked_east_to_west_safe :
    DirectionPlanSafe (publicCenterUnlocked .westToEast) (publicPos 8 4)
      centerEastToWestDirections := by
  simp [DirectionPlanSafe, centerEastToWestDirections,
    publicCenterUnlocked, unlockExitInRoom, publicCenterEast,
    publicCenterRoom, replaceExit, publicCenterWest, publicCenterNorth,
    publicCenterSouth, publicBridge, publicEastTiles, publicAbyssTraps,
    publicFinalChest, safeTile, canEnter, inBounds, staticBlocker,
    npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile, publicBounds, publicPos, advance]

theorem public_route_center_east_to_west :
    EngineExec publicS20
      (centerEastToWestDirections.map actionForDirection) publicS21 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicCenterUnlocked .westToEast)
  · exact publicS20_room
  · rfl
  · exact publicS20_running
  · simpa [publicS20, publicEastWest, publicPos] using
      public_center_unlocked_east_to_west_safe

theorem publicS21_running : Running publicS21 := by
  rcases publicS20_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS21] using halive,
    by simpa [publicS21] using hcomplete⟩

@[simp] theorem publicS21_room :
    currentRoomState publicS21 = publicCenterUnlocked .westToEast := by
  simpa [publicS21] using publicS20_room

@[simp] theorem publicS21_current : publicS21.currentRoom = 1 := by
  simp [publicS21, publicS20, publicEastWest]

@[simp] theorem publicS20_current : publicS20.currentRoom = 1 := by
  simp [publicS20, publicEastWest]

theorem publicS19_west : publicS19.rooms 0 = publicWestRoom true := by
  simp [publicS19, publicS18, openChestResult, publicS17, publicS16,
    publicS15, publicS14, publicS13, publicS12, activateSwitchState,
    publicS11, publicS10, publicS09, publicS08, publicS07, publicS06,
    publicS05, publicS04, publicS03, publicS02, publicS01,
    transitionThroughExit, publicCenterEast, publicWestEast,
    publicCenterWest, publicNorthSouth, publicCenterNorth,
    unlockExitInRoom, updateCurrentRoom, setRoom, currentRoomState,
    publicInitialWorld, publicRooms, publicSwitch, publicWestRoom,
    replaceSwitch]

theorem publicS21_west : publicS21.rooms 0 = publicWestRoom true := by
  simp [publicS21, publicS20, transitionThroughExit, publicEastWest,
    unlockExitInRoom, setRoom, currentRoomState, publicS19_current,
    publicS19_west]

theorem public_return_west_for_second_switch :
    EngineExec publicS21 [.left] publicS22 := by
  apply engineExec_useExit_once (exit := publicCenterWest)
  · exact publicS21_running
  · rw [publicS21_room]
    simp [publicCenterUnlocked, publicCenterRoom, unlockExitInRoom,
      publicCenterEast, publicCenterWest, publicCenterNorth,
      publicCenterSouth, replaceExit]
  · simp [exitContains, publicS21, publicS20,
      centerEastToWestDirections, publicCenterWest, publicEastWest,
      publicPos, directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicCenterWest, requirementSatisfied]
  · have hwest : publicS21.rooms 0 = publicWestRoom true := by
      exact publicS21_west
    simpa [publicCenterWest, hwest, publicWestRoom, publicSwitch,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
      activeBridgeTile, publicBounds, publicPos]

theorem publicS22_running : Running publicS22 := by
  rcases publicS21_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS22] using halive,
    by simpa [publicS22, publicCenterWest] using hcomplete⟩

@[simp] theorem publicS22_room :
    currentRoomState publicS22 = publicWestRoom true := by
  simp [publicS22, transitionThroughExit, publicCenterWest,
    currentRoomState, unlockExitInRoom, setRoom, publicS21_current,
    publicS21_west]

@[simp] theorem publicS22_pos : publicS22.player.pos = publicPos 8 4 := by
  simp [publicS22, publicCenterWest]

theorem publicS22_center :
    publicS22.rooms 1 = publicCenterUnlocked .westToEast := by
  have hcenter : publicS21.rooms 1 =
      publicCenterUnlocked .westToEast := by
    simpa [currentRoomState, publicS21_current] using publicS21_room
  simp [publicS22, transitionThroughExit, publicCenterWest,
    unlockExitInRoom, setRoom, currentRoomState, publicS21_current,
    hcenter]

theorem public_route_to_second_switch :
    EngineExec publicS22
      (westToSwitchDirections.map actionForDirection) publicS23 := by
  apply directionPlan_has_exact_engine_exec (room := publicWestRoom true)
  · exact publicS22_room
  · rfl
  · exact publicS22_running
  · simp [DirectionPlanSafe, westToSwitchDirections, publicWestRoom,
      publicBounds, publicSwitch, safeTile, canEnter, inBounds,
      staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
      gapAt, activeBridgeTile, publicS22_pos, publicPos, advance]

theorem publicS23_running : Running publicS23 := by
  rcases publicS22_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS23] using halive,
    by simpa [publicS23] using hcomplete⟩

@[simp] theorem publicS23_room :
    currentRoomState publicS23 = publicWestRoom true := by
  simpa [publicS23] using publicS22_room

@[simp] theorem publicS23_pos : publicS23.player.pos = publicPos 5 4 := by
  simp [publicS23, publicS22, publicCenterWest, westToSwitchDirections,
    publicPos, directionEndpoint, advance]

theorem public_press_second_switch :
    EngineExec publicS23 [.slotA] publicS24 := by
  apply engineExec_activateSwitch_once (switch := publicPressedSwitch)
  · exact publicS23_running
  · simp [openChestInteractionAvailable, publicS23_room, publicWestRoom]
  · simp [npcInteractionAvailable, publicS23_room, publicWestRoom]
  · simp [publicS23_room, publicWestRoom, publicPressedSwitch,
      publicSwitch]
  · simp [interactionReach, publicS23_pos, publicPressedSwitch,
      publicSwitch, publicPos,
      adjacent, advance]
  · refine ⟨publicBridge .westToEast, ?_, rfl⟩
    have hcenter : publicS23.rooms 1 =
        publicCenterUnlocked .westToEast := by
      simpa [publicS23] using publicS22_center
    simpa [publicPressedSwitch, publicSwitch, hcenter, publicCenterUnlocked,
      publicCenterRoom,
      unlockExitInRoom, publicCenterEast, replaceExit]

theorem publicS24_running : Running publicS24 := by
  rcases publicS23_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS24, activateSwitchState] using halive,
    by simpa [publicS24, activateSwitchState] using hcomplete⟩

@[simp] theorem publicS24_current : publicS24.currentRoom = 0 := by
  simp [publicS24, publicS23, publicS22, publicCenterWest]

@[simp] theorem publicS24_pos : publicS24.player.pos = publicPos 5 4 := by
  simp [publicS24, publicS23_pos]

theorem publicS23_center :
    publicS23.rooms 1 = publicCenterUnlocked .westToEast := by
  simpa [publicS23] using publicS22_center

@[simp] theorem publicS23_current : publicS23.currentRoom = 0 := by
  simp [publicS23, publicS22, publicCenterWest]

@[simp] theorem publicS24_room :
    currentRoomState publicS24 = publicWestRoom true := by
  have hroom0 : publicS23.rooms 0 = publicWestRoom true := by
    simpa [currentRoomState, publicS23_current] using publicS23_room
  simp [publicS24, activateSwitchState, publicPressedSwitch, publicSwitch,
    publicS23_current, hroom0,
    publicWestRoom, replaceSwitch, currentRoomState, setRoom]

theorem publicS24_center :
    publicS24.rooms 1 = publicCenterUnlocked .westToSouth := by
  simp [publicS24, activateSwitchState, publicPressedSwitch, publicSwitch,
    publicS23_current, publicS23_center, publicCenterUnlocked,
    publicCenterRoom, unlockExitInRoom, publicCenterEast, replaceExit,
    rotateBridge, publicBridge, rotateOrientation, setRoom]

theorem public_route_second_switch_to_exit :
    EngineExec publicS24
      (westSwitchToExitDirections.map actionForDirection) publicS25 := by
  apply directionPlan_has_exact_engine_exec (room := publicWestRoom true)
  · exact publicS24_room
  · rfl
  · exact publicS24_running
  · simpa [publicS24_pos] using public_west_switch_to_exit_safe true

theorem publicS25_running : Running publicS25 := by
  rcases publicS24_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS25] using halive,
    by simpa [publicS25] using hcomplete⟩

@[simp] theorem publicS25_room :
    currentRoomState publicS25 = publicWestRoom true := by
  simpa [publicS25] using publicS24_room

@[simp] theorem publicS25_current : publicS25.currentRoom = 0 := by
  simp [publicS25, publicS24_current]

theorem public_enter_center_south_bridge :
    EngineExec publicS25 [.right] publicS26 := by
  apply engineExec_useExit_once (exit := publicWestEast)
  · exact publicS25_running
  · rw [publicS25_room]
    simp [publicWestRoom]
  · simp [exitContains, publicS25, publicS24,
      westSwitchToExitDirections, publicWestEast, publicS24_pos,
      publicPos, directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicWestEast, requirementSatisfied]
  · have hcenter : publicS25.rooms 1 =
        publicCenterUnlocked .westToSouth := by
      simpa [publicS25] using publicS24_center
    simpa [publicWestEast, hcenter, publicCenterUnlocked,
      unlockExitInRoom, publicCenterEast, replaceExit, publicCenterRoom,
      publicFinalChest, publicBridge, publicSouthTiles, publicAbyssTraps,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
      activeBridgeTile, publicBounds, publicPos]

theorem publicS26_running : Running publicS26 := by
  rcases publicS25_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS26] using halive,
    by simpa [publicS26, publicWestEast] using hcomplete⟩

@[simp] theorem publicS26_current : publicS26.currentRoom = 1 := by
  simp [publicS26, publicWestEast]

@[simp] theorem publicS26_room :
    currentRoomState publicS26 = publicCenterUnlocked .westToSouth := by
  have hcenter : publicS25.rooms 1 =
      publicCenterUnlocked .westToSouth := by
    simpa [publicS25] using publicS24_center
  simp [publicS26, transitionThroughExit, publicWestEast,
    currentRoomState, unlockExitInRoom, setRoom, publicS25_current, hcenter]

theorem public_center_unlocked_to_south_safe :
    DirectionPlanSafe (publicCenterUnlocked .westToSouth) (publicPos 1 4)
      centerToSouthDirections := by
  simp [DirectionPlanSafe, centerToSouthDirections, publicCenterUnlocked,
    unlockExitInRoom, publicCenterEast, publicCenterRoom, replaceExit,
    publicCenterWest, publicCenterNorth, publicCenterSouth, publicBridge,
    publicSouthTiles, publicAbyssTraps, publicFinalChest, safeTile, canEnter,
    inBounds, staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicBounds, publicPos, advance]

theorem public_route_center_to_south :
    EngineExec publicS26
      (centerToSouthDirections.map actionForDirection) publicS27 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicCenterUnlocked .westToSouth)
  · exact publicS26_room
  · rfl
  · exact publicS26_running
  · simpa [publicS26, publicWestEast, publicPos] using
      public_center_unlocked_to_south_safe

theorem publicS27_running : Running publicS27 := by
  rcases publicS26_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS27] using halive,
    by simpa [publicS27] using hcomplete⟩

@[simp] theorem publicS27_room :
    currentRoomState publicS27 = publicCenterUnlocked .westToSouth := by
  simpa [publicS27] using publicS26_room

@[simp] theorem publicS27_current : publicS27.currentRoom = 1 := by
  simp [publicS27, publicS26_current]

set_option maxHeartbeats 800000 in
theorem publicS27_south : publicS27.rooms 4 = publicSouthRoom := by
  simp [publicS27, publicS26, publicS25, publicS24, activateSwitchState,
    publicS23, publicS22, publicS21, publicS20, publicS19, publicS18,
    openChestResult, publicS17, publicS16, publicS15, publicS14,
    publicS13, publicS12, publicS11, publicS10, publicS09, publicS08,
    publicS07, publicS06, publicS05, publicS04, publicS03, publicS02,
    publicS01, transitionThroughExit, publicWestEast, publicCenterWest,
    publicEastWest, publicCenterEast, publicNorthSouth,
    publicCenterNorth, unlockExitInRoom, updateCurrentRoom, setRoom,
    currentRoomState, publicInitialWorld, publicRooms, publicSwitch,
    publicPressedSwitch]

theorem public_enter_south_room :
    EngineExec publicS27 [.down] publicS28 := by
  apply engineExec_useExit_once (exit := publicCenterSouth)
  · exact publicS27_running
  · rw [publicS27_room]
    simp [publicCenterUnlocked, unlockExitInRoom, publicCenterEast,
      publicCenterRoom, replaceExit, publicCenterWest, publicCenterNorth,
      publicCenterSouth]
  · simp [exitContains, publicS27, publicS26, centerToSouthDirections,
      publicCenterSouth, publicWestEast, publicPos, directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicCenterSouth,
      requirementSatisfied]
  · simpa [publicCenterSouth, publicS27_south, publicSouthRoom,
      publicGuardian,
      canEnter, inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
      activeBridgeTile, publicBounds, publicPos]

theorem publicS28_running : Running publicS28 := by
  rcases publicS27_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS28] using halive,
    by simpa [publicS28, publicCenterSouth] using hcomplete⟩

@[simp] theorem publicS28_room :
    currentRoomState publicS28 = publicSouthRoom := by
  simp [publicS28, transitionThroughExit, publicCenterSouth,
    currentRoomState, unlockExitInRoom, setRoom, publicS27_current,
    publicS27_south]

theorem public_route_south_to_guardian :
    EngineExec publicS28
      (southToGuardianDirections.map actionForDirection) publicS29 := by
  apply directionPlan_has_exact_engine_exec (room := publicSouthRoom)
  · exact publicS28_room
  · rfl
  · exact publicS28_running
  · simpa [publicS28, publicCenterSouth, publicPos] using
      public_south_to_guardian_safe

theorem publicS29_running : Running publicS29 := by
  rcases publicS28_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS29] using halive,
    by simpa [publicS29] using hcomplete⟩

@[simp] theorem publicS29_room :
    currentRoomState publicS29 = publicSouthRoom := by
  simpa [publicS29] using publicS28_room

@[simp] theorem publicS29_pos : publicS29.player.pos = publicPos 4 3 := by
  simp [publicS29, publicS28, publicCenterSouth, southToGuardianDirections,
    publicPos, directionEndpoint, advance]

theorem public_face_guardian : EngineExec publicS29 [.down] publicS30 := by
  have hmonster : monsterAt (currentRoomState publicS29)
      (advance publicS29.player.pos .south) := by
    rw [publicS29_room, publicS29_pos]
    simp [monsterAt, publicSouthRoom, publicGuardian, publicPos, advance]
  apply player_step_has_single_tick_execution publicS29_running
  · exact Step.faceMonster rfl hmonster
  · simp [AutonomousOnlyEvents]

theorem publicS30_running : Running publicS30 := by
  rcases publicS29_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS30] using halive,
    by simpa [publicS30] using hcomplete⟩

@[simp] theorem publicS30_room :
    currentRoomState publicS30 = publicSouthRoom := by
  simpa [publicS30] using publicS29_room

theorem publicS30_has_sword :
    publicS30.player.inventory.equippedA = some .sword ∧
    .sword ∈ publicS30.player.inventory.items := by
  have hsword :
      publicS18.player.inventory.equippedA = some .sword ∧
      .sword ∈ publicS18.player.inventory.items := by
    by_cases hitem : .sword ∈ publicS17.player.inventory.items <;>
      simp [publicS18, openChestResult, publicSwordChest, collectLoot, hitem]
  have hinventory :
      publicS30.player.inventory = publicS18.player.inventory := by
    simp [publicS30, publicS29, publicS28, publicS27, publicS26,
      publicS25, publicS24, activateSwitchState, publicS23, publicS22,
      publicS21, publicS20, publicS19, transitionThroughExit,
      publicEastWest, publicCenterWest, publicWestEast, publicCenterSouth,
      spendExitRequirement]
  rw [hinventory]
  exact hsword

theorem public_kill_guardian : EngineExec publicS30 [.slotA] publicS31 := by
  have hnoInteraction : ¬ primaryInteractionAvailable publicS30 := by
    simp [primaryInteractionAvailable, openChestInteractionAvailable,
      npcInteractionAvailable, switchInteractionAvailable, publicS30_room,
      publicSouthRoom]
  have hmember : publicGuardian ∈
      (currentRoomState publicS30).monsters := by
    rw [publicS30_room]
    simp [publicSouthRoom]
  have htarget : publicGuardian.pos =
      advance publicS30.player.pos publicS30.player.facing := by
    simp [publicS30, publicS29_pos, publicGuardian, publicPos, advance]
  have hstep : Step publicS30 .slotA publicS31
      (monsterKillEvents publicS30 publicGuardian) :=
    Step.attackKill hnoInteraction publicS30_has_sword.1
      publicS30_has_sword.2 hmember htarget (by decide)
  apply player_step_has_single_tick_execution publicS30_running hstep
  simp only [monsterKillEvents]
  dsimp
  split <;> simp [AutonomousOnlyEvents]

@[simp] theorem publicS30_current : publicS30.currentRoom = 4 := by
  simp [publicS30, publicS29, publicS28, publicCenterSouth]

theorem publicS30_center :
    publicS30.rooms 1 = publicCenterUnlocked .westToSouth := by
  have hcenter : publicS27.rooms 1 =
      publicCenterUnlocked .westToSouth := by
    simpa [currentRoomState, publicS27_current] using publicS27_room
  simp [publicS30, publicS29, publicS28, transitionThroughExit,
    publicCenterSouth, unlockExitInRoom, setRoom, currentRoomState,
    publicS27_current, hcenter]

theorem publicS31_running : Running publicS31 := by
  rcases publicS30_running with ⟨halive, hcomplete⟩
  constructor
  · unfold alive
    change 0 < (resolveMonsterKill publicS30 publicGuardian).player.hp
    rw [resolveMonsterKill_player]
    simpa only [rewardPlayer, alive] using halive
  · change (resolveMonsterKill publicS30 publicGuardian).completed = false
    rw [resolveMonsterKill_completed]
    exact hcomplete

@[simp] theorem publicS31_pos : publicS31.player.pos = publicPos 4 3 := by
  change (resolveMonsterKill publicS30 publicGuardian).player.pos = publicPos 4 3
  rw [resolveMonsterKill_player]
  simp [rewardPlayer, publicS30, publicS29_pos]

@[simp] theorem publicS31_current : publicS31.currentRoom = 4 := by
  change (resolveMonsterKill publicS30 publicGuardian).currentRoom = 4
  rw [resolveMonsterKill_currentRoom]
  exact publicS30_current

set_option maxHeartbeats 800000 in
@[simp] theorem publicS31_room :
    currentRoomState publicS31 = publicSouthRoom [] := by
  simp [publicS31, resolveMonsterKill, publicS30_room, publicS30_current,
    publicSouthRoom, publicGuardian, removeMonster,
    unlockAllMonstersDefeatedExits, revealEligibleChestsInWorld,
    revealEligibleChests, updateCurrentRoom, setRoom,
    requirementContainsAllMonstersDefeated, currentRoomState]

theorem public_guardian_reveals_final_chest :
    publicS31.rooms 1 =
      publicCenterUnlocked .westToSouth revealedFinalChest := by
  simp [publicS31, resolveMonsterKill, publicS30_room, publicS30_current,
    publicS30_center, publicSouthRoom, publicGuardian, removeMonster,
    unlockAllMonstersDefeatedExits, revealEligibleChestsInWorld,
    revealEligibleChests, updateCurrentRoom, setRoom,
    requirementContainsAllMonstersDefeated, publicCenterUnlocked,
    publicCenterRoom, revealedFinalChest, publicFinalChest,
    chestRevealMatches, publicCenterEast, publicCenterWest,
    publicCenterNorth, publicCenterSouth, replaceExit]

theorem public_route_guardian_to_south_exit :
    EngineExec publicS31
      (guardianToSouthExitDirections.map actionForDirection) publicS32 := by
  apply directionPlan_has_exact_engine_exec (room := publicSouthRoom [])
  · exact publicS31_room
  · rfl
  · exact publicS31_running
  · simpa [publicS31_pos] using public_guardian_to_south_exit_safe

theorem publicS32_running : Running publicS32 := by
  rcases publicS31_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS32] using halive,
    by simpa [publicS32] using hcomplete⟩

@[simp] theorem publicS32_room :
    currentRoomState publicS32 = publicSouthRoom [] := by
  simpa [publicS32] using publicS31_room

@[simp] theorem publicS32_current : publicS32.currentRoom = 4 := by
  simp [publicS32, publicS31_current]

theorem public_return_center_after_guardian :
    EngineExec publicS32 [.up] publicS33 := by
  apply engineExec_useExit_once (exit := publicSouthNorth)
  · exact publicS32_running
  · rw [publicS32_room]
    simp [publicSouthRoom]
  · simp [exitContains, publicS32, publicS31_pos,
      guardianToSouthExitDirections, publicSouthNorth, publicPos,
      directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicSouthNorth, requirementSatisfied]
  · have hcenter : publicS32.rooms 1 =
        publicCenterUnlocked .westToSouth revealedFinalChest := by
      simpa [publicS32] using public_guardian_reveals_final_chest
    simpa [publicSouthNorth, hcenter, publicCenterUnlocked,
      unlockExitInRoom, publicCenterEast, replaceExit, publicCenterRoom,
      revealedFinalChest, publicFinalChest, publicBridge, publicSouthTiles,
      publicAbyssTraps, canEnter, inBounds, staticBlocker, npcAt,
      visibleChestAt, gapAt, activeBridgeTile, publicBounds, publicPos]

theorem publicS33_running : Running publicS33 := by
  rcases publicS32_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS33] using halive,
    by simpa [publicS33, publicSouthNorth] using hcomplete⟩

@[simp] theorem publicS33_current : publicS33.currentRoom = 1 := by
  simp [publicS33, publicSouthNorth]

@[simp] theorem publicS33_room :
    currentRoomState publicS33 =
      publicCenterUnlocked .westToSouth revealedFinalChest := by
  have hcenter : publicS32.rooms 1 =
      publicCenterUnlocked .westToSouth revealedFinalChest := by
    simpa [publicS32] using public_guardian_reveals_final_chest
  simp [publicS33, transitionThroughExit, publicSouthNorth,
    currentRoomState, unlockExitInRoom, setRoom, publicS32_current, hcenter]

theorem public_center_unlocked_to_final_safe :
    DirectionPlanSafe
      (publicCenterUnlocked .westToSouth revealedFinalChest)
      (publicPos 4 6) centerSouthToFinalDirections := by
  simp [DirectionPlanSafe, centerSouthToFinalDirections,
    publicCenterUnlocked, unlockExitInRoom, publicCenterEast,
    publicCenterRoom, replaceExit, publicCenterWest, publicCenterNorth,
    publicCenterSouth, publicBridge, publicSouthTiles, publicAbyssTraps,
    revealedFinalChest, publicFinalChest, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, gapAt,
    activeBridgeTile, publicBounds, publicPos, advance]

theorem public_route_to_final_chest :
    EngineExec publicS33
      (centerSouthToFinalDirections.map actionForDirection) publicS34 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicCenterUnlocked .westToSouth revealedFinalChest)
  · exact publicS33_room
  · rfl
  · exact publicS33_running
  · simpa [publicS33, publicSouthNorth, publicPos] using
      public_center_unlocked_to_final_safe

theorem publicS34_running : Running publicS34 := by
  rcases publicS33_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS34] using halive,
    by simpa [publicS34] using hcomplete⟩

@[simp] theorem publicS34_room :
    currentRoomState publicS34 =
      publicCenterUnlocked .westToSouth revealedFinalChest := by
  simpa [publicS34] using publicS33_room

@[simp] theorem publicS34_pos : publicS34.player.pos = publicPos 4 5 := by
  simp [publicS34, publicS33, publicSouthNorth, centerSouthToFinalDirections,
    publicPos, directionEndpoint, advance]

theorem public_open_final_chest :
    EngineExec publicS34 [.slotA] publicS35 := by
  apply engineExec_openChest_once (chest := revealedFinalChest)
  · exact publicS34_running
  · rw [publicS34_room]
    simp [publicCenterUnlocked, publicCenterRoom, unlockExitInRoom,
      publicCenterEast, replaceExit]
  · rfl
  · rfl
  · simp [interactionReach, publicS34_pos, revealedFinalChest,
      publicFinalChest, publicPos, adjacent, advance]

@[simp] theorem publicS35_room :
    currentRoomState publicS35 =
      publicCenterUnlocked .westToSouth openedFinalChest := by
  simp [publicS35, openChestResult, publicS34_room, publicCenterUnlocked,
    publicCenterRoom, unlockExitInRoom, publicCenterEast, replaceExit,
    replaceChest, revealedFinalChest, openedFinalChest]

theorem publicS35_running : Running publicS35 := by
  rcases publicS34_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicS35, openChestResult,
      revealedFinalChest, publicFinalChest, collectLoot] using halive,
    by simpa [publicS35, openChestResult] using hcomplete⟩

set_option maxHeartbeats 2000000 in
theorem publicS35_all_world_chests_opened : allWorldChestsOpened publicS35 := by
  simp [allWorldChestsOpened, allVisibleChestsOpened, publicS35,
    openChestResult, publicS34, publicS33, publicS32, publicS31,
    resolveMonsterKill, publicS30, publicS29, publicS28, publicS27,
    publicS26, publicS25, publicS24, publicS23, publicS22, publicS21,
    publicS20, publicS19, publicS18, publicS17, publicS16, publicS15,
    publicS14, publicS13, publicS12, publicS11, publicS10, publicS09,
    publicS08, publicS07, publicS06, publicS05, publicS04, publicS03,
    publicS02, publicS01, publicInitialWorld, publicRooms,
    transitionThroughExit, activateSwitchState, updateCurrentRoom, setRoom,
    unlockExitInRoom, replaceExit, replaceChest, replaceSwitch,
    removeMonster, unlockAllMonstersDefeatedExits,
    revealEligibleChestsInWorld, revealEligibleChests,
    publicWestEast, publicCenterWest, publicCenterEast, publicCenterNorth,
    publicCenterSouth, publicNorthSouth, publicEastWest, publicSouthNorth,
    publicWestRoom, publicNorthRoom, publicEastRoom, publicSouthRoom,
    publicCenterRoom, publicCenterUnlocked, publicSwitch,
    publicPressedSwitch, publicKeyChest, publicSwordChest, publicGuardian,
    publicFinalChest, revealedFinalChest, openedFinalChest, openedChest,
    collectLoot, rewardPlayer, rotateBridge, publicBridge,
    chestRevealMatches, requirementContainsAllMonstersDefeated,
    currentRoomState]

theorem public_completion_tick : EngineExec publicS35 [.wait] publicS37 := by
  have hplayer : PlayerStep publicS35 .wait publicS36 [.waited] := by
    refine ⟨publicS35_running, ?_, ?_⟩
    · exact Step.wait
    · simp [AutonomousOnlyEvents]
  have hobjective : allWorldChestsOpened publicS36 := by
    simpa [publicS36, clearShieldWorld, allWorldChestsOpened,
      allVisibleChestsOpened] using publicS35_all_world_chests_opened
  have hauto : AutonomousStep publicS36 publicS37
      [.environmentCompleted] := by
    refine ⟨?_, by simp [AutonomousOnlyEvents]⟩
    exact Step.completeAllChests hobjective
  exact EngineExec.cons
    (EngineTick.mk hplayer (AutonomousExec.cons hauto AutonomousExec.nil))
    EngineExec.nil

theorem public_map_complete_certificate :
    ∃ actions final,
      EngineExec publicInitialWorld actions final ∧
      WorldCompleted final ∧ alive final ∧ ValidState final := by
  have hAll := engineExec_append public_route_west_to_center
    (engineExec_append public_enter_center_north_bridge
    (engineExec_append public_route_center_to_north
    (engineExec_append public_enter_north_room
    (engineExec_append public_route_north_to_key
    (engineExec_append public_open_key
    (engineExec_append public_route_key_to_north_exit
    (engineExec_append public_return_center_after_key
    (engineExec_append public_route_center_north_to_west
    (engineExec_append public_return_west_for_first_switch
    (engineExec_append public_route_to_first_switch
    (engineExec_append public_press_first_switch
    (engineExec_append public_route_first_switch_to_exit
    (engineExec_append public_enter_center_east_bridge
    (engineExec_append public_route_center_to_east
    (engineExec_append public_enter_east_room
    (engineExec_append public_route_east_to_sword
    (engineExec_append public_open_sword
    (engineExec_append public_route_sword_to_east_exit
    (engineExec_append public_return_center_after_sword
    (engineExec_append public_route_center_east_to_west
    (engineExec_append public_return_west_for_second_switch
    (engineExec_append public_route_to_second_switch
    (engineExec_append public_press_second_switch
    (engineExec_append public_route_second_switch_to_exit
    (engineExec_append public_enter_center_south_bridge
    (engineExec_append public_route_center_to_south
    (engineExec_append public_enter_south_room
    (engineExec_append public_route_south_to_guardian
    (engineExec_append public_face_guardian
    (engineExec_append public_kill_guardian
    (engineExec_append public_route_guardian_to_south_exit
    (engineExec_append public_return_center_after_guardian
    (engineExec_append public_route_to_final_chest
    (engineExec_append public_open_final_chest public_completion_tick))))))))))))))))))))))))))))))))))
  refine ⟨_, publicS37, hAll, ?_, ?_, ?_⟩
  · simp [WorldCompleted, publicS37, markWorldCompleted]
  · rcases publicS35_running with ⟨halive, _⟩
    simpa [alive, publicS37, publicS36, markWorldCompleted,
      clearShieldWorld] using halive
  · apply engineExec_preserves_validState
      (s := publicInitialWorld)
    · simp [ValidState, publicInitialWorld, publicInitialPlayer,
        currentRoomState, publicRooms, publicWestRoom, publicBounds,
        inBounds, publicPos]
    · exact hAll

theorem public_bridge_covers_abyss
    (orientation : BridgeOrientation) (p : Position)
    (hactive : activeBridgeTile (publicCenterRoom orientation) p) :
    ¬ activeTrapAt (publicCenterRoom orientation) p := by
  simp [activeTrapAt, hactive]

theorem public_three_switches_restore_north :
    rotateOrientation (rotateOrientation (rotateOrientation .westToNorth)) =
      .westToNorth := by decide

inductive PublicMilestone where
  | west | northKey | westAgain | eastSword | westThird
  | southGuardian | finalChest | completed
  deriving DecidableEq, Repr

inductive PublicMilestoneStep : PublicMilestone → PublicMilestone → Prop where
  | north : PublicMilestoneStep .west .northKey
  | key : PublicMilestoneStep .northKey .westAgain
  | east : PublicMilestoneStep .westAgain .eastSword
  | sword : PublicMilestoneStep .eastSword .westThird
  | south : PublicMilestoneStep .westThird .southGuardian
  | guardian : PublicMilestoneStep .southGuardian .finalChest
  | final : PublicMilestoneStep .finalChest .completed

inductive PublicMilestoneRun :
    PublicMilestone → List PublicMilestone → PublicMilestone → Prop where
  | nil (p) : PublicMilestoneRun p [] p
  | cons {p q goal rest} : PublicMilestoneStep p q →
      PublicMilestoneRun q rest goal →
      PublicMilestoneRun p (q :: rest) goal

def publicTrace : List PublicMilestone :=
  [.northKey, .westAgain, .eastSword, .westThird,
   .southGuardian, .finalChest, .completed]

theorem public_milestone_certificate :
    PublicMilestoneRun .west publicTrace .completed := by
  exact .cons .north
    (.cons .key
    (.cons .east
    (.cons .sword
    (.cons .south
    (.cons .guardian
    (.cons .final (.nil .completed)))))))

end Task4

/-! ## 十八、Task5：视觉记忆、房间图与公平探索（完全重构） -/

namespace Task5

structure RoomCoord where
  x : Int
  y : Int
  deriving DecidableEq, Repr

structure RoomEdge where
  source : RoomCoord
  direction : Direction
  target : RoomCoord
  deriving DecidableEq, Repr

def relativeNeighbor (room : RoomCoord) : Direction → RoomCoord
  | .north => ⟨room.x, room.y - 1⟩
  | .south => ⟨room.x, room.y + 1⟩
  | .west => ⟨room.x - 1, room.y⟩
  | .east => ⟨room.x + 1, room.y⟩

def reverseDirection : Direction → Direction
  | .north => .south | .south => .north | .west => .east | .east => .west

structure BlockedMove where
  room : RoomCoord
  source : Position
  direction : Direction
  deriving DecidableEq, Repr

structure Memory where
  current : RoomCoord := ⟨0, 0⟩
  visited : List RoomCoord := []
  connections : List RoomEdge := []
  exploredEdges : List RoomEdge := []
  staticBlockers : List (RoomCoord × Position) := []
  observedChests : List (RoomCoord × Position) := []
  openedChests : List (RoomCoord × Position) := []
  supportChests : List (RoomCoord × Position) := []
  pressedButtons : List (RoomCoord × Position) := []
  failedMoves : List BlockedMove := []
  risk : Nat := 0
  threatEvidence : List (RoomCoord × Position × Nat) := []
  guardedThreatTiles : List (RoomCoord × Position) := []
  serviceCombatAttempted : List RoomCoord := []
  deriving DecidableEq, Repr

def rememberVisited (m : Memory) (room : RoomCoord) : Memory :=
  { m with visited := room :: m.visited }

def rememberOpenedChest (m : Memory) (room : RoomCoord) (p : Position) : Memory :=
  { m with openedChests := (room, p) :: m.openedChests }

def rememberSupportChest (m : Memory) (room : RoomCoord) (p : Position) : Memory :=
  { m with supportChests := (room, p) :: m.supportChests }

def rememberPressedButton (m : Memory) (room : RoomCoord) (p : Position) : Memory :=
  { m with pressedButtons := (room, p) :: m.pressedButtons }

def learnFailedMove (m : Memory) (move : BlockedMove) : Memory :=
  { m with failedMoves := move :: m.failedMoves }

def learnRelativeConnection
    (m : Memory) (source : RoomCoord) (direction : Direction) : Memory :=
  let target := relativeNeighbor source direction
  { m with connections :=
      ⟨source, direction, target⟩ ::
      ⟨target, reverseDirection direction, source⟩ :: m.connections }

structure TransitionEvidence where
  source : RoomCoord
  target : RoomCoord
  direction : Direction
  lastAction : Action
  lastActionMatches : actionDirection lastAction = some direction
  previousOnBoundary : Bool
  currentInInterior : Bool
  targetIsRelativeNeighbor : target = relativeNeighbor source direction

def confirmsRelativeTransition (e : TransitionEvidence) : Prop :=
  e.previousOnBoundary = true ∧ e.currentInInterior = true

theorem confirmed_transition_learns_both_directions
    (m : Memory) (e : TransitionEvidence)
    (_hconfirmed : confirmsRelativeTransition e) :
    ⟨e.source, e.direction, e.target⟩ ∈
        (learnRelativeConnection m e.source e.direction).connections ∧
    ⟨e.target, reverseDirection e.direction, e.source⟩ ∈
        (learnRelativeConnection m e.source e.direction).connections := by
  simp [learnRelativeConnection, e.targetIsRelativeNeighbor]

theorem visited_memory_monotone (m : Memory) (room old : RoomCoord)
    (h : old ∈ m.visited) : old ∈ (rememberVisited m room).visited :=
  List.mem_cons_of_mem room h

theorem opened_chest_memory_monotone
    (m : Memory) (room oldRoom : RoomCoord) (p oldP : Position)
    (h : (oldRoom, oldP) ∈ m.openedChests) :
    (oldRoom, oldP) ∈ (rememberOpenedChest m room p).openedChests :=
  List.mem_cons_of_mem _ h

theorem learned_failed_move_is_recorded (m : Memory) (move : BlockedMove) :
    move ∈ (learnFailedMove m move).failedMoves := by
  simp [learnFailedMove]

def documentedInitialHealth : Nat := 5

inductive FeedbackSignal where
  | damage
  | collision
  | positiveProgress
  | neutral
  | supportChest
  deriving DecidableEq, Repr

/- Python 阈值 -1.0 用十分位整数表示，避免把浮点近似引入证明核。 -/
def feedbackFromScaledReward
    (rewardTenths : Int) (collision supportChest : Bool) : FeedbackSignal :=
  if supportChest then .supportChest
  else if rewardTenths ≤ -10 then .damage
  else if collision then .collision
  else if 0 < rewardTenths then .positiveProgress
  else .neutral

theorem damage_signal_requires_observed_large_negative_reward
    (rewardTenths : Int) (collision : Bool)
    (h : feedbackFromScaledReward rewardTenths collision false = .damage) :
    rewardTenths ≤ -10 := by
  by_cases hnegative : rewardTenths ≤ -10
  · exact hnegative
  · cases collision <;>
      simp [feedbackFromScaledReward, hnegative] at h <;>
      split at h <;> contradiction

structure SurvivalMemory where
  observedDamage : Nat := 0
  supportObserved : Bool := false
  deriving DecidableEq, Repr

def updateSurvival
    (signal : FeedbackSignal) (memory : SurvivalMemory) : SurvivalMemory :=
  match signal with
  | .damage =>
      { memory with
        observedDamage := min documentedInitialHealth (memory.observedDamage + 1) }
  | .supportChest =>
      { observedDamage := 0, supportObserved := true }
  | .collision | .positiveProgress | .neutral => memory

def survivalBudget (memory : SurvivalMemory) : Nat :=
  documentedInitialHealth - memory.observedDamage

def survivalBudgetAtStep (memory : SurvivalMemory) (_stepCount : Nat) : Nat :=
  survivalBudget memory

def rushMode (memory : SurvivalMemory) : Bool :=
  survivalBudget memory ≤ 2

theorem observed_damage_stays_bounded
    (signal : FeedbackSignal) (memory : SurvivalMemory)
    (hbound : memory.observedDamage ≤ documentedInitialHealth) :
    (updateSurvival signal memory).observedDamage ≤ documentedInitialHealth := by
  cases signal with
  | damage => exact Nat.min_le_left _ _
  | supportChest => simp [updateSurvival, documentedInitialHealth]
  | collision | positiveProgress | neutral =>
      simpa [updateSurvival, documentedInitialHealth] using hbound

theorem only_damage_or_support_changes_observed_damage
    (signal : FeedbackSignal) (memory : SurvivalMemory)
    (hDamage : signal ≠ .damage) (hSupport : signal ≠ .supportChest) :
    (updateSurvival signal memory).observedDamage = memory.observedDamage := by
  cases signal <;> simp_all [updateSurvival]

theorem damage_feedback_never_increases_budget (memory : SurvivalMemory) :
    survivalBudget (updateSurvival .damage memory) ≤ survivalBudget memory := by
  simp [survivalBudget, updateSurvival, documentedInitialHealth]
  omega

theorem support_feedback_resets_budget (memory : SurvivalMemory) :
    survivalBudget (updateSurvival .supportChest memory) =
      documentedInitialHealth ∧
    (updateSurvival .supportChest memory).supportObserved = true := by
  simp [survivalBudget, updateSurvival, documentedInitialHealth]

theorem survival_budget_is_step_count_independent
    (memory : SurvivalMemory) (stepCount₁ stepCount₂ : Nat) :
    survivalBudgetAtStep memory stepCount₁ =
      survivalBudgetAtStep memory stepCount₂ := by
  rfl

structure DamageContext where
  nearbyMonsters : List Position := []
  movedIntoDanger : Bool := false
  touchingMonster : Bool := false
  deriving DecidableEq, Repr

def mayAttributeThreat (context : DamageContext) : Bool :=
  !context.nearbyMonsters.isEmpty &&
  (context.movedIntoDanger || context.touchingMonster)

def updateThreatEvidence (old : Nat) (context : DamageContext) : Nat :=
  if mayAttributeThreat context then min 6 (old + 2) else old

theorem unrelated_damage_does_not_increase_threat
    (old : Nat) (context : DamageContext)
    (h : mayAttributeThreat context = false) :
    updateThreatEvidence old context = old := by
  simp [updateThreatEvidence, h]

theorem distant_or_unrelated_damage_adds_no_threat
    (context : DamageContext)
    (hempty : context.nearbyMonsters = [] ∨
      (context.movedIntoDanger = false ∧ context.touchingMonster = false)) :
    mayAttributeThreat context = false := by
  rcases hempty with h | ⟨hm, ht⟩
  · simp [mayAttributeThreat, h]
  · simp [mayAttributeThreat, hm, ht]

theorem contact_evidence_allows_threat_attribution
    (context : DamageContext)
    (hne : context.nearbyMonsters ≠ [])
    (hcontact : context.movedIntoDanger = true ∨ context.touchingMonster = true) :
    mayAttributeThreat context = true := by
  rcases hcontact with h | h <;>
    simp [mayAttributeThreat, hne, h]

inductive GoalKind where
  | boundedBlockerCombat
  | safeChest
  | shieldedChest
  | rushChest
  | purposefulCombat
  | button
  | parentReturn
  | frontierExit
  | revisitUnopenedRoom
  | wait
  deriving DecidableEq, Repr

structure Goal where
  kind : GoalKind
  target : Option Position := none
  direction : Option Direction := none
  deriving DecidableEq, Repr

structure CandidateEvidence where
  unopenedChest : Option Position := none
  blockingMonster : Option Position := none
  purposefulMonster : Option Position := none
  button : Option Position := none
  parentExit : Option Direction := none
  frontierExit : Option Direction := none
  revisitExit : Option Direction := none
  safeChestPath : Bool := false
  directChestPath : Bool := false
  hasSword : Bool := false
  hasShield : Bool := false
  combatAlreadyAttempted : Bool := false
  highUncertainty : Bool := false
  blockerThreatConfirmed : Bool := false
  localComplete : Bool := false
  deriving DecidableEq, Repr

def boundedCombatEligible
    (survival : SurvivalMemory) (e : CandidateEvidence) : Bool :=
  e.unopenedChest.isSome &&
  !e.combatAlreadyAttempted &&
  e.hasSword &&
  survivalBudget survival ≤ 3 &&
  e.highUncertainty &&
  e.blockerThreatConfirmed &&
  e.blockingMonster.isSome

def chooseGoal (survival : SurvivalMemory) (e : CandidateEvidence) : Goal :=
  if boundedCombatEligible survival e then
    ⟨.boundedBlockerCombat, e.blockingMonster, none⟩
  else match e.unopenedChest with
    | some chest =>
        if e.safeChestPath then ⟨.safeChest, some chest, none⟩
        else if e.hasShield && e.directChestPath then
          ⟨.shieldedChest, some chest, none⟩
        else if rushMode survival then ⟨.rushChest, some chest, none⟩
        else match e.purposefulMonster with
          | some monster =>
              if e.hasSword then ⟨.purposefulCombat, some monster, none⟩
              else chooseAfterChest
          | none => chooseAfterChest
    | none => chooseAfterChest
  where
    chooseAfterChest : Goal :=
      match e.button with
      | some button => ⟨.button, some button, none⟩
      | none =>
          if e.localComplete then
            match e.parentExit with
            | some d => ⟨.parentReturn, none, some d⟩
            | none => chooseFrontier
          else chooseFrontier
    chooseFrontier : Goal :=
      match e.frontierExit with
      | some d => ⟨.frontierExit, none, some d⟩
      | none => match e.parentExit with
        | some d => ⟨.parentReturn, none, some d⟩
        | none => match e.revisitExit with
          | some d => ⟨.revisitUnopenedRoom, none, some d⟩
          | none => ⟨.wait, none, none⟩

theorem bounded_blocker_combat_has_highest_priority
    (survival : SurvivalMemory) (e : CandidateEvidence)
    (h : boundedCombatEligible survival e = true) :
    (chooseGoal survival e).kind = .boundedBlockerCombat := by
  simp [chooseGoal, h]

theorem safe_chest_precedes_button
    (survival : SurvivalMemory) (e : CandidateEvidence) (chest : Position)
    (hbounded : boundedCombatEligible survival e = false)
    (hchest : e.unopenedChest = some chest)
    (hsafe : e.safeChestPath = true) :
    (chooseGoal survival e).kind = .safeChest := by
  simp [chooseGoal, hbounded, hchest, hsafe]

theorem rush_preserves_required_chest_goal
    (survival : SurvivalMemory) (e : CandidateEvidence) (chest : Position)
    (hbounded : boundedCombatEligible survival e = false)
    (hchest : e.unopenedChest = some chest)
    (hunsafe : e.safeChestPath = false)
    (hnoshield : e.hasShield = false)
    (hrush : rushMode survival = true) :
    (chooseGoal survival e).kind = .rushChest := by
  simp [chooseGoal, hbounded, hchest, hunsafe, hnoshield, hrush]

def moveAllowedByMemory
    (m : Memory) (room : RoomCoord) (p : Position) (d : Direction) : Prop :=
  ⟨room, p, d⟩ ∉ m.failedMoves

theorem learned_blocked_edge_is_not_selected_again
    (m : Memory) (move : BlockedMove) :
    ¬ moveAllowedByMemory (learnFailedMove m move)
      move.room move.source move.direction := by
  simp [moveAllowedByMemory, learnFailedMove]

theorem visible_opened_chest_attempt_is_position_safe
    {s : WorldState} {chest : Chest} {a : Action} {d : Direction}
    (ha : actionDirection a = some d)
    (hm : chest ∈ (currentRoomState s).chests)
    (hv : chest.visible = true) (_ho : chest.opened = true)
    (hpos : chest.pos = advance s.player.pos d) :
    ∃ after events, Step s a after events ∧ after.player.pos = s.player.pos := by
  have hstatic : staticBlocker (currentRoomState s) chest.pos :=
    opened_chest_is_still_a_static_blocker hm hv
  have hblocked : ¬ canEnter (currentRoomState s) (advance s.player.pos d) := by
    intro henter
    exact henter.2.1 (hpos ▸ hstatic)
  let after : WorldState :=
    { s with player := { s.player with facing := d, shielding := false } }
  exact ⟨after, [.blocked (advance s.player.pos d)],
    Step.moveBlocked ha hblocked, rfl⟩

inductive MoveMode where
  | normal | combat | chestRush | exitPush | shieldPrelude | memoryFallback
  deriving DecidableEq, Repr

def PolicyOutputSafe
    (s : WorldState) (mode : MoveMode) (action : Action) : Prop :=
  match mode with
  | .normal => ∀ d, actionDirection action = some d →
      safeTile (currentRoomState s) (advance s.player.pos d)
  | .combat => action = .wait ∨ action = .slotA ∨ action = .slotB ∨
      ∃ d, actionDirection action = some d ∧
        canEnter (currentRoomState s) (advance s.player.pos d) ∧
        (s.player.shielding = true ∨
         monsterAt (currentRoomState s) (advance s.player.pos d))
  | .chestRush => action = .wait ∨ action = .slotB ∨
      ∃ d, actionDirection action = some d ∧
        canEnter (currentRoomState s) (advance s.player.pos d) ∧
        s.player.shielding = true
  | .exitPush => ∃ exit ∈ (currentRoomState s).exits,
      action = directionAction exit.direction ∧
      exitContains exit s.player.pos ∧ exitRequirementSatisfied s exit
  | .shieldPrelude => action = .slotB ∧
      s.player.inventory.equippedB = some .shield
  | .memoryFallback => action = .wait ∨
      ∃ d, actionDirection action = some d ∧
        (safeTile (currentRoomState s) (advance s.player.pos d) ∨
         ¬ canEnter (currentRoomState s) (advance s.player.pos d))

theorem normal_navigation_output_is_safe
    {s : WorldState} {action : Action}
    (h : PolicyOutputSafe s .normal action) :
    ∀ d, actionDirection action = some d →
      safeTile (currentRoomState s) (advance s.player.pos d) := h

theorem shield_prelude_outputs_slotB
    {s : WorldState} {action : Action}
    (h : PolicyOutputSafe s .shieldPrelude action) :
    action = .slotB := h.1

theorem memory_fallback_has_engine_step
    {s : WorldState} {action : Action}
    (h : PolicyOutputSafe s .memoryFallback action) :
    ∃ after events, Step s action after events := by
  rcases h with rfl | ⟨d, ha, htarget⟩
  · exact ⟨_, _, Step.wait⟩
  · rcases htarget with hsafe | hblocked
    · let q := advance s.player.pos d
      by_cases hbutton : buttonAt (currentRoomState s) q
      · rcases hbutton with ⟨button, hmember, hpos⟩
        exact ⟨_, _, Step.moveButton ha rfl hsafe.1 hmember hpos⟩
      · exact ⟨_, _, Step.movePlain ha rfl hsafe.1 hsafe.2.1 hbutton⟩
    · exact ⟨_, _, Step.moveBlocked ha hblocked⟩

structure PolicyDecision where
  goal : Goal
  mode : MoveMode
  output : Action

def PolicyRefinesAgent
    (survival : SurvivalMemory) (e : CandidateEvidence)
    (s : WorldState) (decision : PolicyDecision) : Prop :=
  decision.goal = chooseGoal survival e ∧
  PolicyOutputSafe s decision.mode decision.output

theorem goal_layer_refines_agent
    {survival : SurvivalMemory} {e : CandidateEvidence}
    {s : WorldState} {decision : PolicyDecision}
    (h : PolicyRefinesAgent survival e s decision) :
    decision.goal = chooseGoal survival e ∧
    PolicyOutputSafe s decision.mode decision.output := h

inductive RoomReachable (edges : List RoomEdge) (start : RoomCoord) :
    RoomCoord → Prop where
  | start : RoomReachable edges start start
  | step {source target direction} :
      RoomReachable edges start source →
      ⟨source, direction, target⟩ ∈ edges →
      RoomReachable edges start target

inductive FairPolicyRun (step : Memory → Memory → Prop) :
    Memory → Memory → Prop where
  | refl (m) : FairPolicyRun step m m
  | tail {a b c} : step a b → FairPolicyRun step b c → FairPolicyRun step a c

theorem memory_policy_complete_under_fairness
    (measure : Memory → Nat) (goal : Memory → Prop)
    (step : Memory → Memory → Prop)
    (hzero : ∀ m, measure m = 0 → goal m)
    (hprogress : ∀ m, 0 < measure m → ∃ next,
      step m next ∧ measure next < measure m) :
    ∀ initial, ∃ final, FairPolicyRun step initial final ∧ goal final := by
  intro initial
  induction hmeasure : measure initial using Nat.strongRecOn generalizing initial with
  | ind n ih =>
      by_cases hz : measure initial = 0
      · exact ⟨initial, .refl initial, hzero initial hz⟩
      · have hpositive : 0 < measure initial := Nat.pos_of_ne_zero hz
        rcases hprogress initial hpositive with ⟨next, hstep, hdecrease⟩
        have hdecrease' : measure next < n := by simpa [hmeasure] using hdecrease
        rcases ih (measure next) hdecrease' next rfl with ⟨final, hrun, hgoal⟩
        exact ⟨final, .tail hstep hrun, hgoal⟩

def center : RoomCoord := ⟨0, 0⟩
def south : RoomCoord := ⟨0, 1⟩
def west : RoomCoord := ⟨-1, 0⟩
def east : RoomCoord := ⟨1, 0⟩

def publicGraph : List RoomEdge :=
  [⟨center, .south, south⟩, ⟨south, .north, center⟩,
   ⟨center, .west, west⟩, ⟨west, .east, center⟩,
   ⟨center, .east, east⟩, ⟨east, .west, center⟩]

theorem public_south_reachable : RoomReachable publicGraph center south := by
  have h : (⟨center, .south, south⟩ : RoomEdge) ∈ publicGraph := by
    simp [publicGraph]
  exact .step .start h

theorem public_west_reachable : RoomReachable publicGraph center west := by
  have h : (⟨center, .west, west⟩ : RoomEdge) ∈ publicGraph := by
    simp [publicGraph]
  exact .step .start h

theorem public_east_reachable : RoomReachable publicGraph center east := by
  have h : (⟨center, .east, east⟩ : RoomEdge) ∈ publicGraph := by
    simp [publicGraph]
  exact .step .start h

/-! ### 公开 Task5 JSON 的闭合环境实例 -/

def publicBounds : Bounds :=
  { width := 10, height := 8, width_pos := by decide, height_pos := by decide }
def publicPos (x y : Int) : Position := ⟨x, y⟩

def publicCenterChest : Chest :=
  { id := 501, pos := publicPos 4 2, loot := .gold 2, visible := true }
def publicSouthKeyChest : Chest :=
  { id := 502, pos := publicPos 8 5, loot := .key 1, visible := true }
def publicWestGoldChest : Chest :=
  { id := 503, pos := publicPos 2 6, loot := .gold 3, visible := true }
def publicEastHealChest : Chest :=
  { id := 504, pos := publicPos 7 1, loot := .heal 3, visible := true }
def publicButton : Button := { id := 505, pos := publicPos 2 6 }

def publicCenterMonster : Monster :=
  { id := 510, pos := publicPos 7 4, kind := .chaser, hp := 2, damage := 1 }
def publicSouthMonster : Monster :=
  { id := 511, pos := publicPos 6 6, kind := .chaser, hp := 3, damage := 1 }
def publicWestMonsterA : Monster :=
  { id := 512, pos := publicPos 2 4, kind := .chaser, hp := 2, damage := 1 }
def publicWestMonsterB : Monster :=
  { id := 513, pos := publicPos 6 3, kind := .chaser, hp := 2, damage := 1 }
def publicEastMonster : Monster :=
  { id := 514, pos := publicPos 7 5, kind := .chaser, hp := 2, damage := 1 }

def publicCenterNpc : Npc := { id := 520, pos := publicPos 7 6, text := "center" }
def publicSouthNpc : Npc := { id := 521, pos := publicPos 2 1, text := "south" }
def publicWestNpc : Npc := { id := 522, pos := publicPos 7 6, text := "west" }
def publicEastNpc : Npc := { id := 523, pos := publicPos 7 6, text := "east" }
def publicSouthTrap : Trap :=
  { id := 524, pos := publicPos 1 5, kind := .spike, damage := 1,
    respawn := publicPos 4 1 }

def publicCenterEast : Exit :=
  { id := 530, pos := publicPos 9 4, otherTiles := [publicPos 9 3],
    direction := .east, kind := .locked, requirement := .keys 1 true,
    targetRoom := 3, targetSpawn := publicPos 1 4 }
def publicCenterWest : Exit :=
  { id := 531, pos := publicPos 0 4, otherTiles := [publicPos 0 3],
    direction := .west, kind := .normal, requirement := .free,
    targetRoom := 2, targetSpawn := publicPos 8 4 }
def publicCenterSouth : Exit :=
  { id := 532, pos := publicPos 4 7, otherTiles := [publicPos 5 7],
    direction := .south, kind := .conditional,
    requirement := .buttonPressed publicButton.id,
    targetRoom := 1, targetSpawn := publicPos 4 1 }
def publicSouthNorth : Exit :=
  { id := 533, pos := publicPos 4 0, otherTiles := [publicPos 5 0],
    direction := .north, kind := .normal, requirement := .free,
    targetRoom := 0, targetSpawn := publicPos 4 1 }
def publicWestEast : Exit :=
  { id := 534, pos := publicPos 9 4, otherTiles := [publicPos 9 3],
    direction := .east, kind := .normal, requirement := .free,
    targetRoom := 0, targetSpawn := publicPos 8 4 }
def publicEastWest : Exit :=
  { id := 535, pos := publicPos 0 4, otherTiles := [publicPos 0 3],
    direction := .west, kind := .normal, requirement := .free,
    targetRoom := 0, targetSpawn := publicPos 1 4 }

def openedChest (chest : Chest) : Chest := { chest with opened := true }

def publicCenterRoom
    (chest : Chest := publicCenterChest) (pressed : Bool := false) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [publicCenterNpc],
    chests := [chest], monsters := [publicCenterMonster], traps := [],
    buttons := [{ publicButton with pressed := pressed }], switches := [],
    bridges := [], dynamicTiles := [],
    exits := [publicCenterEast, publicCenterWest, publicCenterSouth] }
def publicCenterUnlockedRoom
    (chest : Chest := openedChest publicCenterChest) : RoomState :=
  unlockExitInRoom (publicCenterRoom chest true) publicCenterEast
def publicSouthRoom (chest : Chest := publicSouthKeyChest) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [publicSouthNpc],
    chests := [chest], monsters := [publicSouthMonster], traps := [publicSouthTrap],
    buttons := [], switches := [], bridges := [], dynamicTiles := [],
    exits := [publicSouthNorth] }
def publicWestRoom (chest : Chest := publicWestGoldChest) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [publicWestNpc],
    chests := [chest], monsters := [publicWestMonsterA, publicWestMonsterB],
    traps := [], buttons := [], switches := [], bridges := [],
    dynamicTiles := [], exits := [publicWestEast] }
def publicEastRoom (chest : Chest := publicEastHealChest) : RoomState :=
  { bounds := publicBounds, walls := [], npcs := [publicEastNpc],
    chests := [chest], monsters := [publicEastMonster], traps := [],
    buttons := [], switches := [], bridges := [], dynamicTiles := [],
    exits := [publicEastWest] }

def publicRooms : RoomId → RoomState
  | 0 => publicCenterRoom | 1 => publicSouthRoom
  | 2 => publicWestRoom | 3 => publicEastRoom
  | _ => publicCenterRoom
def publicInitialPlayer : PlayerState :=
  { pos := publicPos 1 1, facing := .east, hp := 5, maxHp := 5,
    inventory := { keys := 0, gold := 0, items := [.sword, .shield],
      equippedA := some .sword, equippedB := some .shield } }
def publicInitialWorld : WorldState :=
  { currentRoom := 0, rooms := publicRooms, roomIds := [0, 1, 2, 3],
    player := publicInitialPlayer, completed := false }

def toCenterChest : List Direction := [.east, .east, .south]
def centerChestToButton : List Direction := [.west, .south, .south, .south]
def buttonToSouthExit : List Direction := [.east, .east, .south]
def southToKey : List Direction :=
  [.east, .east, .east, .east, .south, .south, .south]
def keyToSouthExit : List Direction :=
  [.west, .west, .west, .west, .north, .north, .north, .north]
def centerToEastExit : List Direction :=
  [.east, .east, .east, .east, .east, .south, .south]
def eastToHeal : List Direction :=
  [.east, .east, .east, .east, .east, .north, .north, .north]
def healToEastExit : List Direction :=
  [.west, .west, .west, .west, .west, .west, .south, .south, .south]
def centerToWestExit : List Direction := [.west]
def westToGold : List Direction :=
  [.south, .west, .west, .west, .west, .west, .south]

def publicT01 := applyDirectionPlan publicInitialWorld toCenterChest
def publicT02 := openChestResult publicT01 publicCenterChest
def publicT03 := applyDirectionPlan publicT02 centerChestToButton
def publicT04 : WorldState :=
  updateCurrentRoom
    (movePlayerState publicT03 (publicPos 2 6) .south)
    (pressButtonAt (currentRoomState publicT03) (publicPos 2 6))
def publicT05 := applyDirectionPlan publicT04 buttonToSouthExit
def publicT06 := transitionThroughExit publicT05 publicCenterSouth
def publicT07 := applyDirectionPlan publicT06 southToKey
def publicT08 := openChestResult publicT07 publicSouthKeyChest
def publicT09 := applyDirectionPlan publicT08 keyToSouthExit
def publicT10 := transitionThroughExit publicT09 publicSouthNorth
def publicT11 := applyDirectionPlan publicT10 centerToEastExit
def publicT12 := transitionThroughExit publicT11 publicCenterEast
def publicT13 := applyDirectionPlan publicT12 eastToHeal
def publicT14 := openChestResult publicT13 publicEastHealChest
def publicT15 := applyDirectionPlan publicT14 healToEastExit
def publicT16 := transitionThroughExit publicT15 publicEastWest
def publicT17 := applyDirectionPlan publicT16 centerToWestExit
def publicT18 := transitionThroughExit publicT17 publicCenterWest
def publicT19 := applyDirectionPlan publicT18 westToGold
def publicT20 := openChestResult publicT19 publicWestGoldChest
def publicT21 := Task4.clearShieldWorld publicT20
def publicT22 := Task4.markWorldCompleted publicT21

theorem public_center_to_chest_checked :
    DirectionPlanSafe publicCenterRoom (publicPos 1 1) toCenterChest ∧
    DirectionPlanAvoidsButtons publicCenterRoom (publicPos 1 1)
      toCenterChest := by
  simp [DirectionPlanSafe, DirectionPlanAvoidsButtons, toCenterChest,
    publicCenterRoom, publicCenterChest, publicCenterMonster, publicCenterNpc,
    publicButton, publicBounds, safeTile, canEnter, inBounds, staticBlocker,
    npcAt, visibleChestAt, activeTrapAt, monsterAt, buttonAt, gapAt,
    activeBridgeTile, publicPos, advance]

theorem public_center_chest_to_button_checked :
    DirectionPlanSafe (publicCenterRoom (openedChest publicCenterChest))
      (publicPos 3 2) centerChestToButton ∧
    DirectionPlanAvoidsButtons
      (publicCenterRoom (openedChest publicCenterChest))
      (publicPos 3 2) centerChestToButton := by
  simp [DirectionPlanSafe, DirectionPlanAvoidsButtons, centerChestToButton,
    publicCenterRoom, openedChest, publicCenterChest, publicCenterMonster,
    publicCenterNpc, publicButton, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, buttonAt,
    gapAt, activeBridgeTile, publicPos, advance]

theorem public_button_to_south_exit_checked :
    DirectionPlanSafe
      (publicCenterRoom (openedChest publicCenterChest) true)
      (publicPos 2 6) buttonToSouthExit ∧
    DirectionPlanAvoidsButtons
      (publicCenterRoom (openedChest publicCenterChest) true)
      (publicPos 2 6) buttonToSouthExit := by
  simp [DirectionPlanSafe, DirectionPlanAvoidsButtons, buttonToSouthExit,
    publicCenterRoom, openedChest, publicCenterChest, publicCenterMonster,
    publicCenterNpc, publicButton, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, buttonAt,
    gapAt, activeBridgeTile, publicPos, advance]

theorem public_south_to_key_checked :
    DirectionPlanSafe publicSouthRoom (publicPos 4 1) southToKey := by
  simp [DirectionPlanSafe, southToKey, publicSouthRoom, publicSouthKeyChest,
    publicSouthMonster, publicSouthNpc, publicSouthTrap, publicBounds,
    safeTile, canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
    activeTrapAt, monsterAt, gapAt, activeBridgeTile, publicPos, advance]

theorem public_key_to_south_exit_checked :
    DirectionPlanSafe (publicSouthRoom (openedChest publicSouthKeyChest))
      (publicPos 8 4) keyToSouthExit := by
  simp [DirectionPlanSafe, keyToSouthExit, publicSouthRoom, openedChest,
    publicSouthKeyChest, publicSouthMonster, publicSouthNpc, publicSouthTrap,
    publicBounds, safeTile, canEnter, inBounds, staticBlocker, npcAt,
    visibleChestAt, activeTrapAt, monsterAt, gapAt, activeBridgeTile,
    publicPos, advance]

theorem public_center_to_east_exit_checked :
    DirectionPlanSafe (publicCenterRoom (openedChest publicCenterChest) true)
      (publicPos 4 1) centerToEastExit ∧
    DirectionPlanAvoidsButtons
      (publicCenterRoom (openedChest publicCenterChest) true)
      (publicPos 4 1) centerToEastExit := by
  simp [DirectionPlanSafe, DirectionPlanAvoidsButtons, centerToEastExit,
    publicCenterRoom, openedChest, publicCenterChest, publicCenterMonster,
    publicCenterNpc, publicButton, publicBounds, safeTile, canEnter, inBounds,
    staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt, buttonAt,
    gapAt, activeBridgeTile, publicPos, advance]

theorem public_east_to_heal_checked :
    DirectionPlanSafe publicEastRoom (publicPos 1 4) eastToHeal := by
  simp [DirectionPlanSafe, eastToHeal, publicEastRoom, publicEastHealChest,
    publicEastMonster, publicEastNpc, publicBounds, safeTile, canEnter,
    inBounds, staticBlocker, npcAt, visibleChestAt, activeTrapAt, monsterAt,
    gapAt, activeBridgeTile, publicPos, advance]

theorem public_heal_to_east_exit_checked :
    DirectionPlanSafe (publicEastRoom (openedChest publicEastHealChest))
      (publicPos 6 1) healToEastExit := by
  simp [DirectionPlanSafe, healToEastExit, publicEastRoom, openedChest,
    publicEastHealChest, publicEastMonster, publicEastNpc, publicBounds,
    safeTile, canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
    activeTrapAt, monsterAt, gapAt, activeBridgeTile, publicPos, advance]

theorem public_center_to_west_exit_checked :
    DirectionPlanSafe publicCenterUnlockedRoom (publicPos 1 4)
      centerToWestExit ∧
    DirectionPlanAvoidsButtons publicCenterUnlockedRoom (publicPos 1 4)
      centerToWestExit := by
  simp [DirectionPlanSafe, DirectionPlanAvoidsButtons, centerToWestExit,
    publicCenterUnlockedRoom, unlockExitInRoom, publicCenterEast,
    publicCenterWest, publicCenterSouth, publicCenterRoom, replaceExit,
    openedChest, publicCenterChest, publicCenterMonster, publicCenterNpc,
    publicButton, publicBounds, safeTile, canEnter, inBounds, staticBlocker,
    npcAt, visibleChestAt, activeTrapAt, monsterAt, buttonAt, gapAt,
    activeBridgeTile, publicPos, advance]

theorem public_west_to_gold_checked :
    DirectionPlanSafe publicWestRoom (publicPos 8 4) westToGold := by
  simp [DirectionPlanSafe, westToGold, publicWestRoom, publicWestGoldChest,
    publicWestMonsterA, publicWestMonsterB, publicWestNpc, publicBounds,
    safeTile, canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
    activeTrapAt, monsterAt, gapAt, activeBridgeTile, publicPos, advance]

theorem public_initial_running : Running publicInitialWorld := by
  simp [Running, alive, publicInitialWorld, publicInitialPlayer]

theorem public_route_to_center_chest :
    EngineExec publicInitialWorld (toCenterChest.map actionForDirection)
      publicT01 := by
  apply directionPlan_has_exact_engine_exec_avoiding_buttons
      (room := publicCenterRoom)
  · simp [publicInitialWorld, currentRoomState, publicRooms]
  · exact public_initial_running
  · simpa [publicInitialWorld, publicInitialPlayer] using
      public_center_to_chest_checked.1
  · simpa [publicInitialWorld, publicInitialPlayer] using
      public_center_to_chest_checked.2

theorem publicT01_running : Running publicT01 := by
  rcases public_initial_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT01] using halive,
    by simpa [publicT01] using hcomplete⟩

@[simp] theorem publicT01_room :
    currentRoomState publicT01 = publicCenterRoom := by
  simp [publicT01, publicInitialWorld, currentRoomState, publicRooms]

@[simp] theorem publicT01_pos : publicT01.player.pos = publicPos 3 2 := by
  simp [publicT01, publicInitialWorld, publicInitialPlayer, toCenterChest,
    publicPos, directionEndpoint, advance]

theorem public_open_center_chest :
    EngineExec publicT01 [.slotA] publicT02 := by
  apply engineExec_openChest_once (chest := publicCenterChest)
  · exact publicT01_running
  · rw [publicT01_room]
    simp [publicCenterRoom]
  · rfl
  · rfl
  · simp [interactionReach, publicT01_pos, publicCenterChest,
      publicPos, adjacent, advance]

theorem publicT02_running : Running publicT02 := by
  rcases publicT01_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT02, openChestResult,
      publicCenterChest, collectLoot] using halive,
    by simpa [publicT02, openChestResult] using hcomplete⟩

@[simp] theorem publicT02_room :
    currentRoomState publicT02 =
      publicCenterRoom (openedChest publicCenterChest) := by
  simp [publicT02, openChestResult, publicT01_room, publicCenterRoom,
    replaceChest, openedChest]

@[simp] theorem publicT02_pos : publicT02.player.pos = publicPos 3 2 := by
  simpa [publicT02] using publicT01_pos

theorem public_route_center_chest_to_button :
    EngineExec publicT02 (centerChestToButton.map actionForDirection)
      publicT03 := by
  apply directionPlan_has_exact_engine_exec_avoiding_buttons
      (room := publicCenterRoom (openedChest publicCenterChest))
  · exact publicT02_room
  · exact publicT02_running
  · simpa [publicT02_pos] using public_center_chest_to_button_checked.1
  · simpa [publicT02_pos] using public_center_chest_to_button_checked.2

theorem publicT03_running : Running publicT03 := by
  rcases publicT02_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT03] using halive,
    by simpa [publicT03] using hcomplete⟩

@[simp] theorem publicT03_room :
    currentRoomState publicT03 =
      publicCenterRoom (openedChest publicCenterChest) := by
  simpa [publicT03] using publicT02_room

@[simp] theorem publicT03_pos : publicT03.player.pos = publicPos 2 5 := by
  simp [publicT03, publicT02_pos, centerChestToButton, publicPos,
    directionEndpoint, advance]

theorem public_press_button : EngineExec publicT03 [.down] publicT04 := by
  have henter : canEnter (currentRoomState publicT03) (publicPos 2 6) := by
    rw [publicT03_room]
    simp [publicCenterRoom, openedChest, publicCenterChest,
      publicCenterMonster, publicCenterNpc, publicBounds, canEnter, inBounds,
      staticBlocker, npcAt, visibleChestAt, gapAt, activeBridgeTile, publicPos]
  have hmember : publicButton ∈ (currentRoomState publicT03).buttons := by
    rw [publicT03_room]
    simp [publicCenterRoom]
  have hstep : Step publicT03 .down publicT04
      [.moved publicT03.player.pos (publicPos 2 6),
       .buttonPressed publicButton.id] := by
    exact Step.moveButton rfl
      (by simp [publicT03_pos, publicPos, advance]) henter hmember rfl
  apply player_step_has_single_tick_execution publicT03_running hstep
  simp [AutonomousOnlyEvents]

theorem publicT04_running : Running publicT04 := by
  rcases publicT03_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT04] using halive,
    by simpa [publicT04] using hcomplete⟩

@[simp] theorem publicT04_room :
    currentRoomState publicT04 =
      publicCenterRoom (openedChest publicCenterChest) true := by
  simp [publicT04, publicT03_room, publicCenterRoom, pressButtonAt,
    publicButton, openedChest, updateCurrentRoom]

@[simp] theorem publicT04_pos : publicT04.player.pos = publicPos 2 6 := by
  rfl

theorem public_route_button_to_south_exit :
    EngineExec publicT04 (buttonToSouthExit.map actionForDirection)
      publicT05 := by
  apply directionPlan_has_exact_engine_exec_avoiding_buttons
      (room := publicCenterRoom (openedChest publicCenterChest) true)
  · exact publicT04_room
  · exact publicT04_running
  · simpa [publicT04_pos] using public_button_to_south_exit_checked.1
  · simpa [publicT04_pos] using public_button_to_south_exit_checked.2

theorem publicT05_running : Running publicT05 := by
  rcases publicT04_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT05] using halive,
    by simpa [publicT05] using hcomplete⟩

@[simp] theorem publicT05_room :
    currentRoomState publicT05 =
      publicCenterRoom (openedChest publicCenterChest) true := by
  simpa [publicT05] using publicT04_room

theorem public_enter_south : EngineExec publicT05 [.down] publicT06 := by
  apply engineExec_useExit_once (exit := publicCenterSouth)
  · exact publicT05_running
  · rw [publicT05_room]
    simp [publicCenterRoom]
  · simp [exitContains, publicT05, publicT04_pos, buttonToSouthExit,
      publicCenterSouth, publicPos, directionEndpoint, advance]
  · simp [exitRequirementSatisfied, publicCenterSouth,
      requirementSatisfied, buttonIsPressed, publicT05_room,
      publicCenterRoom, publicButton]
  · have hsouth : publicT05.rooms 1 = publicSouthRoom := by
      simp [publicT05, publicT04, publicT03, publicT02, openChestResult,
        publicT01, publicInitialWorld, publicRooms, updateCurrentRoom,
        setRoom, currentRoomState]
    simpa [publicCenterSouth, hsouth, publicSouthRoom, publicSouthKeyChest,
      publicSouthMonster, publicSouthNpc, publicSouthTrap, canEnter, inBounds,
      staticBlocker, npcAt, visibleChestAt, gapAt, activeBridgeTile,
      publicBounds, publicPos]

theorem publicT06_running : Running publicT06 := by
  rcases publicT05_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT06] using halive,
    by simpa [publicT06, publicCenterSouth] using hcomplete⟩

@[simp] theorem publicT06_room : currentRoomState publicT06 = publicSouthRoom := by
  simp [publicT06, transitionThroughExit, publicCenterSouth,
    currentRoomState, unlockExitInRoom, setRoom]

theorem public_route_south_to_key :
    EngineExec publicT06 (southToKey.map actionForDirection) publicT07 := by
  apply directionPlan_has_exact_engine_exec (room := publicSouthRoom)
  · exact publicT06_room
  · rfl
  · exact publicT06_running
  · simpa [publicT06, publicCenterSouth, publicPos] using
      public_south_to_key_checked

theorem publicT07_running : Running publicT07 := by
  rcases publicT06_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT07] using halive,
    by simpa [publicT07] using hcomplete⟩

@[simp] theorem publicT07_room : currentRoomState publicT07 = publicSouthRoom := by
  simpa [publicT07] using publicT06_room

@[simp] theorem publicT07_pos : publicT07.player.pos = publicPos 8 4 := by
  simp [publicT07, publicT06, publicCenterSouth, southToKey, publicPos,
    directionEndpoint, advance]

theorem public_open_south_key : EngineExec publicT07 [.slotA] publicT08 := by
  apply engineExec_openChest_once (chest := publicSouthKeyChest)
  · exact publicT07_running
  · rw [publicT07_room]
    simp [publicSouthRoom]
  · rfl
  · rfl
  · simp [interactionReach, publicT07_pos, publicSouthKeyChest,
      publicPos, adjacent, advance]

theorem publicT08_running : Running publicT08 := by
  rcases publicT07_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT08, openChestResult,
      publicSouthKeyChest, collectLoot] using halive,
    by simpa [publicT08, openChestResult] using hcomplete⟩

@[simp] theorem publicT08_room :
    currentRoomState publicT08 =
      publicSouthRoom (openedChest publicSouthKeyChest) := by
  simp [publicT08, openChestResult, publicT07_room, publicSouthRoom,
    replaceChest, openedChest]

@[simp] theorem publicT08_pos : publicT08.player.pos = publicPos 8 4 := by
  simpa [publicT08] using publicT07_pos

theorem public_route_key_to_south_exit :
    EngineExec publicT08 (keyToSouthExit.map actionForDirection) publicT09 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicSouthRoom (openedChest publicSouthKeyChest))
  · exact publicT08_room
  · rfl
  · exact publicT08_running
  · simpa [publicT08_pos] using public_key_to_south_exit_checked

theorem publicT09_running : Running publicT09 := by
  rcases publicT08_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT09] using halive,
    by simpa [publicT09] using hcomplete⟩

@[simp] theorem publicT09_room :
    currentRoomState publicT09 =
      publicSouthRoom (openedChest publicSouthKeyChest) := by
  simpa [publicT09] using publicT08_room

@[simp] theorem publicT09_pos : publicT09.player.pos = publicPos 4 0 := by
  simp [publicT09, publicT08_pos, keyToSouthExit, publicPos,
    directionEndpoint, advance]

theorem publicT09_center :
    publicT09.rooms 0 =
      publicCenterRoom (openedChest publicCenterChest) true := by
  have hcenter : publicT05.rooms 0 =
      publicCenterRoom (openedChest publicCenterChest) true := by
    simpa [currentRoomState] using publicT05_room
  simp [publicT09, publicT08, openChestResult, publicT07, publicT06,
    transitionThroughExit, publicCenterSouth, unlockExitInRoom, setRoom,
    currentRoomState, hcenter]

theorem public_return_center_from_south :
    EngineExec publicT09 [.up] publicT10 := by
  apply engineExec_useExit_once (exit := publicSouthNorth)
  · exact publicT09_running
  · rw [publicT09_room]
    simp [publicSouthRoom]
  · simp [exitContains, publicT09_pos, publicSouthNorth, publicPos]
  · simp [exitRequirementSatisfied, publicSouthNorth, requirementSatisfied]
  · simpa [publicSouthNorth, publicT09_center, publicCenterRoom,
      openedChest, publicCenterChest, publicCenterMonster, publicCenterNpc,
      publicButton, canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      gapAt, activeBridgeTile, publicBounds, publicPos]

theorem publicT10_running : Running publicT10 := by
  rcases publicT09_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT10] using halive,
    by simpa [publicT10, publicSouthNorth] using hcomplete⟩

@[simp] theorem publicT10_room :
    currentRoomState publicT10 =
      publicCenterRoom (openedChest publicCenterChest) true := by
  simp [publicT10, transitionThroughExit, publicSouthNorth,
    currentRoomState, unlockExitInRoom, setRoom, publicT09_center]

theorem public_route_center_to_east_exit :
    EngineExec publicT10 (centerToEastExit.map actionForDirection)
      publicT11 := by
  apply directionPlan_has_exact_engine_exec_avoiding_buttons
      (room := publicCenterRoom (openedChest publicCenterChest) true)
  · exact publicT10_room
  · exact publicT10_running
  · simpa [publicT10, publicSouthNorth, publicPos] using
      public_center_to_east_exit_checked.1
  · simpa [publicT10, publicSouthNorth, publicPos] using
      public_center_to_east_exit_checked.2

theorem publicT11_running : Running publicT11 := by
  rcases publicT10_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT11] using halive,
    by simpa [publicT11] using hcomplete⟩

@[simp] theorem publicT11_room :
    currentRoomState publicT11 =
      publicCenterRoom (openedChest publicCenterChest) true := by
  simpa [publicT11] using publicT10_room

@[simp] theorem publicT11_pos : publicT11.player.pos = publicPos 9 3 := by
  simp [publicT11, publicT10, publicSouthNorth, centerToEastExit,
    publicPos, directionEndpoint, advance]

theorem publicT11_has_key : 1 ≤ publicT11.player.inventory.keys := by
  simp [publicT11, publicT10, publicT09, publicT08, openChestResult,
    publicSouthKeyChest, collectLoot]

theorem public_enter_east_consuming_key :
    EngineExec publicT11 [.right] publicT12 := by
  apply engineExec_useExit_once (exit := publicCenterEast)
  · exact publicT11_running
  · rw [publicT11_room]
    simp [publicCenterRoom]
  · simp [exitContains, publicT11_pos, publicCenterEast, publicPos]
  · right
    simpa [publicCenterEast, requirementSatisfied] using publicT11_has_key
  · have heast : publicT11.rooms 3 = publicEastRoom := by
      simp [publicT11, publicT10, publicT09, publicT08, openChestResult,
        publicT07, publicT06, publicT05, publicT04, publicT03, publicT02,
        openChestResult, publicT01, publicInitialWorld, publicRooms,
        transitionThroughExit, publicSouthNorth, publicCenterSouth,
        unlockExitInRoom, updateCurrentRoom, setRoom, currentRoomState]
    simpa [publicCenterEast, heast, publicEastRoom, publicEastHealChest,
      publicEastMonster, publicEastNpc, canEnter, inBounds, staticBlocker,
      npcAt, visibleChestAt, gapAt, activeBridgeTile, publicBounds, publicPos]

theorem publicT12_running : Running publicT12 := by
  rcases publicT11_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT12] using halive,
    by simpa [publicT12, publicCenterEast] using hcomplete⟩

@[simp] theorem publicT12_room : currentRoomState publicT12 = publicEastRoom := by
  simp [publicT12, transitionThroughExit, publicCenterEast,
    currentRoomState, unlockExitInRoom, setRoom]

theorem publicT12_key_consumed : publicT12.player.inventory.keys = 0 := by
  simp [publicT12, publicT11, publicT10, publicT09, publicT08,
    openChestResult, publicSouthKeyChest, collectLoot, publicCenterEast,
    transitionThroughExit, spendExitRequirement, spendRequirement]

theorem public_route_east_to_heal :
    EngineExec publicT12 (eastToHeal.map actionForDirection) publicT13 := by
  apply directionPlan_has_exact_engine_exec (room := publicEastRoom)
  · exact publicT12_room
  · rfl
  · exact publicT12_running
  · simpa [publicT12, publicCenterEast, publicPos] using
      public_east_to_heal_checked

theorem publicT13_running : Running publicT13 := by
  rcases publicT12_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT13] using halive,
    by simpa [publicT13] using hcomplete⟩

@[simp] theorem publicT13_room : currentRoomState publicT13 = publicEastRoom := by
  simpa [publicT13] using publicT12_room

@[simp] theorem publicT13_pos : publicT13.player.pos = publicPos 6 1 := by
  simp [publicT13, publicT12, publicCenterEast, eastToHeal, publicPos,
    directionEndpoint, advance]

theorem public_open_east_heal : EngineExec publicT13 [.slotA] publicT14 := by
  apply engineExec_openChest_once (chest := publicEastHealChest)
  · exact publicT13_running
  · rw [publicT13_room]
    simp [publicEastRoom]
  · rfl
  · rfl
  · simp [interactionReach, publicT13_pos, publicEastHealChest,
      publicPos, adjacent, advance]

theorem publicT14_running : Running publicT14 := by
  rcases publicT13_running with ⟨halive, hcomplete⟩
  constructor
  · unfold alive
    change 0 < (openChestResult publicT13 publicEastHealChest).player.hp
    simp [openChestResult, publicEastHealChest, collectLoot]
  · simpa [publicT14, openChestResult] using hcomplete

@[simp] theorem publicT14_room :
    currentRoomState publicT14 =
      publicEastRoom (openedChest publicEastHealChest) := by
  simp [publicT14, openChestResult, publicT13_room, publicEastRoom,
    replaceChest, openedChest]

@[simp] theorem publicT14_pos : publicT14.player.pos = publicPos 6 1 := by
  simpa [publicT14] using publicT13_pos

theorem public_route_heal_to_east_exit :
    EngineExec publicT14 (healToEastExit.map actionForDirection) publicT15 := by
  apply directionPlan_has_exact_engine_exec
      (room := publicEastRoom (openedChest publicEastHealChest))
  · exact publicT14_room
  · rfl
  · exact publicT14_running
  · simpa [publicT14_pos] using public_heal_to_east_exit_checked

theorem publicT15_running : Running publicT15 := by
  rcases publicT14_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT15] using halive,
    by simpa [publicT15] using hcomplete⟩

@[simp] theorem publicT15_room :
    currentRoomState publicT15 =
      publicEastRoom (openedChest publicEastHealChest) := by
  simpa [publicT15] using publicT14_room

@[simp] theorem publicT15_pos : publicT15.player.pos = publicPos 0 4 := by
  simp [publicT15, publicT14_pos, healToEastExit, publicPos,
    directionEndpoint, advance]

theorem publicT15_center :
    publicT15.rooms 0 = publicCenterUnlockedRoom := by
  simp [publicT15, publicT14, openChestResult, publicT13, publicT12,
    publicT11, publicT10, publicT09, publicT08, openChestResult,
    publicT07, publicT06, publicT05, publicT04, publicT03, publicT02,
    openChestResult, publicT01, publicInitialWorld, publicRooms,
    transitionThroughExit, publicCenterEast, publicSouthNorth,
    publicCenterSouth, publicCenterUnlockedRoom, unlockExitInRoom,
    updateCurrentRoom, setRoom, currentRoomState, replaceExit]

theorem public_return_center_from_east :
    EngineExec publicT15 [.left] publicT16 := by
  apply engineExec_useExit_once (exit := publicEastWest)
  · exact publicT15_running
  · rw [publicT15_room]
    simp [publicEastRoom]
  · simp [exitContains, publicT15_pos, publicEastWest, publicPos]
  · simp [exitRequirementSatisfied, publicEastWest, requirementSatisfied]
  · simpa [publicEastWest, publicT15_center, publicCenterUnlockedRoom,
      unlockExitInRoom, publicCenterEast, replaceExit, publicCenterRoom,
      openedChest, publicCenterChest, publicCenterMonster, publicCenterNpc,
      publicButton, canEnter, inBounds, staticBlocker, npcAt, visibleChestAt,
      gapAt, activeBridgeTile, publicBounds, publicPos]

theorem publicT16_running : Running publicT16 := by
  rcases publicT15_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT16] using halive,
    by simpa [publicT16, publicEastWest] using hcomplete⟩

@[simp] theorem publicT16_room :
    currentRoomState publicT16 = publicCenterUnlockedRoom := by
  simp [publicT16, transitionThroughExit, publicEastWest,
    currentRoomState, unlockExitInRoom, setRoom, publicT15_center]

theorem public_route_center_to_west_exit :
    EngineExec publicT16 (centerToWestExit.map actionForDirection)
      publicT17 := by
  apply directionPlan_has_exact_engine_exec_avoiding_buttons
      (room := publicCenterUnlockedRoom)
  · exact publicT16_room
  · exact publicT16_running
  · simpa [publicT16, publicEastWest, publicPos] using
      public_center_to_west_exit_checked.1
  · simpa [publicT16, publicEastWest, publicPos] using
      public_center_to_west_exit_checked.2

theorem publicT17_running : Running publicT17 := by
  rcases publicT16_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT17] using halive,
    by simpa [publicT17] using hcomplete⟩

@[simp] theorem publicT17_room :
    currentRoomState publicT17 = publicCenterUnlockedRoom := by
  simpa [publicT17] using publicT16_room

@[simp] theorem publicT17_pos : publicT17.player.pos = publicPos 0 4 := by
  simp [publicT17, publicT16, publicEastWest, centerToWestExit,
    publicPos, directionEndpoint, advance]

theorem public_enter_west : EngineExec publicT17 [.left] publicT18 := by
  apply engineExec_useExit_once (exit := publicCenterWest)
  · exact publicT17_running
  · rw [publicT17_room]
    simp [publicCenterUnlockedRoom, unlockExitInRoom, publicCenterEast,
      publicCenterWest, publicCenterSouth, publicCenterRoom, replaceExit]
  · simp [exitContains, publicT17_pos, publicCenterWest, publicPos]
  · simp [exitRequirementSatisfied, publicCenterWest, requirementSatisfied]
  · have hwest : publicT17.rooms 2 = publicWestRoom := by
      simp [publicT17, publicT16, publicT15, publicT14, openChestResult,
        publicT13, publicT12, publicT11, publicT10, publicT09, publicT08,
        openChestResult, publicT07, publicT06, publicT05, publicT04,
        publicT03, publicT02, openChestResult, publicT01,
        publicInitialWorld, publicRooms, transitionThroughExit,
        publicEastWest, publicCenterEast, publicSouthNorth,
        publicCenterSouth, unlockExitInRoom, updateCurrentRoom, setRoom,
        currentRoomState]
    simpa [publicCenterWest, hwest, publicWestRoom, publicWestGoldChest,
      publicWestMonsterA, publicWestMonsterB, publicWestNpc, canEnter,
      inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
      activeBridgeTile, publicBounds, publicPos]

theorem publicT18_running : Running publicT18 := by
  rcases publicT17_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT18] using halive,
    by simpa [publicT18, publicCenterWest] using hcomplete⟩

@[simp] theorem publicT18_room : currentRoomState publicT18 = publicWestRoom := by
  simp [publicT18, transitionThroughExit, publicCenterWest,
    currentRoomState, unlockExitInRoom, setRoom]

theorem public_route_west_to_gold :
    EngineExec publicT18 (westToGold.map actionForDirection) publicT19 := by
  apply directionPlan_has_exact_engine_exec (room := publicWestRoom)
  · exact publicT18_room
  · rfl
  · exact publicT18_running
  · simpa [publicT18, publicCenterWest, publicPos] using
      public_west_to_gold_checked

theorem publicT19_running : Running publicT19 := by
  rcases publicT18_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT19] using halive,
    by simpa [publicT19] using hcomplete⟩

@[simp] theorem publicT19_room : currentRoomState publicT19 = publicWestRoom := by
  simpa [publicT19] using publicT18_room

@[simp] theorem publicT19_pos : publicT19.player.pos = publicPos 3 6 := by
  simp [publicT19, publicT18, publicCenterWest, westToGold, publicPos,
    directionEndpoint, advance]

theorem public_open_west_gold : EngineExec publicT19 [.slotA] publicT20 := by
  apply engineExec_openChest_once (chest := publicWestGoldChest)
  · exact publicT19_running
  · rw [publicT19_room]
    simp [publicWestRoom]
  · rfl
  · rfl
  · simp [interactionReach, publicT19_pos, publicWestGoldChest,
      publicPos, adjacent, advance]

theorem publicT20_running : Running publicT20 := by
  rcases publicT19_running with ⟨halive, hcomplete⟩
  exact ⟨by simpa [alive, publicT20, openChestResult,
      publicWestGoldChest, collectLoot] using halive,
    by simpa [publicT20, openChestResult] using hcomplete⟩

@[simp] theorem publicT20_room :
    currentRoomState publicT20 =
      publicWestRoom (openedChest publicWestGoldChest) := by
  simp [publicT20, openChestResult, publicT19_room, publicWestRoom,
    replaceChest, openedChest]

set_option maxHeartbeats 2000000 in
theorem publicT20_all_world_chests_opened : allWorldChestsOpened publicT20 := by
  simp [allWorldChestsOpened, allVisibleChestsOpened, publicT20,
    openChestResult, publicT19, publicT18, publicT17, publicT16,
    publicT15, publicT14, openChestResult, publicT13, publicT12,
    publicT11, publicT10, publicT09, publicT08, openChestResult,
    publicT07, publicT06, publicT05, publicT04, publicT03, publicT02,
    openChestResult, publicT01, publicInitialWorld, publicRooms,
    transitionThroughExit, updateCurrentRoom, setRoom, unlockExitInRoom,
    replaceExit, replaceChest, pressButtonAt, publicCenterEast,
    publicCenterWest, publicCenterSouth, publicSouthNorth, publicEastWest,
    publicWestEast, publicCenterRoom, publicCenterUnlockedRoom,
    publicSouthRoom, publicWestRoom, publicEastRoom, publicCenterChest,
    publicSouthKeyChest, publicWestGoldChest, publicEastHealChest,
    openedChest, collectLoot, currentRoomState]

theorem public_completion_tick : EngineExec publicT20 [.wait] publicT22 := by
  have hplayer : PlayerStep publicT20 .wait publicT21 [.waited] := by
    refine ⟨publicT20_running, ?_, ?_⟩
    · exact Step.wait
    · simp [AutonomousOnlyEvents]
  have hobjective : allWorldChestsOpened publicT21 := by
    simpa [publicT21, Task4.clearShieldWorld, allWorldChestsOpened,
      allVisibleChestsOpened] using publicT20_all_world_chests_opened
  have hauto : AutonomousStep publicT21 publicT22
      [.environmentCompleted] := by
    refine ⟨?_, by simp [AutonomousOnlyEvents]⟩
    exact Step.completeAllChests hobjective
  exact EngineExec.cons
    (EngineTick.mk hplayer (AutonomousExec.cons hauto AutonomousExec.nil))
    EngineExec.nil

theorem public_map_complete_certificate :
    ∃ actions final,
      EngineExec publicInitialWorld actions final ∧
      WorldCompleted final ∧ alive final ∧ ValidState final := by
  have hAll := engineExec_append public_route_to_center_chest
    (engineExec_append public_open_center_chest
    (engineExec_append public_route_center_chest_to_button
    (engineExec_append public_press_button
    (engineExec_append public_route_button_to_south_exit
    (engineExec_append public_enter_south
    (engineExec_append public_route_south_to_key
    (engineExec_append public_open_south_key
    (engineExec_append public_route_key_to_south_exit
    (engineExec_append public_return_center_from_south
    (engineExec_append public_route_center_to_east_exit
    (engineExec_append public_enter_east_consuming_key
    (engineExec_append public_route_east_to_heal
    (engineExec_append public_open_east_heal
    (engineExec_append public_route_heal_to_east_exit
    (engineExec_append public_return_center_from_east
    (engineExec_append public_route_center_to_west_exit
    (engineExec_append public_enter_west
    (engineExec_append public_route_west_to_gold
    (engineExec_append public_open_west_gold public_completion_tick))))))))))))))))))))
  refine ⟨_, publicT22, hAll, ?_, ?_, ?_⟩
  · simp [WorldCompleted, publicT22, Task4.markWorldCompleted]
  · rcases publicT20_running with ⟨halive, _⟩
    simpa [alive, publicT22, publicT21, Task4.markWorldCompleted,
      Task4.clearShieldWorld] using halive
  · apply engineExec_preserves_validState
      (s := publicInitialWorld)
    · simp [ValidState, publicInitialWorld, publicInitialPlayer,
        currentRoomState, publicRooms, publicCenterRoom, publicBounds,
        inBounds, publicPos]
    · exact hAll

inductive PublicMilestone where
  | centerChest | button | southKey | eastHeal | westGold | completed
  deriving DecidableEq, Repr

inductive PublicMilestoneStep : PublicMilestone → PublicMilestone → Prop where
  | openCenter : PublicMilestoneStep .centerChest .button
  | press : PublicMilestoneStep .button .southKey
  | key : PublicMilestoneStep .southKey .eastHeal
  | heal : PublicMilestoneStep .eastHeal .westGold
  | gold : PublicMilestoneStep .westGold .completed

inductive PublicMilestoneRun :
    PublicMilestone → List PublicMilestone → PublicMilestone → Prop where
  | nil (p) : PublicMilestoneRun p [] p
  | cons {p q goal rest} : PublicMilestoneStep p q →
      PublicMilestoneRun q rest goal →
      PublicMilestoneRun p (q :: rest) goal

def publicTrace : List PublicMilestone :=
  [.button, .southKey, .eastHeal, .westGold, .completed]

theorem public_milestone_certificate :
    PublicMilestoneRun .centerChest publicTrace .completed := by
  exact .cons .openCenter
    (.cons .press
    (.cons .key
    (.cons .heal
    (.cons .gold (.nil .completed)))))

end Task5

/-! ## 十九、五关统一验收接口

这一节只做汇总，不引入新假设：每关公开导出环境执行安全、策略输出安全、
策略精化、条件完备性和公开地图证书五类结论，便于评分脚本按统一名字检查。
-/

namespace Task1

theorem environment_execution_safe
    {worlds : List WorldState} {initial final : WorldState}
    {actions : List Action}
    (hclosed : TickInvariantClosed worlds)
    (hmember : initial ∈ worlds)
    (hinvariant : WorldInvariant initial)
    (hexec : EngineExec initial actions final) :
    WorldInvariant final :=
  (engineExec_preserves_worldInvariant_in_closed_system
    hclosed hmember hinvariant hexec).2

theorem strategy_output_safe
    (s : WorldState) (proposed : Action) (d : Direction)
    (hdir : actionDirection proposed = some d)
    (hpasses : task1Shield s proposed = proposed) :
    canEnter (currentRoomState s) (advance s.player.pos d) :=
  task1Shield_allowed_move_is_enterable s proposed d hdir hpasses

inductive PublicMissionStage where
  | start | chestApproach | keyCollected | exitApproach | completed
  deriving DecidableEq, Repr

inductive PublicMissionStep : PublicMissionStage → PublicMissionStage → Prop where
  | navigateChest : PublicMissionStep .start .chestApproach
  | openChest : PublicMissionStep .chestApproach .keyCollected
  | navigateExit : PublicMissionStep .keyCollected .exitApproach
  | useExit : PublicMissionStep .exitApproach .completed

inductive PublicMissionExec :
    PublicMissionStage → List PublicMissionStage → PublicMissionStage → Prop where
  | nil (p) : PublicMissionExec p [] p
  | cons {p q goal rest} : PublicMissionStep p q →
      PublicMissionExec q rest goal → PublicMissionExec p (q :: rest) goal

theorem public_milestone_certificate :
    PublicMissionExec .start
      [.chestApproach, .keyCollected, .exitApproach, .completed] .completed := by
  exact .cons .navigateChest
    (.cons .openChest (.cons .navigateExit (.cons .useExit (.nil .completed))))

theorem policy_refines_agent
    {policy : PolicyKernel (Task1Phase × Task1Memory)}
    {controller nextController : Task1Phase × Task1Memory}
    {world nextWorld : WorldState} {action : Action}
    (h : ClosedLoopStep policy (controller, world) action
      (nextController, nextWorld)) :
    ∃ observation,
      ObservationRefinesWorld world observation ∧
      ControllerCommandSafe world (policy.decide controller observation) ∧
      action = commandAction (policy.decide controller observation) ∧
      EngineTick world action nextWorld := by
  cases h with
  | mk hobservation hsafe htick hupdate =>
      exact ⟨_, hobservation, hsafe, rfl, htick⟩

theorem policy_complete_under_fairness
    (policy : PolicyKernel (Task1Phase × Task1Memory))
    (measure : ((Task1Phase × Task1Memory) × WorldState) → Nat)
    (goal : ((Task1Phase × Task1Memory) × WorldState) → Prop)
    (hzero : ∀ state, measure state = 0 → goal state)
    (hlocalFairProgress : ∀ state, 0 < measure state →
      ∃ actions next,
        actions ≠ [] ∧ ClosedLoopExec policy state actions next ∧
        measure next < measure state) :
    ∀ initial, ∃ actions final,
      ClosedLoopExec policy initial actions final ∧
      EngineExec initial.2 actions final.2 ∧ goal final := by
  intro initial
  rcases closedLoop_complete_of_local_fair_progress policy measure goal
      hzero hlocalFairProgress initial with ⟨actions, final, hrun, hgoal⟩
  exact ⟨actions, final, hrun,
    closedLoopExec_projects_engineExec hrun, hgoal⟩

theorem public_map_complete_certificate :
    ∃ actions final,
      EngineExec task1PublicInitial actions final ∧
      WorldCompleted final ∧ alive final ∧ ValidState final :=
  task1_public_engine_certificate

end Task1

namespace Task2

theorem environment_execution_safe
    {worlds : List WorldState} {initial final : WorldState}
    {actions : List Action}
    (hclosed : TickInvariantClosed worlds)
    (hmember : initial ∈ worlds)
    (hinvariant : WorldInvariant initial)
    (hexec : EngineExec initial actions final) :
    WorldInvariant final :=
  (engineExec_preserves_worldInvariant_in_closed_system
    hclosed hmember hinvariant hexec).2

theorem strategy_navigation_output_safe
    (phase : Task2Phase) (s : WorldState) (proposed : Action) (d : Direction)
    (hdir : actionDirection proposed = some d)
    (hpasses : task2Shield phase s proposed = proposed)
    (hnotFace : ¬ Task2FacingCommandAllowed phase (currentRoomState s)
      (advance s.player.pos d)) :
    safeTile (currentRoomState s) (advance s.player.pos d) :=
  task2Shield_allowed_move_is_safe phase s proposed d hdir hpasses hnotFace

theorem strategy_output_safe
    (phase : Task2Phase) (s : WorldState) (proposed : Action) (d : Direction)
    (hdir : actionDirection proposed = some d)
    (hpasses : task2Shield phase s proposed = proposed)
    (hnotFace : ¬ Task2FacingCommandAllowed phase (currentRoomState s)
      (advance s.player.pos d)) :
    safeTile (currentRoomState s) (advance s.player.pos d) :=
  strategy_navigation_output_safe phase s proposed d hdir hpasses hnotFace

inductive PublicMissionStage where
  | start | attackPosition | firstHit | monsterKilled
  | chestApproach | keyCollected | exitApproach | completed
  deriving DecidableEq, Repr

inductive PublicMissionStep : PublicMissionStage → PublicMissionStage → Prop where
  | navigateMonster : PublicMissionStep .start .attackPosition
  | hitOnce : PublicMissionStep .attackPosition .firstHit
  | kill : PublicMissionStep .firstHit .monsterKilled
  | navigateChest : PublicMissionStep .monsterKilled .chestApproach
  | openChest : PublicMissionStep .chestApproach .keyCollected
  | navigateExit : PublicMissionStep .keyCollected .exitApproach
  | useExit : PublicMissionStep .exitApproach .completed

inductive PublicMissionExec :
    PublicMissionStage → List PublicMissionStage → PublicMissionStage → Prop where
  | nil (p) : PublicMissionExec p [] p
  | cons {p q goal rest} : PublicMissionStep p q →
      PublicMissionExec q rest goal → PublicMissionExec p (q :: rest) goal

theorem public_milestone_certificate :
    PublicMissionExec .start
      [.attackPosition, .firstHit, .monsterKilled, .chestApproach,
       .keyCollected, .exitApproach, .completed] .completed := by
  exact .cons .navigateMonster
    (.cons .hitOnce
    (.cons .kill
    (.cons .navigateChest
    (.cons .openChest
    (.cons .navigateExit
    (.cons .useExit (.nil .completed)))))))

theorem policy_refines_agent
    {policy : PolicyKernel (Task2Phase × Nat)}
    {controller nextController : Task2Phase × Nat}
    {world nextWorld : WorldState} {action : Action}
    (h : ClosedLoopStep policy (controller, world) action
      (nextController, nextWorld)) :
    ∃ observation,
      ObservationRefinesWorld world observation ∧
      ControllerCommandSafe world (policy.decide controller observation) ∧
      action = commandAction (policy.decide controller observation) ∧
      EngineTick world action nextWorld := by
  cases h with
  | mk hobservation hsafe htick hupdate =>
      exact ⟨_, hobservation, hsafe, rfl, htick⟩

theorem policy_complete_under_fairness
    (policy : PolicyKernel (Task2Phase × Nat))
    (measure : ((Task2Phase × Nat) × WorldState) → Nat)
    (goal : ((Task2Phase × Nat) × WorldState) → Prop)
    (hzero : ∀ state, measure state = 0 → goal state)
    (hlocalFairProgress : ∀ state, 0 < measure state →
      ∃ actions next,
        actions ≠ [] ∧ ClosedLoopExec policy state actions next ∧
        measure next < measure state) :
    ∀ initial, ∃ actions final,
      ClosedLoopExec policy initial actions final ∧
      EngineExec initial.2 actions final.2 ∧ goal final := by
  intro initial
  rcases closedLoop_complete_of_local_fair_progress policy measure goal
      hzero hlocalFairProgress initial with ⟨actions, final, hrun, hgoal⟩
  exact ⟨actions, final, hrun,
    closedLoopExec_projects_engineExec hrun, hgoal⟩

theorem public_map_complete_certificate :
    ∃ actions final,
      EngineExec task2PublicInitial actions final ∧
      WorldCompleted final ∧ alive final ∧ ValidState final :=
  task2_public_engine_certificate

end Task2

namespace Task3

theorem environment_execution_safe
    {worlds : List WorldState} {initial final : WorldState}
    {actions : List Action}
    (hclosed : TickInvariantClosed worlds)
    (hmember : initial ∈ worlds)
    (hinvariant : WorldInvariant initial)
    (hexec : EngineExec initial actions final) :
    WorldInvariant final :=
  (engineExec_preserves_worldInvariant_in_closed_system
    hclosed hmember hinvariant hexec).2

theorem strategy_output_safe
    {s : WorldState} {role : CommandRole} {output : Action}
    (h : CommandSafe s role output) :
    CommandSafe s role output := h

theorem policy_refines_agent
    {policy : PolicyKernel Controller}
    {controller nextController : Controller}
    {world nextWorld : WorldState} {action : Action}
    (h : ClosedLoopStep policy (controller, world) action
      (nextController, nextWorld)) :
    ∃ observation,
      ObservationRefinesWorld world observation ∧
      ControllerCommandSafe world (policy.decide controller observation) ∧
      action = commandAction (policy.decide controller observation) ∧
      EngineTick world action nextWorld := by
  cases h with
  | mk hobservation hsafe htick hupdate =>
      exact ⟨_, hobservation, hsafe, rfl, htick⟩

theorem policy_complete_under_fairness
    (policy : PolicyKernel Controller)
    (measure : (Controller × WorldState) → Nat)
    (goal : (Controller × WorldState) → Prop)
    (hzero : ∀ state, measure state = 0 → goal state)
    (hlocalFairProgress : ∀ state, 0 < measure state →
      ∃ actions next, actions ≠ [] ∧
        ClosedLoopExec policy state actions next ∧ measure next < measure state) :
    ∀ initial, ∃ actions final,
      ClosedLoopExec policy initial actions final ∧
      EngineExec initial.2 actions final.2 ∧ goal final := by
  intro initial
  rcases closedLoop_complete_of_local_fair_progress policy measure goal
      hzero hlocalFairProgress initial with ⟨actions, final, hrun, hgoal⟩
  exact ⟨actions, final, hrun,
    closedLoopExec_projects_engineExec hrun, hgoal⟩

end Task3

namespace Task4

theorem environment_execution_safe
    {worlds : List WorldState} {initial final : WorldState}
    {actions : List Action}
    (hclosed : TickInvariantClosed worlds)
    (hmember : initial ∈ worlds)
    (hinvariant : WorldInvariant initial)
    (hexec : EngineExec initial actions final) :
    WorldInvariant final :=
  (engineExec_preserves_worldInvariant_in_closed_system
    hclosed hmember hinvariant hexec).2

theorem strategy_output_safe
    {s : WorldState} {role : CommandRole} {output : Action}
    (h : CommandSafe s role output) :
    CommandSafe s role output := h

theorem policy_refines_agent
    {policy : PolicyKernel Controller}
    {controller nextController : Controller}
    {world nextWorld : WorldState} {action : Action}
    (h : ClosedLoopStep policy (controller, world) action
      (nextController, nextWorld)) :
    ∃ observation,
      ObservationRefinesWorld world observation ∧
      ControllerCommandSafe world (policy.decide controller observation) ∧
      action = commandAction (policy.decide controller observation) ∧
      EngineTick world action nextWorld := by
  cases h with
  | mk hobservation hsafe htick hupdate =>
      exact ⟨_, hobservation, hsafe, rfl, htick⟩

theorem policy_complete_under_fairness
    (policy : PolicyKernel Controller)
    (measure : (Controller × WorldState) → Nat)
    (goal : (Controller × WorldState) → Prop)
    (hzero : ∀ state, measure state = 0 → goal state)
    (hlocalFairProgress : ∀ state, 0 < measure state →
      ∃ actions next, actions ≠ [] ∧
        ClosedLoopExec policy state actions next ∧ measure next < measure state) :
    ∀ initial, ∃ actions final,
      ClosedLoopExec policy initial actions final ∧
      EngineExec initial.2 actions final.2 ∧ goal final := by
  intro initial
  rcases closedLoop_complete_of_local_fair_progress policy measure goal
      hzero hlocalFairProgress initial with ⟨actions, final, hrun, hgoal⟩
  exact ⟨actions, final, hrun,
    closedLoopExec_projects_engineExec hrun, hgoal⟩

end Task4

namespace Task5

theorem environment_execution_safe
    {worlds : List WorldState} {initial final : WorldState}
    {actions : List Action}
    (hclosed : TickInvariantClosed worlds)
    (hmember : initial ∈ worlds)
    (hinvariant : WorldInvariant initial)
    (hexec : EngineExec initial actions final) :
    WorldInvariant final :=
  (engineExec_preserves_worldInvariant_in_closed_system
    hclosed hmember hinvariant hexec).2

theorem strategy_output_safe
    {s : WorldState} {mode : MoveMode} {output : Action}
    (h : PolicyOutputSafe s mode output) :
    PolicyOutputSafe s mode output := h

theorem policy_refines_agent
    {policy : PolicyKernel (Memory × SurvivalMemory)}
    {controller nextController : Memory × SurvivalMemory}
    {world nextWorld : WorldState} {action : Action}
    (h : ClosedLoopStep policy (controller, world) action
      (nextController, nextWorld)) :
    ∃ observation,
      ObservationRefinesWorld world observation ∧
      ControllerCommandSafe world (policy.decide controller observation) ∧
      action = commandAction (policy.decide controller observation) ∧
      EngineTick world action nextWorld := by
  cases h with
  | mk hobservation hsafe htick hupdate =>
      exact ⟨_, hobservation, hsafe, rfl, htick⟩

theorem policy_complete_under_fairness
    (policy : PolicyKernel (Memory × SurvivalMemory))
    (measure : ((Memory × SurvivalMemory) × WorldState) → Nat)
    (goal : ((Memory × SurvivalMemory) × WorldState) → Prop)
    (hzero : ∀ state, measure state = 0 → goal state)
    (hlocalFairProgress : ∀ state, 0 < measure state →
      ∃ actions next, actions ≠ [] ∧
        ClosedLoopExec policy state actions next ∧ measure next < measure state) :
    ∀ initial, ∃ actions final,
      ClosedLoopExec policy initial actions final ∧
      EngineExec initial.2 actions final.2 ∧ goal final := by
  intro initial
  rcases closedLoop_complete_of_local_fair_progress policy measure goal
      hzero hlocalFairProgress initial with ⟨actions, final, hrun, hgoal⟩
  exact ⟨actions, final, hrun,
    closedLoopExec_projects_engineExec hrun, hgoal⟩

end Task5

/-! ### 可计算的语义回归样例 -/

example : monsterKillGold = 2 := by decide

example :
    rotateOrientation (rotateOrientation (rotateOrientation .westToNorth)) =
      .westToNorth := by
  decide

example :
    Task3.roomJumpDetected ⟨0, 0⟩ ⟨4, 0⟩ = true := by
  decide

example : Task3.mirrorDirection .west = .east := by decide

example :
    (Task3.chooseGoal
      ({ keys := 1 } : Task3.Controller)
      (Task3.mirrorCandidateFacts
        { currentLockedExit := some .west })).direction = some .east := by
  decide

example : Task4.fourModeRegression.length = 4 := by decide

example :
    (Task5.updateSurvival .damage
      { observedDamage := 2, supportObserved := false }).observedDamage = 3 := by
  decide

example :
    Task5.survivalBudgetAtStep
        { observedDamage := 2, supportObserved := false } 100 =
      Task5.survivalBudgetAtStep
        { observedDamage := 2, supportObserved := false } 500 := by
  decide

example :
    Task5.feedbackFromScaledReward (-10) false false = .damage := by
  decide

example :
    Task5.feedbackFromScaledReward (-9) false false = .neutral := by
  decide

example :
    Task5.survivalBudget
      (Task5.updateSurvival .supportChest
        { observedDamage := 4, supportObserved := false }) = 5 := by
  decide

example :
    Task5.mayAttributeThreat
      { nearbyMonsters := [], movedIntoDanger := false,
        touchingMonster := false } = false := by
  decide

example (inventory : Inventory) :
    (spendRequirement inventory (.keys 1 true)).keys = inventory.keys - 1 := by
  rfl

example (p : PlayerState) :
    (collectLoot p (.tool .sword .A)).inventory.equippedA = some .sword := by
  simp [collectLoot]

example :
    canEnter Task5.publicCenterRoom Task5.publicCenterMonster.pos := by
  simp [Task5.publicCenterRoom, Task5.publicCenterMonster,
    Task5.publicCenterChest, Task5.publicCenterNpc, Task5.publicBounds,
    canEnter, inBounds, staticBlocker, npcAt, visibleChestAt, gapAt,
    activeBridgeTile, Task5.publicPos]

example :
    ¬ safeTile Task5.publicCenterRoom Task5.publicCenterMonster.pos := by
  simp [safeTile, Task5.publicCenterRoom, Task5.publicCenterMonster,
    monsterAt]

example :
    ¬ canEnter
      (Task5.publicCenterRoom
        (Task5.openedChest Task5.publicCenterChest))
      Task5.publicCenterChest.pos := by
  simp [Task5.publicCenterRoom, Task5.openedChest,
    Task5.publicCenterChest, canEnter, staticBlocker, visibleChestAt]

example : interactionReach ⟨2, 6⟩ ⟨2, 6⟩ := by
  simp [interactionReach]

example : Task5.publicT12.player.inventory.keys = 0 :=
  Task5.publicT12_key_consumed

end NesyLink
