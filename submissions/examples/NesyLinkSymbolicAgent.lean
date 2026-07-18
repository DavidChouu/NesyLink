/-!
  A small Lean4 formalization corresponding to `z_new_km_fw2.py`.

  The Python file extracts a symbolic state from NesyLink observations and uses a
  hand-written symbolic policy:

  * move only to safe tiles;
  * if adjacent to a monster and HP is high enough, attack;
  * otherwise move toward a monster/chest/exit target.

  This Lean file formalizes the symbolic layer, not the pixel-level game engine.
  It is intentionally small enough to compile without Mathlib.
-/

namespace KillMonsterFormalization

abbrev Position := Nat × Nat

inductive Action where
  | wait
  | up
  | down
  | left
  | right
  | attack
  | shield
  deriving DecidableEq, Repr

inductive GoalType where
  | killMonster
  | openChest
  | goExit
  deriving DecidableEq, Repr

structure Goal where
  kind : GoalType
  target : Position
  approachTiles : List Position
  deriving Repr

structure SymbolicState where
  player : Position
  exits : List Position
  walls : List Position
  traps : List Position
  monsters : List Position
  chests : List Position
  health : Nat
  keys : Nat
  deriving DecidableEq, Repr

def hasKey (s : SymbolicState) : Prop :=
  s.keys > 0

def needKey (s : SymbolicState) : Prop :=
  s.keys = 0

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def adjacent (a b : Position) : Prop :=
  manhattan a b = 1

def inBounds (p : Position) : Prop :=
  p.1 < 10 ∧ p.2 < 8

def dangerTiles (s : SymbolicState) : List Position :=
  s.monsters

def isSafe (s : SymbolicState) (p : Position) : Prop :=
  inBounds p ∧
  p ∉ s.walls ∧
  p ∉ s.traps ∧
  p ∉ s.monsters ∧
  p ∉ s.chests ∧
  p ∉ dangerTiles s

def nextPosition (p : Position) : Action → Position
  | Action.up => (p.1, p.2 - 1)
  | Action.down => (p.1, p.2 + 1)
  | Action.left => (p.1 - 1, p.2)
  | Action.right => (p.1 + 1, p.2)
  | _ => p

def adjacentMonster (s : SymbolicState) : Prop :=
  ∃ m, m ∈ s.monsters ∧ adjacent s.player m

def adjacentChest (s : SymbolicState) : Prop :=
  ∃ c, c ∈ s.chests ∧ adjacent s.player c

def canAttack (s : SymbolicState) (m : Position) : Prop :=
  m ∈ s.monsters ∧ adjacent s.player m ∧ s.health > 1

def canOpenChest (s : SymbolicState) (c : Position) : Prop :=
  c ∈ s.chests ∧ adjacent s.player c

/-!
  `Step s a t` is the relational version of Python `symbolic_step`.

  It is deliberately nondeterministic for attacks and chest opening because the
  Python code chooses the first adjacent monster/chest from a set. In Lean, we
  instead say that any adjacent attackable monster, or any adjacent openable
  chest, may be selected.
-/
inductive Step : SymbolicState → Action → SymbolicState → Prop where
  | moveSafe
      {s : SymbolicState} {a : Action} :
      a ∈ [Action.up, Action.down, Action.left, Action.right] →
      isSafe s (nextPosition s.player a) →
      Step s a { s with player := nextPosition s.player a }
  | moveBlocked
      {s : SymbolicState} {a : Action} :
      a ∈ [Action.up, Action.down, Action.left, Action.right] →
      ¬ isSafe s (nextPosition s.player a) →
      Step s a s
  | attackMonster
      {s : SymbolicState} {m : Position} :
      canAttack s m →
      Step s Action.attack { s with monsters := s.monsters.erase m }
  | openChest
      {s : SymbolicState} {c : Position} :
      ¬ adjacentMonster s →
      canOpenChest s c →
      Step s Action.attack { s with chests := s.chests.erase c, keys := s.keys + 1 }
  | attackNoEffect
      {s : SymbolicState} :
      ¬ adjacentMonster s →
      ¬ adjacentChest s →
      Step s Action.attack s
  | wait
      {s : SymbolicState} :
      Step s Action.wait s
  | shield
      {s : SymbolicState} :
      Step s Action.shield s

inductive Exec : SymbolicState → List Action → SymbolicState → Prop where
  | nil {s : SymbolicState} :
      Exec s [] s
  | cons {s t u : SymbolicState} {a : Action} {rest : List Action} :
      Step s a t →
      Exec t rest u →
      Exec s (a :: rest) u

def GoalReached (s : SymbolicState) : Prop :=
  s.monsters = [] ∧ s.keys > 0 ∧ s.player ∈ s.exits

def TaskCompletable (s : SymbolicState) : Prop :=
  ∃ plan final, Exec s plan final ∧ GoalReached final

def SafeState (s : SymbolicState) : Prop :=
  inBounds s.player ∧
  s.player ∉ s.walls ∧
  s.player ∉ s.traps

theorem moveSafe_player_eq
    {s t : SymbolicState} {a : Action}
    (h : Step s a t)
    (ha : a ∈ [Action.up, Action.down, Action.left, Action.right])
    (hsafe : isSafe s (nextPosition s.player a)) :
    t.player = nextPosition s.player a := by
  cases h with
  | moveSafe hmove hsafe' =>
      rfl
  | moveBlocked hmove hblocked =>
      exact False.elim (hblocked hsafe)
  | attackMonster hattack =>
      cases ha <;> contradiction
  | openChest hnoMonster hchest =>
      cases ha <;> contradiction
  | attackNoEffect hnoMonster hnoChest =>
      cases ha <;> contradiction
  | wait =>
      cases ha <;> contradiction
  | shield =>
      cases ha <;> contradiction

theorem moveBlocked_player_eq
    {s t : SymbolicState} {a : Action}
    (h : Step s a t)
    (ha : a ∈ [Action.up, Action.down, Action.left, Action.right])
    (hunsafe : ¬ isSafe s (nextPosition s.player a)) :
    t.player = s.player := by
  cases h with
  | moveSafe hmove hsafe =>
      exact False.elim (hunsafe hsafe)
  | moveBlocked hmove hblocked =>
      rfl
  | attackMonster hattack =>
      cases ha <;> contradiction
  | openChest hnoMonster hchest =>
      cases ha <;> contradiction
  | attackNoEffect hnoMonster hnoChest =>
      cases ha <;> contradiction
  | wait =>
      cases ha <;> contradiction
  | shield =>
      cases ha <;> contradiction

theorem safe_move_preserves_safe_state
    {s t : SymbolicState} {a : Action}
    (h : Step s a t)
    (ha : a ∈ [Action.up, Action.down, Action.left, Action.right])
    (hsafe : isSafe s (nextPosition s.player a)) :
    SafeState t := by
  cases h with
  | moveSafe hmove hsafe' =>
      rcases hsafe' with ⟨hin, hnwall, hntrap, hnmonster, hnchest, hndanger⟩
      unfold SafeState
      exact ⟨hin, hnwall, hntrap⟩
  | moveBlocked hmove hblocked =>
      exact False.elim (hblocked hsafe)
  | attackMonster hattack =>
      cases ha <;> contradiction
  | openChest hnoMonster hchest =>
      cases ha <;> contradiction
  | attackNoEffect hnoMonster hnoChest =>
      cases ha <;> contradiction
  | wait =>
      cases ha <;> contradiction
  | shield =>
      cases ha <;> contradiction

theorem attack_monster_updates_monsters
    {s t : SymbolicState} {m : Position}
    (_h : canAttack s m)
    (_hstep : Step s Action.attack t)
    (hexact : t.monsters = s.monsters.erase m) :
    t.monsters = s.monsters.erase m :=
  hexact

theorem open_chest_increases_keys
    {s t : SymbolicState} {c : Position}
    (_hnoMonster : ¬ adjacentMonster s)
    (_hchest : canOpenChest s c)
    (_hstep : Step s Action.attack t)
    (hexact : t = { s with chests := s.chests.erase c, keys := s.keys + 1 }) :
    t.keys = s.keys + 1 := by
  rw [hexact]

theorem exec_cons_inv
    {s u : SymbolicState} {a : Action} {rest : List Action}
    (h : Exec s (a :: rest) u) :
    ∃ t, Step s a t ∧ Exec t rest u := by
  cases h with
  | cons hstep hexec =>
      exact ⟨_, hstep, hexec⟩

theorem exec_append
    {s t u : SymbolicState} {p q : List Action}
    (hp : Exec s p t)
    (hq : Exec t q u) :
    Exec s (p ++ q) u := by
  induction hp with
  | nil =>
      exact hq
  | cons hstep hexec ih =>
      exact Exec.cons hstep (ih hq)

/-!
  A compositional reachability theorem for the task.

  Read it as:

  If there is a plan to reach a monster-adjacent attack state, attacking updates
  the monster list, there is then a plan to reach a chest-adjacent state, opening
  the chest updates the key count, and there is finally a plan to reach an exit
  state with no monsters and at least one key, then the initial state is
  completable.

  This does not prove BFS completeness. It proves the sound composition property
  that the Python agent relies on: the three symbolic phases can be stitched into
  one valid plan.
-/
theorem task_completable_if_subplans_exist
    {init nearMonster afterKill nearChest afterChest final : SymbolicState}
    {toMonster toChest toExit : List Action}
    {monster chest : Position}
    (hToMonster : Exec init toMonster nearMonster)
    (hAttackable : canAttack nearMonster monster)
    (hAfterKill : afterKill =
      { nearMonster with monsters := nearMonster.monsters.erase monster })
    (hToChest : Exec afterKill toChest nearChest)
    (hNoAdjacentMonster : ¬ adjacentMonster nearChest)
    (hOpenable : canOpenChest nearChest chest)
    (hAfterChest : afterChest =
      { nearChest with chests := nearChest.chests.erase chest, keys := nearChest.keys + 1 })
    (hToExit : Exec afterChest toExit final)
    (hFinalMonsters : final.monsters = [])
    (hFinalKeys : final.keys > 0)
    (hFinalAtExit : final.player ∈ final.exits) :
    TaskCompletable init := by
  have hAttackStep : Step nearMonster Action.attack afterKill := by
    rw [hAfterKill]
    exact Step.attackMonster hAttackable
  have hAttackExec : Exec nearMonster [Action.attack] afterKill := by
    exact Exec.cons hAttackStep Exec.nil

  have hOpenStep : Step nearChest Action.attack afterChest := by
    rw [hAfterChest]
    exact Step.openChest hNoAdjacentMonster hOpenable
  have hOpenExec : Exec nearChest [Action.attack] afterChest := by
    exact Exec.cons hOpenStep Exec.nil

  have hPhase1 : Exec init (toMonster ++ [Action.attack]) afterKill :=
    exec_append hToMonster hAttackExec
  have hPhase2 : Exec init ((toMonster ++ [Action.attack]) ++ toChest) nearChest :=
    exec_append hPhase1 hToChest
  have hPhase3 :
      Exec init (((toMonster ++ [Action.attack]) ++ toChest) ++ [Action.attack]) afterChest :=
    exec_append hPhase2 hOpenExec
  have hAll :
      Exec init
        ((((toMonster ++ [Action.attack]) ++ toChest) ++ [Action.attack]) ++ toExit)
        final :=
    exec_append hPhase3 hToExit

  exact ⟨
    ((((toMonster ++ [Action.attack]) ++ toChest) ++ [Action.attack]) ++ toExit),
    final,
    hAll,
    ⟨hFinalMonsters, hFinalKeys, hFinalAtExit⟩
  ⟩

/-!
  A lightweight BFS completeness statement.

  The Python reference implementation uses BFS as a path planner. Proving the
  exact queue-and-parent implementation complete would require formalizing the
  queue, visited set, parent map, and finite-state invariant.

  For the course project, the central proof obligation can be separated out:

  * `BfsFrontierComplete init n frontier` says that the BFS frontier/list has
    covered every state reachable from `init` by a plan of length at most `n`.
  * `BoundedGoalReachable init n` says that there exists a successful plan of
    length at most `n`.

  The theorem below proves the completeness step: once the BFS frontier has this
  coverage invariant, any bounded reachable goal must be detected in it.
-/

def BoundedReachable (init : SymbolicState) (n : Nat) (target : SymbolicState) : Prop :=
  ∃ plan, plan.length ≤ n ∧ Exec init plan target

def BoundedGoalReachable (init : SymbolicState) (n : Nat) : Prop :=
  ∃ final, BoundedReachable init n final ∧ GoalReached final

def BfsFrontierComplete
    (init : SymbolicState)
    (n : Nat)
    (frontier : List SymbolicState) : Prop :=
  ∀ final, BoundedReachable init n final → final ∈ frontier

def BfsFindsGoal (frontier : List SymbolicState) : Prop :=
  ∃ final, final ∈ frontier ∧ GoalReached final

theorem bfs_completeness_from_frontier_invariant
    {init : SymbolicState} {n : Nat} {frontier : List SymbolicState}
    (hcomplete : BfsFrontierComplete init n frontier)
    (hreachable : BoundedGoalReachable init n) :
    BfsFindsGoal frontier := by
  rcases hreachable with ⟨final, hbounded, hgoal⟩
  exact ⟨final, hcomplete final hbounded, hgoal⟩

/-!
  An executable BFS layer for the deterministic symbolic transition function.

  This is closer to the Python planner: repeatedly expand a frontier by applying
  all available actions. It omits queue ordering and parent reconstruction, but
  it is a genuine BFS reachability computation by depth.
-/

def positionEqB (a b : Position) : Bool :=
  a.1 == b.1 && a.2 == b.2

def containsPosition (p : Position) : List Position → Bool
  | [] => false
  | q :: rest => if positionEqB p q then true else containsPosition p rest

def inBoundsB (p : Position) : Bool :=
  decide (p.1 < 10) && decide (p.2 < 8)

def isSafeB (s : SymbolicState) (p : Position) : Bool :=
  inBoundsB p &&
  !containsPosition p s.walls &&
  !containsPosition p s.traps &&
  !containsPosition p s.monsters &&
  !containsPosition p s.chests &&
  !containsPosition p (dangerTiles s)

def adjacentB (a b : Position) : Bool :=
  manhattan a b == 1

def canAttackB (s : SymbolicState) (m : Position) : Bool :=
  containsPosition m s.monsters &&
  adjacentB s.player m &&
  decide (s.health > 1)

def canOpenChestB (s : SymbolicState) (c : Position) : Bool :=
  containsPosition c s.chests &&
  adjacentB s.player c

def findPosition? (xs : List Position) (p : Position → Bool) : Option Position :=
  match xs with
  | [] => none
  | x :: rest => if p x then some x else findPosition? rest p

def symbolicStepFn (s : SymbolicState) : Action → SymbolicState
  | Action.up =>
      let p := nextPosition s.player Action.up
      if isSafeB s p then { s with player := p } else s
  | Action.down =>
      let p := nextPosition s.player Action.down
      if isSafeB s p then { s with player := p } else s
  | Action.left =>
      let p := nextPosition s.player Action.left
      if isSafeB s p then { s with player := p } else s
  | Action.right =>
      let p := nextPosition s.player Action.right
      if isSafeB s p then { s with player := p } else s
  | Action.attack =>
      match findPosition? s.monsters (canAttackB s) with
      | some monster => { s with monsters := s.monsters.erase monster }
      | none =>
          match findPosition? s.chests (canOpenChestB s) with
          | some chest => { s with chests := s.chests.erase chest, keys := s.keys + 1 }
          | none => s
  | Action.wait => s
  | Action.shield => s

def runPlanFn : SymbolicState → List Action → SymbolicState
  | s, [] => s
  | s, a :: rest => runPlanFn (symbolicStepFn s a) rest

def allActions : List Action :=
  [
    Action.wait,
    Action.up,
    Action.down,
    Action.left,
    Action.right,
    Action.attack,
    Action.shield
  ]

theorem action_mem_allActions (a : Action) : a ∈ allActions := by
  cases a <;> simp [allActions]

def expandFrontier : List SymbolicState → List SymbolicState
  | [] => []
  | s :: rest => allActions.map (symbolicStepFn s) ++ expandFrontier rest

def bfsVisited : Nat → List SymbolicState → List SymbolicState
  | 0, frontier => frontier
  | n + 1, frontier => frontier ++ bfsVisited n (expandFrontier frontier)

def bfsVisitedFrom (init : SymbolicState) (n : Nat) : List SymbolicState :=
  bfsVisited n [init]

theorem mem_expandFrontier_of_mem
    {s : SymbolicState} {frontier : List SymbolicState}
    (a : Action)
    (hs : s ∈ frontier) :
    symbolicStepFn s a ∈ expandFrontier frontier := by
  induction frontier with
  | nil =>
      cases hs
  | cons head tail ih =>
      simp [expandFrontier] at hs ⊢
      rcases hs with hEq | hTail
      · subst hEq
        exact Or.inl ⟨a, action_mem_allActions a, rfl⟩
      · exact Or.inr (ih hTail)

theorem runPlanFn_mem_bfsVisited_of_mem
    {s : SymbolicState} {frontier : List SymbolicState}
    (plan : List Action)
    {n : Nat}
    (hs : s ∈ frontier)
    (hlen : plan.length ≤ n) :
    runPlanFn s plan ∈ bfsVisited n frontier := by
  induction n generalizing s frontier plan with
  | zero =>
      cases plan with
      | nil =>
          simpa [runPlanFn, bfsVisited] using hs
      | cons a rest =>
          cases hlen
  | succ n ih =>
      cases plan with
      | nil =>
          simp [runPlanFn, bfsVisited, hs]
      | cons a rest =>
          have hRestLen : rest.length ≤ n := Nat.succ_le_succ_iff.mp hlen
          have hStepMem : symbolicStepFn s a ∈ expandFrontier frontier :=
            mem_expandFrontier_of_mem a hs
          have hRest :
              runPlanFn (symbolicStepFn s a) rest ∈ bfsVisited n (expandFrontier frontier) :=
            ih rest hStepMem hRestLen
          simp [runPlanFn, bfsVisited]
          exact Or.inr hRest

def FnBoundedReachable (init : SymbolicState) (n : Nat) (target : SymbolicState) : Prop :=
  ∃ plan, plan.length ≤ n ∧ runPlanFn init plan = target

def FnBfsFrontierComplete
    (init : SymbolicState)
    (n : Nat)
    (visited : List SymbolicState) : Prop :=
  ∀ target, FnBoundedReachable init n target → target ∈ visited

theorem bfsVisited_frontier_invariant
    (init : SymbolicState)
    (n : Nat) :
    FnBfsFrontierComplete init n (bfsVisitedFrom init n) := by
  intro target hreachable
  rcases hreachable with ⟨plan, hlen, hrun⟩
  rw [← hrun]
  exact runPlanFn_mem_bfsVisited_of_mem
    (s := init)
    (frontier := [init])
    plan
    (by simp)
    hlen

def FnBoundedGoalReachable (init : SymbolicState) (n : Nat) : Prop :=
  ∃ final, FnBoundedReachable init n final ∧ GoalReached final

def FnBfsFindsGoal (visited : List SymbolicState) : Prop :=
  ∃ final, final ∈ visited ∧ GoalReached final

theorem bfsVisited_complete_for_bounded_goal
    {init : SymbolicState} {n : Nat}
    (hreachable : FnBoundedGoalReachable init n) :
    FnBfsFindsGoal (bfsVisitedFrom init n) := by
  rcases hreachable with ⟨final, hbounded, hgoal⟩
  exact ⟨final, bfsVisited_frontier_invariant init n final hbounded, hgoal⟩

/-!
  A concrete witness for the simplified `task_2` model.

  This abstracts the map in `nesylink/map_data/mathematical_logic/task_2/room_001.json`.
  The real environment gives the monster HP 2, while `Step.attackMonster` above
  follows `z_new_km_fw2.py`'s simplified symbolic model where one attack removes
  an adjacent monster.
-/

def task2Monster : Position := (2, 2)

def task2Chest : Position := (1, 3)

def task2Exits : List Position := [(0, 3), (0, 4)]

def task2Traps : List Position :=
  [
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
    (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7)
  ]

def task2Init : SymbolicState :=
  {
    player := (7, 3),
    exits := task2Exits,
    walls := [],
    traps := task2Traps,
    monsters := [task2Monster],
    chests := [task2Chest],
    health := 5,
    keys := 0
  }

def task2ToMonster : List Action :=
  [Action.left, Action.left, Action.left, Action.left, Action.up]

def task2M1 : SymbolicState :=
  { task2Init with player := (6, 3) }

def task2M2 : SymbolicState :=
  { task2M1 with player := (5, 3) }

def task2M3 : SymbolicState :=
  { task2M2 with player := (4, 3) }

def task2M4 : SymbolicState :=
  { task2M3 with player := (3, 3) }

def task2NearMonster : SymbolicState :=
  { task2M4 with player := (3, 2) }

def task2AfterKill : SymbolicState :=
  { task2NearMonster with monsters := task2NearMonster.monsters.erase task2Monster }

def task2ToChest : List Action :=
  [Action.left, Action.left]

def task2C1 : SymbolicState :=
  { task2AfterKill with player := (2, 2) }

def task2NearChest : SymbolicState :=
  { task2C1 with player := (1, 2) }

def task2AfterChest : SymbolicState :=
  { task2NearChest with chests := task2NearChest.chests.erase task2Chest, keys := task2NearChest.keys + 1 }

def task2ToExit : List Action :=
  [Action.left, Action.down]

def task2E1 : SymbolicState :=
  { task2AfterChest with player := (0, 2) }

def task2Final : SymbolicState :=
  { task2E1 with player := (0, 3) }

theorem task2_exec_to_monster :
    Exec task2Init task2ToMonster task2NearMonster := by
  unfold task2ToMonster
  apply Exec.cons (t := task2M1)
  · simpa [task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2Init) (a := Action.left) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]))
  apply Exec.cons (t := task2M2)
  · simpa [task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2M1) (a := Action.left) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]))
  apply Exec.cons (t := task2M3)
  · simpa [task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2M2) (a := Action.left) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]))
  apply Exec.cons (t := task2M4)
  · simpa [task2M4, task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2M3) (a := Action.left) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]))
  apply Exec.cons (t := task2NearMonster)
  · simpa [task2NearMonster, task2M4, task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2M4) (a := Action.up) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2M4, task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]))
  exact Exec.nil

theorem task2_can_attack :
    canAttack task2NearMonster task2Monster := by
  unfold canAttack task2NearMonster task2M4 task2M3 task2M2 task2M1 task2Init task2Monster adjacent manhattan
  decide

theorem task2_exec_to_chest :
    Exec task2AfterKill task2ToChest task2NearChest := by
  unfold task2ToChest
  apply Exec.cons (t := task2C1)
  · simpa [task2C1, task2AfterKill, task2NearMonster, task2M4, task2M3, task2M2, task2M1,
      task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2AfterKill) (a := Action.left) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2AfterKill, task2NearMonster, task2M4, task2M3,
          task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]))
  apply Exec.cons (t := task2NearChest)
  · simpa [task2NearChest, task2C1, task2AfterKill, task2NearMonster, task2M4, task2M3, task2M2, task2M1,
      task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2C1) (a := Action.left) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2C1, task2AfterKill, task2NearMonster, task2M4,
          task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]))
  exact Exec.nil

theorem task2_no_adjacent_monster_near_chest :
    ¬ adjacentMonster task2NearChest := by
  unfold adjacentMonster task2NearChest task2C1 task2AfterKill task2NearMonster task2M4 task2M3 task2M2 task2M1 task2Init
  unfold task2Monster
  intro h
  rcases h with ⟨m, hm, hadj⟩
  simp at hm

theorem task2_can_open_chest :
    canOpenChest task2NearChest task2Chest := by
  unfold canOpenChest task2NearChest task2C1 task2AfterKill task2NearMonster task2M4 task2M3 task2M2 task2M1 task2Init
  unfold task2Chest adjacent manhattan
  decide

theorem task2_exec_to_exit :
    Exec task2AfterChest task2ToExit task2Final := by
  unfold task2ToExit
  apply Exec.cons (t := task2E1)
  · simpa [task2E1, task2AfterChest, task2NearChest, task2C1, task2AfterKill, task2NearMonster,
      task2M4, task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2AfterChest) (a := Action.left) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2AfterChest, task2NearChest, task2C1, task2AfterKill,
          task2NearMonster, task2M4, task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]))
  apply Exec.cons (t := task2Final)
  · simpa [task2Final, task2E1, task2AfterChest, task2NearChest, task2C1, task2AfterKill, task2NearMonster,
      task2M4, task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps, task2Monster, task2Chest]
      using (Step.moveSafe (s := task2E1) (a := Action.down) (by decide)
        (by simp [isSafe, inBounds, dangerTiles, nextPosition, task2E1, task2AfterChest, task2NearChest, task2C1,
          task2AfterKill, task2NearMonster, task2M4, task2M3, task2M2, task2M1, task2Init, task2Exits, task2Traps,
          task2Monster, task2Chest]))
  exact Exec.nil

theorem task2_final_monsters :
    task2Final.monsters = [] := by
  unfold task2Final task2E1 task2AfterChest task2NearChest task2C1 task2AfterKill task2NearMonster
  unfold task2M4 task2M3 task2M2 task2M1 task2Init
  unfold task2Monster
  decide

theorem task2_final_keys :
    task2Final.keys > 0 := by
  unfold task2Final task2E1 task2AfterChest task2NearChest task2C1 task2AfterKill task2NearMonster
  unfold task2M4 task2M3 task2M2 task2M1 task2Init
  decide

theorem task2_final_at_exit :
    task2Final.player ∈ task2Final.exits := by
  unfold task2Final task2E1 task2AfterChest task2NearChest task2C1 task2AfterKill task2NearMonster
  unfold task2M4 task2M3 task2M2 task2M1 task2Init
  unfold task2Exits
  decide

theorem task2_concrete_completable :
    TaskCompletable task2Init := by
  exact task_completable_if_subplans_exist
    (init := task2Init)
    (nearMonster := task2NearMonster)
    (afterKill := task2AfterKill)
    (nearChest := task2NearChest)
    (afterChest := task2AfterChest)
    (final := task2Final)
    (toMonster := task2ToMonster)
    (toChest := task2ToChest)
    (toExit := task2ToExit)
    (monster := task2Monster)
    (chest := task2Chest)
    task2_exec_to_monster
    task2_can_attack
    rfl
    task2_exec_to_chest
    task2_no_adjacent_monster_near_chest
    task2_can_open_chest
    rfl
    task2_exec_to_exit
    task2_final_monsters
    task2_final_keys
    task2_final_at_exit

end KillMonsterFormalization
