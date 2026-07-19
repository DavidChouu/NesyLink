/-!
  A small, self-contained formalization of the graph model used in the BFS
  document.

  The file deliberately separates three concerns:

  * `GameGraph` is the symbolic game model. A state contains all information
    needed by the transition relation; pixels and CNN errors are outside this
    layer.
  * `GraphPath` is the legal path relation. `legalTransitions` removes edges
    whose source is not expandable, so dead states are not expanded.
  * `bfsRoutesExact` is the mathematical layer specification of BFS. It is
    intentionally a route specification; a queue/visited implementation can
    later be proved equivalent to it.

  All theorems in this file are checked by the Lean kernel without Mathlib.
-/

namespace NesyLinkBFS

/-! ## 1. The finite labelled graph interface -/

structure GameGraph (State Action : Type) where
  transitions : State -> List (Action × State)
  expandable : State -> Bool
  goal : State -> Prop
  start : State

def dead (g : GameGraph State Action) (s : State) : Prop :=
  g.expandable s = false

def legalTransitions
    (g : GameGraph State Action) (s : State) : List (Action × State) :=
  if g.expandable s then g.transitions s else []

theorem legal_edge_source_expandable
    {State Action : Type} {g : GameGraph State Action}
    {s : State} {edge : Action × State}
    (h : edge ∈ legalTransitions g s) :
    g.expandable s = true := by
  by_cases hs : g.expandable s = true
  · exact hs
  · simp [legalTransitions, hs] at h

/-! A path stores the labelled edges after its initial state. -/

inductive GraphPath
    {State Action : Type}
    (g : GameGraph State Action) :
    State -> List (Action × State) -> State -> Prop where
  | nil (s : State) :
      GraphPath g s [] s
  | cons {s goal : State}
      {rest : List (Action × State)}
      (edge : Action × State)
      (hedge : edge ∈ legalTransitions g s)
      (htail : GraphPath g edge.2 rest goal) :
      GraphPath g s (edge :: rest) goal

def routeEnd
    {State Action : Type} :
    State -> List (Action × State) -> State
  | current, [] => current
  | _current, (_action, next) :: rest => routeEnd next rest

def reachable
    {State Action : Type}
    (g : GameGraph State Action) (source target : State) : Prop :=
  ∃ route, GraphPath g source route target

def goalReachable
    {State Action : Type} (g : GameGraph State Action) : Prop :=
  ∃ goal route,
    g.goal goal ∧ GraphPath g g.start route goal

theorem graph_path_endpoint
    {State Action : Type} {g : GameGraph State Action}
    {start goal : State} {route : List (Action × State)}
    (hpath : GraphPath g start route goal) :
    routeEnd start route = goal := by
  induction hpath with
  | nil s => rfl
  | cons edge hedge htail ih =>
      simpa [routeEnd] using ih

theorem reachable_refl
    {State Action : Type} {g : GameGraph State Action}
    (s : State) : reachable g s s := by
  exact ⟨[], GraphPath.nil s⟩

theorem reachable_trans
    {State Action : Type} {g : GameGraph State Action}
    {a b c : State}
    (hab : reachable g a b)
    (hbc : reachable g b c) :
    reachable g a c := by
  rcases hab with ⟨left, hleft⟩
  rcases hbc with ⟨right, hright⟩
  refine ⟨left ++ right, ?_⟩
  induction hleft with
  | nil s =>
      simpa using hright
  | cons edge hedge htail ih =>
      exact GraphPath.cons edge hedge (ih hright)

theorem graph_path_append
    {State Action : Type} {g : GameGraph State Action}
    {a b c : State}
    {left right : List (Action × State)}
    (hleft : GraphPath g a left b)
    (hright : GraphPath g b right c) :
    GraphPath g a (left ++ right) c := by
  induction hleft with
  | nil s =>
      simpa using hright
  | cons edge hedge htail ih =>
      exact GraphPath.cons edge hedge (ih hright)

/-! ## 2. Exact BFS layers -/

def bfsRoutesExact
    {State Action : Type}
    (g : GameGraph State Action) :
    State -> Nat -> List (List (Action × State))
  | _start, 0 => [[]]
  | start, n + 1 =>
      List.flatMap
        (fun edge =>
          (bfsRoutesExact g edge.2 n).map
            (fun route => edge :: route))
        (legalTransitions g start)

def bfsFindsGoalAtDepth
    {State Action : Type}
    (g : GameGraph State Action)
    (start : State) (depth : Nat) : Prop :=
  ∃ route goal,
    route ∈ bfsRoutesExact g start depth ∧
    g.goal goal ∧
    routeEnd start route = goal

def bfsFindsGoalWithin
    {State Action : Type}
    (g : GameGraph State Action)
    (start : State) (maxDepth : Nat) : Prop :=
  ∃ depth,
    depth ≤ maxDepth ∧ bfsFindsGoalAtDepth g start depth

theorem graph_path_mem_bfs_routes_exact
    {State Action : Type}
    {g : GameGraph State Action}
    {start goal : State}
    {route : List (Action × State)}
    (hpath : GraphPath g start route goal) :
    route ∈ bfsRoutesExact g start route.length ∧
    routeEnd start route = goal := by
  induction hpath with
  | nil s =>
      simp [bfsRoutesExact, routeEnd]
  | cons edge hedge htail ih =>
      constructor
      · apply List.mem_flatMap.mpr
        exact ⟨edge, hedge, List.mem_map.mpr ⟨_, ih.1, rfl⟩⟩
      · simpa [routeEnd] using ih.2

theorem bfs_route_layer_sound
    {State Action : Type}
    {g : GameGraph State Action}
    {start : State} {depth : Nat}
    {route : List (Action × State)}
    (hmem : route ∈ bfsRoutesExact g start depth) :
    GraphPath g start route (routeEnd start route) := by
  induction depth generalizing start route with
  | zero =>
      simp [bfsRoutesExact] at hmem
      subst route
      exact GraphPath.nil start
  | succ n ih =>
      rw [bfsRoutesExact] at hmem
      rcases List.mem_flatMap.mp hmem with ⟨edge, hedge, htailMap⟩
      rcases List.mem_map.mp htailMap with ⟨tail, htail, hroute⟩
      subst route
      simpa [routeEnd] using GraphPath.cons edge hedge (ih htail)

theorem bfs_complete_within
    {State Action : Type}
    {g : GameGraph State Action}
    {start : State} {maxDepth : Nat}
    (hreachable : ∃ route goal,
      g.goal goal ∧
      route.length ≤ maxDepth ∧
      GraphPath g start route goal) :
    bfsFindsGoalWithin g start maxDepth := by
  rcases hreachable with ⟨route, goal, hgoal, hbound, hpath⟩
  rcases graph_path_mem_bfs_routes_exact hpath with ⟨hmem, hend⟩
  exact ⟨route.length, hbound, route, goal, hmem, hgoal, hend⟩

theorem bfs_reliable
    {State Action : Type}
    {g : GameGraph State Action}
    {start : State} {depth : Nat}
    (hfound : bfsFindsGoalAtDepth g start depth) :
    ∃ route goal,
      GraphPath g start route goal ∧ g.goal goal := by
  rcases hfound with ⟨route, goal, hmem, hgoal, hend⟩
  have hpath := bfs_route_layer_sound hmem
  rw [hend] at hpath
  exact ⟨route, goal, hpath, hgoal⟩

theorem bfs_complete_for_reachable_goal
    {State Action : Type}
    {g : GameGraph State Action}
    (hgoal : goalReachable g) :
    ∃ depth, bfsFindsGoalAtDepth g g.start depth := by
  rcases hgoal with ⟨goal, route, hgoal, hpath⟩
  rcases graph_path_mem_bfs_routes_exact hpath with ⟨hmem, hend⟩
  exact ⟨route.length, route, goal, hmem, hgoal, hend⟩

theorem bfs_failure_sound
    {State Action : Type}
    {g : GameGraph State Action}
    (hnoGoal : ∀ depth, ¬ bfsFindsGoalAtDepth g g.start depth) :
    ¬ goalReachable g := by
  intro hreachable
  rcases hreachable with ⟨goal, route, hgoal, hpath⟩
  rcases graph_path_mem_bfs_routes_exact hpath with ⟨hmem, hend⟩
  exact hnoGoal route.length ⟨route, goal, hmem, hgoal, hend⟩

theorem bfs_first_goal_layer_is_minimal
    {State Action : Type}
    {g : GameGraph State Action}
    {start : State} {firstDepth : Nat}
    (_hfirst : bfsFindsGoalAtDepth g start firstDepth)
    (hbefore : ∀ depth, depth < firstDepth ->
      ¬ bfsFindsGoalAtDepth g start depth) :
    ∀ depth, bfsFindsGoalAtDepth g start depth -> firstDepth ≤ depth := by
  intro depth hdepth
  by_cases hlt : depth < firstDepth
  · exact False.elim (hbefore depth hlt hdepth)
  · exact Nat.le_of_not_gt hlt

/-! ## 3. A milestone and the stepwise BFS theorem -/

inductive onPath
    {State Action : Type}
    (predicate : State -> Prop) :
    State -> List (Action × State) -> Prop where
  | atStart {s : State} {rest : List (Action × State)}
      (hs : predicate s) :
      onPath predicate s rest
  | atLater {s next : State} {action : Action}
      {rest : List (Action × State)}
      (htail : onPath predicate next rest) :
      onPath predicate s ((action, next) :: rest)

def necessaryMilestone
    {State Action : Type}
    (g : GameGraph State Action)
    (predicate : State -> Prop) : Prop :=
  ∀ {goal : State} {route : List (Action × State)},
    g.goal goal ->
    GraphPath g g.start route goal ->
    onPath predicate g.start route

theorem graph_path_split_at_milestone
    {State Action : Type}
    {g : GameGraph State Action}
    {predicate : State -> Prop}
    {start goal : State}
    {route : List (Action × State)}
    (hpath : GraphPath g start route goal)
    (hmilestone : onPath predicate start route) :
    ∃ milestone pre post,
      predicate milestone ∧
      GraphPath g start pre milestone ∧
      GraphPath g milestone post goal ∧
      route = pre ++ post := by
  induction hmilestone generalizing goal with
  | @atStart current rest hs =>
      exact ⟨current, [], rest, hs, GraphPath.nil current, hpath, by simp⟩
  | @atLater source next action rest htailMilestone ih =>
      cases hpath with
      | cons edge hedge htail =>
          rcases ih htail with
            ⟨milestone, pre, post, hp, hpre, hpost, hroute⟩
          refine ⟨milestone, (action, next) :: pre, post, hp, ?_, hpost, ?_⟩
          · exact GraphPath.cons (action, next) hedge hpre
          · simp [hroute]

theorem stepwise_bfs_theorem
    {State Action : Type}
    {g : GameGraph State Action}
    {predicate : State -> Prop}
    (hnecessary : necessaryMilestone g predicate) :
    goalReachable g ↔
      ∃ milestone goal pre post,
        predicate milestone ∧
        GraphPath g g.start pre milestone ∧
        g.goal goal ∧
        GraphPath g milestone post goal := by
  constructor
  · intro hgoal
    rcases hgoal with ⟨goal, route, hgoal, hpath⟩
    have hmilestone := hnecessary hgoal hpath
    rcases graph_path_split_at_milestone hpath hmilestone with
      ⟨milestone, pre, post, hp, hpre, hpost, hroute⟩
    exact ⟨milestone, goal, pre, post, hp, hpre, hgoal, hpost⟩
  · rintro ⟨milestone, goal, pre, post, hp, hpre, hgoal, hpost⟩
    exact ⟨goal, pre ++ post, hgoal,
      graph_path_append hpre hpost⟩

/-! ## 4. A concrete key-before-door instance -/

inductive KeyState where
  | start
  | keyRoom
  | door
  | victory
  deriving DecidableEq, Repr

inductive KeyAction where
  | walkToKey
  | collectKey
  | walkToDoor
  | openDoor
  deriving DecidableEq, Repr

def keyGraph : GameGraph KeyState KeyAction :=
  { transitions := fun state =>
      match state with
      | .start => [(KeyAction.walkToKey, .keyRoom)]
      | .keyRoom => [(KeyAction.collectKey, .door)]
      | .door => [(KeyAction.openDoor, .victory)]
      | .victory => []
    expandable := fun state =>
      match state with
      | .victory => false
      | _ => true
    goal := fun state => state = .victory
    start := .start }

def hasKey : KeyState -> Prop
  | .start => False
  | .keyRoom => True
  | .door => True
  | .victory => True

theorem keyGraph_has_necessary_milestone :
    necessaryMilestone keyGraph hasKey := by
  intro goal route hgoal hpath
  cases hpath with
  | nil state =>
      simp [keyGraph] at hgoal
  | cons edge hedge htail =>
      have hedgeEq : edge = (KeyAction.walkToKey, KeyState.keyRoom) := by
        simpa [keyGraph, legalTransitions] using hedge
      subst edge
      exact onPath.atLater (onPath.atStart trivial)

theorem keyGraph_stepwise_decomposition :
    goalReachable keyGraph ↔
      ∃ milestone goal pre post,
        hasKey milestone ∧
        GraphPath keyGraph keyGraph.start pre milestone ∧
        keyGraph.goal goal ∧
        GraphPath keyGraph milestone post goal := by
  exact stepwise_bfs_theorem keyGraph_has_necessary_milestone

end NesyLinkBFS
