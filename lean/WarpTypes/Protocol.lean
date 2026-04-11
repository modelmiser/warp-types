import WarpTypes.Csp

/-
  Level 3: Protocol State (Experiment 3)

  Tests whether protocol compliance composes with CspHasType as a SEPARATE
  judgment (Option A) or must thread through the core (Option B).

  Uses the PingPong protocol from research/esp32-compiler/src/protocol.rs
  as the concrete instance.

  Central claim from §3 of complemented-typestate-framework.md:
  In the J1 grid, all cores are always running — protocol branching and
  participant-set divergence are ORTHOGONAL. CspExpr has no conditionals,
  so diverge doesn't create control-flow branches. This means FollowsProtocol
  can be a separate judgment that composes with CspHasType.
-/

-- ============================================================================
-- Protocol Actions and Roles
-- ============================================================================

/-- A single protocol action: directed send or recv between specific participants. -/
inductive ProtoAction (n : Nat)
  | send (self dst : Fin n)
  | recv (self src : Fin n)
  deriving DecidableEq

/-- A protocol role: the expected sequence of actions for one participant.
    Consumed front-to-back as the program executes. -/
abbrev ProtoRole (n : Nat) := List (ProtoAction n)

-- ============================================================================
-- PingPong Protocol (from protocol.rs)
--
-- Core 0: Send to core 1, then Recv from core 1
-- Core 1: Recv from core 0, then Send to core 0
-- ============================================================================

def pingPongCore0 : ProtoRole 6 :=
  [ ProtoAction.send ⟨0, by omega⟩ ⟨1, by omega⟩,
    ProtoAction.recv ⟨0, by omega⟩ ⟨1, by omega⟩ ]

def pingPongCore1 : ProtoRole 6 :=
  [ ProtoAction.recv ⟨1, by omega⟩ ⟨0, by omega⟩,
    ProtoAction.send ⟨1, by omega⟩ ⟨0, by omega⟩ ]

-- ============================================================================
-- FollowsProtocol Judgment (Option A: separate from CspHasType)
-- ============================================================================

/-- Protocol compliance judgment: proto_in ⊢ e ⊣ proto_out

    Tracks protocol consumption through an expression. Send/recv operations
    consume the head of the remaining protocol list; all other expressions
    thread the protocol through their sub-expressions.

    This is a SEPARATE judgment from CspHasType — testing Option A.
    If this composes with CspHasType (both can hold simultaneously),
    Option A works and the paper writes itself. -/
inductive FollowsProtocol {n : Nat} :
    ProtoRole n → CspExpr n → ProtoRole n → Prop

  -- ── Values: no protocol consumption ──

  | groupVal (proto : ProtoRole n) (s : PSet n) :
      FollowsProtocol proto (.groupVal s) proto
  | dataVal (proto : ProtoRole n) :
      FollowsProtocol proto .dataVal proto
  | unitVal (proto : ProtoRole n) :
      FollowsProtocol proto .unitVal proto
  | var (proto : ProtoRole n) (name : String) :
      FollowsProtocol proto (.var name) proto

  -- ── Structural: thread protocol through sub-expressions ──

  | diverge {proto proto' : ProtoRole n} {g : CspExpr n} {pred : PSet n} :
      FollowsProtocol proto g proto' →
      FollowsProtocol proto (.diverge g pred) proto'

  | merge {proto proto' proto'' : ProtoRole n} {g1 g2 : CspExpr n} :
      FollowsProtocol proto g1 proto' →
      FollowsProtocol proto' g2 proto'' →
      FollowsProtocol proto (.merge g1 g2) proto''

  | letBind {proto proto' proto'' : ProtoRole n} {name : String}
      {val body : CspExpr n} :
      FollowsProtocol proto val proto' →
      FollowsProtocol proto' body proto'' →
      FollowsProtocol proto (.letBind name val body) proto''

  | pairVal {proto proto' proto'' : ProtoRole n} {a b : CspExpr n} :
      FollowsProtocol proto a proto' →
      FollowsProtocol proto' b proto'' →
      FollowsProtocol proto (.pairVal a b) proto''

  | fstE {proto proto' : ProtoRole n} {e : CspExpr n} :
      FollowsProtocol proto e proto' →
      FollowsProtocol proto (.fst e) proto'

  | sndE {proto proto' : ProtoRole n} {e : CspExpr n} :
      FollowsProtocol proto e proto' →
      FollowsProtocol proto (.snd e) proto'

  | letPairE {proto proto' proto'' : ProtoRole n} {e : CspExpr n}
      {name1 name2 : String} {body : CspExpr n} :
      FollowsProtocol proto e proto' →
      FollowsProtocol proto' body proto'' →
      FollowsProtocol proto (.letPair e name1 name2 body) proto''

  -- ── Protocol-consuming: send/recv match head of protocol list ──

  /-- Send: thread protocol through sub-expressions, then consume
      Send(self,dst) from the head of the remaining protocol. -/
  | send {proto proto_g proto_p rest : ProtoRole n} {g payload : CspExpr n}
      {self dst : Fin n} :
      FollowsProtocol proto g proto_g →
      FollowsProtocol proto_g payload proto_p →
      proto_p = ProtoAction.send self dst :: rest →
      FollowsProtocol proto (.send g payload self dst) rest

  /-- Recv: thread protocol through sub-expression, then consume
      Recv(self,src) from the head of the remaining protocol. -/
  | recv {proto proto_g rest : ProtoRole n} {g : CspExpr n}
      {self src : Fin n} :
      FollowsProtocol proto g proto_g →
      proto_g = ProtoAction.recv self src :: rest →
      FollowsProtocol proto (.recv g self src) rest

  /-- Collective: synchronization barrier, not a directed message.
      No protocol consumption. -/
  | collective {proto proto' proto'' : ProtoRole n} {g payload : CspExpr n} :
      FollowsProtocol proto g proto' →
      FollowsProtocol proto' payload proto'' →
      FollowsProtocol proto (.collective g payload) proto''

-- ============================================================================
-- Protocol Duality
-- ============================================================================

/-- Two protocol roles are dual: each Send in one matches a Recv in the other
    at the same position, with sender/receiver swapped. -/
def IsDual {n : Nat} : ProtoRole n → ProtoRole n → Prop
  | [], [] => True
  | ProtoAction.send s1 d1 :: rest1, ProtoAction.recv s2 d2 :: rest2 =>
      s1 = d2 ∧ d1 = s2 ∧ IsDual rest1 rest2
  | ProtoAction.recv s1 d1 :: rest1, ProtoAction.send s2 d2 :: rest2 =>
      s1 = d2 ∧ d1 = s2 ∧ IsDual rest1 rest2
  | _, _ => False

/-- PingPong roles are dual. -/
theorem pingPong_dual : IsDual pingPongCore0 pingPongCore1 := by
  unfold pingPongCore0 pingPongCore1
  unfold IsDual
  refine ⟨rfl, rfl, ?_⟩
  unfold IsDual
  exact ⟨rfl, rfl, trivial⟩

-- ============================================================================
-- PingPong Programs
-- ============================================================================

/-- Core 0's PingPong program:
    let g = groupVal All in
    let g2 = send g dataVal 0→1 in       -- Send to core 1
    let (d, g3) = recv g2 0←1 in         -- Recv from core 1
    (d, g3)                               -- consume both bindings -/
def pingPongProg0 : CspExpr 6 :=
  .letBind "g" (.groupVal TileSet.all)
    (.letBind "g2"
      (.send (.var "g") .dataVal ⟨0, by omega⟩ ⟨1, by omega⟩)
      (.letPair
        (.recv (.var "g2") ⟨0, by omega⟩ ⟨1, by omega⟩)
        "d" "g3"
        (.pairVal (.var "d") (.var "g3"))))

/-- Core 1's PingPong program:
    let g = groupVal All in
    let (d, g2) = recv g 1←0 in          -- Recv from core 0
    let g3 = send g2 dataVal 1→0 in      -- Send to core 0
    (d, g3)                               -- consume both bindings -/
def pingPongProg1 : CspExpr 6 :=
  .letBind "g" (.groupVal TileSet.all)
    (.letPair
      (.recv (.var "g") ⟨1, by omega⟩ ⟨0, by omega⟩)
      "d" "g2"
      (.letBind "g3"
        (.send (.var "g2") .dataVal ⟨1, by omega⟩ ⟨0, by omega⟩)
        (.pairVal (.var "d") (.var "g3"))))

-- ============================================================================
-- CspHasType proofs: PingPong programs are well-typed
-- ============================================================================

/-- Core 0's PingPong program is well-typed on the J1 grid.
    The typing derivation threads the group handle linearly through
    send → recv, consuming each binding exactly once. -/
theorem pingPongProg0_typed :
    CspHasType j1Grid ([] : CspCtx 6) pingPongProg0
      (.pair .data (.group TileSet.all)) [] := by
  unfold pingPongProg0
  -- letBind "g" (groupVal All) body
  apply CspHasType.letBind (t1 := .group TileSet.all)
  · exact CspHasType.groupVal _ _
  · rfl  -- freshness
  · -- ctx: [("g", group All)]
    apply CspHasType.letBind (t1 := .group TileSet.all)
    · -- send (var "g") dataVal 0→1
      apply CspHasType.send (s := TileSet.all)
      · exact CspHasType.var _ _ _ rfl
      · decide
      · decide
      · exact CspHasType.dataVal _
    · rfl  -- freshness
    · -- ctx: [("g2", group All)]
      apply CspHasType.letPairE (t1 := .data) (t2 := .group TileSet.all)
      · -- recv (var "g2") 0←1
        apply CspHasType.recv (s := TileSet.all)
        · exact CspHasType.var _ _ _ rfl
        · decide
        · decide
      · decide  -- "d" ≠ "g3"
      · rfl     -- freshness
      · rfl     -- freshness
      · -- ctx: [("g3", group All), ("d", data)]
        apply CspHasType.pairVal
        · exact CspHasType.var _ _ _ rfl
        · exact CspHasType.var _ _ _ rfl
      · rfl  -- linearity
      · rfl  -- linearity
    · rfl  -- linearity
  · rfl  -- linearity

/-- Core 1's PingPong program is well-typed on the J1 grid. -/
theorem pingPongProg1_typed :
    CspHasType j1Grid ([] : CspCtx 6) pingPongProg1
      (.pair .data (.group TileSet.all)) [] := by
  unfold pingPongProg1
  -- letBind "g" (groupVal All) body
  apply CspHasType.letBind (t1 := .group TileSet.all)
  · exact CspHasType.groupVal _ _
  · rfl
  · -- ctx: [("g", group All)]
    apply CspHasType.letPairE (t1 := .data) (t2 := .group TileSet.all)
    · -- recv (var "g") 1←0
      apply CspHasType.recv (s := TileSet.all)
      · exact CspHasType.var _ _ _ rfl
      · decide
      · decide
    · decide  -- "d" ≠ "g2"
    · rfl
    · rfl
    · -- ctx: [("g2", group All), ("d", data)]
      apply CspHasType.letBind (t1 := .group TileSet.all)
      · -- send (var "g2") dataVal 1→0
        apply CspHasType.send (s := TileSet.all)
        · exact CspHasType.var _ _ _ rfl
        · decide
        · decide
        · exact CspHasType.dataVal _
      · rfl
      · -- ctx: [("g3", group All), ("d", data)]
        apply CspHasType.pairVal
        · exact CspHasType.var _ _ _ rfl
        · exact CspHasType.var _ _ _ rfl
      · rfl
    · rfl
    · rfl
  · rfl

-- ============================================================================
-- FollowsProtocol proofs: PingPong programs follow their protocol roles
-- ============================================================================

/-- Core 0's program follows the core0 role: Send(0→1), Recv(0←1).
    Protocol is fully consumed (output = []). -/
theorem pingPongProg0_follows_protocol :
    FollowsProtocol pingPongCore0 pingPongProg0 [] := by
  unfold pingPongProg0 pingPongCore0
  apply FollowsProtocol.letBind
  · exact FollowsProtocol.groupVal _ _
  · apply FollowsProtocol.letBind
    · -- send: consumes Send(0,1) from protocol head
      apply FollowsProtocol.send
      · exact FollowsProtocol.var _ _
      · exact FollowsProtocol.dataVal _
      · rfl  -- [Send(0,1), Recv(0,1)] = Send(0,1) :: [Recv(0,1)]
    · -- letPair: thread protocol [Recv(0,1)] through recv
      apply FollowsProtocol.letPairE
      · -- recv: consumes Recv(0,1) from protocol head
        apply FollowsProtocol.recv
        · exact FollowsProtocol.var _ _
        · rfl  -- [Recv(0,1)] = Recv(0,1) :: []
      · apply FollowsProtocol.pairVal
        · exact FollowsProtocol.var _ _
        · exact FollowsProtocol.var _ _

/-- Core 1's program follows the core1 role: Recv(1←0), Send(1→0).
    Protocol is fully consumed (output = []). -/
theorem pingPongProg1_follows_protocol :
    FollowsProtocol pingPongCore1 pingPongProg1 [] := by
  unfold pingPongProg1 pingPongCore1
  apply FollowsProtocol.letBind
  · exact FollowsProtocol.groupVal _ _
  · apply FollowsProtocol.letPairE
    · -- recv: consumes Recv(1,0) from protocol head
      apply FollowsProtocol.recv
      · exact FollowsProtocol.var _ _
      · rfl
    · apply FollowsProtocol.letBind
      · -- send: consumes Send(1,0) from protocol head
        apply FollowsProtocol.send
        · exact FollowsProtocol.var _ _
        · exact FollowsProtocol.dataVal _
        · rfl
      · apply FollowsProtocol.pairVal
        · exact FollowsProtocol.var _ _
        · exact FollowsProtocol.var _ _

-- ============================================================================
-- Composition Existence: Option A works for PingPong
-- ============================================================================

/-- THE KEY RESULT: Both judgments hold simultaneously.

    CspHasType checks WHO (participant sets, topology constraints).
    FollowsProtocol checks WHAT/WHEN (action sequence matches role).
    They compose as independent judgments — Option A from §3 Level 3.

    This works because CspExpr has no conditional control flow.
    Diverge splits participant GROUPS, not control-flow BRANCHES.
    All cores always run; protocol actions happen in a fixed order
    regardless of diverge state. -/
theorem option_a_composition :
    ∃ (t : CspTy 6) (ctx' : CspCtx 6),
      CspHasType j1Grid [] pingPongProg0 t ctx' ∧
      FollowsProtocol pingPongCore0 pingPongProg0 [] ∧
      CspHasType j1Grid [] pingPongProg1 t ctx' ∧
      FollowsProtocol pingPongCore1 pingPongProg1 [] ∧
      IsDual pingPongCore0 pingPongCore1 :=
  ⟨_, _, pingPongProg0_typed, pingPongProg0_follows_protocol,
         pingPongProg1_typed, pingPongProg1_follows_protocol,
         pingPong_dual⟩

-- ============================================================================
-- Negative instances: each judgment catches DIFFERENT bugs
-- ============================================================================

/-- Protocol violation detected by FollowsProtocol:
    Sending when the protocol expects receiving is untypable.
    (CspHasType would accept this — it only checks topology.) -/
theorem wrong_order_violates_protocol :
    ¬ FollowsProtocol pingPongCore1  -- core1 role: Recv first
      (.send (.groupVal TileSet.all) .dataVal ⟨1, by omega⟩ ⟨0, by omega⟩)
      [] := by
  unfold pingPongCore1
  intro h
  cases h with
  | send hg hp heq =>
    -- After threading through groupVal and dataVal (no consumption),
    -- proto_p = [Recv(1,0), Send(1,0)]. But heq requires head = Send(1,0).
    cases hg with
    | groupVal =>
      cases hp with
      | dataVal =>
        -- heq : [Recv(1,0), Send(1,0)] = Send(1,0) :: ?rest
        -- Head mismatch: Recv ≠ Send
        simp at heq

/-- Topology violation detected by CspHasType:
    Sending to a non-adjacent core is untypable.
    (FollowsProtocol would accept this — it only checks action sequence.) -/
theorem wrong_topology_violates_typing :
    ¬ ∃ t ctx', CspHasType j1Grid []
      (.send (.groupVal TileSet.all) .dataVal ⟨0, by omega⟩ ⟨5, by omega⟩)
      t ctx' :=
  j1_send_opposite_corners_untypable  -- reuse existing theorem

-- ============================================================================
-- Experiment 3b: Branching Protocol (Level 3 stress test)
--
-- PingPong is flat — it never exercises `diverge`. The Option A claim that
-- "protocol branching and participant-set divergence are orthogonal" has
-- therefore not been stressed on a case where diverge and protocol state
-- actually couple.
--
-- This experiment couples them: we split All → leftCol + rightCol via
-- `diverge`, bind both sub-groups with `letPair`, then use the two handles
-- to perform four sends — two in the left column, two in the right column.
--
-- If both `CspHasType` and `FollowsProtocol` hold simultaneously on the
-- same program with a single flat protocol role, Option A survives the
-- stress test: `FollowsProtocol` genuinely does not care which sub-group
-- is doing which action, only the sequence of `self`/`dst` endpoints.
-- ============================================================================

/-- Branching protocol role: four sends in source order.
    Left column: 0→2 then 2→4. Right column: 1→3 then 3→5.
    All four happen in one execution — diverge doesn't gate them away. -/
def branchingCore : ProtoRole 6 :=
  [ ProtoAction.send ⟨0, by omega⟩ ⟨2, by omega⟩,   -- leftCol  step 1
    ProtoAction.send ⟨2, by omega⟩ ⟨4, by omega⟩,   -- leftCol  step 2
    ProtoAction.send ⟨1, by omega⟩ ⟨3, by omega⟩,   -- rightCol step 1
    ProtoAction.send ⟨3, by omega⟩ ⟨5, by omega⟩ ]  -- rightCol step 2

/-- The branching program:

      let g  = groupVal All in
      let (gL, gR) = diverge g leftCol in          -- gL : leftCol, gR : rightCol
      let gL2 = send gL  dataVal 0→2 in            -- consumes Send 0→2
      let gL3 = send gL2 dataVal 2→4 in            -- consumes Send 2→4
      let gR2 = send gR  dataVal 1→3 in            -- consumes Send 1→3
      let gR3 = send gR2 dataVal 3→5 in            -- consumes Send 3→5
      (gL3, gR3)

    Both sub-groups coexist (no control-flow branch). The four sends are
    sequenced by the expression tree — source order IS protocol order. -/
def branchingProg : CspExpr 6 :=
  .letBind "g" (.groupVal TileSet.all)
    (.letPair
      (.diverge (.var "g") TileSet.leftCol)
      "gL" "gR"
      (.letBind "gL2"
        (.send (.var "gL") .dataVal ⟨0, by omega⟩ ⟨2, by omega⟩)
        (.letBind "gL3"
          (.send (.var "gL2") .dataVal ⟨2, by omega⟩ ⟨4, by omega⟩)
          (.letBind "gR2"
            (.send (.var "gR") .dataVal ⟨1, by omega⟩ ⟨3, by omega⟩)
            (.letBind "gR3"
              (.send (.var "gR2") .dataVal ⟨3, by omega⟩ ⟨5, by omega⟩)
              (.pairVal (.var "gL3") (.var "gR3")))))))

-- ============================================================================
-- CspHasType: branchingProg is well-typed on j1Grid
-- ============================================================================

/-- The branching program type-checks on the J1 grid.

    Every send edge (0→2, 2→4, 1→3, 3→5) is a real vertical link in the
    2×3 grid, and every destination lives inside the sub-group doing the
    send, so linearity, adjacency, and active-destination all hold. -/
theorem branchingProg_typed :
    CspHasType j1Grid ([] : CspCtx 6) branchingProg
      (.pair (.group (TileSet.all &&& TileSet.leftCol))
             (.group (TileSet.all &&& ~~~TileSet.leftCol))) [] := by
  unfold branchingProg
  -- let g = groupVal All
  apply CspHasType.letBind (t1 := .group TileSet.all)
  · exact CspHasType.groupVal _ _
  · rfl
  · -- ctx: [("g", group All)]
    -- letPair (diverge (var g) leftCol) "gL" "gR" body
    apply CspHasType.letPairE
      (t1 := .group (TileSet.all &&& TileSet.leftCol))
      (t2 := .group (TileSet.all &&& ~~~TileSet.leftCol))
    · -- diverge (var g) leftCol
      apply CspHasType.diverge (s := TileSet.all)
      exact CspHasType.var _ _ _ rfl
    · decide   -- "gL" ≠ "gR"
    · rfl      -- lookup "gL" in [] = none
    · rfl      -- lookup "gR" in [] = none
    · -- body ctx: [("gR", group (All & ~leftCol)), ("gL", group (All & leftCol))]
      -- let gL2 = send (var gL) dataVal 0 2
      apply CspHasType.letBind (t1 := .group (TileSet.all &&& TileSet.leftCol))
      · apply CspHasType.send (s := TileSet.all &&& TileSet.leftCol)
        · exact CspHasType.var _ _ _ rfl
        · decide   -- bit 2 of (All & leftCol) is true
        · decide   -- adj 0 2 on j1Grid
        · exact CspHasType.dataVal _
      · rfl
      · -- ctx: [("gL2", group leftCol'), ("gR", group rightCol')]
        -- let gL3 = send (var gL2) dataVal 2 4
        apply CspHasType.letBind (t1 := .group (TileSet.all &&& TileSet.leftCol))
        · apply CspHasType.send (s := TileSet.all &&& TileSet.leftCol)
          · exact CspHasType.var _ _ _ rfl
          · decide   -- bit 4
          · decide   -- adj 2 4
          · exact CspHasType.dataVal _
        · rfl
        · -- ctx: [("gL3", group leftCol'), ("gR", group rightCol')]
          -- let gR2 = send (var gR) dataVal 1 3
          apply CspHasType.letBind
            (t1 := .group (TileSet.all &&& ~~~TileSet.leftCol))
          · apply CspHasType.send (s := TileSet.all &&& ~~~TileSet.leftCol)
            · exact CspHasType.var _ _ _ rfl
            · decide   -- bit 3
            · decide   -- adj 1 3
            · exact CspHasType.dataVal _
          · rfl
          · -- ctx: [("gR2", group rightCol'), ("gL3", group leftCol')]
            -- let gR3 = send (var gR2) dataVal 3 5
            apply CspHasType.letBind
              (t1 := .group (TileSet.all &&& ~~~TileSet.leftCol))
            · apply CspHasType.send (s := TileSet.all &&& ~~~TileSet.leftCol)
              · exact CspHasType.var _ _ _ rfl
              · decide   -- bit 5
              · decide   -- adj 3 5
              · exact CspHasType.dataVal _
            · rfl
            · -- ctx: [("gR3", group rightCol'), ("gL3", group leftCol')]
              -- pairVal (var gL3) (var gR3)
              apply CspHasType.pairVal
              · exact CspHasType.var _ _ _ rfl
              · exact CspHasType.var _ _ _ rfl
            · rfl
          · rfl
        · rfl
      · rfl
    · rfl
    · rfl
  · rfl

-- ============================================================================
-- FollowsProtocol: branchingProg follows branchingCore
-- ============================================================================

/-- The branching program consumes `branchingCore` exactly.

    Key threading: `diverge` forwards the protocol through `var g` unchanged
    (no consumption); `letPair` sequences e-then-body; each inner `send`
    consumes the head of the remaining list. The four sub-groups are
    invisible to `FollowsProtocol` — only the `(self, dst)` pairs matter. -/
theorem branchingProg_follows_protocol :
    FollowsProtocol branchingCore branchingProg [] := by
  unfold branchingProg branchingCore
  apply FollowsProtocol.letBind
  · exact FollowsProtocol.groupVal _ _
  · apply FollowsProtocol.letPairE
    · -- diverge (var g) leftCol: threads protocol through var g (no consume)
      apply FollowsProtocol.diverge
      exact FollowsProtocol.var _ _
    · -- body: four sends consuming the role head-by-head
      apply FollowsProtocol.letBind
      · apply FollowsProtocol.send
        · exact FollowsProtocol.var _ _
        · exact FollowsProtocol.dataVal _
        · rfl   -- head = Send 0→2
      · apply FollowsProtocol.letBind
        · apply FollowsProtocol.send
          · exact FollowsProtocol.var _ _
          · exact FollowsProtocol.dataVal _
          · rfl -- head = Send 2→4
        · apply FollowsProtocol.letBind
          · apply FollowsProtocol.send
            · exact FollowsProtocol.var _ _
            · exact FollowsProtocol.dataVal _
            · rfl -- head = Send 1→3
          · apply FollowsProtocol.letBind
            · apply FollowsProtocol.send
              · exact FollowsProtocol.var _ _
              · exact FollowsProtocol.dataVal _
              · rfl -- head = Send 3→5
            · apply FollowsProtocol.pairVal
              · exact FollowsProtocol.var _ _
              · exact FollowsProtocol.var _ _

-- ============================================================================
-- Composition Existence (strengthened): Option A survives the branching stress test
-- ============================================================================

/-- Option A survives when diverge and protocol state are coupled.

    `branchingProg` uses `diverge` + `letPair` to split `All` into
    `leftCol` + `rightCol`, then uses each sub-group handle for a
    distinct send sub-sequence. Yet both judgments still compose:

      • `CspHasType` checks WHO/WHERE — sub-group membership + mesh adjacency.
      • `FollowsProtocol` checks WHEN — the flat sequence of `(self, dst)`
        endpoints in source order.

    Because `CspExpr` has no conditional control flow, `diverge` does NOT
    produce alternative execution paths — it produces two sub-group handles
    that BOTH coexist in the one execution. The protocol is therefore a
    single flat list: the concatenation of all send/recv actions in source
    order. This is the precise sense in which "diverge is orthogonal to
    protocol state" — and Option A composes even when the program exercises
    diverge nontrivially. -/
theorem option_a_branching_composition :
    ∃ (t : CspTy 6) (ctx' : CspCtx 6),
      CspHasType j1Grid [] branchingProg t ctx' ∧
      FollowsProtocol branchingCore branchingProg [] :=
  ⟨_, _, branchingProg_typed, branchingProg_follows_protocol⟩

-- ============================================================================
-- Protocol Trace: structural characterization of FollowsProtocol
--
-- The lemma hinted at by Experiment 3b: FollowsProtocol is EXACTLY
-- "leading prefix of the depth-first action trace of the expression."
-- Once this is proved, every future per-program protocol witness collapses
-- to a single `rfl` — and the "diverge is protocol-invisible" insight
-- becomes a one-line corollary instead of a case analysis.
-- ============================================================================

/-- The protocol trace of an expression: the flat depth-first left-to-right
    sequence of `send` / `recv` actions.

    Every non-protocol constructor (`groupVal`, `dataVal`, `unitVal`, `var`,
    `diverge`, `merge`, `letBind`, `pairVal`, `fst`, `snd`, `letPair`,
    `collective`) is structurally transparent — it contributes only the
    concatenation of its sub-expressions' traces. Only `send` and `recv`
    emit an action.

    Note that `diverge g pred` ignores `pred` entirely: the participant-set
    predicate lives in a different lattice from the protocol, and this
    function is the structural witness of that orthogonality. -/
def protocolTrace {n : Nat} : CspExpr n → List (ProtoAction n)
  | .groupVal _           => []
  | .dataVal              => []
  | .unitVal              => []
  | .var _                => []
  | .diverge g _          => protocolTrace g
  | .merge g1 g2          => protocolTrace g1 ++ protocolTrace g2
  | .letBind _ val body   => protocolTrace val ++ protocolTrace body
  | .pairVal a b          => protocolTrace a ++ protocolTrace b
  | .fst e                => protocolTrace e
  | .snd e                => protocolTrace e
  | .letPair e _ _ body   => protocolTrace e ++ protocolTrace body
  | .send g payload self dst =>
      protocolTrace g ++ protocolTrace payload ++ [ProtoAction.send self dst]
  | .recv g self src =>
      protocolTrace g ++ [ProtoAction.recv self src]
  | .collective g payload => protocolTrace g ++ protocolTrace payload

/-- Forward direction: any `FollowsProtocol` derivation consumes exactly
    `protocolTrace e` from the head of its input protocol.

    Proof by induction on the derivation. Each non-protocol constructor
    simply composes IHs. The `send`/`recv` cases rewrite the explicit
    equality (`proto_p = ProtoAction.send self dst :: rest`) into the
    concatenation shape. -/
theorem followsProtocol_to_trace {n : Nat} {e : CspExpr n}
    {proto proto' : ProtoRole n} :
    FollowsProtocol proto e proto' → proto = protocolTrace e ++ proto' := by
  intro h
  induction h with
  | groupVal _ _ => simp [protocolTrace]
  | dataVal _ => simp [protocolTrace]
  | unitVal _ => simp [protocolTrace]
  | var _ _ => simp [protocolTrace]
  | diverge _ ih => simpa [protocolTrace] using ih
  | merge _ _ ih1 ih2 =>
      simp [protocolTrace, List.append_assoc]
      rw [ih1, ih2]
  | letBind _ _ ih1 ih2 =>
      simp [protocolTrace, List.append_assoc]
      rw [ih1, ih2]
  | pairVal _ _ ih1 ih2 =>
      simp [protocolTrace, List.append_assoc]
      rw [ih1, ih2]
  | fstE _ ih => simpa [protocolTrace] using ih
  | sndE _ ih => simpa [protocolTrace] using ih
  | letPairE _ _ ih1 ih2 =>
      simp [protocolTrace, List.append_assoc]
      rw [ih1, ih2]
  | send _ _ heq ihg ihp =>
      subst heq
      simp [protocolTrace, List.append_assoc]
      rw [ihg, ihp]
  | recv _ heq ih =>
      subst heq
      simp [protocolTrace, List.append_assoc]
      rw [ih]
  | collective _ _ ih1 ih2 =>
      simp [protocolTrace, List.append_assoc]
      rw [ih1, ih2]

/-- Backward direction: any leading prefix equal to `protocolTrace e`
    admits a `FollowsProtocol` derivation.

    Stated as the substitution form `FollowsProtocol (protocolTrace e ++ r) e r`
    so that the induction carries a universally quantified leftover `r`.
    Proof by structural induction on `e`, generalizing `proto'`. -/
theorem trace_to_followsProtocol {n : Nat} (e : CspExpr n)
    (proto' : ProtoRole n) :
    FollowsProtocol (protocolTrace e ++ proto') e proto' := by
  induction e generalizing proto' with
  | groupVal s =>
      simp [protocolTrace]; exact FollowsProtocol.groupVal _ _
  | dataVal =>
      simp [protocolTrace]; exact FollowsProtocol.dataVal _
  | unitVal =>
      simp [protocolTrace]; exact FollowsProtocol.unitVal _
  | var name =>
      simp [protocolTrace]; exact FollowsProtocol.var _ _
  | diverge g pred ih =>
      simp [protocolTrace]
      exact FollowsProtocol.diverge (ih proto')
  | merge g1 g2 ih1 ih2 =>
      simp [protocolTrace, List.append_assoc]
      exact FollowsProtocol.merge
        (ih1 (protocolTrace g2 ++ proto')) (ih2 proto')
  | letBind name val body ihval ihbody =>
      simp [protocolTrace, List.append_assoc]
      exact FollowsProtocol.letBind
        (ihval (protocolTrace body ++ proto')) (ihbody proto')
  | pairVal a b iha ihb =>
      simp [protocolTrace, List.append_assoc]
      exact FollowsProtocol.pairVal
        (iha (protocolTrace b ++ proto')) (ihb proto')
  | fst e ih =>
      simp [protocolTrace]
      exact FollowsProtocol.fstE (ih proto')
  | snd e ih =>
      simp [protocolTrace]
      exact FollowsProtocol.sndE (ih proto')
  | letPair e name1 name2 body ihe ihbody =>
      simp [protocolTrace, List.append_assoc]
      exact FollowsProtocol.letPairE
        (ihe (protocolTrace body ++ proto')) (ihbody proto')
  | send g payload self dst ihg ihp =>
      simp [protocolTrace, List.append_assoc]
      -- Goal shape:
      -- FollowsProtocol (tr g ++ tr p ++ Send self dst :: proto')
      --                 (.send g payload self dst) proto'
      exact FollowsProtocol.send
        (ihg (protocolTrace payload ++ ProtoAction.send self dst :: proto'))
        (ihp (ProtoAction.send self dst :: proto'))
        rfl
  | recv g self src ihg =>
      simp [protocolTrace, List.append_assoc]
      exact FollowsProtocol.recv
        (ihg (ProtoAction.recv self src :: proto'))
        rfl
  | collective g payload ih1 ih2 =>
      simp [protocolTrace, List.append_assoc]
      exact FollowsProtocol.collective
        (ih1 (protocolTrace payload ++ proto')) (ih2 proto')

/-- The structural characterization theorem. `FollowsProtocol` is exactly
    "leading prefix equal to `protocolTrace`". -/
theorem followsProtocol_iff_trace {n : Nat} (e : CspExpr n)
    (proto proto' : ProtoRole n) :
    FollowsProtocol proto e proto' ↔ proto = protocolTrace e ++ proto' := by
  constructor
  · exact followsProtocol_to_trace
  · intro h; subst h; exact trace_to_followsProtocol e proto'

/-- Self-trace corollary: every expression trivially follows its own trace
    with empty leftover. Combined with `rfl` on the trace computation, this
    decides any `FollowsProtocol` goal whose role matches the program's trace. -/
theorem followsProtocol_self {n : Nat} (e : CspExpr n) :
    FollowsProtocol (protocolTrace e) e [] := by
  have h := trace_to_followsProtocol e []
  simpa using h

-- ============================================================================
-- Computational witness: the branching program case falls out by reflexivity
--
-- This is the payoff of the structural lemma — the whole 30-line tactic proof
-- of `branchingProg_follows_protocol` collapses to a trace computation plus
-- `followsProtocol_self`.
-- ============================================================================

/-- `branchingCore` is literally the computed trace of `branchingProg`. -/
theorem branchingCore_eq_trace :
    branchingCore = protocolTrace branchingProg := by
  unfold branchingCore branchingProg protocolTrace
  rfl

/-- `FollowsProtocol` on the branching program, proved by computation
    rather than by a 30-line case walk. Compare with
    `branchingProg_follows_protocol` above. -/
theorem branchingProg_follows_protocol' :
    FollowsProtocol branchingCore branchingProg [] := by
  rw [branchingCore_eq_trace]
  exact followsProtocol_self branchingProg

-- ============================================================================
-- Diverge-is-protocol-invisible, stated as a theorem
-- ============================================================================

/-- **The structural statement of Option A.** The participant-set predicate
    in `diverge` is invisible to the protocol trace. This is the single
    line that captures why `FollowsProtocol` and `CspHasType` compose: the
    protocol layer cannot observe which sub-group is doing what, only the
    depth-first order of `send`/`recv` endpoints.

    For two programs that differ only in their `diverge` predicates, the
    protocol trace — and therefore `FollowsProtocol` — is identical. -/
theorem diverge_pred_is_protocol_invisible {n : Nat}
    (g : CspExpr n) (pred1 pred2 : PSet n) :
    protocolTrace (.diverge g pred1) = protocolTrace (.diverge g pred2) := by
  rfl
