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
