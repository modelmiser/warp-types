import WarpTypes.Generic

/-
  CSP Domain Extension (Level 2b)

  Extends the width-generic PSet framework with CSP (Communicating Sequential
  Processes) operations: send, recv, collective. Parameterized by a topology.

  Demonstrates the central claim of the complemented typestate framework:
  the SAME mechanism that prevents GPU warp divergence bugs ALSO prevents
  CSP protocol violations on a multi-core mesh.

  Key parallels:
  - GPU shuffle requires Warp<All>       ↔  CSP collective requires Group<All>
  - GPU shuffle-after-diverge untypable  ↔  CSP collective-after-diverge untypable
  - GPU has no topology constraint       ↔  CSP send/recv require adjacency (NEW)

  All core theorems (diverge_partition, complement_symmetric) are reused
  from Generic.lean without modification — the width `n` is the only parameter.
-/

-- ============================================================================
-- Topology (§4 of complemented-typestate-framework.md)
-- ============================================================================

/-- A topology on n participants: who can communicate directly.
    Adjacency is symmetric (bidirectional links) and irreflexive (no self-loops). -/
structure Topology (n : Nat) where
  adj : Fin n → Fin n → Bool
  sym : ∀ i j, adj i j = adj j i
  irrefl : ∀ i, adj i i = false

-- ============================================================================
-- J1 2×3 Grid Instance (6 Super-J1 cores)
-- ============================================================================

/-
  Physical layout (from GRID_TOPOLOGY.md):

    J10 ─── J20       (core 0 ─── core 1)
     │       │
    J11 ─── J21       (core 2 ─── core 3)
     │       │
    J12 ─── J22       (core 4 ─── core 5)

  7 bidirectional CSP links. Max 3 hops (corner to opposite corner).
-/
private def j1GridAdj (i j : Fin 6) : Bool :=
  match i.val, j.val with
  | 0, 1 | 1, 0 => true   -- row 0 horizontal
  | 0, 2 | 2, 0 => true   -- col 0 vertical (row 0-1)
  | 1, 3 | 3, 1 => true   -- col 1 vertical (row 0-1)
  | 2, 3 | 3, 2 => true   -- row 1 horizontal
  | 2, 4 | 4, 2 => true   -- col 0 vertical (row 1-2)
  | 3, 5 | 5, 3 => true   -- col 1 vertical (row 1-2)
  | 4, 5 | 5, 4 => true   -- row 2 horizontal
  | _, _ => false

def j1Grid : Topology 6 where
  adj := j1GridAdj
  sym := by decide
  irrefl := by decide

-- ============================================================================
-- TileSet: PSet 6 instantiation (parallel to ActiveSet := PSet 32)
-- ============================================================================

/-- A tile set is a 6-bit bitvector representing which J1 cores are active. -/
abbrev TileSet := PSet 6

namespace TileSet

def all : TileSet := PSet.all 6
def none : TileSet := PSet.none 6
def leftCol : TileSet := 0x15#6     -- cores 0, 2, 4 (left column)
def rightCol : TileSet := 0x2A#6    -- cores 1, 3, 5 (right column)
def topRow : TileSet := 0x03#6      -- cores 0, 1 (top row)
def bottomRow : TileSet := 0x30#6   -- cores 4, 5 (bottom row)

end TileSet

-- ============================================================================
-- Concrete TileSet complement instances
-- ============================================================================

/-- Left and right columns are complements within All.
    Parallel to even_odd_complement in Basic.lean (GPU domain). -/
theorem leftCol_rightCol_complement :
    PSet.IsComplementAll TileSet.leftCol TileSet.rightCol := by
  unfold PSet.IsComplementAll PSet.IsComplement
  unfold PSet.Disjoint PSet.Covers
  unfold TileSet.leftCol TileSet.rightCol PSet.all PSet.none
  constructor <;> decide

/-- Top and bottom rows are complements within their union (NOT within All).
    Demonstrates nested diverge: split All into left/right, then
    further split within a column. -/
theorem topRow_bottomRow_complement_within :
    PSet.IsComplement TileSet.topRow TileSet.bottomRow (0x33#6) := by
  unfold PSet.IsComplement PSet.Disjoint PSet.Covers
  unfold TileSet.topRow TileSet.bottomRow PSet.none
  constructor <;> decide

-- ============================================================================
-- CSP Types (Level 2b)
-- ============================================================================

inductive CspTy (n : Nat)
  | group (s : PSet n)   -- participant set (Warp<S> → Group<S>)
  | data                  -- per-participant payload (PerLane → Data)
  | unit
  | pair (a b : CspTy n)

-- ============================================================================
-- CSP Expressions (Level 1 core + Level 2b extensions)
-- ============================================================================

inductive CspExpr (n : Nat)
  -- Core (Level 1 — same mechanism as GPU)
  | groupVal (s : PSet n)
  | dataVal
  | unitVal
  | var (name : String)
  | diverge (g : CspExpr n) (pred : PSet n)
  | merge (g1 g2 : CspExpr n)
  | letBind (name : String) (val body : CspExpr n)
  | pairVal (a b : CspExpr n)
  | fst (e : CspExpr n)
  | snd (e : CspExpr n)
  | letPair (e : CspExpr n) (name1 name2 : String) (body : CspExpr n)
  -- Level 2b: CSP-specific
  | send (g payload : CspExpr n) (self dst : Fin n)
  | recv (g : CspExpr n) (self src : Fin n)
  | collective (g payload : CspExpr n)

-- ============================================================================
-- CSP Context (linear)
-- ============================================================================

def CspCtx (n : Nat) := List (String × CspTy n)

namespace CspCtx

def lookup {n : Nat} (ctx : CspCtx n) (name : String) : Option (CspTy n) :=
  ctx.find? (fun p => p.1 == name) |>.map Prod.snd

def remove {n : Nat} (ctx : CspCtx n) (name : String) : CspCtx n :=
  ctx.filter (fun p => p.1 != name)

end CspCtx

-- ============================================================================
-- CSP Typing Rules (parameterized by topology)
-- ============================================================================

/-- Linear typing judgement for CSP: Γ ⊢_topo e : τ ⊣ Γ'
    The topology constrains which send/recv operations are well-typed. -/
inductive CspHasType {n : Nat} (topo : Topology n) :
    CspCtx n → CspExpr n → CspTy n → CspCtx n → Prop

  -- ── Core rules (Level 1 — identical mechanism to GPU) ──

  | groupVal (ctx : CspCtx n) (s : PSet n) :
      CspHasType topo ctx (.groupVal s) (.group s) ctx
  | dataVal (ctx : CspCtx n) :
      CspHasType topo ctx .dataVal .data ctx
  | unitVal (ctx : CspCtx n) :
      CspHasType topo ctx .unitVal .unit ctx
  | var (ctx : CspCtx n) (name : String) (t : CspTy n) :
      ctx.lookup name = some t →
      CspHasType topo ctx (.var name) t (ctx.remove name)
  | diverge (ctx ctx' : CspCtx n) (g : CspExpr n) (s pred : PSet n) :
      CspHasType topo ctx g (.group s) ctx' →
      CspHasType topo ctx (.diverge g pred)
        (.pair (.group (s &&& pred)) (.group (s &&& ~~~pred))) ctx'
  | merge (ctx ctx' ctx'' : CspCtx n) (g1 g2 : CspExpr n)
      (s1 s2 parent : PSet n) :
      CspHasType topo ctx g1 (.group s1) ctx' →
      CspHasType topo ctx' g2 (.group s2) ctx'' →
      PSet.IsComplement s1 s2 parent →          -- THE gate (same as GPU)
      CspHasType topo ctx (.merge g1 g2) (.group parent) ctx''
  | letBind (ctx ctx' ctx'' : CspCtx n) (name : String)
      (val body : CspExpr n) (t1 t2 : CspTy n) :
      CspHasType topo ctx val t1 ctx' →
      ctx'.lookup name = none →
      CspHasType topo ((name, t1) :: ctx') body t2 ctx'' →
      ctx''.lookup name = none →
      CspHasType topo ctx (.letBind name val body) t2 ctx''
  | pairVal (ctx ctx' ctx'' : CspCtx n) (a b : CspExpr n) (t1 t2 : CspTy n) :
      CspHasType topo ctx a t1 ctx' →
      CspHasType topo ctx' b t2 ctx'' →
      CspHasType topo ctx (.pairVal a b) (.pair t1 t2) ctx''
  | fstE (ctx ctx' : CspCtx n) (e : CspExpr n) (t1 t2 : CspTy n) :
      CspHasType topo ctx e (.pair t1 t2) ctx' →
      CspHasType topo ctx (.fst e) t1 ctx'
  | sndE (ctx ctx' : CspCtx n) (e : CspExpr n) (t1 t2 : CspTy n) :
      CspHasType topo ctx e (.pair t1 t2) ctx' →
      CspHasType topo ctx (.snd e) t2 ctx'
  | letPairE (ctx ctx' ctx'' : CspCtx n) (e : CspExpr n) (name1 name2 : String)
      (body : CspExpr n) (t1 t2 t : CspTy n) :
      CspHasType topo ctx e (.pair t1 t2) ctx' →
      name1 ≠ name2 →
      ctx'.lookup name1 = none →
      ctx'.lookup name2 = none →
      CspHasType topo ((name2, t2) :: (name1, t1) :: ctx') body t ctx'' →
      ctx''.lookup name1 = none →
      ctx''.lookup name2 = none →
      CspHasType topo ctx (.letPair e name1 name2 body) t ctx''

  -- ── CSP-specific rules (Level 2b) ──

  /-- Send: point-to-point message to an adjacent, active participant.
      Consumes group and payload, returns group (threading the handle). -/
  | send (ctx ctx' ctx'' : CspCtx n) (g payload : CspExpr n)
      (self dst : Fin n) (s : PSet n) :
      CspHasType topo ctx g (.group s) ctx' →
      s.getLsbD dst.val = true →            -- destination is ACTIVE
      topo.adj self dst = true →            -- destination is ADJACENT
      CspHasType topo ctx' payload .data ctx'' →
      CspHasType topo ctx (.send g payload self dst) (.group s) ctx''

  /-- Recv: point-to-point receive from an adjacent, active participant.
      Consumes group, returns pair of (data, group). -/
  | recv (ctx ctx' : CspCtx n) (g : CspExpr n)
      (self src : Fin n) (s : PSet n) :
      CspHasType topo ctx g (.group s) ctx' →
      s.getLsbD src.val = true →            -- source is ACTIVE
      topo.adj self src = true →            -- source is ADJACENT
      CspHasType topo ctx (.recv g self src) (.pair .data (.group s)) ctx'

  /-- Collective: all-to-all operation requiring ALL participants.
      Same gate as GPU shuffle: requires Group<All>. -/
  | collective (ctx ctx' ctx'' : CspCtx n) (g payload : CspExpr n) :
      CspHasType topo ctx g (.group (PSet.all n)) ctx' →
      CspHasType topo ctx' payload .data ctx'' →
      CspHasType topo ctx (.collective g payload) .data ctx''

-- ============================================================================
-- Theorem: CSP diverge partition (delegates to generic)
-- ============================================================================

/-- The diverge partition theorem transfers to CSP unchanged.
    This IS the homomorphism: same theorem, same proof, different domain. -/
theorem csp_diverge_partition {n : Nat} (s pred : PSet n) :
    PSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    PSet.Covers (s &&& pred) (s &&& ~~~pred) s :=
  diverge_partition_generic s pred

-- ============================================================================
-- Theorem: Collective requires all participants (parallel to shuffle)
-- ============================================================================

/-- Collective typing requires the group to contain ALL participants.
    Structurally identical to shuffle_requires_all in the GPU domain. -/
theorem csp_collective_requires_all {n : Nat} {topo : Topology n}
    {ctx ctx'' : CspCtx n} {g payload : CspExpr n} {t : CspTy n} :
    CspHasType topo ctx (.collective g payload) t ctx'' →
    ∃ ctx', CspHasType topo ctx g (.group (PSet.all n)) ctx' := by
  intro h
  cases h with
  | collective _ ctx' _ _ _ hg _ => exact ⟨ctx', hg⟩

-- ============================================================================
-- Theorem: Send requires adjacency (NEW — no GPU analog)
-- ============================================================================

/-- Send is well-typed only if the destination is adjacent in the topology.
    This constraint has no GPU analog — it's purely a CSP/mesh property. -/
theorem csp_send_requires_adjacent {n : Nat} {topo : Topology n}
    {ctx ctx'' : CspCtx n} {g payload : CspExpr n}
    {self dst : Fin n} {t : CspTy n} :
    CspHasType topo ctx (.send g payload self dst) t ctx'' →
    topo.adj self dst = true := by
  intro h
  cases h with
  | send _ _ _ _ _ _ _ _ _ _ hadj _ => exact hadj

-- ============================================================================
-- Theorem: Send requires active destination
-- ============================================================================

theorem csp_send_requires_active {n : Nat} {topo : Topology n}
    {ctx ctx'' : CspCtx n} {g payload : CspExpr n}
    {self dst : Fin n} {t : CspTy n} :
    CspHasType topo ctx (.send g payload self dst) t ctx'' →
    ∃ s : PSet n, s.getLsbD dst.val = true := by
  intro h
  cases h with
  | send _ _ _ _ _ _ _ s _ hactive _ _ => exact ⟨s, hactive⟩

-- ============================================================================
-- Untypability: collective after diverge (CSP analog of GPU bugs 1-5)
-- ============================================================================

/-- Helper: fst of diverge on groupVal always produces a sub-group. -/
private theorem csp_fst_diverge_groupval_type {n : Nat} {topo : Topology n}
    {s pred : PSet n} {t : CspTy n} {ctx' : CspCtx n}
    (ht : CspHasType topo [] (.fst (.diverge (.groupVal s) pred)) t ctx') :
    t = .group (s &&& pred) := by
  cases ht with
  | fstE _ _ _ _ _ he =>
    cases he with
    | diverge _ _ _ _ _ hg =>
      cases hg with
      | groupVal _ _ => rfl

/-- Collective on a diverged sub-group is untypable when the sub-group ≠ All.

    This is the CSP analog of shuffle_diverged_untypable (GPU bugs 1-5).
    Same mechanism, same proof structure, different domain. -/
theorem collective_after_diverge_untypable {n : Nat} {topo : Topology n}
    (s pred : PSet n)
    (hne : s &&& pred ≠ PSet.all n) :
    ¬ ∃ t ctx', CspHasType topo []
      (.collective (.fst (.diverge (.groupVal s) pred)) .dataVal)
      t ctx' := by
  intro ⟨t, ctx', ht⟩
  have ⟨ctx_mid, hg⟩ := csp_collective_requires_all ht
  have heq := csp_fst_diverge_groupval_type hg
  simp only [CspTy.group.injEq] at heq
  exact absurd heq.symm hne

-- ============================================================================
-- Untypability: send to non-adjacent core (NEW — no GPU analog)
-- ============================================================================

/-- Sending from core 0 to core 5 on the J1 grid is untypable.
    These cores are opposite corners (3 hops apart, not adjacent). -/
theorem j1_send_opposite_corners_untypable :
    ¬ ∃ t ctx', CspHasType j1Grid []
      (.send (.groupVal TileSet.all) .dataVal ⟨0, by omega⟩ ⟨5, by omega⟩)
      t ctx' := by
  intro ⟨t, ctx', ht⟩
  have hadj := csp_send_requires_adjacent ht
  simp [j1Grid, j1GridAdj] at hadj

/-- Sending from core 0 to core 3 on the J1 grid is untypable.
    These cores are diagonal (2 hops apart, not adjacent). -/
theorem j1_send_diagonal_untypable :
    ¬ ∃ t ctx', CspHasType j1Grid []
      (.send (.groupVal TileSet.all) .dataVal ⟨0, by omega⟩ ⟨3, by omega⟩)
      t ctx' := by
  intro ⟨t, ctx', ht⟩
  have hadj := csp_send_requires_adjacent ht
  simp [j1Grid, j1GridAdj] at hadj

-- ============================================================================
-- Concrete J1 instance: collective after diverge
-- ============================================================================

/-- Collective after splitting 6 cores into left/right columns is untypable.
    Parallel to bug1_cuda_samples_398 etc. in the GPU domain. -/
theorem j1_collective_after_column_split :
    ¬ ∃ t ctx', CspHasType j1Grid []
      (.collective
        (.fst (.diverge (.groupVal TileSet.all) TileSet.leftCol))
        .dataVal)
      t ctx' :=
  collective_after_diverge_untypable TileSet.all TileSet.leftCol (by decide)

-- ============================================================================
-- Positive instance: send between adjacent cores IS typable
-- ============================================================================

/-- Core 0 sending to adjacent core 1 is well-typed.
    Demonstrates the type system accepts valid communication patterns. -/
theorem j1_send_adjacent_typable :
    ∃ t ctx', CspHasType j1Grid ([] : CspCtx 6)
      (.send (.groupVal TileSet.all) .dataVal ⟨0, by omega⟩ ⟨1, by omega⟩)
      t ctx' := by
  exact ⟨.group TileSet.all, [],
    CspHasType.send [] [] [] _ _ ⟨0, by omega⟩ ⟨1, by omega⟩ TileSet.all
      (CspHasType.groupVal [] TileSet.all)
      (by decide)  -- bit 1 of TileSet.all is true
      (by decide)  -- j1Grid.adj 0 1 is true
      (CspHasType.dataVal [])⟩
