import WarpTypes.Generic

/-
  Tree All-Reduce Domain Extension (Level 2d — experiment C)

  Fourth instance of the complemented typestate framework, testing the
  INSIGHTS.md claim that "the gate is orthogonal to op data-flow shape."

  CSP, Fence, and GPU shuffle all have barriers whose input is a group
  handle (`group (PSet.all n)` = the gate). Tree all-reduce is different:
  its barrier (`finalize`) consumes an *accumulator*, not a group handle.
  The accumulator is the result of folding data values up a tree whose
  leaves are per-participant reductions.

  The question this experiment tests:
  >  Can a recursive accumulator be type-checked using only the existing
  >  `merge`/`diverge`/`letPair` core rules — or do we need a new structural
  >  rule because `data` is opaque (has no participant-set index)?

  Design decision: we introduce a new type constructor `reduced (s : PSet n)`
  that carries the source-group index as part of the type. This is a
  structural extension to the Level 1 type family. `leafReduce : group s →
  reduced s`, `combineRed : reduced s1 → reduced s2 → reduced (s1 ⊔ s2)`
  (requires `IsComplement`), `finalize : reduced (PSet.all n) → data`.

  PREDICTION BEFORE WRITING (recorded here for falsifiability):
  - F3-style outcome: the experiment succeeds but reveals that `combineRed`
    is a *structural clone* of the core `merge` rule, acting on a different
    type family. This is the first measurable sign that the core `merge`
    rule is monomorphic in `.group` where it should be polymorphic over any
    `PSet n`-indexed type family.
  - Refactor implication: a shared `Core.lean` should expose `merge` as a
    rule parameterized by the `PSet n`-indexed family, not hardcoded to
    `.group`.
-/

-- ============================================================================
-- Col: PSet 4 instantiation (smallest tree with depth 2)
-- ============================================================================

/-- A column of 4 participants. Smallest size where a halving tree reduce
    has a non-trivial intermediate step. -/
abbrev Col := PSet 4

namespace Col

def all : Col := PSet.all 4
def none : Col := PSet.none 4
def lowHalf : Col := 0x3#4   -- participants 0, 1
def highHalf : Col := 0xC#4  -- participants 2, 3

end Col

/-- Low and high halves of a 4-participant column are complements within All.
    Third structural analog of leftCol_rightCol (CSP), even_odd (GPU),
    and nibble_complement (Fence). -/
theorem halfway_complement :
    PSet.IsComplementAll Col.lowHalf Col.highHalf := by
  unfold PSet.IsComplementAll PSet.IsComplement
  unfold PSet.Disjoint PSet.Covers
  unfold Col.lowHalf Col.highHalf PSet.all PSet.none
  constructor <;> decide

-- ============================================================================
-- Reduce Types (Level 2d — adds `reduced (s : PSet n)` constructor)
-- ============================================================================

/-- The `reduced` type carries the source-group index. This is the new
    structural extension: previous domains used opaque `data` for fold
    results, which cannot express "this was produced by the full group." -/
inductive ReduceTy (n : Nat)
  | group (s : PSet n)       -- write permission / participant handle
  | data                      -- opaque per-participant value
  | unit                      -- barrier result
  | pair (a b : ReduceTy n)
  | reduced (s : PSet n)     -- NEW: accumulator indexed by source group

-- ============================================================================
-- Reduce Expressions
-- ============================================================================

inductive ReduceExpr (n : Nat)
  -- Core (Level 1 — copy-rename from CspExpr / FenceExpr)
  | groupVal (s : PSet n)
  | dataVal
  | unitVal
  | var (name : String)
  | diverge (g : ReduceExpr n) (pred : PSet n)
  | merge (g1 g2 : ReduceExpr n)
  | letBind (name : String) (val body : ReduceExpr n)
  | pairVal (a b : ReduceExpr n)
  | fst (e : ReduceExpr n)
  | snd (e : ReduceExpr n)
  | letPair (e : ReduceExpr n) (name1 name2 : String) (body : ReduceExpr n)
  -- Level 2d: tree all-reduce specific
  | leafReduce (g : ReduceExpr n)        -- group s → reduced s
  | combineRed (r1 r2 : ReduceExpr n)    -- reduced s1 + reduced s2 → reduced (s1 ⊔ s2)
  | finalize (r : ReduceExpr n)           -- reduced (PSet.all n) → data

-- ============================================================================
-- Reduce Context (linear)
-- ============================================================================

def ReduceCtx (n : Nat) := List (String × ReduceTy n)

namespace ReduceCtx

def lookup {n : Nat} (ctx : ReduceCtx n) (name : String) : Option (ReduceTy n) :=
  ctx.find? (fun p => p.1 == name) |>.map Prod.snd

def remove {n : Nat} (ctx : ReduceCtx n) (name : String) : ReduceCtx n :=
  ctx.filter (fun p => p.1 != name)

end ReduceCtx

-- ============================================================================
-- Reduce Typing Rules
-- ============================================================================

inductive ReduceHasType {n : Nat} :
    ReduceCtx n → ReduceExpr n → ReduceTy n → ReduceCtx n → Prop

  -- ── Core rules (copy-rename from CspHasType / FenceHasType) ──

  | groupVal (ctx : ReduceCtx n) (s : PSet n) :
      ReduceHasType ctx (.groupVal s) (.group s) ctx
  | dataVal (ctx : ReduceCtx n) :
      ReduceHasType ctx .dataVal .data ctx
  | unitVal (ctx : ReduceCtx n) :
      ReduceHasType ctx .unitVal .unit ctx
  | var (ctx : ReduceCtx n) (name : String) (t : ReduceTy n) :
      ctx.lookup name = some t →
      ReduceHasType ctx (.var name) t (ctx.remove name)
  | diverge (ctx ctx' : ReduceCtx n) (g : ReduceExpr n) (s pred : PSet n) :
      ReduceHasType ctx g (.group s) ctx' →
      ReduceHasType ctx (.diverge g pred)
        (.pair (.group (s &&& pred)) (.group (s &&& ~~~pred))) ctx'
  | merge (ctx ctx' ctx'' : ReduceCtx n) (g1 g2 : ReduceExpr n)
      (s1 s2 parent : PSet n) :
      ReduceHasType ctx g1 (.group s1) ctx' →
      ReduceHasType ctx' g2 (.group s2) ctx'' →
      PSet.IsComplement s1 s2 parent →
      ReduceHasType ctx (.merge g1 g2) (.group parent) ctx''
  | letBind (ctx ctx' ctx'' : ReduceCtx n) (name : String)
      (val body : ReduceExpr n) (t1 t2 : ReduceTy n) :
      ReduceHasType ctx val t1 ctx' →
      ctx'.lookup name = none →
      ReduceHasType ((name, t1) :: ctx') body t2 ctx'' →
      ctx''.lookup name = none →
      ReduceHasType ctx (.letBind name val body) t2 ctx''
  | pairVal (ctx ctx' ctx'' : ReduceCtx n) (a b : ReduceExpr n) (t1 t2 : ReduceTy n) :
      ReduceHasType ctx a t1 ctx' →
      ReduceHasType ctx' b t2 ctx'' →
      ReduceHasType ctx (.pairVal a b) (.pair t1 t2) ctx''
  | fstE (ctx ctx' : ReduceCtx n) (e : ReduceExpr n) (t1 t2 : ReduceTy n) :
      ReduceHasType ctx e (.pair t1 t2) ctx' →
      ReduceHasType ctx (.fst e) t1 ctx'
  | sndE (ctx ctx' : ReduceCtx n) (e : ReduceExpr n) (t1 t2 : ReduceTy n) :
      ReduceHasType ctx e (.pair t1 t2) ctx' →
      ReduceHasType ctx (.snd e) t2 ctx'
  | letPairE (ctx ctx' ctx'' : ReduceCtx n) (e : ReduceExpr n) (name1 name2 : String)
      (body : ReduceExpr n) (t1 t2 t : ReduceTy n) :
      ReduceHasType ctx e (.pair t1 t2) ctx' →
      name1 ≠ name2 →
      ctx'.lookup name1 = none →
      ctx'.lookup name2 = none →
      ReduceHasType ((name2, t2) :: (name1, t1) :: ctx') body t ctx'' →
      ctx''.lookup name1 = none →
      ctx''.lookup name2 = none →
      ReduceHasType ctx (.letPair e name1 name2 body) t ctx''

  -- ── Tree all-reduce rules (Level 2d) ──

  /-- Leaf reduction: turn a group handle into an accumulator at the same
      participant set. The resulting `reduced s` tracks the source group
      in the type — this is the whole point of the new type constructor. -/
  | leafReduce (ctx ctx' : ReduceCtx n) (g : ReduceExpr n) (s : PSet n) :
      ReduceHasType ctx g (.group s) ctx' →
      ReduceHasType ctx (.leafReduce g) (.reduced s) ctx'

  /-- Combine two accumulators. THIS IS THE STRUCTURAL TWIN of `merge`.
      Same context threading, same arity, same `IsComplement` gate — but
      acting on `.reduced` instead of `.group`. The duplication is the
      experiment's F3-class finding: `merge` should be polymorphic over
      any `PSet n`-indexed type family, not hardcoded to `.group`. -/
  | combineRed (ctx ctx' ctx'' : ReduceCtx n) (r1 r2 : ReduceExpr n)
      (s1 s2 parent : PSet n) :
      ReduceHasType ctx r1 (.reduced s1) ctx' →
      ReduceHasType ctx' r2 (.reduced s2) ctx'' →
      PSet.IsComplement s1 s2 parent →          -- EXACT same gate as merge
      ReduceHasType ctx (.combineRed r1 r2) (.reduced parent) ctx''

  /-- Finalize: extract the data value from a full-group accumulator.
      Gates on `reduced (PSet.all n)` — structurally parallel to how
      `fence` gates on `group (PSet.all n)`. The gate moved from the
      `.group` family to the `.reduced` family; otherwise unchanged. -/
  | finalize (ctx ctx' : ReduceCtx n) (r : ReduceExpr n) :
      ReduceHasType ctx r (.reduced (PSet.all n)) ctx' →
      ReduceHasType ctx (.finalize r) .data ctx'

-- ============================================================================
-- Theorem: reduce diverge partition (delegates to generic — untouched)
-- ============================================================================

theorem reduce_diverge_partition {n : Nat} (s pred : PSet n) :
    PSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    PSet.Covers (s &&& pred) (s &&& ~~~pred) s :=
  diverge_partition_generic s pred

-- ============================================================================
-- Theorem: finalize requires full-group accumulator
-- ============================================================================

/-- Finalize typing requires the accumulator to span ALL participants.
    Parallel to fence_requires_all and csp_collective_requires_all,
    but the gate lives on `.reduced` instead of `.group`. -/
theorem finalize_requires_all {n : Nat}
    {ctx ctx' : ReduceCtx n} {r : ReduceExpr n} :
    ReduceHasType ctx (.finalize r) .data ctx' →
    ReduceHasType ctx r (.reduced (PSet.all n)) ctx' := by
  intro h
  cases h with
  | finalize _ _ _ hr => exact hr

-- ============================================================================
-- Helper: leafReduce of (fst diverge groupVal) produces a masked accumulator
-- ============================================================================

private theorem reduce_leaf_fst_diverge_type {n : Nat}
    {s pred : PSet n} {t : ReduceTy n} {ctx' : ReduceCtx n}
    (ht : ReduceHasType []
      (.leafReduce (.fst (.diverge (.groupVal s) pred))) t ctx') :
    t = .reduced (s &&& pred) := by
  cases ht with
  | leafReduce _ _ _ _ hg =>
    cases hg with
    | fstE _ _ _ _ _ he =>
      cases he with
      | diverge _ _ _ _ _ hgv =>
        cases hgv with
        | groupVal _ _ => rfl

-- ============================================================================
-- NEGATIVE instance: finalize after partial-group leafReduce is untypable
-- ============================================================================

/-- Finalizing after a leaf reduction on a proper sub-group is untypable.

    Fourth-domain analog of `fence_after_partial_write_untypable`,
    `collective_after_diverge_untypable`, and `shuffle_diverged_untypable`.

    The proof structure is IDENTICAL to the fence case modulo s/fence/finalize
    and s/group/reduced — further evidence that the core mechanism transfers
    unchanged. -/
theorem finalize_after_partial_reduce_untypable {n : Nat}
    (s pred : PSet n)
    (hne : s &&& pred ≠ PSet.all n) :
    ¬ ∃ ctx', ReduceHasType []
      (.finalize (.leafReduce (.fst (.diverge (.groupVal s) pred))))
      .data ctx' := by
  intro ⟨ctx', ht⟩
  have hr := finalize_requires_all ht
  have heq := reduce_leaf_fst_diverge_type hr
  simp only [ReduceTy.reduced.injEq] at heq
  exact absurd heq.symm hne

/-- Concrete Col instance: finalizing after reducing only the low half
    is untypable. Parallel to `bytebuf_fence_after_low_nibble_only`. -/
theorem col_finalize_after_low_half_only :
    ¬ ∃ ctx', ReduceHasType []
      (.finalize
        (.leafReduce (.fst (.diverge (.groupVal Col.all) Col.lowHalf))))
      .data ctx' :=
  finalize_after_partial_reduce_untypable Col.all Col.lowHalf (by decide)

-- ============================================================================
-- POSITIVE instance: full tree reduction is typable
-- ============================================================================

/-- Leaf-reducing the low half, leaf-reducing the high half, combining the
    accumulators via IsComplement, and finalizing is well-typed.

    This is the first positive theorem in the framework where the gate
    (`PSet.all n`) lives on the `.reduced` type family rather than `.group`.
    The proof structure is the same as `fence_after_full_write_typable` —
    which is exactly the duplication the experiment was designed to surface. -/
theorem finalize_tree_reduce_typable :
    ∃ ctx', ReduceHasType ([] : ReduceCtx 4)
      (.finalize
        (.combineRed
          (.leafReduce (.groupVal Col.lowHalf))
          (.leafReduce (.groupVal Col.highHalf))))
      .data ctx' := by
  refine ⟨[], ?_⟩
  apply ReduceHasType.finalize
  exact ReduceHasType.combineRed [] [] [] _ _
    Col.lowHalf Col.highHalf (PSet.all 4)
    (ReduceHasType.leafReduce [] [] _ Col.lowHalf
      (ReduceHasType.groupVal [] Col.lowHalf))
    (ReduceHasType.leafReduce [] [] _ Col.highHalf
      (ReduceHasType.groupVal [] Col.highHalf))
    halfway_complement
