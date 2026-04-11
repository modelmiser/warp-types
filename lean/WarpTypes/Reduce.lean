import WarpTypes.Core
import WarpTypes.CoreMetatheory

/-
  Tree All-Reduce Domain Extension (Level 2d — experiment C, post-port)

  Ported 2026-04-11 onto `WarpTypes.Core`. This file formerly defined
  ReduceTy / ReduceExpr / ReduceCtx / ReduceHasType as a self-contained
  inductive stack (302 lines). Post-port it is a thin domain view over
  `CoreHasType`: concrete instance (Col = PSet 4), four-theorem stack
  (diverge partition delegate, `finalize_requires_all` inversion, negative
  partial-reduce instance, positive full-tree-reduce instance), and the
  nested helper.

  The merge-shaped rule that used to be called `combineRed` is now the
  specialization of `CoreHasType.mergeFamily` at tag `.reduced`; the
  extract-shaped rule that used to be called `finalize` is now
  `CoreHasType.finalizeFamily` at tag `.reduced`. The `.leafReduce` rule
  is a monomorphic cross-family coercion (`.group s → .reduced s`) that
  lives directly in `CoreHasType`.

  Inversion proofs at tag `.reduced` are mirror images of the Fence tag
  `.group` proofs: the live branch becomes the `.reduced` case; the
  `.group` case is dead and closes via constructor-clash contradiction
  on `hExpr : .finalize e = tagToFinalExpr .group e'` which reduces to
  `.finalize e = .fence e'`.
-/

-- ============================================================================
-- Col: PSet 4 instantiation (smallest halving tree with non-trivial middle)
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
    Fourth structural analog of leftCol_rightCol (CSP), even_odd (GPU),
    and nibble_complement (Fence). -/
theorem halfway_complement :
    PSet.IsComplementAll Col.lowHalf Col.highHalf := by
  unfold PSet.IsComplementAll PSet.IsComplement
  unfold PSet.Disjoint PSet.Covers
  unfold Col.lowHalf Col.highHalf PSet.all PSet.none
  constructor <;> decide

-- ============================================================================
-- Theorem: reduce diverge partition (delegates to Core)
-- ============================================================================

theorem reduce_diverge_partition {n : Nat} (s pred : PSet n) :
    PSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    PSet.Covers (s &&& pred) (s &&& ~~~pred) s :=
  core_diverge_partition s pred

-- ============================================================================
-- Theorem: finalize_requires_all — inversion of finalizeFamily at tag .reduced
-- ============================================================================

/-- `finalize` requires the accumulator to span ALL participants.

    Mirror of `fence_requires_all`: the live branch is `finalizeFamily` at
    tag `.reduced`; the `.group` case is dead (its `tagToFinalExpr` maps to
    `.fence`, which clashes with `.finalize r`); `mergeFamily` at both tags
    is also dead (its `tagToMergeExpr` maps to `.merge` / `.combineRed`,
    both clashing with `.finalize r`). -/
theorem finalize_requires_all {n : Nat}
    {ctx ctx' : CoreCtx n} {r : CoreExpr n} :
    CoreHasType ctx (.finalize r) .data ctx' →
    CoreHasType ctx r (.reduced (PSet.all n)) ctx' := by
  intro h
  cases h with
  | finalizeFamily tag _ _ e' expr resultTy hExpr hTy hg =>
    cases tag with
    | group =>
      simp only [tagToFinalExpr] at hExpr
      cases hExpr
    | reduced =>
      simp only [tagToFinalExpr] at hExpr
      simp only [tagToTy] at hg
      injection hExpr with hExpr'
      subst hExpr'
      exact hg
  | mergeFamily tag _ _ _ _ _ expr _ _ _ _ hExpr _ _ _ _ =>
    cases tag with
    | group =>
      simp only [tagToMergeExpr] at hExpr
      cases hExpr
    | reduced =>
      simp only [tagToMergeExpr] at hExpr
      cases hExpr

-- ============================================================================
-- Helper: leafReduce of (fst diverge groupVal) produces a masked accumulator
-- ============================================================================

/-- Walks `leafReduce (fst (diverge (groupVal s) pred))` through four levels
    of nested `cases`, each of which needs the same mergeFamily/finalizeFamily
    dead-branch discharges as Fence's helper (see `fence_fst_diverge_groupval_type`). -/
private theorem reduce_leaf_fst_diverge_type {n : Nat}
    {s pred : PSet n} {t : CoreTy n} {ctx' : CoreCtx n}
    (ht : CoreHasType []
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
        | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
          cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
        | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
          cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
      | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
        cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
      | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
        cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
    cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
  | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
    cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr

-- ============================================================================
-- NEGATIVE instance: finalize after partial-group leafReduce is untypable
-- ============================================================================

/-- Finalizing after a leaf reduction on a proper sub-group is untypable.
    Fourth-domain analog of `fence_after_partial_write_untypable`. -/
theorem finalize_after_partial_reduce_untypable {n : Nat}
    (s pred : PSet n)
    (hne : s &&& pred ≠ PSet.all n) :
    ¬ ∃ ctx', CoreHasType []
      (.finalize (.leafReduce (.fst (.diverge (.groupVal s) pred))))
      .data ctx' := by
  intro ⟨ctx', ht⟩
  have hr := finalize_requires_all ht
  have heq := reduce_leaf_fst_diverge_type hr
  simp only [CoreTy.reduced.injEq] at heq
  exact absurd heq.symm hne

/-- Concrete Col instance: finalizing after reducing only the low half is
    untypable. Parallel to `bytebuf_fence_after_low_nibble_only`. -/
theorem col_finalize_after_low_half_only :
    ¬ ∃ ctx', CoreHasType []
      (.finalize
        (.leafReduce (.fst (.diverge (.groupVal Col.all) Col.lowHalf))))
      .data ctx' :=
  finalize_after_partial_reduce_untypable Col.all Col.lowHalf (by decide)

-- ============================================================================
-- POSITIVE instance: full tree reduction is typable
-- ============================================================================

/-- Leaf-reducing the low half, leaf-reducing the high half, combining the
    accumulators via `IsComplement`, and finalizing is well-typed.

    Exercises `mergeFamily` at tag `.reduced` and `finalizeFamily` at tag
    `.reduced` — the companion to `fence_after_full_write_typable`. The gate
    (`PSet.all n`) now lives on `.reduced` rather than `.group`; the proof
    structure is otherwise identical modulo tag. -/
theorem finalize_tree_reduce_typable :
    ∃ ctx', CoreHasType ([] : CoreCtx 4)
      (.finalize
        (.combineRed
          (.leafReduce (.groupVal Col.lowHalf))
          (.leafReduce (.groupVal Col.highHalf))))
      .data ctx' := by
  refine ⟨[], ?_⟩
  refine CoreHasType.finalizeFamily (n := 4) .reduced [] []
    (.combineRed (.leafReduce (.groupVal Col.lowHalf))
                 (.leafReduce (.groupVal Col.highHalf)))
    _ _ rfl rfl ?_
  refine CoreHasType.mergeFamily (n := 4) .reduced [] [] []
    (.leafReduce (.groupVal Col.lowHalf))
    (.leafReduce (.groupVal Col.highHalf))
    _ Col.lowHalf Col.highHalf (PSet.all 4) _ rfl rfl
    ?_ ?_ halfway_complement
  · exact CoreHasType.leafReduce [] [] _ Col.lowHalf
      (CoreHasType.groupVal [] Col.lowHalf)
  · exact CoreHasType.leafReduce [] [] _ Col.highHalf
      (CoreHasType.groupVal [] Col.highHalf)

-- ============================================================================
-- Metatheory corollaries at Col width (n = 4)
-- ============================================================================
-- Thin specialisations of `CoreMetatheory` theorems at `n = 4`. As with
-- Fence's corollaries, each theorem is a one-liner over the Core version —
-- no Reduce-specific proof content.

/-- Progress for Reduce at Col width. -/
theorem reduce_progress {e : CoreExpr 4} {t : CoreTy 4} {ctx' : CoreCtx 4}
    (ht : CoreHasType [] e t ctx') :
    CoreMetatheory.isValue e = true ∨ ∃ e', CoreMetatheory.Step e e' :=
  CoreMetatheory.progress_closed ht

/-- Preservation for Reduce at Col width. -/
theorem reduce_preservation {e e' : CoreExpr 4} {t : CoreTy 4} {ctx ctx' : CoreCtx 4}
    (ht : CoreHasType ctx e t ctx') (hs : CoreMetatheory.Step e e') :
    CoreHasType ctx e' t ctx' :=
  CoreMetatheory.preservation ht hs

/-- Multi-step type safety for Reduce at Col width. -/
theorem reduce_type_safety {e e' : CoreExpr 4} {t : CoreTy 4} {ctx' : CoreCtx 4}
    (ht : CoreHasType [] e t ctx') (hstar : CoreMetatheory.Star CoreMetatheory.Step e e') :
    CoreMetatheory.isValue e' = true ∨ ∃ e'', CoreMetatheory.Step e' e'' :=
  CoreMetatheory.type_safety ht hstar
