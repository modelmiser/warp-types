import WarpTypes.Core
import WarpTypes.CoreMetatheory

/-
  Fence / Partial-Write Domain Extension (Level 2c — experiment B, post-port)

  Ported 2026-04-11 onto `WarpTypes.Core`. This file formerly defined
  FenceTy / FenceExpr / FenceCtx / FenceHasType as a self-contained inductive
  stack (271 lines). Post-port, it is a thin domain view over `CoreHasType`:
  concrete instance (ByteBuf = PSet 8), the four theorem stack (diverge
  partition delegate, `fence_requires_all` inversion, negative partial-write
  instance, positive full-write instance), and two helpers.

  The ~87 lines of core typing rules previously duplicated between this file
  and Csp.lean / Reduce.lean now live in Core.lean as monomorphic + parametric
  rules. Fence's `merge` and `fence` are subsumed by `CoreHasType.mergeFamily`
  and `CoreHasType.finalizeFamily` respectively, both at tag `.group`.

  Inversion proofs that previously used a single-case `cases` on the concrete
  rule now case on the parametric rule and discriminate by tag, closing dead
  branches (the `.reduced` tag and the cross-rule `mergeFamily` branch) via
  constructor-clash contradiction. This is the standard probe-3b pattern.

  Domain interpretation (unchanged):
  - PSet n   : write mask over an n-byte buffer (bit i ↔ byte i)
  - .group s : linear permission to write exactly bytes in s
  - diverge  : split a permission by predicate
  - merge    : combine complementary sub-permissions (THE gate)
  - write    : bulk write payload under permission; threads permission
  - fence    : barrier; requires full-buffer permission (gate on `.group (PSet.all n)`)
-/

-- ============================================================================
-- ByteBuf: PSet 8 instantiation
-- ============================================================================

/-- An 8-bit write mask: one bit per byte of a word-sized buffer. -/
abbrev ByteBuf := PSet 8

namespace ByteBuf

def all : ByteBuf := PSet.all 8
def none : ByteBuf := PSet.none 8
def lowNibble : ByteBuf := 0x0F#8   -- bytes 0-3
def highNibble : ByteBuf := 0xF0#8  -- bytes 4-7

end ByteBuf

/-- Low and high nibbles are complements within All. Structural analog of
    `leftCol_rightCol_complement` (CSP), `even_odd_complement` (GPU), and
    `halfway_complement` (Reduce). -/
theorem nibble_complement :
    PSet.IsComplementAll ByteBuf.lowNibble ByteBuf.highNibble := by
  unfold PSet.IsComplementAll PSet.IsComplement
  unfold PSet.Disjoint PSet.Covers
  unfold ByteBuf.lowNibble ByteBuf.highNibble PSet.all PSet.none
  constructor <;> decide

-- ============================================================================
-- Theorem: fence diverge partition (delegates to Core, which delegates to Generic)
-- ============================================================================

/-- Third instance of the diverge partition homomorphism. -/
theorem fence_diverge_partition {n : Nat} (s pred : PSet n) :
    PSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    PSet.Covers (s &&& pred) (s &&& ~~~pred) s :=
  core_diverge_partition s pred

-- ============================================================================
-- Theorem: fence_requires_all — inversion of finalizeFamily at tag .group
-- ============================================================================

/-- `fence` requires the group to contain ALL bytes.

    Proof structure (probe 3b, double-witness variant):
    `cases` generates a branch for every `CoreHasType` constructor whose
    conclusion can unify with `CoreHasType ctx (.fence g) .unit ctx'`. The
    monomorphic constructors are all eliminated by constructor clash at the
    unifier level. The two parametric rules (`mergeFamily`, `finalizeFamily`)
    survive `cases` because their conclusion `expr` and `ty` are free pattern
    variables. Inside each surviving branch, case on `tag` and discharge:
    - `finalizeFamily .group`: live branch — extract the premise `hg`.
    - `finalizeFamily .reduced`: dead — `tagToFinalExpr .reduced e' = .finalize e'`
      clashes with `.fence g`.
    - `mergeFamily .group`: dead — `tagToMergeExpr .group e1 e2 = .merge e1 e2`
      clashes with `.fence g`.
    - `mergeFamily .reduced`: dead — `.combineRed e1 e2` clashes with `.fence g`. -/
theorem fence_requires_all {n : Nat}
    {ctx ctx' : CoreCtx n} {g : CoreExpr n} :
    CoreHasType ctx (.fence g) .unit ctx' →
    CoreHasType ctx g (.group (PSet.all n)) ctx' := by
  intro h
  cases h with
  | finalizeFamily tag _ _ e' expr resultTy hExpr hTy hg =>
    cases tag with
    | group =>
      simp only [tagToFinalExpr] at hExpr
      simp only [tagToTy] at hg
      injection hExpr with hExpr'
      subst hExpr'
      exact hg
    | reduced =>
      simp only [tagToFinalExpr] at hExpr
      cases hExpr
  | mergeFamily tag _ _ _ _ _ expr _ _ _ _ hExpr _ _ _ _ =>
    cases tag with
    | group =>
      simp only [tagToMergeExpr] at hExpr
      cases hExpr
    | reduced =>
      simp only [tagToMergeExpr] at hExpr
      cases hExpr

-- ============================================================================
-- Helper: fst of diverge on groupVal produces the masked sub-group
-- ============================================================================

/-- Walks a nested expression `fst (diverge (groupVal s) pred)` through the
    `fstE`, `diverge`, and `groupVal` rules. Each `cases` level now has to
    explicitly discharge `mergeFamily` and `finalizeFamily` branches — they
    survive `cases` (their `expr` is a free pattern variable that unifies
    with any concrete expression), even though `tagToMergeExpr tag _ _` and
    `tagToFinalExpr tag _` constructor-clash against `.fst`, `.diverge`,
    and `.groupVal` at every tag. The discharge is always the same
    one-liner: `cases tag; simp only [dispatcher] at hExpr; cases hExpr`. -/
private theorem fence_fst_diverge_groupval_type {n : Nat}
    {s pred : PSet n} {t : CoreTy n} {ctx' : CoreCtx n}
    (ht : CoreHasType [] (.fst (.diverge (.groupVal s) pred)) t ctx') :
    t = .group (s &&& pred) := by
  cases ht with
  | fstE _ _ _ _ _ he =>
    cases he with
    | diverge _ _ _ _ _ hg =>
      cases hg with
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

-- ============================================================================
-- NEGATIVE instance: fence after partial write is untypable
-- ============================================================================

/-- Fencing after writing only a proper sub-region of the buffer is untypable.

    Fence-domain analog of `collective_after_diverge_untypable` (CSP) and
    `shuffle_diverged_untypable` (GPU). Proof body unchanged from pre-port:
    `fence_requires_all` has the same statement, and the helper `fence_fst_
    diverge_groupval_type` is unchanged. -/
theorem fence_after_partial_write_untypable {n : Nat}
    (s pred : PSet n)
    (hne : s &&& pred ≠ PSet.all n) :
    ¬ ∃ ctx', CoreHasType []
      (.fence (.fst (.diverge (.groupVal s) pred)))
      .unit ctx' := by
  intro ⟨ctx', ht⟩
  have hg := fence_requires_all ht
  have heq := fence_fst_diverge_groupval_type hg
  simp only [CoreTy.group.injEq] at heq
  exact absurd heq.symm hne

/-- Concrete ByteBuf instance: fencing after writing only the low nibble is
    untypable. Parallel to `j1_collective_after_column_split`. -/
theorem bytebuf_fence_after_low_nibble_only :
    ¬ ∃ ctx', CoreHasType []
      (.fence
        (.fst (.diverge (.groupVal ByteBuf.all) ByteBuf.lowNibble)))
      .unit ctx' :=
  fence_after_partial_write_untypable ByteBuf.all ByteBuf.lowNibble (by decide)

-- ============================================================================
-- POSITIVE instance: merge two nibble permissions and fence
-- ============================================================================

/-- Writing the low nibble, writing the high nibble, merging the permissions
    via `IsComplement`, and fencing is well-typed. Validates that
    `mergeFamily` at tag `.group` and `finalizeFamily` at tag `.group`
    compose correctly. -/
theorem fence_after_full_write_typable :
    ∃ ctx', CoreHasType ([] : CoreCtx 8)
      (.fence
        (.merge
          (.write (.groupVal ByteBuf.lowNibble) .dataVal)
          (.write (.groupVal ByteBuf.highNibble) .dataVal)))
      .unit ctx' := by
  refine ⟨[], ?_⟩
  refine CoreHasType.finalizeFamily (n := 8) .group [] []
    (.merge (.write (.groupVal ByteBuf.lowNibble) .dataVal)
            (.write (.groupVal ByteBuf.highNibble) .dataVal))
    _ _ rfl rfl ?_
  refine CoreHasType.mergeFamily (n := 8) .group [] [] []
    (.write (.groupVal ByteBuf.lowNibble) .dataVal)
    (.write (.groupVal ByteBuf.highNibble) .dataVal)
    _ ByteBuf.lowNibble ByteBuf.highNibble (PSet.all 8) _ rfl rfl
    ?_ ?_ nibble_complement
  · exact CoreHasType.write [] [] [] _ _ ByteBuf.lowNibble
      (CoreHasType.groupVal [] ByteBuf.lowNibble)
      (CoreHasType.dataVal [])
  · exact CoreHasType.write [] [] [] _ _ ByteBuf.highNibble
      (CoreHasType.groupVal [] ByteBuf.highNibble)
      (CoreHasType.dataVal [])

-- ============================================================================
-- Metatheory corollaries at ByteBuf width (n = 8)
-- ============================================================================
-- Thin specialisations of `CoreMetatheory` theorems at `n = 8`. They demonstrate
-- that the factored metatheory in `CoreMetatheory.lean` covers the Fence domain
-- at the same depth as Basic.lean's `Metatheory.lean` covers GPU. Each theorem
-- is a one-liner over the Core version — no Fence-specific proof content.

/-- Progress for Fence at ByteBuf width: every closed well-typed Fence
    expression is either a value or can take a step. -/
theorem fence_progress {e : CoreExpr 8} {t : CoreTy 8} {ctx' : CoreCtx 8}
    (ht : CoreHasType [] e t ctx') :
    CoreMetatheory.isValue e = true ∨ ∃ e', CoreMetatheory.Step e e' :=
  CoreMetatheory.progress_closed ht

/-- Preservation for Fence at ByteBuf width: stepping a well-typed Fence
    expression yields a term of the same type at the same context pair. -/
theorem fence_preservation {e e' : CoreExpr 8} {t : CoreTy 8} {ctx ctx' : CoreCtx 8}
    (ht : CoreHasType ctx e t ctx') (hs : CoreMetatheory.Step e e') :
    CoreHasType ctx e' t ctx' :=
  CoreMetatheory.preservation ht hs

/-- Multi-step type safety for Fence at ByteBuf width. -/
theorem fence_type_safety {e e' : CoreExpr 8} {t : CoreTy 8} {ctx' : CoreCtx 8}
    (ht : CoreHasType [] e t ctx') (hstar : CoreMetatheory.Star CoreMetatheory.Step e e') :
    CoreMetatheory.isValue e' = true ∨ ∃ e'', CoreMetatheory.Step e' e'' :=
  CoreMetatheory.type_safety ht hstar
