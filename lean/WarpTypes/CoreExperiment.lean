import WarpTypes.Generic

/-
  Experiment D — Feasibility Probe (2026-04-11)

  Question: can Lean 4's inductive type system accept a typing rule whose
  result type is mediated by a *type-family constructor passed as a rule
  parameter*? This is the shape §9.2 of the research note proposes for a
  higher-order `Core.lean` refactor that would factor the ~170 lines of
  duplication between Fence.lean's `merge` / `fence_requires_all` and
  Reduce.lean's `combineRed` / `finalize_requires_all`.

  The real thing (sketch from §9.2):

      | mergeFamily (T : PSet n → CoreTy n) ... :
          CoreHasType ctx e1 (T s1) ctx' →
          CoreHasType ctx' e2 (T s2) ctx'' →
          PSet.IsComplement s1 s2 parent →
          CoreHasType ctx (mkExpr e1 e2) (T parent) ctx''

  This file is a standalone, minimal toy that mirrors that shape — two
  PSet-indexed type-family constructors, two expression constructors, a
  SINGLE `mergeFamily` rule parameterized by both. If Lean accepts the
  inductive AND admits two instantiations (T := .groupToy and T := .reducedToy),
  the probe returns D1 (go). If Lean rejects the inductive, the probe returns
  D2 (classify the rejection). If the inductive is accepted but the
  instantiations won't elaborate, that is also D2.

  Hard constraints:
  - Must not import or modify any existing domain file (Basic/Csp/Fence/Reduce).
  - Must not be imported from WarpTypes.lean. Built directly via
    `lake build WarpTypes.CoreExperiment`.
  - No sorry, no axiom, no admit.
  - Generic.lean is the ONLY existing file this probe depends on (for PSet
    and IsComplement). Its md5 must be unchanged at the end of the experiment.
-/

namespace CoreExperiment

/-- Toy type family with two PSet-indexed constructors plus a terminal
    sentinel. `finalTy` is the shared extract-side result type, structural
    analog of Fence's `.unit` and (in the Core.lean refactor) Reduce's
    `.data` (finalize pushes us out of the family back into a plain value). -/
inductive ToyTy (n : Nat)
  | groupToy (s : PSet n)
  | reducedToy (s : PSet n)
  | finalTy

/-- Toy expression language. Two distinct `merge`-shaped constructors, two
    distinct `finalize`-shaped constructors. The duplication on the expression
    side is essential — the probe has to pass two DIFFERENT expression
    constructors to the same parametric rule. -/
inductive ToyExpr (n : Nat)
  | leaf (s : PSet n)
  | mergeGroup    (e1 e2 : ToyExpr n)
  | mergeReduced  (e1 e2 : ToyExpr n)
  | finalizeGroup   (e : ToyExpr n)
  | finalizeReduced (e : ToyExpr n)

/-- THE PROBE. One inductive, three rules. The `mergeFamily` rule takes the
    type-family constructor `T` and the expression-level constructor `mkExpr`
    as rule parameters. Everything else — the complement gate, the arity,
    the result shape — is fixed.

    If Lean accepts this inductive, strict positivity isn't in the way: `T`
    and `mkExpr` are universally-quantified function parameters, not
    occurrences of the inductive being defined. -/
inductive ToyHasType : {n : Nat} → ToyExpr n → ToyTy n → Prop
  | leafGroup {n : Nat} (s : PSet n) :
      ToyHasType (.leaf s) (.groupToy s)
  | leafReduced {n : Nat} (s : PSet n) :
      ToyHasType (.leaf s) (.reducedToy s)
  | mergeFamily {n : Nat}
      (T : PSet n → ToyTy n)
      (mkExpr : ToyExpr n → ToyExpr n → ToyExpr n)
      (e1 e2 : ToyExpr n)
      (s1 s2 parent : PSet n) :
      ToyHasType e1 (T s1) →
      ToyHasType e2 (T s2) →
      PSet.IsComplement s1 s2 parent →
      ToyHasType (mkExpr e1 e2) (T parent)
  | finalizeFamily {n : Nat}
      (T : PSet n → ToyTy n)
      (mkFinal : ToyExpr n → ToyExpr n)
      (e : ToyExpr n) :
      ToyHasType e (T (PSet.all n)) →
      ToyHasType (mkFinal e) .finalTy

-- ============================================================================
-- Smoke test: instantiate mergeFamily at two different (T, mkExpr) pairs
-- ============================================================================

/-- Instantiation 1: T := .groupToy, mkExpr := .mergeGroup.
    This is the shape of `Fence.lean`'s `merge` rule, factored out. -/
theorem probe_group_instantiation
    {n : Nat} (s1 s2 parent : PSet n)
    (h : PSet.IsComplement s1 s2 parent) :
    ToyHasType (ToyExpr.mergeGroup (.leaf s1) (.leaf s2)) (ToyTy.groupToy parent) :=
  ToyHasType.mergeFamily
    ToyTy.groupToy
    ToyExpr.mergeGroup
    (.leaf s1) (.leaf s2)
    s1 s2 parent
    (ToyHasType.leafGroup s1)
    (ToyHasType.leafGroup s2)
    h

/-- Instantiation 2: T := .reducedToy, mkExpr := .mergeReduced.
    This is the shape of `Reduce.lean`'s `combineRed` rule, factored out.
    SAME constructor, different parametric instantiation — the whole point
    of the probe. -/
theorem probe_reduced_instantiation
    {n : Nat} (s1 s2 parent : PSet n)
    (h : PSet.IsComplement s1 s2 parent) :
    ToyHasType (ToyExpr.mergeReduced (.leaf s1) (.leaf s2)) (ToyTy.reducedToy parent) :=
  ToyHasType.mergeFamily
    ToyTy.reducedToy
    ToyExpr.mergeReduced
    (.leaf s1) (.leaf s2)
    s1 s2 parent
    (ToyHasType.leafReduced s1)
    (ToyHasType.leafReduced s2)
    h

/-- Bonus: confirm the two instantiations are distinguishable at the type
    level. Both halves use the same underlying rule, but their conclusion
    types are structurally different (`groupToy p` vs `reducedToy p`) — if
    Lean were silently conflating the two instantiations, this would fail
    to elaborate. -/
theorem probe_instantiations_distinct :
    (∀ (s1 s2 p : PSet 4), PSet.IsComplement s1 s2 p →
        ToyHasType (ToyExpr.mergeGroup (.leaf s1) (.leaf s2)) (ToyTy.groupToy p))
    ∧
    (∀ (s1 s2 p : PSet 4), PSet.IsComplement s1 s2 p →
        ToyHasType (ToyExpr.mergeReduced (.leaf s1) (.leaf s2)) (ToyTy.reducedToy p)) :=
  ⟨fun s1 s2 p h => probe_group_instantiation s1 s2 p h,
   fun s1 s2 p h => probe_reduced_instantiation s1 s2 p h⟩

-- ============================================================================
-- PROBE 2: Inversion-side test — does `cases` unify through mkFinal?
-- ============================================================================

/-
  The probe-2 theorem below is the analog of `Fence.lean`'s
  `fence_requires_all` under the Core.lean refactor hypothesis — the
  caller writes the CONCRETE expression constructor `ToyExpr.finalizeGroup g`
  and expects inversion to deliver `ToyHasType g (.groupToy (PSet.all n))`.

  Attempted proof (verbatim, kept in this comment as the negative artifact):

      theorem probe_fence_requires_all_direct {n : Nat} {g : ToyExpr n} :
          ToyHasType (ToyExpr.finalizeGroup g) ToyTy.finalTy →
          ToyHasType g (ToyTy.groupToy (PSet.all n)) := by
        intro h
        cases h with
        | finalizeFamily T mkFinal e' hg => sorry

  Result (Lean 4.28.0):

      error: Dependent elimination failed: Failed to solve equation
        g.finalizeGroup = mkFinal e'

  Classification: OUTCOME D2 (inversion side).

  Why: `cases h` on a `ToyHasType (.finalizeGroup g) .finalTy` goal has
  exactly one matching constructor, `finalizeFamily`. To reach the
  `finalizeFamily` branch, Lean must unify
      mkFinal' e'  =?=  ToyExpr.finalizeGroup g
  where `mkFinal'` is a *function-typed* pattern variable. This is a
  higher-order unification problem (specifically, a pattern-unification
  problem where the flex side is `(mkFinal' e')` with the flex head a
  bound variable applied to another pattern variable). Lean 4 does not
  solve this — it does not invent `mkFinal' := .finalizeGroup` and
  `e' := g`, because in general such a unifier is not unique (e.g.
  `mkFinal' := fun _ => .finalizeGroup g`, `e' := anything` is also a
  solution, and `mkFinal' := id`, `e' := .finalizeGroup g` is yet another).

  Consequence for the Core.lean refactor: the direct parametric approach
  works for CONSTRUCTION (see probe 1) but fails for INVERSION. A working
  Core.lean would need additional machinery — §9.2 of the research note
  lists the candidates: (a) enum-tag + `tagToTy`/`tagToExpr` dispatch
  functions (case analysis lifts to the tag, which is first-order), or
  (b) typeclass encoding with injectivity instances on each expression
  constructor, or (c) reflective representation of the type family. None
  of these is a showstopper; they all raise the refactor cost.

  A third probe (tag-based inversion) would test whether the cheapest
  option — an enum tag — restores inversion without losing the
  construction-side parametricity. That probe is deliberately not in
  this file: the 30-minute scope ends here.
-/

end CoreExperiment

-- ============================================================================
-- PROBE 3: Tag-based inversion — does first-order dispatch fix D2?
-- ============================================================================

/-
  Probe 2 established that function-typed rule parameters (`T`, `mkFinal`)
  block `cases`-based inversion because the resulting unification goal is
  higher-order. Probe 3 tests the cheapest candidate fix from §9.2: replace
  the function parameters with a first-order enum tag + pattern-matching
  dispatch functions. After `cases` on the rule and `cases` on the tag,
  the dispatch functions reduce and the unification becomes first-order
  (equality on concrete `ToyExpr` constructors), which Lean handles fine.
-/

namespace CoreExperiment.Tagged

open CoreExperiment (ToyTy ToyExpr)

/-- Enum tag selecting which PSet-indexed type-family branch a rule is in.
    First-order: two constructors, `Decidable` equality, case-analyzable. -/
inductive TyTag
  | group
  | reduced
  deriving DecidableEq

/-- Tag → type-family constructor. Pattern-matched, not a function parameter.
    Reducible so `cases`/`simp` can unfold it during unification. -/
@[reducible]
def tagToTy {n : Nat} : TyTag → PSet n → ToyTy n
  | .group,   s => .groupToy s
  | .reduced, s => .reducedToy s

/-- Tag → finalize-expression constructor. Same discipline. -/
@[reducible]
def tagToFinalExpr {n : Nat} : TyTag → ToyExpr n → ToyExpr n
  | .group,   e => .finalizeGroup e
  | .reduced, e => .finalizeReduced e

/-- Tagged inductive — ONE `finalizeTagged` rule parameterized by a tag.
    Same homomorphism target as `finalizeFamily` from probe 2, but with
    first-order dispatch instead of function-valued parameters. -/
inductive ToyHasTypeT : {n : Nat} → ToyExpr n → ToyTy n → Prop
  | leafGroup {n : Nat} (s : PSet n) :
      ToyHasTypeT (.leaf s) (.groupToy s)
  | leafReduced {n : Nat} (s : PSet n) :
      ToyHasTypeT (.leaf s) (.reducedToy s)
  | finalizeTagged {n : Nat} (tag : TyTag) (e : ToyExpr n) :
      ToyHasTypeT e (tagToTy tag (PSet.all n)) →
      ToyHasTypeT (tagToFinalExpr tag e) .finalTy

/-
  First attempt: tactic-mode `cases` on `ToyHasTypeT` followed by nested
  `cases` on the tag. This is the naive reading of the fix and it FAILS
  with the same kind of error as probe 2 — `cases h` tries to solve the
  unification equation *before* the inner `cases tag` can reduce the
  dispatch. Preserved as negative artifact:

      theorem probe3_v1 {n : Nat} {g : ToyExpr n} :
          ToyHasTypeT (ToyExpr.finalizeGroup g) ToyTy.finalTy →
          ToyHasTypeT g (ToyTy.groupToy (PSet.all n)) := by
        intro h
        cases h with
        | finalizeTagged tag e' hg => cases tag with
          | group => exact hg
          | reduced => exact hg

      error: Dependent elimination failed: Failed to solve equation
        g.finalizeGroup =
          match tag, e' with
          | TyTag.group, e => e.finalizeGroup
          | TyTag.reduced, e => e.finalizeReduced

  The elaborator doesn't case-split on `tag` during the outer `cases`
  even when `tagToFinalExpr` is `@[reducible]`. This is an ordering issue,
  not a fundamental obstruction. Workaround follows: use an EXPLICIT
  equality hypothesis on the rule.
-/

/-- Second rule shape: the rule takes the expression as a parameter and
    an explicit equality witness stating it equals `tagToFinalExpr tag e`.
    `cases` then gives the equation as a hypothesis rather than forcing
    it into the dependent unification, and we can `simp`/`injection`/
    `subst` in whatever order works. Standard Lean idiom for
    higher-order dispatch in inductive conclusions. -/
inductive ToyHasTypeE : {n : Nat} → ToyExpr n → ToyTy n → Prop
  | leafGroup {n : Nat} (s : PSet n) :
      ToyHasTypeE (.leaf s) (.groupToy s)
  | leafReduced {n : Nat} (s : PSet n) :
      ToyHasTypeE (.leaf s) (.reducedToy s)
  | finalizeTagged {n : Nat} (tag : TyTag) (e expr : ToyExpr n)
      (heq : expr = tagToFinalExpr tag e) :
      ToyHasTypeE e (tagToTy tag (PSet.all n)) →
      ToyHasTypeE expr .finalTy

/-- Probe 3b: tagged inversion via explicit equality hypothesis. -/
theorem probe3_fence_requires_all {n : Nat} {g : ToyExpr n} :
    ToyHasTypeE (ToyExpr.finalizeGroup g) ToyTy.finalTy →
    ToyHasTypeE g (ToyTy.groupToy (PSet.all n)) := by
  intro h
  cases h with
  | finalizeTagged tag e' expr heq hg =>
    cases tag with
    | group =>
      -- tagToFinalExpr .group e' reduces to .finalizeGroup e' ;
      -- tagToTy .group (PSet.all n) reduces to .groupToy (PSet.all n).
      simp only [tagToFinalExpr] at heq
      simp only [tagToTy] at hg
      -- heq : .finalizeGroup g = .finalizeGroup e'  (or the reverse)
      injection heq with heq'
      subst heq'
      exact hg
    | reduced =>
      -- tagToFinalExpr .reduced e' reduces to .finalizeReduced e',
      -- structurally distinct from .finalizeGroup g. Dead branch.
      simp only [tagToFinalExpr] at heq
      contradiction

/-- Mirror theorem on the reduced branch. -/
theorem probe3_finalize_requires_all {n : Nat} {r : ToyExpr n} :
    ToyHasTypeE (ToyExpr.finalizeReduced r) ToyTy.finalTy →
    ToyHasTypeE r (ToyTy.reducedToy (PSet.all n)) := by
  intro h
  cases h with
  | finalizeTagged tag e' expr heq hg =>
    cases tag with
    | group =>
      simp only [tagToFinalExpr] at heq
      contradiction
    | reduced =>
      simp only [tagToFinalExpr] at heq
      simp only [tagToTy] at hg
      injection heq with heq'
      subst heq'
      exact hg

end CoreExperiment.Tagged
