import WarpTypes.Core

/-
  CoreMetatheory — Reduction semantics and progress/preservation for
  the family-parametric Core typing judgment `CoreHasType`.

  Scope. This file inherits from `Core.lean` but does not modify it
  (Core.lean must stay byte-frozen so §3.2's narrative anchor holds).
  It defines `isValue` / `subst` / `Step` on `CoreExpr`, proves canonical
  forms / progress / preservation / type safety for `CoreHasType`, and
  is inherited at concrete widths by Fence (n = 8) and Reduce (n = 4).

  Design note: canonical reduced value. `CoreExpr` has no `.reducedVal`
  constructor; the canonical form of type `.reduced s` is
  `.leafReduce (.groupVal s)`. This keeps `CoreExpr` byte-frozen and
  exposes `IsComplement`'s `Covers` clause at the operational-semantics
  level: `combineRed` reduces by merging the underlying group values,
  which makes the gate do metatheoretic work in `preservation`.

  Basic.lean's GPU-specific `Metatheory.lean` is untouched. The two
  metatheories are parallel — one for `Basic.HasType` over Basic.Expr,
  one for `CoreHasType` over `CoreExpr`.

  All declarations live in the `CoreMetatheory` namespace to avoid
  collision with `Metatheory.lean`'s root-namespace lemmas of the same
  name (`subst`, `isValue`, `Step`, `remove_comm`, etc.).
-/

namespace CoreMetatheory

-- ============================================================================
-- Values
-- ============================================================================

/-- Values of `CoreExpr`. The canonical form of `.reduced s` is
    `.leafReduce (.groupVal s)` — see the design-note headnote. -/
def isValue {n : Nat} : CoreExpr n → Bool
  | .groupVal _ => true
  | .dataVal => true
  | .unitVal => true
  | .leafReduce (.groupVal _) => true
  | .pairVal a b => isValue a && isValue b
  | _ => false

-- ============================================================================
-- Substitution
-- ============================================================================

/-- Capture-avoiding substitution on `CoreExpr`. Binders shadow per the
    standard letBind / letPair rules. Adapted from `Metatheory.lean`'s
    `subst` for Basic.Expr; structurally identical where the expression
    languages overlap. -/
def subst {n : Nat} (e : CoreExpr n) (x : String) (v : CoreExpr n) : CoreExpr n :=
  match e with
  | .groupVal s => .groupVal s
  | .dataVal => .dataVal
  | .unitVal => .unitVal
  | .var name => if name == x then v else .var name
  | .diverge g pred => .diverge (subst g x v) pred
  | .merge g1 g2 => .merge (subst g1 x v) (subst g2 x v)
  | .combineRed r1 r2 => .combineRed (subst r1 x v) (subst r2 x v)
  | .letBind name val body =>
      if name == x then .letBind name (subst val x v) body
      else .letBind name (subst val x v) (subst body x v)
  | .pairVal a b => .pairVal (subst a x v) (subst b x v)
  | .fst e => .fst (subst e x v)
  | .snd e => .snd (subst e x v)
  | .letPair e n1 n2 body =>
      if n1 == x || n2 == x then .letPair (subst e x v) n1 n2 body
      else .letPair (subst e x v) n1 n2 (subst body x v)
  | .write g payload => .write (subst g x v) (subst payload x v)
  | .leafReduce g => .leafReduce (subst g x v)
  | .fence g => .fence (subst g x v)
  | .finalize r => .finalize (subst r x v)

-- ============================================================================
-- Context infrastructure lemmas
-- ============================================================================
-- Structurally identical to the `Metatheory.lean` lemmas for `Ctx n`.
-- `CoreCtx n` has the same underlying List (String × _) representation with
-- identical `lookup` / `remove` implementations, so the proofs carry over
-- verbatim with a mechanical `Ctx`→`CoreCtx`, `Ty`→`CoreTy` rename.
-- Duplicated here (rather than generalised into Generic.lean) to keep
-- Generic.lean's md5 invariant stable — the §3.2 narrative anchor.

/-- Looking up a name in a context from which it was removed yields none. -/
theorem remove_lookup_self {n : Nat} (ctx : CoreCtx n) (name : String) :
    (ctx.remove name).lookup name = none := by
  induction ctx with
  | nil => rfl
  | cons p tl ih =>
    simp only [CoreCtx.remove, CoreCtx.lookup]
    unfold List.filter
    by_cases h : (p.1 != name) = true
    · simp only [h, List.find?]
      have hne : (p.1 == name) = false := by
        simp only [bne_iff_ne, ne_eq] at h; simp [h]
      simp only [hne, Option.map, CoreCtx.lookup, CoreCtx.remove] at ih ⊢
      exact ih
    · simp only [Bool.not_eq_true] at h; simp only [h]
      exact ih

/-- Looking up a different name is unaffected by remove. -/
theorem remove_lookup_ne {n : Nat} {name name' : String} (hne : name ≠ name') (ctx : CoreCtx n) :
    (ctx.remove name).lookup name' = ctx.lookup name' := by
  induction ctx with
  | nil => rfl
  | cons p tl ih =>
    simp only [CoreCtx.remove, CoreCtx.lookup]
    by_cases hp : p.1 = name
    · have h_bne : (p.1 != name) = false := by rw [hp]; simp [bne_iff_ne]
      simp only [List.filter, h_bne]
      have h_beq : (p.1 == name') = false := by rw [hp]; simp [beq_iff_eq, hne]
      simp only [List.find?, h_beq]
      exact ih
    · have h_bne : (p.1 != name) = true := by simp [bne_iff_ne, hp]
      simp only [List.filter, h_bne, List.find?]
      by_cases hq : p.1 = name'
      · simp [beq_iff_eq, hq]
      · have h_beq : (p.1 == name') = false := by simp [beq_iff_eq, hq]
        simp only [h_beq]
        exact ih

/-- Removing from a context where the name is absent is identity. -/
theorem remove_of_lookup_none {n : Nat} {ctx : CoreCtx n} {name : String}
    (h : ctx.lookup name = none) : ctx.remove name = ctx := by
  induction ctx with
  | nil => rfl
  | cons p tl ih =>
    simp only [CoreCtx.lookup, List.find?] at h
    by_cases hq : (p.1 == name) = true
    · simp [hq, Option.map] at h
    · simp only [Bool.not_eq_true] at hq
      simp only [hq] at h
      simp only [CoreCtx.remove, List.filter]
      have : (p.1 != name) = true := by
        simp [bne_iff_ne, ne_eq]; simp [beq_iff_eq] at hq; exact hq
      simp only [this]
      congr 1
      exact ih h

/-- Lookup on cons with matching name. -/
theorem lookup_cons_eq {n : Nat} (name : String) (t : CoreTy n) (ctx : CoreCtx n) :
    CoreCtx.lookup ((name, t) :: ctx) name = some t := by
  simp [CoreCtx.lookup, List.find?]

/-- Lookup on cons with different name. -/
theorem lookup_cons_ne {n : Nat} {name name' : String} (hne : name ≠ name') (t : CoreTy n)
    (ctx : CoreCtx n) :
    CoreCtx.lookup ((name, t) :: ctx) name' = ctx.lookup name' := by
  simp only [CoreCtx.lookup, List.find?]
  have : (name == name') = false := by simp [beq_iff_eq, hne]
  simp [this]

/-- Remove on cons with matching name. -/
theorem remove_cons_eq {n : Nat} (name : String) (t : CoreTy n) (ctx : CoreCtx n) :
    CoreCtx.remove ((name, t) :: ctx) name = CoreCtx.remove ctx name := by
  simp only [CoreCtx.remove, List.filter]
  have : ((name, t).1 != name) = false := by simp [bne_iff_ne, ne_eq]
  simp [this]

/-- Remove on cons with different name. -/
theorem remove_cons_ne {n : Nat} {name name' : String} (hne : name ≠ name') (t : CoreTy n)
    (ctx : CoreCtx n) :
    CoreCtx.remove ((name', t) :: ctx) name = (name', t) :: CoreCtx.remove ctx name := by
  simp only [CoreCtx.remove, List.filter]
  have : ((name', t).1 != name) = true := by simp [bne_iff_ne, ne_eq]; exact Ne.symm hne
  simp [this]

/-- Remove commutes: order of removal doesn't matter. -/
theorem remove_comm {n : Nat} (ctx : CoreCtx n) (a b : String) :
    CoreCtx.remove (CoreCtx.remove ctx a) b = CoreCtx.remove (CoreCtx.remove ctx b) a := by
  simp only [CoreCtx.remove, List.filter_filter]
  congr 1; ext p; simp [Bool.and_comm]

-- ============================================================================
-- Small-Step Reduction
-- ============================================================================

/-- Small-step reduction on `CoreExpr`.

    Design note — canonical reduced value. `.reduced s` has no dedicated
    value constructor; its canonical value form is `.leafReduce (.groupVal s)`.
    The `combineRedVal` rule below therefore reduces
    `combineRed (.leafReduce (.groupVal s1)) (.leafReduce (.groupVal s2))`
    to `.leafReduce (.groupVal (s1 ||| s2))`, walking through the underlying
    group spines. This makes `IsComplement`'s `Covers` clause load-bearing
    in `preservation` rather than only at typing-rule construction time —
    the gate is used operationally, not just as a typing side condition.
    (See INSIGHTS §N+50.)

    The family-parametric typing rules `mergeFamily` and `finalizeFamily`
    are *not* mirrored by a single parametric Step rule. Instead, the four
    concrete reduction rules (`mergeVal`, `combineRedVal`, `fenceVal`,
    `finalizeVal`) each live as monomorphic Step constructors. `preservation`
    does the tag-dispatch manually. Attempting a single parametric
    `mergeFamilyVal` rule would hit the same stuck-dispatch problem
    Experiment D probe 3a documented for typing — better to keep Step
    first-order and pay the discharge cost in one place (preservation)
    rather than sprinkle it across the Step inductive as well. -/
inductive Step {n : Nat} : CoreExpr n → CoreExpr n → Prop
  -- ── Value-producing reductions ──
  | divergeVal (s pred : PSet n) :
      Step (.diverge (.groupVal s) pred)
           (.pairVal (.groupVal (s &&& pred)) (.groupVal (s &&& ~~~pred)))
  | mergeVal (s1 s2 : PSet n) :
      Step (.merge (.groupVal s1) (.groupVal s2)) (.groupVal (s1 ||| s2))
  | combineRedVal (s1 s2 : PSet n) :
      Step (.combineRed (.leafReduce (.groupVal s1)) (.leafReduce (.groupVal s2)))
           (.leafReduce (.groupVal (s1 ||| s2)))
  | writeVal (s : PSet n) :
      Step (.write (.groupVal s) .dataVal) (.groupVal s)
  | fenceVal (s : PSet n) :
      Step (.fence (.groupVal s)) .unitVal
  | finalizeVal (s : PSet n) :
      Step (.finalize (.leafReduce (.groupVal s))) .dataVal
  | letVal (name : String) (v body : CoreExpr n) :
      isValue v = true →
      Step (.letBind name v body) (subst body name v)
  | fstVal (a b : CoreExpr n) :
      isValue a = true → isValue b = true →
      Step (.fst (.pairVal a b)) a
  | sndVal (a b : CoreExpr n) :
      isValue a = true → isValue b = true →
      Step (.snd (.pairVal a b)) b
  | letPairVal (name1 name2 : String) (v1 v2 body : CoreExpr n) :
      isValue v1 = true → isValue v2 = true →
      Step (.letPair (.pairVal v1 v2) name1 name2 body)
           (subst (subst body name1 v1) name2 v2)
  -- ── Congruence rules ──
  | divergeCong (g g' : CoreExpr n) (pred : PSet n) :
      Step g g' → Step (.diverge g pred) (.diverge g' pred)
  | mergeLeft (g1 g1' g2 : CoreExpr n) :
      Step g1 g1' → Step (.merge g1 g2) (.merge g1' g2)
  | mergeRight (v1 g2 g2' : CoreExpr n) :
      isValue v1 = true → Step g2 g2' →
      Step (.merge v1 g2) (.merge v1 g2')
  | combineRedLeft (r1 r1' r2 : CoreExpr n) :
      Step r1 r1' → Step (.combineRed r1 r2) (.combineRed r1' r2)
  | combineRedRight (v1 r2 r2' : CoreExpr n) :
      isValue v1 = true → Step r2 r2' →
      Step (.combineRed v1 r2) (.combineRed v1 r2')
  | letCong (name : String) (val val' body : CoreExpr n) :
      Step val val' →
      Step (.letBind name val body) (.letBind name val' body)
  | pairLeftCong (a a' b : CoreExpr n) :
      Step a a' → Step (.pairVal a b) (.pairVal a' b)
  | pairRightCong (a b b' : CoreExpr n) :
      isValue a = true → Step b b' →
      Step (.pairVal a b) (.pairVal a b')
  | fstCong (e e' : CoreExpr n) :
      Step e e' → Step (.fst e) (.fst e')
  | sndCong (e e' : CoreExpr n) :
      Step e e' → Step (.snd e) (.snd e')
  | letPairCong (e e' : CoreExpr n) (name1 name2 : String) (body : CoreExpr n) :
      Step e e' → Step (.letPair e name1 name2 body) (.letPair e' name1 name2 body)
  | writeLeft (g g' payload : CoreExpr n) :
      Step g g' → Step (.write g payload) (.write g' payload)
  | writeRight (v payload payload' : CoreExpr n) :
      isValue v = true → Step payload payload' →
      Step (.write v payload) (.write v payload')
  | leafReduceCong (g g' : CoreExpr n) :
      Step g g' → Step (.leafReduce g) (.leafReduce g')
  | fenceCong (g g' : CoreExpr n) :
      Step g g' → Step (.fence g) (.fence g')
  | finalizeCong (r r' : CoreExpr n) :
      Step r r' → Step (.finalize r) (.finalize r')

-- ============================================================================
-- Canonical forms
-- ============================================================================
-- A closed value at each `CoreTy` constructor has a predictable syntactic
-- shape. The `canonical_reduced` case is the interesting one: under the
-- option-(a) design (INSIGHTS §N+50) it forces
-- `e = .leafReduce (.groupVal s)`, not a dedicated `.reducedVal` constructor.

/-- A closed value at type `.group s` is syntactically `.groupVal s`. -/
theorem canonical_group {n : Nat} {e : CoreExpr n} {s : PSet n} {ctx' : CoreCtx n}
    (ht : CoreHasType [] e (.group s) ctx') (hv : isValue e = true) :
    e = .groupVal s := by
  match ht with
  | .groupVal _ _ => rfl
  | .var _ _ _ hlook => simp [CoreCtx.lookup, List.find?] at hlook
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .write _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr hTy _ _ _ =>
    cases tag
    case group =>
      simp [tagToMergeExpr] at hExpr
      subst hExpr
      simp [isValue] at hv
    case reduced => simp [tagToTy] at hTy
  | .finalizeFamily tag _ _ _ _ _ hExpr hTy _ =>
    cases tag <;> simp [tagToFinalTy] at hTy

/-- A closed value at type `.data` is syntactically `.dataVal`. -/
theorem canonical_data {n : Nat} {e : CoreExpr n} {ctx' : CoreCtx n}
    (ht : CoreHasType [] e .data ctx') (hv : isValue e = true) :
    e = .dataVal := by
  match ht with
  | .dataVal _ => rfl
  | .var _ _ _ hlook => simp [CoreCtx.lookup, List.find?] at hlook
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ _ hTy _ _ _ =>
    cases tag <;> simp [tagToTy] at hTy
  | .finalizeFamily tag _ _ _ _ _ hExpr hTy _ =>
    cases tag
    case group => simp [tagToFinalTy] at hTy
    case reduced =>
      simp [tagToFinalExpr] at hExpr
      subst hExpr
      simp [isValue] at hv

/-- A closed value at type `.unit` is syntactically `.unitVal`. -/
theorem canonical_unit {n : Nat} {e : CoreExpr n} {ctx' : CoreCtx n}
    (ht : CoreHasType [] e .unit ctx') (hv : isValue e = true) :
    e = .unitVal := by
  match ht with
  | .unitVal _ => rfl
  | .var _ _ _ hlook => simp [CoreCtx.lookup, List.find?] at hlook
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ _ hTy _ _ _ =>
    cases tag <;> simp [tagToTy] at hTy
  | .finalizeFamily tag _ _ _ _ _ hExpr hTy _ =>
    cases tag
    case group =>
      simp [tagToFinalExpr] at hExpr
      subst hExpr
      simp [isValue] at hv
    case reduced => simp [tagToFinalTy] at hTy

/-- A closed value at type `.pair t1 t2` is syntactically `.pairVal v1 v2`
    where both components are themselves values. -/
theorem canonical_pair {n : Nat} {e : CoreExpr n} {t1 t2 : CoreTy n} {ctx' : CoreCtx n}
    (ht : CoreHasType [] e (.pair t1 t2) ctx') (hv : isValue e = true) :
    ∃ v1 v2, e = .pairVal v1 v2 ∧ isValue v1 = true ∧ isValue v2 = true := by
  match ht with
  | .pairVal _ _ _ a b _ _ _ _ =>
    simp [isValue] at hv; exact ⟨a, b, rfl, hv.1, hv.2⟩
  | .var _ _ _ hlook => simp [CoreCtx.lookup, List.find?] at hlook
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ _ hTy _ _ _ =>
    cases tag <;> simp [tagToTy] at hTy
  | .finalizeFamily tag _ _ _ _ _ _ hTy _ =>
    cases tag <;> simp [tagToFinalTy] at hTy

/-- A closed value at type `.reduced s` is syntactically `.leafReduce (.groupVal s)`
    — the canonical form dictated by the option-(a) design. This is the
    validation moment for INSIGHTS §N+50: if this proof forces a dedicated
    `.reducedVal` constructor, the design must be retracted. It does not. -/
theorem canonical_reduced {n : Nat} {e : CoreExpr n} {s : PSet n} {ctx' : CoreCtx n}
    (ht : CoreHasType [] e (.reduced s) ctx') (hv : isValue e = true) :
    e = .leafReduce (.groupVal s) := by
  match ht with
  | .var _ _ _ hlook => simp [CoreCtx.lookup, List.find?] at hlook
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .leafReduce _ _ g _ hg =>
    -- isValue (.leafReduce g) forces g = .groupVal _; then hg's typing forces
    -- the groupVal's PSet index to equal s.
    match hg with
    | .groupVal _ _ => rfl
    | .var _ _ _ hlook => simp [CoreCtx.lookup, List.find?] at hlook
    | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .fstE _ _ _ _ _ _ => simp [isValue] at hv
    | .sndE _ _ _ _ _ _ => simp [isValue] at hv
    | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .write _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;>
        (simp [tagToMergeExpr] at hExpr; subst hExpr; simp [isValue] at hv)
    | .finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;>
        (simp [tagToFinalExpr] at hExpr; subst hExpr; simp [isValue] at hv)
  | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr hTy _ _ _ =>
    cases tag
    case group => simp [tagToTy] at hTy
    case reduced =>
      simp [tagToMergeExpr] at hExpr
      subst hExpr
      simp [isValue] at hv
  | .finalizeFamily tag _ _ _ _ _ _ hTy _ =>
    cases tag <;> simp [tagToFinalTy] at hTy

-- ============================================================================
-- Values preserve contexts (values consume no linear resources)
-- ============================================================================

/-- If a well-typed expression is syntactically a value, its input and output
    contexts coincide — values consume no linear resources. -/
theorem value_preserves_ctx {n : Nat} {ctx ctx' : CoreCtx n} {v : CoreExpr n} {t : CoreTy n}
    (ht : CoreHasType ctx v t ctx') (hv : isValue v = true) : ctx = ctx' := by
  match ht with
  | .groupVal _ _ => rfl
  | .dataVal _ => rfl
  | .unitVal _ => rfl
  | .pairVal _ ctx_mid _ a b _ _ ha hb =>
    simp [isValue] at hv
    have h1 := value_preserves_ctx ha hv.1
    have h2 := value_preserves_ctx hb hv.2
    subst h1; subst h2; rfl
  | .var _ _ _ _ => simp [isValue] at hv
  | .diverge _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .write _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .leafReduce _ _ g _ hg =>
    -- .leafReduce (.groupVal s) is a value; inner typing gives ctx = ctx via groupVal's rule.
    -- For every other shape of g, `isValue (.leafReduce g) = false` so hv contradicts.
    match hg with
    | .groupVal _ _ => rfl
    | .var _ _ _ _ => simp [isValue] at hv
    | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .fstE _ _ _ _ _ _ => simp [isValue] at hv
    | .sndE _ _ _ _ _ _ => simp [isValue] at hv
    | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .write _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;>
        (simp [tagToMergeExpr] at hExpr; subst hExpr; simp [isValue] at hv)
    | .finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;>
        (simp [tagToFinalExpr] at hExpr; subst hExpr; simp [isValue] at hv)
  | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
    cases tag <;>
      (simp [tagToMergeExpr] at hExpr; subst hExpr; simp [isValue] at hv)
  | .finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
    cases tag <;>
      (simp [tagToFinalExpr] at hExpr; subst hExpr; simp [isValue] at hv)

-- ============================================================================
-- subst commutes with the tag dispatchers
-- ============================================================================
-- These two lemmas let every parametric-rule case of `subst_typing`,
-- `progress`, and `preservation` rewrite through `tagToMergeExpr` and
-- `tagToFinalExpr` without per-tag repetition.

theorem subst_tagToMergeExpr {n : Nat} (tag : TyTag) (e1 e2 : CoreExpr n)
    (nm : String) (v : CoreExpr n) :
    subst (tagToMergeExpr tag e1 e2) nm v
      = tagToMergeExpr tag (subst e1 nm v) (subst e2 nm v) := by
  cases tag <;> rfl

theorem subst_tagToFinalExpr {n : Nat} (tag : TyTag) (e : CoreExpr n)
    (nm : String) (v : CoreExpr n) :
    subst (tagToFinalExpr tag e) nm v = tagToFinalExpr tag (subst e nm v) := by
  cases tag <;> rfl

-- ============================================================================
-- Output context bindings come from input context
-- ============================================================================

/-- Any binding in the output context was present (with the same type) in input. -/
theorem output_binding_from_input {n : Nat} {ctx ctx' : CoreCtx n}
    {e : CoreExpr n} {t : CoreTy n}
    (ht : CoreHasType ctx e t ctx')
    {x : String} {tx : CoreTy n}
    (hout : ctx'.lookup x = some tx) :
    ctx.lookup x = some tx := by
  induction ht with
  | groupVal _ _ => exact hout
  | dataVal _ => exact hout
  | unitVal _ => exact hout
  | var ctx₀ name₀ t₀ hlook =>
    by_cases hxn : x = name₀
    · subst hxn; rw [remove_lookup_self] at hout; exact absurd hout (by simp)
    · rwa [remove_lookup_ne (Ne.symm hxn)] at hout
  | diverge _ _ _ _ _ _ ih => exact ih hout
  | letBind ctx₀ ctx_mid ctx_body name' _ _ _ _ _ hfresh₀ _ hcons₀ ih_val ih_body =>
    have h_body := ih_body hout
    by_cases hxn : x = name'
    · subst hxn; rw [hcons₀] at hout; exact absurd hout (by simp)
    · rw [lookup_cons_ne (Ne.symm hxn)] at h_body
      exact ih_val h_body
  | pairVal _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1 (ih2 hout)
  | fstE _ _ _ _ _ _ ih => exact ih hout
  | sndE _ _ _ _ _ _ ih => exact ih hout
  | letPairE ctx₀ ctx_mid ctx_body _ n1 n2 _ _ _ _ he hdist hfresh1 hfresh2 hbody hcons1 hcons2 ih_e ih_body =>
    have h_body := ih_body hout
    have hxn1 : x ≠ n1 := by intro h; subst h; rw [hcons1] at hout; simp at hout
    have hxn2 : x ≠ n2 := by intro h; subst h; rw [hcons2] at hout; simp at hout
    rw [lookup_cons_ne (Ne.symm hxn2), lookup_cons_ne (Ne.symm hxn1)] at h_body
    exact ih_e h_body
  | write _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1 (ih2 hout)
  | leafReduce _ _ _ _ _ ih => exact ih hout
  | mergeFamily _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1 (ih2 hout)
  | finalizeFamily _ _ _ _ _ _ _ _ _ ih => exact ih hout

-- ============================================================================
-- Values can be typed in any context
-- ============================================================================

/-- Helper for `value_any_ctx` — values produce an any-context typing.
    Needed separately because the direct version hits a dependent-index
    constraint when the inner value's types are dispatched. -/
private theorem value_any_ctx_aux {n : Nat} {v : CoreExpr n} {t : CoreTy n}
    {ctx₁ ctx₂ : CoreCtx n}
    (hv : isValue v = true)
    (ht : CoreHasType ctx₁ v t ctx₂) :
    ∀ ctx₃, CoreHasType ctx₃ v t ctx₃ := by
  match ht with
  | .groupVal _ s => intro ctx₃; exact CoreHasType.groupVal ctx₃ s
  | .dataVal _ => intro ctx₃; exact CoreHasType.dataVal ctx₃
  | .unitVal _ => intro ctx₃; exact CoreHasType.unitVal ctx₃
  | .pairVal _ _ _ a b _ _ ha hb =>
    simp [isValue] at hv
    have iha := value_any_ctx_aux hv.1 ha
    have ihb := value_any_ctx_aux hv.2 hb
    intro ctx₃
    exact CoreHasType.pairVal ctx₃ ctx₃ ctx₃ a b _ _ (iha ctx₃) (ihb ctx₃)
  | .var _ _ _ _ => simp [isValue] at hv
  | .diverge _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .write _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .leafReduce _ _ g _ hg =>
    -- .leafReduce (.groupVal s) is a value; the inner groupVal types at any context.
    match hg with
    | .groupVal _ s =>
      intro ctx₃
      exact CoreHasType.leafReduce ctx₃ ctx₃ _ s (CoreHasType.groupVal ctx₃ s)
    | .var _ _ _ _ => simp [isValue] at hv
    | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .fstE _ _ _ _ _ _ => simp [isValue] at hv
    | .sndE _ _ _ _ _ _ => simp [isValue] at hv
    | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .write _ _ _ _ _ _ _ _ => simp [isValue] at hv
    | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;>
        (simp [tagToMergeExpr] at hExpr; subst hExpr; simp [isValue] at hv)
    | .finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;>
        (simp [tagToFinalExpr] at hExpr; subst hExpr; simp [isValue] at hv)
  | .mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
    cases tag <;>
      (simp [tagToMergeExpr] at hExpr; subst hExpr; simp [isValue] at hv)
  | .finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
    cases tag <;>
      (simp [tagToFinalExpr] at hExpr; subst hExpr; simp [isValue] at hv)

/-- Values can be typed in any context, producing the same context unchanged. -/
theorem value_any_ctx {n : Nat} {v : CoreExpr n} {t : CoreTy n} {ctx₁ : CoreCtx n}
    (hv : isValue v = true)
    (ht : CoreHasType ctx₁ v t ctx₁) :
    ∀ ctx₂, CoreHasType ctx₂ v t ctx₂ :=
  value_any_ctx_aux hv ht

-- ============================================================================
-- Substitution preserves typing — the long-pole generalised lemma
-- ============================================================================

/-- The var case of `subst_typing`, extracted so the main function can pattern-
    match on non-var cases without nesting this branch. -/
private theorem subst_typing_var {n : Nat}
    {nm : String} {t_v : CoreTy n} {v : CoreExpr n}
    (hv : isValue v = true)
    (ht_v : ∀ ctx₂, CoreHasType ctx₂ v t_v ctx₂)
    {ctx₀ : CoreCtx n} {name' : String} {t' : CoreTy n}
    (hlook : ctx₀.lookup name' = some t')
    (hname : ∀ t'', ctx₀.lookup nm = some t'' → t'' = t_v) :
    CoreHasType (ctx₀.remove nm) (subst (.var name') nm v) t'
            (CoreCtx.remove (ctx₀.remove name') nm) := by
  simp only [subst]
  by_cases hxn : name' = nm
  · -- name' = nm: substitute with v
    have hbeq : (name' == nm) = true := by simp [beq_iff_eq, hxn]
    simp only [hbeq]
    have hlook_nm : ctx₀.lookup nm = some t' := hxn ▸ hlook
    have : t' = t_v := hname t' hlook_nm; subst this
    rw [hxn, remove_of_lookup_none (remove_lookup_self ctx₀ nm)]
    exact ht_v _
  · -- name' ≠ nm: no substitution, just adjust contexts
    have hbeq : (name' == nm) = false := by simp [beq_iff_eq, hxn]
    simp only [hbeq]
    have hlook' : (ctx₀.remove nm).lookup name' = some t' := by
      rw [remove_lookup_ne (Ne.symm hxn)]; exact hlook
    have hgoal := CoreHasType.var (ctx₀.remove nm) name' t' hlook'
    rw [remove_comm] at hgoal
    exact hgoal

/-- Core substitution theorem: substituting a value for a name removes that
    name's binding from both input and output contexts.

    Generalises over where in the context the binding appears, which is
    essential for the pair/merge/combineRed/write cases where the binding may
    have been threaded past the first sub-expression. -/
theorem subst_typing {n : Nat}
    {nm : String} {t_v : CoreTy n} {v : CoreExpr n}
    (hv : isValue v = true)
    (ht_v : ∀ ctx₂, CoreHasType ctx₂ v t_v ctx₂)
    {ctx : CoreCtx n} {e : CoreExpr n} {t : CoreTy n} {ctx' : CoreCtx n}
    (hte : CoreHasType ctx e t ctx')
    (hname : ∀ t', ctx.lookup nm = some t' → t' = t_v) :
    CoreHasType (ctx.remove nm) (subst e nm v) t (ctx'.remove nm) :=
  match hte with
  | .groupVal _ s => by simp [subst]; exact CoreHasType.groupVal _ s
  | .dataVal _ => by simp [subst]; exact CoreHasType.dataVal _
  | .unitVal _ => by simp [subst]; exact CoreHasType.unitVal _
  | .var _ _ _ hlook => subst_typing_var hv ht_v hlook hname
  | .diverge _ _ _ _ _ hg => by
    simp [subst]
    exact CoreHasType.diverge _ _ _ _ _ (subst_typing hv ht_v hg hname)
  | .letBind _ ctx_mid ctx_body name' _ _ t1 _ hval hfresh hbody hconsumed => by
    simp only [subst]
    by_cases hxn : name' = nm
    · -- name' = nm: body NOT substituted (shadowing)
      simp [show (name' == nm) = true from by simp [beq_iff_eq, hxn]]
      have hfresh_nm : ctx_mid.lookup nm = none := hxn ▸ hfresh
      have hconsumed_nm : ctx_body.lookup nm = none := hxn ▸ hconsumed
      have hval' := subst_typing hv ht_v hval hname
      rw [remove_of_lookup_none hfresh_nm] at hval'
      rw [remove_of_lookup_none hconsumed_nm]
      exact CoreHasType.letBind _ _ _ _ _ _ _ _ hval' hfresh hbody hconsumed
    · -- name' ≠ nm: both val and body substituted
      simp [show (name' == nm) = false from by simp [beq_iff_eq, hxn]]
      have hval' := subst_typing hv ht_v hval hname
      have hfresh' : (ctx_mid.remove nm).lookup name' = none := by
        rw [remove_lookup_ne (Ne.symm hxn)]; exact hfresh
      have hname_body : ∀ t', CoreCtx.lookup ((name', t1) :: ctx_mid) nm = some t' → t' = t_v := by
        intro t' hl
        have : CoreCtx.lookup ctx_mid nm = some t' := by
          rwa [lookup_cons_ne hxn] at hl
        exact hname t' (output_binding_from_input hval this)
      have hbody' := subst_typing hv ht_v hbody hname_body
      rw [remove_cons_ne (Ne.symm hxn)] at hbody'
      have hconsumed' : (ctx_body.remove nm).lookup name' = none := by
        rw [remove_lookup_ne (Ne.symm hxn)]; exact hconsumed
      exact CoreHasType.letBind _ _ _ _ _ _ _ _ hval' hfresh' hbody' hconsumed'
  | .pairVal _ ctx_mid _ _ _ _ _ ha hb => by
    simp [subst]
    have hname_mid : ∀ t', ctx_mid.lookup nm = some t' → t' = t_v := by
      intro t' hl; exact hname t' (output_binding_from_input ha hl)
    exact CoreHasType.pairVal _ _ _ _ _ _ _
      (subst_typing hv ht_v ha hname) (subst_typing hv ht_v hb hname_mid)
  | .fstE _ _ _ _ _ he => by
    simp [subst]; exact CoreHasType.fstE _ _ _ _ _ (subst_typing hv ht_v he hname)
  | .sndE _ _ _ _ _ he => by
    simp [subst]; exact CoreHasType.sndE _ _ _ _ _ (subst_typing hv ht_v he hname)
  | .write _ ctx_mid _ _ _ _ hg hpayload => by
    simp [subst]
    have hname_mid : ∀ t', ctx_mid.lookup nm = some t' → t' = t_v := by
      intro t' hl; exact hname t' (output_binding_from_input hg hl)
    exact CoreHasType.write _ _ _ _ _ _
      (subst_typing hv ht_v hg hname) (subst_typing hv ht_v hpayload hname_mid)
  | .leafReduce _ _ _ _ hg => by
    simp [subst]
    exact CoreHasType.leafReduce _ _ _ _ (subst_typing hv ht_v hg hname)
  | .mergeFamily tag _ ctx_mid _ _ _ _ _ _ _ _ hExpr hTy hw1 hw2 hcomp => by
    subst hExpr
    subst hTy
    rw [subst_tagToMergeExpr]
    have hname_mid : ∀ t', ctx_mid.lookup nm = some t' → t' = t_v := by
      intro t' hl; exact hname t' (output_binding_from_input hw1 hl)
    exact CoreHasType.mergeFamily tag _ _ _ _ _ _ _ _ _ _ rfl rfl
      (subst_typing hv ht_v hw1 hname)
      (subst_typing hv ht_v hw2 hname_mid)
      hcomp
  | .finalizeFamily tag _ _ _ _ _ hExpr hTy hw => by
    subst hExpr
    subst hTy
    rw [subst_tagToFinalExpr]
    exact CoreHasType.finalizeFamily tag _ _ _ _ _ rfl rfl
      (subst_typing hv ht_v hw hname)
  | .letPairE _ ctx_mid ctx_body _ n1 n2 _ t1 t2 _ he hdist hfresh1 hfresh2 hbody hcons1 hcons2 => by
    simp only [subst]
    by_cases hxn1 : n1 = nm
    · -- n1 = nm: body NOT substituted (shadowing by n1)
      have hor : (n1 == nm || n2 == nm) = true := by simp [beq_iff_eq, hxn1]
      simp only [hor]
      have hfresh_nm1 : ctx_mid.lookup nm = none := hxn1 ▸ hfresh1
      have hcons_nm1 : ctx_body.lookup nm = none := hxn1 ▸ hcons1
      have he' := subst_typing hv ht_v he hname
      rw [remove_of_lookup_none hfresh_nm1] at he'
      rw [remove_of_lookup_none hcons_nm1]
      exact CoreHasType.letPairE _ _ _ _ _ _ _ _ _ _ he' hdist hfresh1 hfresh2 hbody hcons1 hcons2
    · by_cases hxn2 : n2 = nm
      · -- n2 = nm: body NOT substituted (shadowing by n2)
        have hor : (n1 == nm || n2 == nm) = true := by simp [beq_iff_eq, hxn2]
        simp only [hor]
        have hfresh_nm2 : ctx_mid.lookup nm = none := hxn2 ▸ hfresh2
        have hcons_nm2 : ctx_body.lookup nm = none := hxn2 ▸ hcons2
        have he' := subst_typing hv ht_v he hname
        rw [remove_of_lookup_none hfresh_nm2] at he'
        rw [remove_of_lookup_none hcons_nm2]
        exact CoreHasType.letPairE _ _ _ _ _ _ _ _ _ _ he' hdist hfresh1 hfresh2 hbody hcons1 hcons2
      · -- Neither n1 nor n2 = nm: both e and body substituted
        have hor : (n1 == nm || n2 == nm) = false := by
          simp [beq_iff_eq, hxn1, hxn2]
        simp only [hor]
        have he' := subst_typing hv ht_v he hname
        have hfresh1' : (ctx_mid.remove nm).lookup n1 = none := by
          rw [remove_lookup_ne (Ne.symm hxn1)]; exact hfresh1
        have hfresh2' : (ctx_mid.remove nm).lookup n2 = none := by
          rw [remove_lookup_ne (Ne.symm hxn2)]; exact hfresh2
        have hname_body : ∀ t', CoreCtx.lookup ((n2, t2) :: (n1, t1) :: ctx_mid) nm = some t' → t' = t_v := by
          intro t' hl
          rw [lookup_cons_ne hxn2, lookup_cons_ne hxn1] at hl
          exact hname t' (output_binding_from_input he hl)
        have hbody' := subst_typing hv ht_v hbody hname_body
        rw [remove_cons_ne (Ne.symm hxn2), remove_cons_ne (Ne.symm hxn1)] at hbody'
        have hcons1' : (ctx_body.remove nm).lookup n1 = none := by
          rw [remove_lookup_ne (Ne.symm hxn1)]; exact hcons1
        have hcons2' : (ctx_body.remove nm).lookup n2 = none := by
          rw [remove_lookup_ne (Ne.symm hxn2)]; exact hcons2
        exact CoreHasType.letPairE _ _ _ _ _ _ _ _ _ _ he' hdist hfresh1' hfresh2' hbody' hcons1' hcons2'

-- ============================================================================
-- Substitution preserves typing — convenience wrapper over subst_typing
-- ============================================================================

/-- Substitution lemma as needed by preservation's letVal case.
    Takes the value's typing at a general (ctx, ctx') pair: value_preserves_ctx
    forces ctx' = ctx internally, then subst_typing fires. -/
theorem subst_preserves_typing {n : Nat}
    {ctx ctx' ctx'' : CoreCtx n} {name : String} {v : CoreExpr n} {t_v : CoreTy n}
    {body : CoreExpr n} {t : CoreTy n}
    (hval : CoreHasType ctx v t_v ctx')
    (hfresh : ctx'.lookup name = none)
    (hbody : CoreHasType ((name, t_v) :: ctx') body t ctx'')
    (hconsumed : ctx''.lookup name = none)
    (hv : isValue v = true) :
    CoreHasType ctx (subst body name v) t ctx'' := by
  have hctx_eq := value_preserves_ctx hval hv
  subst hctx_eq
  have ht_v := value_any_ctx hv hval
  have hname_top : ∀ t', CoreCtx.lookup ((name, t_v) :: ctx) name = some t' → t' = t_v := by
    intro t' h; simp [CoreCtx.lookup, List.find?] at h; exact h.symm
  have h := subst_typing hv ht_v hbody hname_top
  rw [remove_cons_eq] at h
  rw [remove_of_lookup_none hfresh] at h
  rw [remove_of_lookup_none hconsumed] at h
  exact h

-- ============================================================================
-- Progress: a closed well-typed expression is either a value or can step
-- ============================================================================

/-- Progress theorem for `CoreHasType`. Every closed well-typed term either
    is already a value or can take a small step under `Step`.

    The `mergeFamily` and `finalizeFamily` cases use the double-witness
    substitution idiom (`subst hExpr; subst hTy`) to expose the parametric
    dispatchers in the goal, then `cases tag` to select the concrete Step
    constructor. Under option (a) (`.leafReduce (.groupVal s)` as the
    canonical value of `.reduced s`), the `.reduced`-tag branch of
    `mergeFamily` / `finalizeFamily` invokes `canonical_reduced` to force
    the shape — the gate's `Covers` clause will become load-bearing in
    `preservation`, not here. -/
theorem progress {n : Nat} {ctx e_ctx' : CoreCtx n} {e : CoreExpr n} {t : CoreTy n}
    (ht : CoreHasType ctx e t e_ctx') (hempty : ctx = []) :
    isValue e = true ∨ ∃ e', Step e e' := by
  induction ht with
  | groupVal _ _ => left; rfl
  | dataVal _ => left; rfl
  | unitVal _ => left; rfl
  | var _ _ _ hlook =>
    subst hempty; simp [CoreCtx.lookup, List.find?] at hlook
  | diverge _ _ g s pred hg ih =>
    have ih_g := ih hempty
    match ih_g with
    | .inl hv =>
      subst hempty
      have := canonical_group hg hv; subst this
      exact .inr ⟨_, Step.divergeVal s pred⟩
    | .inr ⟨g', hg'⟩ =>
      exact .inr ⟨_, Step.divergeCong g g' pred hg'⟩
  | letBind _ _ _ name val body _ _ hval _ _ _ ih_val _ =>
    have ih := ih_val hempty
    match ih with
    | .inl hv => exact .inr ⟨_, Step.letVal name val body hv⟩
    | .inr ⟨val', hval'⟩ =>
      exact .inr ⟨_, Step.letCong name val val' body hval'⟩
  | pairVal _ ctx_mid _ a b _ _ ha hb iha ihb =>
    have iha' := iha hempty
    match iha' with
    | .inr ⟨a', ha'⟩ =>
      exact .inr ⟨_, Step.pairLeftCong a a' b ha'⟩
    | .inl hva =>
      subst hempty
      have hctx := value_preserves_ctx ha hva
      have ihb' := ihb hctx.symm
      match ihb' with
      | .inl hvb =>
        left; simp [isValue]; exact ⟨hva, hvb⟩
      | .inr ⟨b', hb''⟩ =>
        exact .inr ⟨_, Step.pairRightCong a b b' hva hb''⟩
  | fstE _ _ e _ _ he ih =>
    have ih_e := ih hempty
    match ih_e with
    | .inl hv =>
      subst hempty
      have ⟨v1, v2, heq, hv1, hv2⟩ := canonical_pair he hv
      subst heq
      exact .inr ⟨v1, Step.fstVal v1 v2 hv1 hv2⟩
    | .inr ⟨e', he'⟩ =>
      exact .inr ⟨_, Step.fstCong e e' he'⟩
  | sndE _ _ e _ _ he ih =>
    have ih_e := ih hempty
    match ih_e with
    | .inl hv =>
      subst hempty
      have ⟨v1, v2, heq, hv1, hv2⟩ := canonical_pair he hv
      subst heq
      exact .inr ⟨v2, Step.sndVal v1 v2 hv1 hv2⟩
    | .inr ⟨e', he'⟩ =>
      exact .inr ⟨_, Step.sndCong e e' he'⟩
  | letPairE _ _ _ e _ _ body _ _ _ he _ _ _ _ _ _ ih_e _ =>
    have ih := ih_e hempty
    match ih with
    | .inl hv =>
      subst hempty
      have ⟨v1, v2, heq, hv1, hv2⟩ := canonical_pair he hv
      subst heq
      exact .inr ⟨_, Step.letPairVal _ _ v1 v2 body hv1 hv2⟩
    | .inr ⟨e', he'⟩ =>
      exact .inr ⟨_, Step.letPairCong e e' _ _ body he'⟩
  | write _ ctx_mid _ g payload s hg hpayload ih_g ih_p =>
    have ihg := ih_g hempty
    match ihg with
    | .inr ⟨g', hg'⟩ =>
      exact .inr ⟨_, Step.writeLeft g g' payload hg'⟩
    | .inl hvg =>
      subst hempty
      have hctx := value_preserves_ctx hg hvg
      have ihp := ih_p hctx.symm
      match ihp with
      | .inl hvp =>
        have h1 := canonical_group hg hvg; subst h1
        have hpayload' := hctx.symm ▸ hpayload
        have h2 := canonical_data hpayload' hvp; subst h2
        exact .inr ⟨_, Step.writeVal s⟩
      | .inr ⟨p', hp'⟩ =>
        exact .inr ⟨_, Step.writeRight g payload p' hvg hp'⟩
  | leafReduce _ _ g s hg ih =>
    have ih_g := ih hempty
    match ih_g with
    | .inl hvg =>
      subst hempty
      have := canonical_group hg hvg; subst this
      left; simp [isValue]
    | .inr ⟨g', hg'⟩ =>
      exact .inr ⟨_, Step.leafReduceCong g g' hg'⟩
  | mergeFamily tag _ ctx_mid _ e1 e2 _ s1 s2 _ _ hExpr hTy hw1 hw2 _ ih1 ih2 =>
    have ih1' := ih1 hempty
    match ih1' with
    | .inr ⟨e1', he1'⟩ =>
      rw [hExpr]
      cases tag
      · exact .inr ⟨_, Step.mergeLeft e1 e1' e2 he1'⟩
      · exact .inr ⟨_, Step.combineRedLeft e1 e1' e2 he1'⟩
    | .inl hv1 =>
      subst hempty
      have hctx := value_preserves_ctx hw1 hv1
      have ih2' := ih2 hctx.symm
      match ih2' with
      | .inr ⟨e2', he2'⟩ =>
        rw [hExpr]
        cases tag
        · exact .inr ⟨_, Step.mergeRight e1 e2 e2' hv1 he2'⟩
        · exact .inr ⟨_, Step.combineRedRight e1 e2 e2' hv1 he2'⟩
      | .inl hv2 =>
        rw [hExpr]
        cases tag
        · have h1 := canonical_group hw1 hv1; subst h1
          have hw2' : CoreHasType [] e2 (.group s2) _ := hctx ▸ hw2
          have h2 := canonical_group hw2' hv2; subst h2
          exact .inr ⟨_, Step.mergeVal s1 s2⟩
        · have h1 := canonical_reduced hw1 hv1; subst h1
          have hw2' : CoreHasType [] e2 (.reduced s2) _ := hctx ▸ hw2
          have h2 := canonical_reduced hw2' hv2; subst h2
          exact .inr ⟨_, Step.combineRedVal s1 s2⟩
  | finalizeFamily tag _ _ e _ _ hExpr hTy hw ih =>
    have ih' := ih hempty
    match ih' with
    | .inr ⟨e', he'⟩ =>
      rw [hExpr]
      cases tag
      · exact .inr ⟨_, Step.fenceCong e e' he'⟩
      · exact .inr ⟨_, Step.finalizeCong e e' he'⟩
    | .inl hv =>
      rw [hExpr]
      cases tag
      · subst hempty
        have := canonical_group hw hv; subst this
        exact .inr ⟨_, Step.fenceVal (PSet.all n)⟩
      · subst hempty
        have := canonical_reduced hw hv; subst this
        exact .inr ⟨_, Step.finalizeVal (PSet.all n)⟩

/-- Progress at the empty input context — the usual form. -/
theorem progress_closed {n : Nat} {e : CoreExpr n} {t : CoreTy n} {ctx' : CoreCtx n}
    (ht : CoreHasType [] e t ctx') :
    isValue e = true ∨ ∃ e', Step e e' :=
  progress ht rfl

-- ============================================================================
-- Preservation: if Γ ⊢ e : t ⊣ Γ' and e ⟶ e', then Γ ⊢ e' : t ⊣ Γ'
-- ============================================================================

/-- Preservation theorem for `CoreHasType`. Stepping a well-typed term
    yields a term of the same type at the same context pair.

    The structure of the proof mirrors `Metatheory.lean`'s `preservation`:
    induction on `hs` with contexts generalised, and inner `cases ht`
    per Step constructor to extract the typing structure.

    Every outer `cases ht` whose Step target is a concrete expression
    (e.g. `divergeVal` targets `.diverge (.groupVal s) pred`) produces
    mergeFamily and finalizeFamily dead branches alongside the real
    match. These are discharged by the pattern:
        `cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr`
    which case-splits the tag, unfolds the dispatcher, and constructor-
    clashes `hExpr` against the concrete expression shape.

    The parametric Step cases (`mergeVal`, `combineRedVal`, `fenceVal`,
    `finalizeVal`) are where the gate does metatheoretic work: the
    `mergeVal` / `combineRedVal` cases unpack `IsComplement`'s `Covers`
    clause to conclude `s1 ||| s2 = parent`, and the fence / finalize
    cases use `canonical_group` / `canonical_reduced` to force the
    `PSet.all n` gate from the typing rule. -/
theorem preservation {n : Nat} {e e' : CoreExpr n} {t : CoreTy n} {ctx ctx' : CoreCtx n}
    (ht : CoreHasType ctx e t ctx') (hs : Step e e') :
    CoreHasType ctx e' t ctx' := by
  induction hs generalizing t ctx ctx' with
  -- ── Value-producing reductions ──
  | divergeVal s pred =>
    cases ht with
    | diverge _ _ _ _ _ hg =>
      cases hg with
      | groupVal _ _ =>
        exact CoreHasType.pairVal _ _ _ _ _ _ _
          (CoreHasType.groupVal _ _) (CoreHasType.groupVal _ _)
      | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
        cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
      | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
        cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | mergeVal s1 s2 =>
    cases ht with
    | mergeFamily tag _ _ _ _ _ _ _ _ parent _ hExpr hTy hw1 hw2 hcomp =>
      cases tag with
      | group =>
        simp only [tagToMergeExpr] at hExpr
        simp only [tagToTy] at hTy
        injection hExpr with he1 he2
        subst he1; subst he2; subst hTy
        cases hw1 with
        | groupVal _ _ =>
          cases hw2 with
          | groupVal _ _ =>
            have ⟨_, hcov⟩ := hcomp
            unfold PSet.Covers at hcov
            rw [hcov]
            exact CoreHasType.groupVal _ _
          | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
            cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
          | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
            cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
        | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
          cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
        | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
          cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
      | reduced =>
        simp only [tagToMergeExpr] at hExpr
        cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | combineRedVal s1 s2 =>
    cases ht with
    | mergeFamily tag _ _ _ _ _ _ _ _ parent _ hExpr hTy hw1 hw2 hcomp =>
      cases tag with
      | group =>
        simp only [tagToMergeExpr] at hExpr
        cases hExpr
      | reduced =>
        simp only [tagToMergeExpr] at hExpr
        simp only [tagToTy] at hTy
        injection hExpr with he1 he2
        subst he1; subst he2; subst hTy
        -- hw1 : CoreHasType ctx (.leafReduce (.groupVal s1)) (.reduced s1) ctx_mid
        -- hw2 : CoreHasType ctx_mid (.leafReduce (.groupVal s2)) (.reduced s2) ctx'
        cases hw1 with
        | leafReduce _ _ _ _ hg1 =>
          cases hg1 with
          | groupVal _ _ =>
            cases hw2 with
            | leafReduce _ _ _ _ hg2 =>
              cases hg2 with
              | groupVal _ _ =>
                have ⟨_, hcov⟩ := hcomp
                unfold PSet.Covers at hcov
                rw [hcov]
                exact CoreHasType.leafReduce _ _ _ _ (CoreHasType.groupVal _ _)
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
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | writeVal s =>
    cases ht with
    | write _ _ _ _ _ _ hg hpayload =>
      cases hg with
      | groupVal _ _ =>
        cases hpayload with
        | dataVal _ => exact CoreHasType.groupVal _ _
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
  | fenceVal s =>
    cases ht with
    | finalizeFamily tag _ _ _ _ _ hExpr hTy hg =>
      cases tag with
      | group =>
        simp only [tagToFinalExpr] at hExpr
        simp only [tagToFinalTy] at hTy
        injection hExpr with he
        subst he; subst hTy
        -- Now hg : CoreHasType ctx (.groupVal s) (.group (PSet.all n)) ctx'
        cases hg with
        | groupVal _ _ => exact CoreHasType.unitVal _
        | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
          cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
        | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
          cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
      | reduced =>
        simp only [tagToFinalExpr] at hExpr
        cases hExpr
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
  | finalizeVal s =>
    cases ht with
    | finalizeFamily tag _ _ _ _ _ hExpr hTy hr =>
      cases tag with
      | group =>
        simp only [tagToFinalExpr] at hExpr
        cases hExpr
      | reduced =>
        simp only [tagToFinalExpr] at hExpr
        simp only [tagToFinalTy] at hTy
        injection hExpr with he
        subst he; subst hTy
        -- Now hr : CoreHasType ctx (.leafReduce (.groupVal s)) (.reduced (PSet.all n)) ctx'
        cases hr with
        | leafReduce _ _ _ _ hg =>
          cases hg with
          | groupVal _ _ => exact CoreHasType.dataVal _
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
  | letVal name v body hv =>
    cases ht with
    | letBind _ _ _ _ _ _ _ _ hval hfresh hbody hconsumed =>
      exact subst_preserves_typing hval hfresh hbody hconsumed hv
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | fstVal a b hva hvb =>
    cases ht with
    | fstE _ _ _ _ _ he =>
      cases he with
      | pairVal _ ctx_mid _ _ _ _ _ ha hb =>
        have := value_preserves_ctx hb hvb
        subst this
        exact ha
      | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
        cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
      | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
        cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | sndVal a b hva hvb =>
    cases ht with
    | sndE _ _ _ _ _ he =>
      cases he with
      | pairVal _ ctx_mid _ _ _ _ _ ha hb =>
        have := value_preserves_ctx ha hva
        subst this
        exact hb
      | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
        cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
      | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
        cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | letPairVal name1 name2 v1 v2 body hv1 hv2 =>
    cases ht with
    | letPairE _ _ _ _ _ _ _ t1 t2 _ he hdist hfresh1 hfresh2 hbody hcons1 hcons2 =>
      cases he with
      | pairVal _ ctx_a _ _ _ _ _ ha hb =>
        have hctx_a := value_preserves_ctx ha hv1; subst hctx_a
        have hctx_mid := value_preserves_ctx hb hv2; subst hctx_mid
        have ht_v1 := value_any_ctx hv1 ha
        have hname_top : ∀ t', CoreCtx.lookup ((name2, t2) :: (name1, t1) :: ctx) name1 = some t' → t' = t1 := by
          intro t' h
          rw [lookup_cons_ne (Ne.symm hdist)] at h
          simp [lookup_cons_eq] at h
          exact h.symm
        have h1 := subst_typing hv1 ht_v1 hbody hname_top
        rw [remove_cons_ne hdist, remove_cons_eq,
            remove_of_lookup_none hfresh1] at h1
        rw [remove_of_lookup_none hcons1] at h1
        exact subst_preserves_typing hb hfresh2 h1 hcons2 hv2
      | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
        cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
      | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
        cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  -- ── Congruence cases ──
  | divergeCong g g' pred _ ih =>
    cases ht with
    | diverge _ _ _ s _ hw =>
      exact CoreHasType.diverge _ _ _ _ _ (ih hw)
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | mergeLeft g1 g1' g2 _ ih =>
    cases ht with
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr hTy hw1 hw2 hcomp =>
      cases tag with
      | group =>
        simp only [tagToMergeExpr] at hExpr
        simp only [tagToTy] at hTy
        injection hExpr with he1 he2
        subst he1; subst he2; subst hTy
        exact CoreHasType.mergeFamily .group _ _ _ _ _ _ _ _ _ _ rfl rfl
          (ih hw1) hw2 hcomp
      | reduced =>
        simp only [tagToMergeExpr] at hExpr
        cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | mergeRight v1 g2 g2' _ _ ih =>
    cases ht with
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr hTy hw1 hw2 hcomp =>
      cases tag with
      | group =>
        simp only [tagToMergeExpr] at hExpr
        simp only [tagToTy] at hTy
        injection hExpr with he1 he2
        subst he1; subst he2; subst hTy
        exact CoreHasType.mergeFamily .group _ _ _ _ _ _ _ _ _ _ rfl rfl
          hw1 (ih hw2) hcomp
      | reduced =>
        simp only [tagToMergeExpr] at hExpr
        cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | combineRedLeft r1 r1' r2 _ ih =>
    cases ht with
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr hTy hw1 hw2 hcomp =>
      cases tag with
      | group =>
        simp only [tagToMergeExpr] at hExpr
        cases hExpr
      | reduced =>
        simp only [tagToMergeExpr] at hExpr
        simp only [tagToTy] at hTy
        injection hExpr with he1 he2
        subst he1; subst he2; subst hTy
        exact CoreHasType.mergeFamily .reduced _ _ _ _ _ _ _ _ _ _ rfl rfl
          (ih hw1) hw2 hcomp
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | combineRedRight v1 r2 r2' _ _ ih =>
    cases ht with
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr hTy hw1 hw2 hcomp =>
      cases tag with
      | group =>
        simp only [tagToMergeExpr] at hExpr
        cases hExpr
      | reduced =>
        simp only [tagToMergeExpr] at hExpr
        simp only [tagToTy] at hTy
        injection hExpr with he1 he2
        subst he1; subst he2; subst hTy
        exact CoreHasType.mergeFamily .reduced _ _ _ _ _ _ _ _ _ _ rfl rfl
          hw1 (ih hw2) hcomp
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | letCong name val val' body _ ih =>
    cases ht with
    | letBind _ _ _ _ _ _ _ _ hval hfresh hbody hconsumed =>
      exact CoreHasType.letBind _ _ _ _ _ _ _ _ (ih hval) hfresh hbody hconsumed
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | pairLeftCong a a' b _ ih =>
    cases ht with
    | pairVal _ _ _ _ _ _ _ ha hb =>
      exact CoreHasType.pairVal _ _ _ _ _ _ _ (ih ha) hb
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | pairRightCong a b b' _ _ ih =>
    cases ht with
    | pairVal _ _ _ _ _ _ _ ha hb =>
      exact CoreHasType.pairVal _ _ _ _ _ _ _ ha (ih hb)
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | fstCong e e' _ ih =>
    cases ht with
    | fstE _ _ _ _ _ he =>
      exact CoreHasType.fstE _ _ _ _ _ (ih he)
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | sndCong e e' _ ih =>
    cases ht with
    | sndE _ _ _ _ _ he =>
      exact CoreHasType.sndE _ _ _ _ _ (ih he)
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | letPairCong e e' name1 name2 body _ ih =>
    cases ht with
    | letPairE _ _ _ _ _ _ _ _ _ _ he hdist hfresh1 hfresh2 hbody hcons1 hcons2 =>
      exact CoreHasType.letPairE _ _ _ _ _ _ _ _ _ _ (ih he) hdist hfresh1 hfresh2 hbody hcons1 hcons2
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | writeLeft g g' payload _ ih =>
    cases ht with
    | write _ _ _ _ _ _ hg hpayload =>
      exact CoreHasType.write _ _ _ _ _ _ (ih hg) hpayload
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | writeRight v payload payload' _ _ ih =>
    cases ht with
    | write _ _ _ _ _ _ hg hpayload =>
      exact CoreHasType.write _ _ _ _ _ _ hg (ih hpayload)
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | leafReduceCong g g' _ ih =>
    cases ht with
    | leafReduce _ _ _ _ hg =>
      exact CoreHasType.leafReduce _ _ _ _ (ih hg)
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
  | fenceCong g g' _ ih =>
    cases ht with
    | finalizeFamily tag _ _ _ _ _ hExpr hTy hg =>
      cases tag with
      | group =>
        simp only [tagToFinalExpr] at hExpr
        simp only [tagToFinalTy] at hTy
        injection hExpr with he
        subst he; subst hTy
        exact CoreHasType.finalizeFamily .group _ _ _ _ _ rfl rfl (ih hg)
      | reduced =>
        simp only [tagToFinalExpr] at hExpr
        cases hExpr
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
  | finalizeCong r r' _ ih =>
    cases ht with
    | finalizeFamily tag _ _ _ _ _ hExpr hTy hr =>
      cases tag with
      | group =>
        simp only [tagToFinalExpr] at hExpr
        cases hExpr
      | reduced =>
        simp only [tagToFinalExpr] at hExpr
        simp only [tagToFinalTy] at hTy
        injection hExpr with he
        subst he; subst hTy
        exact CoreHasType.finalizeFamily .reduced _ _ _ _ _ rfl rfl (ih hr)
    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr

end CoreMetatheory
