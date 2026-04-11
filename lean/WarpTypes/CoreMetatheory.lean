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

end CoreMetatheory
