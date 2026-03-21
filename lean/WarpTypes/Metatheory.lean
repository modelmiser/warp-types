import WarpTypes.Basic

/-
  Metatheory: Progress and Preservation for Warp Typestate

  - Capture-avoiding substitution
  - Small-step reduction (Step)
  - Canonical forms lemmas
  - Progress theorem
  - Preservation theorem (substitution lemma proved via context removal)
  - Untypability proofs for 5 documented GPU bugs
-/

-- ============================================================================
-- Substitution
-- ============================================================================

def subst (e : Expr) (x : String) (v : Expr) : Expr :=
  match e with
  | .warpVal s => .warpVal s
  | .perLaneVal => .perLaneVal
  | .unitVal => .unitVal
  | .var name => if name == x then v else .var name
  | .diverge w pred => .diverge (subst w x v) pred
  | .merge w1 w2 => .merge (subst w1 x v) (subst w2 x v)
  | .shuffle w data => .shuffle (subst w x v) (subst data x v)
  | .letBind name val body =>
      if name == x then .letBind name (subst val x v) body
      else .letBind name (subst val x v) (subst body x v)
  | .pairVal a b => .pairVal (subst a x v) (subst b x v)
  | .fst e => .fst (subst e x v)
  | .snd e => .snd (subst e x v)
  | .letPair e n1 n2 body =>
      if n1 == x || n2 == x then .letPair (subst e x v) n1 n2 body
      else .letPair (subst e x v) n1 n2 (subst body x v)

-- ============================================================================
-- Small-Step Reduction
-- ============================================================================

inductive Step : Expr → Expr → Prop
  | divergeVal (s pred : ActiveSet) :
      Step (.diverge (.warpVal s) pred)
           (.pairVal (.warpVal (s &&& pred)) (.warpVal (s &&& ~~~pred)))
  | mergeVal (s1 s2 : ActiveSet) :
      Step (.merge (.warpVal s1) (.warpVal s2)) (.warpVal (s1 ||| s2))
  | shuffleVal (s : ActiveSet) :
      Step (.shuffle (.warpVal s) .perLaneVal) .perLaneVal
  | letVal (name : String) (v body : Expr) :
      isValue v = true →
      Step (.letBind name v body) (subst body name v)
  | fstVal (a b : Expr) :
      isValue a = true → isValue b = true →
      Step (.fst (.pairVal a b)) a
  | sndVal (a b : Expr) :
      isValue a = true → isValue b = true →
      Step (.snd (.pairVal a b)) b
  | divergeCong (w w' : Expr) (pred : ActiveSet) :
      Step w w' → Step (.diverge w pred) (.diverge w' pred)
  | mergeLeft (w1 w1' w2 : Expr) :
      Step w1 w1' → Step (.merge w1 w2) (.merge w1' w2)
  | mergeRight (v1 w2 w2' : Expr) :
      isValue v1 = true → Step w2 w2' →
      Step (.merge v1 w2) (.merge v1 w2')
  | shuffleLeft (w w' data : Expr) :
      Step w w' → Step (.shuffle w data) (.shuffle w' data)
  | shuffleRight (v data data' : Expr) :
      isValue v = true → Step data data' →
      Step (.shuffle v data) (.shuffle v data')
  | letCong (name : String) (val val' body : Expr) :
      Step val val' →
      Step (.letBind name val body) (.letBind name val' body)
  | pairLeftCong (a a' b : Expr) :
      Step a a' → Step (.pairVal a b) (.pairVal a' b)
  | pairRightCong (a b b' : Expr) :
      isValue a = true → Step b b' →
      Step (.pairVal a b) (.pairVal a b')
  | fstCong (e e' : Expr) :
      Step e e' → Step (.fst e) (.fst e')
  | sndCong (e e' : Expr) :
      Step e e' → Step (.snd e) (.snd e')
  | letPairVal (name1 name2 : String) (v1 v2 body : Expr) :
      isValue v1 = true → isValue v2 = true →
      Step (.letPair (.pairVal v1 v2) name1 name2 body)
           (subst (subst body name1 v1) name2 v2)
  | letPairCong (e e' : Expr) (name1 name2 : String) (body : Expr) :
      Step e e' → Step (.letPair e name1 name2 body) (.letPair e' name1 name2 body)

-- ============================================================================
-- Values preserve contexts (values don't consume linear resources)
-- ============================================================================

theorem value_preserves_ctx {ctx ctx' : Ctx} {v : Expr} {t : Ty}
    (ht : HasType ctx v t ctx') (hv : isValue v = true) : ctx = ctx' := by
  match ht with
  | .warpVal _ _ => rfl
  | .perLaneVal _ => rfl
  | .unitVal _ => rfl
  | .pairVal _ ctx_mid _ a b _ _ ha hb =>
    simp [isValue] at hv
    have h1 := value_preserves_ctx ha hv.1
    have h2 := value_preserves_ctx hb hv.2
    subst h1; subst h2; rfl
  | .var _ _ _ _ => simp [isValue] at hv
  | .diverge _ _ _ _ _ _ => simp [isValue] at hv
  | .merge _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .shuffle _ _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv

-- ============================================================================
-- Canonical Forms
-- ============================================================================

theorem canonical_warp {e : Expr} {s : ActiveSet} {ctx' : Ctx}
    (ht : HasType [] e (.warp s) ctx') (hv : isValue e = true) :
    e = .warpVal s := by
  match ht with
  | .warpVal _ _ => rfl
  | .var _ _ _ hlook => simp [Ctx.lookup, List.find?] at hlook
  | .merge _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv

theorem canonical_perLane {e : Expr} {ctx' : Ctx}
    (ht : HasType [] e .perLane ctx') (hv : isValue e = true) :
    e = .perLaneVal := by
  match ht with
  | .perLaneVal _ => rfl
  | .var _ _ _ hlook => simp [Ctx.lookup, List.find?] at hlook
  | .shuffle _ _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv

theorem canonical_pair {e : Expr} {t1 t2 : Ty} {ctx' : Ctx}
    (ht : HasType [] e (.pair t1 t2) ctx') (hv : isValue e = true) :
    ∃ v1 v2, e = .pairVal v1 v2 ∧ isValue v1 = true ∧ isValue v2 = true := by
  match ht with
  | .pairVal _ _ _ a b _ _ _ _ =>
    simp [isValue] at hv; exact ⟨a, b, rfl, hv.1, hv.2⟩
  | .var _ _ _ hlook => simp [Ctx.lookup, List.find?] at hlook
  | .diverge _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv

-- ============================================================================
-- Progress
-- ============================================================================

/-- Progress: A closed well-typed expression is either a value or can step. -/
theorem progress {e : Expr} {t : Ty} {ctx' : Ctx}
    (ht : HasType [] e t ctx') :
    isValue e = true ∨ ∃ e', Step e e' := by
  match ht with
  | .warpVal _ _ => left; rfl
  | .perLaneVal _ => left; rfl
  | .unitVal _ => left; rfl
  | .var _ _ _ hlook => simp [Ctx.lookup, List.find?] at hlook
  | .diverge _ _ w s pred hw =>
    have ih_w := progress hw
    match ih_w with
    | .inl hv =>
      have := canonical_warp hw hv; subst this
      exact .inr ⟨_, Step.divergeVal s pred⟩
    | .inr ⟨w', hw'⟩ =>
      exact .inr ⟨_, Step.divergeCong w w' pred hw'⟩
  | .merge _ ctx_mid _ w1 w2 s1 s2 _ hw1 hw2 hcomp =>
    have ih1 := progress hw1
    match ih1 with
    | .inr ⟨w1', hw1'⟩ =>
      exact .inr ⟨_, Step.mergeLeft w1 w1' w2 hw1'⟩
    | .inl hv1 =>
      -- w1 is a value, so ctx_mid = [] by value_preserves_ctx
      have hctx := value_preserves_ctx hw1 hv1
      -- Now hw2 has empty input context
      have hw2' : HasType [] w2 (.warp s2) ctx' := hctx ▸ hw2
      have ih2 := progress hw2'
      match ih2 with
      | .inl hv2 =>
        have h1 := canonical_warp hw1 hv1; subst h1
        have h2 := canonical_warp hw2' hv2; subst h2
        exact .inr ⟨_, Step.mergeVal s1 s2⟩
      | .inr ⟨w2', hw2'⟩ =>
        exact .inr ⟨_, Step.mergeRight w1 w2 w2' hv1 hw2'⟩
  | .shuffle _ ctx_mid _ w data hw hd =>
    have ihw := progress hw
    match ihw with
    | .inr ⟨w', hw'⟩ =>
      exact .inr ⟨_, Step.shuffleLeft w w' data hw'⟩
    | .inl hv =>
      have hctx := value_preserves_ctx hw hv
      have hd' : HasType [] data .perLane ctx' := hctx ▸ hd
      have ihd := progress hd'
      match ihd with
      | .inl hvd =>
        have h1 := canonical_warp hw hv; subst h1
        have h2 := canonical_perLane hd' hvd; subst h2
        exact .inr ⟨_, Step.shuffleVal ActiveSet.all⟩
      | .inr ⟨d', hd''⟩ =>
        exact .inr ⟨_, Step.shuffleRight w data d' hv hd''⟩
  | .letBind _ _ _ name val body _ _ hval _ hbody _ =>
    have ihv := progress hval
    match ihv with
    | .inl hv => exact .inr ⟨_, Step.letVal name val body hv⟩
    | .inr ⟨val', hval'⟩ =>
      exact .inr ⟨_, Step.letCong name val val' body hval'⟩
  | .pairVal _ ctx_mid _ a b _ _ ha hb =>
    have iha := progress ha
    match iha with
    | .inr ⟨a', ha'⟩ =>
      exact .inr ⟨_, Step.pairLeftCong a a' b ha'⟩
    | .inl hva =>
      have hctx := value_preserves_ctx ha hva
      have hb' : HasType [] b _ ctx' := hctx ▸ hb
      have ihb := progress hb'
      match ihb with
      | .inl hvb =>
        left; simp [isValue]; exact ⟨hva, hvb⟩
      | .inr ⟨b', hb''⟩ =>
        exact .inr ⟨_, Step.pairRightCong a b b' hva hb''⟩
  | .fstE _ _ e _ _ he =>
    have ihe := progress he
    match ihe with
    | .inl hv =>
      have ⟨v1, v2, heq, hv1, hv2⟩ := canonical_pair he hv
      subst heq
      exact .inr ⟨v1, Step.fstVal v1 v2 hv1 hv2⟩
    | .inr ⟨e', he'⟩ =>
      exact .inr ⟨_, Step.fstCong e e' he'⟩
  | .sndE _ _ e _ _ he =>
    have ihe := progress he
    match ihe with
    | .inl hv =>
      have ⟨v1, v2, heq, hv1, hv2⟩ := canonical_pair he hv
      subst heq
      exact .inr ⟨v2, Step.sndVal v1 v2 hv1 hv2⟩
    | .inr ⟨e', he'⟩ =>
      exact .inr ⟨_, Step.sndCong e e' he'⟩
  | .letPairE _ _ _ e _ _ body _ _ _ he _ _ _ hbody _ _ =>
    have ihe := progress he
    match ihe with
    | .inl hv =>
      have ⟨v1, v2, heq, hv1, hv2⟩ := canonical_pair he hv
      subst heq
      exact .inr ⟨_, Step.letPairVal _ _ v1 v2 body hv1 hv2⟩
    | .inr ⟨e', he'⟩ =>
      exact .inr ⟨_, Step.letPairCong e e' _ _ body he'⟩

-- ============================================================================
-- Preservation
-- ============================================================================

-- ============================================================================
-- Context infrastructure lemmas
-- ============================================================================

/-- Looking up a name in a context from which it was removed yields none. -/
theorem remove_lookup_self (ctx : Ctx) (name : String) :
    (ctx.remove name).lookup name = none := by
  induction ctx with
  | nil => rfl
  | cons p tl ih =>
    simp only [Ctx.remove, Ctx.lookup]
    unfold List.filter
    by_cases h : (p.1 != name) = true
    · -- p kept: but p.1 ≠ name, so find? skips it
      simp only [h, List.find?]
      have hne : (p.1 == name) = false := by
        simp only [bne_iff_ne, ne_eq, beq_iff_eq] at h; simp [beq_iff_eq, h]
      simp only [hne, Option.map, Ctx.lookup, Ctx.remove] at ih ⊢
      exact ih
    · -- p removed (p.1 = name)
      simp only [Bool.not_eq_true] at h; simp only [h]
      exact ih

/-- Looking up a different name is unaffected by remove. -/
theorem remove_lookup_ne {name name' : String} (hne : name ≠ name') (ctx : Ctx) :
    (ctx.remove name).lookup name' = ctx.lookup name' := by
  induction ctx with
  | nil => rfl
  | cons p tl ih =>
    simp only [Ctx.remove, Ctx.lookup]
    by_cases hp : p.1 = name
    · -- p.1 = name, so p is filtered out (bne = false)
      have h_bne : (p.1 != name) = false := by rw [hp]; simp [bne_iff_ne]
      simp only [List.filter, h_bne]
      -- p would be skipped by find? anyway (p.1 = name ≠ name')
      have h_beq : (p.1 == name') = false := by rw [hp]; simp [beq_iff_eq, hne]
      simp only [List.find?, h_beq]
      exact ih
    · -- p.1 ≠ name, so p is kept
      have h_bne : (p.1 != name) = true := by simp [bne_iff_ne, hp]
      simp only [List.filter, h_bne, List.find?]
      by_cases hq : p.1 = name'
      · simp [beq_iff_eq, hq]
      · have h_beq : (p.1 == name') = false := by simp [beq_iff_eq, hq]
        simp only [h_beq]
        exact ih

/-- Removing from a context where the name is absent is identity. -/
theorem remove_of_lookup_none {ctx : Ctx} {name : String}
    (h : ctx.lookup name = none) : ctx.remove name = ctx := by
  induction ctx with
  | nil => rfl
  | cons p tl ih =>
    simp only [Ctx.lookup, List.find?] at h
    by_cases hq : (p.1 == name) = true
    · simp [hq, Option.map] at h
    · simp only [Bool.not_eq_true] at hq
      simp only [hq] at h
      simp only [Ctx.remove, List.filter]
      have : (p.1 != name) = true := by simp [bne_iff_ne, ne_eq, beq_iff_eq]; simp [beq_iff_eq] at hq; exact hq
      simp only [this]
      congr 1
      exact ih h

/-- Lookup on cons with matching name. -/
theorem lookup_cons_eq (name : String) (t : Ty) (ctx : Ctx) :
    Ctx.lookup ((name, t) :: ctx) name = some t := by
  simp [Ctx.lookup, List.find?, beq_self_eq_true]

/-- Lookup on cons with different name. -/
theorem lookup_cons_ne {name name' : String} (hne : name ≠ name') (t : Ty) (ctx : Ctx) :
    Ctx.lookup ((name, t) :: ctx) name' = ctx.lookup name' := by
  simp only [Ctx.lookup, List.find?]
  have : (name == name') = false := by simp [beq_iff_eq, hne]
  simp [this]

/-- Remove on cons with matching name. -/
theorem remove_cons_eq (name : String) (t : Ty) (ctx : Ctx) :
    Ctx.remove ((name, t) :: ctx) name = Ctx.remove ctx name := by
  simp only [Ctx.remove, List.filter]
  have : ((name, t).1 != name) = false := by simp [bne_iff_ne, ne_eq]
  simp [this]

/-- Remove on cons with different name. -/
theorem remove_cons_ne {name name' : String} (hne : name ≠ name') (t : Ty) (ctx : Ctx) :
    Ctx.remove ((name', t) :: ctx) name = (name', t) :: Ctx.remove ctx name := by
  simp only [Ctx.remove, List.filter]
  have : ((name', t).1 != name) = true := by simp [bne_iff_ne, ne_eq]; exact Ne.symm hne
  simp [this]

/-- Remove commutes: order of removal doesn't matter. -/
theorem remove_comm (ctx : Ctx) (a b : String) :
    Ctx.remove (Ctx.remove ctx a) b = Ctx.remove (Ctx.remove ctx b) a := by
  simp only [Ctx.remove, List.filter_filter]
  congr 1; ext p; simp [Bool.and_comm]

-- ============================================================================
-- Values can be typed in any context
-- ============================================================================

/-- Helper: values produce any-context typing (avoids index constraint). -/
private theorem value_any_ctx_aux {v : Expr} {t : Ty} {ctx₁ ctx₂ : Ctx}
    (hv : isValue v = true)
    (ht : HasType ctx₁ v t ctx₂) :
    ∀ ctx₃, HasType ctx₃ v t ctx₃ := by
  match ht with
  | .warpVal _ s => intro ctx₃; exact HasType.warpVal ctx₃ s
  | .perLaneVal _ => intro ctx₃; exact HasType.perLaneVal ctx₃
  | .unitVal _ => intro ctx₃; exact HasType.unitVal ctx₃
  | .pairVal _ _ _ a b _ _ ha hb =>
    simp [isValue] at hv
    have iha := value_any_ctx_aux hv.1 ha
    have ihb := value_any_ctx_aux hv.2 hb
    intro ctx₃
    exact HasType.pairVal ctx₃ ctx₃ ctx₃ a b _ _ (iha ctx₃) (ihb ctx₃)
  | .var _ _ _ _ => simp [isValue] at hv
  | .diverge _ _ _ _ _ _ => simp [isValue] at hv
  | .merge _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .shuffle _ _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv
  | .letPairE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv

/-- Values can be typed in any context, producing the same context unchanged. -/
theorem value_any_ctx {v : Expr} {t : Ty} {ctx₁ : Ctx}
    (hv : isValue v = true)
    (ht : HasType ctx₁ v t ctx₁) :
    ∀ ctx₂, HasType ctx₂ v t ctx₂ :=
  value_any_ctx_aux hv ht

-- ============================================================================
-- Output context bindings come from input context
-- ============================================================================

/-- Any binding in the output context was present (with the same type) in input. -/
theorem output_binding_from_input {ctx ctx' : Ctx} {e : Expr} {t : Ty}
    (ht : HasType ctx e t ctx')
    {x : String} {tx : Ty}
    (hout : ctx'.lookup x = some tx) :
    ctx.lookup x = some tx := by
  induction ht with
  | warpVal _ _ => exact hout
  | perLaneVal _ => exact hout
  | unitVal _ => exact hout
  | var ctx₀ name₀ t₀ hlook =>
    -- ctx' = ctx₀.remove name₀
    by_cases hxn : x = name₀
    · subst hxn; rw [remove_lookup_self] at hout; exact absurd hout (by simp)
    · rwa [remove_lookup_ne (Ne.symm hxn)] at hout
  | diverge _ _ _ _ _ _ ih => exact ih hout
  | merge _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1 (ih2 hout)
  | shuffle _ _ _ _ _ _ _ ih1 ih2 => exact ih1 (ih2 hout)
  | letBind ctx₀ ctx_mid ctx_body name' _ _ _ _ _ hfresh₀ _ hcons₀ ih_val ih_body =>
    -- ctx' = ctx_body, ctx = ctx₀
    -- hout : ctx_body.lookup x = some tx
    -- hcons₀ : ctx_body.lookup name' = none
    -- ih_body : ctx_body.lookup x = some tx → ((name', _) :: ctx_mid).lookup x = some tx
    -- ih_val : ((name', _) :: ctx_mid).lookup x = some tx → ctx₀.lookup x = some tx
    -- But ih_body's premise is about the output of body's typing, which IS ctx_body = ctx'
    -- Actually the IH is already correctly instantiated by `induction`:
    -- ih_body gives us binding in (name', t1) :: ctx_mid
    -- ih_val gives us binding in ctx₀
    have h_body := ih_body hout
    -- h_body : ((name', _) :: ctx_mid).lookup x = some tx
    -- Need to strip the (name', _) prefix. x ≠ name' because x is in ctx_body but name' isn't
    by_cases hxn : x = name'
    · -- x = name', but ctx_body.lookup name' = none contradicts hout
      subst hxn; rw [hcons₀] at hout; exact absurd hout (by simp)
    · -- x ≠ name', so lookup skips the cons
      rw [lookup_cons_ne (Ne.symm hxn)] at h_body
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

-- ============================================================================
-- Generalized Substitution Lemma
-- ============================================================================

/-- Substitution for the var case (extracted to avoid scoping issues). -/
private theorem subst_typing_var
    {nm : String} {t_v : Ty} {v : Expr}
    (hv : isValue v = true)
    (ht_v : ∀ ctx₂, HasType ctx₂ v t_v ctx₂)
    {ctx₀ : Ctx} {name' : String} {t' : Ty}
    (hlook : ctx₀.lookup name' = some t')
    (hname : ∀ t'', ctx₀.lookup nm = some t'' → t'' = t_v) :
    HasType (ctx₀.remove nm) (subst (.var name') nm v) t'
            (Ctx.remove (ctx₀.remove name') nm) := by
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
    have hgoal := HasType.var (ctx₀.remove nm) name' t' hlook'
    rw [remove_comm] at hgoal
    exact hgoal

/-- Core substitution theorem: substituting a value for a name removes that
    name's binding from both input and output contexts.

    This generalizes over WHERE in the context the binding appears, which is
    essential for the merge/shuffle/pair cases where the binding may have
    been threaded past the first sub-expression. -/
theorem subst_typing
    {nm : String} {t_v : Ty} {v : Expr}
    (hv : isValue v = true)
    (ht_v : ∀ ctx₂, HasType ctx₂ v t_v ctx₂)
    {ctx : Ctx} {e : Expr} {t : Ty} {ctx' : Ctx}
    (hte : HasType ctx e t ctx')
    (hname : ∀ t', ctx.lookup nm = some t' → t' = t_v) :
    HasType (ctx.remove nm) (subst e nm v) t (ctx'.remove nm) :=
  match hte with
  | .warpVal _ s => by simp [subst]; exact HasType.warpVal _ s
  | .perLaneVal _ => by simp [subst]; exact HasType.perLaneVal _
  | .unitVal _ => by simp [subst]; exact HasType.unitVal _
  | .var _ _ _ hlook => subst_typing_var hv ht_v hlook hname
  | .diverge _ _ _ _ _ hw => by
    simp [subst]
    exact HasType.diverge _ _ _ _ _ (subst_typing hv ht_v hw hname)
  | .merge _ ctx_mid _ _ _ _ _ _ hw1 hw2 hcomp => by
    simp [subst]
    have hname_mid : ∀ t', ctx_mid.lookup nm = some t' → t' = t_v := by
      intro t' hl; exact hname t' (output_binding_from_input hw1 hl)
    exact HasType.merge _ _ _ _ _ _ _ _
      (subst_typing hv ht_v hw1 hname) (subst_typing hv ht_v hw2 hname_mid) hcomp
  | .shuffle _ ctx_mid _ _ _ hw hd => by
    simp [subst]
    have hname_mid : ∀ t', ctx_mid.lookup nm = some t' → t' = t_v := by
      intro t' hl; exact hname t' (output_binding_from_input hw hl)
    exact HasType.shuffle _ _ _ _ _
      (subst_typing hv ht_v hw hname) (subst_typing hv ht_v hd hname_mid)
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
      exact HasType.letBind _ _ _ _ _ _ _ _ hval' hfresh hbody hconsumed
    · -- name' ≠ nm: both val and body substituted
      simp [show (name' == nm) = false from by simp [beq_iff_eq, hxn]]
      have hval' := subst_typing hv ht_v hval hname
      have hfresh' : (ctx_mid.remove nm).lookup name' = none := by
        rw [remove_lookup_ne (Ne.symm hxn)]; exact hfresh
      have hname_body : ∀ t', Ctx.lookup ((name', t1) :: ctx_mid) nm = some t' → t' = t_v := by
        intro t' hl
        have : Ctx.lookup ctx_mid nm = some t' := by
          rwa [lookup_cons_ne hxn] at hl
        exact hname t' (output_binding_from_input hval this)
      have hbody' := subst_typing hv ht_v hbody hname_body
      rw [remove_cons_ne (Ne.symm hxn)] at hbody'
      have hconsumed' : (ctx_body.remove nm).lookup name' = none := by
        rw [remove_lookup_ne (Ne.symm hxn)]; exact hconsumed
      exact HasType.letBind _ _ _ _ _ _ _ _ hval' hfresh' hbody' hconsumed'
  | .pairVal _ ctx_mid _ _ _ _ _ ha hb => by
    simp [subst]
    have hname_mid : ∀ t', ctx_mid.lookup nm = some t' → t' = t_v := by
      intro t' hl; exact hname t' (output_binding_from_input ha hl)
    exact HasType.pairVal _ _ _ _ _ _ _
      (subst_typing hv ht_v ha hname) (subst_typing hv ht_v hb hname_mid)
  | .fstE _ _ _ _ _ he => by
    simp [subst]; exact HasType.fstE _ _ _ _ _ (subst_typing hv ht_v he hname)
  | .sndE _ _ _ _ _ he => by
    simp [subst]; exact HasType.sndE _ _ _ _ _ (subst_typing hv ht_v he hname)
  | .letPairE _ ctx_mid ctx_body _ n1 n2 body t1 t2 _ he hdist hfresh1 hfresh2 hbody hcons1 hcons2 => by
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
      exact HasType.letPairE _ _ _ _ _ _ _ _ _ _ he' hdist hfresh1 hfresh2 hbody hcons1 hcons2
    · by_cases hxn2 : n2 = nm
      · -- n2 = nm: body NOT substituted (shadowing by n2)
        have hor : (n1 == nm || n2 == nm) = true := by simp [beq_iff_eq, hxn2]
        simp only [hor]
        have hfresh_nm2 : ctx_mid.lookup nm = none := hxn2 ▸ hfresh2
        have hcons_nm2 : ctx_body.lookup nm = none := hxn2 ▸ hcons2
        have he' := subst_typing hv ht_v he hname
        rw [remove_of_lookup_none hfresh_nm2] at he'
        rw [remove_of_lookup_none hcons_nm2]
        exact HasType.letPairE _ _ _ _ _ _ _ _ _ _ he' hdist hfresh1 hfresh2 hbody hcons1 hcons2
      · -- Neither n1 nor n2 = nm: both e and body substituted
        have hor : (n1 == nm || n2 == nm) = false := by
          simp [beq_iff_eq, hxn1, hxn2]
        simp only [hor]
        have he' := subst_typing hv ht_v he hname
        have hfresh1' : (ctx_mid.remove nm).lookup n1 = none := by
          rw [remove_lookup_ne (Ne.symm hxn1)]; exact hfresh1
        have hfresh2' : (ctx_mid.remove nm).lookup n2 = none := by
          rw [remove_lookup_ne (Ne.symm hxn2)]; exact hfresh2
        have hname_body : ∀ t', Ctx.lookup ((n2, t2) :: (n1, t1) :: ctx_mid) nm = some t' → t' = t_v := by
          intro t' hl
          rw [lookup_cons_ne hxn2, lookup_cons_ne hxn1] at hl
          exact hname t' (output_binding_from_input he hl)
        have hbody' := subst_typing hv ht_v hbody hname_body
        rw [remove_cons_ne (Ne.symm hxn2), remove_cons_ne (Ne.symm hxn1)] at hbody'
        have hcons1' : (ctx_body.remove nm).lookup n1 = none := by
          rw [remove_lookup_ne (Ne.symm hxn1)]; exact hcons1
        have hcons2' : (ctx_body.remove nm).lookup n2 = none := by
          rw [remove_lookup_ne (Ne.symm hxn2)]; exact hcons2
        exact HasType.letPairE _ _ _ _ _ _ _ _ _ _ he' hdist hfresh1' hfresh2' hbody' hcons1' hcons2'

-- ============================================================================
-- Substitution preserves typing (wrapper for preservation)
-- ============================================================================

/-- Substitution lemma as needed by preservation's letVal case. -/
theorem subst_preserves_typing
    {ctx ctx' ctx'' : Ctx} {name : String} {v : Expr} {t_v : Ty}
    {e : Expr} {t : Ty}
    (hval : HasType ctx v t_v ctx')
    (hfresh : ctx'.lookup name = none)
    (hbody : HasType ((name, t_v) :: ctx') e t ctx'')
    (hconsumed : ctx''.lookup name = none)
    (hv : isValue v = true) :
    HasType ctx (subst e name v) t ctx'' := by
  have hctx_eq := value_preserves_ctx hval hv
  subst hctx_eq  -- ctx' replaced by ctx; hval : HasType ctx v t_v ctx
  have ht_v := value_any_ctx hv hval
  have hname_top : ∀ t', Ctx.lookup ((name, t_v) :: ctx) name = some t' → t' = t_v := by
    intro t' h; simp [lookup_cons_eq] at h; exact h.symm
  have h := subst_typing hv ht_v hbody hname_top
  rw [remove_cons_eq] at h
  rw [remove_of_lookup_none hfresh] at h
  rw [remove_of_lookup_none hconsumed] at h
  exact h

/-- Preservation: if Γ ⊢ e : t ⊣ Γ' and e ⟶ e', then Γ ⊢ e' : t ⊣ Γ'. -/
theorem preservation {e e' : Expr} {t : Ty} {ctx ctx' : Ctx}
    (ht : HasType ctx e t ctx') (hs : Step e e') :
    HasType ctx e' t ctx' := by
  induction hs generalizing t ctx ctx' with
  | divergeVal s pred =>
    cases ht with
    | diverge _ _ _ _ _ hw =>
      cases hw with
      | warpVal _ _ =>
        exact HasType.pairVal _ _ _ _ _ _ _
          (HasType.warpVal _ _) (HasType.warpVal _ _)
  | mergeVal s1 s2 =>
    cases ht with
    | merge _ _ _ _ _ _ _ _ hw1 hw2 hcomp =>
      cases hw1 with
      | warpVal _ _ =>
        cases hw2 with
        | warpVal _ _ =>
          have ⟨_, hcov⟩ := hcomp
          unfold ActiveSet.Covers at hcov
          rw [hcov]
          exact HasType.warpVal _ _
  | shuffleVal s =>
    cases ht with
    | shuffle _ _ _ _ _ hw hd =>
      cases hw with
      | warpVal _ _ =>
        cases hd with
        | perLaneVal _ => exact HasType.perLaneVal _
  | letVal name v body hv =>
    cases ht with
    | letBind _ _ _ _ _ _ _ _ hval hfresh hbody hconsumed =>
      exact subst_preserves_typing hval hfresh hbody hconsumed hv
  | fstVal a b hva hvb =>
    cases ht with
    | fstE _ _ _ t1 t2 he =>
      cases he with
      | pairVal _ ctx_mid _ _ _ _ _ ha hb =>
        have := value_preserves_ctx hb hvb
        subst this; exact ha
  | sndVal a b hva hvb =>
    cases ht with
    | sndE _ _ _ t1 t2 he =>
      cases he with
      | pairVal _ ctx_mid _ _ _ _ _ ha hb =>
        have := value_preserves_ctx ha hva
        subst this; exact hb
  | divergeCong w w' pred _ ih =>
    cases ht with
    | diverge _ _ _ s _ hw =>
      exact HasType.diverge _ _ _ _ _ (ih hw)
  | mergeLeft w1 w1' w2 _ ih =>
    cases ht with
    | merge _ _ _ _ _ _ _ _ hw1 hw2 hcomp =>
      exact HasType.merge _ _ _ _ _ _ _ _ (ih hw1) hw2 hcomp
  | mergeRight v1 w2 w2' _ _ ih =>
    cases ht with
    | merge _ _ _ _ _ _ _ _ hw1 hw2 hcomp =>
      exact HasType.merge _ _ _ _ _ _ _ _ hw1 (ih hw2) hcomp
  | shuffleLeft w w' data _ ih =>
    cases ht with
    | shuffle _ _ _ _ _ hw hd =>
      exact HasType.shuffle _ _ _ _ _ (ih hw) hd
  | shuffleRight v data data' _ _ ih =>
    cases ht with
    | shuffle _ _ _ _ _ hw hd =>
      exact HasType.shuffle _ _ _ _ _ hw (ih hd)
  | letCong name val val' body _ ih =>
    cases ht with
    | letBind _ _ _ _ _ _ _ _ hval hfresh hbody hconsumed =>
      exact HasType.letBind _ _ _ _ _ _ _ _ (ih hval) hfresh hbody hconsumed
  | pairLeftCong a a' b _ ih =>
    cases ht with
    | pairVal _ _ _ _ _ _ _ ha hb =>
      exact HasType.pairVal _ _ _ _ _ _ _ (ih ha) hb
  | pairRightCong a b b' _ _ ih =>
    cases ht with
    | pairVal _ _ _ _ _ _ _ ha hb =>
      exact HasType.pairVal _ _ _ _ _ _ _ ha (ih hb)
  | fstCong e e' _ ih =>
    cases ht with
    | fstE _ _ _ _ _ he =>
      exact HasType.fstE _ _ _ _ _ (ih he)
  | sndCong e e' _ ih =>
    cases ht with
    | sndE _ _ _ _ _ he =>
      exact HasType.sndE _ _ _ _ _ (ih he)
  | letPairVal name1 name2 v1 v2 body hv1 hv2 =>
    cases ht with
    | letPairE _ _ _ _ _ _ _ t1 t2 _ he hdist hfresh1 hfresh2 hbody hcons1 hcons2 =>
      cases he with
      | pairVal _ ctx_a _ _ _ _ _ ha hb =>
        -- ha : HasType ctx v1 t1 ctx_a
        -- hb : HasType ctx_a v2 t2 ctx_mid
        -- Both values, so ctx = ctx_a = ctx_mid
        have hctx_a := value_preserves_ctx ha hv1; subst hctx_a
        have hctx_mid := value_preserves_ctx hb hv2; subst hctx_mid
        -- Step 1: substitute name1 → v1 via subst_typing
        have ht_v1 := value_any_ctx hv1 ha
        have hname_top : ∀ t', Ctx.lookup ((name2, t2) :: (name1, t1) :: ctx) name1 = some t' → t' = t1 := by
          intro t' h; rw [lookup_cons_ne (Ne.symm hdist)] at h; simp [lookup_cons_eq] at h; exact h.symm
        have h1 := subst_typing hv1 ht_v1 hbody hname_top
        rw [remove_cons_ne hdist, remove_cons_eq,
            remove_of_lookup_none hfresh1] at h1
        rw [remove_of_lookup_none hcons1] at h1
        -- Step 2: substitute name2 → v2 via subst_preserves_typing
        exact subst_preserves_typing hb hfresh2 h1 hcons2 hv2
  | letPairCong e e' name1 name2 body _ ih =>
    cases ht with
    | letPairE _ _ _ _ _ _ _ _ _ _ he hdist hfresh1 hfresh2 hbody hcons1 hcons2 =>
      exact HasType.letPairE _ _ _ _ _ _ _ _ _ _ (ih he) hdist hfresh1 hfresh2 hbody hcons1 hcons2

-- ============================================================================
-- Multi-step Type Safety (Corollary 4.3)
-- ============================================================================

/-- Multi-step type safety: a closed well-typed term that reduces in zero or more
    steps never reaches a stuck non-value state. That is, every reachable
    expression is either a value or can take another step.
    Corollary 4.3 from the paper — follows by induction on `Star Step` from
    progress + preservation. -/
theorem type_safety {e e' : Expr} {t : Ty} {ctx' : Ctx}
    (ht : HasType [] e t ctx') (hstar : Star Step e e') :
    (isValue e' = true) ∨ (∃ e'', Step e' e'') := by
  induction hstar with
  | refl => exact progress ht
  | step h1 _ ih => exact ih (preservation ht h1)

-- ============================================================================
-- Untypability: real GPU bugs cannot be typed
--
-- Each theorem proves that a specific bug pattern (shuffle on a diverged
-- warp) has NO typing derivation. All five documented bugs share the same
-- root cause: attempting to use shuffle (which requires Warp<All>) on a
-- sub-warp obtained via diverge.
-- ============================================================================

/-- The type of fst(diverge(warpVal(all), pred)) is always Warp<all &&& pred>. -/
private theorem fst_diverge_warpval_type {pred : ActiveSet} {t : Ty} {ctx' : Ctx}
    (ht : HasType [] (.fst (.diverge (.warpVal ActiveSet.all) pred)) t ctx') :
    t = .warp (ActiveSet.all &&& pred) := by
  match ht with
  | .fstE _ _ _ _ _ he =>
    match he with
    | .diverge _ _ _ _ _ hwv =>
      match hwv with
      | .warpVal _ _ => rfl

/-- Shuffle on a diverged warp is untypable when the predicate ≠ all. -/
private theorem shuffle_diverged_untypable
    (pred : ActiveSet)
    (hne : ActiveSet.all &&& pred ≠ ActiveSet.all) :
    ¬ ∃ t ctx', HasType []
      (.shuffle (.fst (.diverge (.warpVal ActiveSet.all) pred)) .perLaneVal)
      t ctx' := by
  intro ⟨t, ctx', ht⟩
  -- shuffle requires warp arg to have type Warp<All>
  have ⟨ctx_mid, hw⟩ := shuffle_requires_all ht
  -- But fst(diverge(warpVal(all), pred)) has type Warp<all &&& pred>
  have heq := fst_diverge_warpval_type hw
  -- heq : Ty.warp ActiveSet.all = Ty.warp (ActiveSet.all &&& pred)
  -- Extract: ActiveSet.all = ActiveSet.all &&& pred
  simp only [Ty.warp.injEq] at heq
  exact absurd heq.symm hne

/-- Bug 1 (cuda-samples #398): Shuffle after extracting lane 0.
    ActiveSet.all &&& 0x1 = 0x1 ≠ ActiveSet.all -/
theorem bug1_cuda_samples_398 :
    ¬ ∃ t ctx', HasType []
      (.shuffle (.fst (.diverge (.warpVal ActiveSet.all) (0x00000001#32))) .perLaneVal)
      t ctx' :=
  shuffle_diverged_untypable (0x00000001#32) (by decide)

/-- Bug 2 (CUB/CCCL #854): Shuffle on 16-lane sub-warp.
    ActiveSet.all &&& 0xFFFF = 0xFFFF ≠ ActiveSet.all -/
theorem bug2_cccl_854 :
    ¬ ∃ t ctx', HasType []
      (.shuffle (.fst (.diverge (.warpVal ActiveSet.all) (0x0000FFFF#32))) .perLaneVal)
      t ctx' :=
  shuffle_diverged_untypable (0x0000FFFF#32) (by decide)

/-- Bug 3 (PIConGPU #2514): Ballot (= shuffle) on diverged subset.
    ActiveSet.all &&& 0xFFFF = 0xFFFF ≠ ActiveSet.all -/
theorem bug3_picongpu_2514 :
    ¬ ∃ t ctx', HasType []
      (.shuffle (.fst (.diverge (.warpVal ActiveSet.all) (0x0000FFFF#32))) .perLaneVal)
      t ctx' :=
  shuffle_diverged_untypable (0x0000FFFF#32) (by decide)

/-- Bug 4 (LLVM #155682): Shuffle after lane-0 conditional.
    ActiveSet.all &&& 0x1 = 0x1 ≠ ActiveSet.all -/
theorem bug4_llvm_155682 :
    ¬ ∃ t ctx', HasType []
      (.shuffle (.fst (.diverge (.warpVal ActiveSet.all) (0x00000001#32))) .perLaneVal)
      t ctx' :=
  shuffle_diverged_untypable (0x00000001#32) (by decide)

/-- Bug 5 (demo): Shuffle after even/odd divergence.
    ActiveSet.all &&& ActiveSet.even = ActiveSet.even ≠ ActiveSet.all -/
theorem bug5_shuffle_after_diverge :
    ¬ ∃ t ctx', HasType []
      (.shuffle (.fst (.diverge (.warpVal ActiveSet.all) ActiveSet.even)) .perLaneVal)
      t ctx' :=
  shuffle_diverged_untypable ActiveSet.even (by decide)
