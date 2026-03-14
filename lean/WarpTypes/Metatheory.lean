import WarpTypes.Basic

/-
  Metatheory: Progress and Preservation for Session-Typed Divergence

  - Capture-avoiding substitution
  - Small-step reduction (Step)
  - Canonical forms lemmas
  - Progress theorem (zero sorry)
  - Preservation theorem (axiom: substitution lemma for linear contexts)
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

-- ============================================================================
-- Small-Step Reduction
-- ============================================================================

inductive Step : Expr → Expr → Prop
  | divergeVal (s pred : ActiveSet) :
      Step (.diverge (.warpVal s) pred)
           (.pairVal (.warpVal (s &&& pred)) (.warpVal (s &&& ~~~pred)))
  | mergeVal (s1 s2 : ActiveSet) :
      Step (.merge (.warpVal s1) (.warpVal s2)) (.warpVal ActiveSet.all)
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
  | .merge _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .shuffle _ _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv

-- ============================================================================
-- Canonical Forms
-- ============================================================================

theorem canonical_warp {e : Expr} {s : ActiveSet} {ctx' : Ctx}
    (ht : HasType [] e (.warp s) ctx') (hv : isValue e = true) :
    e = .warpVal s := by
  match ht with
  | .warpVal _ _ => rfl
  | .var _ _ _ hlook => simp [Ctx.lookup, List.find?] at hlook
  | .merge _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv

theorem canonical_perLane {e : Expr} {ctx' : Ctx}
    (ht : HasType [] e .perLane ctx') (hv : isValue e = true) :
    e = .perLaneVal := by
  match ht with
  | .perLaneVal _ => rfl
  | .var _ _ _ hlook => simp [Ctx.lookup, List.find?] at hlook
  | .shuffle _ _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv

theorem canonical_pair {e : Expr} {t1 t2 : Ty} {ctx' : Ctx}
    (ht : HasType [] e (.pair t1 t2) ctx') (hv : isValue e = true) :
    ∃ v1 v2, e = .pairVal v1 v2 ∧ isValue v1 = true ∧ isValue v2 = true := by
  match ht with
  | .pairVal _ _ _ a b _ _ _ _ =>
    simp [isValue] at hv; exact ⟨a, b, rfl, hv.1, hv.2⟩
  | .var _ _ _ hlook => simp [Ctx.lookup, List.find?] at hlook
  | .diverge _ _ _ _ _ _ => simp [isValue] at hv
  | .letBind _ _ _ _ _ _ _ _ _ _ => simp [isValue] at hv
  | .fstE _ _ _ _ _ _ => simp [isValue] at hv
  | .sndE _ _ _ _ _ _ => simp [isValue] at hv

-- ============================================================================
-- Progress (zero sorry)
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
  | .merge _ ctx_mid _ w1 w2 s1 s2 hw1 hw2 hcomp =>
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
  | .letBind _ _ _ name val body _ _ hval hbody =>
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

-- ============================================================================
-- Preservation
-- ============================================================================

/-- Substitution preserves typing in linear contexts.
    Standard substitution lemma — requires structural induction on typing
    with careful context splitting for the linear discipline. -/
axiom subst_preserves_typing :
    ∀ {ctx ctx' ctx'' : Ctx} {name : String} {v : Expr} {t_v : Ty}
      {e : Expr} {t : Ty},
    HasType ctx v t_v ctx' →
    HasType ((name, t_v) :: ctx') e t ctx'' →
    HasType ctx (subst e name v) t ctx''

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
    | merge _ _ _ _ _ _ _ hw1 hw2 _ =>
      cases hw1 with
      | warpVal _ _ =>
        cases hw2 with
        | warpVal _ _ => exact HasType.warpVal _ _
  | shuffleVal s =>
    cases ht with
    | shuffle _ _ _ _ _ hw hd =>
      cases hw with
      | warpVal _ _ =>
        cases hd with
        | perLaneVal _ => exact HasType.perLaneVal _
  | letVal name v body hv =>
    cases ht with
    | letBind _ _ _ _ _ _ _ _ hval hbody =>
      exact subst_preserves_typing hval hbody
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
    | merge _ _ _ _ _ _ _ hw1 hw2 hcomp =>
      exact HasType.merge _ _ _ _ _ _ _ (ih hw1) hw2 hcomp
  | mergeRight v1 w2 w2' _ _ ih =>
    cases ht with
    | merge _ _ _ _ _ _ _ hw1 hw2 hcomp =>
      exact HasType.merge _ _ _ _ _ _ _ hw1 (ih hw2) hcomp
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
    | letBind _ _ _ _ _ _ _ _ hval hbody =>
      exact HasType.letBind _ _ _ _ _ _ _ _ (ih hval) hbody
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
