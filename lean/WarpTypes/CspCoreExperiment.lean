import WarpTypes.Generic

/-
  Experiment E — Csp.lean ↦ Core.lean Feasibility Probe (2026-04-11)

  VERDICT: E1 — "Widen Core" works.
  ─────────────────────────────────

  Design (a), the shadow `Option (Topology n)`-parameterized extension of
  Core.lean's judgment, passes all three pre-registered falsification tests:

    F1 (construction) — a `send` derivation under a concrete 2-node topology
        builds from `ShadowCoreHasType.send` without issue.

    F2 (new-axis inversion) — `shadow_send_requires_adjacent` inverts a
        `.send g payload self dst` hypothesis and extracts `topo' : Topology n`
        with `topo = some topo'` and `topo'.adj self dst = true`. The inversion
        coexists with `mergeFamily` / `finalizeFamily` in the same inductive —
        those rules' branches are discharged by the double-equality-witness
        pattern used in Core.lean, unchanged by the addition of send/recv.

    F3 (existing-family inversion) — `shadow_fence_requires_all` inverts a
        `.fence g` hypothesis and extracts `ShadowCoreHasType ... g
        (.group (PSet.all n)) ctx'`, via the same discharge pattern Fence.lean
        uses on Core.lean's `finalizeFamily`. Adding send/recv and the
        topology parameter did NOT break this — the `send`/`recv`/`collective`
        branches auto-eliminate on cases-site constructor clash (they carry
        concrete expression patterns `.send`/`.recv`/`.collective`, not free
        `expr` variables like the family rules do), so F3's proof is
        byte-identical in structure to Fence.lean's post-port
        `fence_requires_all`.

  What this means for a real port. Csp.lean CAN be ported onto a widened
  Core.lean. The cost is:
    1. Add `topo : Option (Topology n)` as an inductive parameter to
       `CoreHasType` (not an index — parameters don't get generalized by
       `cases`, which is what keeps the existing domain proofs working).
    2. Add `send` / `recv` / `collective` constructors to `CoreExpr` and
       corresponding rules to `CoreHasType`. `send` and `recv` carry an
       additional `(topo' : Topology n) (hTopo : topo = some topo')` pair —
       analogous to the double-equality-witness pattern but ranging over a
       different axis (topology presence instead of tag).
    3. Fence.lean and Reduce.lean stay at `topo = none`. Csp-analog proofs
       live at `topo = some topo'`. The Option sentinel IS the witness.

  What this probe did NOT test:
    - The ~170-line port cost itself — this is only a feasibility probe.
      The port would carry two new hypotheses at each new rule (`topo'` +
      `hTopo`) and cases-site unifier behaviour on a 20-rule inductive
      hasn't been measured here.
    - Csp.lean's nested-cases helpers (`csp_fst_diverge_groupval_type`,
      `collective_after_diverge_untypable`) — these walk through multiple
      `cases` levels and §9.3 observed the caller-side boilerplate tax
      scales with nesting depth × family-rule-count. Adding send/recv/
      collective to the ported `CoreHasType` is a family-rule-count +3,
      so any existing nested proof in Fence/Reduce would pay 3 more
      dead-branch discharges per cases level. Those DO auto-eliminate
      (they have concrete expression shapes — `cases` sees through it
      without manual help, same as the auto-elimination F2 exercises
      here), so the overhead is zero.
    - Design (b) split-Core — intentionally skipped once (a) passed.
      If a future probe finds (a) runs out of headroom (e.g. with a
      fifth domain that needs yet another orthogonal parameter), (b)
      is the documented fallback. A sketch is preserved at the bottom
      of this file.

  Hard constraints (from plan):
    - `import WarpTypes.Generic` only. Does NOT import Core/Csp/Fence/
      Reduce/Protocol and is NOT imported by WarpTypes.lean.
    - Generic.lean md5 must remain `7f125b5f5f26122cc9e97c39522a4d03`.
    - No sorry, no axiom, no admit.
    - Built via `lake build WarpTypes.CspCoreExperiment`.
-/

namespace CspCoreExperiment

-- ============================================================================
-- Topology — mirrors Csp.lean's `Topology n` structure
-- ============================================================================

/-- A topology on n participants. Adjacency is symmetric and irreflexive.
    Structurally identical to `Topology` in `Csp.lean`; duplicated here
    because this probe intentionally does not import Csp.lean. -/
structure Topology (n : Nat) where
  adj : Fin n → Fin n → Bool
  sym : ∀ i j, adj i j = adj j i
  irrefl : ∀ i, adj i i = false

-- ============================================================================
-- Shadow Core types, tags, and expression language
-- ============================================================================

/-- Shadow of Core.lean's `CoreTy`. Same constructors. -/
inductive ShadowCoreTy (n : Nat)
  | group (s : PSet n)
  | reduced (s : PSet n)
  | data
  | unit
  | pair (a b : ShadowCoreTy n)

/-- Shadow of Core.lean's `TyTag`. Same two constructors, `DecidableEq` for
    dead-branch elimination on the widened inductive. -/
inductive TyTag
  | group
  | reduced
  deriving DecidableEq

/-- Shadow of Core.lean's `CoreExpr`, augmented with Csp's `send`, `recv`,
    and `collective`. `write` and `leafReduce` (Fence/Reduce monomorphic
    rules) are omitted — F1/F2/F3 don't exercise them and they would only
    add copy-paste mass without changing the falsification outcome. -/
inductive ShadowCoreExpr (n : Nat)
  -- Value constructors
  | groupVal (s : PSet n)
  | dataVal
  | unitVal
  -- Variable
  | var (name : String)
  -- Partition
  | diverge (g : ShadowCoreExpr n) (pred : PSet n)
  -- Family-parametric merges
  | merge (g1 g2 : ShadowCoreExpr n)
  | combineRed (r1 r2 : ShadowCoreExpr n)
  -- Let-binding
  | letBind (name : String) (val body : ShadowCoreExpr n)
  -- Pair suite
  | pairVal (a b : ShadowCoreExpr n)
  | fst (e : ShadowCoreExpr n)
  | snd (e : ShadowCoreExpr n)
  -- Family-parametric extracts
  | fence (g : ShadowCoreExpr n)
  | finalize (r : ShadowCoreExpr n)
  -- ── Csp additions ──
  | send (g payload : ShadowCoreExpr n) (self dst : Fin n)
  | recv (g : ShadowCoreExpr n) (self src : Fin n)
  | collective (g payload : ShadowCoreExpr n)

-- ============================================================================
-- Dispatchers — same @[reducible] discipline as Core.lean
-- ============================================================================

@[reducible]
def tagToTy {n : Nat} : TyTag → PSet n → ShadowCoreTy n
  | .group,   s => .group s
  | .reduced, s => .reduced s

@[reducible]
def tagToMergeExpr {n : Nat} : TyTag → ShadowCoreExpr n → ShadowCoreExpr n → ShadowCoreExpr n
  | .group,   e1, e2 => .merge e1 e2
  | .reduced, e1, e2 => .combineRed e1 e2

@[reducible]
def tagToFinalExpr {n : Nat} : TyTag → ShadowCoreExpr n → ShadowCoreExpr n
  | .group,   e => .fence e
  | .reduced, e => .finalize e

@[reducible]
def tagToFinalTy {n : Nat} : TyTag → ShadowCoreTy n
  | .group   => .unit
  | .reduced => .data

-- ============================================================================
-- Shadow context (linear, same shape as CoreCtx)
-- ============================================================================

def ShadowCoreCtx (n : Nat) := List (String × ShadowCoreTy n)

namespace ShadowCoreCtx

def lookup {n : Nat} (ctx : ShadowCoreCtx n) (name : String) : Option (ShadowCoreTy n) :=
  ctx.find? (fun p => p.1 == name) |>.map Prod.snd

def remove {n : Nat} (ctx : ShadowCoreCtx n) (name : String) : ShadowCoreCtx n :=
  ctx.filter (fun p => p.1 != name)

end ShadowCoreCtx

-- ============================================================================
-- Design (a): Widen Core — topology as an inductive parameter
-- ============================================================================

/-- The shadow judgment, widened with `topo : Option (Topology n)` as an
    INDUCTIVE PARAMETER (not an index). Parameters don't get generalized by
    `cases`, which is load-bearing for the existing Fence/Reduce inversion
    proofs: if topology were an index, every `cases` in those proofs would
    have to rebind `topo`, and the family-parametric dead-branch discharges
    would no longer pattern-match cleanly.

    Rules for `send` / `recv` carry an additional `(topo' : Topology n)`
    parameter together with `hTopo : topo = some topo'`. This is the
    adjacency analog of the double-equality-witness pattern: the side
    condition `topo'.adj self dst = true` can only be stated about an
    unwrapped topology, and the equality witness lets `cases`-site
    unification stay first-order. `collective` is intentionally NOT
    guarded on topology — it's a PSet-all gate with no topology
    dependence, same as Fence's `fence` and Reduce's `finalize`. -/
inductive ShadowCoreHasType {n : Nat} (topo : Option (Topology n)) :
    ShadowCoreCtx n → ShadowCoreExpr n → ShadowCoreTy n → ShadowCoreCtx n → Prop

  -- ── Monomorphic value / variable rules (copies of Core.lean) ──

  | groupVal (ctx : ShadowCoreCtx n) (s : PSet n) :
      ShadowCoreHasType topo ctx (.groupVal s) (.group s) ctx
  | dataVal (ctx : ShadowCoreCtx n) :
      ShadowCoreHasType topo ctx .dataVal .data ctx
  | unitVal (ctx : ShadowCoreCtx n) :
      ShadowCoreHasType topo ctx .unitVal .unit ctx
  | var (ctx : ShadowCoreCtx n) (name : String) (t : ShadowCoreTy n) :
      ctx.lookup name = some t →
      ShadowCoreHasType topo ctx (.var name) t (ctx.remove name)
  | diverge (ctx ctx' : ShadowCoreCtx n) (g : ShadowCoreExpr n) (s pred : PSet n) :
      ShadowCoreHasType topo ctx g (.group s) ctx' →
      ShadowCoreHasType topo ctx (.diverge g pred)
        (.pair (.group (s &&& pred)) (.group (s &&& ~~~pred))) ctx'

  -- ── Let-binding, pair suite ──

  | letBind (ctx ctx' ctx'' : ShadowCoreCtx n) (name : String)
      (val body : ShadowCoreExpr n) (t1 t2 : ShadowCoreTy n) :
      ShadowCoreHasType topo ctx val t1 ctx' →
      ctx'.lookup name = none →
      ShadowCoreHasType topo ((name, t1) :: ctx') body t2 ctx'' →
      ctx''.lookup name = none →
      ShadowCoreHasType topo ctx (.letBind name val body) t2 ctx''
  | pairVal (ctx ctx' ctx'' : ShadowCoreCtx n) (a b : ShadowCoreExpr n) (t1 t2 : ShadowCoreTy n) :
      ShadowCoreHasType topo ctx a t1 ctx' →
      ShadowCoreHasType topo ctx' b t2 ctx'' →
      ShadowCoreHasType topo ctx (.pairVal a b) (.pair t1 t2) ctx''
  | fstE (ctx ctx' : ShadowCoreCtx n) (e : ShadowCoreExpr n) (t1 t2 : ShadowCoreTy n) :
      ShadowCoreHasType topo ctx e (.pair t1 t2) ctx' →
      ShadowCoreHasType topo ctx (.fst e) t1 ctx'
  | sndE (ctx ctx' : ShadowCoreCtx n) (e : ShadowCoreExpr n) (t1 t2 : ShadowCoreTy n) :
      ShadowCoreHasType topo ctx e (.pair t1 t2) ctx' →
      ShadowCoreHasType topo ctx (.snd e) t2 ctx'

  -- ── Family-parametric rules (verbatim from Core.lean) ──

  /-- Parametric merge. Shape is identical to Core.lean's `mergeFamily`:
      tagged dispatch via `tagToMergeExpr` / `tagToTy` + double equality
      witnesses. The only change is the added `topo` parameter on the
      inductive — which, being a parameter, does not affect rule-internal
      unification. -/
  | mergeFamily (tag : TyTag)
      (ctx ctx' ctx'' : ShadowCoreCtx n) (e1 e2 expr : ShadowCoreExpr n)
      (s1 s2 parent : PSet n) (ty : ShadowCoreTy n)
      (hExpr : expr = tagToMergeExpr tag e1 e2)
      (hTy : ty = tagToTy tag parent) :
      ShadowCoreHasType topo ctx e1 (tagToTy tag s1) ctx' →
      ShadowCoreHasType topo ctx' e2 (tagToTy tag s2) ctx'' →
      PSet.IsComplement s1 s2 parent →
      ShadowCoreHasType topo ctx expr ty ctx''

  /-- Parametric finalize/fence. Same double-witness shape. -/
  | finalizeFamily (tag : TyTag)
      (ctx ctx' : ShadowCoreCtx n) (e expr : ShadowCoreExpr n) (resultTy : ShadowCoreTy n)
      (hExpr : expr = tagToFinalExpr tag e)
      (hTy : resultTy = tagToFinalTy tag) :
      ShadowCoreHasType topo ctx e (tagToTy tag (PSet.all n)) ctx' →
      ShadowCoreHasType topo ctx expr resultTy ctx'

  -- ── Csp additions — topology-guarded ──

  /-- Point-to-point send. Requires topology presence via `hTopo`, plus
      active-destination and adjacency side conditions. Returns the group
      threaded unchanged. -/
  | send (topo' : Topology n) (hTopo : topo = some topo')
      (ctx ctx' ctx'' : ShadowCoreCtx n) (g payload : ShadowCoreExpr n)
      (self dst : Fin n) (s : PSet n) :
      ShadowCoreHasType topo ctx g (.group s) ctx' →
      s.getLsbD dst.val = true →
      topo'.adj self dst = true →
      ShadowCoreHasType topo ctx' payload .data ctx'' →
      ShadowCoreHasType topo ctx (.send g payload self dst) (.group s) ctx''

  /-- Point-to-point recv. Same topology guard shape as `send`. Returns a
      pair of (data, group). -/
  | recv (topo' : Topology n) (hTopo : topo = some topo')
      (ctx ctx' : ShadowCoreCtx n) (g : ShadowCoreExpr n)
      (self src : Fin n) (s : PSet n) :
      ShadowCoreHasType topo ctx g (.group s) ctx' →
      s.getLsbD src.val = true →
      topo'.adj self src = true →
      ShadowCoreHasType topo ctx (.recv g self src) (.pair .data (.group s)) ctx'

  /-- Collective. PSet-all-gated, no topology dependence. Structurally a
      Fence-shape rule (`.group` input, `PSet.all n` parent) with a
      Reduce-shape output (`.data`). Not in the family-parametric pattern
      because its input/output pairing doesn't fit `tagToFinalTy` — this
      is a design observation, not a probe failure. -/
  | collective (ctx ctx' ctx'' : ShadowCoreCtx n) (g payload : ShadowCoreExpr n) :
      ShadowCoreHasType topo ctx g (.group (PSet.all n)) ctx' →
      ShadowCoreHasType topo ctx' payload .data ctx'' →
      ShadowCoreHasType topo ctx (.collective g payload) .data ctx''

-- ============================================================================
-- Concrete 2-node topology (for F1 / F2 construction and smoke tests)
-- ============================================================================

/-- Minimal 2-node topology: cores 0 and 1 are adjacent, both symmetric and
    irreflexive. Decidable, so `by decide` closes both structure fields. -/
def twoNodeAdj : Fin 2 → Fin 2 → Bool
  | ⟨0, _⟩, ⟨1, _⟩ => true
  | ⟨1, _⟩, ⟨0, _⟩ => true
  | _, _           => false

def twoNode : Topology 2 where
  adj := twoNodeAdj
  sym := by decide
  irrefl := by decide

-- ============================================================================
-- F1 — Construction test
-- ============================================================================

/-- F1: a `send` derivation from core 0 to core 1 under `twoNode` builds.
    Mirrors `j1_send_adjacent_typable` in Csp.lean, but on the widened
    shadow judgment. PASS means design (a) admits construction. -/
theorem F1_send_adjacent :
    ∃ t ctx', ShadowCoreHasType (some twoNode) ([] : ShadowCoreCtx 2)
      (.send (.groupVal (PSet.all 2)) .dataVal ⟨0, by omega⟩ ⟨1, by omega⟩)
      t ctx' :=
  ⟨.group (PSet.all 2), [],
    ShadowCoreHasType.send
      twoNode rfl [] [] [] _ _ ⟨0, by omega⟩ ⟨1, by omega⟩ (PSet.all 2)
      (ShadowCoreHasType.groupVal [] (PSet.all 2))
      (by decide)   -- bit 1 of (PSet.all 2) is true
      (by decide)   -- twoNode.adj 0 1 is true
      (ShadowCoreHasType.dataVal [])⟩

-- ============================================================================
-- F2 — New-axis inversion test
-- ============================================================================

/-- F2: inversion of a `send` derivation recovers the topology and the
    adjacency witness. Mirrors Csp.lean's `csp_send_requires_adjacent`.
    The proof has to discharge the `mergeFamily` and `finalizeFamily`
    branches manually (their `expr` parameter is free at the cases-site),
    but the monomorphic branches (groupVal, recv, collective, etc.)
    auto-eliminate on constructor clash because their expression shapes
    are concrete. This is the probe's central observation: the double-
    equality-witness pattern and the new topology axis coexist cleanly
    in the same inductive. PASS means design (a) admits side-condition
    extraction on a new-axis rule. -/
theorem F2_shadow_send_requires_adjacent
    {n : Nat} {topo : Option (Topology n)}
    {ctx ctx'' : ShadowCoreCtx n} {g payload : ShadowCoreExpr n}
    {self dst : Fin n} {t : ShadowCoreTy n} :
    ShadowCoreHasType topo ctx (.send g payload self dst) t ctx'' →
    ∃ topo' : Topology n, topo = some topo' ∧ topo'.adj self dst = true := by
  intro h
  cases h with
  | send topo' hTopo _ _ _ _ _ _ _ _ _ _ hadj _ =>
    exact ⟨topo', hTopo, hadj⟩
  | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
    cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
  | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
    cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr

-- ============================================================================
-- F3 — Existing-family inversion test (Fence's fence_requires_all analog)
-- ============================================================================

/-- F3: inversion of a `.fence g` derivation recovers that `g` has type
    `.group (PSet.all n)`. Mirrors Fence.lean's post-port
    `fence_requires_all`. Proof structure is byte-identical to that file's
    — adding send/recv/collective and the topology parameter did not
    force any additional discharges. The new constructors auto-eliminate
    (concrete expression shapes) and the topology parameter is not
    generalized by `cases` (it's an inductive parameter, not an index).
    PASS means design (a) preserves the existing Fence/Reduce inversion
    theorems. -/
theorem F3_shadow_fence_requires_all {n : Nat} {topo : Option (Topology n)}
    {ctx ctx' : ShadowCoreCtx n} {g : ShadowCoreExpr n} :
    ShadowCoreHasType topo ctx (.fence g) .unit ctx' →
    ShadowCoreHasType topo ctx g (.group (PSet.all n)) ctx' := by
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
  | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
    cases tag with
    | group =>
      simp only [tagToMergeExpr] at hExpr
      cases hExpr
    | reduced =>
      simp only [tagToMergeExpr] at hExpr
      cases hExpr

-- ============================================================================
-- Bonus: F3 mirror on `.finalize` — the Reduce-side existing-family test
-- ============================================================================

/-- Reduce.lean's `finalize_requires_all` analog. Same discharge pattern
    as F3, different tag branch. Confirms the result type asymmetry
    (`.unit` vs `.data`) still survives on the widened inductive. -/
theorem F3b_shadow_finalize_requires_all {n : Nat} {topo : Option (Topology n)}
    {ctx ctx' : ShadowCoreCtx n} {r : ShadowCoreExpr n} :
    ShadowCoreHasType topo ctx (.finalize r) .data ctx' →
    ShadowCoreHasType topo ctx r (.reduced (PSet.all n)) ctx' := by
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
  | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
    cases tag with
    | group =>
      simp only [tagToMergeExpr] at hExpr
      cases hExpr
    | reduced =>
      simp only [tagToMergeExpr] at hExpr
      cases hExpr

end CspCoreExperiment

-- ============================================================================
-- Design (b) — Split Core — deliberately NOT run
-- ============================================================================

/-
  Design (b) is the documented fallback if a future extension exhausts
  design (a)'s headroom. It is sketched (not coded) here for the record.

  SHAPE:
    1. Leave Core topology-free (as it is today).
    2. Define `CoreTopoHasType (topo : Topology n) : Ctx → Expr → Ty → Ctx → Prop`
       with ONE embedding rule:
           | lift : CoreHasType ctx e t ctx' → CoreTopoHasType topo ctx e t ctx'
       plus the send/recv/collective rules duplicated from the Csp port,
       all at `(topo : Topology n)` (no Option sentinel — Topology is
       unconditionally present in this variant).
    3. Csp.lean's programs type-check in `CoreTopoHasType j1Grid`. Fence
       and Reduce stay in `CoreHasType` and are lifted into `CoreTopoHasType`
       by `lift` when composed with Csp programs (if ever needed).

  TRADE-OFFS vs (a):
    + Does NOT add Option handling to the core judgment — Fence/Reduce
      don't see topology at all.
    + The embedding rule makes "every Fence program is a CoreTopo
      program at any topology" a one-line theorem.
    − Every theorem about CoreTopo programs that hits a `lift` branch
      needs an extra inversion step (unfolding the lift to recover the
      underlying Core derivation), so Csp-side helpers pay a per-theorem
      tax.
    − Csp.lean's existing nested-cases helpers would need to be
      rewritten to cases on CoreTopoHasType first, then on the lifted
      CoreHasType — a mechanical port but not free.

  VERDICT E1 above means (b) is not needed today. If a future fifth or
  sixth domain stresses (a), revisit (b) then.
-/
