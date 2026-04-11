import WarpTypes.Generic

/-
  Core.lean — Higher-order factoring of Fence.lean (Level 2c) and
  Reduce.lean (Level 2d)  (Experiment D, committed 2026-04-11)

  Purpose. Csp.lean / Fence.lean / Reduce.lean share ~170 lines of structural
  duplication in their core typing rules. The duplication is not stylistic: the
  `merge` rule in Fence and the `combineRed` rule in Reduce are byte-identical
  modulo `.group ↔ .reduced`, and the same holds for `fence_requires_all` vs
  `finalize_requires_all`. §9.2 of `research/complemented-typestate-framework.md`
  identifies this as evidence that the `merge` rule should be polymorphic over
  any `PSet n`-indexed type family, not hardcoded to `.group`.

  Rule shape. Experiment D probe 3b (see `CoreExperiment.lean`) validated the
  "tagged dispatch + explicit equality witness" pattern as the working way to
  encode such a family-parametric rule in Lean 4. Function-valued rule
  parameters block `cases`-site unification (probe 2); `@[reducible]`
  dispatchers alone don't fix it (probe 3a); the explicit equality witness in
  the rule parameters does (probe 3b). See INSIGHTS §N+40 for the elaboration
  detail.

  This file adds a **double** equality witness (one for the expression, one for
  the conclusion type) to both `mergeFamily` and `finalizeFamily`. Probe 3b
  used a single witness because its toy rule's conclusion type was the
  concrete `.finalTy`, not a stuck dispatch. In the real port, both the
  expression AND the output type go through tagged dispatch functions, so both
  need to be abstracted out of the conclusion or `cases` on a concrete-shape
  hypothesis (e.g. `CoreHasType ctx (.fence g) .unit ctx'`) hits the same
  stuck unification in the `mergeFamily` branch.

  Generic.lean is the only dependency. Its md5 must remain unchanged.
-/

-- ============================================================================
-- Types and tags
-- ============================================================================

/-- Core type family. Covers every type that appears in Fence or Reduce.

    `.group` / `.reduced` are both `PSet n`-indexed — they are the two type
    families the parametric rules dispatch over. Adding a future family
    means adding one constructor here plus one `TyTag` constructor plus one
    clause in each dispatcher. -/
inductive CoreTy (n : Nat)
  | group (s : PSet n)
  | reduced (s : PSet n)
  | data
  | unit
  | pair (a b : CoreTy n)

/-- Tag selecting a `PSet n`-indexed type-family branch. First-order enum
    (unlike a function-valued parameter it admits `cases`-based dispatch
    through `@[reducible]` unfolding). `deriving DecidableEq` is load-bearing
    for the dead-branch contradiction proofs. -/
inductive TyTag
  | group
  | reduced
  deriving DecidableEq

-- ============================================================================
-- Expression language (Level 1 core ⊔ Fence-specific ⊔ Reduce-specific)
-- ============================================================================

/-- Union of `FenceExpr` and `ReduceExpr` expression constructors.

    The choice to inline the full expression language into `Core.lean`
    (rather than have each domain define its own inductive and coerce into
    Core) is forced by Lean 4: inductives are closed, and the parametric
    typing rules need a single `CoreExpr` type on which to uniformly state
    `(expr : CoreExpr n)` as a pattern variable. See CoreExperiment probe 3b
    for the validation of this shape. -/
inductive CoreExpr (n : Nat)
  -- Value constructors
  | groupVal (s : PSet n)
  | dataVal
  | unitVal
  -- Variable
  | var (name : String)
  -- Partition (group-only: `diverge` always operates on a `.group s`)
  | diverge (g : CoreExpr n) (pred : PSet n)
  -- Family-parametric merges (merge: `.group`; combineRed: `.reduced`)
  | merge (g1 g2 : CoreExpr n)
  | combineRed (r1 r2 : CoreExpr n)
  -- Let-binding
  | letBind (name : String) (val body : CoreExpr n)
  -- Pair suite
  | pairVal (a b : CoreExpr n)
  | fst (e : CoreExpr n)
  | snd (e : CoreExpr n)
  | letPair (e : CoreExpr n) (name1 name2 : String) (body : CoreExpr n)
  -- Fence-specific: bulk write
  | write (g payload : CoreExpr n)
  -- Reduce-specific: leaf reduction from group to accumulator
  | leafReduce (g : CoreExpr n)
  -- Family-parametric extracts (fence: `.group` → `.unit`; finalize: `.reduced` → `.data`)
  | fence (g : CoreExpr n)
  | finalize (r : CoreExpr n)

-- ============================================================================
-- Dispatchers — @[reducible] so `simp only` can unfold them at use sites
-- ============================================================================

/-- Tag → type-family constructor. The core of the polymorphism: when a
    parametric rule says "the result is `tagToTy tag parent`", it refers to
    `.group parent` under tag `.group` and `.reduced parent` under tag
    `.reduced`. -/
@[reducible]
def tagToTy {n : Nat} : TyTag → PSet n → CoreTy n
  | .group,   s => .group s
  | .reduced, s => .reduced s

/-- Tag → binary merge expression constructor. `mergeFamily` maps to
    `.merge e1 e2` under tag `.group` and `.combineRed e1 e2` under tag
    `.reduced`. The two merge-shaped expression constructors are kept
    distinct in `CoreExpr` (not unified into one) so the mapping is
    structural and `simp only [tagToMergeExpr]` reduces cleanly. -/
@[reducible]
def tagToMergeExpr {n : Nat} : TyTag → CoreExpr n → CoreExpr n → CoreExpr n
  | .group,   e1, e2 => .merge e1 e2
  | .reduced, e1, e2 => .combineRed e1 e2

/-- Tag → finalize/fence expression constructor. `.fence` consumes a
    `.group`, `.finalize` consumes a `.reduced`. -/
@[reducible]
def tagToFinalExpr {n : Nat} : TyTag → CoreExpr n → CoreExpr n
  | .group,   e => .fence e
  | .reduced, e => .finalize e

/-- Tag → finalize result type. THIS IS THE ASYMMETRY between the two
    domains: Fence's `fence` returns `.unit` (barrier), Reduce's `finalize`
    returns `.data` (extracted value). The gate is the same (`PSet.all n`);
    the output type differs. -/
@[reducible]
def tagToFinalTy {n : Nat} : TyTag → CoreTy n
  | .group   => .unit
  | .reduced => .data

-- ============================================================================
-- Context
-- ============================================================================

/-- Linear context: ordered list of name-type bindings. Structurally
    identical to `FenceCtx` and `ReduceCtx`. -/
def CoreCtx (n : Nat) := List (String × CoreTy n)

namespace CoreCtx

def lookup {n : Nat} (ctx : CoreCtx n) (name : String) : Option (CoreTy n) :=
  ctx.find? (fun p => p.1 == name) |>.map Prod.snd

def remove {n : Nat} (ctx : CoreCtx n) (name : String) : CoreCtx n :=
  ctx.filter (fun p => p.1 != name)

end CoreCtx

-- ============================================================================
-- Typing judgment — Γ ⊢ e : τ ⊣ Γ'
-- ============================================================================

/-- Linear typing for the unified core language.

    Of the 14 constructors below, 12 are monomorphic (structural rules copied
    from Fence / Reduce with at most a rename) and 2 are the family-parametric
    rules validated by Experiment D:

    - `mergeFamily` — subsumes Fence's `merge` and Reduce's `combineRed`.
      Parameterized by a `TyTag` and by double equality witnesses over both
      the expression and the output type.
    - `finalizeFamily` — subsumes Fence's `fence` and Reduce's `finalize`.
      Same double-witness shape. -/
inductive CoreHasType {n : Nat} :
    CoreCtx n → CoreExpr n → CoreTy n → CoreCtx n → Prop

  -- ── Monomorphic value / variable / partition rules ──

  | groupVal (ctx : CoreCtx n) (s : PSet n) :
      CoreHasType ctx (.groupVal s) (.group s) ctx
  | dataVal (ctx : CoreCtx n) :
      CoreHasType ctx .dataVal .data ctx
  | unitVal (ctx : CoreCtx n) :
      CoreHasType ctx .unitVal .unit ctx
  | var (ctx : CoreCtx n) (name : String) (t : CoreTy n) :
      ctx.lookup name = some t →
      CoreHasType ctx (.var name) t (ctx.remove name)
  | diverge (ctx ctx' : CoreCtx n) (g : CoreExpr n) (s pred : PSet n) :
      CoreHasType ctx g (.group s) ctx' →
      CoreHasType ctx (.diverge g pred)
        (.pair (.group (s &&& pred)) (.group (s &&& ~~~pred))) ctx'

  -- ── Let-binding, pair suite ──

  | letBind (ctx ctx' ctx'' : CoreCtx n) (name : String)
      (val body : CoreExpr n) (t1 t2 : CoreTy n) :
      CoreHasType ctx val t1 ctx' →
      ctx'.lookup name = none →
      CoreHasType ((name, t1) :: ctx') body t2 ctx'' →
      ctx''.lookup name = none →
      CoreHasType ctx (.letBind name val body) t2 ctx''
  | pairVal (ctx ctx' ctx'' : CoreCtx n) (a b : CoreExpr n) (t1 t2 : CoreTy n) :
      CoreHasType ctx a t1 ctx' →
      CoreHasType ctx' b t2 ctx'' →
      CoreHasType ctx (.pairVal a b) (.pair t1 t2) ctx''
  | fstE (ctx ctx' : CoreCtx n) (e : CoreExpr n) (t1 t2 : CoreTy n) :
      CoreHasType ctx e (.pair t1 t2) ctx' →
      CoreHasType ctx (.fst e) t1 ctx'
  | sndE (ctx ctx' : CoreCtx n) (e : CoreExpr n) (t1 t2 : CoreTy n) :
      CoreHasType ctx e (.pair t1 t2) ctx' →
      CoreHasType ctx (.snd e) t2 ctx'
  | letPairE (ctx ctx' ctx'' : CoreCtx n) (e : CoreExpr n) (name1 name2 : String)
      (body : CoreExpr n) (t1 t2 t : CoreTy n) :
      CoreHasType ctx e (.pair t1 t2) ctx' →
      name1 ≠ name2 →
      ctx'.lookup name1 = none →
      ctx'.lookup name2 = none →
      CoreHasType ((name2, t2) :: (name1, t1) :: ctx') body t ctx'' →
      ctx''.lookup name1 = none →
      ctx''.lookup name2 = none →
      CoreHasType ctx (.letPair e name1 name2 body) t ctx''

  -- ── Domain-specific monomorphic rules ──

  /-- Fence-domain `write`: consume a payload under a group permission,
      threading the group unchanged. Monomorphic in `.group`. -/
  | write (ctx ctx' ctx'' : CoreCtx n) (g payload : CoreExpr n) (s : PSet n) :
      CoreHasType ctx g (.group s) ctx' →
      CoreHasType ctx' payload .data ctx'' →
      CoreHasType ctx (.write g payload) (.group s) ctx''

  /-- Reduce-domain `leafReduce`: cross-family coercion `.group s → .reduced s`.
      Monomorphic — it is precisely the operation that lifts a permission
      handle into an accumulator at the same participant set. -/
  | leafReduce (ctx ctx' : CoreCtx n) (g : CoreExpr n) (s : PSet n) :
      CoreHasType ctx g (.group s) ctx' →
      CoreHasType ctx (.leafReduce g) (.reduced s) ctx'

  -- ── Family-parametric rules (the factoring target) ──

  /-- Parametric merge. Subsumes Fence's `merge` (tag `.group`) and Reduce's
      `combineRed` (tag `.reduced`).

      Both the expression conclusion AND the output type are abstracted via
      equality witnesses — required because both `tagToMergeExpr tag e1 e2`
      and `tagToTy tag parent` are stuck on `tag` at the `cases`-site
      unification stage. Without abstraction, `cases h` on a concrete-shape
      hypothesis fails dependent elimination in this branch. -/
  | mergeFamily (tag : TyTag)
      (ctx ctx' ctx'' : CoreCtx n) (e1 e2 expr : CoreExpr n)
      (s1 s2 parent : PSet n) (ty : CoreTy n)
      (hExpr : expr = tagToMergeExpr tag e1 e2)
      (hTy : ty = tagToTy tag parent) :
      CoreHasType ctx e1 (tagToTy tag s1) ctx' →
      CoreHasType ctx' e2 (tagToTy tag s2) ctx'' →
      PSet.IsComplement s1 s2 parent →
      CoreHasType ctx expr ty ctx''

  /-- Parametric finalize/fence. Subsumes Fence's `fence` (tag `.group`,
      returns `.unit`) and Reduce's `finalize` (tag `.reduced`, returns
      `.data`). Same double-witness shape as `mergeFamily`. -/
  | finalizeFamily (tag : TyTag)
      (ctx ctx' : CoreCtx n) (e expr : CoreExpr n) (resultTy : CoreTy n)
      (hExpr : expr = tagToFinalExpr tag e)
      (hTy : resultTy = tagToFinalTy tag) :
      CoreHasType ctx e (tagToTy tag (PSet.all n)) ctx' →
      CoreHasType ctx expr resultTy ctx'

-- ============================================================================
-- Generic lemma — the diverge partition transfers to every domain unchanged
-- ============================================================================

/-- The diverge partition property, stated in Core. Fence and Reduce each
    re-export this under their domain-specific name. -/
theorem core_diverge_partition {n : Nat} (s pred : PSet n) :
    PSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    PSet.Covers (s &&& pred) (s &&& ~~~pred) s :=
  diverge_partition_generic s pred
