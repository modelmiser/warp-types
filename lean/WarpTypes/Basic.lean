import WarpTypes.Generic

/-
  Warp Typestate: Lean 4 Formalization

  This file formalizes the core type system from
  "Type-Safe GPU Warp Programming via Linear Typestate."

  Width-generic core (PSet n) is in Generic.lean.
  This file provides:
  - GPU instantiation: ActiveSet := PSet 32
  - Concrete GPU lane patterns (even, odd, lowHalf, highHalf)
  - Width-parameterized type system (Ty n, Expr n, HasType, Step)
  - Generic theorems (diverge_partition, complement_symmetric, shuffle_requires_all)
  - Concrete n=32 theorems (even_odd_complement, all_lanes_active, etc.)
-/

-- ============================================================================
-- GPU Instantiation: ActiveSet = PSet 32 (§3.2)
-- ============================================================================

/-- An active set is a 32-bit bitvector representing which lanes are active. -/
abbrev ActiveSet := PSet 32

namespace ActiveSet

def all : ActiveSet := PSet.all 32
def none : ActiveSet := PSet.none 32
def even : ActiveSet := 0x55555555#32
def odd : ActiveSet := 0xAAAAAAAA#32
def lowHalf : ActiveSet := 0x0000FFFF#32
def highHalf : ActiveSet := 0xFFFF0000#32

/-- Two sets are disjoint if their intersection is zero. -/
def Disjoint (a b : ActiveSet) : Prop := PSet.Disjoint a b

/-- Two sets cover a parent if their union equals the parent. -/
def Covers (a b parent : ActiveSet) : Prop := PSet.Covers a b parent

/-- Two sets are complements within a parent: disjoint and covering. -/
def IsComplement (a b parent : ActiveSet) : Prop := PSet.IsComplement a b parent

/-- Complements within All. -/
def IsComplementAll (a b : ActiveSet) : Prop := PSet.IsComplementAll a b

end ActiveSet

-- ============================================================================
-- Types and Expressions (§3.3) — parameterized by width n
-- ============================================================================

inductive Ty (n : Nat)
  | warp (s : PSet n)
  | perLane
  | unit
  | pair (a b : Ty n)      -- Product type for diverge results

inductive Expr (n : Nat)
  | warpVal (s : PSet n)
  | perLaneVal
  | unitVal
  | var (name : String)
  | diverge (w : Expr n) (pred : PSet n)
  | merge (w1 w2 : Expr n)
  | shuffle (w data : Expr n)
  | letBind (name : String) (val body : Expr n)
  | pairVal (a b : Expr n)  -- Pair constructor
  | fst (e : Expr n)        -- First projection
  | snd (e : Expr n)        -- Second projection
  | letPair (e : Expr n) (name1 name2 : String) (body : Expr n)  -- Linear pair destructor
  | loopUniform (k : Nat) (warpName : String) (warp body : Expr n)  -- §5.1 uniform loop
  | loopVarying (warp body : Expr n)  -- §5.1 varying loop (warp-free body)
  | loopPhased (k : Nat) (warpName : String) (warp uniformBody varyingBody : Expr n)  -- §5.1 phased
  | loopConvergent (k : Nat) (warpName : String) (warp body : Expr n)  -- §5.1 convergent loop

-- ============================================================================
-- Typing Context (linear) — parameterized by width n
-- ============================================================================

def Ctx (n : Nat) := List (String × Ty n)

def Ctx.lookup {n : Nat} (ctx : Ctx n) (name : String) : Option (Ty n) :=
  ctx.find? (fun p => p.1 == name) |>.map Prod.snd

def Ctx.remove {n : Nat} (ctx : Ctx n) (name : String) : Ctx n :=
  ctx.filter (fun p => p.1 != name)

-- ============================================================================
-- Warp-free predicate (§5.1 LOOP-VARYING)
-- ============================================================================

/-- An expression is warp-free if it contains no warp operations.
    Such expressions cannot introduce divergence bugs — the warp
    passes through unchanged. -/
def warpFree {n : Nat} : Expr n → Bool
  | .warpVal _ => false
  | .diverge _ _ => false
  | .merge _ _ => false
  | .shuffle _ _ => false
  | .loopUniform _ _ _ _ => false
  | .loopVarying _ _ => false
  | .loopPhased _ _ _ _ _ => false
  | .loopConvergent _ _ _ _ => false
  | .perLaneVal => true
  | .unitVal => true
  | .var _ => true
  | .letBind _ val body => warpFree val && warpFree body
  | .pairVal a b => warpFree a && warpFree b
  | .fst e => warpFree e
  | .snd e => warpFree e
  | .letPair e _ _ body => warpFree e && warpFree body

-- ============================================================================
-- Typing Rules (§3.3) — parameterized by width n
-- ============================================================================

/-- Linear typing judgement: Γ ⊢ e : τ ⊣ Γ' -/
inductive HasType {n : Nat} : Ctx n → Expr n → Ty n → Ctx n → Prop
  | warpVal (ctx : Ctx n) (s : PSet n) :
      HasType ctx (.warpVal s) (.warp s) ctx
  | perLaneVal (ctx : Ctx n) :
      HasType ctx .perLaneVal .perLane ctx
  | unitVal (ctx : Ctx n) :
      HasType ctx .unitVal .unit ctx
  | var (ctx : Ctx n) (name : String) (t : Ty n) :
      ctx.lookup name = some t →
      HasType ctx (.var name) t (ctx.remove name)
  | diverge (ctx ctx' : Ctx n) (w : Expr n) (s pred : PSet n) :
      HasType ctx w (.warp s) ctx' →
      HasType ctx (.diverge w pred)
        (.pair (.warp (s &&& pred)) (.warp (s &&& ~~~pred))) ctx'
  | merge (ctx ctx' ctx'' : Ctx n) (w1 w2 : Expr n) (s1 s2 parent : PSet n) :
      HasType ctx w1 (.warp s1) ctx' →
      HasType ctx' w2 (.warp s2) ctx'' →
      PSet.IsComplement s1 s2 parent →
      HasType ctx (.merge w1 w2) (.warp parent) ctx''
  | shuffle (ctx ctx' ctx'' : Ctx n) (w data : Expr n) :
      HasType ctx w (.warp (PSet.all n)) ctx' →
      HasType ctx' data .perLane ctx'' →
      HasType ctx (.shuffle w data) .perLane ctx''
  | letBind (ctx ctx' ctx'' : Ctx n) (name : String) (val body : Expr n) (t1 t2 : Ty n) :
      HasType ctx val t1 ctx' →
      ctx'.lookup name = none →          -- freshness: no shadowing
      HasType ((name, t1) :: ctx') body t2 ctx'' →
      ctx''.lookup name = none →         -- linearity: binding was consumed
      HasType ctx (.letBind name val body) t2 ctx''
  | pairVal (ctx ctx' ctx'' : Ctx n) (a b : Expr n) (t1 t2 : Ty n) :
      HasType ctx a t1 ctx' →
      HasType ctx' b t2 ctx'' →
      HasType ctx (.pairVal a b) (.pair t1 t2) ctx''
  | fstE (ctx ctx' : Ctx n) (e : Expr n) (t1 t2 : Ty n) :
      HasType ctx e (.pair t1 t2) ctx' →
      HasType ctx (.fst e) t1 ctx'
  | sndE (ctx ctx' : Ctx n) (e : Expr n) (t1 t2 : Ty n) :
      HasType ctx e (.pair t1 t2) ctx' →
      HasType ctx (.snd e) t2 ctx'
  | letPairE (ctx ctx' ctx'' : Ctx n) (e : Expr n) (name1 name2 : String)
      (body : Expr n) (t1 t2 t : Ty n) :
      HasType ctx e (.pair t1 t2) ctx' →
      name1 ≠ name2 →
      ctx'.lookup name1 = none →
      ctx'.lookup name2 = none →
      HasType ((name2, t2) :: (name1, t1) :: ctx') body t ctx'' →
      ctx''.lookup name1 = none →
      ctx''.lookup name2 = none →
      HasType ctx (.letPair e name1 name2 body) t ctx''
  | loopUniform (ctx ctx' : Ctx n) (k : Nat) (warpName : String)
      (warp body : Expr n) (s : PSet n) :
      HasType ctx warp (.warp s) ctx' →
      ctx'.lookup warpName = none →
      HasType ((warpName, .warp s) :: ctx') body (.warp s) ctx' →
      HasType ctx (.loopUniform k warpName warp body) (.warp s) ctx'
  | loopVarying (ctx ctx' : Ctx n) (warp body : Expr n) (s : PSet n) :
      HasType ctx warp (.warp s) ctx' →
      warpFree body = true →
      HasType ctx (.loopVarying warp body) (.warp s) ctx'
  | loopPhased (ctx ctx' : Ctx n) (k : Nat) (warpName : String)
      (warp uniformBody varyingBody : Expr n) (s : PSet n) :
      HasType ctx warp (.warp s) ctx' →
      ctx'.lookup warpName = none →
      HasType ((warpName, .warp s) :: ctx') uniformBody (.warp s) ctx' →
      warpFree varyingBody = true →
      HasType ctx (.loopPhased k warpName warp uniformBody varyingBody) (.warp s) ctx'
  | loopConvergent (ctx ctx' : Ctx n) (k : Nat) (warpName : String)
      (warp body : Expr n) (s : PSet n) :
      HasType ctx warp (.warp s) ctx' →
      ctx'.lookup warpName = none →
      HasType ((warpName, .warp s) :: ctx') body (.warp s) ctx' →
      HasType ctx (.loopConvergent k warpName warp body) (.warp s) ctx'

-- ============================================================================
-- Theorem 4.1: Diverge Partition (width-generic, delegates to Generic.lean)
-- ============================================================================

/-- diverge produces sets that are disjoint and cover the parent.
    This is the core soundness property: S = (S∩P) ⊔ (S∩¬P) with (S∩P) ⊓ (S∩¬P) = ∅.
    Now generic over any width n. -/
theorem diverge_partition {n : Nat} (s pred : PSet n) :
    PSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    PSet.Covers (s &&& pred) (s &&& ~~~pred) s :=
  diverge_partition_generic s pred

-- ============================================================================
-- Theorem: Shuffle requires All (width-generic)
-- ============================================================================

theorem shuffle_requires_all {n : Nat} {ctx ctx'' : Ctx n} {w data : Expr n} {t : Ty n} :
    HasType ctx (.shuffle w data) t ctx'' →
    ∃ ctx', HasType ctx w (.warp (PSet.all n)) ctx' := by
  intro h
  cases h with
  | shuffle _ ctx' _ _ _ hw _ => exact ⟨ctx', hw⟩

-- ============================================================================
-- Lemma: Complement Symmetry (width-generic, delegates to Generic.lean)
-- ============================================================================

theorem complement_symmetric {n : Nat} {a b : PSet n} :
    PSet.IsComplementAll a b → PSet.IsComplementAll b a :=
  complement_symmetric_generic

-- ============================================================================
-- Concrete Complement Instances (n=32)
-- ============================================================================

theorem even_odd_complement : ActiveSet.IsComplementAll ActiveSet.even ActiveSet.odd := by
  unfold ActiveSet.IsComplementAll ActiveSet.even ActiveSet.odd
  unfold PSet.IsComplementAll PSet.IsComplement
  unfold PSet.Disjoint PSet.Covers PSet.all PSet.none
  constructor <;> decide

theorem lowHalf_highHalf_complement :
    ActiveSet.IsComplementAll ActiveSet.lowHalf ActiveSet.highHalf := by
  unfold ActiveSet.IsComplementAll ActiveSet.lowHalf ActiveSet.highHalf
  unfold PSet.IsComplementAll PSet.IsComplement
  unfold PSet.Disjoint PSet.Covers PSet.all PSet.none
  constructor <;> decide

-- ============================================================================
-- Nested Complement Instances (§3.4, n=32)
-- ============================================================================

namespace ActiveSet

/-- EvenLow: lanes that are both even AND in the low half. -/
def evenLow : ActiveSet := 0x00005555#32

/-- EvenHigh: lanes that are both even AND in the high half. -/
def evenHigh : ActiveSet := 0x55550000#32

end ActiveSet

/-- EvenLow and EvenHigh are complements within Even (NOT within All).
    This demonstrates nested divergence: diverge into Even/Odd, then
    further diverge Even into EvenLow/EvenHigh. -/
theorem evenLow_evenHigh_complement_within_even :
    ActiveSet.IsComplement ActiveSet.evenLow ActiveSet.evenHigh ActiveSet.even := by
  unfold ActiveSet.IsComplement ActiveSet.evenLow ActiveSet.evenHigh ActiveSet.even
  unfold PSet.IsComplement PSet.Disjoint PSet.Covers PSet.none
  constructor <;> decide

-- ============================================================================
-- Theorem: All Lanes Active (Lemma 4.6 correspondence, n=32)
-- ============================================================================

/-- Every lane is active in the `all` set (Lemma 4.6 correspondence). -/
theorem all_lanes_active (i : Fin 32) : ActiveSet.all[i] = true := by
  revert i; decide

-- ============================================================================
-- POSITIVE instance: merge reconverges warp, enabling shuffle (n=32)
-- ============================================================================

/-- Merging even and odd lane handles (which are complements within All)
    reconverges the warp, enabling a shuffle on the full participant set.
    This is the GPU-domain positive typability witness, parallel to
    `fence_after_full_write_typable` (Fence), `finalize_tree_reduce_typable`
    (Reduce), and `j1_send_adjacent_typable` (CSP). -/
theorem merge_then_shuffle_typable :
    ∃ ctx', HasType (n := 32) []
      (.shuffle (.merge (.warpVal ActiveSet.even) (.warpVal ActiveSet.odd)) .perLaneVal)
      .perLane ctx' := by
  refine ⟨[], ?_⟩
  exact HasType.shuffle [] [] []
    (.merge (.warpVal ActiveSet.even) (.warpVal ActiveSet.odd))
    .perLaneVal
    (HasType.merge [] [] []
      (.warpVal ActiveSet.even) (.warpVal ActiveSet.odd)
      ActiveSet.even ActiveSet.odd (PSet.all 32)
      (HasType.warpVal [] ActiveSet.even)
      (HasType.warpVal [] ActiveSet.odd)
      even_odd_complement)
    (HasType.perLaneVal [])

-- ============================================================================
-- Values — parameterized by width n
-- ============================================================================

def isValue {n : Nat} : Expr n → Bool
  | .warpVal _ => true
  | .perLaneVal => true
  | .unitVal => true
  | .pairVal a b => isValue a && isValue b
  | .letPair _ _ _ _ => false
  | .loopUniform _ _ _ _ => false
  | .loopVarying _ _ => false
  | .loopPhased _ _ _ _ _ => false
  | .loopConvergent _ _ _ _ => false
  | _ => false

-- ============================================================================
-- Reflexive-Transitive Closure
-- ============================================================================

/-- Reflexive-transitive closure of a relation. Used for multi-step reduction. -/
inductive Star (R : α → α → Prop) : α → α → Prop
  | refl : Star R a a
  | step : R a b → Star R b c → Star R a c
