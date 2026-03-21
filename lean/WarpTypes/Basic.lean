import Std.Tactic.BVDecide

/-
  Warp Typestate: Lean 4 Formalization

  This file formalizes the core type system from
  "Type-Safe GPU Warp Programming via Linear Typestate."

  Machine-checked proofs:
  1. diverge_partition: diverge produces disjoint, covering sub-sets
  2. complement_symmetric: complement relation is symmetric
  3. shuffle_requires_all: shuffle typing requires Warp<All>
  4. even_odd_complement: Even ∧ Odd are complements
  5. lowHalf_highHalf_complement: LowHalf ∧ HighHalf are complements
  6. all_lanes_active: every lane is active in `all` (Lemma 4.6)
-/

-- ============================================================================
-- Active Sets (§3.2) — 32-bit bitmasks
-- ============================================================================

/-- An active set is a 32-bit bitvector representing which lanes are active. -/
abbrev ActiveSet := BitVec 32

namespace ActiveSet

def all : ActiveSet := 0xFFFFFFFF#32
def none : ActiveSet := 0x0#32
def even : ActiveSet := 0x55555555#32
def odd : ActiveSet := 0xAAAAAAAA#32
def lowHalf : ActiveSet := 0x0000FFFF#32
def highHalf : ActiveSet := 0xFFFF0000#32

/-- Two sets are disjoint if their intersection is zero. -/
def Disjoint (a b : ActiveSet) : Prop := a &&& b = none

/-- Two sets cover a parent if their union equals the parent. -/
def Covers (a b parent : ActiveSet) : Prop := a ||| b = parent

/-- Two sets are complements within a parent: disjoint and covering. -/
def IsComplement (a b parent : ActiveSet) : Prop :=
  Disjoint a b ∧ Covers a b parent

/-- Complements within All. -/
def IsComplementAll (a b : ActiveSet) : Prop :=
  IsComplement a b all

end ActiveSet

-- ============================================================================
-- Types and Expressions (§3.3)
-- ============================================================================

inductive Ty
  | warp (s : ActiveSet)
  | perLane
  | unit
  | pair (a b : Ty)      -- Product type for diverge results

inductive Expr
  | warpVal (s : ActiveSet)
  | perLaneVal
  | unitVal
  | var (name : String)
  | diverge (w : Expr) (pred : ActiveSet)
  | merge (w1 w2 : Expr)
  | shuffle (w data : Expr)
  | letBind (name : String) (val body : Expr)
  | pairVal (a b : Expr)  -- Pair constructor
  | fst (e : Expr)        -- First projection
  | snd (e : Expr)        -- Second projection
  | letPair (e : Expr) (name1 name2 : String) (body : Expr)  -- Linear pair destructor
  | loopUniform (n : Nat) (warpName : String) (warp body : Expr)  -- §5.1 uniform loop
  | loopVarying (warp body : Expr)  -- §5.1 varying loop (warp-free body)
  | loopPhased (n : Nat) (warpName : String) (warp uniformBody varyingBody : Expr)  -- §5.1 phased

-- ============================================================================
-- Typing Context (linear)
-- ============================================================================

def Ctx := List (String × Ty)

def Ctx.lookup (ctx : Ctx) (name : String) : Option Ty :=
  ctx.find? (fun p => p.1 == name) |>.map Prod.snd

def Ctx.remove (ctx : Ctx) (name : String) : Ctx :=
  ctx.filter (fun p => p.1 != name)

-- ============================================================================
-- Warp-free predicate (§5.1 LOOP-VARYING)
-- ============================================================================

/-- An expression is warp-free if it contains no warp operations.
    Such expressions cannot introduce divergence bugs — the warp
    passes through unchanged. -/
def warpFree : Expr → Bool
  | .warpVal _ => false
  | .diverge _ _ => false
  | .merge _ _ => false
  | .shuffle _ _ => false
  | .loopUniform _ _ _ _ => false
  | .loopVarying _ _ => false
  | .loopPhased _ _ _ _ _ => false
  | .perLaneVal => true
  | .unitVal => true
  | .var _ => true
  | .letBind _ val body => warpFree val && warpFree body
  | .pairVal a b => warpFree a && warpFree b
  | .fst e => warpFree e
  | .snd e => warpFree e
  | .letPair e _ _ body => warpFree e && warpFree body

-- ============================================================================
-- Typing Rules (§3.3)
-- ============================================================================

/-- Linear typing judgement: Γ ⊢ e : τ ⊣ Γ' -/
inductive HasType : Ctx → Expr → Ty → Ctx → Prop
  | warpVal (ctx : Ctx) (s : ActiveSet) :
      HasType ctx (.warpVal s) (.warp s) ctx
  | perLaneVal (ctx : Ctx) :
      HasType ctx .perLaneVal .perLane ctx
  | unitVal (ctx : Ctx) :
      HasType ctx .unitVal .unit ctx
  | var (ctx : Ctx) (name : String) (t : Ty) :
      ctx.lookup name = some t →
      HasType ctx (.var name) t (ctx.remove name)
  | diverge (ctx ctx' : Ctx) (w : Expr) (s pred : ActiveSet) :
      HasType ctx w (.warp s) ctx' →
      HasType ctx (.diverge w pred)
        (.pair (.warp (s &&& pred)) (.warp (s &&& ~~~pred))) ctx'
  | merge (ctx ctx' ctx'' : Ctx) (w1 w2 : Expr) (s1 s2 parent : ActiveSet) :
      HasType ctx w1 (.warp s1) ctx' →
      HasType ctx' w2 (.warp s2) ctx'' →
      ActiveSet.IsComplement s1 s2 parent →
      HasType ctx (.merge w1 w2) (.warp parent) ctx''
  | shuffle (ctx ctx' ctx'' : Ctx) (w data : Expr) :
      HasType ctx w (.warp ActiveSet.all) ctx' →
      HasType ctx' data .perLane ctx'' →
      HasType ctx (.shuffle w data) .perLane ctx''
  | letBind (ctx ctx' ctx'' : Ctx) (name : String) (val body : Expr) (t1 t2 : Ty) :
      HasType ctx val t1 ctx' →
      ctx'.lookup name = none →          -- freshness: no shadowing
      HasType ((name, t1) :: ctx') body t2 ctx'' →
      ctx''.lookup name = none →         -- linearity: binding was consumed
      HasType ctx (.letBind name val body) t2 ctx''
  | pairVal (ctx ctx' ctx'' : Ctx) (a b : Expr) (t1 t2 : Ty) :
      HasType ctx a t1 ctx' →
      HasType ctx' b t2 ctx'' →
      HasType ctx (.pairVal a b) (.pair t1 t2) ctx''
  | fstE (ctx ctx' : Ctx) (e : Expr) (t1 t2 : Ty) :
      HasType ctx e (.pair t1 t2) ctx' →
      HasType ctx (.fst e) t1 ctx'
  | sndE (ctx ctx' : Ctx) (e : Expr) (t1 t2 : Ty) :
      HasType ctx e (.pair t1 t2) ctx' →
      HasType ctx (.snd e) t2 ctx'
  | letPairE (ctx ctx' ctx'' : Ctx) (e : Expr) (name1 name2 : String)
      (body : Expr) (t1 t2 t : Ty) :
      HasType ctx e (.pair t1 t2) ctx' →
      name1 ≠ name2 →
      ctx'.lookup name1 = none →
      ctx'.lookup name2 = none →
      HasType ((name2, t2) :: (name1, t1) :: ctx') body t ctx'' →
      ctx''.lookup name1 = none →
      ctx''.lookup name2 = none →
      HasType ctx (.letPair e name1 name2 body) t ctx''
  | loopUniform (ctx ctx' : Ctx) (n : Nat) (warpName : String)
      (warp body : Expr) (s : ActiveSet) :
      HasType ctx warp (.warp s) ctx' →
      ctx'.lookup warpName = none →
      HasType ((warpName, .warp s) :: ctx') body (.warp s) ctx' →
      HasType ctx (.loopUniform n warpName warp body) (.warp s) ctx'
  | loopVarying (ctx ctx' : Ctx) (warp body : Expr) (s : ActiveSet) :
      HasType ctx warp (.warp s) ctx' →
      warpFree body = true →
      HasType ctx (.loopVarying warp body) (.warp s) ctx'
  | loopPhased (ctx ctx' : Ctx) (n : Nat) (warpName : String)
      (warp uniformBody varyingBody : Expr) (s : ActiveSet) :
      HasType ctx warp (.warp s) ctx' →
      ctx'.lookup warpName = none →
      HasType ((warpName, .warp s) :: ctx') uniformBody (.warp s) ctx' →
      warpFree varyingBody = true →
      HasType ctx (.loopPhased n warpName warp uniformBody varyingBody) (.warp s) ctx'

-- ============================================================================
-- Theorem 4.1: Diverge Partition
-- ============================================================================

/-- diverge produces sets that are disjoint and cover the parent.
    This is the core soundness property: S = (S∩P) ⊔ (S∩¬P) with (S∩P) ⊓ (S∩¬P) = ∅. -/
theorem diverge_partition (s pred : ActiveSet) :
    ActiveSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    ActiveSet.Covers (s &&& pred) (s &&& ~~~pred) s := by
  unfold ActiveSet.Disjoint ActiveSet.Covers ActiveSet.none
  constructor
  · ext i; simp_all
  · ext i; simp_all; cases s[i] <;> simp

-- ============================================================================
-- Theorem: Shuffle requires All
-- ============================================================================

theorem shuffle_requires_all {ctx ctx'' : Ctx} {w data : Expr} {t : Ty} :
    HasType ctx (.shuffle w data) t ctx'' →
    ∃ ctx', HasType ctx w (.warp ActiveSet.all) ctx' := by
  intro h
  cases h with
  | shuffle _ ctx' _ _ _ hw _ => exact ⟨ctx', hw⟩

-- ============================================================================
-- Lemma: Complement Symmetry
-- ============================================================================

theorem complement_symmetric {a b : ActiveSet} :
    ActiveSet.IsComplementAll a b → ActiveSet.IsComplementAll b a := by
  intro ⟨hdisj, hcov⟩
  unfold ActiveSet.IsComplementAll ActiveSet.IsComplement at *
  unfold ActiveSet.Disjoint ActiveSet.Covers at *
  constructor
  · rw [BitVec.and_comm]; exact hdisj
  · rw [BitVec.or_comm]; exact hcov

-- ============================================================================
-- Concrete Complement Instances
-- ============================================================================

theorem even_odd_complement : ActiveSet.IsComplementAll ActiveSet.even ActiveSet.odd := by
  unfold ActiveSet.IsComplementAll ActiveSet.IsComplement
  unfold ActiveSet.Disjoint ActiveSet.Covers
  unfold ActiveSet.even ActiveSet.odd ActiveSet.all ActiveSet.none
  constructor <;> decide

theorem lowHalf_highHalf_complement :
    ActiveSet.IsComplementAll ActiveSet.lowHalf ActiveSet.highHalf := by
  unfold ActiveSet.IsComplementAll ActiveSet.IsComplement
  unfold ActiveSet.Disjoint ActiveSet.Covers
  unfold ActiveSet.lowHalf ActiveSet.highHalf ActiveSet.all ActiveSet.none
  constructor <;> decide

-- ============================================================================
-- Nested Complement Instances (§3.4)
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
  unfold ActiveSet.IsComplement ActiveSet.Disjoint ActiveSet.Covers
  unfold ActiveSet.evenLow ActiveSet.evenHigh ActiveSet.even ActiveSet.none
  constructor <;> decide

-- ============================================================================
-- Theorem: All Lanes Active (Lemma 4.6 correspondence)
-- ============================================================================

/-- Every lane is active in the `all` set (Lemma 4.6 correspondence). -/
theorem all_lanes_active (i : Fin 32) : ActiveSet.all[i] = true := by
  revert i; decide

-- ============================================================================
-- Values
-- ============================================================================

def isValue : Expr → Bool
  | .warpVal _ => true
  | .perLaneVal => true
  | .unitVal => true
  | .pairVal a b => isValue a && isValue b
  | .letPair _ _ _ _ => false
  | .loopUniform _ _ _ _ => false
  | .loopVarying _ _ => false
  | .loopPhased _ _ _ _ _ => false
  | _ => false

-- ============================================================================
-- Reflexive-Transitive Closure
-- ============================================================================

/-- Reflexive-transitive closure of a relation. Used for multi-step reduction. -/
inductive Star (R : α → α → Prop) : α → α → Prop
  | refl : Star R a a
  | step : R a b → Star R b c → Star R a c

