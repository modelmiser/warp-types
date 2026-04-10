import Std.Tactic.BVDecide

/-
  Generic Participant Sets — Width-Parameterized Core

  PSet n is a generic n-bit participant set. The GPU-specific ActiveSet (n=32)
  is defined in Basic.lean as an instantiation. All bitwise-algebraic theorems
  that don't depend on a concrete width live here.
-/

-- ============================================================================
-- Level 0: Generic Participant Sets
-- ============================================================================

/-- A participant set is an n-bit bitvector representing which participants are active. -/
abbrev PSet (n : Nat) := BitVec n

namespace PSet

def all (n : Nat) : PSet n := BitVec.allOnes n
def none (n : Nat) : PSet n := 0#n

/-- Two sets are disjoint if their intersection is zero. -/
def Disjoint {n : Nat} (a b : PSet n) : Prop := a &&& b = none n

/-- Two sets cover a parent if their union equals the parent. -/
def Covers {n : Nat} (a b parent : PSet n) : Prop := a ||| b = parent

/-- Two sets are complements within a parent: disjoint and covering. -/
def IsComplement {n : Nat} (a b parent : PSet n) : Prop :=
  Disjoint a b ∧ Covers a b parent

/-- Complements within All. -/
def IsComplementAll {n : Nat} (a b : PSet n) : Prop :=
  IsComplement a b (all n)

end PSet

-- ============================================================================
-- Level 1: Width-Generic Theorems
-- ============================================================================

/-- diverge produces sets that are disjoint and cover the parent.
    This is the core soundness property: S = (S∩P) ⊔ (S∩¬P) with (S∩P) ⊓ (S∩¬P) = ∅.
    Generic over any width n. -/
theorem diverge_partition_generic {n : Nat} (s pred : PSet n) :
    PSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    PSet.Covers (s &&& pred) (s &&& ~~~pred) s := by
  unfold PSet.Disjoint PSet.Covers PSet.none
  constructor
  · ext i; simp_all
  · ext i; simp_all; cases s[i] <;> simp

/-- Complement relation is symmetric. Generic over any width n. -/
theorem complement_symmetric_generic {n : Nat} {a b : PSet n} :
    PSet.IsComplementAll a b → PSet.IsComplementAll b a := by
  intro ⟨hdisj, hcov⟩
  unfold PSet.IsComplementAll PSet.IsComplement at *
  unfold PSet.Disjoint PSet.Covers at *
  constructor
  · rw [BitVec.and_comm]; exact hdisj
  · rw [BitVec.or_comm]; exact hcov
