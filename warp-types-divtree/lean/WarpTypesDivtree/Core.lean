import WarpTypesBitwise.CUDA

/-!
# WarpTypesDivtree.Core

Nested divergence partition tree at generic `BitVec n` width.

## Overview

A `DivTree n` is a binary tree whose internal nodes split a parent
bitmask into two complementary halves via a predicate:

- **leaf** — a terminal bitmask
- **node** — a parent mask + a predicate + left/right subtrees

A tree is `WellFormed` when each internal node's children carry the
predicate-split of the parent: `root left = mask &&& pred` and
`root right = mask &&& ~~~pred`. The four soundness theorems together
state that a well-formed tree's leaves form a disjoint cover of its
root, which is the soundness property for arbitrarily-nested warp
divergence:

- `leaves_cover_root` — OR-folding all leaves recovers the root
- `leaves_pairwise_disjoint` — leaves are pairwise disjoint
- `leaf_subset_root` — every leaf is an AND-subset of the root
- `leaves_length` — a tree with `k` internal nodes has `k + 1` leaves

## Dependency posture

Mathlib-free and Sol-free. Uses only Lean 4.28 core's `BitVec` /
`List` APIs plus `ext` + `simp` proofs of the underlying bitwise
identities, plus a Lake path dependency on `warp-types-bitwise` for
the shared `foldl_or_lift` helper and the `ballot_split` fold-append
theorem (hoisted in v0.2.0 — v0.1.0 inlined these as private
theorems). The remaining helpers (`and_complement_cover`,
`subset_of_subset_and`, `cross_disjoint`, `foldl_or_singleton`) are
divtree-specific shapes about predicate splits and complement
disjointness that have no analog in bitwise; they stay local.
-/

namespace WarpTypesDivtree

section
variable {n : Nat}

-- =========================================================================
-- 1. Divtree-local bitwise helper lemmas
-- =========================================================================
-- These correspond to Sol's `sol_bv_*` tactics for predicate-split and
-- complement-disjointness shapes. They have no analog in
-- `warp-types-bitwise`, so they stay local to divtree. The fold-lift
-- and fold-append helpers were hoisted to `warp-types-bitwise` in
-- v0.2.0 (`foldl_or_lift` and `ballot_split`); see the leaves_cover_root
-- proof below for the call sites.

/-- `(m &&& p) ||| (m &&& ~~~p) = m` — a mask equals its predicate-split
    halves OR'd together. The key algebraic fact behind the diverge
    tree: each split is a lossless partition. -/
private theorem and_complement_cover (m p : BitVec n) :
    (m &&& p) ||| (m &&& ~~~p) = m := by
  ext i hi
  simp [BitVec.getElem_or, BitVec.getElem_and, BitVec.getElem_not]
  cases p[i]'hi <;> simp

/-- `List.foldl (· ||| ·) 0 [x] = x` — OR-fold of a singleton is the
    element. Used as the leaf case of `leaves_cover_root`. -/
private theorem foldl_or_singleton (x : BitVec n) :
    List.foldl (· ||| ·) 0 [x] = x := by
  show 0 ||| x = x
  exact BitVec.zero_or

/-- If `m` is contained in `mask &&& pred`, then `m` is contained in
    `mask`. Per-bit: if every `true` bit of `m` is in both `mask` and
    `pred`, then every `true` bit of `m` is in `mask`.

    Proof note: after `simp [BitVec.getElem_and]`, the hypothesis and
    goal are both in implication form (pattern 1 from
    `feedback_lean4_bv_proofs.md`) — use `intro` + `.1` projection, do
    NOT `cases` on `m[i]`. -/
private theorem subset_of_subset_and {m mask pred : BitVec n}
    (h : m &&& (mask &&& pred) = m) : m &&& mask = m := by
  ext i hi
  have h_bit : (m &&& (mask &&& pred))[i]'hi = m[i]'hi :=
    congrArg (·[i]'hi) h
  simp [BitVec.getElem_and] at h_bit
  -- h_bit : m[i] = true → mask[i] = true ∧ pred[i] = true
  simp [BitVec.getElem_and]
  -- Goal: m[i] = true → mask[i] = true
  intro hm
  exact (h_bit hm).1

/-- Two masks contained in complementary halves of a parent are disjoint.
    If `a ⊆ mask & pred` and `b ⊆ mask & ~pred`, then `a & b = 0`. -/
private theorem cross_disjoint {a b mask pred : BitVec n}
    (ha : a &&& (mask &&& pred) = a)
    (hb : b &&& (mask &&& ~~~pred) = b) :
    a &&& b = 0#n := by
  ext i hi
  simp [BitVec.getElem_and, BitVec.getElem_zero]
  have ha_bit : (a &&& (mask &&& pred))[i]'hi = a[i]'hi :=
    congrArg (·[i]'hi) ha
  have hb_bit : (b &&& (mask &&& ~~~pred))[i]'hi = b[i]'hi :=
    congrArg (·[i]'hi) hb
  simp [BitVec.getElem_and, BitVec.getElem_not] at ha_bit hb_bit
  cases hai : a[i]'hi
  · simp
  · cases hbi : b[i]'hi
    · simp
    · -- a[i] = true and b[i] = true; derive contradiction from pred[i]
      simp [hai] at ha_bit
      simp [hbi] at hb_bit
      -- ha_bit : mask[i] ∧ pred[i] = true  (both true)
      -- hb_bit : mask[i] ∧ ¬pred[i] = true
      exact absurd ha_bit.2 (by simp [hb_bit.2])

end

-- =========================================================================
-- 2. The DivTree type
-- =========================================================================

/-- A diverge tree at generic width `n`. Each internal node splits its
    mask into `mask &&& pred` (left) and `mask &&& ~~~pred` (right). -/
inductive DivTree (n : Nat) where
  | leaf (mask : BitVec n)
  | node (mask pred : BitVec n) (left right : DivTree n)

namespace DivTree

section
variable {n : Nat}

/-- The root bitmask of a tree. -/
def root : DivTree n → BitVec n
  | .leaf mask => mask
  | .node mask _ _ _ => mask

/-- Collect all leaf bitmasks left-to-right. -/
def leaves : DivTree n → List (BitVec n)
  | .leaf mask => [mask]
  | .node _ _ left right => left.leaves ++ right.leaves

/-- Number of internal nodes in a tree. -/
def nodeCount : DivTree n → Nat
  | .leaf _ => 0
  | .node _ _ left right => 1 + left.nodeCount + right.nodeCount

-- =========================================================================
-- 3. Well-formedness: each node's children partition the parent
-- =========================================================================

/-- A tree is well-formed when each internal node's children are the
    predicate-split of the parent mask. -/
inductive WellFormed : DivTree n → Prop where
  | leaf (mask : BitVec n) : WellFormed (.leaf mask)
  | node {mask pred : BitVec n} {left right : DivTree n} :
      root left = mask &&& pred →
      root right = mask &&& ~~~pred →
      WellFormed left → WellFormed right →
      WellFormed (.node mask pred left right)

-- =========================================================================
-- 4. Soundness theorems
-- =========================================================================

/-- OR-ing all leaves of a well-formed tree recovers the root.
    After arbitrarily many diverge operations, merging all leaf warps
    (via OR) recovers the original state. -/
theorem leaves_cover_root {t : DivTree n} (hwf : WellFormed t) :
    List.foldl (· ||| ·) 0 (leaves t) = root t := by
  induction hwf with
  | leaf mask => exact foldl_or_singleton mask
  | node hleft hright _wfl _wfr ihl ihr =>
    simp only [leaves, root]
    rw [WarpTypesBitwise.ballot_split, ihl, ihr, hleft, hright]
    exact and_complement_cover _ _

/-- A tree with `k` internal nodes has `k + 1` leaves. Size invariant
    that justifies static leaf-count capacity bounds in downstream RTL. -/
theorem leaves_length {t : DivTree n} :
    (leaves t).length = nodeCount t + 1 := by
  induction t with
  | leaf _ => rfl
  | node _ _ left right ihl ihr =>
    simp only [leaves, nodeCount, List.length_append, ihl, ihr]
    omega

/-- Every leaf mask is contained in (AND-subset of) the tree root.
    Combined with `leaves_cover_root`, this gives a full partition. -/
theorem leaf_subset_root {t : DivTree n} (hwf : WellFormed t)
    {m : BitVec n} (hm : m ∈ t.leaves) : m &&& t.root = m := by
  induction hwf with
  | leaf mask =>
    simp [leaves] at hm
    subst hm
    -- After `subst hm`, goal is `mask &&& (DivTree.leaf mask).root = mask`.
    -- `simp [root]` unfolds root and closes via Bool's `and_self` simp lemma.
    simp [root]
  | node hleft hright _wfl _wfr ihl ihr =>
    simp only [leaves, List.mem_append] at hm
    simp only [root]
    cases hm with
    | inl hml =>
      have := ihl hml
      rw [hleft] at this
      exact subset_of_subset_and this
    | inr hmr =>
      have := ihr hmr
      rw [hright] at this
      exact subset_of_subset_and this

/-- All leaves of a well-formed tree are pairwise disjoint.
    Together with `leaves_cover_root`, this gives a full partition:
    the leaves are a disjoint cover of the root. -/
theorem leaves_pairwise_disjoint {t : DivTree n} (hwf : WellFormed t) :
    List.Pairwise (fun a b => a &&& b = 0#n) (leaves t) := by
  induction hwf with
  | leaf _ => exact List.pairwise_singleton _ _
  | node hleft hright wfl wfr ihl ihr =>
    simp only [leaves]
    rw [List.pairwise_append]
    refine ⟨ihl, ihr, fun a ha b hb => ?_⟩
    have hsa := leaf_subset_root wfl ha
    rw [hleft] at hsa
    have hsb := leaf_subset_root wfr hb
    rw [hright] at hsb
    exact cross_disjoint hsa hsb

end
end DivTree
end WarpTypesDivtree
