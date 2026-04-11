/-!
# WarpTypesBitwise.CUDA

CUDA warp-mask bitvector lemmas.

Six lemmas covering the mask-safety and warp-vote shapes that arise when
CUDA kernels are verified. Three of them (`ballot_split`, `all_sync_split`,
`any_sync_monotone`) are fold-algebra theorems over lane-wise aggregation
and may migrate to a future `warp-types-ballot` sibling crate.

All lemmas are width-parametric in `n : Nat` — specialise at `n = 32` for
NVIDIA warps, or `n = 64` for AMD wavefronts.
-/

namespace WarpTypesBitwise

section
variable {n : Nat}

-- ============================================================================
-- Fold-lift helpers (public so sibling crates can depend on them)
-- ============================================================================
-- These were private in v0.1.0 (used only by ballot_split and all_sync_split
-- below). They were promoted to public in v0.2.0 so warp-types-divtree and
-- warp-types-ballot can import them via Lake path dependency rather than
-- copy-pasting their bodies. The bodies themselves are unchanged.

theorem foldl_or_lift (xs : List (BitVec n)) (acc : BitVec n) :
    List.foldl (· ||| ·) acc xs = acc ||| List.foldl (· ||| ·) 0 xs := by
  induction xs generalizing acc with
  | nil =>
    simp only [List.foldl_nil]
    ext i hi
    simp
  | cons b bs ih =>
    simp only [List.foldl_cons]
    rw [ih (acc ||| b), ih (0 ||| b)]
    ext i hi
    simp [BitVec.getElem_or, Bool.or_assoc]

theorem foldl_and_lift (xs : List (BitVec n)) (acc : BitVec n) :
    List.foldl (· &&& ·) acc xs = acc &&& List.foldl (· &&& ·) (BitVec.allOnes n) xs := by
  induction xs generalizing acc with
  | nil =>
    simp only [List.foldl_nil]
    ext i hi
    simp
  | cons b bs ih =>
    simp only [List.foldl_cons]
    rw [ih (acc &&& b), ih (BitVec.allOnes n &&& b)]
    ext i hi
    simp [BitVec.getElem_and, Bool.and_assoc]

-- ============================================================================
-- Mask safety: subset and containment
-- ============================================================================

/-- Masking a child mask by the parent produces a disjoint pair:
    `(child & parent) & (child & ~parent) == 0`. -/
theorem mask_produces_subset (child parent : BitVec n) :
    (child &&& parent) &&& (child &&& ~~~parent) = 0#n := by
  ext i hi
  simp [BitVec.getElem_and, BitVec.getElem_not, BitVec.getElem_zero]
  cases parent[i]'hi <;> simp

/-- A child mask AND-ed with parent cannot have bits outside the parent:
    `(child & parent) & ~parent == 0`. -/
theorem child_within_parent (child parent : BitVec n) :
    (child &&& parent) &&& ~~~parent = 0#n := by
  ext i hi
  simp [BitVec.getElem_and, BitVec.getElem_not, BitVec.getElem_zero]

/-- Passing a submask to `__syncwarp` is safe when the submask is a subset
    of the active mask: if `submask & ~activemask == 0`, then
    `submask & activemask == submask`. -/
theorem syncwarp_safe {submask activemask : BitVec n}
    (h_subset : submask &&& ~~~activemask = 0#n) :
    submask &&& activemask = submask := by
  ext i hi
  have h_bit : submask[i]'hi = true → activemask[i]'hi = true := by
    have := congrArg (·[i]'hi) h_subset
    simpa [BitVec.getElem_and, BitVec.getElem_not, BitVec.getElem_zero] using this
  simp [BitVec.getElem_and]
  cases hs : submask[i]'hi
  · simp
  · simp [h_bit hs]

-- ============================================================================
-- Vote operations: fold algebra over lane lists
-- ============================================================================
-- These three may migrate to warp-types-ballot when that sibling exists.

/-- `__ballot_sync` composes over list concatenation: the OR-fold of an
    appended list equals the OR of its halves. -/
theorem ballot_split (left_lanes right_lanes : List (BitVec n)) :
    List.foldl (· ||| ·) 0 (left_lanes ++ right_lanes) =
    List.foldl (· ||| ·) 0 left_lanes |||
    List.foldl (· ||| ·) 0 right_lanes := by
  rw [List.foldl_append, foldl_or_lift right_lanes (List.foldl (· ||| ·) 0 left_lanes)]

/-- `__all_sync` composes over list concatenation: the AND-fold starting
    from all-ones over an appended list equals the AND of its halves. -/
theorem all_sync_split (left_lanes right_lanes : List (BitVec n)) :
    List.foldl (· &&& ·) (BitVec.allOnes n) (left_lanes ++ right_lanes) =
    List.foldl (· &&& ·) (BitVec.allOnes n) left_lanes &&&
    List.foldl (· &&& ·) (BitVec.allOnes n) right_lanes := by
  rw [List.foldl_append,
      foldl_and_lift right_lanes (List.foldl (· &&& ·) (BitVec.allOnes n) left_lanes)]

/-- `__any_sync` is monotone: if the OR of two masks is non-zero, at least
    one of them is non-zero. -/
theorem any_sync_monotone {a b : BitVec n}
    (h : a ||| b ≠ 0#n) : a ≠ 0#n ∨ b ≠ 0#n := by
  by_cases ha : a = 0#n
  · right
    intro hb
    apply h
    rw [ha, hb]
    ext i hi
    simp [BitVec.getElem_zero]
  · left
    exact ha

end
end WarpTypesBitwise
