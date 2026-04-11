/-!
# WarpTypesBallot.Core

GPU warp-vote fold-algebra lemmas. The three CUDA warp-vote intrinsics
(`__ballot_sync`, `__all_sync`, `__any_sync`) are modeled as fold
algebra over lists of per-lane bitvectors:

- **`__ballot_sync(mask, pred)`** — OR-fold: each lane contributes its
  per-lane mask bit, and the result is the OR-fold over the list of
  per-lane contributions.
- **`__all_sync(mask, pred)`** — AND-fold starting from
  `BitVec.allOnes n`: each lane contributes its per-lane value and the
  result is the AND across all lanes (allOnes is the identity for
  AND, giving "no lanes ⇒ all predicate true" semantics).
- **`__any_sync(mask, pred)`** — monotone existence: if the combined
  ballot is non-zero, at least one input half was non-zero.

Seven theorems total:

| Theorem | Shape |
|---|---|
| `ballot_nil` | `foldl (· ||| ·) 0 [] = 0` |
| `ballot_singleton` | `foldl (· ||| ·) 0 [x] = x` |
| `ballot_split` | `foldl (· ||| ·) 0 (xs ++ ys) = foldl (…) xs ||| foldl (…) ys` |
| `all_sync_nil` | `foldl (· &&& ·) (allOnes n) [] = allOnes n` |
| `all_sync_singleton` | `foldl (· &&& ·) (allOnes n) [x] = x` |
| `all_sync_split` | `foldl (· &&& ·) (allOnes n) (xs ++ ys) = foldl (…) xs &&& foldl (…) ys` |
| `any_sync_monotone` | `a ||| b ≠ 0 → a ≠ 0 ∨ b ≠ 0` |

## Dependency posture

Mathlib-free and Sol-free. Uses only Lean 4.28 core's `BitVec` /
`List` APIs plus `ext` + `simp` proofs. Sol's original proof of
`any_sync_monotone` used `push_neg` (which lives in
`Mathlib.Logic.Basic`); this crate uses the `by_cases` + explicit
rewriting pattern from `warp-types-bitwise` (pattern 3 from
`feedback_lean4_bv_proofs.md`).

## Duplication note

Three of these theorems (`ballot_split`, `all_sync_split`,
`any_sync_monotone`) are structurally identical to lemmas already in
`warp-types-bitwise`'s `WarpTypesBitwise.CUDA` module. Likewise, the
two private fold-lift helpers (`foldl_or_lift`, `foldl_and_lift`) are
byte-identical to bitwise's private helpers. v0.1.0 duplicates them
here per the sibling-plan's "each crate standalone" rule; a family-wide
refactor scheduled after ballot lands will hoist the shared helpers
into a common module so divtree, ballot, and bitwise all import from
the same source.
-/

namespace WarpTypesBallot

section
variable {n : Nat}

-- =========================================================================
-- 1. Private fold-lift helpers
-- =========================================================================
-- Byte-identical to warp-types-bitwise.foldl_{or,and}_lift.
-- Tracked for post-ballot family-wide refactor.

/-- Accumulator-lift for OR-fold: folding into an arbitrary accumulator
    equals folding into zero, then OR-ing the accumulator. -/
private theorem foldl_or_lift (xs : List (BitVec n)) (acc : BitVec n) :
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

/-- Accumulator-lift for AND-fold: folding into an arbitrary accumulator
    equals folding into `allOnes`, then AND-ing the accumulator. -/
private theorem foldl_and_lift (xs : List (BitVec n)) (acc : BitVec n) :
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

-- =========================================================================
-- 2. Ballot (OR-fold)
-- =========================================================================

/-- An empty ballot is zero: no lanes, no bits set. -/
theorem ballot_nil : List.foldl (· ||| ·) 0 ([] : List (BitVec n)) = 0 := rfl

/-- A single-lane ballot is the lane's mask: the OR-fold over `[x]`
    reduces to `0 ||| x`, which is `x`. -/
theorem ballot_singleton (x : BitVec n) :
    List.foldl (· ||| ·) 0 [x] = x := by
  show 0 ||| x = x
  exact BitVec.zero_or

/-- `__ballot_sync` composes over list concatenation: the OR-fold of an
    appended list equals the OR of its halves. Structurally identical
    to `WarpTypesBitwise.ballot_split`. -/
theorem ballot_split (left_lanes right_lanes : List (BitVec n)) :
    List.foldl (· ||| ·) 0 (left_lanes ++ right_lanes) =
    List.foldl (· ||| ·) 0 left_lanes |||
    List.foldl (· ||| ·) 0 right_lanes := by
  rw [List.foldl_append, foldl_or_lift right_lanes (List.foldl (· ||| ·) 0 left_lanes)]

-- =========================================================================
-- 3. All-sync (AND-fold starting from allOnes)
-- =========================================================================

/-- An empty all-sync is `allOnes`: vacuously, all zero lanes satisfy
    any predicate. This gives `all_sync` its identity element for
    AND-fold composition. -/
theorem all_sync_nil :
    List.foldl (· &&& ·) (BitVec.allOnes n) ([] : List (BitVec n)) = BitVec.allOnes n := rfl

/-- A single-lane all-sync is the lane's mask: the AND-fold over `[x]`
    starting from `allOnes` reduces to `allOnes &&& x`, which is `x`. -/
theorem all_sync_singleton (x : BitVec n) :
    List.foldl (· &&& ·) (BitVec.allOnes n) [x] = x := by
  show BitVec.allOnes n &&& x = x
  exact BitVec.allOnes_and

/-- `__all_sync` composes over list concatenation: the AND-fold starting
    from `allOnes` over an appended list equals the AND of its halves.
    Structurally identical to `WarpTypesBitwise.all_sync_split`. -/
theorem all_sync_split (left_lanes right_lanes : List (BitVec n)) :
    List.foldl (· &&& ·) (BitVec.allOnes n) (left_lanes ++ right_lanes) =
    List.foldl (· &&& ·) (BitVec.allOnes n) left_lanes &&&
    List.foldl (· &&& ·) (BitVec.allOnes n) right_lanes := by
  rw [List.foldl_append,
      foldl_and_lift right_lanes (List.foldl (· &&& ·) (BitVec.allOnes n) left_lanes)]

-- =========================================================================
-- 4. Any-sync (monotone existence)
-- =========================================================================

/-- `__any_sync` is monotone: if the OR of two masks is non-zero, at
    least one of them is non-zero. Structurally identical to
    `WarpTypesBitwise.any_sync_monotone`; Sol's original version used
    `push_neg` (Mathlib), this version uses the core `by_cases` +
    explicit rewriting pattern. -/
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

end WarpTypesBallot
