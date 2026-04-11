import WarpTypesBitwise.CUDA

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

## Dependency posture

v0.2.0 imports `warp-types-bitwise` via Lake path dependency. The
three theorems shared with `WarpTypesBitwise.CUDA` (`ballot_split`,
`all_sync_split`, `any_sync_monotone`) are re-exported under the
`WarpTypesBallot` namespace via `export` so consumers can continue
to refer to `WarpTypesBallot.ballot_split` etc. without breakage.
The two private fold-lift helpers (`foldl_or_lift`, `foldl_and_lift`)
that were inlined in v0.1.0 are deleted — bitwise's now-public
versions cover them. The four ballot-specific boundary-case theorems
(`ballot_nil`, `ballot_singleton`, `all_sync_nil`,
`all_sync_singleton`) stay local — they complete the API surface
that bitwise's CUDA module doesn't enumerate.
-/

namespace WarpTypesBallot

-- =========================================================================
-- 1. Re-exports from WarpTypesBitwise.CUDA
-- =========================================================================
-- Three theorems (`ballot_split`, `all_sync_split`, `any_sync_monotone`)
-- have byte-identical canonical versions in `warp-types-bitwise`. The
-- `export` declaration makes them resolvable as
-- `WarpTypesBallot.ballot_split` etc. so any consumer that imports
-- WarpTypesBallot can continue to use the names without qualifying
-- them — the redirection is invisible. The previously-private
-- `foldl_or_lift` / `foldl_and_lift` helpers are now provided by
-- bitwise (also public as of bitwise v0.2.0) and used internally by
-- the re-exported theorems.

export WarpTypesBitwise (ballot_split all_sync_split any_sync_monotone)

section
variable {n : Nat}

-- =========================================================================
-- 2. Ballot-specific boundary cases (OR-fold)
-- =========================================================================

/-- An empty ballot is zero: no lanes, no bits set. -/
theorem ballot_nil : List.foldl (· ||| ·) 0 ([] : List (BitVec n)) = 0 := rfl

/-- A single-lane ballot is the lane's mask: the OR-fold over `[x]`
    reduces to `0 ||| x`, which is `x`. -/
theorem ballot_singleton (x : BitVec n) :
    List.foldl (· ||| ·) 0 [x] = x := by
  show 0 ||| x = x
  exact BitVec.zero_or

-- =========================================================================
-- 3. All-sync boundary cases (AND-fold starting from allOnes)
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

end

end WarpTypesBallot
