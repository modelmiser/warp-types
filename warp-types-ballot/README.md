# warp-types-ballot

GPU warp-vote fold-algebra lemmas for warp-types verification, packaged as a standalone Lean 4 library.

## What this is

A library of seven lemmas modeling the CUDA warp-vote intrinsics (`__ballot_sync`, `__all_sync`, `__any_sync`) as fold algebra over lists of per-lane bitvectors. The fifth and final Source B sibling crate in the warp-types family, landing the 2026-04-11 obligation migration triggered by Sol's `CUDA.lean:113-145` Vote section.

## Theorems

| Theorem | Shape | CUDA intrinsic |
|---|---|---|
| `ballot_nil` | `foldl (·\|\|\|·) 0 [] = 0` | empty ballot |
| `ballot_singleton` | `foldl (·\|\|\|·) 0 [x] = x` | single-lane ballot |
| `ballot_split` | `foldl (·\|\|\|·) 0 (xs ++ ys) = … \|\|\| …` | `__ballot_sync` composition |
| `all_sync_nil` | `foldl (·&&&·) allOnes [] = allOnes` | empty all-sync |
| `all_sync_singleton` | `foldl (·&&&·) allOnes [x] = x` | single-lane all-sync |
| `all_sync_split` | `foldl (·&&&·) allOnes (xs ++ ys) = … &&& …` | `__all_sync` composition |
| `any_sync_monotone` | `a \|\|\| b ≠ 0 → a ≠ 0 ∨ b ≠ 0` | `__any_sync` existence |

All seven are width-parametric in `n : Nat`. Specialize at `n = 32` for NVIDIA warps or `n = 64` for AMD wavefronts.

## What this is not

- Not a substitute for `warp-types-bitwise`. Three of the seven theorems (`ballot_split`, `all_sync_split`, `any_sync_monotone`) are structurally identical to lemmas already in `warp-types-bitwise`'s `CUDA.lean` module. The v0.1.0 release duplicates them per the sibling-plan's "each crate standalone" discipline; a family-wide refactor scheduled after all five Source B siblings land will hoist the shared helpers into a common module that divtree, ballot, and bitwise all import from.
- Not a decision procedure. The theorems are closed by `ext` + `simp` + list induction for the compositional ones, and by `by_cases` + explicit rewriting (pattern 3 from `feedback_lean4_bv_proofs.md`) for `any_sync_monotone`. No Mathlib `push_neg` or `tauto`.
- Not a simulation model. `ballot_split` (and its cousins) capture the *algebraic shape* of warp-vote composition — specifically that results compose over disjoint subsets of the warp. The mapping from CUDA's actual `__ballot_sync(mask, pred)` lane-by-lane evaluation to this fold representation is part of a consumer's modeling layer, not part of this crate.

## The "nil + singleton + split" completeness story

The four `*_nil` and `*_singleton` theorems are new in ballot (not in bitwise). They exist to let consumers case-analyze lists of per-lane masks without dropping to raw `List.foldl` reduction:

```lean
-- Case-analyze a list of lane contributions
match lanes with
| [] => WarpTypesBallot.ballot_nil         -- nil case
| [x] => WarpTypesBallot.ballot_singleton x -- singleton case
| _ :: _ :: _ =>
    -- General case: split the list and apply ballot_split
    ...
```

Without `ballot_nil` and `ballot_singleton`, the boundary cases would require either `rfl` (fragile under future simp normalization changes) or a manual `simp [List.foldl]` cascade. The four trivial completions are cheap (one or two lines each) and complete the API.

## Using from a Lean project

Add a Lake dependency in your `lakefile.toml`:

```toml
[[require]]
name = "WarpTypesBallot"
path = "path/to/warp-types/warp-types-ballot/lean"
```

Or, once published:

```toml
[[require]]
name = "WarpTypesBallot"
git = "https://github.com/modelmiser/warp-types"
subDir = "warp-types-ballot/lean"
```

Then import in your `.lean` files:

```lean
import WarpTypesBallot

-- __ballot_sync composition
example (L R : List (BitVec 32)) :
    List.foldl (· ||| ·) 0 (L ++ R) =
    List.foldl (· ||| ·) 0 L ||| List.foldl (· ||| ·) 0 R :=
  WarpTypesBallot.ballot_split L R

-- __all_sync empty case
example : List.foldl (· &&& ·) (BitVec.allOnes 32) ([] : List (BitVec 32)) = BitVec.allOnes 32 :=
  WarpTypesBallot.all_sync_nil

-- __any_sync existence
example (a b : BitVec 32) (h : a ||| b ≠ 0#32) : a ≠ 0#32 ∨ b ≠ 0#32 :=
  WarpTypesBallot.any_sync_monotone h
```

## Building

```bash
cd lean
lake build
```

Requires Lean 4.28.0 (pinned via `lean-toolchain`). No Mathlib dependency; uses only Lean core's `BitVec` / `List` APIs, `BitVec.allOnes_and`, `BitVec.zero_or`, and `ext` + `simp` proofs.

## Relationship to the warp-types workspace

`warp-types-ballot` is a Cargo workspace member of the root `warp-types` crate at version 0.1.0. The Rust `lib.rs` is a minimal marker; the actual library is the Lean project in `lean/`. This layout keeps versioning and release cadence consistent across the warp-types sibling crate family.

**As of 2026-04-11, ballot completes the Source B sibling family**: `warp-types-bitwise`, `warp-types-invariant`, `warp-types-overflow`, `warp-types-divtree`, `warp-types-ballot`. The next planned workflow step is a family-wide refactor pass to eliminate helper duplication between these five crates and then a Sol migration pass (post-paper-submission).

## License

MIT. See the repository root for the full license text.
