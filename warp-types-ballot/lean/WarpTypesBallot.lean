import WarpTypesBallot.Core

/-!
# WarpTypesBallot

GPU warp-vote fold-algebra lemmas for warp-types verification.

This module re-exports every public lemma from the `Core` submodule:

- `WarpTypesBallot.Core` — seven lemmas modeling `__ballot_sync`
  (OR-fold), `__all_sync` (AND-fold starting from `BitVec.allOnes`),
  and `__any_sync` (monotone existence). Three split lemmas, two
  nil-case dispatchers, two singleton-case dispatchers, plus
  `any_sync_monotone`. Width-parametric, no Mathlib or Sol dependency.

All lemmas live under the `WarpTypesBallot` namespace. For the
sub-namespace, import the corresponding module directly.
-/
