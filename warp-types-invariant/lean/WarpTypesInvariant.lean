import WarpTypesInvariant.Core

/-!
# WarpTypesInvariant

State-machine induction combinators for warp-types verification.

This module re-exports every public combinator from the `Core`
submodule:

- `WarpTypesInvariant.Core` — two foundational combinators
  (`iterate_invariant`, `foldl_invariant`) and two derived corollaries
  (`iterate_fixpoint`, `foldl_constant`) for state-machine invariant
  proofs. Width-parametric, hardware-agnostic, no Mathlib or Sol
  dependency.

All lemmas live under the `WarpTypesInvariant` namespace. For the
sub-namespace, import the corresponding module directly.
-/
