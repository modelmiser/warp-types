import WarpTypesOverflow.Core

/-!
# WarpTypesOverflow

Rust arithmetic overflow-freedom lemmas for warp-types verification.

This module re-exports every public lemma from the `Core` submodule:

- `WarpTypesOverflow.Core` — five overflow-freedom theorems
  (`add_no_wrap`, `mul_no_wrap`, `sub_no_wrap`, `add_half_range`,
  `value_in_range`) covering the arithmetic obligations a Rust
  verification plugin emits for `#[sol_verify(overflow_free)]`-style
  annotations. Width-parametric, no Mathlib or Sol dependency.

All lemmas live under the `WarpTypesOverflow` namespace. For the
sub-namespace, import the corresponding module directly.
-/
