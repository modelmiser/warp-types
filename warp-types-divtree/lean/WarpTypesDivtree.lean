import WarpTypesDivtree.Core

/-!
# WarpTypesDivtree

Nested divergence partition tree for warp-types verification.

This module re-exports every public declaration from the `Core`
submodule:

- `WarpTypesDivtree.Core` — the `DivTree` inductive type, the
  `WellFormed` predicate, the `root` / `leaves` / `nodeCount`
  accessors, and four soundness theorems (`leaves_cover_root`,
  `leaves_pairwise_disjoint`, `leaf_subset_root`, `leaves_length`).
  Width-parametric, no Mathlib or Sol dependency.

All declarations live under the `WarpTypesDivtree.DivTree` namespace.
For the sub-namespace, import the corresponding module directly.
-/
