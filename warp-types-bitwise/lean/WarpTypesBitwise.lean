import WarpTypesBitwise.Rust
import WarpTypesBitwise.CUDA
import WarpTypesBitwise.Verilog

/-!
# WarpTypesBitwise

Bitvector tautology and mask-algebra lemmas for warp-types verification.

This module re-exports every public lemma from the three shape-specific
submodules:

- `WarpTypesBitwise.Rust` — 5 lemmas covering Rust's mask-algebra obligation
  shapes: `mask_idempotent`, `disjoint_masks`, `field_insert_read`,
  `counter_mask_valid`, `disjoint_update`.
- `WarpTypesBitwise.CUDA` — 6 lemmas covering CUDA's warp-mask obligation
  shapes: `ballot_split`, `all_sync_split`, `any_sync_monotone`,
  `mask_produces_subset`, `child_within_parent`, `syncwarp_safe`.
- `WarpTypesBitwise.Verilog` — 3 lemmas covering Verilog's else-path
  algebra obligation shapes: `else_complement`, `else_disjoint_from_taken`,
  `rtl_else_xor`.

All lemmas live under the `WarpTypesBitwise` namespace. For the
shape-specific sub-namespaces, import the corresponding module directly.
-/
