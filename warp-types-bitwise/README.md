# warp-types-bitwise

Bitvector tautology and mask-algebra lemmas for warp-types verification, packaged as a standalone Lean 4 library.

## What this is

A lemma library for the 14 bitwise-tautology obligations that the warp-types project's verification workflow routinely needs. Originally extracted from Sol's obligation receiver files (`active/sol/lean/Sol/Rust.lean`, `CUDA.lean`, `Verilog.lean`) after the 2026-04-11 obligation inventory identified bitwise mask algebra as the largest unhandled obligation bin (14 theorems, 20% of Sol's current 70-obligation backlog).

## What this is not

- Not a solver. The lemmas are closed at compile time by Lean 4's built-in `bv_decide` and per-bit extensionality; there is no runtime SAT or SMT engine inside this crate.
- Not a tactic library. For the `sol_bv_decide` and `sol_bv_hyp_decide` tactics and their friends, see the [Sol](https://github.com/modelmiser/sol) project (if you need hypothesis-aware decision procedures on top of the underlying `bv_decide`).
- Not a discharger for runtime-generated obligations. The lemmas here are named, fixed theorems — you prove your own new goals by citing them or by using Lean's tactics directly.

## Lemmas

Fourteen lemmas across three shapes:

| Shape | Count | Examples |
|---|---|---|
| Rust mask-algebra | 5 | `mask_idempotent`, `disjoint_masks`, `field_insert_read`, `counter_mask_valid`, `disjoint_update` |
| CUDA warp-mask | 6 | `ballot_split`, `all_sync_split`, `any_sync_monotone`, `mask_produces_subset`, `child_within_parent`, `syncwarp_safe` |
| Verilog else-path | 3 | `else_complement`, `else_disjoint_from_taken`, `rtl_else_xor` |

### About the fold lemmas

`ballot_split`, `all_sync_split`, and `any_sync_monotone` are fold-algebra theorems (list homomorphisms over the bitvector OR/AND monoids). They are included in this crate because they arose in the same Sol obligation bin, but they may relocate to a separate `warp-types-ballot` crate when that sibling is built. If you use these three, expect a future re-import path.

## Using from a Lean project

Add a Lake dependency in your `lakefile.toml`:

```toml
[[require]]
name = "WarpTypesBitwise"
path = "path/to/warp-types/warp-types-bitwise/lean"
```

Or, once published:

```toml
[[require]]
name = "WarpTypesBitwise"
git = "https://github.com/modelmiser/warp-types"
subDir = "warp-types-bitwise/lean"
```

Then import in your `.lean` files:

```lean
import WarpTypesBitwise

example (x mask : BitVec 32) : (x &&& mask) &&& mask = x &&& mask :=
  WarpTypesBitwise.mask_idempotent x mask
```

## Building

```bash
cd lean
lake build
```

Requires Lean 4.28.0 (pinned via `lean-toolchain`). No Mathlib dependency; uses only Lean core's `BitVec` API and `bv_decide` tactic.

## Relationship to the warp-types workspace

`warp-types-bitwise` is a Cargo workspace member of the root `warp-types` crate at version 0.1.0. The Rust `lib.rs` is a minimal marker; the actual library is the Lean project in `lean/`. This layout keeps versioning and release cadence consistent across the warp-types sibling crate family (`warp-types-sat`, `warp-types-bitwise`, and forthcoming `warp-types-invariant`, `warp-types-overflow`, `warp-types-divtree`, `warp-types-ballot`).

## License

MIT. See the repository root for the full license text.
