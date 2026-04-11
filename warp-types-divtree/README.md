# warp-types-divtree

Nested divergence partition tree for warp-types verification, packaged as a standalone Lean 4 library.

## What this is

A formalization of the *diverge tree* — a binary tree whose internal nodes split a parent bitmask into two complementary halves via a predicate — together with four soundness theorems proving that a well-formed tree's leaves form a disjoint cover of its root. Originally extracted from Sol's `active/sol/lean/Sol/DivTree.lean` after the 2026-04-11 obligation inventory identified nested divergence as the fourth bin to migrate (after bitwise, invariant, and overflow) in the warp-types sibling crate family.

## Why it matters

warp-types verifies warp divergence at a single level via the `ActiveSet` typestate. A full hardware pipeline may nest diverge operations several levels deep (a divergent branch inside a divergent branch inside a loop, etc.). The diverge tree captures arbitrarily-deep nested divergence as a single structure, and the soundness theorems prove that no matter how a kernel branches, merging all the leaf warps (via OR) recovers the original state and the leaves never overlap. This is the proof obligation that gates Phase V hardware verification for warp-core, warp-pipe, warp-display, warp-cdc, and warp-mesh — all the ULX3S RTL variants inherit from the same soundness contract.

## Theorems

| Theorem | Signature | Says |
|---|---|---|
| `leaves_cover_root` | `WellFormed t → foldl (· ||| ·) 0 t.leaves = t.root` | OR-folding all leaves recovers the root mask |
| `leaves_pairwise_disjoint` | `WellFormed t → List.Pairwise (· &&& · = 0) t.leaves` | All leaf masks are pairwise disjoint |
| `leaf_subset_root` | `WellFormed t → m ∈ t.leaves → m &&& t.root = m` | Every leaf is an AND-subset of the root |
| `leaves_length` | `t.leaves.length = t.nodeCount + 1` | A tree with `k` internal nodes has `k + 1` leaves |

All four are width-parametric in `n : Nat`. `leaves_length` does not require `WellFormed` — it's a pure structural invariant.

Plus the public surface:

- `DivTree n` — the inductive type (`leaf` + `node`)
- `DivTree.root`, `DivTree.leaves`, `DivTree.nodeCount` — accessors
- `DivTree.WellFormed` — the predicate stating each node's children are the predicate-split of the parent

## What this is not

- Not a decision procedure. The theorems are closed by `ext` + `simp` + manual induction over `WellFormed`. There is no runtime SAT or SMT.
- Not a substitute for `warp-types-bitwise`. The diverge tree uses five private helper lemmas (`and_complement_cover`, `foldl_or_singleton`, `foldl_or_lift`, `foldl_or_append`, `subset_of_subset_and`, `cross_disjoint`) that overlap with bitwise's lemma set. A future family-wide refactor will eliminate this duplication via a Lake path dependency; v0.1.0 keeps each sibling crate standalone per the sibling-plan discipline.
- Not a hardware model. `DivTree` captures the *algebraic shape* of nested divergence. The mapping to a specific RTL divergence stack (warp-core's `DivState`, etc.) is a separate step that lives in each RTL variant's verification crate, not here.

## Using from a Lean project

Add a Lake dependency in your `lakefile.toml`:

```toml
[[require]]
name = "WarpTypesDivtree"
path = "path/to/warp-types/warp-types-divtree/lean"
```

Or, once published:

```toml
[[require]]
name = "WarpTypesDivtree"
git = "https://github.com/modelmiser/warp-types"
subDir = "warp-types-divtree/lean"
```

Then import in your `.lean` files:

```lean
import WarpTypesDivtree

open WarpTypesDivtree

-- Build a tiny tree and prove its leaves cover the root
example (parent pred : BitVec 32)
    (t : DivTree 32)
    (hwf : DivTree.WellFormed t) :
    List.foldl (· ||| ·) 0 t.leaves = t.root :=
  DivTree.leaves_cover_root hwf

-- Capacity bound for downstream RTL: 7 nodes ⇒ 8 leaves
example (t : DivTree 32) (h : t.nodeCount = 7) :
    t.leaves.length = 8 := by
  rw [DivTree.leaves_length, h]
```

## Building

```bash
cd lean
lake build
```

Requires Lean 4.28.0 (pinned via `lean-toolchain`). No Mathlib dependency; uses only Lean core's `BitVec` / `List` APIs and `ext` + `simp` proofs of the underlying bitwise identities.

## Relationship to the warp-types workspace

`warp-types-divtree` is a Cargo workspace member of the root `warp-types` crate at version 0.1.0. The Rust `lib.rs` is a minimal marker; the actual library is the Lean project in `lean/`. This layout keeps versioning and release cadence consistent across the warp-types sibling crate family (`warp-types-sat`, `warp-types-bitwise`, `warp-types-invariant`, `warp-types-overflow`, `warp-types-divtree`, and forthcoming `warp-types-ballot`).

## License

MIT. See the repository root for the full license text.
