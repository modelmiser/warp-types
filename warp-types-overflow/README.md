# warp-types-overflow

Rust arithmetic overflow-freedom lemmas for warp-types verification, packaged as a standalone Lean 4 library.

## What this is

A small library of five lemmas covering the *overflow-freedom* obligation class that a Rust verification plugin emits when checking annotations like `#[sol_verify(overflow_free)] fn transfer(...)`. Originally extracted from Sol's `active/sol/lean/Sol/Rust.lean` after the 2026-04-11 obligation inventory identified arithmetic no-wrap as the third bin to migrate (after bitwise and invariant) in the warp-types sibling crate family.

| Lemma | Precondition | Conclusion |
|---|---|---|
| `add_no_wrap` | `a.toNat + b.toNat < 2^n` | `(a + b).toNat = a.toNat + b.toNat` |
| `mul_no_wrap` | `a.toNat * b.toNat < 2^n` | `(a * b).toNat = a.toNat * b.toNat` |
| `sub_no_wrap` | `b.toNat ≤ a.toNat` | `(a - b).toNat = a.toNat - b.toNat` |
| `add_half_range` | `a, b < 2^(n-1)` and `0 < n` | `(a + b).toNat = a.toNat + b.toNat` |
| `value_in_range` | — | `a.toNat < 2^n` |

All five are width-parametric (`∀ n : Nat`). `value_in_range` is a trivial wrapper over `BitVec.isLt`, included for call-site uniformity so a plugin can treat "result-fits-in-width" and "operand-in-width" obligations through a single lemma surface.

## What this is not

- Not a tactic. Sol's `sol_overflow` macro packages these three no-wrap rewrites as a `first` combinator, but it depends on Sol's internal namespace for `sub_no_wrap` dispatch. This crate exposes the raw lemmas; users call `WarpTypesOverflow.add_no_wrap h` directly or write their own tactic wrapper.
- Not a substitute for `omega`. Bounds obligations (`i < arr.length`, `base + offset < len`, etc.) are discharged by Lean core's `omega` without any lemma library — there is no `warp-types-bounds` crate because `omega` already fully covers that territory.
- Not a discharger for combined overflow + bitwise obligations. Sol's `checked_add_properties` (from `Rust.lean:206`) spans both this crate and `warp-types-bitwise`. Consumers combine the two by hand — one `add_no_wrap` call for the Nat-arithmetic side, one `sol_bv_decide` (or a bitwise lemma from `warp-types-bitwise`) for the decomposition side.

## The Mathlib-boundary footnote

`add_half_range` uses the identity `2^(n-1) + 2^(n-1) = 2^n` to reduce to `add_no_wrap`. Sol's version of this lemma used Mathlib's `pow_succ`; this crate uses Lean core's `Nat.pow_succ` instead. Same content, different namespace. If you are migrating a Sol-shaped proof that uses `pow_succ`, rewrite to `Nat.pow_succ` — or better, let this lemma absorb the rewrite so the migration is a single identifier swap.

## Using from a Lean project

Add a Lake dependency in your `lakefile.toml`:

```toml
[[require]]
name = "WarpTypesOverflow"
path = "path/to/warp-types/warp-types-overflow/lean"
```

Or, once published:

```toml
[[require]]
name = "WarpTypesOverflow"
git = "https://github.com/modelmiser/warp-types"
subDir = "warp-types-overflow/lean"
```

Then import in your `.lean` files:

```lean
import WarpTypesOverflow

-- Rust checked_add shape
example (a b : BitVec 32) (h : a.toNat + b.toNat < 2 ^ 32) :
    (a + b).toNat = a.toNat + b.toNat :=
  WarpTypesOverflow.add_no_wrap h

-- Rust checked_sub shape
example (a b : BitVec 64) (h : b.toNat ≤ a.toNat) :
    (a - b).toNat = a.toNat - b.toNat :=
  WarpTypesOverflow.sub_no_wrap h

-- Two u32 halves sum safely
example (a b : BitVec 32)
    (ha : a.toNat < 2 ^ 31) (hb : b.toNat < 2 ^ 31) :
    (a + b).toNat = a.toNat + b.toNat :=
  WarpTypesOverflow.add_half_range (by omega) ha hb
```

## Building

```bash
cd lean
lake build
```

Requires Lean 4.28.0 (pinned via `lean-toolchain`). No Mathlib dependency; uses only Lean core's `BitVec` API, `Nat.mod_eq_of_lt`, `Nat.add_mod_right`, `Nat.pow_succ`, and `omega`.

## Relationship to the warp-types workspace

`warp-types-overflow` is a Cargo workspace member of the root `warp-types` crate at version 0.1.0. The Rust `lib.rs` is a minimal marker; the actual library is the Lean project in `lean/`. This layout keeps versioning and release cadence consistent across the warp-types sibling crate family (`warp-types-sat`, `warp-types-bitwise`, `warp-types-invariant`, `warp-types-overflow`, and forthcoming `warp-types-divtree`, `warp-types-ballot`).

## License

MIT. See the repository root for the full license text.
