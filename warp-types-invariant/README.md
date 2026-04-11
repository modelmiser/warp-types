# warp-types-invariant

State-machine induction combinators for warp-types verification, packaged as a standalone Lean 4 library.

## What this is

A tiny combinator library for proving `∀ n, P (step^n s₀)`-shaped state-machine invariants. Originally extracted from Sol's `active/sol/lean/Sol/Invariant.lean` after the 2026-04-11 obligation inventory identified state-machine induction as the second obligation bin to migrate (after bitwise mask algebra) in the warp-types sibling crate family.

Two shapes are supported:

| Shape | Combinator | Goal form |
|---|---|---|
| Autonomous | `iterate_invariant` | `∀ n, P (iterate f n s₀)` |
| Input-driven | `foldl_invariant` | `∀ inputs : List I, P (inputs.foldl step s₀)` |

Plus two derived corollaries:

| Corollary | Use case |
|---|---|
| `iterate_fixpoint` | `f s₀ = s₀ → ∀ n, iterate f n s₀ = s₀` (reset-state fixpoints in pipelined RTL) |
| `foldl_constant` | Trajectory-independent predicates (typing-witnessed invariants) |

## What this is not

- Not a tactic library. Sol's `sol_invariant` elab tactic dispatches to `sol_auto` for subgoal closing, which lives in Sol. This crate exposes the raw combinators; users call `apply WarpTypesInvariant.iterate_invariant` and close subgoals with whatever tactics they have available.
- Not a substitute for Mathlib's `Function.iterate`. This crate defines its own local `iterate` function because Lean 4.28 core exports neither `Function.iterate` nor the `f^[n]` notation — both live in `Mathlib.Logic.Function.Iterate`. The two definitions differ in recursion order (outer-cons vs inner-cons on the `succ` case) but are provably equal; downstream migration requires replacing `f^[n] s₀` with `WarpTypesInvariant.iterate f n s₀`.
- Not a discharger for runtime-generated obligations. The combinators are fixed theorems; consumers write their own invariant definitions and cite these combinators in the proof.

## The higher-order unification pitfall

`apply iterate_invariant` uses first-order unification and will fail to synthesize `?P` when the goal's predicate is a bare lambda:

```lean
-- ❌ `apply` cannot unify `?P (iterate counterTick n 0)` against
--    `(fun _ : Nat => True) (iterate counterTick n 0)`
example : ∀ n, (fun _ : Nat => True) (iterate counterTick n 0) := by
  apply iterate_invariant  -- FAILS: higher-order unification
  ...
```

Two workarounds:

1. **Use a named definition for `P`** — `def P : Nat → Prop := fun _ => True`, then the goal is `∀ n, P (iterate ...)` and unification succeeds.
2. **Use a partially-applied predicate that is not a beta-redex** — e.g., `0 ≤ _` (which elaborates to `LE.le 0 _`, a partial application, not a lambda).

The validation tests inside `Core.lean` take approach (2). Sol's `Invariant.lean` doc comment warns about the same issue for `sol_invariant`.

## Using from a Lean project

Add a Lake dependency in your `lakefile.toml`:

```toml
[[require]]
name = "WarpTypesInvariant"
path = "path/to/warp-types/warp-types-invariant/lean"
```

Or, once published:

```toml
[[require]]
name = "WarpTypesInvariant"
git = "https://github.com/modelmiser/warp-types"
subDir = "warp-types-invariant/lean"
```

Then import in your `.lean` files:

```lean
import WarpTypesInvariant

open WarpTypesInvariant

-- Autonomous: a counter that only increments stays ≥ 0
example (tick : Nat → Nat) (h : ∀ s, s ≤ tick s) :
    ∀ n, 0 ≤ iterate tick n 0 := by
  apply iterate_invariant
  · exact Nat.zero_le _
  · intro s _; exact Nat.zero_le _

-- Input-driven: a running sum from 0 stays ≥ 0
example : ∀ xs : List Nat, 0 ≤ xs.foldl (· + ·) 0 := by
  apply foldl_invariant
  · exact Nat.zero_le _
  · intro _ _ _; exact Nat.zero_le _
```

## Building

```bash
cd lean
lake build
```

Requires Lean 4.28.0 (pinned via `lean-toolchain`). No Mathlib dependency; uses only Lean core's `List.foldl` and a locally-defined `iterate`.

## Relationship to the warp-types workspace

`warp-types-invariant` is a Cargo workspace member of the root `warp-types` crate at version 0.1.0. The Rust `lib.rs` is a minimal marker; the actual library is the Lean project in `lean/`. This layout keeps versioning and release cadence consistent across the warp-types sibling crate family (`warp-types-sat`, `warp-types-bitwise`, `warp-types-invariant`, and forthcoming `warp-types-overflow`, `warp-types-divtree`, `warp-types-ballot`).

## License

MIT. See the repository root for the full license text.
