/-!
# WarpTypesInvariant.Core

State-machine induction combinators. Given a step function and a
predicate `P` that holds at the initial state and is preserved by one
step, these combinators conclude that `P` holds at every reachable
state. Two shapes are supported:

- **Autonomous** — `f : α → α`, conclude `∀ n, P (iterate f n s₀)`.
- **Input-driven** — `step : S → I → S`, conclude
  `∀ inputs : List I, P (inputs.foldl step s₀)`.

Two derived corollaries cover common special cases:

- **`iterate_fixpoint`** — if `f s₀ = s₀`, then
  `∀ n, iterate f n s₀ = s₀`. Useful for reset-state fixpoints in
  pipelined hardware proofs.
- **`foldl_constant`** — if the step function preserves a
  trajectory-independent predicate, any trace from an initial state
  satisfying `P` still satisfies `P`. This is the shape used by
  typing-witnessed invariants like "the mask remains a valid n-bit
  vector".

## Dependency posture

Everything here is Mathlib-free and Sol-free: Lean 4.28 core has neither
`Function.iterate` nor the `f^[n]` notation (both live in Mathlib's
`Logic.Function.Iterate`), so this module defines a local `iterate`
function and proves induction principles over it. `List.foldl` is in
core and is used directly for the input-driven combinator.

Consumers migrating from Sol's `Function.iterate`-style proofs should
replace `f^[n] s₀` with `WarpTypesInvariant.iterate f n s₀`. The two
definitions are provably equal (outer-cons vs inner-cons on the `succ`
case), but this crate's variant makes `iterate f (n+1) s₀ =
f (iterate f n s₀)` hold *definitionally*, which shortens induction
proofs by one rewrite step.
-/

namespace WarpTypesInvariant

-- =========================================================================
-- 1. Local iterate (Mathlib-free)
-- =========================================================================

/-- `iterate f n x` applies `f` to `x` exactly `n` times. Defined locally
    because Lean 4.28 core does not export `Function.iterate`. The
    recursion order `iterate f (n+1) x = f (iterate f n x)` makes the
    step lemma definitional. -/
def iterate {α : Sort u} (f : α → α) : Nat → α → α
  | 0, a => a
  | n + 1, a => f (iterate f n a)

/-- `iterate f 0 x = x` — definitional. -/
@[simp] theorem iterate_zero {α : Sort u} (f : α → α) (x : α) :
    iterate f 0 x = x := rfl

/-- `iterate f (n+1) x = f (iterate f n x)` — definitional. -/
@[simp] theorem iterate_succ {α : Sort u} (f : α → α) (n : Nat) (x : α) :
    iterate f (n + 1) x = f (iterate f n x) := rfl

-- =========================================================================
-- 2. Foundational combinators
-- =========================================================================

/-- Invariant induction for autonomous step functions.

    Given `P s₀` and `∀ s, P s → P (f s)`, conclude
    `∀ n, P (iterate f n s₀)`.

    Proof: induction on `n`. The `zero` case is `base` unchanged; the
    `succ` case uses `step` on the IH, relying on the definitional
    unfolding of `iterate`. -/
theorem iterate_invariant {α : Sort u} (f : α → α) (s₀ : α) (P : α → Prop)
    (base : P s₀) (step : ∀ s, P s → P (f s)) :
    ∀ n, P (iterate f n s₀) := by
  intro n
  induction n with
  | zero => exact base
  | succ n ih => exact step _ ih

/-- Invariant induction for input-driven step functions.

    Given `P s₀` and `∀ s i, P s → P (step s i)`, conclude
    `∀ inputs : List I, P (inputs.foldl step s₀)`.

    Proof: strengthen the goal to quantify over the starting state, then
    induct on the input list. The cons case unfolds `List.foldl` one
    step and feeds the result into the IH. -/
theorem foldl_invariant {S : Type u} {I : Type v}
    (step : S → I → S) (s₀ : S) (P : S → Prop)
    (base : P s₀) (consec : ∀ s i, P s → P (step s i)) :
    ∀ inputs : List I, P (inputs.foldl step s₀) := by
  suffices h : ∀ (inputs : List I) (s : S), P s → P (inputs.foldl step s) from
    fun inputs => h inputs s₀ base
  intro inputs
  induction inputs with
  | nil => intro s hs; simpa [List.foldl] using hs
  | cons i rest ih =>
    intro s hs
    show P (rest.foldl step (step s i))
    exact ih (step s i) (consec s i hs)

-- =========================================================================
-- 3. Derived corollaries
-- =========================================================================

/-- Fixpoint iteration: if `f` fixes `s₀`, then iterating `f` any number
    of times leaves `s₀` unchanged. Degenerate case of
    `iterate_invariant` with `P s := s = s₀`. -/
theorem iterate_fixpoint {α : Sort u} {f : α → α} {s₀ : α} (h : f s₀ = s₀) :
    ∀ n, iterate f n s₀ = s₀ := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
    show f (iterate f n s₀) = s₀
    rw [ih]; exact h

/-- Trajectory-independent invariant: if every state satisfies `P`
    unconditionally (e.g. a typing-witnessed property like "is a valid
    n-bit vector"), then any trace from an initial `P`-state stays in
    `P`. Specialization of `foldl_invariant` with a consecution
    hypothesis that ignores the previous state. -/
theorem foldl_constant {S : Type u} {I : Type v}
    (step : S → I → S) (s₀ : S) (P : S → Prop)
    (base : P s₀) (universal : ∀ s i, P (step s i)) :
    ∀ inputs : List I, P (inputs.foldl step s₀) :=
  foldl_invariant step s₀ P base (fun s i _ => universal s i)

-- =========================================================================
-- 4. Validation — exercise combinators against a toy state machine
-- =========================================================================

-- A tiny counter state machine: state is a natural, step adds an input.
-- No hardware dependency, no BitVec, no Sol. The purpose is to force
-- type-checking of both combinators against concrete arguments.

private def counterStep (s : Nat) (i : Nat) : Nat := s + i
private def counterTick (s : Nat) : Nat := s + 1

/-- Autonomous counter is monotone: after `n` ticks from 0, state is ≥ 0.
    Trivial via `Nat.zero_le`, but exercises `iterate_invariant` against
    a concrete step function and a partially-applied predicate
    (`LE.le 0`) — `apply` unifies cleanly when `P` is not a bare lambda.
    See INSIGHTS for the higher-order unification pitfall. -/
private theorem counter_tick_nonneg :
    ∀ n, 0 ≤ iterate counterTick n 0 := by
  apply iterate_invariant
  · exact Nat.zero_le _
  · intro s _; exact Nat.zero_le _

/-- Reset fixpoint: the identity function iterated from 0 stays at 0.
    Exercises `iterate_fixpoint` on a concrete function. -/
private theorem counter_reset_fixpoint :
    ∀ n, iterate (fun s : Nat => s) n 0 = 0 :=
  iterate_fixpoint rfl

/-- Input-driven counter: sum of inputs from 0 is ≥ 0. Exercises
    `foldl_invariant` against `counterStep`. -/
private theorem counter_foldl_nonneg :
    ∀ inputs : List Nat, 0 ≤ inputs.foldl counterStep 0 := by
  apply foldl_invariant
  · exact Nat.zero_le _
  · intro _ _ _; exact Nat.zero_le _

/-- Trajectory-independent variant: exercises `foldl_constant`. Uses
    `0 ≤ _` (non-lambda) to avoid higher-order unification issues. -/
private theorem counter_foldl_constant_nonneg :
    ∀ inputs : List Nat, 0 ≤ inputs.foldl counterStep 0 := by
  apply foldl_constant
  · exact Nat.zero_le _
  · intro _ _; exact Nat.zero_le _

end WarpTypesInvariant
