/-!
# WarpTypesOverflow.Core

Rust arithmetic overflow-freedom lemmas. Every theorem is
width-parametric in `n : Nat` and proves that a bitvector operation
commutes with `toNat` under a "result fits in n bits" precondition.

Five lemmas:

- **`add_no_wrap`** — addition doesn't wrap when the sum fits:
  `a.toNat + b.toNat < 2^n → (a + b).toNat = a.toNat + b.toNat`
- **`mul_no_wrap`** — multiplication doesn't wrap when the product fits:
  `a.toNat * b.toNat < 2^n → (a * b).toNat = a.toNat * b.toNat`
- **`sub_no_wrap`** — subtraction doesn't wrap when the left operand
  dominates: `b.toNat ≤ a.toNat → (a - b).toNat = a.toNat - b.toNat`
- **`add_half_range`** — corollary: two values each less than half the
  range cannot overflow when added
- **`value_in_range`** — every bitvector fits in its width (trivial
  wrapper over `BitVec.isLt`, included for call-site uniformity)

These are the obligations a Rust verification plugin would emit for
`#[sol_verify(overflow_free)] fn transfer(from, to, amount) { ... }`
style annotations, where the checker expects the plugin to bridge from
`BitVec` operations to `Nat` arithmetic via `toNat`.

## Dependency posture

Mathlib-free and Sol-free. Uses only Lean 4.28 core:
`BitVec.toNat_add`, `BitVec.toNat_mul`, `BitVec.toNat_sub`,
`BitVec.isLt`, `Nat.mod_eq_of_lt`, `Nat.add_mod_right`, `Nat.pow_succ`,
and `omega`. The one gotcha is that Mathlib's `pow_succ` is replaced by
`Nat.pow_succ` — same content, different name.
-/

namespace WarpTypesOverflow

section
variable {n : Nat}

-- =========================================================================
-- 1. Core no-wrap lemmas
-- =========================================================================

/-- Addition doesn't wrap when the sum fits in n bits.
    This is the obligation a Rust plugin emits for `checked_add`. -/
theorem add_no_wrap {a b : BitVec n} (h : a.toNat + b.toNat < 2 ^ n) :
    (a + b).toNat = a.toNat + b.toNat := by
  rw [BitVec.toNat_add]
  exact Nat.mod_eq_of_lt h

/-- Multiplication doesn't wrap when the product fits in n bits.
    This is the obligation a Rust plugin emits for `checked_mul`. -/
theorem mul_no_wrap {a b : BitVec n} (h : a.toNat * b.toNat < 2 ^ n) :
    (a * b).toNat = a.toNat * b.toNat := by
  rw [BitVec.toNat_mul]
  exact Nat.mod_eq_of_lt h

/-- Subtraction doesn't wrap when the left operand is ≥ the right.
    This is the obligation a Rust plugin emits for `checked_sub`.

    Proof: `BitVec.toNat_sub` produces `(2^n - b.toNat + a.toNat) % 2^n`,
    which omega cannot simplify directly because of the modular arithmetic.
    We rewrite the inner expression as `(a.toNat - b.toNat) + 2^n`, apply
    `Nat.add_mod_right` to strip the `+ 2^n`, then close with
    `Nat.mod_eq_of_lt` using the fact that `a.toNat - b.toNat < 2^n`
    (from `a.toNat < 2^n` via `BitVec.isLt`). -/
theorem sub_no_wrap {a b : BitVec n} (h : b.toNat ≤ a.toNat) :
    (a - b).toNat = a.toNat - b.toNat := by
  rw [BitVec.toNat_sub]
  have ha := a.isLt
  have h_lt : a.toNat - b.toNat < 2 ^ n := by omega
  have h_rw : 2 ^ n - b.toNat + a.toNat = (a.toNat - b.toNat) + 2 ^ n := by omega
  rw [h_rw, Nat.add_mod_right]
  exact Nat.mod_eq_of_lt h_lt

-- =========================================================================
-- 2. Corollaries and utility wrappers
-- =========================================================================

/-- Every bitvector value is strictly less than `2^n`. Trivial wrapper
    over `BitVec.isLt`, included for call-site uniformity: a Rust plugin
    can treat `value_in_range` as the "type-witnessed bound" obligation
    without knowing whether it discharges to a theorem or to a core
    instance. -/
theorem value_in_range (a : BitVec n) : a.toNat < 2 ^ n := BitVec.isLt a

/-- Addition of two values that are each strictly less than half the
    range cannot wrap. Common in Rust: two `u32` values each known to be
    < 2^31 sum safely as a `u32`.

    Proof: reduce to `add_no_wrap` by showing
    `2^(n-1) + 2^(n-1) = 2^n`. Case-split on `n` because `n - 1` is
    `Nat.sub` (truncating) — the `n = 0` branch is vacuously closed by
    `omega` on the premises `0 < n`. -/
theorem add_half_range {a b : BitVec n} (hn : 0 < n)
    (ha : a.toNat < 2 ^ (n - 1)) (hb : b.toNat < 2 ^ (n - 1)) :
    (a + b).toNat = a.toNat + b.toNat := by
  apply add_no_wrap
  have key : 2 ^ (n - 1) + 2 ^ (n - 1) = 2 ^ n := by
    cases n with
    | zero => omega
    | succ m =>
      show 2 ^ m + 2 ^ m = 2 ^ (m + 1)
      rw [Nat.pow_succ]; omega
  have hab : a.toNat + b.toNat < 2 ^ (n - 1) + 2 ^ (n - 1) :=
    Nat.add_lt_add ha hb
  rw [key] at hab
  exact hab

end

-- =========================================================================
-- 3. Validation — concrete-width witnesses
-- =========================================================================

-- These exercise the lemmas at fixed widths the way a Rust plugin would
-- emit them. No BitVec-agnostic generality, no Sol-specific structure.

/-- Proc-macro shape test (from `Sol/Rust.lean:88`): the exact form a
    Rust `#[sol_verify(overflow_free)]` plugin emits for `u32` subtract. -/
private example (a b : BitVec 32) (h : b.toNat ≤ a.toNat) :
    (a - b).toNat = a.toNat - b.toNat :=
  sub_no_wrap h

/-- Proc-macro shape test for `u64` add. -/
private example (a b : BitVec 64) (h : a.toNat + b.toNat < 2 ^ 64) :
    (a + b).toNat = a.toNat + b.toNat :=
  add_no_wrap h

/-- Proc-macro shape test for `u16` mul. -/
private example (a b : BitVec 16) (h : a.toNat * b.toNat < 2 ^ 16) :
    (a * b).toNat = a.toNat * b.toNat :=
  mul_no_wrap h

/-- `add_half_range` witnessed at `u32`: two values < 2^31 sum safely. -/
private example (a b : BitVec 32)
    (ha : a.toNat < 2 ^ 31) (hb : b.toNat < 2 ^ 31) :
    (a + b).toNat = a.toNat + b.toNat :=
  add_half_range (by omega) ha hb

/-- `value_in_range` witnessed at `u8`. -/
private example (a : BitVec 8) : a.toNat < 2 ^ 8 := value_in_range a

end WarpTypesOverflow
