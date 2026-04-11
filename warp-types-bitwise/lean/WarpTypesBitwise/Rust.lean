/-!
# WarpTypesBitwise.Rust

Rust-shaped bitvector tautology and mask-algebra lemmas.

These are the lemmas that arise when Rust code annotated with
`#[sol_verify(mask_safe)]` (or similar) is lowered to bitvector
obligations. All five theorems are width-parametric (`∀ n : Nat`).
-/

namespace WarpTypesBitwise

section
variable {n : Nat}

/-- Masking is idempotent: applying a mask twice is the same as once.
    Arises in Rust systems code: `(x & MASK) & MASK == x & MASK`. -/
theorem mask_idempotent (x mask : BitVec n) :
    (x &&& mask) &&& mask = x &&& mask := by
  ext i hi
  simp [BitVec.getElem_and]

/-- Disjoint masks produce zero: `(x & 0xF0) & 0x0F == 0`. -/
theorem disjoint_masks (x m1 m2 : BitVec n)
    (h : m1 &&& m2 = 0#n) :
    (x &&& m1) &&& m2 = 0#n := by
  ext i hi
  have h_bit : m1[i]'hi = true → m2[i]'hi = false := by
    have := congrArg (·[i]'hi) h
    simpa [BitVec.getElem_and, BitVec.getElem_zero] using this
  simp [BitVec.getElem_and, BitVec.getElem_zero]
  cases hm1 : m1[i]'hi
  · simp
  · simp [h_bit hm1]

/-- Bit extraction and reinsertion: clear field, set field, read field back.
    `((val & ~mask) | (field & mask)) & mask == field & mask` -/
theorem field_insert_read (val field mask : BitVec n) :
    ((val &&& ~~~mask) ||| (field &&& mask)) &&& mask = field &&& mask := by
  ext i hi
  simp [BitVec.getElem_and, BitVec.getElem_or, BitVec.getElem_not]
  cases mask[i]'hi <;> simp

/-- `CounterLE counter bound` holds when `counter` has no bits outside `bound`. -/
def CounterLE {n : Nat} (counter bound : BitVec n) : Prop :=
  counter &&& ~~~bound = 0#n

/-- Masking a counter by its bound always produces a valid counter:
    `(counter & bound) & ~bound == 0`. -/
theorem counter_mask_valid (counter bound : BitVec n) :
    CounterLE (counter &&& bound) bound := by
  unfold CounterLE
  ext i hi
  simp [BitVec.getElem_and, BitVec.getElem_not, BitVec.getElem_zero]

/-- Disjoint resource tracking: updating one mask-gated region does not
    affect another disjoint mask-gated region. -/
theorem disjoint_update {state mask_a mask_b new_a : BitVec n}
    (h_disjoint : mask_a &&& mask_b = 0#n) :
    ((state &&& ~~~mask_a) ||| (new_a &&& mask_a)) &&& mask_b =
    state &&& mask_b := by
  ext i hi
  have h_bit : mask_a[i]'hi = true → mask_b[i]'hi = false := by
    have := congrArg (·[i]'hi) h_disjoint
    simpa [BitVec.getElem_and, BitVec.getElem_zero] using this
  simp [BitVec.getElem_and, BitVec.getElem_or, BitVec.getElem_not]
  cases hma : mask_a[i]'hi
  · simp
  · -- mask_a[i] = true, so mask_b[i] = false
    simp [h_bit hma]

end
end WarpTypesBitwise
