/-!
# WarpTypesBitwise.Verilog

Verilog else-path algebra lemmas.

Three lemmas covering the bitvector identities that arise when a hardware
divergence stack's else-path is modeled: specifically, the fact that
`parent & ~taken` equals `parent & ~pred` when `taken = parent & pred`,
and that this is in turn equal to `parent XOR (parent & pred)`.

All three are width-parametric (`∀ n : Nat`).
-/

namespace WarpTypesBitwise

section
variable {n : Nat}

/-- ELSE produces the complement: after `diverge(mask, pred)` where
    `taken = mask & pred`, the else path is `mask & ~taken = mask & ~pred`. -/
theorem else_complement (mask pred : BitVec n) :
    mask &&& ~~~(mask &&& pred) = mask &&& ~~~pred := by
  ext i hi
  simp [BitVec.getElem_and, BitVec.getElem_not]
  by_cases hm : mask[i]'hi = true
  · simp [hm]
  · simp at hm; simp [hm]

/-- ELSE result is disjoint from the taken path:
    `(mask & pred) & (mask & ~pred) == 0`. -/
theorem else_disjoint_from_taken (mask pred : BitVec n) :
    (mask &&& pred) &&& (mask &&& ~~~pred) = 0#n := by
  ext i hi
  simp [BitVec.getElem_and, BitVec.getElem_not, BitVec.getElem_zero]
  by_cases hp : pred[i]'hi = true
  · simp [hp]
  · simp at hp; simp [hp]

/-- Hardware equivalence: `parent XOR (parent & pred) == parent & ~pred`.
    This is how the RTL computes the else mask from the stack top and the
    current diverge predicate. -/
theorem rtl_else_xor (parent pred : BitVec n) :
    parent ^^^ (parent &&& pred) = parent &&& ~~~pred := by
  ext i hi
  simp [BitVec.getElem_xor, BitVec.getElem_and, BitVec.getElem_not]
  by_cases hp : pred[i]'hi = true
  · simp [hp]
  · simp at hp; simp [hp]

end
end WarpTypesBitwise
