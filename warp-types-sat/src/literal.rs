//! SAT literals and variables.
//!
//! A literal is a variable with polarity. Variable 3 positive = Lit(6),
//! variable 3 negative = Lit(7). This is the standard DIMACS encoding:
//! `var_index = lit / 2`, `is_negated = lit & 1`.

/// A boolean variable (0-indexed).
pub type Variable = u32;

/// A literal: a variable with polarity.
///
/// Encoding: `variable * 2 + polarity` where polarity 0 = positive, 1 = negative.
/// This is the standard internal encoding used by MiniSat, CaDiCaL, etc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Lit(u32);

impl Lit {
    /// Create a positive literal for variable `v`.
    ///
    /// # Panics
    /// Panics if `v > u32::MAX / 2` — the `var * 2` encoding would overflow
    /// and silently alias another variable in release builds.
    pub fn pos(v: Variable) -> Self {
        assert!(v <= u32::MAX / 2, "variable {v} overflows Lit encoding");
        Lit(v * 2)
    }

    /// Create a negative literal for variable `v`.
    ///
    /// # Panics
    /// Panics if `v > u32::MAX / 2` — the `var * 2 + 1` encoding would overflow
    /// and silently alias another variable in release builds.
    pub fn neg(v: Variable) -> Self {
        assert!(v <= u32::MAX / 2, "variable {v} overflows Lit encoding");
        Lit(v * 2 + 1)
    }

    /// The underlying variable.
    pub fn var(self) -> Variable {
        self.0 / 2
    }

    /// Whether this literal is negated.
    pub fn is_negated(self) -> bool {
        self.0 & 1 == 1
    }

    /// The complementary literal (negate).
    pub fn complement(self) -> Self {
        Lit(self.0 ^ 1)
    }

    /// Raw encoding.
    pub fn code(self) -> u32 {
        self.0
    }

    /// Inverse of [`Lit::code`]. Total, and safe: `Lit` is a
    /// `#[repr(transparent)]` newtype over `u32` with no niche and no validity
    /// invariant, so every `u32` denotes some literal. Exists so branchless
    /// selection over `code()` values does not need a `transmute` — see the
    /// partner select in `watch.rs`. Unlike `pos`/`neg` it cannot overflow the
    /// encoding, because it does not perform one.
    ///
    /// `#[inline]` is load-bearing, not decoration: without it the partner
    /// select in BCP's hot loop measured ~2.5% slower at random_3sat n=50 than
    /// the `transmute` it replaced, because a non-inlined call sat in the
    /// middle of a branchless sequence.
    #[inline]
    pub fn from_code(code: u32) -> Self {
        Lit(code)
    }

    /// Evaluate this literal given a variable assignment.
    /// `assign[var]`: Some(true) = true, Some(false) = false, None = unassigned.
    pub fn eval(self, assign: &[Option<bool>]) -> Option<bool> {
        assign
            .get(self.var() as usize)
            .copied()
            .flatten()
            .map(|val| if self.is_negated() { !val } else { val })
    }
}

impl core::fmt::Display for Lit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_negated() {
            write!(f, "¬x{}", self.var())
        } else {
            write!(f, "x{}", self.var())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_encoding() {
        let p = Lit::pos(3);
        let n = Lit::neg(3);
        assert_eq!(p.var(), 3);
        assert_eq!(n.var(), 3);
        assert!(!p.is_negated());
        assert!(n.is_negated());
        assert_eq!(p.complement(), n);
        assert_eq!(n.complement(), p);
    }

    #[test]
    #[should_panic(expected = "overflows Lit encoding")]
    fn pos_overflow_panics() {
        // v * 2 wraps for v > u32::MAX / 2, silently aliasing another
        // variable in release builds before the release assert.
        let _ = Lit::pos(u32::MAX / 2 + 1);
    }

    #[test]
    #[should_panic(expected = "overflows Lit encoding")]
    fn neg_overflow_panics() {
        let _ = Lit::neg(u32::MAX / 2 + 1);
    }

    #[test]
    fn literal_eval() {
        let assign = vec![None, Some(true), Some(false)];
        assert_eq!(Lit::pos(0).eval(&assign), None); // unassigned
        assert_eq!(Lit::pos(1).eval(&assign), Some(true)); // x1 = true
        assert_eq!(Lit::neg(1).eval(&assign), Some(false)); // ¬x1 = false
        assert_eq!(Lit::pos(2).eval(&assign), Some(false)); // x2 = false
        assert_eq!(Lit::neg(2).eval(&assign), Some(true)); // ¬x2 = true
    }
}
