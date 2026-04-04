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
pub struct Lit(u32);

impl Lit {
    /// Create a positive literal for variable `v`.
    pub fn pos(v: Variable) -> Self {
        Lit(v * 2)
    }

    /// Create a negative literal for variable `v`.
    pub fn neg(v: Variable) -> Self {
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

    /// Evaluate this literal given a variable assignment.
    /// `assign[var]`: Some(true) = true, Some(false) = false, None = unassigned.
    pub fn eval(self, assign: &[Option<bool>]) -> Option<bool> {
        assign.get(self.var() as usize).copied().flatten().map(|val| {
            if self.is_negated() {
                !val
            } else {
                val
            }
        })
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
    fn literal_eval() {
        let assign = vec![None, Some(true), Some(false)];
        assert_eq!(Lit::pos(0).eval(&assign), None);        // unassigned
        assert_eq!(Lit::pos(1).eval(&assign), Some(true));   // x1 = true
        assert_eq!(Lit::neg(1).eval(&assign), Some(false));  // ¬x1 = false
        assert_eq!(Lit::pos(2).eval(&assign), Some(false));  // x2 = false
        assert_eq!(Lit::neg(2).eval(&assign), Some(true));   // ¬x2 = true
    }
}
