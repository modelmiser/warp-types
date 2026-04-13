//! Cube representation for PDR.
//!
//! A cube is a conjunction of literals — a partial assignment representing
//! a set of states. PDR blocks cubes by adding their negation (a clause)
//! to the frame sequence.

use warp_types_sat::literal::Lit;

/// A cube: conjunction of literals representing a set of states.
///
/// In PDR, cubes are extracted from SAT models (counterexample-to-induction
/// states) and blocked by adding their negation as clauses to frames.
#[derive(Debug, Clone)]
pub struct Cube {
    /// Literals in the cube (all must hold simultaneously).
    pub lits: Vec<Lit>,
}

impl Cube {
    /// Create a cube from a list of literals.
    pub fn new(lits: Vec<Lit>) -> Self {
        Cube { lits }
    }

    /// Negate the cube: returns a clause (disjunction of complemented literals).
    /// ¬(l₁ ∧ l₂ ∧ ... ∧ lₙ) = (¬l₁ ∨ ¬l₂ ∨ ... ∨ ¬lₙ)
    pub fn negate(&self) -> Vec<Lit> {
        self.lits.iter().map(|l| l.complement()).collect()
    }

    /// Shift all variable indices by `offset`.
    /// Used to map current-state cubes to next-state encoding.
    pub fn shift(&self, offset: u32) -> Cube {
        Cube {
            lits: self.lits.iter().map(|&l| shift_lit(l, offset)).collect(),
        }
    }

    /// Extract a cube from a SAT assignment over the first `num_state_vars` variables.
    pub fn from_assignment(assignment: &[bool], num_state_vars: u32) -> Cube {
        let lits: Vec<Lit> = (0..num_state_vars)
            .map(|v| {
                if assignment[v as usize] {
                    Lit::pos(v)
                } else {
                    Lit::neg(v)
                }
            })
            .collect();
        Cube { lits }
    }

    /// Extract a cube from next-state variables in a SAT assignment.
    /// Maps variables `[offset, offset + num_state_vars)` back to `[0, num_state_vars)`.
    pub fn from_assignment_next_state(assignment: &[bool], num_state_vars: u32) -> Cube {
        let offset = num_state_vars as usize;
        let lits: Vec<Lit> = (0..num_state_vars)
            .map(|v| {
                if assignment[offset + v as usize] {
                    Lit::pos(v)
                } else {
                    Lit::neg(v)
                }
            })
            .collect();
        Cube { lits }
    }

    /// Convert to a boolean state vector.
    pub fn to_state_vec(&self, num_state_vars: u32) -> Vec<bool> {
        let mut state = vec![false; num_state_vars as usize];
        for &lit in &self.lits {
            let v = lit.var() as usize;
            if v < state.len() {
                state[v] = !lit.is_negated();
            }
        }
        state
    }
}

/// Shift a literal's variable by `offset`.
pub(crate) fn shift_lit(lit: Lit, offset: u32) -> Lit {
    let var = lit.var() + offset;
    if lit.is_negated() {
        Lit::neg(var)
    } else {
        Lit::pos(var)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_negate() {
        let cube = Cube::new(vec![Lit::pos(0), Lit::neg(1)]);
        let clause = cube.negate();
        assert_eq!(clause.len(), 2);
        assert_eq!(clause[0], Lit::neg(0));
        assert_eq!(clause[1], Lit::pos(1));
    }

    #[test]
    fn cube_shift() {
        let cube = Cube::new(vec![Lit::pos(0), Lit::neg(1)]);
        let shifted = cube.shift(3);
        assert_eq!(shifted.lits[0], Lit::pos(3));
        assert_eq!(shifted.lits[1], Lit::neg(4));
    }

    #[test]
    fn cube_from_assignment() {
        let assign = vec![true, false, true];
        let cube = Cube::from_assignment(&assign, 3);
        assert_eq!(cube.lits[0], Lit::pos(0));
        assert_eq!(cube.lits[1], Lit::neg(1));
        assert_eq!(cube.lits[2], Lit::pos(2));
    }

    #[test]
    fn cube_roundtrip() {
        let assign = vec![true, false, true, false];
        let cube = Cube::from_assignment(&assign, 4);
        let state = cube.to_state_vec(4);
        assert_eq!(state, assign);
    }
}
