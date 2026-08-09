//! Transition system model for bounded model checking.
//!
//! A transition system is (S, I, T, P) where:
//! - S is a set of boolean state variables
//! - I(s) is the initial state predicate (CNF over state vars at time 0)
//! - T(s, s') is the transition relation (CNF over state vars at time k and k+1)
//! - P(s) is the safety property (CNF — satisfied in safe states)
//!
//! BMC asks: ∃ s₀, s₁, ..., sₖ such that I(s₀) ∧ T(s₀,s₁) ∧ ... ∧ T(sₖ₋₁,sₖ) ∧ ¬P(sₖ)?
//! If SAT: counterexample trace of length k. If UNSAT: no bug at depth ≤ k.

use warp_types_sat::literal::Lit;

/// A state variable in the transition system.
/// At unroll depth k, state variable `v` becomes SAT variable `v + k * num_state_vars`.
pub type StateVar = u32;

/// A clause over state variables. Literals reference state variables
/// (not time-indexed SAT variables — the unroller does the indexing).
#[derive(Debug, Clone)]
pub struct ModelClause {
    /// Literals in this clause. Each literal's variable is a `StateVar`.
    pub lits: Vec<Lit>,
}

/// A transition clause relating current-state and next-state variables.
/// Current-state literals use variables `0..num_vars`.
/// Next-state literals use variables `num_vars..2*num_vars`.
#[derive(Debug, Clone)]
pub struct TransitionClause {
    pub lits: Vec<Lit>,
}

/// A transition system: the input to bounded model checking.
///
/// Fields are private: every clause must enter through the validating
/// `add_*` methods, which reject literals outside the declared variable
/// ranges. Out-of-range variables would be silently time-shifted into
/// later unrolling frames, corrupting the encoding (and the verdict) —
/// so there is deliberately no way to construct or mutate the clause
/// lists without going through validation.
///
/// ```compile_fail
/// use warp_types_bmc::model::TransitionSystem;
/// // Struct-literal construction bypasses validation — must not compile.
/// let sys = TransitionSystem {
///     num_state_vars: 1,
///     initial: Vec::new(),
///     transition: Vec::new(),
///     property: Vec::new(),
/// };
/// ```
///
/// ```compile_fail
/// use warp_types_bmc::model::{ModelClause, TransitionSystem};
/// use warp_types_sat::literal::Lit;
/// // Direct field push bypasses validation — must not compile.
/// let mut sys = TransitionSystem::new(1);
/// sys.initial.push(ModelClause { lits: vec![Lit::pos(99)] });
/// ```
#[derive(Debug, Clone)]
pub struct TransitionSystem {
    /// Number of boolean state variables.
    num_state_vars: u32,
    /// Initial state predicate I(s₀): CNF over state vars [0, num_state_vars).
    initial: Vec<ModelClause>,
    /// Transition relation T(s, s'): CNF over [0, 2*num_state_vars).
    /// Variables [0, num_state_vars) are current-state, [num_state_vars, 2*num_state_vars) are next-state.
    transition: Vec<TransitionClause>,
    /// Safety property P(s): CNF over [0, num_state_vars). Satisfied = safe.
    /// BMC checks ¬P, so a SAT result means the property is violated.
    property: Vec<ModelClause>,
}

impl TransitionSystem {
    /// Create a new transition system.
    pub fn new(num_state_vars: u32) -> Self {
        TransitionSystem {
            num_state_vars,
            initial: Vec::new(),
            transition: Vec::new(),
            property: Vec::new(),
        }
    }

    /// Number of boolean state variables.
    pub fn num_state_vars(&self) -> u32 {
        self.num_state_vars
    }

    /// Initial state predicate I(s₀): CNF over state vars [0, num_state_vars).
    pub fn initial(&self) -> &[ModelClause] {
        &self.initial
    }

    /// Transition relation T(s, s'): CNF over [0, 2*num_state_vars).
    /// Variables [0, num_state_vars) are current-state, [num_state_vars, 2*num_state_vars) are next-state.
    pub fn transition(&self) -> &[TransitionClause] {
        &self.transition
    }

    /// Safety property P(s): CNF over [0, num_state_vars). Satisfied = safe.
    /// BMC checks ¬P, so a SAT result means the property is violated.
    pub fn property(&self) -> &[ModelClause] {
        &self.property
    }

    /// Add an initial-state clause.
    ///
    /// # Panics
    /// Panics if any literal references a variable outside `[0, num_state_vars)`.
    /// Out-of-range variables would be silently time-shifted into later
    /// unrolling frames, corrupting the encoding (and the verdict).
    pub fn add_initial(&mut self, lits: Vec<Lit>) {
        for &lit in &lits {
            assert!(
                lit.var() < self.num_state_vars,
                "initial-state clause literal references variable {}, but valid \
                 state variables are [0, {})",
                lit.var(),
                self.num_state_vars
            );
        }
        self.initial.push(ModelClause { lits });
    }

    /// Add a transition clause (current-state vars in [0, n), next-state in [n, 2n)).
    ///
    /// # Panics
    /// Panics if any literal references a variable outside `[0, 2*num_state_vars)`.
    /// Out-of-range variables would be silently time-shifted into later
    /// unrolling frames, corrupting the encoding (and the verdict).
    pub fn add_transition(&mut self, lits: Vec<Lit>) {
        for &lit in &lits {
            assert!(
                lit.var() < 2 * self.num_state_vars,
                "transition clause literal references variable {}, but valid \
                 variables are [0, {}) (current-state [0, {}), next-state [{}, {}))",
                lit.var(),
                2 * self.num_state_vars,
                self.num_state_vars,
                self.num_state_vars,
                2 * self.num_state_vars
            );
        }
        self.transition.push(TransitionClause { lits });
    }

    /// Add a safety property clause (negated for BMC — SAT means violation).
    ///
    /// # Panics
    /// Panics if any literal references a variable outside `[0, num_state_vars)`.
    /// Out-of-range variables would be silently time-shifted into later
    /// unrolling frames, corrupting the encoding (and the verdict).
    pub fn add_property(&mut self, lits: Vec<Lit>) {
        for &lit in &lits {
            assert!(
                lit.var() < self.num_state_vars,
                "property clause literal references variable {}, but valid \
                 state variables are [0, {})",
                lit.var(),
                self.num_state_vars
            );
        }
        self.property.push(ModelClause { lits });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accessors expose exactly what the validating add_* methods stored.
    #[test]
    fn accessors_reflect_validated_construction() {
        let mut sys = TransitionSystem::new(2);
        sys.add_initial(vec![Lit::neg(0)]);
        sys.add_transition(vec![Lit::pos(0), Lit::pos(2)]);
        sys.add_property(vec![Lit::neg(0), Lit::neg(1)]);

        assert_eq!(sys.num_state_vars(), 2);
        assert_eq!(sys.initial().len(), 1);
        assert_eq!(sys.initial()[0].lits, vec![Lit::neg(0)]);
        assert_eq!(sys.transition().len(), 1);
        assert_eq!(sys.transition()[0].lits, vec![Lit::pos(0), Lit::pos(2)]);
        assert_eq!(sys.property().len(), 1);
        assert_eq!(sys.property()[0].lits, vec![Lit::neg(0), Lit::neg(1)]);
    }

    #[test]
    #[should_panic(expected = "initial-state clause literal references variable 2")]
    fn add_initial_rejects_out_of_range() {
        let mut sys = TransitionSystem::new(2);
        sys.add_initial(vec![Lit::pos(2)]);
    }

    #[test]
    #[should_panic(expected = "transition clause literal references variable 4")]
    fn add_transition_rejects_out_of_range() {
        let mut sys = TransitionSystem::new(2);
        sys.add_transition(vec![Lit::pos(4)]);
    }

    #[test]
    #[should_panic(expected = "property clause literal references variable 2")]
    fn add_property_rejects_out_of_range() {
        let mut sys = TransitionSystem::new(2);
        sys.add_property(vec![Lit::pos(2)]);
    }
}
