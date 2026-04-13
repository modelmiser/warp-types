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
#[derive(Debug, Clone)]
pub struct TransitionSystem {
    /// Number of boolean state variables.
    pub num_state_vars: u32,
    /// Initial state predicate I(s₀): CNF over state vars [0, num_state_vars).
    pub initial: Vec<ModelClause>,
    /// Transition relation T(s, s'): CNF over [0, 2*num_state_vars).
    /// Variables [0, num_state_vars) are current-state, [num_state_vars, 2*num_state_vars) are next-state.
    pub transition: Vec<TransitionClause>,
    /// Safety property P(s): CNF over [0, num_state_vars). Satisfied = safe.
    /// BMC checks ¬P, so a SAT result means the property is violated.
    pub property: Vec<ModelClause>,
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

    /// Add an initial-state clause.
    pub fn add_initial(&mut self, lits: Vec<Lit>) {
        self.initial.push(ModelClause { lits });
    }

    /// Add a transition clause (current-state vars in [0, n), next-state in [n, 2n)).
    pub fn add_transition(&mut self, lits: Vec<Lit>) {
        self.transition.push(TransitionClause { lits });
    }

    /// Add a safety property clause (negated for BMC — SAT means violation).
    pub fn add_property(&mut self, lits: Vec<Lit>) {
        self.property.push(ModelClause { lits });
    }
}
