//! End-to-end integration tests for warp-types-bmc.
//!
//! Each test exercises the full BMC loop:
//! transition system → unrolling → SAT oracle → result.

use warp_types_bmc::{check, BmcResult, TransitionSystem};
use warp_types_sat::literal::Lit;

// ============================================================================
// Known-Unsafe: 2-bit counter reaches 11 at depth 3
// ============================================================================

/// 2-bit counter: s₀ toggles, s₁ = s₁ ⊕ s₀. Counts 00 → 01 → 10 → 11.
/// Property ¬(s₀ ∧ s₁) is violated at depth 3, and no earlier.
fn two_bit_counter() -> TransitionSystem {
    let mut sys = TransitionSystem::new(2);
    sys.add_initial(vec![Lit::neg(0)]);
    sys.add_initial(vec![Lit::neg(1)]);
    // s₀' = ¬s₀
    sys.add_transition(vec![Lit::pos(0), Lit::pos(2)]);
    sys.add_transition(vec![Lit::neg(0), Lit::neg(2)]);
    // s₁' = s₀ ⊕ s₁
    sys.add_transition(vec![Lit::neg(0), Lit::neg(1), Lit::neg(3)]);
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(0), Lit::pos(1), Lit::neg(3)]);
    // P: ¬(s₀ ∧ s₁)
    sys.add_property(vec![Lit::neg(0), Lit::neg(1)]);
    sys
}

#[test]
fn unsafe_counter_found_at_depth_3() {
    let sys = two_bit_counter();
    match check(&sys, 10, 0) {
        BmcResult::CounterexampleFound { depth, trace } => {
            assert_eq!(depth, 3, "counterexample must be found at depth 3");
            assert_eq!(trace.len(), 4);
            // Initial state 00
            assert!(!trace[0][0] && !trace[0][1]);
            // Final state 11 (violates ¬(s₀ ∧ s₁))
            assert!(trace[3][0] && trace[3][1]);
        }
        other => panic!("expected CounterexampleFound at depth 3, got {:?}", other),
    }
}

#[test]
fn unsafe_counter_bounded_safe_below_bug_depth() {
    // Same counter, but max_depth = 2: the bug at depth 3 is out of reach.
    let sys = two_bit_counter();
    match check(&sys, 2, 0) {
        BmcResult::BoundedSafe { max_depth } => assert_eq!(max_depth, 2),
        other => panic!("expected BoundedSafe at max_depth 2, got {:?}", other),
    }
}

// ============================================================================
// Known-BoundedSafe: 1-bit identity system
// ============================================================================

#[test]
fn bounded_safe_identity() {
    // s₀ starts 0 and never changes; property ¬s₀ holds at every depth.
    let mut sys = TransitionSystem::new(1);
    sys.add_initial(vec![Lit::neg(0)]);
    // s₀' = s₀
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1)]);
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1)]);
    sys.add_property(vec![Lit::neg(0)]);

    match check(&sys, 10, 0) {
        BmcResult::BoundedSafe { max_depth } => assert_eq!(max_depth, 10),
        other => panic!("expected BoundedSafe, got {:?}", other),
    }
}

// ============================================================================
// Empty property (finding D): vacuously true ⇒ BoundedSafe at every depth
// ============================================================================

#[test]
fn empty_property_is_bounded_safe() {
    // The empty conjunction of property clauses is `true`; ¬P = false, so
    // no depth can have a counterexample. Regression: the encoder used to
    // emit nothing for ¬P, making the instance I ∧ T^k — typically SAT —
    // and reporting a false counterexample at depth 0.
    for max_depth in [0, 3, 10] {
        let mut sys = TransitionSystem::new(1);
        sys.add_initial(vec![Lit::neg(0)]);
        // s₀' = ¬s₀ (toggle)
        sys.add_transition(vec![Lit::pos(0), Lit::pos(1)]);
        sys.add_transition(vec![Lit::neg(0), Lit::neg(1)]);
        // No property clauses.

        match check(&sys, max_depth, 0) {
            BmcResult::BoundedSafe { max_depth: d } => assert_eq!(d, max_depth),
            other => panic!(
                "empty property must be BoundedSafe at max_depth {}, got {:?}",
                max_depth, other
            ),
        }
    }
}

// ============================================================================
// Model validation (finding E): out-of-range variables are rejected
// ============================================================================

#[test]
#[should_panic(expected = "initial-state clause")]
fn add_initial_rejects_out_of_range_var() {
    let mut sys = TransitionSystem::new(2);
    // Var 2 is outside [0, 2): would silently alias into time frame 1.
    sys.add_initial(vec![Lit::pos(2)]);
}

#[test]
#[should_panic(expected = "property clause")]
fn add_property_rejects_out_of_range_var() {
    let mut sys = TransitionSystem::new(2);
    sys.add_property(vec![Lit::neg(2)]);
}

#[test]
#[should_panic(expected = "transition clause")]
fn add_transition_rejects_out_of_range_var() {
    let mut sys = TransitionSystem::new(2);
    // Var 4 is outside [0, 4): would silently alias two time frames ahead.
    sys.add_transition(vec![Lit::pos(4)]);
}

#[test]
fn add_transition_accepts_next_state_vars() {
    let mut sys = TransitionSystem::new(2);
    // Vars in [0, 2n) are the documented contract: current [0, n), next [n, 2n).
    sys.add_transition(vec![Lit::neg(0), Lit::pos(3)]);
}
