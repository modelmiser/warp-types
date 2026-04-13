//! End-to-end integration tests for warp-types-pdr.
//!
//! Each test exercises the full IC3/PDR loop:
//! transition system → frame sequence → SAT queries → result.

use warp_types_bmc::TransitionSystem;
use warp_types_pdr::{check, PdrResult};
use warp_types_sat::literal::Lit;

// ============================================================================
// Test 1: Trivially safe — constant 0
// ============================================================================

#[test]
fn safe_constant() {
    let mut sys = TransitionSystem::new(1);
    sys.add_initial(vec![Lit::neg(0)]);
    // Identity: s₀' = s₀
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1)]);
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1)]);
    sys.add_property(vec![Lit::neg(0)]);

    match check(&sys, 20, 0) {
        PdrResult::Safe { .. } => {}
        other => panic!("expected Safe, got {:?}", other),
    }
}

// ============================================================================
// Test 2: Unsafe counter — reaches bad state at depth 3
// ============================================================================

#[test]
fn unsafe_counter() {
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
    sys.add_property(vec![Lit::neg(0), Lit::neg(1)]);

    match check(&sys, 20, 0) {
        PdrResult::CounterexampleFound { depth, trace } => {
            assert_eq!(depth, 3);
            assert_eq!(trace.len(), 4);
            // Initial state: 00
            assert!(!trace[0][0] && !trace[0][1]);
            // Final state: 11 (violates property)
            assert!(trace[3][0] && trace[3][1]);
        }
        other => panic!("expected CounterexampleFound, got {:?}", other),
    }
}

// ============================================================================
// Test 3: Initial violation — property false at depth 0
// ============================================================================

#[test]
fn initial_violation() {
    let mut sys = TransitionSystem::new(1);
    sys.add_initial(vec![Lit::pos(0)]); // s₀ = 1
    // Identity
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1)]);
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1)]);
    sys.add_property(vec![Lit::neg(0)]); // ¬s₀ — violated initially

    match check(&sys, 20, 0) {
        PdrResult::CounterexampleFound { depth, trace } => {
            assert_eq!(depth, 0);
            assert_eq!(trace.len(), 1);
            assert!(trace[0][0]); // s₀ = true
        }
        other => panic!("expected CounterexampleFound at depth 0, got {:?}", other),
    }
}

// ============================================================================
// Test 4: Invariant discovery — s₁ never set (requires strengthening)
// ============================================================================

#[test]
fn invariant_discovery() {
    let mut sys = TransitionSystem::new(2);
    sys.add_initial(vec![Lit::neg(0)]);
    sys.add_initial(vec![Lit::neg(1)]);
    // s₀' = ¬s₀ (toggle)
    sys.add_transition(vec![Lit::pos(0), Lit::pos(2)]);
    sys.add_transition(vec![Lit::neg(0), Lit::neg(2)]);
    // s₁' = s₁ (stays)
    sys.add_transition(vec![Lit::neg(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(1), Lit::neg(3)]);
    sys.add_property(vec![Lit::neg(1)]); // ¬s₁

    match check(&sys, 20, 0) {
        PdrResult::Safe { .. } => {}
        other => panic!("expected Safe, got {:?}", other),
    }
}

// ============================================================================
// Test 5: Frame budget exhaustion
// ============================================================================

#[test]
fn budget_exhaustion() {
    // Use the unsafe counter but with max_frames=1 (not enough to find cex)
    let mut sys = TransitionSystem::new(2);
    sys.add_initial(vec![Lit::neg(0)]);
    sys.add_initial(vec![Lit::neg(1)]);
    sys.add_transition(vec![Lit::pos(0), Lit::pos(2)]);
    sys.add_transition(vec![Lit::neg(0), Lit::neg(2)]);
    sys.add_transition(vec![Lit::neg(0), Lit::neg(1), Lit::neg(3)]);
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(0), Lit::pos(1), Lit::neg(3)]);
    sys.add_property(vec![Lit::neg(0), Lit::neg(1)]);

    // With max_frames=1, PDR can block at level 1 but can't explore enough
    // to find the depth-3 counterexample
    let result = check(&sys, 1, 0);
    // Should either find a counterexample (if blocking reaches level 0 within 1 frame)
    // or exhaust frames — either is acceptable for this test
    match result {
        PdrResult::Exhausted { .. } | PdrResult::CounterexampleFound { .. } => {}
        PdrResult::Safe { .. } => panic!("counter should not be safe"),
    }
}

// ============================================================================
// Test 6: Tautology property — always safe
// ============================================================================

#[test]
fn tautology_property() {
    // 1-bit toggle, property: (s₀ ∨ ¬s₀) encoded as two unit property clauses
    // Actually, any non-empty property that is always true works.
    // Simplest: property has no clauses (vacuously true) → but we need at least
    // one clause for the Tseitin encoding. Use: property = {s₀, ¬s₀} = tautology.
    // Actually a single clause (s₀ ∨ ¬s₀) IS a tautology.
    let mut sys = TransitionSystem::new(1);
    sys.add_initial(vec![Lit::neg(0)]);
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1)]); // identity
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1)]);
    // Property: (s₀ ∨ ¬s₀) — always true
    sys.add_property(vec![Lit::pos(0), Lit::neg(0)]);

    match check(&sys, 20, 0) {
        PdrResult::Safe { .. } => {}
        other => panic!("tautology property should be safe, got {:?}", other),
    }
}
