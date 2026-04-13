//! BMC command-line tool.
//!
//! Checks safety properties on simple transition systems.
//! Currently demonstrates the phase-typed BMC loop on built-in examples.

use warp_types_bmc::{check, BmcResult, TransitionSystem};
use warp_types_sat::literal::Lit;

/// Build a simple 2-bit counter that wraps: 00 → 01 → 10 → 11 → 00.
/// Property: state is never 11 (both bits set). This is UNSAFE — the counter
/// reaches 11 at depth 3.
fn counter_reaches_bad_state() -> TransitionSystem {
    let mut sys = TransitionSystem::new(2);

    // Initial state: both bits 0
    // I(s) = ¬s₀ ∧ ¬s₁
    sys.add_initial(vec![Lit::neg(0)]); // ¬s₀
    sys.add_initial(vec![Lit::neg(1)]); // ¬s₁

    // Transition: increment by 1 (binary addition)
    // s₀' = ¬s₀  (low bit flips every step)
    // s₁' = s₀ ⊕ s₁ (high bit flips when low bit was 1)
    //
    // s₀' = ¬s₀:
    //   (s₀ ∨ s₀')  — if s₀=0 then s₀'=1
    //   (¬s₀ ∨ ¬s₀') — if s₀=1 then s₀'=0
    // State vars: 0=s₀, 1=s₁, 2=s₀', 3=s₁'
    sys.add_transition(vec![Lit::pos(0), Lit::pos(2)]); // s₀ ∨ s₀'
    sys.add_transition(vec![Lit::neg(0), Lit::neg(2)]); // ¬s₀ ∨ ¬s₀'

    // s₁' = s₀ ⊕ s₁ (XOR):
    //   (¬s₀ ∨ ¬s₁ ∨ ¬s₁') — if both 1, next high bit is 0
    //   (¬s₀ ∨ s₁ ∨ s₁')   — if s₀=1, s₁=0, next high bit is 1
    //   (s₀ ∨ ¬s₁ ∨ s₁')   — if s₀=0, s₁=1, next high bit stays 1
    //   (s₀ ∨ s₁ ∨ ¬s₁')   — if both 0, next high bit stays 0
    sys.add_transition(vec![Lit::neg(0), Lit::neg(1), Lit::neg(3)]);
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(0), Lit::pos(1), Lit::neg(3)]);

    // Property: ¬(s₀ ∧ s₁) — "never both bits set"
    // As CNF: (¬s₀ ∨ ¬s₁) — a single clause
    sys.add_property(vec![Lit::neg(0), Lit::neg(1)]);

    sys
}

/// Build a 1-bit toggle that stays at 0 forever.
/// Property: state is never 1. This is SAFE at any depth.
fn safe_constant() -> TransitionSystem {
    let mut sys = TransitionSystem::new(1);

    // I(s) = ¬s₀
    sys.add_initial(vec![Lit::neg(0)]);

    // T(s, s') = s₀' ↔ s₀ (identity — state never changes)
    //   (¬s₀ ∨ s₀') ∧ (s₀ ∨ ¬s₀')
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1)]);
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1)]);

    // P(s) = ¬s₀ (safe if bit is 0)
    sys.add_property(vec![Lit::neg(0)]);

    sys
}

fn main() {
    println!("warp-types-bmc: Phase-typed Bounded Model Checker\n");

    // Example 1: counter that reaches bad state at depth 3
    println!("── Example 1: 2-bit counter, property ¬(s₀ ∧ s₁) ──");
    let sys = counter_reaches_bad_state();
    let result = check(&sys, 10, 0);
    match &result {
        BmcResult::CounterexampleFound { depth, trace } => {
            println!("  UNSAFE: counterexample at depth {depth}");
            for (t, frame) in trace.iter().enumerate() {
                let bits: String = frame.iter().map(|&b| if b { '1' } else { '0' }).collect();
                println!("    t={t}: {bits}");
            }
        }
        BmcResult::BoundedSafe { max_depth } => {
            println!("  SAFE up to depth {max_depth}");
        }
        BmcResult::Exhausted { depth } => {
            println!("  EXHAUSTED at depth {depth}");
        }
    }

    // Example 2: constant system that is always safe
    println!("\n── Example 2: constant 0, property ¬s₀ ──");
    let sys = safe_constant();
    let result = check(&sys, 10, 0);
    match &result {
        BmcResult::CounterexampleFound { depth, trace } => {
            println!("  UNSAFE: counterexample at depth {depth}");
            for (t, frame) in trace.iter().enumerate() {
                let bits: String = frame.iter().map(|&b| if b { '1' } else { '0' }).collect();
                println!("    t={t}: {bits}");
            }
        }
        BmcResult::BoundedSafe { max_depth } => {
            println!("  SAFE up to depth {max_depth}");
        }
        BmcResult::Exhausted { depth } => {
            println!("  EXHAUSTED at depth {depth}");
        }
    }
}
