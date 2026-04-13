//! PDR command-line tool.
//!
//! Demonstrates Property-Directed Reachability on simple transition systems.

use warp_types_bmc::TransitionSystem;
use warp_types_pdr::{check, PdrResult};
use warp_types_sat::literal::Lit;

/// 1-bit toggle that stays at 0 forever. Property: ¬s₀.
/// PDR should prove SAFE (the invariant ¬s₀ is immediately inductive).
fn safe_constant() -> TransitionSystem {
    let mut sys = TransitionSystem::new(1);
    sys.add_initial(vec![Lit::neg(0)]); // s₀ = 0
                                        // Identity transition: s₀' = s₀
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1)]); // ¬s₀ ∨ s₀'
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1)]); // s₀ ∨ ¬s₀'
    sys.add_property(vec![Lit::neg(0)]); // P: ¬s₀
    sys
}

/// 2-bit counter: 00 → 01 → 10 → 11 → 00. Property: ¬(s₀ ∧ s₁).
/// Counter reaches 11 at depth 3 — PDR should find the counterexample.
fn unsafe_counter() -> TransitionSystem {
    let mut sys = TransitionSystem::new(2);
    sys.add_initial(vec![Lit::neg(0)]); // s₀ = 0
    sys.add_initial(vec![Lit::neg(1)]); // s₁ = 0

    // s₀' = ¬s₀
    sys.add_transition(vec![Lit::pos(0), Lit::pos(2)]); // s₀ ∨ s₀'
    sys.add_transition(vec![Lit::neg(0), Lit::neg(2)]); // ¬s₀ ∨ ¬s₀'

    // s₁' = s₀ ⊕ s₁
    sys.add_transition(vec![Lit::neg(0), Lit::neg(1), Lit::neg(3)]);
    sys.add_transition(vec![Lit::neg(0), Lit::pos(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(0), Lit::neg(1), Lit::pos(3)]);
    sys.add_transition(vec![Lit::pos(0), Lit::pos(1), Lit::neg(3)]);

    sys.add_property(vec![Lit::neg(0), Lit::neg(1)]); // ¬(s₀ ∧ s₁)
    sys
}

/// 2-bit system: s₀ toggles, s₁ stays constant.
/// Initial: s₀=0, s₁=0. Transitions: s₀' = ¬s₀, s₁' = s₁.
/// Property: ¬s₁ (bit 1 is never set).
/// SAFE — but requires PDR to discover the invariant ¬s₁, because
/// the state space includes {s₀=0,s₁=1} and {s₀=1,s₁=1} which
/// violate the property but are unreachable from the initial state.
fn invariant_discovery() -> TransitionSystem {
    let mut sys = TransitionSystem::new(2);
    sys.add_initial(vec![Lit::neg(0)]); // s₀ = 0
    sys.add_initial(vec![Lit::neg(1)]); // s₁ = 0

    // s₀' = ¬s₀ (toggle):
    //   (s₀ ∨ s₀') and (¬s₀ ∨ ¬s₀')
    sys.add_transition(vec![Lit::pos(0), Lit::pos(2)]); // s₀ ∨ s₀'
    sys.add_transition(vec![Lit::neg(0), Lit::neg(2)]); // ¬s₀ ∨ ¬s₀'

    // s₁' = s₁ (identity):
    //   (¬s₁ ∨ s₁') and (s₁ ∨ ¬s₁')
    sys.add_transition(vec![Lit::neg(1), Lit::pos(3)]); // ¬s₁ ∨ s₁'
    sys.add_transition(vec![Lit::pos(1), Lit::neg(3)]); // s₁ ∨ ¬s₁'

    // Property: ¬s₁
    sys.add_property(vec![Lit::neg(1)]);
    sys
}

fn main() {
    println!("warp-types-pdr: Phase-typed Property-Directed Reachability\n");

    // Example 1: safe constant
    println!("── Example 1: constant 0, property ¬s₀ ──");
    print_result(&check(&safe_constant(), 20, 0));

    // Example 2: unsafe counter
    println!("\n── Example 2: 2-bit counter, property ¬(s₀ ∧ s₁) ──");
    print_result(&check(&unsafe_counter(), 20, 0));

    // Example 3: invariant discovery
    println!("\n── Example 3: invariant discovery, property ¬(s₁ ∧ ¬s₀) ──");
    print_result(&check(&invariant_discovery(), 20, 0));
}

fn print_result(result: &PdrResult) {
    match result {
        PdrResult::Safe { invariant_frame } => {
            println!("  SAFE: inductive invariant at frame {invariant_frame}");
        }
        PdrResult::CounterexampleFound { depth, trace } => {
            println!("  UNSAFE: counterexample at depth {depth}");
            for (t, frame) in trace.iter().enumerate() {
                let bits: String = frame.iter().map(|&b| if b { '1' } else { '0' }).collect();
                println!("    t={t}: {bits}");
            }
        }
        PdrResult::Exhausted { frames_explored } => {
            println!("  EXHAUSTED: explored {frames_explored} frames");
        }
    }
}
