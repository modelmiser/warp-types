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

fn run_aiger(args: &[String]) {
    let mut path: Option<&str> = None;
    let mut inv_path: Option<&str> = None;
    let mut max_frames: u32 = 1000;
    let mut budget: u64 = 0;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--max-frames" => {
                i += 1;
                max_frames = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--budget" => {
                i += 1;
                budget = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--invariant" => {
                i += 1;
                inv_path = Some(args.get(i).map(String::as_str).unwrap_or_else(|| usage()));
            }
            a if a.starts_with("--") => usage(),
            a => {
                if path.is_some() {
                    usage();
                }
                path = Some(a);
            }
        }
        i += 1;
    }
    let Some(path) = path else { usage() };
    let aig = warp_types_bmc::aiger::read_file(path).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    let (sys, lw) = warp_types_bmc::aiger::lower(&aig).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "c {}: M={} I={} L={} A={} -> {} state vars",
        path,
        aig.max_var,
        aig.inputs.len(),
        aig.latches.len(),
        aig.ands.len(),
        sys.num_state_vars()
    );
    let t = std::time::Instant::now();
    let result = check(&sys, max_frames, budget);
    let secs = t.elapsed().as_secs_f64();
    match &result {
        PdrResult::Safe {
            invariant_frame,
            invariant,
        } => {
            println!(
                "s UNSAT frame={invariant_frame} clauses={} time={secs:.3}",
                invariant.len()
            );
            if let Some(out) = inv_path {
                // The inductive invariant is F_k ∧ P: write the frame's
                // clauses and the property clause, as AIGER literals
                // (2*index + negation; the constant-true variable is literal 1),
                // one clause per line, 0-terminated.
                let mut text = String::new();
                text.push_str(&format!(
                    "c inductive invariant from warp-types-pdr for {path}: frame {invariant_frame}, \
                     {} clauses + property; AIGER literals, 0-terminated\n",
                    invariant.len()
                ));
                let mut clauses: Vec<&Vec<_>> = invariant.iter().collect();
                let prop: Vec<Vec<_>> = sys.property().iter().map(|c| c.lits.clone()).collect();
                clauses.extend(prop.iter());
                for c in clauses {
                    for &l in c {
                        text.push_str(&lw.aiger_lit(l).to_string());
                        text.push(' ');
                    }
                    text.push_str("0\n");
                }
                std::fs::write(out, text).unwrap_or_else(|e| {
                    eprintln!("Error writing {out}: {e}");
                    std::process::exit(1);
                });
                eprintln!("c invariant written to {out}");
            }
        }
        PdrResult::CounterexampleFound { depth, .. } => {
            println!("s SAT depth={depth} time={secs:.3}")
        }
        PdrResult::Exhausted { frames_explored } => {
            println!("s EXHAUSTED frames={frames_explored} time={secs:.3}")
        }
    }
}

fn usage() -> ! {
    eprintln!("Usage: pdr [<file.aig|aag> [--max-frames N] [--budget C] [--invariant OUT]]");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        run_aiger(&args);
        return;
    }
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
        PdrResult::Safe {
            invariant_frame,
            invariant,
        } => {
            println!(
                "  SAFE: inductive invariant at frame {invariant_frame} ({} learned clauses)",
                invariant.len()
            );
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
