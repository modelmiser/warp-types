//! BMC command-line tool.
//!
//! `bmc <file.aig|aag> [--max-depth N] [--budget C] [--trace OUT]` runs bounded model checking
//! on an AIGER safety instance (one bad literal or one output) and prints a
//! verdict line: `s SAT depth=K` (counterexample), `s BOUNDED_SAFE depth=N`,
//! or `s EXHAUSTED depth=K`. `--trace OUT` writes a counterexample as
//! `step aiger-literal value` lines (inputs every step, latches at step 0) for
//! `tools/replay-trace.py`. With no arguments it demonstrates the
//! phase-typed BMC loop on built-in examples.

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

fn run_aiger(args: &[String]) {
    let mut path: Option<&str> = None;
    let mut max_depth: u32 = 1000;
    let mut budget: u64 = 0;
    let mut coi = true;
    let mut trace_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-coi" => coi = false,
            "--trace" => {
                i += 1;
                trace_path = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--max-depth" => {
                i += 1;
                max_depth = args
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
    let lowered = if coi {
        warp_types_bmc::aiger::lower_coi(&aig)
    } else {
        warp_types_bmc::aiger::lower(&aig)
    };
    let (sys, lw) = lowered.unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "c {}: M={} I={} L={} A={} B={} C={} -> {} state vars, {} init / {} transition clauses",
        path,
        aig.max_var,
        aig.inputs.len(),
        aig.latches.len(),
        aig.ands.len(),
        aig.bad.len(),
        aig.constraints.len(),
        sys.num_state_vars(),
        sys.initial().len(),
        sys.transition().len()
    );
    let t = std::time::Instant::now();
    let result = check(&sys, max_depth, budget);
    let secs = t.elapsed().as_secs_f64();
    match result {
        BmcResult::CounterexampleFound { depth, trace } => {
            println!("s SAT depth={depth} time={secs:.3}");
            if let Some(tp) = &trace_path {
                let mut out = String::new();
                warp_types_bmc::aiger::write_trace(&aig, &lw, &trace, path, &mut out);
                std::fs::write(tp, out).unwrap_or_else(|e| {
                    eprintln!("Error writing {tp}: {e}");
                    std::process::exit(1);
                });
                eprintln!("c trace written to {tp}");
            }
        }
        BmcResult::BoundedSafe { max_depth } => {
            println!("s BOUNDED_SAFE depth={max_depth} time={secs:.3}")
        }
        BmcResult::Exhausted { depth } => println!("s EXHAUSTED depth={depth} time={secs:.3}"),
    }
}

fn usage() -> ! {
    eprintln!("Usage: bmc [<file.aig|aag> [--max-depth N] [--budget C] [--trace OUT] [--no-coi]]");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        run_aiger(&args);
        return;
    }
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
