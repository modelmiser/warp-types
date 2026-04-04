#![allow(dead_code)] // Bench utilities only used in #[cfg(test)]

//! Benchmarking: scheduled vs basic BCP.
//!
//! Generates random 3-SAT instances at the phase transition ratio (~4.267 clauses/var)
//! and measures propagation throughput for both BCP implementations.
//!
//! Not a criterion benchmark — a simple timing harness that prints results.
//! Run with: `cargo test -p warp-types-sat --release bench_ -- --nocapture`

use crate::bcp::{self, BcpResult, ClauseDb};
use crate::literal::Lit;
use crate::scheduler;
use crate::session;
use crate::trail::Trail;
use std::time::Instant;

// ============================================================================
// Random 3-SAT generator
// ============================================================================

/// Simple LCG random number generator (deterministic, no dependency).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // LCG from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u32() as usize) % bound
    }

    fn next_bool(&mut self) -> bool {
        self.next_u32() & 1 == 1
    }
}

/// Generate a random k-SAT instance.
///
/// `num_vars`: number of variables
/// `num_clauses`: number of clauses
/// `k`: literals per clause
/// `seed`: deterministic seed
pub fn generate_k_sat(num_vars: u32, num_clauses: usize, k: usize, seed: u64) -> ClauseDb {
    assert!(
        k <= num_vars as usize,
        "k={k} exceeds num_vars={num_vars} — cannot pick {k} distinct variables from {num_vars}"
    );
    let mut rng = Rng::new(seed);
    let mut db = ClauseDb::new();

    for _ in 0..num_clauses {
        let mut lits = Vec::with_capacity(k);
        let mut used_vars = Vec::with_capacity(k);
        for _ in 0..k {
            // Avoid duplicate variables within a clause (prevents tautologies
            // like x ∨ ¬x and duplicate literals like x ∨ x).
            let mut var = rng.next_usize(num_vars as usize) as u32;
            while used_vars.contains(&var) {
                var = rng.next_usize(num_vars as usize) as u32;
            }
            used_vars.push(var);
            let lit = if rng.next_bool() {
                Lit::pos(var)
            } else {
                Lit::neg(var)
            };
            lits.push(lit);
        }
        db.add_clause(lits);
    }

    db
}

/// Generate a random 3-SAT instance at the phase transition ratio.
pub fn generate_3sat_phase_transition(num_vars: u32, seed: u64) -> ClauseDb {
    let num_clauses = ((num_vars as f64) * 4.267).ceil() as usize;
    generate_k_sat(num_vars, num_clauses, 3, seed)
}

// ============================================================================
// Benchmark runner
// ============================================================================

/// Result of a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub num_vars: u32,
    pub num_clauses: usize,
    pub elapsed_us: u128,
    pub propagated: usize,
    pub is_conflict: bool,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let outcome = if self.is_conflict { "CONFLICT" } else { "OK" };
        write!(
            f,
            "{:<20} vars={:<6} cls={:<6} time={:<8}µs props={:<6} [{}]",
            self.name, self.num_vars, self.num_clauses, self.elapsed_us, self.propagated, outcome
        )
    }
}

/// Run basic BCP and return timing.
fn bench_basic_bcp(db: &ClauseDb, num_vars: usize) -> BenchResult {
    let mut trail = Trail::new(num_vars);
    trail.new_decision(Lit::pos(0));
    let before = trail.len();
    let start = Instant::now();

    let result = session::with_session(|s| {
        let p = s.decide().propagate();
        bcp::run_bcp(db, &mut trail, &p)
    });

    let elapsed = start.elapsed();
    let propagated = trail.len() - before;
    let is_conflict = matches!(result, BcpResult::Conflict { .. });

    BenchResult {
        name: "basic".to_string(),
        num_vars: num_vars as u32,
        num_clauses: db.len(),
        elapsed_us: elapsed.as_micros(),
        propagated,
        is_conflict,
    }
}

/// Run scheduled BCP and return timing.
fn bench_scheduled_bcp(db: &ClauseDb, num_vars: usize) -> BenchResult {
    let mut trail = Trail::new(num_vars);
    trail.new_decision(Lit::pos(0));
    let before = trail.len();
    let start = Instant::now();

    let result = session::with_session(|s| {
        let p = s.decide().propagate();
        scheduler::run_bcp_scheduled(db, &mut trail, &p)
    });

    let elapsed = start.elapsed();
    let propagated = trail.len() - before;
    let is_conflict = matches!(result, BcpResult::Conflict { .. });

    BenchResult {
        name: "scheduled".to_string(),
        num_vars: num_vars as u32,
        num_clauses: db.len(),
        elapsed_us: elapsed.as_micros(),
        propagated,
        is_conflict,
    }
}

/// Run comparison benchmark at a given size.
pub fn bench_comparison(num_vars: u32, seed: u64) -> (BenchResult, BenchResult) {
    let db = generate_3sat_phase_transition(num_vars, seed);

    let basic = bench_basic_bcp(&db, num_vars as usize);
    let scheduled = bench_scheduled_bcp(&db, num_vars as usize);

    (basic, scheduled)
}

/// Run the full benchmark suite across sizes.
pub fn bench_suite() -> Vec<(BenchResult, BenchResult)> {
    let sizes = [20, 50, 100, 200, 500, 1000];
    let mut results = Vec::new();

    for &n in &sizes {
        // Average over 3 seeds for stability
        let mut basic_total = 0u128;
        let mut sched_total = 0u128;
        let mut last_basic = None;
        let mut last_sched = None;

        for seed in 0..3 {
            let (b, s) = bench_comparison(n, seed);
            basic_total += b.elapsed_us;
            sched_total += s.elapsed_us;
            last_basic = Some(b);
            last_sched = Some(s);
        }

        let mut b = last_basic.unwrap();
        let mut s = last_sched.unwrap();
        b.elapsed_us = basic_total / 3;
        s.elapsed_us = sched_total / 3;
        results.push((b, s));
    }

    results
}

// ============================================================================
// Tests (run with --release --nocapture for meaningful timing)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_3sat_smoke() {
        let db = generate_3sat_phase_transition(50, 42);
        // ~213 clauses for 50 vars at 4.267 ratio
        assert!(db.len() > 200);
        assert!(db.len() < 220);
    }

    #[test]
    fn basic_and_scheduled_agree() {
        for seed in 0..5 {
            let db = generate_3sat_phase_transition(30, seed);

            let mut trail1 = Trail::new(30);
            trail1.new_decision(Lit::pos(0));
            let mut trail2 = Trail::new(30);
            trail2.new_decision(Lit::pos(0));

            let r1 = session::with_session(|s| {
                let p = s.decide().propagate();
                bcp::run_bcp(&db, &mut trail1, &p)
            });
            let r2 = session::with_session(|s| {
                let p = s.decide().propagate();
                scheduler::run_bcp_scheduled(&db, &mut trail2, &p)
            });

            assert_eq!(r1, r2, "seed {seed}: basic and scheduled disagree");
            assert_eq!(
                trail1.assignments(),
                trail2.assignments(),
                "seed {seed}: assignments diverge"
            );
        }
    }

    #[test]
    fn bench_small() {
        let (b, s) = bench_comparison(50, 42);
        println!("\n{b}");
        println!("{s}");
        // Both should complete (no panic) and agree on conflict/no-conflict
        assert_eq!(b.is_conflict, s.is_conflict);
    }

    #[test]
    fn bench_suite_runs() {
        // Full suite — prints comparison table
        // Run with: cargo test -p warp-types-sat --release bench_suite -- --nocapture
        let results = bench_suite();
        println!("\n=== BCP Benchmark: basic vs scheduled ===");
        println!(
            "{:<8} {:<8} {:>10} {:>10} {:>8}",
            "vars", "clauses", "basic(µs)", "sched(µs)", "ratio"
        );
        println!("{}", "-".repeat(50));
        for (b, s) in &results {
            let ratio = if s.elapsed_us > 0 {
                b.elapsed_us as f64 / s.elapsed_us as f64
            } else {
                f64::NAN
            };
            println!(
                "{:<8} {:<8} {:>10} {:>10} {:>8.2}x",
                b.num_vars, b.num_clauses, b.elapsed_us, s.elapsed_us, ratio
            );
        }
    }

    #[test]
    fn implication_chain_scaling() {
        // Pure implication chain: x0 → x1 → ... → xN
        // Best case for BCP — every clause becomes unit sequentially
        println!("\n=== Implication Chain Scaling ===");
        println!(
            "{:<8} {:>10} {:>10} {:>8}",
            "length", "basic(µs)", "sched(µs)", "ratio"
        );
        println!("{}", "-".repeat(42));

        for &n in &[10, 50, 100, 500, 1000] {
            let mut db = ClauseDb::new();
            for i in 0..n {
                db.add_clause(vec![Lit::neg(i), Lit::pos(i + 1)]);
            }

            let num = (n + 1) as usize;
            let mut trail_b = Trail::new(num);
            trail_b.new_decision(Lit::pos(0));
            let mut trail_s = Trail::new(num);
            trail_s.new_decision(Lit::pos(0));

            let start = Instant::now();
            session::with_session(|s| {
                let p = s.decide().propagate();
                bcp::run_bcp(&db, &mut trail_b, &p)
            });
            let basic_us = start.elapsed().as_micros();

            let start = Instant::now();
            session::with_session(|s| {
                let p = s.decide().propagate();
                scheduler::run_bcp_scheduled(&db, &mut trail_s, &p)
            });
            let sched_us = start.elapsed().as_micros();

            let ratio = if sched_us > 0 {
                basic_us as f64 / sched_us as f64
            } else {
                f64::NAN
            };
            println!("{:<8} {:>10} {:>10} {:>8.2}x", n, basic_us, sched_us, ratio);

            assert_eq!(trail_b.assignments(), trail_s.assignments());
        }
    }
}
