//! Random 3-SAT at the phase-transition ratio (~4.267).
//!
//! Purpose: smoke-test that our CDCL performance sits in the
//! `batsat`/`splr` range on peer-competitive input. If we're
//! dramatically slower than either, something in BCP or VSIDS
//! regressed. If we're dramatically faster, re-read the review
//! in INSIGHTS.md before celebrating — probably a measurement bug.
//!
//! We do NOT benchmark against Kissat / CaDiCaL here. See
//! `benches/README.md` for why.
//!
//! Peer comparison runs under `--features compare`:
//!   cargo bench -p warp-types-sat --bench random_3sat --features compare

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use warp_types_sat::dimacs;
use warp_types_sat::solver;

/// Deterministic random 3-CNF generator. Seeded so runs are reproducible.
/// Ratio 4.267 is near the satisfiability phase transition for 3-SAT.
fn generate_random_3cnf(num_vars: u32, seed: u64) -> String {
    let ratio = 4.267;
    let num_clauses = (num_vars as f64 * ratio) as u32;

    // Linear congruential RNG — deterministic, adequate for benchmark inputs.
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };

    let mut out = format!("p cnf {} {}\n", num_vars, num_clauses);
    for _ in 0..num_clauses {
        let mut lits: Vec<i32> = Vec::with_capacity(3);
        while lits.len() < 3 {
            let v = (next() % num_vars as u64) as i32 + 1;
            let lit = if next() & 1 == 0 { v } else { -v };
            if !lits.iter().any(|&l| l.abs() == lit.abs()) {
                lits.push(lit);
            }
        }
        for l in lits {
            out.push_str(&format!("{} ", l));
        }
        out.push_str("0\n");
    }
    out
}

const SIZES: &[u32] = &[50, 75, 100];
const SEED: u64 = 0xDEADBEEF;

fn bench_our_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_3sat/warp-types-sat");
    for &n in SIZES {
        let cnf = generate_random_3cnf(n, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &cnf, |b, cnf| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
                let _ = solver::solve(black_box(inst.db), black_box(inst.num_vars));
            });
        });
    }
    group.finish();
}

// -------- peer-solver benches (feature-gated) --------

#[cfg(feature = "compare")]
mod peer {
    use super::*;
    use std::sync::Once;

    /// Convert our DIMACS text into splr-native clauses (Vec<Vec<i32>>).
    /// Our generator emits well-formed `p cnf` + `l1 l2 l3 0\n`, so a
    /// hand-rolled split is adequate here — no need for full DIMACS.
    fn dimacs_to_clauses(cnf: &str) -> Vec<Vec<i32>> {
        let mut out = Vec::new();
        for line in cnf.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('c') || line.starts_with('p') {
                continue;
            }
            let clause: Vec<i32> = line
                .split_whitespace()
                .map(|s| s.parse::<i32>().expect("dimacs int"))
                .take_while(|&l| l != 0)
                .collect();
            if !clause.is_empty() {
                out.push(clause);
            }
        }
        out
    }

    fn solve_warp(cnf: &str) -> bool {
        use warp_types_sat::solver::SolveResult;
        let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
        matches!(solver::solve(inst.db, inst.num_vars), SolveResult::Sat(_))
    }

    fn solve_batsat(cnf: &str) -> bool {
        use batsat::callbacks::Basic;
        use batsat::{dimacs as bdimacs, lbool, Solver, SolverInterface, SolverOpts};
        use std::io::Cursor;
        let mut reader = Cursor::new(cnf.as_bytes());
        let mut s: Solver<Basic> = Solver::new(SolverOpts::default(), Basic::new());
        bdimacs::parse(&mut reader, &mut s, false, false).expect("batsat parse");
        if !s.simplify() {
            return false;
        }
        s.solve_limited(&[]) == lbool::TRUE
    }

    fn solve_splr(clauses: &[Vec<i32>]) -> bool {
        use splr::{Certificate, SolverError};
        match Certificate::try_from(clauses.to_vec()) {
            Ok(Certificate::SAT(_)) => true,
            Ok(Certificate::UNSAT) => false,
            Err(SolverError::EmptyClause) => false,
            Err(e) => panic!("splr solver error: {:?}", e),
        }
    }

    /// Runs once per process — verifies all three solvers return the
    /// same SAT/UNSAT verdict on the benchmark inputs. If this fails,
    /// we've wired an API wrong and any timing numbers are meaningless.
    pub fn check_agreement() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            for &n in SIZES {
                let cnf = generate_random_3cnf(n, SEED);
                let clauses = dimacs_to_clauses(&cnf);
                let w = solve_warp(&cnf);
                let b = solve_batsat(&cnf);
                let sp = solve_splr(&clauses);
                assert!(
                    w == b && b == sp,
                    "solver disagreement at n={}: warp={} batsat={} splr={}",
                    n, w, b, sp
                );
                eprintln!("[agreement] n={}: all three solvers -> {}", n,
                    if w { "SAT" } else { "UNSAT" });
            }
        });
    }

    pub fn bench_batsat(c: &mut Criterion) {
        check_agreement();
        let mut group = c.benchmark_group("random_3sat/batsat");
        for &n in SIZES {
            let cnf = generate_random_3cnf(n, SEED);
            group.bench_with_input(BenchmarkId::from_parameter(n), &cnf, |b, cnf| {
                b.iter(|| {
                    let _ = solve_batsat(black_box(cnf));
                });
            });
        }
        group.finish();
    }

    pub fn bench_splr(c: &mut Criterion) {
        check_agreement();
        let mut group = c.benchmark_group("random_3sat/splr");
        for &n in SIZES {
            let cnf = generate_random_3cnf(n, SEED);
            let clauses = dimacs_to_clauses(&cnf);
            group.bench_with_input(BenchmarkId::from_parameter(n), &clauses, |b, cls| {
                b.iter(|| {
                    // Certificate::try_from consumes the Vec, so we
                    // clone per iteration. At n=100 that's ~427
                    // clauses × 3 i32s — negligible vs CDCL runtime.
                    let _ = solve_splr(black_box(cls));
                });
            });
        }
        group.finish();
    }
}

#[cfg(feature = "compare")]
criterion_group!(benches, bench_our_solver, peer::bench_batsat, peer::bench_splr);
#[cfg(not(feature = "compare"))]
criterion_group!(benches, bench_our_solver);

criterion_main!(benches);
