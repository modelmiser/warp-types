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
//! TODO(tier-2 follow-up): wire `batsat` and `splr` comparison
//! passes. Do NOT guess their APIs — verify current versions and
//! public surfaces on docs.rs, then add `dev-dependencies` to
//! Cargo.toml and matching `bench_batsat` / `bench_splr` functions
//! below, gated behind a `compare` feature flag so the default
//! bench run still works without the extra deps.

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

fn bench_our_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_3sat/warp-types-sat");
    for &n in &[50u32, 75, 100] {
        let cnf = generate_random_3cnf(n, 0xDEADBEEF);
        group.bench_with_input(BenchmarkId::from_parameter(n), &cnf, |b, cnf| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
                let _ = solver::solve(black_box(inst.db), black_box(inst.num_vars));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_our_solver);
criterion_main!(benches);
