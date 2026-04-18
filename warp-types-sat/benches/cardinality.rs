//! Cardinality-heavy and parity instances.
//!
//! Purpose: showcase where `warp-types-sat` could plausibly compete —
//! the gradient path (FastFourierSAT lineage) on instances with many
//! cardinality constraints, where CDCL struggles with the combinatorial
//! blowup of the CNF encoding.
//!
//! Three columns per input family:
//!
//! - **`/cdcl`** — pure `solver::solve` (CDCL).
//! - **`/gradient`** — pure `gradient::gradient_search` with the
//!   `enhanced` preset (momentum + clause-weight EMA).
//! - **`/hybrid`** — `gradient::hybrid_solve`, which seeds VSIDS phase
//!   hints + initial activities from gradient confidence, then runs CDCL.
//!
//! Two input families:
//!
//! - **Pigeonhole PHP(n)**: n+1 pigeons into n holes. Classically hard
//!   for resolution-based CDCL (exponential lower bound). Pure gradient
//!   runs its budget without finding a solution (there isn't one), so
//!   the measurement is "gradient gives up on a bounded budget" vs.
//!   "CDCL takes exponentially long to prove UNSAT." Any time we beat
//!   CDCL here with the gradient or hybrid column, that is the pitch.
//!
//! - **Parity XOR chains**: stacked `a ≠ b` constraints. SAT. Measures
//!   whether gradient converges to a satisfying alternating assignment
//!   faster than CDCL.
//!
//! The gradient config is deliberately sized for bounded bench runtime
//! (`num_starts=8, max_iters=100`) rather than the library default
//! (`32, 1000`). If real performance comparison needs different
//! settings, tune in `bench_gradient_config`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use warp_types_sat::dimacs;
use warp_types_sat::gradient;
use warp_types_sat::solver;

/// Bench-sized gradient config: bounded runtime, still exercises the
/// `enhanced` features (momentum, clause-weight EMA).
fn bench_gradient_config() -> gradient::GradientConfig {
    gradient::GradientConfig {
        num_starts: 8,
        max_iters: 100,
        ..gradient::GradientConfig::enhanced()
    }
}

/// Pigeonhole CNF: "n+1 pigeons into n holes".
/// Variables x_{i,j} = pigeon i is in hole j (1-indexed).
fn pigeonhole(n: u32) -> String {
    let pigeons = n + 1;
    let holes = n;
    let var = |i: u32, j: u32| -> i32 { ((i - 1) * holes + j) as i32 };

    let mut clauses: Vec<Vec<i32>> = Vec::new();
    for i in 1..=pigeons {
        clauses.push((1..=holes).map(|j| var(i, j)).collect());
    }
    for j in 1..=holes {
        for i in 1..=pigeons {
            for k in (i + 1)..=pigeons {
                clauses.push(vec![-var(i, j), -var(k, j)]);
            }
        }
    }

    let num_vars = pigeons * holes;
    let mut out = format!("p cnf {} {}\n", num_vars, clauses.len());
    for c in &clauses {
        for l in c {
            out.push_str(&format!("{} ", l));
        }
        out.push_str("0\n");
    }
    out
}

/// Parity chain: k XOR-2 constraints in CNF (each XOR = 2 clauses for XOR-2).
fn parity_chain(k: u32) -> String {
    let num_vars = k + 1;
    let mut clauses: Vec<Vec<i32>> = Vec::new();
    for i in 1..=k {
        let a = i as i32;
        let b = (i + 1) as i32;
        clauses.push(vec![a, b]);
        clauses.push(vec![-a, -b]);
    }
    let mut out = format!("p cnf {} {}\n", num_vars, clauses.len());
    for c in &clauses {
        for l in c {
            out.push_str(&format!("{} ", l));
        }
        out.push_str("0\n");
    }
    out
}

// ─── Pigeonhole ────────────────────────────────────────────────────────

fn bench_pigeonhole_cdcl(c: &mut Criterion) {
    let mut group = c.benchmark_group("pigeonhole/cdcl");
    for &n in &[4u32, 5, 6] {
        let cnf = pigeonhole(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &cnf, |b, cnf| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
                let _ = solver::solve(black_box(inst.db), black_box(inst.num_vars));
            });
        });
    }
    group.finish();
}

fn bench_pigeonhole_gradient(c: &mut Criterion) {
    let cfg = bench_gradient_config();
    let mut group = c.benchmark_group("pigeonhole/gradient");
    for &n in &[4u32, 5, 6] {
        let cnf = pigeonhole(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &cnf, |b, cnf| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
                let _ = gradient::gradient_search(
                    black_box(&inst.db),
                    black_box(inst.num_vars),
                    &cfg,
                );
            });
        });
    }
    group.finish();
}

fn bench_pigeonhole_hybrid(c: &mut Criterion) {
    let cfg = bench_gradient_config();
    let mut group = c.benchmark_group("pigeonhole/hybrid");
    for &n in &[4u32, 5, 6] {
        let cnf = pigeonhole(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &cnf, |b, cnf| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
                let _ = gradient::hybrid_solve(
                    black_box(inst.db),
                    black_box(inst.num_vars),
                    &cfg,
                );
            });
        });
    }
    group.finish();
}

// ─── Parity ────────────────────────────────────────────────────────────

fn bench_parity_cdcl(c: &mut Criterion) {
    let mut group = c.benchmark_group("parity/cdcl");
    for &k in &[10u32, 20, 40] {
        let cnf = parity_chain(k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &cnf, |b, cnf| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
                let _ = solver::solve(black_box(inst.db), black_box(inst.num_vars));
            });
        });
    }
    group.finish();
}

fn bench_parity_gradient(c: &mut Criterion) {
    let cfg = bench_gradient_config();
    let mut group = c.benchmark_group("parity/gradient");
    for &k in &[10u32, 20, 40] {
        let cnf = parity_chain(k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &cnf, |b, cnf| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
                let _ = gradient::gradient_search(
                    black_box(&inst.db),
                    black_box(inst.num_vars),
                    &cfg,
                );
            });
        });
    }
    group.finish();
}

fn bench_parity_hybrid(c: &mut Criterion) {
    let cfg = bench_gradient_config();
    let mut group = c.benchmark_group("parity/hybrid");
    for &k in &[10u32, 20, 40] {
        let cnf = parity_chain(k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &cnf, |b, cnf| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(cnf).expect("parse");
                let _ = gradient::hybrid_solve(
                    black_box(inst.db),
                    black_box(inst.num_vars),
                    &cfg,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pigeonhole_cdcl,
    bench_pigeonhole_gradient,
    bench_pigeonhole_hybrid,
    bench_parity_cdcl,
    bench_parity_gradient,
    bench_parity_hybrid,
);
criterion_main!(benches);
