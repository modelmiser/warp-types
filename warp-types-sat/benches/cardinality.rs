//! Cardinality-heavy and parity instances.
//!
//! Purpose: showcase where `warp-types-sat` could plausibly compete —
//! the gradient path (FastFourierSAT lineage) on instances with many
//! cardinality constraints, where CDCL struggles with the combinatorial
//! blowup of the CNF encoding.
//!
//! Two input families:
//!
//! - **Pigeonhole PHP(n)**: n+1 pigeons into n holes, classically hard
//!   for resolution-based CDCL (exponential lower bound). Any time we
//!   beat CDCL here with the gradient solver, that is the pitch.
//!
//! - **Parity XOR chains**: stacked XOR constraints expanded to CNF.
//!   Cardinality-dense, well-behaved for gradient methods.
//!
//! Today this file measures the CDCL path on both input classes. Add
//! `gradient::solve` benchmarks alongside once that public API has
//! stabilized — the point of this file is to justify the gradient-path
//! promotion in the README with real numbers.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use warp_types_sat::dimacs;
use warp_types_sat::solver;

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

// TODO(tier-2 follow-up): add gradient-path benches once the public API
// stabilizes. This is the load-bearing benchmark for deciding whether the
// gradient solver earns README promotion — on pigeonhole in particular,
// any meaningful advantage over the CDCL column is the pitch.
//
//     fn bench_pigeonhole_gradient(c: &mut Criterion) { ... }
//     fn bench_parity_gradient(c: &mut Criterion) { ... }

criterion_group!(benches, bench_pigeonhole_cdcl, bench_parity_cdcl);
criterion_main!(benches);
