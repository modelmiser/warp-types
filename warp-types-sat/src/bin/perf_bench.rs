//! Focused benchmark for `perf stat` measurement.
//!
//! Solves a single 500v random 3-SAT instance (seed 1, ratio 3.5).
//! Run with: taskset -c 16 perf stat -e <events> target/release/deps/perf_bench-*
//!
//! Or: cargo build --release -p warp-types-sat --bin perf_bench && \
//!     taskset -c 16 perf stat -e cpu_atom/instructions/,cpu_atom/cycles/,cpu_atom/cache-misses/,cpu_atom/branch-misses/ \
//!     target/release/perf_bench

use warp_types_sat::bench::generate_k_sat;
use warp_types_sat::solver::{solve_watched_stats, SolveResult};

fn main() {
    let num_vars: u32 = 500;
    let ratio: f64 = 3.5;
    let seed: u64 = 1;
    let num_clauses = ((num_vars as f64) * ratio).ceil() as usize;

    let db = generate_k_sat(num_vars, num_clauses, 3, seed);
    let (result, stats) = solve_watched_stats(db, num_vars);

    let tag = match result {
        SolveResult::Sat(_) => "SAT",
        SolveResult::Unsat => "UNSAT",
    };
    eprintln!(
        "{tag}: {conf} conflicts, {props} propagations, {dec} decisions",
        conf = stats.conflicts,
        props = stats.propagations,
        dec = stats.decisions,
    );
    eprintln!(
        "bcp={bcp}ns analyze={ana}ns vsids={vs}ns",
        bcp = stats.bcp_ns,
        ana = stats.analyze_ns,
        vs = stats.vsids_ns,
    );
}
