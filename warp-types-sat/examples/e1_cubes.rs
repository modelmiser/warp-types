//! E1 runner — least-confident cubes on `hybrid_solve` (pre-registered 2026-11-01,
//! `research/gpu-sat-partition.md` §3). Prints one CSV row per (instance, method).
//!
//! Primary metric is TOTAL CONFLICTS (deterministic); wall time is secondary.
//! Everything runs single-threaded; cubes are solved sequentially on fresh
//! clause databases (no lemma sharing).
//!
//! Run: `cargo run -p warp-types-sat --release --example e1_cubes > e1.csv`
//! Env:  E1_SEEDS (default 20), E1_PHP_MAX (default 7), E1_N (default "100,150,200")
//!       E1_D (default "3,5,7"), E1_SKIP_CDCL_PHP (default unset; set to skip pure CDCL on PHP≥7)

use std::time::Instant;
use warp_types_sat::bench::{generate_3sat_phase_transition, parity_chain_dimacs, pigeonhole_dimacs};
use warp_types_sat::bcp::ClauseDb;
use warp_types_sat::dimacs::parse_dimacs_str;
use warp_types_sat::gradient::{self, CubeSelector, GradientConfig};
use warp_types_sat::solver::{self, SolveResult};

fn env_list(name: &str, default: &str) -> Vec<u32> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn tuned() -> GradientConfig {
    GradientConfig { num_starts: 8, max_iters: 100, ..GradientConfig::enhanced() }
}

struct Instance {
    name: String,
    cnf: String,
    num_vars: u32,
}

fn parse(cnf: &str) -> (ClauseDb, u32) {
    let inst = parse_dimacs_str(cnf).expect("parse");
    (inst.db, inst.num_vars)
}

fn tag(r: &SolveResult) -> &'static str {
    match r {
        SolveResult::Sat(_) => "SAT",
        SolveResult::Unsat => "UNSAT",
        SolveResult::Unknown => "UNKNOWN",
    }
}

fn main() {
    let seeds = env_list("E1_SEEDS", "20").first().copied().unwrap_or(20) as u64;
    let php_max = env_list("E1_PHP_MAX", "7").first().copied().unwrap_or(7);
    let ns = env_list("E1_N", "100,150,200");
    let ds: Vec<usize> = env_list("E1_D", "3,5,7").into_iter().map(|d| d as usize).collect();
    let skip_cdcl_php = std::env::var("E1_SKIP_CDCL_PHP").is_ok();

    let mut instances: Vec<Instance> = Vec::new();
    for n in 4..=php_max {
        instances.push(Instance { name: format!("php{n}"), cnf: pigeonhole_dimacs(n), num_vars: (n + 1) * n });
    }
    for k in [10u32, 20, 40] {
        instances.push(Instance { name: format!("parity{k}"), cnf: parity_chain_dimacs(k), num_vars: k + 1 });
    }
    for &n in &ns {
        for seed in 0..seeds {
            let db = generate_3sat_phase_transition(n, seed);
            instances.push(Instance { name: format!("r3sat_n{n}_s{seed}"), cnf: db_to_dimacs(&db, n), num_vars: n });
        }
    }

    println!("instance,num_vars,class,config,method,d,selector,result,conflicts,decisions,propagations,cubes_solved,cubes_unsat,gradient_sat,wall_ms");
    for inst in &instances {
        // Pure CDCL baseline — also fixes the instance CLASS (SAT/UNSAT) used by the analysis.
        let (class, cdcl_row) = if skip_cdcl_php && inst.name.starts_with("php") && inst.num_vars >= 56 {
            ("UNSAT".to_string(), None)
        } else {
            let (db, n) = parse(&inst.cnf);
            let t = Instant::now();
            let (r, s) = solver::solve_watched_stats(db, n);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            (tag(&r).to_string(), Some((tag(&r), s.conflicts, s.decisions, s.propagations, ms)))
        };
        if let Some((r, c, d, p, ms)) = cdcl_row {
            println!("{},{},{},none,cdcl,0,none,{},{},{},{},1,{},false,{:.3}", inst.name, inst.num_vars, class, r, c, d, p, (r == "UNSAT") as u8, ms);
        }
        for (cname, cfg) in [("tuned", tuned()), ("default", GradientConfig::default())] {
            // No-cube hybrid baseline.
            let (db, n) = parse(&inst.cnf);
            let t = Instant::now();
            let (r, s) = gradient::hybrid_solve_stats(db, n, &cfg);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!("{},{},{},{},hybrid,0,none,{},{},{},{},1,{},{},{:.3}", inst.name, inst.num_vars, class, cname, tag(&r), s.conflicts, s.decisions, s.propagations, (tag(&r) == "UNSAT") as u8, s.conflicts == 0 && tag(&r) == "SAT", ms);
            // Cubes: hypothesis + two controls, every d.
            for &d in &ds {
                for (sname, sel) in [
                    ("least", CubeSelector::LeastConfident),
                    ("most", CubeSelector::MostConfident),
                    ("random", CubeSelector::Random { seed: 0xE1 }),
                ] {
                    let (db, n) = parse(&inst.cnf);
                    let t = Instant::now();
                    let (r, st) = gradient::hybrid_solve_cubes(&db, n, &cfg, d, sel);
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    println!("{},{},{},{},cubes,{},{},{},{},{},{},{},{},{},{:.3}", inst.name, inst.num_vars, class, cname, d, sname, tag(&r), st.total_conflicts, st.total_decisions, st.total_propagations, st.cubes_solved, st.cubes_unsat, st.gradient_sat, ms);
                }
            }
        }
    }
}

/// Serialize a generated `ClauseDb` to DIMACS so every method parses the same text.
fn db_to_dimacs(db: &ClauseDb, num_vars: u32) -> String {
    warp_types_sat::bench::db_to_dimacs(db, num_vars)
}
