//! SATLIB corpus sweep.
//!
//! Auto-discovers `benches/satlib/*.cnf` (non-recursive; drop files
//! flat after `tar --strip-components=1`). Runs warp-types-sat on
//! each; under `--features compare`, also runs `batsat 0.6` and
//! `splr 0.17` for per-instance ratios.
//!
//! If the directory is empty or missing, prints a one-line skip
//! message and returns — `cargo bench` never fails on a missing
//! corpus. See `benches/satlib/README.md` for the fetch URL.
//!
//! By default the first 10 files in lexicographic order are
//! benched. Override with `SATLIB_MAX=<n>` at bench time.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::fs;
use std::path::PathBuf;
use warp_types_sat::dimacs;
use warp_types_sat::solver;

const DEFAULT_MAX_FILES: usize = 10;

fn discover_cnfs() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/satlib");
    let Ok(read_dir) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = read_dir
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("cnf"))
        .collect();
    paths.sort();
    let max = std::env::var("SATLIB_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_FILES);
    paths.truncate(max);
    paths
}

#[cfg(feature = "compare")]
mod peer {
    // Intentional duplication with random_3sat.rs's peer module.
    // Criterion bench targets compile as separate binaries, so a
    // shared `mod common` would require a `#[path = "..."]` import
    // in each target. At ~40 lines per copy and two bench files,
    // duplicate-and-inline is cheaper than the abstraction.

    pub fn dimacs_to_clauses(cnf: &str) -> Vec<Vec<i32>> {
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

    pub fn solve_batsat(cnf: &str) -> bool {
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

    pub fn solve_splr(clauses: &[Vec<i32>]) -> bool {
        use splr::{Certificate, SolverError};
        match Certificate::try_from(clauses.to_vec()) {
            Ok(Certificate::SAT(_)) => true,
            Ok(Certificate::UNSAT) => false,
            Err(SolverError::EmptyClause) => false,
            Err(e) => panic!("splr solver error: {:?}", e),
        }
    }
}

fn bench_satlib(c: &mut Criterion) {
    let files = discover_cnfs();
    if files.is_empty() {
        eprintln!(
            "[satlib_sweep] benches/satlib/ is empty — skipping. \
             See benches/satlib/README.md for the fetch URL."
        );
        return;
    }
    eprintln!("[satlib_sweep] benching {} instance(s)", files.len());

    for path in &files {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let cnf = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[satlib_sweep] skip {} ({})", stem, e);
                continue;
            }
        };

        let mut group = c.benchmark_group(format!("satlib/{}", stem));

        group.bench_function("warp-types-sat", |b| {
            b.iter(|| {
                let inst = dimacs::parse_dimacs_str(&cnf).expect("parse");
                let _ = solver::solve(black_box(inst.db), black_box(inst.num_vars));
            });
        });

        #[cfg(feature = "compare")]
        {
            group.bench_function("batsat", |b| {
                b.iter(|| {
                    let _ = peer::solve_batsat(black_box(cnf.as_str()));
                });
            });

            let clauses = peer::dimacs_to_clauses(&cnf);
            group.bench_function("splr", |b| {
                b.iter(|| {
                    let _ = peer::solve_splr(black_box(&clauses));
                });
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_satlib);
criterion_main!(benches);
