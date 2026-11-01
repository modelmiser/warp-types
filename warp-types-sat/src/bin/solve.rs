//! CLI entry point: solve a DIMACS CNF file.
//!
//! Usage: cargo run -p warp-types-sat --bin solve -- [--drat <proof.drat>] [--reduce-interval <n>] <file.cnf>
//!
//! `--drat FILE` writes a DRAT proof of an UNSATISFIABLE answer to FILE
//! (checkable with `drat-trim FILE.cnf FILE.drat`). On SATISFIABLE the file is
//! still written but proves nothing. See `warp_types_sat::proof`.
//!
//! `--reduce-interval N` sets the conflicts between learned-clause reductions
//! (default 2000). A diagnostic: a small N makes short solves exercise clause
//! deletion, which is what proof-logging coverage needs. Honoured with and
//! without `--drat`, so timings of the two are like-for-like.
//!
//! Note: loads the entire file into memory via `fs::read_to_string`.
//! Not suitable for adversarial input without external size limits.

use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::process;

fn usage(prog: &str) -> ! {
    eprintln!("Usage: {prog} [--drat <proof.drat>] [--reduce-interval <n>] <file.cnf>");
    process::exit(1);
}

/// True when `proof` names the same file as `cnf`. The proof file may not
/// exist yet, so compare the canonical parent directory plus file name.
fn same_file(cnf: &str, proof: &str) -> bool {
    use std::path::Path;
    let canon = |p: &str| -> Option<std::path::PathBuf> {
        let p = Path::new(p);
        let parent = p
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        Some(fs::canonicalize(parent).ok()?.join(p.file_name()?))
    };
    match (canon(cnf), canon(proof)) {
        (Some(a), Some(b)) => a == b,
        _ => cnf == proof,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let prog = args.first().map(String::as_str).unwrap_or("solve");
    let mut drat_path: Option<String> = None;
    let mut reduce_interval: u64 = warp_types_sat::solver::DEFAULT_REDUCE_INTERVAL;
    let mut cnf_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--drat" => {
                i += 1;
                if i >= args.len() || drat_path.is_some() {
                    usage(prog);
                }
                drat_path = Some(args[i].clone());
            }
            "--reduce-interval" => {
                i += 1;
                reduce_interval = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .filter(|&n: &u64| n > 0)
                    .unwrap_or_else(|| usage(prog));
            }
            a if a.starts_with("--") => usage(prog),
            a => {
                if cnf_path.is_some() {
                    usage(prog);
                }
                cnf_path = Some(a.to_string());
            }
        }
        i += 1;
    }
    let Some(cnf_path) = cnf_path else {
        usage(prog)
    };
    if let Some(d) = &drat_path {
        if same_file(&cnf_path, d) {
            eprintln!("Error: --drat path {d} is the input CNF; refusing to overwrite it");
            process::exit(1);
        }
    }

    let contents = fs::read_to_string(&cnf_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", cnf_path, e);
        process::exit(1);
    });

    let instance = warp_types_sat::dimacs::parse_dimacs_str(&contents).unwrap_or_else(|e| {
        eprintln!("Parse error: {}", e);
        process::exit(1);
    });

    eprintln!(
        "c {} vars, {} clauses",
        instance.num_vars, instance.num_clauses,
    );

    let result = match drat_path {
        None => warp_types_sat::solver::solve_watched_with(
            instance.db,
            instance.num_vars,
            reduce_interval,
        ),
        Some(path) => {
            let file = fs::File::create(&path).unwrap_or_else(|e| {
                eprintln!("Error creating {}: {}", path, e);
                process::exit(1);
            });
            let mut out = BufWriter::new(file);
            let r = warp_types_sat::solver::solve_watched_drat_with(
                instance.db,
                instance.num_vars,
                &mut out,
                reduce_interval,
            )
            .and_then(|r| out.flush().map(|_| r))
            .unwrap_or_else(|e| {
                eprintln!("Error writing proof to {}: {}", path, e);
                process::exit(1);
            });
            eprintln!("c DRAT proof written to {}", path);
            r
        }
    };

    match result {
        warp_types_sat::solver::SolveResult::Sat(assign) => {
            println!("s SATISFIABLE");
            print!("v");
            for (i, &val) in assign.iter().enumerate() {
                let lit = if val {
                    (i + 1) as i32
                } else {
                    -((i + 1) as i32)
                };
                print!(" {}", lit);
            }
            println!(" 0");
        }
        warp_types_sat::solver::SolveResult::Unsat => {
            println!("s UNSATISFIABLE");
        }
        warp_types_sat::solver::SolveResult::Unknown => {
            println!("s UNKNOWN");
        }
    }
}
