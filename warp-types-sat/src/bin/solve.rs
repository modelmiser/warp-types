//! CLI entry point: solve a DIMACS CNF file.
//!
//! Usage: cargo run -p warp-types-sat --bin solve -- <file.cnf>
//!
//! Note: loads the entire file into memory via `fs::read_to_string`.
//! Not suitable for adversarial input without external size limits.

use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file.cnf>", args[0]);
        process::exit(1);
    }

    let contents = fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", args[1], e);
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

    match warp_types_sat::solver::solve(instance.db, instance.num_vars) {
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
    }
}
