//! aig-witness — turn a `pdr --invariant` file into a Certifaiger witness circuit.
//!
//! Usage: aig-witness MODEL.aig INVARIANT.inv WITNESS.aag
//!
//! The witness is the model with its bad signal replaced by `¬(Inv ∧ P)`;
//! check it with `certifaiger MODEL.aig WITNESS.aag` (see
//! `warp_types_bmc::aiger::write_witness`).
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();
    let [_, model, inv, out] = args.as_slice() else {
        eprintln!("Usage: aig-witness MODEL.aig INVARIANT.inv WITNESS.aag");
        process::exit(1);
    };
    let bytes = fs::read(model).unwrap_or_else(|e| {
        eprintln!("Error reading {model}: {e}");
        process::exit(1);
    });
    let aig = warp_types_bmc::aiger::parse(&bytes).unwrap_or_else(|e| {
        eprintln!("AIGER error in {model}: {e}");
        process::exit(1);
    });
    let text = fs::read_to_string(inv).unwrap_or_else(|e| {
        eprintln!("Error reading {inv}: {e}");
        process::exit(1);
    });
    let clauses = warp_types_bmc::aiger::parse_invariant_file(&text).unwrap_or_else(|e| {
        eprintln!("Invariant error in {inv}: {e}");
        process::exit(1);
    });
    let name = std::path::Path::new(model)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(model);
    let w = warp_types_bmc::aiger::write_witness(&aig, &clauses, name);
    fs::write(out, &w).unwrap_or_else(|e| {
        eprintln!("Error writing {out}: {e}");
        process::exit(1);
    });
    eprintln!(
        "c witness written to {out}: {} clause(s), {} gates (model had {})",
        clauses.len(),
        w.lines()
            .next()
            .and_then(|h| h.split_whitespace().nth(5))
            .unwrap_or("?"),
        aig.ands.len()
    );
}
