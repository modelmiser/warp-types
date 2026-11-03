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
    // The invariant file names its model by fingerprint (pdr --invariant since
    // 2026-11-03); a mismatch means this witness would certify the wrong
    // circuit, so it is refused. Older files carry none: warned, not refused.
    let fp = warp_types_bmc::aiger::fingerprint(&bytes);
    match warp_types_bmc::aiger::invariant_file_fingerprint(&text) {
        Some(want) if want != fp => {
            eprintln!(
                "Error: {inv} was produced for a model with fnv1a64 {want:016x}, but {model} has {fp:016x}"
            );
            process::exit(1);
        }
        Some(_) => {}
        None => eprintln!(
            "c warning: {inv} records no model fingerprint; cannot confirm it belongs to {model}"
        ),
    }
    let clauses = warp_types_bmc::aiger::parse_invariant_file(&text).unwrap_or_else(|e| {
        eprintln!("Invariant error in {inv}: {e}");
        process::exit(1);
    });
    let name = std::path::Path::new(model)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(model);
    let w =
        warp_types_bmc::aiger::write_witness(&aig, &clauses, &format!("{name} fnv1a64 {fp:016x}"));
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
