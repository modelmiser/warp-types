//! lrat-trim — backward-trim an LRAT proof (text or binary in, text or binary out).
//!
//! Usage: lrat-trim IN.lrat OUT.lrat [--binary]
//!
//! Keeps only the lemmas the first empty clause depends on, and the deletions
//! of kept clauses; ids are unchanged, so the output verifies with the same
//! checkers as the input. No propagation is done here — LRAT names every
//! antecedent already (this is `drat-trim`'s backward pass without its
//! search; see `warp_types_sat::lrat::trim`).
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();
    let (input, output, binary) = match args.as_slice() {
        [_, i, o] => (i, o, false),
        [_, i, o, flag] if flag == "--binary" => (i, o, true),
        _ => {
            eprintln!("Usage: lrat-trim IN.lrat OUT.lrat [--binary]");
            process::exit(1);
        }
    };
    let bytes = fs::read(input).unwrap_or_else(|e| {
        eprintln!("Error reading {input}: {e}");
        process::exit(1);
    });
    let proof = warp_types_sat::lrat::parse_any(&bytes).unwrap_or_else(|e| {
        eprintln!("Parse error in {input}: {e}");
        process::exit(1);
    });
    let (trimmed, st) = warp_types_sat::lrat::trim(&proof);
    let out = if binary {
        warp_types_sat::lrat::write_binary(&trimmed)
    } else {
        warp_types_sat::lrat::write_text(&trimmed).into_bytes()
    };
    fs::write(output, &out).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });
    eprintln!(
        "c trimmed {}: lemmas {} -> {}, deletions {} -> {}, bytes {} -> {}",
        input,
        st.lemmas_in,
        st.lemmas_out,
        st.deletions_in,
        st.deletions_out,
        bytes.len(),
        out.len()
    );
}
