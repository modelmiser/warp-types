#![no_main]
//! `parse_dimacs_str` on arbitrary input, then hand the result to the solver.
//!
//! ORACLE — a cross-boundary one, not "didn't crash". The parser is total over
//! `&str`: every input either parses or returns `DimacsError`, so a panic
//! inside the parser is a finding. But the sharper property is what happens
//! AFTER: `solve_watched_budget` asserts `db.max_variable() < num_vars` at
//! entry, and that precondition is ESTABLISHED by the parser, three call
//! frames away. If the parser can emit an instance that trips it, a malformed
//! file panics deep in the solver instead of being rejected at the boundary —
//! the precondition would be asserted where it is used and never where it is
//! made. DIMACS is the only input this crate takes from outside itself.
//!
//! A tiny conflict budget keeps each case fast; the point is reaching the
//! entry asserts and the arena decoding, not solving anything hard.
use libfuzzer_sys::fuzz_target;
use warp_types_sat::dimacs::parse_dimacs_str;
use warp_types_sat::solver::solve_watched_budget;

fuzz_target!(|data: &str| {
    if let Ok(inst) = parse_dimacs_str(data) {
        // Guard the fuzzer against its own success: a header can declare a
        // huge var count, and allocating trail arrays for it is a timeout,
        // not a bug.
        if inst.num_vars > 2_000 || inst.db.len() > 5_000 {
            return;
        }
        let _ = solve_watched_budget(inst.db, inst.num_vars, 50);
    }
});
