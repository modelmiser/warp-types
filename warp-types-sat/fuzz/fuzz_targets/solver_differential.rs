#![no_main]
//! Two BCP implementations must agree, and a claimed model must be real.
//!
//! ORACLE, in two layers — the second is why this is worth more than a
//! crash-finder:
//!
//! 1. AGREEMENT. `solve` uses the original scan-every-clause BCP; `solve_watched`
//!    uses the two-watched-literal hot loop with raw-pointer compaction and
//!    unchecked arena indexing. They must return the same verdict. This
//!    generalises the ten fixed seeds in `watch::tests::watched_agrees_with_
//!    original_bcp` to arbitrary CNF.
//!
//! 2. MODEL CHECK. Agreement alone cannot catch both being wrong together, and
//!    the two share `ClauseDb`. So any `Sat` verdict is verified against the
//!    clauses directly — this harness built them, so it can check them without
//!    trusting either solver. An UNSAT claim has no cheap witness and is left
//!    to layer 1.
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use warp_types_sat::bcp::ClauseDb;
use warp_types_sat::literal::Lit;
use warp_types_sat::solver::{solve, solve_watched, SolveResult};

#[derive(Arbitrary, Debug)]
struct Cnf {
    vars: u8,
    clauses: Vec<Vec<i8>>,
}

/// Bound the instance hard. Both solvers are exponential in the worst case and
/// `solve`'s BCP is O(clauses) per propagation; the interesting behaviour is in
/// the watch-list bookkeeping, which small instances exercise fully.
fn normalise(c: &Cnf) -> (u32, Vec<Vec<Lit>>) {
    let num_vars = (c.vars % 12) as u32 + 1;
    let clauses = c
        .clauses
        .iter()
        .take(40)
        .map(|raw| {
            raw.iter()
                .take(5)
                .filter(|&&x| x != 0)
                .map(|&x| {
                    let v = (x.unsigned_abs() as u32 - 1) % num_vars;
                    if x < 0 {
                        Lit::neg(v)
                    } else {
                        Lit::pos(v)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    (num_vars, clauses)
}

fn build(clauses: &[Vec<Lit>]) -> ClauseDb {
    let mut db = ClauseDb::new();
    for c in clauses {
        db.add_clause(c.clone());
    }
    db
}

fuzz_target!(|c: Cnf| {
    let (num_vars, clauses) = normalise(&c);

    let r_plain = solve(build(&clauses), num_vars);
    let r_watched = solve_watched(build(&clauses), num_vars);

    let tag = |r: &SolveResult| match r {
        SolveResult::Sat(_) => "sat",
        SolveResult::Unsat => "unsat",
        SolveResult::Unknown => "unknown",
    };
    assert_eq!(
        tag(&r_plain),
        tag(&r_watched),
        "BCP implementations disagree on {clauses:?} (num_vars {num_vars})"
    );

    for r in [&r_plain, &r_watched] {
        if let SolveResult::Sat(assign) = r {
            for cl in &clauses {
                let satisfied = cl.iter().any(|l| {
                    assign
                        .get(l.var() as usize)
                        .is_some_and(|&v| v != l.is_negated())
                });
                assert!(
                    satisfied,
                    "claimed model leaves {cl:?} unsatisfied (assign {assign:?})"
                );
            }
        }
    }
});
