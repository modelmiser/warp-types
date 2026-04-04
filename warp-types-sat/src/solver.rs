//! Top-level CDCL solver.
//!
//! Connects trail, BCP, conflict analysis, and the phase-typed session.
//! The trail is the single source of truth for assignments — BCP writes
//! through it, backtracking retracts through it. No ghost assignments.

use crate::analyze;
use crate::bcp::{self, BcpResult, ClauseDb};
use crate::literal::Lit;
use crate::session;
use crate::trail::Trail;

/// Result of solving a SAT instance.
#[derive(Debug)]
pub enum SolveResult {
    /// Satisfiable. Contains the variable assignment (index = var, value = polarity).
    Sat(Vec<bool>),
    /// Unsatisfiable.
    Unsat,
}

/// Solve a CNF instance.
///
/// Takes ownership of the clause database (learned clauses are appended).
pub fn solve(mut db: ClauseDb, num_vars: u32) -> SolveResult {
    // Empty clause → trivially UNSAT (a disjunction of zero literals is false).
    for i in 0..db.len() {
        if db.clause(i).literals.is_empty() {
            return SolveResult::Unsat;
        }
    }
    // Zero variables: vacuously SAT if no clauses, UNSAT if any clause exists
    // (clauses reference variables that can't be assigned).
    if num_vars == 0 {
        return if db.is_empty() {
            SolveResult::Sat(vec![])
        } else {
            SolveResult::Unsat
        };
    }

    let mut trail = Trail::new(num_vars as usize);

    session::with_session(|initial_session| {
        // ── Initial BCP (Idle → Propagate) ──
        let propagate = initial_session.propagate();
        if let BcpResult::Conflict { .. } = bcp::run_bcp(&db, &mut trail, &propagate) {
            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
            return SolveResult::Unsat;
        }
        let mut idle = propagate.finish_no_conflict();

        // ── Main CDCL loop ──
        loop {
            if trail.all_assigned() {
                let _ = idle.sat();
                return SolveResult::Sat(trail.assignment_vec());
            }

            // ── Decide ──
            let var = pick_variable(trail.assignments());
            trail.new_decision(Lit::pos(var));
            let mut propagate = idle.decide().propagate();
            let mut bcp_result = bcp::run_bcp(&db, &mut trail, &propagate);

            // ── Inner conflict resolution loop ──
            loop {
                match bcp_result {
                    BcpResult::Ok => {
                        idle = propagate.finish_no_conflict();
                        break;
                    }
                    BcpResult::Conflict { clause_index } => {
                        if trail.current_level() == 0 {
                            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                            return SolveResult::Unsat;
                        }

                        let analysis = analyze::analyze_conflict(&trail, &db, clause_index);

                        let conflict = propagate.finish_conflict();
                        let analyzed = conflict.analyze();

                        if analysis.learned.is_empty() {
                            let _ = analyzed.backtrack().unsat();
                            return SolveResult::Unsat;
                        }

                        trail.backtrack_to(analysis.backtrack_level);

                        let asserting_lit = analysis.learned[0];
                        let clause_idx = db.add_clause(analysis.learned);
                        trail.record_propagation(asserting_lit, clause_idx);

                        let bt = analyzed.backtrack();
                        propagate = bt.propagate();
                        bcp_result = bcp::run_bcp(&db, &mut trail, &propagate);
                    }
                }
            }
        }
    })
}

/// Pick the next unassigned variable. Simple sequential scan.
fn pick_variable(assignments: &[Option<bool>]) -> u32 {
    assignments
        .iter()
        .position(|a| a.is_none())
        .expect("pick_variable called with all variables assigned") as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimacs;

    #[test]
    fn trivial_sat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);

        match solve(db, 1) {
            SolveResult::Sat(assign) => assert!(assign[0]),
            SolveResult::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn trivial_unsat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);
        db.add_clause(vec![Lit::neg(0)]);

        match solve(db, 1) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT"),
        }
    }

    #[test]
    fn simple_sat_two_vars() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);

        match solve(db, 2) {
            SolveResult::Sat(assign) => assert!(assign[1]),
            SolveResult::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn pigeonhole_2_1_unsat() {
        let cnf = "p cnf 2 3\n1 0\n2 0\n-1 -2 0\n";
        let inst = dimacs::parse_dimacs_str(cnf).unwrap();
        match solve(inst.db, inst.num_vars) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT"),
        }
    }

    #[test]
    fn pigeonhole_3_2_unsat() {
        let cnf = "\
p cnf 6 9
1 2 0
3 4 0
5 6 0
-1 -3 0
-1 -5 0
-3 -5 0
-2 -4 0
-2 -6 0
-4 -6 0
";
        let inst = dimacs::parse_dimacs_str(cnf).unwrap();
        match solve(inst.db, inst.num_vars) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT"),
        }
    }

    #[test]
    fn satisfiable_3sat() {
        let cnf = "\
p cnf 3 4
1 2 0
-1 3 0
2 3 0
-2 -3 0
";
        let inst = dimacs::parse_dimacs_str(cnf).unwrap();
        match solve(inst.db, inst.num_vars) {
            SolveResult::Sat(_) => {}
            SolveResult::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn empty_clause_is_unsat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![]); // empty clause = trivially false
        match solve(db, 1) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT for empty clause"),
        }
    }

    #[test]
    fn zero_vars_no_clauses_is_sat() {
        let db = ClauseDb::new();
        match solve(db, 0) {
            SolveResult::Sat(assign) => assert!(assign.is_empty()),
            SolveResult::Unsat => panic!("expected SAT for vacuous formula"),
        }
    }

    #[test]
    fn zero_vars_with_clauses_is_unsat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);
        match solve(db, 0) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT for 0-var formula with clauses"),
        }
    }

    #[test]
    fn empty_db_is_sat() {
        let db = ClauseDb::new();
        match solve(db, 5) {
            SolveResult::Sat(_) => {}
            SolveResult::Unsat => panic!("expected SAT for vacuously satisfiable formula"),
        }
    }

    #[test]
    fn large_clause_does_not_panic() {
        // Clause with 64 literals — exceeds tile size (32), should not panic.
        let mut db = ClauseDb::new();
        let lits: Vec<Lit> = (0..64).map(Lit::pos).collect();
        db.add_clause(lits);
        db.add_clause(vec![Lit::neg(0)]); // force x0=false

        match solve(db, 64) {
            SolveResult::Sat(assign) => {
                assert!(!assign[0]); // x0 must be false
                // The 64-lit clause needs at least one true lit — any of x1..x63 works
                assert!(assign[1..].iter().any(|&v| v));
            }
            SolveResult::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn verify_sat_assignment() {
        // Verify the returned assignment actually satisfies every clause.
        let cnf = "\
p cnf 5 10
1 2 3 0
-1 2 4 0
1 -3 5 0
-2 3 -4 0
1 4 5 0
-1 -2 5 0
2 -4 -5 0
-1 3 4 0
1 -2 -3 0
3 4 -5 0
";
        let inst = dimacs::parse_dimacs_str(cnf).unwrap();
        if let SolveResult::Sat(assign) = solve(inst.db, inst.num_vars) {
            let clauses: Vec<Vec<i32>> = vec![
                vec![1, 2, 3],
                vec![-1, 2, 4],
                vec![1, -3, 5],
                vec![-2, 3, -4],
                vec![1, 4, 5],
                vec![-1, -2, 5],
                vec![2, -4, -5],
                vec![-1, 3, 4],
                vec![1, -2, -3],
                vec![3, 4, -5],
            ];
            for clause in &clauses {
                let satisfied = clause.iter().any(|lit: &i32| {
                    let var = (lit.unsigned_abs() - 1) as usize;
                    let pos = *lit > 0;
                    assign[var] == pos
                });
                assert!(
                    satisfied,
                    "clause {:?} not satisfied by {:?}",
                    clause, assign
                );
            }
        }
    }
}
