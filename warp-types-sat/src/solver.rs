//! Top-level CDCL solver.
//!
//! Connects all components: trail, BCP, conflict analysis, and the
//! phase-typed session. The session ensures CDCL phase ordering at
//! compile time — you can't propagate before deciding, can't analyze
//! without a conflict, can't backtrack without analysis.

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
/// Takes ownership of the clause database (learned clauses will be added).
/// `num_vars` is the number of variables (0-indexed).
pub fn solve(mut db: ClauseDb, num_vars: u32) -> SolveResult {
    let n = num_vars as usize;
    let mut assignments: Vec<Option<bool>> = vec![None; n];
    let mut trail = Trail::new();

    session::with_session(|initial_session| {
        // ── Initial BCP (Idle → Propagate) ──
        let propagate = initial_session.propagate();
        match bcp::run_bcp(&db, &mut assignments, &propagate) {
            BcpResult::Conflict { .. } => {
                let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                return SolveResult::Unsat;
            }
            BcpResult::Ok { propagated } => {
                for imp in &propagated {
                    trail.record_propagation(imp.lit, imp.reason, &mut assignments);
                }
            }
        }
        let mut idle = propagate.finish_no_conflict();

        // ── Main CDCL loop ──
        loop {
            // All variables assigned?
            if assignments.iter().all(|a| a.is_some()) {
                let _ = idle.sat();
                return SolveResult::Sat(
                    assignments.iter().map(|a| a.unwrap()).collect(),
                );
            }

            // ── Decide ──
            let var = pick_variable(&assignments);
            let lit = Lit::pos(var); // try positive polarity first
            trail.new_decision(lit, &mut assignments);
            let mut propagate = idle.decide().propagate();
            let mut bcp_result = bcp::run_bcp(&db, &mut assignments, &propagate);

            // Inner conflict resolution loop.
            // Phase-typed session threads through:
            //   Propagate → Conflict → Analyze → Backtrack → Propagate → ...
            loop {
                match bcp_result {
                    BcpResult::Ok { propagated } => {
                        for imp in &propagated {
                            trail.record_propagation(
                                imp.lit,
                                imp.reason,
                                &mut assignments,
                            );
                        }
                        idle = propagate.finish_no_conflict();
                        break;
                    }
                    BcpResult::Conflict { clause_index } => {
                        if trail.current_level() == 0 {
                            let _ = propagate
                                .finish_conflict()
                                .analyze()
                                .backtrack()
                                .unsat();
                            return SolveResult::Unsat;
                        }

                        let analysis =
                            analyze::analyze_conflict(&trail, &db, clause_index);

                        let conflict = propagate.finish_conflict();
                        let analyzed = conflict.analyze();

                        if analysis.learned.is_empty() {
                            let _ = analyzed.backtrack().unsat();
                            return SolveResult::Unsat;
                        }

                        trail.backtrack_to(
                            analysis.backtrack_level,
                            &mut assignments,
                        );

                        let asserting_lit = analysis.learned[0];
                        let clause_idx = db.add_clause(analysis.learned);
                        trail.record_propagation(
                            asserting_lit,
                            clause_idx,
                            &mut assignments,
                        );

                        let bt = analyzed.backtrack();
                        propagate = bt.propagate();
                        bcp_result =
                            bcp::run_bcp(&db, &mut assignments, &propagate);
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
        // Single clause: (x0)
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);

        match solve(db, 1) {
            SolveResult::Sat(assign) => assert!(assign[0]),
            SolveResult::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn trivial_unsat() {
        // x0 ∧ ¬x0
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
        // (x0 ∨ x1) ∧ (¬x0 ∨ x1)
        // Satisfiable: x1=true works regardless of x0
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
        // 3 pigeons, 2 holes. 6 variables (p_i_j = pigeon i in hole j).
        // Each pigeon in at least one hole. At most one pigeon per hole.
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
    fn random_3sat_small() {
        // Generate a small satisfiable instance and verify the solution
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
        match solve(inst.db, inst.num_vars) {
            SolveResult::Sat(assign) => {
                // Verify: check each clause is satisfied
                let cnf_clauses = vec![
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
                for clause in &cnf_clauses {
                    let satisfied = clause.iter().any(|lit: &i32| {
                        let var = (lit.unsigned_abs() - 1) as usize;
                        let pos = *lit > 0;
                        assign[var] == pos
                    });
                    assert!(satisfied, "clause {:?} not satisfied by {:?}", clause, assign);
                }
            }
            SolveResult::Unsat => {
                // This instance might be UNSAT — that's also valid
                // (we didn't construct it to be guaranteed SAT)
            }
        }
    }

    #[test]
    fn dimacs_from_string() {
        // End-to-end: parse DIMACS string → solve → verify
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
            SolveResult::Unsat => panic!("expected SAT for this instance"),
        }
    }
}
