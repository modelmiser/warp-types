//! Top-level CDCL solver.
//!
//! Connects trail, BCP, conflict analysis, and the phase-typed session.
//! The trail is the single source of truth for assignments — BCP writes
//! through it, backtracking retracts through it. No ghost assignments.

use crate::analyze::{self, AnalyzeWork};
use crate::bcp::{self, BcpResult, ClauseDb};
use crate::literal::Lit;
use crate::restart::LubyRestarts;
use crate::session;
use crate::trail::{Reason, Trail};
use crate::vsids::Vsids;
use crate::watch;

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

    // Validate that clause database doesn't reference variables beyond num_vars.
    let max_var = db.max_variable();
    assert!(
        db.is_empty() || max_var < num_vars,
        "clause database references variable {max_var} but only {num_vars} variables declared",
    );

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

/// Solve a CNF instance using watched-literal BCP with VSIDS and Luby restarts.
///
/// Full CDCL loop: two-watched-literal BCP, VSIDS branching heuristic with
/// phase saving, Luby restart policy (base interval 100 conflicts).
pub fn solve_watched(db: ClauseDb, num_vars: u32) -> SolveResult {
    solve_cdcl_core(db, num_vars, Vsids::new(num_vars))
}

/// Internal CDCL solver: watched literals + VSIDS + restarts + phase saving.
///
/// Accepts a pre-configured `Vsids` so callers (e.g., hybrid solver) can
/// warm-start activity scores and phase hints from external sources.
pub(crate) fn solve_cdcl_core(
    mut db: ClauseDb,
    num_vars: u32,
    mut vsids: Vsids,
) -> SolveResult {
    for i in 0..db.len() {
        if db.clause(i).literals.is_empty() {
            return SolveResult::Unsat;
        }
    }
    if num_vars == 0 {
        return if db.is_empty() {
            SolveResult::Sat(vec![])
        } else {
            SolveResult::Unsat
        };
    }

    let max_var = db.max_variable();
    assert!(
        db.is_empty() || max_var < num_vars,
        "clause database references variable {max_var} but only {num_vars} variables declared",
    );

    let mut trail = Trail::new(num_vars as usize);
    let mut watches = watch::Watches::new(&db, num_vars);
    let mut restarts = LubyRestarts::new(32);
    let mut restart_pending = false;
    let mut analyze_work = AnalyzeWork::new(num_vars as usize);

    // LBD clause deletion with periodic compaction
    db.freeze_original();
    let mut conflicts: u64 = 0;
    let reduce_interval: u64 = 2000;
    let mut next_reduce: u64 = reduce_interval;

    session::with_session(|initial_session| {
        let propagate = initial_session.propagate();
        if let BcpResult::Conflict { .. } =
            watch::run_bcp_watched(&db, &mut watches, &mut trail, &propagate)
        {
            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
            return SolveResult::Unsat;
        }
        let mut idle = propagate.finish_no_conflict();

        loop {
            // ── Execute pending restart ──
            if restart_pending && trail.current_level() > 0 {
                for entry in trail.entries_above(0) {
                    vsids.save_phase(entry.lit.var(), !entry.lit.is_negated());
                    vsids.notify_unassigned(entry.lit.var());
                }
                trail.backtrack_to(0);
                watches.notify_backtrack(trail.len());
                restart_pending = false;

                // ── Learned clause deletion + compaction ──
                if conflicts >= next_reduce {
                    let locked = build_locked_set(&trail, db.len());
                    let deleted = db.reduce_learned(&locked);
                    if !deleted.is_empty() {
                        // Compact db → contiguous indices for cache locality
                        let remap = db.compact();
                        trail.remap_reasons(&remap);
                        // Rebuild watches from the now-compact database
                        watches = watch::Watches::new(&db, num_vars);
                        watches.set_queue_head(trail.len());
                    }
                    next_reduce = conflicts + reduce_interval;
                }
            }

            if trail.all_assigned() {
                let _ = idle.sat();
                return SolveResult::Sat(trail.assignment_vec());
            }

            // ── VSIDS decision (highest activity + saved phase) ──
            let (var, polarity) = vsids.pick(trail.assignments());
            let lit = if polarity { Lit::pos(var) } else { Lit::neg(var) };
            trail.new_decision(lit);
            let mut propagate = idle.decide().propagate();
            let mut bcp_result =
                watch::run_bcp_watched(&db, &mut watches, &mut trail, &propagate);

            // ── Inner conflict resolution loop ──
            loop {
                match bcp_result {
                    BcpResult::Ok => {
                        idle = propagate.finish_no_conflict();
                        break;
                    }
                    BcpResult::Conflict { clause_index } => {
                        conflicts += 1;

                        if trail.current_level() == 0 {
                            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                            return SolveResult::Unsat;
                        }

                        let analysis = analyze::analyze_conflict_with(
                            &mut analyze_work, &trail, &db, clause_index,
                        );

                        // ── VSIDS: bump learned clause variables, decay ──
                        for &learned_lit in &analysis.learned {
                            vsids.bump(learned_lit.var());
                        }
                        vsids.decay();

                        let conflict = propagate.finish_conflict();
                        let analyzed = conflict.analyze();

                        if analysis.learned.is_empty() {
                            let _ = analyzed.backtrack().unsat();
                            return SolveResult::Unsat;
                        }

                        // ── Phase saving + heap re-insertion before backtrack ──
                        for entry in trail.entries_above(analysis.backtrack_level) {
                            vsids.save_phase(
                                entry.lit.var(),
                                !entry.lit.is_negated(),
                            );
                            vsids.notify_unassigned(entry.lit.var());
                        }

                        trail.backtrack_to(analysis.backtrack_level);
                        watches.notify_backtrack(trail.len());

                        let asserting_lit = analysis.learned[0];
                        let lbd = analysis.lbd;
                        let clause_idx = db.add_clause(analysis.learned);
                        db.set_lbd(clause_idx, lbd as u16);
                        watches.add_clause(&db, clause_idx);
                        trail.record_propagation(asserting_lit, clause_idx);

                        // ── Restart check ──
                        if restarts.on_conflict() {
                            restart_pending = true;
                        }

                        let bt = analyzed.backtrack();
                        propagate = bt.propagate();
                        bcp_result =
                            watch::run_bcp_watched(&db, &mut watches, &mut trail, &propagate);
                    }
                }
            }
        }
    })
}

/// Build a boolean array marking which clauses are "locked" — currently
/// serving as a propagation reason for an assignment on the trail.
/// Locked clauses must not be deleted.
fn build_locked_set(trail: &Trail, num_clauses: usize) -> Vec<bool> {
    let mut locked = vec![false; num_clauses];
    for entry in trail.entries() {
        if let Reason::Propagation(ci) = entry.reason {
            if ci < locked.len() {
                locked[ci] = true;
            }
        }
    }
    locked
}

/// Pick the next unassigned variable. Simple sequential scan.
///
/// Always decides positive polarity (`Lit::pos`). This is correct but naive —
/// VSIDS or phase-saving heuristics would improve performance on hard instances.
/// The solver backtracks and learns the correct polarity via conflict analysis.
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

    // ── solve_watched tests: must match solve on every instance ──

    #[test]
    fn watched_trivial_sat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);
        match solve_watched(db, 1) {
            SolveResult::Sat(a) => assert!(a[0]),
            SolveResult::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn watched_trivial_unsat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);
        db.add_clause(vec![Lit::neg(0)]);
        match solve_watched(db, 1) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT"),
        }
    }

    #[test]
    fn watched_pigeonhole_3_2() {
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
        match solve_watched(inst.db, inst.num_vars) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT"),
        }
    }

    #[test]
    fn watched_satisfiable_3sat() {
        let cnf = "\
p cnf 3 4
1 2 0
-1 3 0
2 3 0
-2 -3 0
";
        let inst = dimacs::parse_dimacs_str(cnf).unwrap();
        match solve_watched(inst.db, inst.num_vars) {
            SolveResult::Sat(_) => {}
            SolveResult::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn watched_agrees_with_solve() {
        // Both solvers must agree on SAT/UNSAT for random 3-SAT instances.
        use crate::bench::generate_3sat_phase_transition;
        for seed in 0..20 {
            let db1 = generate_3sat_phase_transition(30, seed);
            let db2 = generate_3sat_phase_transition(30, seed);
            let r1 = solve(db1, 30);
            let r2 = solve_watched(db2, 30);
            assert_eq!(
                matches!(r1, SolveResult::Sat(_)),
                matches!(r2, SolveResult::Sat(_)),
                "seed {seed}: solve and solve_watched disagree"
            );
        }
    }

    #[test]
    fn watched_scaling_benchmark() {
        // Performance comparison. Run with --release --nocapture.
        use crate::bench::generate_3sat_phase_transition;
        use std::time::Instant;

        println!("\n=== solve (old BCP) vs solve_watched (watched literals) ===");
        println!("{:<6} {:>10} {:>10} {:>8}", "vars", "old(us)", "watch(us)", "speedup");
        println!("{}", "-".repeat(40));

        for &n in &[20, 50, 100] {
            let seeds = 3u64;
            let (mut ot, mut wt) = (0u128, 0u128);
            let skip_old = false;

            for seed in 0..seeds {
                if !skip_old {
                    let db = generate_3sat_phase_transition(n, seed);
                    let t = Instant::now();
                    let _ = solve(db, n);
                    ot += t.elapsed().as_micros();
                }

                let db = generate_3sat_phase_transition(n, seed);
                let t = Instant::now();
                let _ = solve_watched(db, n);
                wt += t.elapsed().as_micros();
            }

            let old_str = if skip_old {
                "n/a".to_string()
            } else {
                format!("{}", ot / seeds as u128)
            };
            let speedup = if skip_old || wt == 0 {
                "n/a".to_string()
            } else {
                format!("{:.1}x", ot as f64 / wt as f64)
            };
            println!(
                "{:<6} {:>10} {:>10} {:>8}",
                n,
                old_str,
                wt / seeds as u128,
                speedup
            );
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
    fn watched_200var_phase_transition() {
        // 200-var random 3-SAT at phase transition — previously hung with
        // sequential decision heuristic. VSIDS + restarts make SAT instances
        // sub-second. UNSAT instances are still hard without clause deletion.
        use crate::bench::generate_3sat_phase_transition;
        use crate::gradient;
        use std::time::Instant;

        println!("\n=== 200-var CDCL scalability (VSIDS + restarts) ===");
        println!("{:<6} {:>10} {:>8}", "seed", "time(ms)", "result");
        println!("{}", "-".repeat(30));

        let mut sat_count = 0;
        for seed in 0..10 {
            let db = generate_3sat_phase_transition(200, seed);
            let t = Instant::now();
            let result = solve_watched(db, 200);
            let elapsed = t.elapsed().as_millis();
            let tag = match &result {
                SolveResult::Sat(a) => {
                    let db = generate_3sat_phase_transition(200, seed);
                    assert!(gradient::verify(&db, a), "seed {seed}: invalid assignment");
                    sat_count += 1;
                    // SAT instances must be fast in release mode.
                    // Debug mode is ~13x slower due to bounds checking;
                    // use a higher threshold to avoid false failures.
                    let threshold = if cfg!(debug_assertions) { 60_000 } else { 5_000 };
                    assert!(
                        elapsed < threshold,
                        "seed {seed}: SAT took {elapsed}ms — should be sub-{threshold}ms"
                    );
                    "SAT"
                }
                SolveResult::Unsat => "UNSAT",
            };
            println!("{:<6} {:>10} {:>8}", seed, elapsed, tag);
            // Skip slow UNSAT seeds — clause deletion not yet implemented
            if elapsed > 10_000 {
                println!("(skipping remaining seeds — UNSAT is slow without clause deletion)");
                break;
            }
        }
        assert!(sat_count > 0, "should find at least one SAT instance in 10 seeds");
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
