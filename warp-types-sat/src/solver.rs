//! Top-level CDCL solver.
//!
//! Connects trail, BCP, conflict analysis, and the phase-typed session.
//! The trail is the single source of truth for assignments — BCP writes
//! through it, backtracking retracts through it. No ghost assignments.

use std::collections::HashSet;
use std::time::Instant;

use crate::analyze::{self, AnalyzeWork};
use crate::bcp::{self, BcpResult, CRef, ClauseDb};
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
    /// Conflict budget exhausted — neither SAT nor UNSAT proven.
    Unknown,
}

/// Statistics from a solve run.
#[derive(Debug, Default, Clone)]
pub struct SolveStats {
    pub conflicts: u64,
    pub decisions: u64,
    pub propagations: u64,
    /// Nanoseconds spent in BCP (all `run_bcp_watched` calls).
    pub bcp_ns: u64,
    /// Nanoseconds spent in 1-UIP conflict analysis (includes clause minimization).
    pub analyze_ns: u64,
    /// Nanoseconds spent in VSIDS (pick + bump + decay).
    pub vsids_ns: u64,
    /// Nanoseconds within analysis spent on 1-UIP resolution.
    pub analyze_resolve_ns: u64,
    /// Nanoseconds within analysis spent on clause minimization.
    pub analyze_minimize_ns: u64,
    /// Nanoseconds spent in trail-gradient probes.
    pub trail_gradient_ns: u64,
    /// Number of trail-gradient probes performed.
    pub trail_gradient_probes: u64,
}

/// Solve a CNF instance.
///
/// Takes ownership of the clause database (learned clauses are appended).
pub fn solve(mut db: ClauseDb, num_vars: u32) -> SolveResult {
    // Empty clause → trivially UNSAT (a disjunction of zero literals is false).
    for cref in db.iter_crefs() {
        if db.clause(cref).literals.is_empty() {
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
                    BcpResult::Conflict { clause } => {
                        if trail.current_level() == 0 {
                            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                            return SolveResult::Unsat;
                        }

                        let analysis = analyze::analyze_conflict(&trail, &db, clause);

                        let conflict = propagate.finish_conflict();
                        let analyzed = conflict.analyze();

                        if analysis.learned.is_empty() {
                            let _ = analyzed.backtrack().unsat();
                            return SolveResult::Unsat;
                        }

                        trail.backtrack_to(analysis.backtrack_level);

                        let asserting_lit = analysis.learned[0];
                        let cref = db.add_clause(analysis.learned);
                        trail.record_propagation(asserting_lit, cref);

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

/// Solve with stats: returns both the result and performance counters.
pub fn solve_watched_stats(db: ClauseDb, num_vars: u32) -> (SolveResult, SolveStats) {
    solve_cdcl_core_stats(db, num_vars, Vsids::new(num_vars))
}

/// Internal CDCL solver with stats.
fn solve_cdcl_core_stats(
    mut db: ClauseDb,
    num_vars: u32,
    mut vsids: Vsids,
) -> (SolveResult, SolveStats) {
    let mut stats = SolveStats::default();
    let result = solve_cdcl_core_inner(&mut db, num_vars, &mut vsids, &mut stats, 0, 0.0, 0);
    (result, stats)
}

/// Solve with a conflict budget. Returns Unknown if budget exhausted.
pub fn solve_watched_budget(db: ClauseDb, num_vars: u32, conflict_limit: u64) -> (SolveResult, SolveStats) {
    let mut db = db;
    let mut vsids = Vsids::new(num_vars);
    let mut stats = SolveStats::default();
    let result = solve_cdcl_core_inner(&mut db, num_vars, &mut vsids, &mut stats, 0, 0.0, conflict_limit);
    (result, stats)
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
    solve_cdcl_core_inner(&mut db, num_vars, &mut vsids, &mut SolveStats::default(), 0, 0.0, 0)
}

/// CDCL solver with periodic trail-gradient probing (seed-1a).
///
/// Every `probe_interval` conflicts, computes a single gradient at the current
/// trail position and uses it to set phase hints and optionally boost activities
/// for unassigned variables.
pub fn solve_watched_trail_gradient(
    db: ClauseDb,
    num_vars: u32,
    probe_interval: u64,
    boost_scale: f64,
) -> SolveResult {
    let mut db = db;
    let mut vsids = Vsids::new(num_vars);
    vsids.initialize_from_clauses(&db);
    solve_cdcl_core_inner(&mut db, num_vars, &mut vsids, &mut SolveStats::default(), probe_interval, boost_scale, 0)
}

/// CDCL with trail-gradient probing, returning stats.
pub fn solve_watched_trail_gradient_stats(
    db: ClauseDb,
    num_vars: u32,
    probe_interval: u64,
    boost_scale: f64,
    conflict_limit: u64,
) -> (SolveResult, SolveStats) {
    let mut db = db;
    let mut vsids = Vsids::new(num_vars);
    vsids.initialize_from_clauses(&db);
    let mut stats = SolveStats::default();
    let result = solve_cdcl_core_inner(&mut db, num_vars, &mut vsids, &mut stats, probe_interval, boost_scale, conflict_limit);
    (result, stats)
}

fn solve_cdcl_core_inner(
    db: &mut ClauseDb,
    num_vars: u32,
    vsids: &mut Vsids,
    stats: &mut SolveStats,
    trail_gradient_interval: u64,
    trail_gradient_boost: f64,
    conflict_limit: u64,
) -> SolveResult {
    for cref in db.iter_crefs() {
        if db.clause(cref).literals.is_empty() {
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
    let mut watches = watch::Watches::new(db, num_vars);
    let mut restarts = LubyRestarts::new(32);
    let mut restart_pending = false;
    let mut analyze_work = AnalyzeWork::new(num_vars as usize);

    // LBD clause deletion with periodic compaction
    db.freeze_original();
    let mut conflicts: u64 = 0;
    let reduce_interval: u64 = 2000;
    let mut next_reduce: u64 = reduce_interval;

    // Trail-gradient probe schedule (seed-1a)
    let mut next_probe: u64 = trail_gradient_interval;

    session::with_session(|initial_session| {
        let propagate = initial_session.propagate();
        if let BcpResult::Conflict { .. } =
            watch::run_bcp_watched(db, &mut watches, &mut trail, &propagate)
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
                    let locked = build_locked_set(&trail);
                    let deleted = db.reduce_learned(&locked);
                    if !deleted.is_empty() {
                        // Compact db → contiguous CRefs for cache locality
                        let remap = db.compact();
                        trail.remap_reasons(&remap);
                        // Rebuild watches from the now-compact database
                        watches = watch::Watches::new(db, num_vars);
                        watches.set_queue_head(trail.len());
                    }
                    next_reduce = conflicts + reduce_interval;
                }
            }

            // ── Trail-gradient probe (seed-1a) ──
            if trail_gradient_interval > 0 && conflicts >= next_probe {
                let t = Instant::now();
                let tg = crate::gradient::gradient_at_trail(db, num_vars, trail.assignments());
                vsids.apply_trail_gradient(
                    &tg.magnitudes,
                    &tg.polarities,
                    trail.assignments(),
                    trail_gradient_boost,
                );
                stats.trail_gradient_ns += t.elapsed().as_nanos() as u64;
                stats.trail_gradient_probes += 1;
                next_probe = conflicts + trail_gradient_interval;
            }

            if trail.all_assigned() {
                let _ = idle.sat();
                return SolveResult::Sat(trail.assignment_vec());
            }

            // ── VSIDS decision (highest activity + saved phase) ──
            let t = Instant::now();
            let (var, polarity) = vsids.pick(trail.assignments());
            stats.vsids_ns += t.elapsed().as_nanos() as u64;
            let lit = if polarity { Lit::pos(var) } else { Lit::neg(var) };
            stats.decisions += 1;
            let trail_before = trail.len();
            trail.new_decision(lit);
            let mut propagate = idle.decide().propagate();
            let t = Instant::now();
            let mut bcp_result =
                watch::run_bcp_watched(db, &mut watches, &mut trail, &propagate);
            stats.bcp_ns += t.elapsed().as_nanos() as u64;
            stats.propagations += (trail.len() - trail_before - 1) as u64; // -1 for the decision

            // ── Inner conflict resolution loop ──
            loop {
                match bcp_result {
                    BcpResult::Ok => {
                        idle = propagate.finish_no_conflict();
                        break;
                    }
                    BcpResult::Conflict { clause } => {
                        conflicts += 1;
                        stats.conflicts += 1;

                        // Conflict budget exhausted
                        if conflict_limit > 0 && conflicts >= conflict_limit {
                            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                            return SolveResult::Unknown;
                        }

                        if trail.current_level() == 0 {
                            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                            return SolveResult::Unsat;
                        }

                        let t = Instant::now();
                        let analysis = analyze::analyze_conflict_with(
                            &mut analyze_work, &trail, &db, clause,
                        );
                        stats.analyze_ns += t.elapsed().as_nanos() as u64;
                        stats.analyze_resolve_ns += analysis.resolve_ns;
                        stats.analyze_minimize_ns += analysis.minimize_ns;

                        // ── VSIDS: bump learned clause variables, decay ──
                        let t = Instant::now();
                        for &learned_lit in &analysis.learned {
                            vsids.bump(learned_lit.var());
                        }
                        vsids.decay();
                        stats.vsids_ns += t.elapsed().as_nanos() as u64;

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
                        let cref = db.add_clause(analysis.learned);
                        db.set_lbd(cref, lbd as u16);
                        watches.add_clause(db, cref);
                        trail.record_propagation(asserting_lit, cref);

                        // ── Restart check ──
                        if restarts.on_conflict() {
                            restart_pending = true;
                        }

                        let bt = analyzed.backtrack();
                        propagate = bt.propagate();
                        let trail_before_bcp = trail.len();
                        let t = Instant::now();
                        bcp_result =
                            watch::run_bcp_watched(db, &mut watches, &mut trail, &propagate);
                        stats.bcp_ns += t.elapsed().as_nanos() as u64;
                        stats.propagations += (trail.len() - trail_before_bcp) as u64;
                    }
                }
            }
        }
    })
}

/// Build a set of CRefs for clauses that are "locked" — currently
/// serving as a propagation reason for an assignment on the trail.
/// Locked clauses must not be deleted.
fn build_locked_set(trail: &Trail) -> HashSet<CRef> {
    let mut locked = HashSet::new();
    for entry in trail.entries() {
        if let Reason::Propagation(cref) = entry.reason {
            locked.insert(cref);
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
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
        }
    }

    #[test]
    fn pigeonhole_2_1_unsat() {
        let cnf = "p cnf 2 3\n1 0\n2 0\n-1 -2 0\n";
        let inst = dimacs::parse_dimacs_str(cnf).unwrap();
        match solve(inst.db, inst.num_vars) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT"),
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
        }
    }

    #[test]
    fn empty_clause_is_unsat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![]); // empty clause = trivially false
        match solve(db, 1) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT for empty clause"),
            _ => panic!("unexpected Unknown"),
        }
    }

    #[test]
    fn zero_vars_no_clauses_is_sat() {
        let db = ClauseDb::new();
        match solve(db, 0) {
            SolveResult::Sat(assign) => assert!(assign.is_empty()),
            SolveResult::Unsat => panic!("expected SAT for vacuous formula"),
            _ => panic!("unexpected Unknown"),
        }
    }

    #[test]
    fn zero_vars_with_clauses_is_unsat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);
        match solve(db, 0) {
            SolveResult::Unsat => {}
            SolveResult::Sat(_) => panic!("expected UNSAT for 0-var formula with clauses"),
            _ => panic!("unexpected Unknown"),
        }
    }

    #[test]
    fn empty_db_is_sat() {
        let db = ClauseDb::new();
        match solve(db, 5) {
            SolveResult::Sat(_) => {}
            SolveResult::Unsat => panic!("expected SAT for vacuously satisfiable formula"),
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
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
            _ => panic!("unexpected Unknown"),
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
                _ => panic!("unexpected Unknown"),
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

    #[test]
    fn per_conflict_cost_profile() {
        use crate::bench::generate_3sat_phase_transition;
        use std::time::Instant;

        println!("\n=== Per-conflict cost breakdown (200 vars) ===");
        println!("{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12} {:>8}",
            "seed", "result", "ms", "conflicts", "decisions", "props",
            "us/conf", "prop/conf");
        println!("{}", "-".repeat(90));

        for seed in 0..10 {
            let db = generate_3sat_phase_transition(200, seed);
            let t = Instant::now();
            let (result, stats) = solve_watched_stats(db, 200);
            let elapsed_us = t.elapsed().as_micros();
            let tag = match result {
                SolveResult::Sat(_) => "SAT",
                SolveResult::Unsat => "UNSAT",
                _ => panic!("unexpected Unknown"),
            };
            let us_per_conf = if stats.conflicts > 0 {
                elapsed_us as f64 / stats.conflicts as f64
            } else { 0.0 };
            let props_per_conf = if stats.conflicts > 0 {
                stats.propagations as f64 / stats.conflicts as f64
            } else { 0.0 };
            println!("{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12.1} {:>8.1}",
                seed, tag, elapsed_us / 1000, stats.conflicts, stats.decisions,
                stats.propagations, us_per_conf, props_per_conf);
        }
    }

    #[test]
    fn per_conflict_cost_scaling() {
        use crate::bench::generate_k_sat;
        use std::time::Instant;

        // How does per-conflict BCP cost scale with problem size?
        // Uses decreasing clause/var ratios at larger sizes to ensure
        // instances remain solvable (phase transition is ~4.267 for 3-SAT).
        // At larger sizes, watch lists and clause DB grow, stressing cache.
        println!("\n=== Per-conflict cost scaling ===");
        println!("{:<6} {:>6} {:<6} {:>8} {:>8} {:>10} {:>10} {:>12} {:>8}",
            "vars", "ratio", "seed", "result", "ms", "conflicts", "props",
            "us/conf", "prop/conf");
        println!("{}", "-".repeat(88));

        // (vars, clause_ratio): lower ratio = easier (more SAT, fewer conflicts)
        let configs: &[(u32, f64)] = &[
            (200, 4.267), (300, 4.0), (500, 3.5), (700, 3.0), (1000, 2.5),
        ];

        for &(n, ratio) in configs {
            let num_clauses = ((n as f64) * ratio).ceil() as usize;
            let mut completed = 0;
            for seed in 0..30u64 {
                let db = generate_k_sat(n, num_clauses, 3, seed);
                let t = Instant::now();
                let (result, stats) = solve_watched_stats(db, n);
                let elapsed_us = t.elapsed().as_micros();
                let elapsed_ms = elapsed_us / 1000;

                // Hard timeout: skip if too slow (UNSAT or hard SAT)
                if elapsed_ms > 5_000 {
                    continue;
                }
                // Need enough conflicts for stable per-conflict measurement
                if stats.conflicts < 500 {
                    continue;
                }

                let tag = match result {
                    SolveResult::Sat(_) => "SAT",
                    SolveResult::Unsat => "UNSAT",
                    _ => panic!("unexpected Unknown"),
                };
                let us_per_conf = elapsed_us as f64 / stats.conflicts as f64;
                let props_per_conf = stats.propagations as f64 / stats.conflicts as f64;
                println!("{:<6} {:>6.3} {:<6} {:>8} {:>8} {:>10} {:>10} {:>12.1} {:>8.1}",
                    n, ratio, seed, tag, elapsed_ms, stats.conflicts, stats.propagations,
                    us_per_conf, props_per_conf);

                completed += 1;
                if completed >= 3 {
                    break;
                }
            }
            if completed == 0 {
                println!("{:<6} {:>6.3} (no seeds with >=500 conflicts in <5s)", n, ratio);
            }
        }
    }

    #[test]
    fn phase_timing_profile() {
        use crate::bench::generate_k_sat;
        use std::time::Instant;

        // Per-phase timing breakdown with analysis sub-profiling.
        // Shows where time goes: BCP, analysis (split into resolve + minimize), VSIDS.
        println!("\n=== Phase timing breakdown ===");
        println!("{:<6} {:>6} {:<6} {:>7} {:>9} {:>6} {:>6} {:>6} {:>6} {:>6} {:>10} {:>10}",
            "vars", "ratio", "seed", "ms", "conflicts",
            "bcp%", "resv%", "min%", "vsid%", "otr%",
            "ns/prop", "ns/conf");
        println!("{}", "-".repeat(110));

        let configs: &[(u32, f64)] = &[
            (200, 4.267), (300, 4.0), (500, 3.5), (700, 3.0), (1000, 2.5),
        ];

        for &(n, ratio) in configs {
            let num_clauses = ((n as f64) * ratio).ceil() as usize;
            let mut completed = 0;
            for seed in 0..30u64 {
                let db = generate_k_sat(n, num_clauses, 3, seed);
                let t = Instant::now();
                let (result, stats) = solve_watched_stats(db, n);
                let wall_ns = t.elapsed().as_nanos() as u64;
                let wall_ms = wall_ns / 1_000_000;

                if wall_ms > 5_000 { continue; }
                if stats.conflicts < 500 { continue; }

                let _ = match result {
                    SolveResult::Sat(_) => "SAT",
                    SolveResult::Unsat => "UNSAT",
                    _ => panic!("unexpected Unknown"),
                };

                let total_accounted = stats.bcp_ns + stats.analyze_ns + stats.vsids_ns;
                let other_ns = wall_ns.saturating_sub(total_accounted);

                let pct = |ns: u64| -> f64 { 100.0 * ns as f64 / wall_ns as f64 };
                let ns_per_prop = if stats.propagations > 0 {
                    stats.bcp_ns as f64 / stats.propagations as f64
                } else { 0.0 };
                let ns_per_conf = if stats.conflicts > 0 {
                    wall_ns as f64 / stats.conflicts as f64
                } else { 0.0 };

                println!("{:<6} {:>6.3} {:<6} {:>7} {:>9} {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>10.1} {:>10.0}",
                    n, ratio, seed, wall_ms, stats.conflicts,
                    pct(stats.bcp_ns), pct(stats.analyze_resolve_ns),
                    pct(stats.analyze_minimize_ns),
                    pct(stats.vsids_ns), pct(other_ns),
                    ns_per_prop, ns_per_conf);

                completed += 1;
                if completed >= 3 { break; }
            }
            if completed == 0 {
                println!("{:<6} {:>6.3} (no seeds with >=500 conflicts in <5s)", n, ratio);
            }
        }
    }

    #[test]
    fn seed_1a_kill_signal() {
        // Kill signal: 100 instances at n=300, ratio=4.26.
        // If trail-gradient doesn't improve median solve time by >10%, it's dead.
        //
        // Uses a conflict budget (50K) instead of wall-clock timeout to bound
        // UNSAT instances. Instances exceeding the budget return Unknown and
        // are skipped — they're too hard for any config to solve.
        use crate::bench::generate_k_sat;

        let n = 300u32;
        let ratio = 4.26;
        let num_clauses = ((n as f64) * ratio).ceil() as usize;
        let num_instances = 100u64;
        let conflict_budget = 50_000u64;

        struct Config {
            name: &'static str,
            interval: u64,
            boost: f64,
        }
        let configs = [
            Config { name: "CDCL (baseline)", interval: 0, boost: 0.0 },
            Config { name: "TG phase K=50", interval: 50, boost: 0.0 },
            Config { name: "TG phase K=200", interval: 200, boost: 0.0 },
            Config { name: "TG ph+bst K=50 b=1", interval: 50, boost: 1.0 },
            Config { name: "TG ph+bst K=200 b=1", interval: 200, boost: 1.0 },
            Config { name: "TG ph+bst K=50 b=5", interval: 50, boost: 5.0 },
        ];

        println!("\n=== Seed-1a Kill Signal: n={n}, ratio={ratio}, {num_instances} instances, budget={conflict_budget} conflicts ===");
        println!("{:<25} {:>12} {:>12} {:>12} {:>8} {:>8} {:>8} {:>12}",
            "solver", "p25(us)", "p50(us)", "p75(us)", "SAT", "UNK", "UNSAT", "probe_ns/p");
        println!("{}", "-".repeat(105));

        for cfg in &configs {
            let mut times = Vec::new();
            let mut sat_count = 0u32;
            let mut unknown_count = 0u32;
            let mut unsat_count = 0u32;
            let mut total_probe_ns = 0u64;
            let mut total_probes = 0u64;

            for seed in 0..num_instances {
                let db = generate_k_sat(n, num_clauses, 3, seed);
                let t = Instant::now();
                let (result, stats) = if cfg.interval == 0 {
                    solve_watched_budget(db, n, conflict_budget)
                } else {
                    solve_watched_trail_gradient_stats(db, n, cfg.interval, cfg.boost, conflict_budget)
                };
                let elapsed = t.elapsed();

                match result {
                    SolveResult::Unknown => { unknown_count += 1; continue; }
                    SolveResult::Sat(_) => { sat_count += 1; }
                    SolveResult::Unsat => { unsat_count += 1; }
                }

                times.push(elapsed);
                total_probe_ns += stats.trail_gradient_ns;
                total_probes += stats.trail_gradient_probes;
            }

            times.sort();
            let pct = |v: &[std::time::Duration], p: usize| -> u128 {
                if v.is_empty() { return 0; }
                v[v.len() * p / 100].as_micros()
            };
            let ns_per_probe = if total_probes > 0 {
                total_probe_ns / total_probes
            } else { 0 };

            println!("{:<25} {:>12} {:>12} {:>12} {:>8} {:>8} {:>8} {:>12}",
                cfg.name,
                pct(&times, 25), pct(&times, 50), pct(&times, 75),
                sat_count, unknown_count, unsat_count, ns_per_probe);
        }
    }
}