//! Top-level CDCL solver.
//!
//! Connects trail, BCP, conflict analysis, and the phase-typed session.
//! The trail is the single source of truth for assignments — BCP writes
//! through it, backtracking retracts through it. No ghost assignments.

use std::collections::HashSet;
use std::time::Instant;

use crate::analyze::{self, AnalyzeWork, ConflictProfile};
use crate::bcp::{self, BcpResult, CRef, ClauseDb};
use crate::literal::Lit;
use crate::restart::LubyRestarts;
use crate::session;
use crate::theory::{NoTheory, TheoryContext, TheoryResult, TheorySolver};
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

/// Default pivot bump scale for production solver.
/// 0.5 gives ~35% conflict reduction on 200-var random 3-SAT without
/// overshooting. Derived from pivot centrality correlation experiments.
const DEFAULT_PIVOT_BUMP_SCALE: f64 = 0.5;

/// Solve a CNF instance using watched-literal BCP with VSIDS and Luby restarts.
///
/// Full CDCL loop: two-watched-literal BCP, VSIDS branching heuristic with
/// phase saving + pivot-augmented bumps, Luby restart policy (base interval
/// 100 conflicts).
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
    let result = solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut stats,
        0,
        0.0,
        0,
        DEFAULT_PIVOT_BUMP_SCALE,
        &mut NoTheory,
    );
    (result, stats)
}

/// Solve with a conflict budget. Returns Unknown if budget exhausted.
pub fn solve_watched_budget(
    db: ClauseDb,
    num_vars: u32,
    conflict_limit: u64,
) -> (SolveResult, SolveStats) {
    let mut db = db;
    let mut vsids = Vsids::new(num_vars);
    let mut stats = SolveStats::default();
    let result = solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut stats,
        0,
        0.0,
        conflict_limit,
        DEFAULT_PIVOT_BUMP_SCALE,
        &mut NoTheory,
    );
    (result, stats)
}

/// Internal CDCL solver: watched literals + VSIDS + restarts + phase saving.
///
/// Accepts a pre-configured `Vsids` so callers (e.g., hybrid solver) can
/// warm-start activity scores and phase hints from external sources.
pub(crate) fn solve_cdcl_core(mut db: ClauseDb, num_vars: u32, mut vsids: Vsids) -> SolveResult {
    solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut SolveStats::default(),
        0,
        0.0,
        0,
        DEFAULT_PIVOT_BUMP_SCALE,
        &mut NoTheory,
    )
}

/// CDCL solver with periodic trail-gradient probing.
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
    solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut SolveStats::default(),
        probe_interval,
        boost_scale,
        0,
        0.0,
        &mut NoTheory,
    )
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
    let result = solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut stats,
        probe_interval,
        boost_scale,
        conflict_limit,
        0.0,
        &mut NoTheory,
    );
    (result, stats)
}

/// CDCL with both trail-gradient probing AND pivot-augmented VSIDS.
///
/// Combines gradient (phase hints + activity boost) and pivot-augmented
/// VSIDS (decision bumps from resolution pivots). These are orthogonal signals: gradients shape
/// polarity (which value to try), pivots shape variable ordering (which
/// variable to decide next).
pub fn solve_watched_combined(
    db: ClauseDb,
    num_vars: u32,
    probe_interval: u64,
    gradient_boost: f64,
    pivot_bump_scale: f64,
    conflict_limit: u64,
) -> (SolveResult, SolveStats) {
    let mut db = db;
    let mut vsids = Vsids::new(num_vars);
    vsids.initialize_from_clauses(&db);
    let mut stats = SolveStats::default();
    let result = solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut stats,
        probe_interval,
        gradient_boost,
        conflict_limit,
        pivot_bump_scale,
        &mut NoTheory,
    );
    (result, stats)
}

#[allow(clippy::too_many_arguments)]
fn solve_cdcl_core_inner<T: TheorySolver>(
    db: &mut ClauseDb,
    num_vars: u32,
    vsids: &mut Vsids,
    stats: &mut SolveStats,
    trail_gradient_interval: u64,
    trail_gradient_boost: f64,
    conflict_limit: u64,
    pivot_bump_scale: f64,
    theory: &mut T,
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

    // Warm-start VSIDS from clause occurrence counts. Variables appearing in
    // more clauses are more constrained — decide them first. Adds to any
    // pre-seeded activities (e.g., from gradient solver), then rebuilds heap.
    vsids.initialize_from_clauses(db);

    // LBD clause deletion with periodic compaction
    db.freeze_original();
    let mut conflicts: u64 = 0;
    let reduce_interval: u64 = 2000;
    let mut next_reduce: u64 = reduce_interval;

    // Trail-gradient probe schedule
    let mut next_probe: u64 = trail_gradient_interval;

    session::with_session(|initial_session| {
        let propagate = initial_session.propagate();
        let mut initial_bcp = watch::run_bcp_watched(db, &mut watches, &mut trail, &propagate);

        // ── Initial BCP + theory check loop (level 0) ──
        // Theory may propagate at level 0, which needs further BCP.
        // Theory conflict at level 0 means UNSAT (no backtrack possible).
        loop {
            match initial_bcp {
                BcpResult::Conflict { .. } => {
                    let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                    return SolveResult::Unsat;
                }
                BcpResult::Ok => {
                    let theory_result = {
                        let ctx = TheoryContext {
                            trail: &trail,
                            db: &*db,
                            num_vars,
                        };
                        theory.check(&ctx)
                    };
                    match theory_result {
                        TheoryResult::Consistent => break,
                        TheoryResult::Propagate(props) => {
                            for p in props {
                                trail.record_theory_propagation(p.lit, p.key);
                            }
                            initial_bcp =
                                watch::run_bcp_watched(db, &mut watches, &mut trail, &propagate);
                            continue;
                        }
                        TheoryResult::Conflict(_) => {
                            // Theory conflict at level 0 → UNSAT
                            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                            return SolveResult::Unsat;
                        }
                    }
                }
            }
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
                theory.backtrack(0);
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

            // ── Trail-gradient probe ──
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
            let lit = if polarity {
                Lit::pos(var)
            } else {
                Lit::neg(var)
            };
            stats.decisions += 1;
            let trail_before = trail.len();
            trail.new_decision(lit);
            let mut propagate = idle.decide().propagate();
            let t = Instant::now();
            let mut bcp_result = watch::run_bcp_watched(db, &mut watches, &mut trail, &propagate);
            stats.bcp_ns += t.elapsed().as_nanos() as u64;
            stats.propagations += (trail.len() - trail_before - 1) as u64; // -1 for the decision

            // ── Inner conflict resolution loop ──
            loop {
                match bcp_result {
                    BcpResult::Ok => {
                        // ── Theory consistency check (DPLL(T) hook) ──
                        // After BCP fixpoint, ask the theory if the current
                        // partial assignment is consistent.
                        let theory_result = {
                            let ctx = TheoryContext {
                                trail: &trail,
                                db: &*db,
                                num_vars,
                            };
                            theory.check(&ctx)
                        };
                        match theory_result {
                            TheoryResult::Consistent => {
                                idle = propagate.finish_no_conflict();
                                break;
                            }
                            TheoryResult::Propagate(props) => {
                                // Record theory-implied literals on the trail,
                                // then re-run BCP to propagate consequences.
                                let trail_before_theory = trail.len();
                                for p in props {
                                    trail.record_theory_propagation(p.lit, p.key);
                                }
                                let t = Instant::now();
                                bcp_result = watch::run_bcp_watched(
                                    db,
                                    &mut watches,
                                    &mut trail,
                                    &propagate,
                                );
                                stats.bcp_ns += t.elapsed().as_nanos() as u64;
                                stats.propagations += (trail.len() - trail_before_theory) as u64;
                                // Continue inner loop: re-check theory after BCP
                                continue;
                            }
                            TheoryResult::Conflict(lits) => {
                                // Theory found the current assignment inconsistent.
                                // Add the theory conflict clause to the DB and
                                // enter conflict resolution.
                                let cref = db.add_clause(lits);
                                watches.add_clause(db, cref);
                                bcp_result = BcpResult::Conflict { clause: cref };
                                continue;
                            }
                        }
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
                        let analysis = analyze::analyze_conflict_with_theory(
                            &mut analyze_work,
                            &trail,
                            db,
                            clause,
                            theory,
                        );
                        stats.analyze_ns += t.elapsed().as_nanos() as u64;
                        stats.analyze_resolve_ns += analysis.resolve_ns;
                        stats.analyze_minimize_ns += analysis.minimize_ns;

                        // ── VSIDS: bump learned clause variables + pivots, decay ──
                        let t = Instant::now();
                        for &learned_lit in &analysis.learned {
                            vsids.bump(learned_lit.var());
                        }
                        if pivot_bump_scale > 0.0 {
                            for &pv in &analyze_work.pivots {
                                vsids.bump_scaled(pv, pivot_bump_scale);
                            }
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
                            vsids.save_phase(entry.lit.var(), !entry.lit.is_negated());
                            vsids.notify_unassigned(entry.lit.var());
                        }

                        trail.backtrack_to(analysis.backtrack_level);
                        theory.backtrack(analysis.backtrack_level);
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

// ============================================================================
// Theory-aware solver entry points (DPLL(T))
// ============================================================================

/// Solve a CNF instance with a theory solver (DPLL(T)).
///
/// The SAT solver interleaves BCP with theory consistency checks:
/// after each BCP fixpoint, `theory.check()` is called. The theory can
/// propagate implied literals, report conflicts, or confirm consistency.
/// During conflict analysis, `theory.explain()` lazily produces explanation
/// clauses for theory-propagated literals.
///
/// For pure SAT, use [`solve_watched`] (equivalent to `NoTheory`).
pub fn solve_with_theory<T: TheorySolver>(
    db: ClauseDb,
    num_vars: u32,
    theory: &mut T,
) -> SolveResult {
    let mut db = db;
    let mut vsids = Vsids::new(num_vars);
    vsids.initialize_from_clauses(&db);
    solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut SolveStats::default(),
        0,
        0.0,
        0,
        DEFAULT_PIVOT_BUMP_SCALE,
        theory,
    )
}

/// Solve with a theory solver, returning both the result and performance stats.
pub fn solve_with_theory_stats<T: TheorySolver>(
    db: ClauseDb,
    num_vars: u32,
    theory: &mut T,
) -> (SolveResult, SolveStats) {
    let mut db = db;
    let mut vsids = Vsids::new(num_vars);
    vsids.initialize_from_clauses(&db);
    let mut stats = SolveStats::default();
    let result = solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut stats,
        0,
        0.0,
        0,
        DEFAULT_PIVOT_BUMP_SCALE,
        theory,
    );
    (result, stats)
}

/// Solve with a theory solver and a conflict budget.
pub fn solve_with_theory_budget<T: TheorySolver>(
    db: ClauseDb,
    num_vars: u32,
    conflict_limit: u64,
    theory: &mut T,
) -> (SolveResult, SolveStats) {
    let mut db = db;
    let mut vsids = Vsids::new(num_vars);
    vsids.initialize_from_clauses(&db);
    let mut stats = SolveStats::default();
    let result = solve_cdcl_core_inner(
        &mut db,
        num_vars,
        &mut vsids,
        &mut stats,
        0,
        0.0,
        conflict_limit,
        DEFAULT_PIVOT_BUMP_SCALE,
        theory,
    );
    (result, stats)
}

/// Result of an instrumented solve run.
#[derive(Debug)]
pub struct InstrumentedResult {
    pub result: SolveResult,
    /// Per-conflict profiles (Level 2).
    pub profiles: Vec<ConflictProfile>,
    /// CRef of each learned clause, parallel to `profiles`.
    pub learned_crefs: Vec<CRef>,
    /// Final VSIDS activity scores per variable.
    pub vsids_activities: Vec<f64>,
}

/// Solve with resolution chain instrumentation for proof DAG mining.
///
/// Returns an `InstrumentedResult` containing the solve result, per-conflict
/// profiles, learned clause CRefs, and final VSIDS activities.
pub fn solve_instrumented(db: ClauseDb, num_vars: u32, conflict_limit: u64) -> InstrumentedResult {
    solve_instrumented_inner(db, num_vars, conflict_limit, 0.0, 0.0)
}

/// Instrumented solve with depth-weighted clause deletion.
///
/// `depth_weight > 0.0` scores deletion candidates by `LBD + depth_weight * resolution_depth`.
/// Use `depth_weight = 0.0` for baseline LBD-only deletion.
pub fn solve_instrumented_depth_weighted(
    db: ClauseDb,
    num_vars: u32,
    conflict_limit: u64,
    depth_weight: f64,
) -> InstrumentedResult {
    solve_instrumented_inner(db, num_vars, conflict_limit, depth_weight, 0.0)
}

/// Instrumented solve with pivot-augmented VSIDS.
///
/// During conflict analysis, in addition to the standard learned-clause bumps,
/// each pivot variable in the resolution chain receives a bump scaled by
/// `pivot_bump_scale × increment`. This tests whether feeding structural
/// centrality (pivot frequency) back into VSIDS improves solving.
///
/// `pivot_bump_scale = 0.0` is baseline (no pivot bumps).
/// `pivot_bump_scale = 1.0` gives pivots the same bump as learned-clause vars.
pub fn solve_instrumented_pivot_augmented(
    db: ClauseDb,
    num_vars: u32,
    conflict_limit: u64,
    pivot_bump_scale: f64,
) -> InstrumentedResult {
    solve_instrumented_inner(db, num_vars, conflict_limit, 0.0, pivot_bump_scale)
}

fn solve_instrumented_inner(
    mut db: ClauseDb,
    num_vars: u32,
    conflict_limit: u64,
    depth_weight: f64,
    pivot_bump_scale: f64,
) -> InstrumentedResult {
    let mut profiles: Vec<ConflictProfile> = Vec::new();
    let mut learned_crefs: Vec<CRef> = Vec::new();

    let mk = |result, profiles, learned_crefs, activities: Vec<f64>| InstrumentedResult {
        result,
        profiles,
        learned_crefs,
        vsids_activities: activities,
    };

    for cref in db.iter_crefs() {
        if db.clause(cref).literals.is_empty() {
            return mk(SolveResult::Unsat, profiles, learned_crefs, vec![]);
        }
    }
    if num_vars == 0 {
        return if db.is_empty() {
            mk(SolveResult::Sat(vec![]), profiles, learned_crefs, vec![])
        } else {
            mk(SolveResult::Unsat, profiles, learned_crefs, vec![])
        };
    }

    let mut trail = Trail::new(num_vars as usize);
    let mut watches = watch::Watches::new(&db, num_vars);
    let mut restarts = LubyRestarts::new(32);
    let mut restart_pending = false;
    let mut analyze_work = AnalyzeWork::new(num_vars as usize);
    let mut vsids = Vsids::new(num_vars);

    db.freeze_original();
    let mut conflicts: u64 = 0;
    let reduce_interval: u64 = 2000;
    let mut next_reduce: u64 = reduce_interval;

    session::with_session(|initial_session| {
        let propagate = initial_session.propagate();
        if let BcpResult::Conflict { .. } =
            watch::run_bcp_watched(&mut db, &mut watches, &mut trail, &propagate)
        {
            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
            return mk(
                SolveResult::Unsat,
                profiles,
                learned_crefs,
                vsids.activities().to_vec(),
            );
        }
        let mut idle = propagate.finish_no_conflict();

        loop {
            if restart_pending && trail.current_level() > 0 {
                for entry in trail.entries_above(0) {
                    vsids.save_phase(entry.lit.var(), !entry.lit.is_negated());
                    vsids.notify_unassigned(entry.lit.var());
                }
                trail.backtrack_to(0);
                watches.notify_backtrack(trail.len());
                restart_pending = false;

                if conflicts >= next_reduce {
                    let locked = build_locked_set(&trail);
                    let deleted = if depth_weight > 0.0 {
                        db.reduce_learned_weighted(&locked, depth_weight)
                    } else {
                        db.reduce_learned(&locked)
                    };
                    if !deleted.is_empty() {
                        let remap = db.compact();
                        trail.remap_reasons(&remap);
                        // Remap learned_crefs to new positions
                        for lc in &mut learned_crefs {
                            if let Ok(idx) = remap.binary_search_by_key(lc, |&(old, _)| old) {
                                *lc = remap[idx].1;
                            }
                        }
                        watches = watch::Watches::new(&db, num_vars);
                        watches.set_queue_head(trail.len());
                    }
                    next_reduce = conflicts + reduce_interval;
                }
            }

            if trail.all_assigned() {
                let _ = idle.sat();
                return mk(
                    SolveResult::Sat(trail.assignment_vec()),
                    profiles,
                    learned_crefs,
                    vsids.activities().to_vec(),
                );
            }

            let (var, polarity) = vsids.pick(trail.assignments());
            let lit = if polarity {
                Lit::pos(var)
            } else {
                Lit::neg(var)
            };
            let trail_before_decision = trail.len();
            trail.new_decision(lit);
            let mut propagate = idle.decide().propagate();
            let mut bcp_result =
                watch::run_bcp_watched(&mut db, &mut watches, &mut trail, &propagate);
            // BCP propagations = trail growth from BCP (excludes the decision itself)
            let mut bcp_props_this_cycle = (trail.len() - trail_before_decision - 1) as u32;

            loop {
                match bcp_result {
                    BcpResult::Ok => {
                        idle = propagate.finish_no_conflict();
                        break;
                    }
                    BcpResult::Conflict { clause } => {
                        conflicts += 1;

                        if conflict_limit > 0 && conflicts >= conflict_limit {
                            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                            return mk(
                                SolveResult::Unknown,
                                profiles,
                                learned_crefs,
                                vsids.activities().to_vec(),
                            );
                        }

                        if trail.current_level() == 0 {
                            let _ = propagate.finish_conflict().analyze().backtrack().unsat();
                            return mk(
                                SolveResult::Unsat,
                                profiles,
                                learned_crefs,
                                vsids.activities().to_vec(),
                            );
                        }

                        let trail_size = trail.len();
                        let current_level = trail.current_level();

                        // Use instrumented analysis
                        let analysis = analyze::analyze_conflict_instrumented(
                            &mut analyze_work,
                            &trail,
                            &db,
                            clause,
                        );

                        let resolution_depth = analysis.resolution_chain.len() as u32;

                        profiles.push(ConflictProfile {
                            conflict_id: conflicts - 1, // 0-indexed
                            decision_level: current_level,
                            resolution_depth,
                            learned_clause_size: analysis.learned.len(),
                            learned_lbd: analysis.lbd,
                            backtrack_distance: current_level - analysis.backtrack_level,
                            trail_size_at_conflict: trail_size,
                            resolution_chain: analysis.resolution_chain,
                            bcp_propagations: bcp_props_this_cycle,
                        });

                        for &learned_lit in &analysis.learned {
                            vsids.bump(learned_lit.var());
                        }
                        // Pivot-augmented VSIDS: bump pivots from resolution chain.
                        // Pivots are structurally central (C3: r_s=0.524) but only
                        // partially captured by learned-clause bumps (26%). This
                        // feeds the 74% gap back into the decision heuristic.
                        if pivot_bump_scale > 0.0 {
                            if let Some(profile) = profiles.last() {
                                for step in &profile.resolution_chain {
                                    vsids.bump_scaled(step.pivot_var, pivot_bump_scale);
                                }
                            }
                        }
                        vsids.decay();

                        let conflict = propagate.finish_conflict();
                        let analyzed = conflict.analyze();

                        if analysis.learned.is_empty() {
                            let _ = analyzed.backtrack().unsat();
                            return mk(
                                SolveResult::Unsat,
                                profiles,
                                learned_crefs,
                                vsids.activities().to_vec(),
                            );
                        }

                        for entry in trail.entries_above(analysis.backtrack_level) {
                            vsids.save_phase(entry.lit.var(), !entry.lit.is_negated());
                            vsids.notify_unassigned(entry.lit.var());
                        }

                        trail.backtrack_to(analysis.backtrack_level);
                        watches.notify_backtrack(trail.len());

                        let asserting_lit = analysis.learned[0];
                        let lbd = analysis.lbd;
                        let cref = db.add_clause(analysis.learned);
                        db.set_lbd(cref, lbd as u16);
                        if depth_weight > 0.0 {
                            db.set_depth(cref, resolution_depth.min(u16::MAX as u32) as u16);
                        }
                        watches.add_clause(&db, cref);
                        trail.record_propagation(asserting_lit, cref);
                        learned_crefs.push(cref);

                        if restarts.on_conflict() {
                            restart_pending = true;
                        }

                        let bt = analyzed.backtrack();
                        propagate = bt.propagate();
                        let trail_before_bcp = trail.len();
                        bcp_result =
                            watch::run_bcp_watched(&mut db, &mut watches, &mut trail, &propagate);
                        // Update BCP prop count for next potential conflict in this cycle
                        bcp_props_this_cycle = (trail.len() - trail_before_bcp) as u32;
                    }
                }
            }
        }
    })
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
    #[ignore] // benchmark — too slow for CI
    fn watched_scaling_benchmark() {
        // Performance comparison. Run with --release --nocapture.
        use crate::bench::generate_3sat_phase_transition;
        use std::time::Instant;

        println!("\n=== solve (old BCP) vs solve_watched (watched literals) ===");
        println!(
            "{:<6} {:>10} {:>10} {:>8}",
            "vars", "old(us)", "watch(us)", "speedup"
        );
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
                    let threshold = if cfg!(debug_assertions) {
                        60_000
                    } else {
                        5_000
                    };
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
        assert!(
            sat_count > 0,
            "should find at least one SAT instance in 10 seeds"
        );
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
        println!(
            "{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12} {:>8}",
            "seed", "result", "ms", "conflicts", "decisions", "props", "us/conf", "prop/conf"
        );
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
            } else {
                0.0
            };
            let props_per_conf = if stats.conflicts > 0 {
                stats.propagations as f64 / stats.conflicts as f64
            } else {
                0.0
            };
            println!(
                "{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12.1} {:>8.1}",
                seed,
                tag,
                elapsed_us / 1000,
                stats.conflicts,
                stats.decisions,
                stats.propagations,
                us_per_conf,
                props_per_conf
            );
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn per_conflict_cost_scaling() {
        use crate::bench::generate_k_sat;
        use std::time::Instant;

        // How does per-conflict BCP cost scale with problem size?
        // Uses decreasing clause/var ratios at larger sizes to ensure
        // instances remain solvable (phase transition is ~4.267 for 3-SAT).
        // At larger sizes, watch lists and clause DB grow, stressing cache.
        println!("\n=== Per-conflict cost scaling ===");
        println!(
            "{:<6} {:>6} {:<6} {:>8} {:>8} {:>10} {:>10} {:>12} {:>8}",
            "vars", "ratio", "seed", "result", "ms", "conflicts", "props", "us/conf", "prop/conf"
        );
        println!("{}", "-".repeat(88));

        // (vars, clause_ratio): lower ratio = easier (more SAT, fewer conflicts)
        let configs: &[(u32, f64)] = &[
            (200, 4.267),
            (300, 4.0),
            (500, 3.5),
            (700, 3.0),
            (1000, 2.5),
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
                println!(
                    "{:<6} {:>6.3} {:<6} {:>8} {:>8} {:>10} {:>10} {:>12.1} {:>8.1}",
                    n,
                    ratio,
                    seed,
                    tag,
                    elapsed_ms,
                    stats.conflicts,
                    stats.propagations,
                    us_per_conf,
                    props_per_conf
                );

                completed += 1;
                if completed >= 3 {
                    break;
                }
            }
            if completed == 0 {
                println!(
                    "{:<6} {:>6.3} (no seeds with >=500 conflicts in <5s)",
                    n, ratio
                );
            }
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn phase_timing_profile() {
        use crate::bench::generate_k_sat;
        use std::time::Instant;

        // Per-phase timing breakdown with analysis sub-profiling.
        // Shows where time goes: BCP, analysis (split into resolve + minimize), VSIDS.
        println!("\n=== Phase timing breakdown ===");
        println!(
            "{:<6} {:>6} {:<6} {:>7} {:>9} {:>6} {:>6} {:>6} {:>6} {:>6} {:>10} {:>10}",
            "vars",
            "ratio",
            "seed",
            "ms",
            "conflicts",
            "bcp%",
            "resv%",
            "min%",
            "vsid%",
            "otr%",
            "ns/prop",
            "ns/conf"
        );
        println!("{}", "-".repeat(110));

        let configs: &[(u32, f64)] = &[
            (200, 4.267),
            (300, 4.0),
            (500, 3.5),
            (700, 3.0),
            (1000, 2.5),
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

                if wall_ms > 5_000 {
                    continue;
                }
                if stats.conflicts < 500 {
                    continue;
                }

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
                } else {
                    0.0
                };
                let ns_per_conf = if stats.conflicts > 0 {
                    wall_ns as f64 / stats.conflicts as f64
                } else {
                    0.0
                };

                println!("{:<6} {:>6.3} {:<6} {:>7} {:>9} {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>10.1} {:>10.0}",
                    n, ratio, seed, wall_ms, stats.conflicts,
                    pct(stats.bcp_ns), pct(stats.analyze_resolve_ns),
                    pct(stats.analyze_minimize_ns),
                    pct(stats.vsids_ns), pct(other_ns),
                    ns_per_prop, ns_per_conf);

                completed += 1;
                if completed >= 3 {
                    break;
                }
            }
            if completed == 0 {
                println!(
                    "{:<6} {:>6.3} (no seeds with >=500 conflicts in <5s)",
                    n, ratio
                );
            }
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn trail_gradient_kill_signal() {
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
            Config {
                name: "CDCL (baseline)",
                interval: 0,
                boost: 0.0,
            },
            Config {
                name: "TG phase K=50",
                interval: 50,
                boost: 0.0,
            },
            Config {
                name: "TG phase K=200",
                interval: 200,
                boost: 0.0,
            },
            Config {
                name: "TG ph+bst K=50 b=1",
                interval: 50,
                boost: 1.0,
            },
            Config {
                name: "TG ph+bst K=200 b=1",
                interval: 200,
                boost: 1.0,
            },
            Config {
                name: "TG ph+bst K=50 b=5",
                interval: 50,
                boost: 5.0,
            },
        ];

        println!("\n=== Trail-Gradient Kill Signal: n={n}, ratio={ratio}, {num_instances} instances, budget={conflict_budget} conflicts ===");
        println!(
            "{:<25} {:>12} {:>12} {:>12} {:>8} {:>8} {:>8} {:>12}",
            "solver", "p25(us)", "p50(us)", "p75(us)", "SAT", "UNK", "UNSAT", "probe_ns/p"
        );
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
                    solve_watched_trail_gradient_stats(
                        db,
                        n,
                        cfg.interval,
                        cfg.boost,
                        conflict_budget,
                    )
                };
                let elapsed = t.elapsed();

                match result {
                    SolveResult::Unknown => {
                        unknown_count += 1;
                        continue;
                    }
                    SolveResult::Sat(_) => {
                        sat_count += 1;
                    }
                    SolveResult::Unsat => {
                        unsat_count += 1;
                    }
                }

                times.push(elapsed);
                total_probe_ns += stats.trail_gradient_ns;
                total_probes += stats.trail_gradient_probes;
            }

            times.sort();
            let pct = |v: &[std::time::Duration], p: usize| -> u128 {
                if v.is_empty() {
                    return 0;
                }
                v[v.len() * p / 100].as_micros()
            };
            let ns_per_probe = total_probe_ns.checked_div(total_probes).unwrap_or(0);

            println!(
                "{:<25} {:>12} {:>12} {:>12} {:>8} {:>8} {:>8} {:>12}",
                cfg.name,
                pct(&times, 25),
                pct(&times, 50),
                pct(&times, 75),
                sat_count,
                unknown_count,
                unsat_count,
                ns_per_probe
            );
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn resolution_depth_kill_signal() {
        // Kill signal: if median resolution depth < 3, DAGs are
        // too shallow to mine. Run on 200-var and 300-var instances.
        use crate::bench::generate_k_sat;

        let conflict_budget = 50_000u64;

        for &(n, ratio) in &[(200u32, 4.267), (300, 4.26)] {
            let num_clauses = ((n as f64) * ratio).ceil() as usize;
            let mut all_depths: Vec<u32> = Vec::new();
            let mut instances_solved = 0u32;
            let mut instances_unknown = 0u32;

            for seed in 0..20u64 {
                let db = generate_k_sat(n, num_clauses, 3, seed);
                let ir = solve_instrumented(db, n, conflict_budget);
                let (result, profiles) = (ir.result, ir.profiles);

                all_depths.extend(profiles.iter().map(|p| p.resolution_depth));
                match result {
                    SolveResult::Sat(_) => instances_solved += 1,
                    SolveResult::Unsat => instances_solved += 1,
                    SolveResult::Unknown => instances_unknown += 1,
                }
            }

            all_depths.sort();
            let len = all_depths.len();
            if len == 0 {
                println!("n={n}: no conflicts recorded");
                continue;
            }

            let median = all_depths[len / 2];
            let p25 = all_depths[len / 4];
            let p75 = all_depths[3 * len / 4];
            let p90 = all_depths[9 * len / 10];
            let p99 = all_depths[99 * len / 100];
            let max = *all_depths.last().unwrap();
            let mean = all_depths.iter().map(|&d| d as f64).sum::<f64>() / len as f64;

            // Depth distribution histogram (buckets: 0, 1, 2, 3-5, 6-10, 11-20, 21+)
            let mut buckets = [0u32; 7];
            for &d in &all_depths {
                let b = match d {
                    0 => 0,
                    1 => 1,
                    2 => 2,
                    3..=5 => 3,
                    6..=10 => 4,
                    11..=20 => 5,
                    _ => 6,
                };
                buckets[b] += 1;
            }

            println!("\n=== Resolution Depth Kill Signal: n={n}, ratio={ratio} ===");
            println!(
                "  Instances: {} solved, {} budget-exhausted, {} total conflicts",
                instances_solved, instances_unknown, len
            );
            println!("  Depth: p25={p25}, median={median}, mean={mean:.1}, p75={p75}, p90={p90}, p99={p99}, max={max}");
            println!("  Distribution:");
            println!(
                "    depth=0:    {:>6} ({:>5.1}%)",
                buckets[0],
                100.0 * buckets[0] as f64 / len as f64
            );
            println!(
                "    depth=1:    {:>6} ({:>5.1}%)",
                buckets[1],
                100.0 * buckets[1] as f64 / len as f64
            );
            println!(
                "    depth=2:    {:>6} ({:>5.1}%)",
                buckets[2],
                100.0 * buckets[2] as f64 / len as f64
            );
            println!(
                "    depth=3-5:  {:>6} ({:>5.1}%)",
                buckets[3],
                100.0 * buckets[3] as f64 / len as f64
            );
            println!(
                "    depth=6-10: {:>6} ({:>5.1}%)",
                buckets[4],
                100.0 * buckets[4] as f64 / len as f64
            );
            println!(
                "    depth=11-20:{:>6} ({:>5.1}%)",
                buckets[5],
                100.0 * buckets[5] as f64 / len as f64
            );
            println!(
                "    depth=21+:  {:>6} ({:>5.1}%)",
                buckets[6],
                100.0 * buckets[6] as f64 / len as f64
            );

            // Kill signal: median depth < 3 means DAGs are too shallow
            if median < 3 {
                println!(
                    "  *** KILL SIGNAL: median depth {} < 3 — DAGs too shallow ***",
                    median
                );
            } else {
                println!(
                    "  Resolution chains have structure (median depth {} >= 3)",
                    median
                );
            }
        }
    }

    #[test]
    fn conflict_profiles_structural_invariants() {
        // Level 2 structural invariants: every ConflictProfile field must
        // be self-consistent with the solver's state at conflict time.
        use crate::analyze::{clause_reuse_frequency, pivot_frequency, working_width_profile};
        use crate::bench::generate_k_sat;

        let n = 200u32;
        let num_clauses = ((n as f64) * 4.267).ceil() as usize;
        let db = generate_k_sat(n, num_clauses, 3, 42);
        let ir = solve_instrumented(db, n, 10_000);
        let (result, profiles) = (ir.result, ir.profiles);

        assert!(!profiles.is_empty(), "should have conflicts to profile");

        // ── Per-profile invariants ──
        for (i, p) in profiles.iter().enumerate() {
            assert_eq!(
                p.conflict_id, i as u64,
                "conflict_id must be sequential 0-indexed"
            );

            assert!(
                p.decision_level > 0,
                "conflict at level 0 should terminate, not profile"
            );

            assert_eq!(
                p.resolution_depth,
                p.resolution_chain.len() as u32,
                "resolution_depth must match chain length"
            );

            assert!(
                p.learned_clause_size > 0,
                "learned clause must have at least 1 literal (asserting)"
            );

            assert!(
                p.learned_lbd > 0 && p.learned_lbd <= p.learned_clause_size as u32,
                "LBD must be in [1, clause_size], got lbd={} size={}",
                p.learned_lbd,
                p.learned_clause_size
            );

            assert!(
                p.backtrack_distance <= p.decision_level,
                "can't backtrack further than current level"
            );

            assert!(
                p.trail_size_at_conflict > 0,
                "trail must have at least one assignment at conflict"
            );

            // Working widths in chain should all be > 0
            let widths = working_width_profile(&p.resolution_chain);
            for (j, &w) in widths.iter().enumerate() {
                assert!(w > 0, "working width at step {} must be > 0", j);
            }
        }

        // ── Topology: pivot frequency ──
        let pivots = pivot_frequency(&profiles);
        if !pivots.is_empty() {
            let max_pivot_freq = *pivots.values().max().unwrap();
            println!(
                "  Pivot frequency: {} unique vars, max freq {}",
                pivots.len(),
                max_pivot_freq
            );
            // At least some variables should be pivoted more than once
            // (on 200-var instances with 10K conflicts, this is guaranteed)
            assert!(max_pivot_freq > 1, "should see repeated pivots");
        }

        // ── Topology: clause reuse ──
        let reuse = clause_reuse_frequency(&profiles);
        if !reuse.is_empty() {
            let max_reuse = *reuse.values().max().unwrap();
            println!(
                "  Clause reuse: {} unique clauses, max reuse {}",
                reuse.len(),
                max_reuse
            );
            assert!(max_reuse > 1, "should see clause reuse");
        }

        // ── Scatter data: resolution depth vs learned clause size ──
        let mut depth_vs_size: Vec<(u32, usize)> = profiles
            .iter()
            .map(|p| (p.resolution_depth, p.learned_clause_size))
            .collect();
        depth_vs_size.sort();

        println!("\n=== Level 2: Conflict Profiles (n={n}, 10K budget) ===");
        println!("  {} profiles collected", profiles.len());
        println!(
            "  Result: {:?}",
            match &result {
                SolveResult::Sat(_) => "SAT",
                SolveResult::Unsat => "UNSAT",
                SolveResult::Unknown => "UNKNOWN",
            }
        );

        // Depth distribution
        let depths: Vec<u32> = profiles.iter().map(|p| p.resolution_depth).collect();
        let avg_depth = depths.iter().map(|&d| d as f64).sum::<f64>() / depths.len() as f64;
        let avg_size = profiles
            .iter()
            .map(|p| p.learned_clause_size as f64)
            .sum::<f64>()
            / profiles.len() as f64;
        let avg_lbd =
            profiles.iter().map(|p| p.learned_lbd as f64).sum::<f64>() / profiles.len() as f64;
        let avg_bt = profiles
            .iter()
            .map(|p| p.backtrack_distance as f64)
            .sum::<f64>()
            / profiles.len() as f64;
        let avg_bcp = profiles
            .iter()
            .map(|p| p.bcp_propagations as f64)
            .sum::<f64>()
            / profiles.len() as f64;

        println!("  Avg depth={avg_depth:.1}, size={avg_size:.1}, lbd={avg_lbd:.1}, backtrack={avg_bt:.1}, bcp_props={avg_bcp:.1}");

        // Correlation: resolution depth vs learned clause size
        // Pearson r — positive correlation expected (deeper chains → larger clauses)
        if profiles.len() > 10 {
            let n_f = profiles.len() as f64;
            let sum_d: f64 = depths.iter().map(|&d| d as f64).sum();
            let sum_s: f64 = profiles.iter().map(|p| p.learned_clause_size as f64).sum();
            let sum_dd: f64 = depths.iter().map(|&d| (d as f64) * (d as f64)).sum();
            let sum_ss: f64 = profiles
                .iter()
                .map(|p| (p.learned_clause_size as f64).powi(2))
                .sum();
            let sum_ds: f64 = profiles
                .iter()
                .map(|p| p.resolution_depth as f64 * p.learned_clause_size as f64)
                .sum();

            let numer = n_f * sum_ds - sum_d * sum_s;
            let denom = ((n_f * sum_dd - sum_d * sum_d) * (n_f * sum_ss - sum_s * sum_s)).sqrt();
            if denom > 0.0 {
                let r = numer / denom;
                println!("  Pearson r(depth, clause_size) = {r:.3}");
            }
        }
    }

    #[test]
    fn proof_dag_topology() {
        // Level 3: Build proof DAG from conflict profiles, validate topology.
        use crate::analyze::ProofDag;
        use crate::bench::generate_k_sat;

        let n = 200u32;
        let num_clauses = ((n as f64) * 4.267).ceil() as usize;
        let db = generate_k_sat(n, num_clauses, 3, 42);
        let ir = solve_instrumented(db, n, 10_000);
        let (result, profiles, learned_crefs) = (ir.result, ir.profiles, ir.learned_crefs);

        assert!(!profiles.is_empty(), "need conflicts to build DAG");
        assert_eq!(
            profiles.len(),
            learned_crefs.len(),
            "one learned CRef per profile"
        );

        // Build the proof DAG
        let dag = ProofDag::build(&profiles, &learned_crefs);
        let summary = dag.summary();

        println!("\n=== Level 3: Proof DAG Topology (n={n}, 10K budget) ===");
        println!(
            "  Result: {:?}",
            match &result {
                SolveResult::Sat(_) => "SAT",
                SolveResult::Unsat => "UNSAT",
                SolveResult::Unknown => "UNKNOWN",
            }
        );
        println!(
            "  Nodes: {} total ({} input, {} learned)",
            summary.num_nodes, summary.num_input, summary.num_learned
        );
        println!(
            "  Edges: {} total, {} unique",
            summary.total_edges, summary.unique_edges
        );
        println!(
            "  Sharing ratio: {:.3} (1.0 = tree, lower = more sharing)",
            summary.sharing_ratio
        );
        println!("  Max depth: {}", summary.max_depth);
        println!(
            "  Max fan-out: {} (most-reused clause)",
            summary.max_fan_out
        );
        println!(
            "  Max fan-in: {} (deepest resolution chain)",
            summary.max_fan_in
        );
        println!("  Avg fan-out: {:.2}", summary.avg_fan_out);

        // ── Structural invariants ──

        // Every learned clause should appear as a node
        assert!(summary.num_learned > 0, "should have learned clauses");

        // Learned clauses have fan_in > 0 (they were derived by resolution)
        // Input clauses have fan_in == 0
        let learned_set: std::collections::HashSet<CRef> =
            dag.learned_crefs.iter().copied().collect();
        for (&cref, node) in &dag.nodes {
            if !learned_set.contains(&cref) {
                // Input clauses: never produced by resolution
                assert_eq!(node.fan_in, 0, "input clause should have fan_in=0");
                assert_eq!(node.depth, 0, "input clause must have depth 0");
            }
            // Learned clauses at depth > 0 is expected (derived from
            // input clauses or earlier learned clauses)
        }

        // Sharing ratio: should be < 1.0 for any non-trivial solve
        // (many chains share the same reason clauses)
        assert!(
            summary.sharing_ratio < 1.0,
            "should have sharing (ratio={:.3})",
            summary.sharing_ratio
        );

        // Width profile
        let widths = dag.width_profile();
        println!("  Width profile (first 10 depths):");
        for (d, &w) in widths.iter().enumerate().take(10) {
            println!("    depth {d}: {w} nodes");
        }

        // Depth 0 is the widest layer (input clauses that were used as reasons)
        assert!(widths[0] > 0, "depth 0 should have nodes");

        // Variable centrality (top 10)
        let centrality = ProofDag::variable_centrality(&profiles);
        println!("  Top 10 pivot variables (var, count):");
        for &(var, count) in centrality.iter().take(10) {
            println!("    x{var}: {count}");
        }

        // The most central variable should have significant frequency
        if let Some(&(top_var, top_count)) = centrality.first() {
            assert!(
                top_count > 100,
                "top pivot var x{} has count {} — expected > 100 on 200-var instance",
                top_var,
                top_count
            );
        }

        // Total edges should equal sum of all resolution chain lengths
        let expected_edges: usize = profiles.iter().map(|p| p.resolution_chain.len()).sum();
        assert_eq!(
            dag.total_edges, expected_edges,
            "total edges must equal sum of chain lengths"
        );
    }

    #[test]
    fn level_4_correlations() {
        // Level 4: The payoff. Four correlations between DAG topology and
        // solver behavior. Any strong correlation (|r| > 0.3) suggests a
        // concrete heuristic improvement.
        use crate::analyze::{
            correlate_depth_vs_clause_reuse, correlate_depth_vs_next_bcp,
            correlate_pivot_vs_gradient,
        };
        use crate::bench::generate_k_sat;

        let n = 200u32;
        let num_clauses = ((n as f64) * 4.267).ceil() as usize;

        // Run multiple seeds for statistical robustness
        let mut all_correlations: Vec<[f64; 4]> = Vec::new();

        println!("\n=== Level 4: Topology ↔ Solver Correlations ===");

        for seed in [42u64, 7, 99, 123, 256] {
            let db = generate_k_sat(n, num_clauses, 3, seed);

            // Compute cold gradient (all unassigned) on the original formula
            let cold_assignments: Vec<Option<bool>> = vec![None; n as usize];
            let tg = crate::gradient::gradient_at_trail(&db, n, &cold_assignments);

            let ir = solve_instrumented(db, n, 10_000);

            if ir.profiles.len() < 100 {
                println!(
                    "  seed {seed}: only {} conflicts, skipping",
                    ir.profiles.len()
                );
                continue;
            }

            // Correlation 1: depth(C) vs bcp_props(C+1)
            let c1 = correlate_depth_vs_next_bcp(&ir.profiles);

            // Correlation 2: depth vs learned clause reuse
            let c2 = correlate_depth_vs_clause_reuse(&ir.profiles, &ir.learned_crefs);

            // Correlation 3 (fixed): pivot centrality vs actual VSIDS activity (Spearman rank)
            let c3 =
                crate::analyze::correlate_centrality_vs_vsids(&ir.profiles, &ir.vsids_activities);

            // Correlation 4: pivot frequency vs gradient magnitude
            let c4 = correlate_pivot_vs_gradient(&ir.profiles, &tg.magnitudes);

            let result_str = match &ir.result {
                SolveResult::Sat(_) => "SAT",
                SolveResult::Unsat => "UNSAT",
                SolveResult::Unknown => "UNK",
            };
            println!(
                "  seed {seed} ({result_str}, {} conflicts):",
                ir.profiles.len()
            );
            println!(
                "    C1 depth→next_bcp:     r={:+.3}  r²={:.3}  n={}",
                c1.r, c1.r_squared, c1.n
            );
            println!(
                "    C2 depth→clause_reuse: r={:+.3}  r²={:.3}  n={}",
                c2.r, c2.r_squared, c2.n
            );
            println!(
                "    C3 centrality→VSIDS:   r={:+.3}  r²={:.3}  n={} (Spearman rank)",
                c3.r, c3.r_squared, c3.n
            );
            println!(
                "    C4 pivot→gradient:     r={:+.3}  r²={:.3}  n={}",
                c4.r, c4.r_squared, c4.n
            );

            all_correlations.push([c1.r, c2.r, c3.r, c4.r]);
        }

        // Aggregate: mean |r| across seeds
        if !all_correlations.is_empty() {
            let k = all_correlations.len() as f64;
            let names = [
                "depth→next_bcp",
                "depth→clause_reuse",
                "centrality→VSIDS",
                "pivot→gradient",
            ];
            println!("\n  Aggregate across {} seeds:", all_correlations.len());
            for (j, name) in names.iter().enumerate() {
                let mean_r: f64 = all_correlations.iter().map(|c| c[j]).sum::<f64>() / k;
                let mean_abs_r: f64 = all_correlations.iter().map(|c| c[j].abs()).sum::<f64>() / k;
                let consistent_sign = all_correlations
                    .iter()
                    .all(|c| c[j].signum() == all_correlations[0][j].signum());
                let sign_tag = if consistent_sign {
                    "consistent"
                } else {
                    "mixed"
                };
                println!(
                    "    {name}: mean_r={mean_r:+.3}, mean_|r|={mean_abs_r:.3} ({sign_tag} sign)"
                );
            }

            // Significance threshold: |r| > 0.1 consistently across seeds
            // Strong signal: |r| > 0.3
            // The test passes regardless — this is an observatory, not a gate
            println!("\n  Legend: |r| > 0.3 = strong, |r| > 0.1 = weak, < 0.1 = noise");
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn depth_weighted_deletion_ab_test() {
        // C2 experiment: does depth-weighted clause deletion reduce conflict count?
        //
        // A = baseline LBD-only deletion (depth_weight = 0.0)
        // B = depth-weighted deletion (depth_weight = α)
        //
        // Run on 20 seeds × 200-var phase transition, 50K conflict budget.
        // Compare mean conflict counts.
        use crate::bench::generate_k_sat;

        let n = 200u32;
        let num_clauses = ((n as f64) * 4.267).ceil() as usize;
        let conflict_budget = 50_000u64;
        let num_seeds = 20;

        // Test several depth weights
        let weights = [0.0, 0.1, 0.25, 0.5, 1.0];

        println!("\n=== C2 Experiment: Depth-Weighted Deletion ===");
        println!("  n={n}, ratio=4.267, budget={conflict_budget}, seeds=0..{num_seeds}");
        println!(
            "  {:>8} {:>10} {:>10} {:>10} {:>10}",
            "α", "conflicts", "solved", "unknown", "vs_base%"
        );

        let mut baseline_conflicts = 0u64;

        for &weight in &weights {
            let mut total_conflicts = 0u64;
            let mut solved = 0u32;
            let mut unknown = 0u32;

            for seed in 0..num_seeds {
                let db = generate_k_sat(n, num_clauses, 3, seed);
                let ir = solve_instrumented_depth_weighted(db, n, conflict_budget, weight);
                let n_conflicts = ir.profiles.len() as u64;
                total_conflicts += n_conflicts;
                match ir.result {
                    SolveResult::Sat(_) | SolveResult::Unsat => solved += 1,
                    SolveResult::Unknown => unknown += 1,
                }
            }

            if weight == 0.0 {
                baseline_conflicts = total_conflicts;
            }

            let vs_base = if baseline_conflicts > 0 {
                ((total_conflicts as f64 / baseline_conflicts as f64) - 1.0) * 100.0
            } else {
                0.0
            };

            println!(
                "  {:>8.2} {:>10} {:>10} {:>10} {:>+10.1}",
                weight, total_conflicts, solved, unknown, vs_base
            );
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn pivot_augmented_vsids_ab_test() {
        // C3 follow-up: does feeding pivot centrality back into VSIDS
        // reduce conflict count?
        //
        // C3 showed r_s = +0.524 between VSIDS activity and pivot frequency,
        // meaning VSIDS captures ~26% of pivot centrality. The 74% gap
        // represents structural information that VSIDS misses.
        //
        // Experiment: during conflict analysis, bump pivot variables by
        // `scale × increment` in addition to the standard learned-clause bumps.
        //
        // A = baseline (pivot_bump_scale = 0.0)
        // B = pivot-augmented (pivot_bump_scale > 0.0)
        //
        // 20 seeds × 200-var phase transition, 50K conflict budget.
        use crate::bench::generate_k_sat;

        let n = 200u32;
        let num_clauses = ((n as f64) * 4.267).ceil() as usize;
        let conflict_budget = 50_000u64;
        let num_seeds = 20;

        let scales = [0.0, 0.25, 0.5, 1.0, 2.0];

        println!("\n=== C3 Experiment: Pivot-Augmented VSIDS ===");
        println!("  n={n}, ratio=4.267, budget={conflict_budget}, seeds=0..{num_seeds}");
        println!(
            "  {:>8} {:>10} {:>10} {:>10} {:>10}",
            "scale", "conflicts", "solved", "unknown", "vs_base%"
        );

        let mut baseline_conflicts = 0u64;
        let mut baseline_solved = 0u32;

        for &scale in &scales {
            let mut total_conflicts = 0u64;
            let mut solved = 0u32;
            let mut unknown = 0u32;

            for seed in 0..num_seeds {
                let db = generate_k_sat(n, num_clauses, 3, seed);
                let ir = solve_instrumented_pivot_augmented(db, n, conflict_budget, scale);
                let n_conflicts = ir.profiles.len() as u64;
                total_conflicts += n_conflicts;
                match ir.result {
                    SolveResult::Sat(_) | SolveResult::Unsat => solved += 1,
                    SolveResult::Unknown => unknown += 1,
                }
            }

            if scale == 0.0 {
                baseline_conflicts = total_conflicts;
                baseline_solved = solved;
            }

            let vs_base = if baseline_conflicts > 0 {
                ((total_conflicts as f64 / baseline_conflicts as f64) - 1.0) * 100.0
            } else {
                0.0
            };

            println!(
                "  {:>8.2} {:>10} {:>10} {:>10} {:>+10.1}",
                scale, total_conflicts, solved, unknown, vs_base
            );
        }

        // Also run on 300-var to check scaling behavior
        let n_large = 300u32;
        let num_clauses_large = ((n_large as f64) * 4.267).ceil() as usize;

        println!("\n  --- Scaling check: n={n_large} ---");
        println!(
            "  {:>8} {:>10} {:>10} {:>10} {:>10}",
            "scale", "conflicts", "solved", "unknown", "vs_base%"
        );

        let mut baseline_large = 0u64;

        for &scale in &scales {
            let mut total_conflicts = 0u64;
            let mut solved = 0u32;
            let mut unknown = 0u32;

            for seed in 0..num_seeds {
                let db = generate_k_sat(n_large, num_clauses_large, 3, seed);
                let ir = solve_instrumented_pivot_augmented(db, n_large, conflict_budget, scale);
                let n_conflicts = ir.profiles.len() as u64;
                total_conflicts += n_conflicts;
                match ir.result {
                    SolveResult::Sat(_) | SolveResult::Unsat => solved += 1,
                    SolveResult::Unknown => unknown += 1,
                }
            }

            if scale == 0.0 {
                baseline_large = total_conflicts;
            }

            let vs_base = if baseline_large > 0 {
                ((total_conflicts as f64 / baseline_large as f64) - 1.0) * 100.0
            } else {
                0.0
            };

            println!(
                "  {:>8.2} {:>10} {:>10} {:>10} {:>+10.1}",
                scale, total_conflicts, solved, unknown, vs_base
            );
        }

        println!("\n  Baseline 200v: solved {baseline_solved}/{num_seeds}, conflicts {baseline_conflicts}");
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn pivot_scaling_diagnosis() {
        // Diagnose WHY pivot-augmented VSIDS doesn't help at 300v:
        //   (a) Budget ceiling? → re-run with 200K budget
        //   (b) Pivot entropy collapse? → measure Shannon entropy of pivot freq
        //
        // If (a): budget was masking the signal; mechanism scales.
        // If (b): pivot distribution flattens at larger n; mechanism is
        //         fundamentally size-limited.
        use crate::analyze::pivot_frequency;
        use crate::bench::generate_k_sat;

        let num_seeds = 20;
        let scales = [0.0, 0.5, 1.0];

        // ── Part A: 300v with 200K budget ──────────────────────────────
        let n = 300u32;
        let num_clauses = ((n as f64) * 4.267).ceil() as usize;
        let budget = 200_000u64;

        println!("\n=== Scaling Diagnosis: 300v × 200K budget ===");
        println!(
            "  {:>8} {:>10} {:>10} {:>10} {:>10}",
            "scale", "conflicts", "solved", "unknown", "vs_base%"
        );

        let mut baseline = 0u64;

        for &scale in &scales {
            let mut total_conflicts = 0u64;
            let mut solved = 0u32;
            let mut unknown = 0u32;

            for seed in 0..num_seeds {
                let db = generate_k_sat(n, num_clauses, 3, seed);
                let ir = solve_instrumented_pivot_augmented(db, n, budget, scale);
                total_conflicts += ir.profiles.len() as u64;
                match ir.result {
                    SolveResult::Sat(_) | SolveResult::Unsat => solved += 1,
                    SolveResult::Unknown => unknown += 1,
                }
            }

            if scale == 0.0 {
                baseline = total_conflicts;
            }

            let vs_base = if baseline > 0 {
                ((total_conflicts as f64 / baseline as f64) - 1.0) * 100.0
            } else {
                0.0
            };

            println!(
                "  {:>8.2} {:>10} {:>10} {:>10} {:>+10.1}",
                scale, total_conflicts, solved, unknown, vs_base
            );
        }

        // ── Part B: Pivot frequency entropy at 200v vs 300v ────────────
        // Shannon entropy H = -Σ p_i log2(p_i) where p_i = freq_i / total
        // Normalized entropy H/log2(n_pivots) ∈ [0,1]. Near 1 = uniform
        // (pivots are interchangeable), near 0 = skewed (some pivots dominate).
        println!("\n=== Pivot Frequency Entropy ===");
        println!(
            "  {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "n", "seed", "n_pivots", "total", "H_bits", "H_norm", "max_freq"
        );

        for &nv in &[200u32, 300u32] {
            let nc = ((nv as f64) * 4.267).ceil() as usize;
            // Use 50K budget to match original conditions
            for seed in [0u64, 5, 10, 15] {
                let db = generate_k_sat(nv, nc, 3, seed);
                let ir = solve_instrumented(db, nv, 50_000);
                let freq = pivot_frequency(&ir.profiles);

                if freq.is_empty() {
                    continue;
                }

                let total: usize = freq.values().sum();
                let n_pivots = freq.len();
                let max_f = *freq.values().max().unwrap();

                // Shannon entropy
                let h: f64 = freq
                    .values()
                    .map(|&f| {
                        let p = f as f64 / total as f64;
                        if p > 0.0 {
                            -p * p.log2()
                        } else {
                            0.0
                        }
                    })
                    .sum();

                let h_max = (n_pivots as f64).log2();
                let h_norm = if h_max > 0.0 { h / h_max } else { 0.0 };

                println!(
                    "  {:>6} {:>6} {:>10} {:>10} {:>10.3} {:>10.4} {:>10}",
                    nv, seed, n_pivots, total, h, h_norm, max_f
                );
            }
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn combined_gradient_pivot_ab_test() {
        // Combined experiment: gradient phase hints + pivot decision bumps.
        //
        // These are orthogonal signals:
        //   - Gradient: shapes POLARITY (which value to try)
        //   - Pivot: shapes VARIABLE ORDER (which variable to decide)
        //
        // If they're truly orthogonal, the combined effect should be
        // multiplicative (or at least additive). If they interfere, combined
        // could be worse than either alone.
        //
        // Test: 20 seeds × 200v, 50K budget. Compare:
        //   (A) baseline
        //   (B) gradient-only (interval=50, boost=1.0)
        //   (C) pivot-only (scale=0.5)
        //   (D) combined (gradient + pivot)
        use crate::bench::generate_k_sat;

        let n = 200u32;
        let num_clauses = ((n as f64) * 4.267).ceil() as usize;
        let budget = 50_000u64;
        let num_seeds = 20;

        struct Config {
            name: &'static str,
            gradient_interval: u64,
            gradient_boost: f64,
            pivot_scale: f64,
        }

        let configs = [
            Config {
                name: "A: baseline",
                gradient_interval: 0,
                gradient_boost: 0.0,
                pivot_scale: 0.0,
            },
            Config {
                name: "B: gradient-only",
                gradient_interval: 50,
                gradient_boost: 1.0,
                pivot_scale: 0.0,
            },
            Config {
                name: "C: pivot-only",
                gradient_interval: 0,
                gradient_boost: 0.0,
                pivot_scale: 0.5,
            },
            Config {
                name: "D: combined",
                gradient_interval: 50,
                gradient_boost: 1.0,
                pivot_scale: 0.5,
            },
            Config {
                name: "E: combined(1.0)",
                gradient_interval: 50,
                gradient_boost: 1.0,
                pivot_scale: 1.0,
            },
            Config {
                name: "F: grad+pivot(K200)",
                gradient_interval: 200,
                gradient_boost: 1.0,
                pivot_scale: 0.5,
            },
        ];

        println!("\n=== Combined Gradient+Pivot A/B Test ===");
        println!("  n={n}, ratio=4.267, budget={budget}, seeds=0..{num_seeds}");
        println!(
            "  {:<25} {:>10} {:>10} {:>10} {:>10}",
            "config", "conflicts", "solved", "unknown", "vs_base%"
        );
        println!("  {}", "-".repeat(70));

        let mut baseline_conflicts = 0u64;

        for cfg in &configs {
            let mut total_conflicts = 0u64;
            let mut solved = 0u32;
            let mut unknown = 0u32;

            for seed in 0..num_seeds {
                let db = generate_k_sat(n, num_clauses, 3, seed);
                let (result, _stats) = solve_watched_combined(
                    db,
                    n,
                    cfg.gradient_interval,
                    cfg.gradient_boost,
                    cfg.pivot_scale,
                    budget,
                );

                match result {
                    SolveResult::Sat(_) | SolveResult::Unsat => solved += 1,
                    SolveResult::Unknown => unknown += 1,
                }
                // Count conflicts from Unknown instances too
                total_conflicts += _stats.conflicts;
            }

            if cfg.pivot_scale == 0.0 && cfg.gradient_interval == 0 {
                baseline_conflicts = total_conflicts;
            }

            let vs_base = if baseline_conflicts > 0 {
                ((total_conflicts as f64 / baseline_conflicts as f64) - 1.0) * 100.0
            } else {
                0.0
            };

            println!(
                "  {:<25} {:>10} {:>10} {:>10} {:>+10.1}",
                cfg.name, total_conflicts, solved, unknown, vs_base
            );
        }
    }

    // ── Theory solver integration tests ──

    use crate::theory::{TheoryContext, TheoryProp, TheoryResult};

    /// A simple theory: variables `a` and `b` cannot both be true.
    /// Equivalent to adding the clause (¬a ∨ ¬b), but enforced through
    /// the theory mechanism to exercise the full DPLL(T) integration.
    struct ExclusiveTheory {
        var_a: u32,
        var_b: u32,
    }

    impl TheorySolver for ExclusiveTheory {
        fn check(&mut self, ctx: &TheoryContext<'_>) -> TheoryResult {
            let a = ctx.trail.value(self.var_a);
            let b = ctx.trail.value(self.var_b);
            match (a, b) {
                (Some(true), Some(true)) => {
                    // Both true → conflict: theory lemma ¬a ∨ ¬b
                    TheoryResult::Conflict(vec![Lit::neg(self.var_a), Lit::neg(self.var_b)])
                }
                (Some(true), None) => {
                    // a=true, b unassigned → propagate ¬b
                    TheoryResult::Propagate(vec![TheoryProp {
                        lit: Lit::neg(self.var_b),
                        key: 0,
                    }])
                }
                (None, Some(true)) => {
                    // b=true, a unassigned → propagate ¬a
                    TheoryResult::Propagate(vec![TheoryProp {
                        lit: Lit::neg(self.var_a),
                        key: 1,
                    }])
                }
                _ => TheoryResult::Consistent,
            }
        }

        fn backtrack(&mut self, _new_level: u32) {
            // Stateless theory — nothing to retract
        }

        fn explain(&mut self, _lit: Lit, _key: u32) -> Vec<Lit> {
            // Both keys have the same explanation: ¬a ∨ ¬b
            vec![Lit::neg(self.var_a), Lit::neg(self.var_b)]
        }
    }

    #[test]
    fn theory_exclusive_sat() {
        // Formula: (x0 ∨ x1) ∧ (x1 ∨ x2)
        // Theory: x0 and x1 cannot both be true
        // Expected: SAT (e.g., x0=true, x1=false, x2=true)
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1)]); // x0 ∨ x1
        db.add_clause(vec![Lit::pos(1), Lit::pos(2)]); // x1 ∨ x2

        let mut theory = ExclusiveTheory { var_a: 0, var_b: 1 };

        match solve_with_theory(db, 3, &mut theory) {
            SolveResult::Sat(assign) => {
                // Theory constraint: not both true
                assert!(
                    !(assign[0] && assign[1]),
                    "theory violated: x0={} x1={} — both true",
                    assign[0],
                    assign[1]
                );
                // Clause (x0 ∨ x1) satisfied
                assert!(assign[0] || assign[1], "clause (x0 ∨ x1) violated");
                // Clause (x1 ∨ x2) satisfied
                assert!(assign[1] || assign[2], "clause (x1 ∨ x2) violated");
            }
            SolveResult::Unsat => panic!("expected SAT"),
            SolveResult::Unknown => panic!("unexpected Unknown"),
        }
    }

    #[test]
    fn theory_exclusive_unsat() {
        // Formula: (x0) ∧ (x1)  — forces both true
        // Theory: x0 and x1 cannot both be true
        // Expected: UNSAT
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]); // x0 must be true
        db.add_clause(vec![Lit::pos(1)]); // x1 must be true

        let mut theory = ExclusiveTheory { var_a: 0, var_b: 1 };

        match solve_with_theory(db, 2, &mut theory) {
            SolveResult::Unsat => {}
            SolveResult::Sat(a) => panic!("expected UNSAT, got SAT: {:?}", a),
            SolveResult::Unknown => panic!("unexpected Unknown"),
        }
    }

    #[test]
    fn theory_no_theory_matches_pure_sat() {
        // Verify NoTheory produces the same result as solve_watched
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
        let inst1 = dimacs::parse_dimacs_str(cnf).unwrap();
        let inst2 = dimacs::parse_dimacs_str(cnf).unwrap();

        let r1 = solve_watched(inst1.db, inst1.num_vars);
        let r2 = solve_with_theory(inst2.db, inst2.num_vars, &mut NoTheory);

        match (&r1, &r2) {
            (SolveResult::Unsat, SolveResult::Unsat) => {}
            (SolveResult::Sat(_), SolveResult::Sat(_)) => {}
            _ => panic!(
                "NoTheory and pure SAT disagree: {:?} vs {:?}",
                matches!(r1, SolveResult::Sat(_)),
                matches!(r2, SolveResult::Sat(_))
            ),
        }
    }
}
