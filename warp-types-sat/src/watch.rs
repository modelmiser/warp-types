//! Two-watched-literal BCP — O(propagations) instead of O(propagations × clauses).
//!
//! Each clause watches two literals. When a watched literal becomes false,
//! we inspect only that clause — not every clause in the database. Either:
//! - A replacement watch exists (clause has ≥2 unresolved literals) → swap watch
//! - The clause is unit (propagate the other watched literal)
//! - The clause is a conflict (all literals false)
//!
//! No watch restoration on backtrack. The two-watched-literal invariant
//! survives because unassigning a variable only strengthens it.

use crate::bcp::{BcpResult, ClauseDb};
use crate::literal::Lit;
use crate::phase::Propagate;
use crate::trail::{BcpTrail, Trail};

/// A single entry in a literal's watch list.
///
/// Stores the clause index (as u32 for cache density) and a "blocker" literal.
/// The blocker is a speculative hint — if it evaluates to true, the clause is
/// satisfied and we skip it without any clause DB lookup. This eliminates
/// 50-70% of clause lookups in typical BCP (MiniSat's key optimization).
///
/// Size: 8 bytes (same as a bare `usize` on 64-bit), but carries the blocker for free.
#[derive(Clone, Copy)]
struct WatchEntry {
    /// Clause index in the ClauseDb.
    clause: u32,
    /// Blocker: the other watched literal at watch setup time. If this literal
    /// is true, the clause is satisfied — skip without touching the clause DB.
    /// May be stale (clause's watches may have moved), but a stale-but-true
    /// blocker is still a valid skip.
    blocker: Lit,
}

/// Two-watched-literal data structure with blocking literals.
pub struct Watches {
    /// Per-literal watch lists with blocker hints.
    lists: Vec<Vec<WatchEntry>>,
    /// Per-clause watched literal pair.
    watched: Vec<[Lit; 2]>,
    /// Trail position processed up to (for incremental propagation).
    queue_head: usize,
    /// Whether the one-time unit/empty clause scan has been performed.
    initial_scan_done: bool,
}

impl Watches {
    /// Initialize watches for all clauses. Clauses with <2 literals get a
    /// placeholder (handled as unit/empty in the BCP loop directly).
    pub fn new(db: &ClauseDb, num_vars: u32) -> Self {
        let num_lits = 2 * num_vars as usize;
        let mut lists = vec![Vec::new(); num_lits];
        let mut watched = Vec::with_capacity(db.len());

        for ci in 0..db.len() {
            let lits = &db.clause(ci).literals;
            if lits.len() < 2 {
                watched.push([Lit::pos(0), Lit::pos(0)]); // placeholder
                continue;
            }
            let w0 = lits[0];
            let w1 = lits[1];
            // Each watch entry stores the *other* watched literal as blocker
            lists[w0.code() as usize].push(WatchEntry { clause: ci as u32, blocker: w1 });
            lists[w1.code() as usize].push(WatchEntry { clause: ci as u32, blocker: w0 });
            watched.push([w0, w1]);
        }

        Watches {
            lists,
            watched,
            queue_head: 0,
            initial_scan_done: false,
        }
    }

    /// Add watches for a newly learned clause.
    pub fn add_clause(&mut self, db: &ClauseDb, clause_idx: usize) {
        let lits = &db.clause(clause_idx).literals;
        if lits.len() < 2 {
            self.watched.push([Lit::pos(0), Lit::pos(0)]);
            return;
        }
        let w0 = lits[0];
        let w1 = lits[1];
        let ci = clause_idx as u32;
        self.lists[w0.code() as usize].push(WatchEntry { clause: ci, blocker: w1 });
        self.lists[w1.code() as usize].push(WatchEntry { clause: ci, blocker: w0 });
        self.watched.push([w0, w1]);
    }

    /// Reset queue head after backtracking (trail is shorter now).
    pub fn notify_backtrack(&mut self, new_trail_len: usize) {
        self.queue_head = self.queue_head.min(new_trail_len);
    }

    /// Set queue head and mark initial scan as done (used after watch rebuild).
    pub fn set_queue_head(&mut self, pos: usize) {
        self.queue_head = pos;
        self.initial_scan_done = true;
    }
}

/// Evaluate a literal: Some(true) if satisfied, Some(false) if falsified, None if unassigned.
#[inline]
fn eval_lit(lit: Lit, assignments: &[Option<bool>]) -> Option<bool> {
    assignments[lit.var() as usize].map(|val| if lit.is_negated() { !val } else { val })
}

/// Unchecked eval_lit for the BCP hot loop.
///
/// # Safety
/// `lit.var()` must be < `assignments.len()`. This is guaranteed when:
/// - All literals come from the clause DB
/// - `db.max_variable() < num_vars` was asserted at solver startup
/// - `assignments.len() == num_vars`
#[inline]
unsafe fn eval_lit_unchecked(lit: Lit, assignments: &[Option<bool>]) -> Option<bool> {
    unsafe { *assignments.get_unchecked(lit.var() as usize) }
        .map(|val| if lit.is_negated() { !val } else { val })
}

/// Watched-literal BCP. Processes trail entries from `queue_head` onward.
///
/// Complexity: O(propagations × avg_watches_per_literal).
/// For random 3-SAT at ratio 4.267: ~13 watch entries per literal on average,
/// vs scanning all ~4n clauses in the old BCP.
pub fn run_bcp_watched(
    db: &ClauseDb,
    watches: &mut Watches,
    trail: &mut Trail,
    _phase: &crate::session::SolverSession<'_, Propagate>,
) -> BcpResult {
    // Split trail: bt.assigns is a &mut [Option<bool>] (stable pointer).
    // bt.record_propagation writes entries/var_position (disjoint fields),
    // so the compiler keeps the assigns pointer in a register across propagations.
    let mut bt = trail.bcp_split();

    // Handle unit/empty original clauses once at initialization.
    if !watches.initial_scan_done {
        watches.initial_scan_done = true;
        for ci in 0..db.len() {
            if db.is_deleted(ci) {
                continue;
            }
            let lits = &db.clause(ci).literals;
            if lits.is_empty() {
                return BcpResult::Conflict { clause_index: ci };
            }
            if lits.len() == 1 {
                let lit = lits[0];
                match eval_lit(lit, bt.assigns) {
                    None => bt.record_propagation(lit, ci),
                    Some(false) => return BcpResult::Conflict { clause_index: ci },
                    Some(true) => {}
                }
            }
        }
    }

    // Main propagation loop: process trail entries from queue_head.
    // bt.assigns pointer is stable throughout — no re-derivation after propagations.
    //
    // SAFETY of unchecked indexing throughout this loop:
    // - All literals come from clauses in the DB or from watched[] (derived from DB)
    // - solve_cdcl_core_inner asserts db.max_variable() < num_vars at startup
    // - bt.assigns.len() == num_vars (from Trail::new)
    // - Therefore lit.var() < bt.assigns.len() for every literal encountered
    while watches.queue_head < bt.len() {
        let assigned_lit = bt.entry_at(watches.queue_head).lit;
        watches.queue_head += 1;
        let false_lit = assigned_lit.complement();

        let mut ws = std::mem::take(&mut watches.lists[false_lit.code() as usize]);
        let mut j = 0;

        let mut i = 0;
        while i < ws.len() {
            let entry = ws[i];
            let ci = entry.clause as usize;

            if db.is_deleted(ci) {
                i += 1;
                continue;
            }

            // ── Blocker fast-path ──
            // SAFETY: blocker literal comes from a clause in the DB (see above)
            if unsafe { eval_lit_unchecked(entry.blocker, bt.assigns) } == Some(true) {
                ws[j] = entry;
                j += 1;
                i += 1;
                continue;
            }

            let [w0, w1] = watches.watched[ci];

            let (partner, watch_pos) = if w0 == false_lit {
                (w1, 0usize)
            } else {
                debug_assert_eq!(w1, false_lit, "clause {ci} in watch list for {false_lit} but watches {w0},{w1}");
                (w0, 1usize)
            };

            // SAFETY: partner is one of the watched literals, from the DB
            if unsafe { eval_lit_unchecked(partner, bt.assigns) } == Some(true) {
                ws[j] = WatchEntry { clause: entry.clause, blocker: partner };
                j += 1;
                i += 1;
                continue;
            }

            // Search clause for a replacement watch.
            let clause_lits = db.clause(ci).literals;
            let mut replacement = None;
            for &lit in clause_lits {
                if lit == w0 || lit == w1 {
                    continue;
                }
                // SAFETY: lit comes from a clause in the DB
                if unsafe { eval_lit_unchecked(lit, bt.assigns) } != Some(false) {
                    replacement = Some(lit);
                    break;
                }
            }

            if let Some(new_watch) = replacement {
                watches.watched[ci][watch_pos] = new_watch;
                watches.lists[new_watch.code() as usize].push(
                    WatchEntry { clause: entry.clause, blocker: partner }
                );
                i += 1;
                continue;
            }

            ws[j] = entry;
            j += 1;
            i += 1;

            // SAFETY: partner is a watched literal from the DB
            let partner_val = unsafe { eval_lit_unchecked(partner, bt.assigns) };
            if partner_val == Some(false) {
                while i < ws.len() {
                    ws[j] = ws[i];
                    j += 1;
                    i += 1;
                }
                ws.truncate(j);
                watches.lists[false_lit.code() as usize] = ws;
                return BcpResult::Conflict { clause_index: ci };
            } else if partner_val.is_none() {
                bt.record_propagation(partner, ci);
            }
            // else: partner is true — satisfied during this BCP round
        }

        ws.truncate(j);
        watches.lists[false_lit.code() as usize] = ws;
    }

    BcpResult::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session;

    fn bcp_after_decision(
        db: &ClauseDb,
        watches: &mut Watches,
        trail: &mut Trail,
    ) -> BcpResult {
        session::with_session(|s| {
            let p = s.decide().propagate();
            run_bcp_watched(db, watches, trail, &p)
        })
    }

    #[test]
    fn simple_unit_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);

        let mut w = Watches::new(&db, 3);
        let mut trail = Trail::new(3);
        trail.new_decision(Lit::pos(0));
        assert_eq!(bcp_after_decision(&db, &mut w, &mut trail), BcpResult::Ok);
        assert_eq!(trail.value(1), Some(true));
        assert_eq!(trail.value(2), Some(true));
    }

    #[test]
    fn conflict_via_unit_clause() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0)]); // unit: ¬x0

        let mut w = Watches::new(&db, 1);
        let mut trail = Trail::new(1);
        trail.new_decision(Lit::pos(0));
        assert_eq!(
            bcp_after_decision(&db, &mut w, &mut trail),
            BcpResult::Conflict { clause_index: 0 }
        );
    }

    #[test]
    fn chain_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::neg(2), Lit::pos(3)]);

        let mut w = Watches::new(&db, 4);
        let mut trail = Trail::new(4);
        trail.new_decision(Lit::pos(0));
        let before = trail.len();
        assert_eq!(bcp_after_decision(&db, &mut w, &mut trail), BcpResult::Ok);
        assert_eq!(trail.len() - before, 3);
        assert_eq!(trail.value(3), Some(true));
    }

    #[test]
    fn conflict_after_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(0), Lit::neg(1)]);

        let mut w = Watches::new(&db, 2);
        let mut trail = Trail::new(2);
        trail.new_decision(Lit::pos(0));
        match bcp_after_decision(&db, &mut w, &mut trail) {
            BcpResult::Conflict { .. } => {}
            other => panic!("expected Conflict, got {:?}", other),
        }
    }

    #[test]
    fn three_literal_clause_finds_replacement() {
        // (¬x0 ∨ x1 ∨ x2): when x0=true, watches ¬x0 becomes false.
        // Should find x1 or x2 as replacement watch, NOT propagate.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1), Lit::pos(2)]);

        let mut w = Watches::new(&db, 3);
        let mut trail = Trail::new(3);
        trail.new_decision(Lit::pos(0));
        assert_eq!(bcp_after_decision(&db, &mut w, &mut trail), BcpResult::Ok);
        // Neither x1 nor x2 should be propagated (clause has 2 unresolved lits).
        assert_eq!(trail.value(1), None);
        assert_eq!(trail.value(2), None);
    }

    #[test]
    fn watched_agrees_with_original_bcp() {
        use crate::bench::generate_3sat_phase_transition;
        use crate::bcp;

        for seed in 0..10 {
            let db = generate_3sat_phase_transition(30, seed);

            // Original BCP
            let mut trail1 = Trail::new(30);
            trail1.new_decision(Lit::pos(0));
            let r1 = session::with_session(|s| {
                let p = s.decide().propagate();
                bcp::run_bcp(&db, &mut trail1, &p)
            });

            // Watched-literal BCP
            let mut w = Watches::new(&db, 30);
            let mut trail2 = Trail::new(30);
            trail2.new_decision(Lit::pos(0));
            let r2 = session::with_session(|s| {
                let p = s.decide().propagate();
                run_bcp_watched(&db, &mut w, &mut trail2, &p)
            });

            // Must agree on Ok vs Conflict.
            assert_eq!(
                matches!(r1, BcpResult::Ok),
                matches!(r2, BcpResult::Ok),
                "seed {seed}: old={r1:?}, new={r2:?}"
            );
            // If both Ok, assignments must match.
            if matches!(r1, BcpResult::Ok) {
                assert_eq!(
                    trail1.assignments(),
                    trail2.assignments(),
                    "seed {seed}: assignments diverge"
                );
            }
        }
    }
}
