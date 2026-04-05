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
use crate::trail::Trail;

/// Two-watched-literal data structure.
pub struct Watches {
    /// Per-literal watch lists. `lists[lit.code()]` = clause indices watching `lit`.
    lists: Vec<Vec<usize>>,
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
            lists[w0.code() as usize].push(ci);
            lists[w1.code() as usize].push(ci);
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
        self.lists[w0.code() as usize].push(clause_idx);
        self.lists[w1.code() as usize].push(clause_idx);
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
    trail.ensure_capacity(db.max_variable() as usize + 1);

    // Handle unit/empty original clauses once at initialization.
    // Must not re-run after restarts — deleted clauses have empty literals
    // that look like trivially-false clauses, and the O(n) scan is expensive.
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
                match eval_lit(lit, trail.assignments()) {
                    None => trail.record_propagation(lit, ci),
                    Some(false) => return BcpResult::Conflict { clause_index: ci },
                    Some(true) => {}
                }
            }
        }
    }

    // Main propagation loop: process trail entries from queue_head.
    while watches.queue_head < trail.len() {
        let assigned_lit = trail.entries()[watches.queue_head].lit;
        watches.queue_head += 1;
        let false_lit = assigned_lit.complement(); // this literal just became false

        // Take the watch list (avoids borrow conflict with trail/watches).
        let mut ws = std::mem::take(&mut watches.lists[false_lit.code() as usize]);
        let mut j = 0; // compaction write index

        let mut i = 0;
        while i < ws.len() {
            let ci = ws[i];

            // Lazy cleanup: skip deleted clauses (compacts them out of the list)
            if db.is_deleted(ci) {
                i += 1;
                continue;
            }

            let [w0, w1] = watches.watched[ci];

            // Which watch is false_lit? The other is the partner.
            let (partner, watch_pos) = if w0 == false_lit {
                (w1, 0usize)
            } else {
                debug_assert_eq!(w1, false_lit, "clause {ci} in watch list for {false_lit} but watches {w0},{w1}");
                (w0, 1usize)
            };

            // Fast path: if partner is true, clause is satisfied — keep watching.
            if eval_lit(partner, trail.assignments()) == Some(true) {
                ws[j] = ci;
                j += 1;
                i += 1;
                continue;
            }

            // Search clause for a replacement watch (unassigned or true literal).
            let clause_lits = db.clause(ci).literals;
            let mut replacement = None;
            for &lit in clause_lits {
                if lit == w0 || lit == w1 {
                    continue;
                }
                if eval_lit(lit, trail.assignments()) != Some(false) {
                    replacement = Some(lit);
                    break;
                }
            }

            if let Some(new_watch) = replacement {
                // Swap watch: move clause from false_lit's list to new_watch's list.
                watches.watched[ci][watch_pos] = new_watch;
                watches.lists[new_watch.code() as usize].push(ci);
                i += 1;
                continue; // don't write back to ws — clause removed from this list
            }

            // No replacement: all non-watched literals are false. Keep in list.
            ws[j] = ci;
            j += 1;
            i += 1;

            // Check partner to determine unit vs conflict.
            match eval_lit(partner, trail.assignments()) {
                Some(false) => {
                    // CONFLICT: all literals false.
                    // Flush remaining entries back to list before returning.
                    while i < ws.len() {
                        ws[j] = ws[i];
                        j += 1;
                        i += 1;
                    }
                    ws.truncate(j);
                    watches.lists[false_lit.code() as usize] = ws;
                    return BcpResult::Conflict { clause_index: ci };
                }
                None => {
                    // UNIT: propagate partner.
                    trail.record_propagation(partner, ci);
                }
                Some(true) => {
                    // Partner became true during this BCP round. Satisfied.
                }
            }
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
