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
//!
//! Watched positions are stored inline in the clause arena: c[0] and c[1]
//! are always the two watched literals. When a watch changes, the new literal
//! is swapped into position. This co-locates watch data with clause data
//! (one fewer cache line access vs a separate `watched` array) and lets the
//! replacement search start at c[2] (no w0/w1 comparison).

use crate::bcp::{BcpResult, ClauseDb};
use crate::literal::Lit;
use crate::phase::Propagate;
use crate::trail::Trail;

/// Binary flag encoded in bit 31 of the clause index.
const BINARY_FLAG: u32 = 0x8000_0000;

/// A single entry in a literal's watch list.
///
/// Stores the clause index (as u32 for cache density) and a "blocker" literal.
/// The blocker is a speculative hint — if it evaluates to true, the clause is
/// satisfied and we skip it without any clause DB lookup. This eliminates
/// 50-70% of clause lookups in typical BCP (MiniSat's key optimization).
///
/// For binary clauses (2 literals), bit 31 of `clause_and_flags` is set.
/// The blocker is always the exact partner literal (never stale), so BCP
/// can skip the clause DB access entirely — propagation or conflict is
/// determined from the blocker value alone.
///
/// Size: 8 bytes (same as a bare `usize` on 64-bit), but carries the blocker for free.
#[derive(Clone, Copy)]
struct WatchEntry {
    /// Clause index in the ClauseDb, with binary flag in bit 31.
    /// Bit 31 set → binary clause (blocker is exact partner, no clause DB access needed).
    clause_and_flags: u32,
    /// Blocker: the other watched literal at watch setup time. If this literal
    /// is true, the clause is satisfied — skip without touching the clause DB.
    /// For long clauses: may be stale (clause's watches may have moved),
    /// but a stale-but-true blocker is still a valid skip.
    /// For binary clauses: always exact (the partner never changes).
    blocker: Lit,
}

impl WatchEntry {
    #[inline]
    fn new(clause_idx: u32, blocker: Lit, binary: bool) -> Self {
        let flags = if binary { BINARY_FLAG } else { 0 };
        WatchEntry {
            clause_and_flags: clause_idx | flags,
            blocker,
        }
    }

    #[inline]
    fn clause_index(&self) -> usize {
        (self.clause_and_flags & !BINARY_FLAG) as usize
    }

    #[inline]
    fn is_binary(&self) -> bool {
        self.clause_and_flags & BINARY_FLAG != 0
    }
}

/// Two-watched-literal data structure with blocking literals.
///
/// Watch positions are stored inline in the clause arena (c[0] and c[1]),
/// not in a separate array. This struct only holds per-literal watch lists
/// and the BCP queue head.
pub struct Watches {
    /// Per-literal watch lists with blocker hints.
    lists: Vec<Vec<WatchEntry>>,
    /// Trail position processed up to (for incremental propagation).
    queue_head: usize,
    /// Whether the one-time unit/empty clause scan has been performed.
    initial_scan_done: bool,
}

impl Watches {
    /// Initialize watches for all clauses. Clauses with <2 literals get
    /// no watch entries (handled as unit/empty in the BCP loop directly).
    ///
    /// Reads c[0] and c[1] from each clause as the watched pair — the
    /// inline-watch invariant must already hold (true at init and maintained
    /// by swap_literal_unchecked during BCP).
    pub fn new(db: &ClauseDb, num_vars: u32) -> Self {
        let num_lits = 2 * num_vars as usize;
        let mut lists = vec![Vec::new(); num_lits];

        for ci in 0..db.len() {
            let lits = &db.clause(ci).literals;
            if lits.len() < 2 {
                continue;
            }
            let w0 = lits[0];
            let w1 = lits[1];
            let binary = lits.len() == 2;
            // Each watch entry stores the *other* watched literal as blocker
            lists[w0.code() as usize].push(WatchEntry::new(ci as u32, w1, binary));
            lists[w1.code() as usize].push(WatchEntry::new(ci as u32, w0, binary));
        }

        Watches {
            lists,
            queue_head: 0,
            initial_scan_done: false,
        }
    }

    /// Add watches for a newly learned clause.
    ///
    /// Reads c[0] and c[1] as the watched pair (caller must ensure the
    /// asserting literal is at c[0] and the second watch is at c[1]).
    pub fn add_clause(&mut self, db: &ClauseDb, clause_idx: usize) {
        let lits = &db.clause(clause_idx).literals;
        if lits.len() < 2 {
            return;
        }
        let w0 = lits[0];
        let w1 = lits[1];
        let ci = clause_idx as u32;
        let binary = lits.len() == 2;
        self.lists[w0.code() as usize].push(WatchEntry::new(ci, w1, binary));
        self.lists[w1.code() as usize].push(WatchEntry::new(ci, w0, binary));
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

/// Evaluate a literal using the literal-indexed assignment array.
/// Single array lookup — no polarity branch.
///
/// # Safety
/// `lit.code()` must be < `lit_values.len()`. This is guaranteed when:
/// - All literals come from the clause DB
/// - `db.max_variable() < num_vars` was asserted at solver startup
/// - `lit_values.len() == 2 * num_vars`
#[inline]
unsafe fn eval_lit_indexed(lit: Lit, lit_values: &[Option<bool>]) -> Option<bool> {
    *lit_values.get_unchecked(lit.code() as usize)
}

/// Watched-literal BCP with inline watch positions.
///
/// Processes trail entries from `queue_head` onward. Clause positions c[0]
/// and c[1] are always the watched pair — when a replacement is found, it's
/// swapped into the watched position via `swap_literal_unchecked`.
///
/// Takes `&mut ClauseDb` for in-place literal swapping.
pub fn run_bcp_watched(
    db: &mut ClauseDb,
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
                match bt.lit_values[lit.code() as usize] {
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
    // - All literals come from clauses in the DB (c[0], c[1], c[k])
    // - solve_cdcl_core_inner asserts db.max_variable() < num_vars at startup
    // - bt.assigns.len() == num_vars (from Trail::new)
    // - Therefore lit.var() < bt.assigns.len() for every literal encountered
    while watches.queue_head < bt.len() {
        let assigned_lit = bt.entry_at(watches.queue_head).lit;
        watches.queue_head += 1;
        let false_lit = assigned_lit.complement();

        // SAFETY for watches.lists unchecked accesses:
        // false_lit and new_watch are literals from clauses in the DB.
        // All literals satisfy lit.code() < 2*num_vars (validated at solver startup).
        // watches.lists.len() == 2*num_vars (from Watches::new).
        let ws_slot = unsafe { watches.lists.get_unchecked_mut(false_lit.code() as usize) };
        let mut ws = std::mem::take(ws_slot);
        let mut j = 0;

        // SAFETY for ws unchecked accesses throughout this loop:
        // Invariant: j <= i. j starts at 0, incremented only when i increments.
        // i < ws.len() is the loop condition. Therefore j <= i < ws.len(),
        // so both ws[i] reads and ws[j] writes are in bounds.
        let mut i = 0;
        while i < ws.len() {
            let entry = unsafe { *ws.get_unchecked(i) };
            let ci = entry.clause_index();

            // SAFETY: ci comes from WatchEntry, set only from valid clause
            // indices during Watches::new() or add_clause().
            if unsafe { db.is_deleted_unchecked(ci) } {
                i += 1;
                continue;
            }

            // ── Blocker check ──
            // SAFETY: blocker literal comes from a clause in the DB.
            // Store result — reused by the binary fast path below.
            let blocker_val = unsafe { eval_lit_indexed(entry.blocker, bt.lit_values) };
            if blocker_val == Some(true) {
                unsafe { *ws.get_unchecked_mut(j) = entry };
                j += 1;
                i += 1;
                continue;
            }

            // ── Binary clause fast path ──
            // For binary clauses, the blocker is the exact partner (never stale).
            // No clause DB access needed — decide propagation/conflict from
            // the blocker value alone.
            if entry.is_binary() {
                unsafe { *ws.get_unchecked_mut(j) = entry };
                j += 1;
                i += 1;
                // blocker_val is Some(false) or None (Some(true) handled above)
                if blocker_val == Some(false) {
                    // Both literals false → CONFLICT
                    while i < ws.len() {
                        unsafe { *ws.get_unchecked_mut(j) = *ws.get_unchecked(i) };
                        j += 1;
                        i += 1;
                    }
                    ws.truncate(j);
                    *unsafe { watches.lists.get_unchecked_mut(false_lit.code() as usize) } = ws;
                    return BcpResult::Conflict { clause_index: ci };
                }
                // blocker_val is None → propagate partner
                bt.record_propagation(entry.blocker, ci);
                continue;
            }

            // ── Long clause path (≥3 literals) ──

            // Read watched pair from inline positions c[0], c[1].
            // SAFETY: ci < db.len(), positions 0 and 1 exist (clauses with <2
            // literals never get watch entries).
            let (c0, c1) = {
                let c = unsafe { db.clause_unchecked(ci) };
                (c.literals[0], c.literals[1])
            }; // borrow of db released — enables mutable access below

            // Determine partner (the other watched literal) and which position
            // in the clause holds false_lit.
            let (partner, false_pos) = if c0 == false_lit {
                (c1, 0usize)
            } else {
                debug_assert_eq!(c1, false_lit,
                    "clause {ci} in watch list for {false_lit} but c[0]={c0}, c[1]={c1}");
                (c0, 1usize)
            };

            // ── Partner satisfied → clause satisfied, keep watch ──
            // SAFETY: partner is a watched literal from the DB
            if unsafe { eval_lit_indexed(partner, bt.lit_values) } == Some(true) {
                unsafe { *ws.get_unchecked_mut(j) = WatchEntry::new(
                    entry.clause_and_flags & !BINARY_FLAG, partner, false,
                ) };
                j += 1;
                i += 1;
                continue;
            }

            // ── Search for replacement watch starting at c[2] ──
            // No need to compare against c[0]/c[1] — they're at known positions.
            let replacement = {
                let c = unsafe { db.clause_unchecked(ci) };
                let mut found = None;
                for k in 2..c.literals.len() {
                    let lit = c.literals[k];
                    // SAFETY: lit comes from a clause in the DB
                    if unsafe { eval_lit_indexed(lit, bt.lit_values) } != Some(false) {
                        found = Some((lit, k));
                        break;
                    }
                }
                found
            }; // borrow of db released

            if let Some((new_watch, k)) = replacement {
                // Swap replacement into the watched position (c[false_pos])
                // SAFETY: ci < db.len(), false_pos ∈ {0,1}, k ∈ [2, clause_len)
                unsafe { db.swap_literal_unchecked(ci, false_pos, k) };
                // Add watch for the new literal (long clause, not binary)
                // SAFETY: new_watch.code() < 2*num_vars
                unsafe { watches.lists.get_unchecked_mut(new_watch.code() as usize) }.push(
                    WatchEntry::new(ci as u32, partner, false)
                );
                // Entry removed from false_lit's list (not copied to ws[j])
                i += 1;
                continue;
            }

            // No replacement found — clause is unit under current assignment.
            // Keep this entry in the watch list.
            unsafe { *ws.get_unchecked_mut(j) = entry };
            j += 1;
            i += 1;

            // SAFETY: partner is a watched literal from the DB
            let partner_val = unsafe { eval_lit_indexed(partner, bt.lit_values) };
            if partner_val == Some(false) {
                // Both watched literals false, no replacement → CONFLICT
                // Drain remaining entries before returning
                while i < ws.len() {
                    unsafe { *ws.get_unchecked_mut(j) = *ws.get_unchecked(i) };
                    j += 1;
                    i += 1;
                }
                ws.truncate(j);
                // SAFETY: false_lit.code() < 2*num_vars
                *unsafe { watches.lists.get_unchecked_mut(false_lit.code() as usize) } = ws;
                return BcpResult::Conflict { clause_index: ci };
            } else if partner_val.is_none() {
                // Partner unassigned → unit clause, propagate partner
                bt.record_propagation(partner, ci);
            }
            // else: partner is true — satisfied during this BCP round
        }

        ws.truncate(j);
        // SAFETY: false_lit.code() < 2*num_vars
        *unsafe { watches.lists.get_unchecked_mut(false_lit.code() as usize) } = ws;
    }

    BcpResult::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session;

    fn bcp_after_decision(
        db: &mut ClauseDb,
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
        assert_eq!(bcp_after_decision(&mut db, &mut w, &mut trail), BcpResult::Ok);
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
            bcp_after_decision(&mut db, &mut w, &mut trail),
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
        assert_eq!(bcp_after_decision(&mut db, &mut w, &mut trail), BcpResult::Ok);
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
        match bcp_after_decision(&mut db, &mut w, &mut trail) {
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
        assert_eq!(bcp_after_decision(&mut db, &mut w, &mut trail), BcpResult::Ok);
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
            let mut db2 = generate_3sat_phase_transition(30, seed);
            let mut w = Watches::new(&db2, 30);
            let mut trail2 = Trail::new(30);
            trail2.new_decision(Lit::pos(0));
            let r2 = session::with_session(|s| {
                let p = s.decide().propagate();
                run_bcp_watched(&mut db2, &mut w, &mut trail2, &p)
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
