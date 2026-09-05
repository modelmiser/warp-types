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

use crate::bcp::{BcpResult, CRef, ClauseDb};
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
    fn clause_ref(&self) -> CRef {
        self.clause_and_flags & !BINARY_FLAG
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
pub(crate) struct Watches {
    /// Per-literal watch lists with blocker hints.
    lists: Vec<Vec<WatchEntry>>,
    /// Trail position processed up to (for incremental propagation).
    queue_head: usize,
    /// Whether the one-time unit/empty clause scan has been performed.
    initial_scan_done: bool,
    /// Upper bound on stored CRefs: the source DB's arena length at build time
    /// (kept current by `add_clause`). `run_bcp_watched` asserts this against
    /// the DB it is given, catching watches built from a different/older DB.
    cref_bound: usize,
}

impl Watches {
    /// Initialize watches for all clauses. Clauses with <2 literals get
    /// no watch entries (handled as unit/empty in the BCP loop directly).
    ///
    /// Reads c[0] and c[1] from each clause as the watched pair — the
    /// inline-watch invariant must already hold (true at init and maintained
    /// by swap_literal_unchecked during BCP).
    pub(crate) fn new(db: &ClauseDb, num_vars: u32) -> Self {
        let num_lits = 2 * num_vars as usize;
        let mut lists = vec![Vec::new(); num_lits];

        for cref in db.iter_crefs() {
            if db.is_deleted(cref) {
                continue;
            }
            let lits = &db.clause(cref).literals;
            if lits.len() < 2 {
                continue;
            }
            let w0 = lits[0];
            let w1 = lits[1];
            let binary = lits.len() == 2;
            // Each watch entry stores the *other* watched literal as blocker
            lists[w0.code() as usize].push(WatchEntry::new(cref, w1, binary));
            lists[w1.code() as usize].push(WatchEntry::new(cref, w0, binary));
        }

        Watches {
            lists,
            queue_head: 0,
            initial_scan_done: false,
            cref_bound: db.arena_len(),
        }
    }

    /// Add watches for a newly learned clause.
    ///
    /// Reads c[0] and c[1] as the watched pair (caller must ensure the
    /// asserting literal is at c[0] and the second watch is at c[1]).
    pub(crate) fn add_clause(&mut self, db: &ClauseDb, cref: CRef) {
        self.cref_bound = self.cref_bound.max(db.arena_len());
        let lits = &db.clause(cref).literals;
        if lits.len() < 2 {
            return;
        }
        let w0 = lits[0];
        let w1 = lits[1];
        let binary = lits.len() == 2;
        self.lists[w0.code() as usize].push(WatchEntry::new(cref, w1, binary));
        self.lists[w1.code() as usize].push(WatchEntry::new(cref, w0, binary));
    }

    /// Reset queue head after backtracking (trail is shorter now).
    pub(crate) fn notify_backtrack(&mut self, new_trail_len: usize) {
        self.queue_head = self.queue_head.min(new_trail_len);
    }

    /// Set queue head and mark initial scan as done (used after watch rebuild).
    pub(crate) fn set_queue_head(&mut self, pos: usize) {
        self.queue_head = pos;
        self.initial_scan_done = true;
    }
}

/// A watch list lifted out of `Watches::lists` for pointer-based compaction,
/// with its slot restored on every exit path — including an unwind.
///
/// BCP compacts a literal's watch list in place through raw `src`/`dst`
/// pointers while also pushing onto *other* literals' lists, so the list
/// being compacted has to leave the outer `Vec` for the duration. Restoring
/// it by hand at the end of the loop is correct only if the loop always
/// reaches the end: `BcpTrail::record_propagation` carries a release
/// `assert!` on double assignment, and if that ever fires the taken list is
/// dropped and the literal is left watching NOTHING. BCP would then silently
/// miss every propagation on that literal — a wrong SAT/UNSAT answer with no
/// diagnostic, which is the worst failure this solver has.
///
/// Today no in-crate caller catches that unwind and `Watches` is
/// `pub(crate)` and rebuilt inside every `solve*` call, so the window is not
/// reachable from outside. It becomes reachable the moment an incremental
/// API keeps a `Watches` across a fallible call. Structure it so that
/// question never has to be asked again.
///
/// Access to the other lists goes through `self.lists` — `ws` and `lists`
/// are disjoint fields, so the compaction pointers into `ws` stay valid
/// across pushes to the rest.
struct TakenList<'a> {
    lists: &'a mut Vec<Vec<WatchEntry>>,
    idx: usize,
    ws: Vec<WatchEntry>,
}

impl Drop for TakenList<'_> {
    fn drop(&mut self) {
        // `get_mut`, not indexing: a panic here during an unwind would abort,
        // and turning a recoverable assert into an abort is not an upgrade.
        if let Some(slot) = self.lists.get_mut(self.idx) {
            *slot = std::mem::take(&mut self.ws);
        }
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
///
/// # Cross-argument invariants (asserted once per call)
///
/// The unchecked indexing in the hot loop relies on these invariants, which
/// this function validates at entry (a mismatched call panics instead of
/// causing undefined behavior):
///
/// - `db.max_variable() < trail.num_vars()` (unless the DB is empty) — every
///   literal in the DB indexes within the trail's assignment arrays.
/// - `watches` covers at least `2 * trail.num_vars()` literals — every literal
///   on the trail indexes within the watch lists.
/// - `watches` was built from (a prefix of) THIS `db` — asserted via the
///   arena-length bound recorded at `Watches::new`/`add_clause` time. A
///   same-length but *different* DB cannot be detected by this check; the
///   caller must not mix watches across databases.
pub(crate) fn run_bcp_watched(
    db: &mut ClauseDb,
    watches: &mut Watches,
    trail: &mut Trail,
    _phase: &crate::session::SolverSession<'_, Propagate>,
) -> BcpResult {
    // ── Boundary checks: once per call, NOT in the propagation loop ──
    let num_vars = trail.num_vars();
    assert!(
        db.is_empty() || (db.max_variable() as usize) < num_vars,
        "run_bcp_watched: clause DB references variable {} but trail covers only {} variables",
        db.max_variable(),
        num_vars
    );
    assert!(
        watches.lists.len() >= 2 * num_vars,
        "run_bcp_watched: watch lists cover {} literal slots but trail requires {}",
        watches.lists.len(),
        2 * num_vars
    );
    assert!(
        watches.cref_bound <= db.arena_len(),
        "run_bcp_watched: watches reference clause offsets up to {} but DB arena has {} words \
         (watches built from a different DB?)",
        watches.cref_bound,
        db.arena_len()
    );

    // Split trail: bt.assigns is a &mut [Option<bool>] (stable pointer).
    // bt.record_propagation writes entries/var_position (disjoint fields),
    // so the compiler keeps the assigns pointer in a register across propagations.
    let mut bt = trail.bcp_split();

    // Handle unit/empty original clauses once at initialization.
    if !watches.initial_scan_done {
        watches.initial_scan_done = true;
        for cref in db.iter_crefs() {
            if db.is_deleted(cref) {
                continue;
            }
            let lits = &db.clause(cref).literals;
            if lits.is_empty() {
                return BcpResult::Conflict { clause: cref };
            }
            if lits.len() == 1 {
                let lit = lits[0];
                match bt.lit_values[lit.code() as usize] {
                    None => bt.record_propagation(lit, cref),
                    Some(false) => return BcpResult::Conflict { clause: cref },
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
        let idx = false_lit.code() as usize;
        let taken = std::mem::take(unsafe { watches.lists.get_unchecked_mut(idx) });
        // Restores `lists[idx]` on every exit from this iteration, unwind
        // included — see `TakenList`.
        let mut ws = TakenList {
            lists: &mut watches.lists,
            idx,
            ws: taken,
        };

        // Pointer-based iteration (MiniSat's pattern): src/dst/end instead
        // of i/j/ws.len(). Three pointer registers vs 4-5 index registers.
        // The compiler no longer maintains redundant loop counters (countdown,
        // i, i+1, ptr offset) — a single pointer advance per iteration.
        //
        // SAFETY invariant: dst <= src <= ws_end throughout.
        // src starts at ws.as_mut_ptr(), dst = src, ws_end = src + ws.len().
        // dst only advances when src advances (compaction: skip deleted).
        // All entries within [ws.as_ptr(), ws_end) are valid WatchEntry values.
        let ws_base = ws.ws.as_mut_ptr();
        let ws_end = unsafe { ws_base.add(ws.ws.len()) };
        let mut src = ws_base;
        let mut dst = ws_base;

        while src < ws_end {
            let entry = unsafe { *src };
            src = unsafe { src.add(1) };

            // ── Blocker check FIRST (no clause DB access) ──
            // This is the hot path: 50-70% of entries are skipped here.
            // Checking blocker before deleted avoids a random arena access
            // for the majority of watch entries (MiniSat's approach).
            // SAFETY: blocker literal comes from a clause in the DB.
            let blocker_val = unsafe { eval_lit_indexed(entry.blocker, bt.lit_values) };
            if blocker_val == Some(true) {
                unsafe { *dst = entry };
                dst = unsafe { dst.add(1) };
                continue;
            }

            let cref = entry.clause_ref();

            // ── Deleted check (deferred past blocker) ──
            // Stale entries for deleted clauses that pass the blocker check
            // are harmless — cleaned up during the next watch rebuild after
            // compaction (solver.rs Watches::new rebuild).
            // SAFETY: cref comes from WatchEntry, set only from valid clause
            // CRefs during Watches::new() or add_clause().
            if unsafe { db.is_deleted_unchecked(cref) } {
                continue;
            }

            // ── Binary clause fast path ──
            // For binary clauses, the blocker is the exact partner (never stale).
            // No clause DB access needed — decide propagation/conflict from
            // the blocker value alone.
            if entry.is_binary() {
                unsafe { *dst = entry };
                dst = unsafe { dst.add(1) };
                // blocker_val is Some(false) or None (Some(true) handled above)
                if blocker_val == Some(false) {
                    // Both literals false → CONFLICT
                    // Drain remaining entries (memmove: regions may overlap)
                    let remaining = unsafe { ws_end.offset_from(src) } as usize;
                    unsafe { std::ptr::copy(src, dst, remaining) };
                    dst = unsafe { dst.add(remaining) };
                    let new_len = unsafe { dst.offset_from(ws_base) } as usize;
                    unsafe { ws.ws.set_len(new_len) };
                    // `ws` restores lists[idx] as it drops on this return.
                    return BcpResult::Conflict { clause: cref };
                }
                // blocker_val is None → propagate partner
                bt.record_propagation(entry.blocker, cref);
                continue;
            }

            // ── Long clause path (≥3 literals) ──

            // Single clause access: read c[0], c[1] and search for replacement
            // in one borrow scope. NLL releases the borrow after the last use
            // of `lits` (the replacement search loop), before the mutable
            // swap_literal_unchecked call below.
            //
            // SAFETY: cref points to a valid header, clause has ≥2 literals
            // (watch invariant), all literal codes < 2*num_vars.
            let c = unsafe { db.clause_unchecked(cref) };
            let lits = c.literals;
            let c0 = unsafe { *lits.get_unchecked(0) };
            let c1 = unsafe { *lits.get_unchecked(1) };

            // Branchless partner/false_pos: exactly one of c[0],c[1] equals
            // false_lit. Select the *other* as partner. The compiler emitted a
            // branch for the original if/else (~8% of BCP branch misses) because
            // the two code paths loaded from different clause positions. Bitmask
            // selection eliminates the branch entirely.
            debug_assert!(
                c0 == false_lit || c1 == false_lit,
                "clause cref={cref} in watch list for {false_lit} but c[0]={c0}, c[1]={c1}"
            );
            let c0_is_false = (c0 == false_lit) as u32;
            let mask = c0_is_false.wrapping_neg(); // 0xFFFF_FFFF or 0
                                                   // SAFETY: Lit is #[repr(transparent)] over u32; both codes are valid.
            let partner: Lit =
                unsafe { std::mem::transmute((c1.code() & mask) | (c0.code() & !mask)) };
            let false_pos = (1 ^ c0_is_false) as usize;

            // ── Partner satisfied → clause satisfied, keep watch ──
            // SAFETY: partner is a watched literal from the DB
            if unsafe { eval_lit_indexed(partner, bt.lit_values) } == Some(true) {
                unsafe {
                    *dst = WatchEntry::new(entry.clause_and_flags & !BINARY_FLAG, partner, false)
                };
                dst = unsafe { dst.add(1) };
                continue;
            }

            // ── Search for replacement watch starting at c[2] ──
            // No need to compare against c[0]/c[1] — they're at known positions.
            let mut replacement = None;
            for k in 2..lits.len() {
                let lit = unsafe { *lits.get_unchecked(k) };
                // SAFETY: lit comes from a clause in the DB
                if unsafe { eval_lit_indexed(lit, bt.lit_values) } != Some(false) {
                    replacement = Some((lit, k));
                    break;
                }
            }
            // NLL: c/lits borrow ends here (last use was in the loop).
            // Mutable db access for swap_literal_unchecked is now safe.

            if let Some((new_watch, k)) = replacement {
                // Swap replacement into the watched position (c[false_pos])
                // SAFETY: cref valid, false_pos ∈ {0,1}, k ∈ [2, clause_len)
                unsafe { db.swap_literal_unchecked(cref, false_pos, k) };
                // Add watch for the new literal (long clause, not binary)
                // SAFETY: new_watch.code() < 2*num_vars
                unsafe { ws.lists.get_unchecked_mut(new_watch.code() as usize) }
                    .push(WatchEntry::new(cref, partner, false));
                // Entry removed from false_lit's list (not copied to dst)
                continue;
            }

            // No replacement found — clause is unit under current assignment.
            // Keep this entry in the watch list.
            unsafe { *dst = entry };
            dst = unsafe { dst.add(1) };

            // SAFETY: partner is a watched literal from the DB
            let partner_val = unsafe { eval_lit_indexed(partner, bt.lit_values) };
            if partner_val == Some(false) {
                // Both watched literals false, no replacement → CONFLICT
                // Drain remaining entries
                let remaining = unsafe { ws_end.offset_from(src) } as usize;
                unsafe { std::ptr::copy(src, dst, remaining) };
                dst = unsafe { dst.add(remaining) };
                let new_len = unsafe { dst.offset_from(ws_base) } as usize;
                unsafe { ws.ws.set_len(new_len) };
                // `ws` restores lists[idx] as it drops on this return.
                return BcpResult::Conflict { clause: cref };
            } else if partner_val.is_none() {
                // Partner unassigned → unit clause, propagate partner
                bt.record_propagation(partner, cref);
            }
            // else: partner is true — satisfied during this BCP round
        }

        let new_len = unsafe { dst.offset_from(ws_base) } as usize;
        unsafe { ws.ws.set_len(new_len) };
        // `ws` restores lists[idx] as it drops at the end of this iteration.
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
        assert_eq!(
            bcp_after_decision(&mut db, &mut w, &mut trail),
            BcpResult::Ok
        );
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
            BcpResult::Conflict { clause: 0 }
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
        assert_eq!(
            bcp_after_decision(&mut db, &mut w, &mut trail),
            BcpResult::Ok
        );
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
        assert_eq!(
            bcp_after_decision(&mut db, &mut w, &mut trail),
            BcpResult::Ok
        );
        // Neither x1 nor x2 should be propagated (clause has 2 unresolved lits).
        assert_eq!(trail.value(1), None);
        assert_eq!(trail.value(2), None);
    }

    #[test]
    #[should_panic(expected = "references variable")]
    fn mismatched_trail_panics() {
        // DB references variable 6, but the trail only covers 2 variables.
        // Before the boundary asserts this was UB in release (unchecked
        // indexing into lit_values); now it panics at the call boundary.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(5), Lit::pos(6)]);
        let mut w = Watches::new(&db, 7);
        let mut trail = Trail::new(2);
        trail.new_decision(Lit::pos(1));
        bcp_after_decision(&mut db, &mut w, &mut trail);
    }

    #[test]
    #[should_panic(expected = "watch lists")]
    fn undersized_watches_panics() {
        // Watches built for 2 variables, trail covers 5 — a decision on a
        // high variable would index the watch lists out of bounds.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        let mut w = Watches::new(&db, 2);
        let mut trail = Trail::new(5);
        trail.new_decision(Lit::pos(0));
        bcp_after_decision(&mut db, &mut w, &mut trail);
    }

    #[test]
    #[should_panic(expected = "different DB")]
    fn watches_from_bigger_db_panics() {
        // Watches built from a larger DB hold CRefs past the end of the
        // smaller DB's arena — previously UB via is_deleted_unchecked.
        let mut big = ClauseDb::new();
        big.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        big.add_clause(vec![Lit::neg(1), Lit::pos(2)]);
        let mut w = Watches::new(&big, 3);

        let mut small = ClauseDb::new();
        small.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        let mut trail = Trail::new(3);
        trail.new_decision(Lit::pos(0));
        bcp_after_decision(&mut small, &mut w, &mut trail);
    }

    #[test]
    fn conflict_return_restores_the_watch_list() {
        // The binary-clause conflict path returns from the middle of the
        // compaction loop. The list it was compacting is put back by
        // `TakenList`'s Drop, not by a statement on that path — if the guard
        // were wired wrong, x0's watch list would come back empty and every
        // later BCP would miss both clauses.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(0), Lit::neg(1)]);

        let mut w = Watches::new(&db, 2);
        let mut trail = Trail::new(2);
        let false_lit = Lit::pos(0).complement();
        let before = w.lists[false_lit.code() as usize].len();
        assert_eq!(before, 2, "both clauses should watch this literal");

        trail.new_decision(Lit::pos(0));
        match bcp_after_decision(&mut db, &mut w, &mut trail) {
            BcpResult::Conflict { .. } => {}
            other => panic!("expected Conflict, got {other:?}"),
        }
        assert_eq!(
            w.lists[false_lit.code() as usize].len(),
            before,
            "conflict return dropped the watch list it had taken"
        );
    }

    // Only in debug: the panic this provokes is a `debug_assert!` on the
    // inline-watch invariant. It is the one panic reachable inside the
    // compaction window from a constructible state — BCP's own value checks
    // filter out the double-propagation that would trip
    // `record_propagation`'s release assert, which is why that one is a
    // latent hazard rather than a live bug. The guard's correctness does not
    // depend on which panic unwinds through it.
    #[cfg(debug_assertions)]
    #[test]
    fn unwind_mid_compaction_leaves_the_watch_list_intact() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1), Lit::pos(2)]); // cref 0
        let cref_b = db.add_clause(vec![Lit::pos(3), Lit::pos(4), Lit::pos(5)]);

        let mut w = Watches::new(&db, 6);
        let false_lit = Lit::pos(0).complement();
        let slot = false_lit.code() as usize;

        // Clause B is watched by x3/x4, not by ¬x0. Filing it under ¬x0
        // breaks the inline-watch invariant that the branchless
        // partner/false_pos selection depends on.
        w.lists[slot].push(WatchEntry::new(cref_b, Lit::pos(3), false));
        let before = w.lists[slot].len();
        assert_eq!(before, 2);

        let mut trail = Trail::new(6);
        trail.new_decision(Lit::pos(0));

        let r = catch_unwind(AssertUnwindSafe(|| {
            bcp_after_decision(&mut db, &mut w, &mut trail)
        }));
        assert!(r.is_err(), "expected the broken watch invariant to panic");

        // Before the guard, the taken list was dropped during the unwind and
        // ¬x0 was left watching nothing — BCP would then silently miss every
        // propagation on x0, which is a wrong answer with no diagnostic.
        assert!(
            !w.lists[slot].is_empty(),
            "unwind dropped the watch list for {false_lit}"
        );
    }

    #[test]
    fn watched_agrees_with_original_bcp() {
        use crate::bcp;
        use crate::bench::generate_3sat_phase_transition;

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

    /// Surface guard for the `add_clause` UB leg.
    ///
    /// `Watches`, `Watches::add_clause`, and `run_bcp_watched` are `pub(crate)`.
    /// The historical UB path was: external safe code hand-crafts a `Watches`,
    /// calls `add_clause` with a mid-clause / bogus `CRef`, and the next
    /// `run_bcp_watched` reaches `lits.get_unchecked(k)` — undefined behavior
    /// from safe code. With these items `pub(crate)`, no code OUTSIDE this
    /// crate can name `Watches` or reach `add_clause`/`run_bcp_watched` at all,
    /// so the leg is closed structurally (a compile-level fact, not a runtime
    /// check). This test is the in-crate witness that `add_clause` is reachable
    /// ONLY from within the crate and, fed a genuine `CRef` from
    /// `ClauseDb::add_clause` (the only way the solver ever calls it), behaves
    /// correctly.
    ///
    /// If `Watches` or `add_clause` are ever made `pub` again, this comment is
    /// the record of why they must not be: the unchecked BCP replacement search
    /// trusts every stored `CRef` to be a genuine clause header.
    #[test]
    fn add_clause_leg_is_crate_internal_and_sound_with_genuine_cref() {
        // (¬x0 ∨ x1) present at init; then learn (¬x1 ∨ x2) the way the solver
        // does — append via ClauseDb::add_clause (genuine CRef), register with
        // Watches::add_clause, then propagate.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        let mut w = Watches::new(&db, 3);

        let learned = db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);
        w.add_clause(&db, learned); // genuine, in-bounds header CRef

        let mut trail = Trail::new(3);
        trail.new_decision(Lit::pos(0));
        assert_eq!(
            bcp_after_decision(&mut db, &mut w, &mut trail),
            BcpResult::Ok
        );
        // x0 ⇒ x1 (original) ⇒ x2 (learned): both must propagate true.
        assert_eq!(trail.value(1), Some(true));
        assert_eq!(trail.value(2), Some(true));
    }
}
