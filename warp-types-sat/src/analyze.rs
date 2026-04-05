//! 1-UIP conflict analysis with clause minimization.
//!
//! Given a conflict clause and the current trail, resolves backward through
//! implication reasons until exactly one literal at the current decision level
//! remains. That literal is the First Unique Implication Point (1-UIP).
//!
//! After deriving the learned clause, clause minimization (MiniSat's
//! `litRedundant`) removes literals whose assignments are already implied
//! by other literals in the clause. This typically removes 20-30% of
//! learned clause literals, improving both LBD scores and BCP speed.
//!
//! Returns the learned clause and the backtrack level.

use std::time::Instant;

use crate::bcp::ClauseDb;
use crate::literal::Lit;
use crate::trail::{Reason, Trail};

/// Result of conflict analysis.
pub struct AnalysisResult {
    /// The learned clause (asserting clause). The first literal is the
    /// asserting literal (the 1-UIP, negated).
    pub learned: Vec<Lit>,
    /// The decision level to backtrack to.
    pub backtrack_level: u32,
    /// Literal Block Distance: number of distinct decision levels among the
    /// learned clause's literals. Lower = more useful (a "glue clause" has LBD ≤ 2).
    pub lbd: u32,
    /// Nanoseconds spent in 1-UIP resolution (backward trail scan + reason clause iteration).
    pub resolve_ns: u64,
    /// Nanoseconds spent in clause minimization (litRedundant DFS).
    pub minimize_ns: u64,
}

/// Persistent scratch buffers for conflict analysis.
///
/// Allocated once at solver init, reused across all conflicts. Eliminates
/// the per-conflict heap allocation of `seen` and `levels_seen` vectors.
pub struct AnalyzeWork {
    /// Per-variable seen flag. Sized for num_vars at init, cleared
    /// incrementally via `touched` after each analysis.
    seen: Vec<bool>,
    /// Variables touched during this analysis (for incremental clear).
    touched: Vec<u32>,
    /// Stack for clause minimization DFS.
    min_stack: Vec<Lit>,
    /// Temporary for `to_clear` in minimization.
    min_to_clear: Vec<u32>,
    /// Level-seen flags for LBD computation.
    levels_seen: Vec<bool>,
}

impl AnalyzeWork {
    /// Create scratch buffers for a solver with `num_vars` variables.
    pub fn new(num_vars: usize) -> Self {
        AnalyzeWork {
            seen: vec![false; num_vars],
            touched: Vec::with_capacity(64),
            min_stack: Vec::with_capacity(32),
            min_to_clear: Vec::with_capacity(32),
            levels_seen: Vec::new(), // grown on demand per analysis
        }
    }

    /// Ensure buffers cover at least `num_vars` variables (for learned clauses
    /// that introduce new variable indices — shouldn't happen, but defensive).
    fn ensure_capacity(&mut self, num_vars: usize) {
        if num_vars > self.seen.len() {
            self.seen.resize(num_vars, false);
        }
    }

    /// Clear all seen flags touched during the last analysis.
    fn clear_seen(&mut self) {
        for &var in &self.touched {
            self.seen[var as usize] = false;
        }
        self.touched.clear();
    }

    /// Mark a variable as seen and record it for cleanup.
    #[inline]
    fn mark_seen(&mut self, var: u32) {
        self.seen[var as usize] = true;
        self.touched.push(var);
    }
}

/// Run 1-UIP conflict analysis (allocates fresh scratch buffers).
///
/// Convenience wrapper for callers that don't reuse buffers (old solver, tests).
pub fn analyze_conflict(trail: &Trail, db: &ClauseDb, conflict_clause: usize) -> AnalysisResult {
    let num_vars = trail.num_vars().max(db.max_variable() as usize + 1);
    let mut work = AnalyzeWork::new(num_vars);
    analyze_conflict_with(&mut work, trail, db, conflict_clause)
}

/// Run 1-UIP conflict analysis using persistent scratch buffers.
///
/// `conflict_clause` is the index of the clause that caused the conflict.
/// Returns the learned clause and backtrack level.
pub fn analyze_conflict_with(
    work: &mut AnalyzeWork,
    trail: &Trail,
    db: &ClauseDb,
    conflict_clause: usize,
) -> AnalysisResult {
    let current_level = trail.current_level();
    let t_resolve = Instant::now();

    // Ensure seen array covers all variables (defensive — shouldn't grow).
    let max_var = trail.num_vars();
    work.ensure_capacity(max_var);

    // Start with the conflict clause's literals.
    // SAFETY: conflict_clause < db.len() (caller invariant).
    let mut learned = Vec::new();
    let mut num_at_current_level = 0;

    for &lit in unsafe { db.clause_unchecked(conflict_clause) }.literals {
        let var = lit.var();
        if !work.seen[var as usize] {
            work.mark_seen(var);
            // SAFETY: var comes from clause DB, validated var < num_vars at startup.
            let entry = unsafe { trail.entry_for_var_unchecked(var) };
            debug_assert!(
                entry.is_some(),
                "variable {} in conflict clause has no trail entry (unassigned in a conflict?)",
                var
            );
            match entry {
                Some(e) if e.level == current_level => {
                    num_at_current_level += 1;
                }
                Some(_) | None => {
                    learned.push(lit);
                }
            }
        }
    }

    // Resolve backward through trail until 1 literal at current level remains
    let entries = trail.entries();
    let mut trail_idx = entries.len();

    while num_at_current_level > 1 {
        // Find the most recent trail entry at the current level that we've seen
        trail_idx -= 1;
        let entry = &entries[trail_idx];
        if entry.level != current_level || !work.seen[entry.lit.var() as usize] {
            continue;
        }

        // This literal is at the current level and in our working clause.
        // Resolve it away using its reason clause.
        match entry.reason {
            Reason::Decision => {
                // Decisions can't be resolved — this shouldn't happen if
                // there's more than 1 literal at the current level
                debug_assert!(
                    num_at_current_level <= 1,
                    "hit decision during resolution with {} literals remaining at current level",
                    num_at_current_level
                );
                break;
            }
            Reason::Propagation(reason_clause) => {
                debug_assert!(
                    !db.is_deleted(reason_clause),
                    "resolving through deleted clause {reason_clause} for var {}",
                    entry.lit.var()
                );
                num_at_current_level -= 1;
                work.seen[entry.lit.var() as usize] = false; // resolved away
                // Add all other literals from the reason clause.
                // SAFETY: reason_clause < db.len() (valid propagation reasons).
                for &lit in unsafe { db.clause_unchecked(reason_clause) }.literals {
                    let var = lit.var();
                    if var == entry.lit.var() {
                        continue; // skip the resolved variable
                    }
                    if !work.seen[var as usize] {
                        work.mark_seen(var);
                        // SAFETY: var from clause DB, validated var < num_vars.
                        let reason_entry = unsafe { trail.entry_for_var_unchecked(var) };
                        match reason_entry {
                            Some(e) if e.level == current_level => {
                                num_at_current_level += 1;
                            }
                            Some(_) | None => {
                                learned.push(lit);
                            }
                        }
                    }
                }
            }
        }
    }

    // Find the 1-UIP: the single remaining seen variable at the current level.
    // Scan trail backward — the most recent seen entry at this level is the UIP.
    let mut asserting_lit = None;
    for entry in entries.iter().rev() {
        if entry.level == current_level && work.seen[entry.lit.var() as usize] {
            asserting_lit = Some(entry.lit.complement());
            break;
        }
    }

    let lit = asserting_lit
        .expect("1-UIP resolution must find an asserting literal at the current decision level");
    learned.insert(0, lit); // asserting literal first

    // Select optimal second watch: the literal with the highest decision level
    // among non-asserting literals. This ensures that after backtracking to
    // backtrack_level, c[1] is still assigned (false) — making the clause
    // immediately unit from BCP's perspective. MiniSat's standard technique.
    if learned.len() >= 3 {
        let mut best_pos = 1;
        let mut best_level = trail.entry_for_var(learned[1].var())
            .map(|e| e.level).unwrap_or(0);
        for i in 2..learned.len() {
            let level = trail.entry_for_var(learned[i].var())
                .map(|e| e.level).unwrap_or(0);
            if level > best_level {
                best_level = level;
                best_pos = i;
            }
        }
        if best_pos != 1 {
            learned.swap(1, best_pos);
        }
    }

    let resolve_ns = t_resolve.elapsed().as_nanos() as u64;

    // ── Clause minimization ───��──────────────────────────────��─────
    // Remove literals whose propagation reasons are already implied by
    // other literals in the clause. MiniSat's litRedundant algorithm.
    let t_minimize = Instant::now();

    let abstract_levels = {
        let mut mask = 0u64;
        for &l in &learned {
            if let Some(e) = trail.entry_for_var(l.var()) {
                mask |= 1u64 << (e.level % 64);
            }
        }
        mask
    };

    work.min_to_clear.clear();
    let mut minimized = Vec::with_capacity(learned.len());
    minimized.push(learned[0]); // asserting literal always kept

    for &l in &learned[1..] {
        if lit_redundant_with(work, trail, db, l, abstract_levels) {
            work.seen[l.var() as usize] = false; // no longer in clause
        } else {
            minimized.push(l);
        }
    }

    // Clean up DFS marks from successful redundancy proofs
    for &var in &work.min_to_clear {
        work.seen[var as usize] = false;
    }
    work.min_to_clear.clear();

    let minimize_ns = t_minimize.elapsed().as_nanos() as u64;
    let learned = minimized;

    // Backtrack level: highest level among learned clause literals,
    // excluding the asserting literal (which is at current_level).
    let backtrack_level = learned
        .iter()
        .skip(1) // skip asserting literal
        .filter_map(|lit| trail.entry_for_var(lit.var()).map(|e| e.level))
        .max()
        .unwrap_or(0);

    // LBD: count distinct decision levels in the learned clause.
    let lbd = {
        let level_count = current_level as usize + 1;
        work.levels_seen.clear();
        work.levels_seen.resize(level_count, false);
        let mut count = 0u32;
        for lit in &learned {
            if let Some(e) = trail.entry_for_var(lit.var()) {
                let lv = e.level as usize;
                if lv < level_count && !work.levels_seen[lv] {
                    work.levels_seen[lv] = true;
                    count += 1;
                }
            }
        }
        count
    };

    // Clean up seen flags for next conflict
    work.clear_seen();

    AnalysisResult {
        learned,
        backtrack_level,
        lbd,
        resolve_ns,
        minimize_ns,
    }
}

/// Check if a literal is redundant using persistent work buffers.
///
/// Same algorithm as the standalone `lit_redundant`, but reuses
/// `work.min_stack` and `work.min_to_clear` across calls.
fn lit_redundant_with(
    work: &mut AnalyzeWork,
    trail: &Trail,
    db: &ClauseDb,
    lit: Lit,
    abstract_levels: u64,
) -> bool {
    let top = work.min_to_clear.len(); // snapshot for rollback on failure
    work.min_stack.clear();

    // Start by examining the reason for lit's assignment.
    // SAFETY: lit.var() comes from learned clause, all vars < num_vars.
    let entry = match unsafe { trail.entry_for_var_unchecked(lit.var()) } {
        Some(e) => e,
        None => return false,
    };
    let reason_clause = match entry.reason {
        Reason::Decision => return false, // decisions are never redundant
        Reason::Propagation(ci) => ci,
    };

    // Push reason clause literals (except lit itself) onto stack.
    // SAFETY: reason_clause is a valid propagation reason from the trail.
    for &reason_lit in unsafe { db.clause_unchecked(reason_clause) }.literals {
        let rv = reason_lit.var();
        if rv == lit.var() {
            continue;
        }
        if work.seen[rv as usize] {
            continue; // in clause or already proven redundant
        }
        // SAFETY: rv from clause DB, validated var < num_vars.
        if let Some(re) = unsafe { trail.entry_for_var_unchecked(rv) } {
            if re.level == 0 {
                continue; // level 0 literals are always satisfied
            }
        }
        work.min_stack.push(reason_lit);
    }

    while let Some(l) = work.min_stack.pop() {
        let var = l.var();

        // SAFETY: var from clause DB, validated var < num_vars.
        let re = match unsafe { trail.entry_for_var_unchecked(var) } {
            Some(e) => e,
            None => {
                // Unassigned — rollback
                for v in work.min_to_clear.drain(top..) {
                    work.seen[v as usize] = false;
                }
                return false;
            }
        };

        // Abstract level filter: if this level can't be in the clause, fail fast
        if (abstract_levels >> (re.level % 64)) & 1 == 0 {
            for v in work.min_to_clear.drain(top..) {
                work.seen[v as usize] = false;
            }
            return false;
        }

        let ci = match re.reason {
            Reason::Decision => {
                // Decision variable at a level that MIGHT be in the clause
                // (passed abstract filter) but isn't actually in the clause
                // (not in `seen`). Can't be proven redundant.
                for v in work.min_to_clear.drain(top..) {
                    work.seen[v as usize] = false;
                }
                return false;
            }
            Reason::Propagation(ci) => ci,
        };

        // Mark as visited (will be cleaned up on failure or at end)
        work.seen[var as usize] = true;
        work.min_to_clear.push(var);

        // Explore reason clause.
        // SAFETY: ci is a valid propagation reason from the trail.
        for &reason_lit in unsafe { db.clause_unchecked(ci) }.literals {
            let rv = reason_lit.var();
            if rv == var {
                continue;
            }
            if work.seen[rv as usize] {
                continue;
            }
            // SAFETY: rv from clause DB, validated var < num_vars.
            if let Some(rre) = unsafe { trail.entry_for_var_unchecked(rv) } {
                if rre.level == 0 {
                    continue;
                }
            }
            work.min_stack.push(reason_lit);
        }
    }

    true // all paths lead to in-clause or level-0
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Lit;
    use crate::trail::Trail;

    #[test]
    fn simple_conflict_analysis() {
        // Setup: x0=T (decision), x1=T (propagated by clause 0: ¬x0∨x1),
        //        conflict on clause 1: ¬x0∨¬x1
        //
        // Conflict clause: {¬x0, ¬x1}, both at level 1.
        // Resolve ¬x1 with reason clause 0: (¬x0 ∨ x1) → resolvent: {¬x0}
        // Only one lit at level 1 (¬x0 is the decision) → UIP found.
        // Learned clause: {¬x0}
        // Backtrack level: 0

        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // clause 0: ¬x0 ∨ x1
        db.add_clause(vec![Lit::neg(0), Lit::neg(1)]); // clause 1: ¬x0 ∨ ¬x1

        let mut trail = Trail::new(2);

        trail.new_decision(Lit::pos(0)); // level 1: x0=T
        trail.record_propagation(Lit::pos(1), 0); // x1=T from clause 0

        let result = analyze_conflict(&trail, &db, 1);

        // Learned clause should contain ¬x0 (the asserting literal)
        assert_eq!(result.learned.len(), 1);
        assert_eq!(result.learned[0], Lit::neg(0));
        assert_eq!(result.backtrack_level, 0);
    }

    #[test]
    fn two_level_conflict() {
        // Level 1: decide x0=T
        // Level 2: decide x1=T, propagate x2=T (from clause 0: ¬x1∨x2)
        //          conflict on clause 1: ¬x0∨¬x2
        //
        // Conflict clause: {¬x0, ¬x2}
        //   ¬x0 at level 1, ¬x2 at level 2
        //   1 literal at current level (x2 was assigned at level 2)
        //   Standard 1-UIP: count==1 immediately → stop.
        //   Asserting literal: ¬x2 (complement of trail entry for x2)
        // Learned: {¬x2, ¬x0}
        // Backtrack level: 1 (from ¬x0 at level 1)

        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]); // clause 0: ¬x1 ∨ x2
        db.add_clause(vec![Lit::neg(0), Lit::neg(2)]); // clause 1: ¬x0 ∨ ¬x2

        let mut trail = Trail::new(3);

        trail.new_decision(Lit::pos(0)); // level 1: x0=T
        trail.new_decision(Lit::pos(1)); // level 2: x1=T
        trail.record_propagation(Lit::pos(2), 0); // x2=T from clause 0

        let result = analyze_conflict(&trail, &db, 1);

        assert_eq!(result.learned[0], Lit::neg(2)); // asserting: ¬x2
        assert!(result.learned.contains(&Lit::neg(0))); // from level 1
        assert_eq!(result.backtrack_level, 1);
    }

    #[test]
    fn minimization_removes_redundant_literal() {
        // Build a case where clause minimization can remove a literal.
        //
        // Clauses:
        //   c0: ¬x0 ∨ x1           (x0=T → x1=T)
        //   c1: ¬x1 ∨ x2           (x1=T → x2=T)
        //   c2: ¬x0 ∨ ¬x2 ∨ x3    (x0=T ∧ x2=T → x3=T)
        //   c3: ¬x3 ∨ ¬x4          (conflict when x3=T ∧ x4=T)
        //
        // Trail:
        //   Level 1: decide x0=T
        //   Level 1: propagate x1=T (from c0)
        //   Level 1: propagate x2=T (from c1)
        //   Level 1: propagate x3=T (from c2)
        //   Level 2: decide x4=T
        //   Conflict on c3: ¬x3 ∨ ¬x4
        //
        // 1-UIP analysis:
        //   Conflict clause: {¬x3, ¬x4}
        //   ¬x3 at level 1, ¬x4 at level 2 (current)
        //   Only 1 lit at current level → UIP is x4
        //   Raw learned: {¬x4, ¬x3}
        //
        // But x3 was propagated by c2: ¬x0 ∨ ¬x2 ∨ x3
        //   x0 is the decision at level 1 (not redundant — it's a decision)
        //   x2 was propagated by c1: ¬x1 ∨ x2
        //     x1 was propagated by c0: ¬x0 ∨ x1
        //       x0 is a decision (stops here)
        //   So ¬x3's reason traces to x0, which IS in the clause... wait.
        //
        // Actually ¬x3 is NOT redundant here because its reason (c2)
        // involves x0 and x2. x0 is NOT in the learned clause (only ¬x3 and ¬x4).
        // So ¬x3 can't be removed.
        //
        // Let me construct a better example where minimization actually fires.

        // Better example:
        //   Level 1: decide x0=T
        //   Level 1: propagate x1=T (from c0: ¬x0 ∨ x1)
        //   Level 1: propagate x2=T (from c1: ¬x1 ∨ x2)
        //   Level 2: decide x3=T
        //   Level 2: propagate x4=T (from c2: ¬x3 ∨ x4)
        //   Conflict on c3: ¬x0 ∨ ¬x2 ∨ ¬x4
        //
        // 1-UIP:
        //   Conflict clause literals: ¬x0 (level 1), ¬x2 (level 1), ¬x4 (level 2)
        //   x4 is at current level, count=1 → UIP is x4
        //   Raw learned: {¬x4, ¬x0, ¬x2}
        //
        // Minimization: can ¬x2 be removed?
        //   x2 was propagated by c1 (¬x1 ∨ x2). Reason lits: {¬x1}
        //   Is ¬x1 in learned? No. Is x1 redundant?
        //     x1 was propagated by c0 (¬x0 ∨ x1). Reason lits: {¬x0}
        //     Is ¬x0 in learned? YES (seen[0] = true).
        //   So x1 is redundant, therefore x2 is redundant.
        //   ¬x2 can be removed!
        //
        // Minimized: {¬x4, ¬x0}

        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // c0: ¬x0 ∨ x1
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]); // c1: ¬x1 ∨ x2
        db.add_clause(vec![Lit::neg(3), Lit::pos(4)]); // c2: ¬x3 ∨ x4
        db.add_clause(vec![Lit::neg(0), Lit::neg(2), Lit::neg(4)]); // c3: ¬x0 ∨ ¬x2 ∨ ¬x4

        let mut trail = Trail::new(5);
        trail.new_decision(Lit::pos(0)); // level 1: x0=T
        trail.record_propagation(Lit::pos(1), 0); // x1=T from c0
        trail.record_propagation(Lit::pos(2), 1); // x2=T from c1
        trail.new_decision(Lit::pos(3)); // level 2: x3=T
        trail.record_propagation(Lit::pos(4), 2); // x4=T from c2

        let result = analyze_conflict(&trail, &db, 3);

        // Asserting literal: ¬x4
        assert_eq!(result.learned[0], Lit::neg(4));
        // ¬x2 should be removed (redundant via x1→x0 chain)
        assert!(!result.learned.contains(&Lit::neg(2)), "¬x2 should be minimized away");
        // ¬x0 should remain (it's a decision, not redundant)
        assert!(result.learned.contains(&Lit::neg(0)));
        // Final clause: {¬x4, ¬x0}
        assert_eq!(result.learned.len(), 2);
        assert_eq!(result.backtrack_level, 1);
    }

    #[test]
    fn minimization_keeps_non_redundant() {
        // Example where no literal is redundant:
        //   Level 1: decide x0=T
        //   Level 2: decide x1=T
        //   Conflict on c0: ¬x0 ∨ ¬x1
        //
        // Learned: {¬x1, ¬x0}
        // Both are decisions — neither can be removed.

        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::neg(1)]); // c0: ¬x0 ∨ ¬x1

        let mut trail = Trail::new(2);
        trail.new_decision(Lit::pos(0)); // level 1
        trail.new_decision(Lit::pos(1)); // level 2

        let result = analyze_conflict(&trail, &db, 0);

        assert_eq!(result.learned[0], Lit::neg(1)); // asserting
        assert!(result.learned.contains(&Lit::neg(0)));
        assert_eq!(result.learned.len(), 2); // no minimization possible
    }

    #[test]
    fn minimization_with_level_zero() {
        // Level 0 literals in reason clauses are always satisfied and
        // should be treated as "free" during redundancy checks.
        //
        //   Level 0: propagate x0=T (from unit clause c0: x0)
        //   Level 1: decide x1=T
        //   Level 1: propagate x2=T (from c1: ¬x0 ∨ ¬x1 ∨ x2)
        //   Level 2: decide x3=T
        //   Conflict on c2: ¬x2 ∨ ¬x3
        //
        // Raw learned: {¬x3, ¬x2}
        // Can ¬x2 be removed?
        //   x2 propagated by c1 (¬x0 ∨ ¬x1 ∨ x2). Reason lits: {¬x0, ¬x1}
        //   x0 is at level 0 → skip (always true)
        //   x1 is a decision at level 1, NOT in learned → not redundant
        // So ¬x2 can't be removed. Clause stays {¬x3, ¬x2}.

        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]); // c0: x0 (unit)
        db.add_clause(vec![Lit::neg(0), Lit::neg(1), Lit::pos(2)]); // c1: ¬x0 ∨ ¬x1 ∨ x2
        db.add_clause(vec![Lit::neg(2), Lit::neg(3)]); // c2: ¬x2 ∨ ¬x3

        let mut trail = Trail::new(4);
        trail.record_propagation(Lit::pos(0), 0); // level 0: x0=T from c0
        trail.new_decision(Lit::pos(1)); // level 1: x1=T
        trail.record_propagation(Lit::pos(2), 1); // x2=T from c1
        trail.new_decision(Lit::pos(3)); // level 2: x3=T

        let result = analyze_conflict(&trail, &db, 2);

        assert_eq!(result.learned[0], Lit::neg(3));
        assert!(result.learned.contains(&Lit::neg(2)));
        assert_eq!(result.learned.len(), 2); // ¬x2 not redundant (x1 blocks it)
    }
}
