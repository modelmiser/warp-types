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
}

/// Run 1-UIP conflict analysis.
///
/// `conflict_clause` is the index of the clause that caused the conflict.
/// Returns the learned clause and backtrack level.
pub fn analyze_conflict(trail: &Trail, db: &ClauseDb, conflict_clause: usize) -> AnalysisResult {
    let current_level = trail.current_level();

    // Seen set: tracks which variables have been visited during resolution
    let max_var = db.max_variable().max(
        trail
            .entries()
            .iter()
            .map(|e| e.lit.var())
            .max()
            .unwrap_or(0),
    );
    let mut seen = vec![false; max_var as usize + 1];

    // Start with the conflict clause's literals
    let mut learned = Vec::new();
    let mut num_at_current_level = 0;

    for &lit in db.clause(conflict_clause).literals {
        let var = lit.var();
        if !seen[var as usize] {
            seen[var as usize] = true;
            let entry = trail.entry_for_var(var);
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
        if entry.level != current_level || !seen[entry.lit.var() as usize] {
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
                seen[entry.lit.var() as usize] = false; // resolved away
                                                        // Add all other literals from the reason clause
                for &lit in db.clause(reason_clause).literals {
                    let var = lit.var();
                    if var == entry.lit.var() {
                        continue; // skip the resolved variable
                    }
                    if !seen[var as usize] {
                        seen[var as usize] = true;
                        let reason_entry = trail.entry_for_var(var);
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
        if entry.level == current_level && seen[entry.lit.var() as usize] {
            asserting_lit = Some(entry.lit.complement());
            break;
        }
    }

    let lit = asserting_lit
        .expect("1-UIP resolution must find an asserting literal at the current decision level");
    learned.insert(0, lit); // asserting literal first

    // ── Clause minimization ────────────────────────────────────────
    // Remove literals whose propagation reasons are already implied by
    // other literals in the clause. MiniSat's litRedundant algorithm.

    let abstract_levels = {
        let mut mask = 0u64;
        for &l in &learned {
            if let Some(e) = trail.entry_for_var(l.var()) {
                mask |= 1u64 << (e.level % 64);
            }
        }
        mask
    };

    let mut to_clear: Vec<u32> = Vec::new();
    let mut minimized = Vec::with_capacity(learned.len());
    minimized.push(learned[0]); // asserting literal always kept

    for &l in &learned[1..] {
        if lit_redundant(trail, db, l, abstract_levels, &mut seen, &mut to_clear) {
            seen[l.var() as usize] = false; // no longer in clause
        } else {
            minimized.push(l);
        }
    }

    // Clean up DFS marks from successful redundancy proofs
    for &var in &to_clear {
        seen[var as usize] = false;
    }

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
        let mut levels_seen = vec![false; current_level as usize + 1];
        let mut count = 0u32;
        for lit in &learned {
            if let Some(e) = trail.entry_for_var(lit.var()) {
                let lv = e.level as usize;
                if lv < levels_seen.len() && !levels_seen[lv] {
                    levels_seen[lv] = true;
                    count += 1;
                }
            }
        }
        count
    };

    AnalysisResult {
        learned,
        backtrack_level,
        lbd,
    }
}

/// Check if a literal is redundant in the learned clause.
///
/// A literal is redundant if all literals in its reason clause are either:
/// - Already in the learned clause (marked in `seen`)
/// - At decision level 0 (always true, can't contribute to conflict)
/// - Themselves redundant (recursive, via DFS)
///
/// Uses `abstract_levels` (bitmask of decision levels in the learned clause)
/// as a fast filter: if a variable's level has no bit set, it can't be in
/// the clause, so the literal isn't redundant.
///
/// Marks from successful proofs persist in `seen` (transitivity cache).
/// On failure, all marks from this check are rolled back via `to_clear`.
fn lit_redundant(
    trail: &Trail,
    db: &ClauseDb,
    lit: Lit,
    abstract_levels: u64,
    seen: &mut [bool],
    to_clear: &mut Vec<u32>,
) -> bool {
    let top = to_clear.len(); // snapshot for rollback on failure
    let mut stack: Vec<Lit> = Vec::new();

    // Start by examining the reason for lit's assignment
    let entry = match trail.entry_for_var(lit.var()) {
        Some(e) => e,
        None => return false,
    };
    let reason_clause = match entry.reason {
        Reason::Decision => return false, // decisions are never redundant
        Reason::Propagation(ci) => ci,
    };

    // Push reason clause literals (except lit itself) onto stack
    for &reason_lit in db.clause(reason_clause).literals {
        let rv = reason_lit.var();
        if rv == lit.var() {
            continue;
        }
        if seen[rv as usize] {
            continue; // in clause or already proven redundant
        }
        if let Some(re) = trail.entry_for_var(rv) {
            if re.level == 0 {
                continue; // level 0 literals are always satisfied
            }
        }
        stack.push(reason_lit);
    }

    while let Some(l) = stack.pop() {
        let var = l.var();

        let re = match trail.entry_for_var(var) {
            Some(e) => e,
            None => {
                // Unassigned — rollback
                for v in to_clear.drain(top..) {
                    seen[v as usize] = false;
                }
                return false;
            }
        };

        // Abstract level filter: if this level can't be in the clause, fail fast
        if (abstract_levels >> (re.level % 64)) & 1 == 0 {
            for v in to_clear.drain(top..) {
                seen[v as usize] = false;
            }
            return false;
        }

        let ci = match re.reason {
            Reason::Decision => {
                // Decision variable at a level that MIGHT be in the clause
                // (passed abstract filter) but isn't actually in the clause
                // (not in `seen`). Can't be proven redundant.
                for v in to_clear.drain(top..) {
                    seen[v as usize] = false;
                }
                return false;
            }
            Reason::Propagation(ci) => ci,
        };

        // Mark as visited (will be cleaned up on failure or at end)
        seen[var as usize] = true;
        to_clear.push(var);

        // Explore reason clause
        for &reason_lit in db.clause(ci).literals {
            let rv = reason_lit.var();
            if rv == var {
                continue;
            }
            if seen[rv as usize] {
                continue;
            }
            if let Some(rre) = trail.entry_for_var(rv) {
                if rre.level == 0 {
                    continue;
                }
            }
            stack.push(reason_lit);
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
