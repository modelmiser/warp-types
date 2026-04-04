//! 1-UIP conflict analysis.
//!
//! Given a conflict clause and the current trail, resolves backward through
//! implication reasons until exactly one literal at the current decision level
//! remains. That literal is the First Unique Implication Point (1-UIP).
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

    for &lit in &db.clause(conflict_clause).literals {
        let var = lit.var();
        if !seen[var as usize] {
            seen[var as usize] = true;
            let entry = trail.entry_for_var(var);
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
                num_at_current_level -= 1;
                seen[entry.lit.var() as usize] = false; // resolved away
                                                        // Add all other literals from the reason clause
                for &lit in &db.clause(reason_clause).literals {
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

    // Backtrack level: highest level among learned clause literals,
    // excluding the asserting literal (which is at current_level).
    let backtrack_level = learned
        .iter()
        .skip(1) // skip asserting literal
        .filter_map(|lit| trail.entry_for_var(lit.var()).map(|e| e.level))
        .max()
        .unwrap_or(0);

    AnalysisResult {
        learned,
        backtrack_level,
    }
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
}
