//! Assignment trail with decision levels and reasons.
//!
//! The trail records every variable assignment in order, annotated with:
//! - The decision level at which it was made
//! - The reason (decision or propagation by a specific clause)
//!
//! Backtracking unwinds the trail to a target level, retracting assignments.

use crate::literal::Lit;

/// Why a variable was assigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Chosen by the decide heuristic.
    Decision,
    /// Forced by unit propagation from the clause at this index.
    Propagation(usize),
}

/// A single trail entry.
#[derive(Debug, Clone, Copy)]
pub struct TrailEntry {
    pub lit: Lit,
    pub level: u32,
    pub reason: Reason,
}

/// The assignment trail.
pub struct Trail {
    entries: Vec<TrailEntry>,
    /// `level_starts[i]` = index in `entries` where decision level `i` begins.
    level_starts: Vec<usize>,
    current_level: u32,
}

impl Trail {
    pub fn new() -> Self {
        Trail {
            entries: Vec::new(),
            level_starts: vec![0], // level 0 starts at index 0
            current_level: 0,
        }
    }

    pub fn current_level(&self) -> u32 {
        self.current_level
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Record a new decision: increments the decision level and pushes the literal.
    pub fn new_decision(&mut self, lit: Lit, assignments: &mut [Option<bool>]) {
        self.current_level += 1;
        self.level_starts.push(self.entries.len());
        let value = !lit.is_negated();
        assignments[lit.var() as usize] = Some(value);
        self.entries.push(TrailEntry {
            lit,
            level: self.current_level,
            reason: Reason::Decision,
        });
    }

    /// Record a propagated literal at the current decision level.
    pub fn record_propagation(
        &mut self,
        lit: Lit,
        reason_clause: usize,
        assignments: &mut [Option<bool>],
    ) {
        let value = !lit.is_negated();
        assignments[lit.var() as usize] = Some(value);
        self.entries.push(TrailEntry {
            lit,
            level: self.current_level,
            reason: Reason::Propagation(reason_clause),
        });
    }

    /// Backtrack to the given decision level, retracting all assignments above it.
    /// Returns the retracted entries (for diagnostics; can be ignored).
    pub fn backtrack_to(&mut self, target_level: u32, assignments: &mut [Option<bool>]) {
        let start = self.level_starts[target_level as usize + 1];
        for entry in &self.entries[start..] {
            assignments[entry.lit.var() as usize] = None;
        }
        self.entries.truncate(start);
        self.level_starts.truncate(target_level as usize + 1);
        self.current_level = target_level;
    }

    /// Get the trail entry for a variable (most recent assignment).
    /// Linear scan from the end — fine for small instances.
    pub fn entry_for_var(&self, var: u32) -> Option<&TrailEntry> {
        self.entries.iter().rev().find(|e| e.lit.var() == var)
    }

    /// Iterate entries in assignment order.
    pub fn entries(&self) -> &[TrailEntry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_and_propagation() {
        let mut trail = Trail::new();
        let mut assign = vec![None; 4];

        // Level 0: propagate x0
        trail.record_propagation(Lit::pos(0), 0, &mut assign);
        assert_eq!(trail.current_level(), 0);
        assert_eq!(assign[0], Some(true));

        // Level 1: decide x1=true, propagate x2=true
        trail.new_decision(Lit::pos(1), &mut assign);
        assert_eq!(trail.current_level(), 1);
        trail.record_propagation(Lit::pos(2), 1, &mut assign);
        assert_eq!(assign[2], Some(true));

        // Level 2: decide x3=false
        trail.new_decision(Lit::neg(3), &mut assign);
        assert_eq!(trail.current_level(), 2);
        assert_eq!(assign[3], Some(false));

        assert_eq!(trail.len(), 4);
    }

    #[test]
    fn backtrack() {
        let mut trail = Trail::new();
        let mut assign = vec![None; 4];

        trail.record_propagation(Lit::pos(0), 0, &mut assign);
        trail.new_decision(Lit::pos(1), &mut assign);
        trail.record_propagation(Lit::pos(2), 1, &mut assign);
        trail.new_decision(Lit::neg(3), &mut assign);

        // Backtrack to level 1: undo x3 decision
        trail.backtrack_to(1, &mut assign);
        assert_eq!(trail.current_level(), 1);
        assert_eq!(assign[3], None); // retracted
        assert_eq!(assign[2], Some(true)); // kept (level 1)
        assert_eq!(assign[0], Some(true)); // kept (level 0)
        assert_eq!(trail.len(), 3); // x0, x1, x2

        // Backtrack to level 0: undo x1 decision and x2 propagation
        trail.backtrack_to(0, &mut assign);
        assert_eq!(trail.current_level(), 0);
        assert_eq!(assign[1], None);
        assert_eq!(assign[2], None);
        assert_eq!(assign[0], Some(true)); // kept
        assert_eq!(trail.len(), 1); // only x0
    }

    #[test]
    fn entry_lookup() {
        let mut trail = Trail::new();
        let mut assign = vec![None; 3];

        trail.new_decision(Lit::pos(0), &mut assign);
        trail.record_propagation(Lit::neg(1), 0, &mut assign);

        let e = trail.entry_for_var(1).unwrap();
        assert!(e.lit.is_negated());
        assert_eq!(e.level, 1);
        assert_eq!(e.reason, Reason::Propagation(0));

        assert!(trail.entry_for_var(2).is_none());
    }
}
