//! Boolean Constraint Propagation (BCP) engine.
//!
//! Runs unit propagation using tile-local clause checking. All propagated
//! assignments are written through the [`Trail`], which is the single source
//! of truth for variable assignments. This prevents ghost assignments — values
//! in the assignment array with no trail entry — which would corrupt conflict
//! analysis and backtracking.
//!
//! Currently CPU-only (SimWarp simulation). The ballot-based clause checking
//! pattern maps directly to GPU `Tile<SIZE>` when Rust nightly gets pred
//! register support for nvptx64 ballot.
//!
//! The phase-typed session ensures BCP only runs during the Propagate phase.

use crate::clause::ClausePool;
use crate::clause_tile::{self, ClauseStatus, ClauseTile};
use crate::literal::Lit;
use crate::phase::Propagate;
use crate::trail::Trail;

// ============================================================================
// BCP result
// ============================================================================

/// Result of running BCP to fixpoint.
///
/// Propagated literals are recorded on the trail (not returned here).
/// Query `trail.entries()` or `trail.len()` to inspect what was propagated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BcpResult {
    /// Propagation completed without conflict.
    Ok,
    /// A conflict was found. Contains the conflicting clause index.
    Conflict { clause_index: usize },
}

// ============================================================================
// Clause database
// ============================================================================

/// A clause: a disjunction of literals.
#[derive(Debug, Clone)]
pub struct Clause {
    /// The literals in this clause.
    pub literals: Vec<Lit>,
}

/// Clause database storing original and learned clauses.
///
/// Clauses are indexed by insertion order (0, 1, 2, ...).
/// The solver appends learned clauses during conflict analysis.
/// After `freeze_original()`, clauses at indices `0..num_original` are
/// protected from deletion; all subsequent clauses are "learned" and
/// eligible for LBD-based garbage collection.
pub struct ClauseDb {
    clauses: Vec<Clause>,
    /// Number of original (input) clauses. Set by `freeze_original()`.
    num_original: usize,
    /// LBD (Literal Block Distance) score per clause. 0 for original clauses.
    lbd: Vec<u16>,
    /// Tombstone flag: true if the clause has been deleted.
    deleted: Vec<bool>,
}

impl ClauseDb {
    pub fn new() -> Self {
        ClauseDb {
            clauses: Vec::new(),
            num_original: 0,
            lbd: Vec::new(),
            deleted: Vec::new(),
        }
    }

    /// Add a clause, returns its index.
    pub fn add_clause(&mut self, literals: Vec<Lit>) -> usize {
        let idx = self.clauses.len();
        self.clauses.push(Clause { literals });
        self.lbd.push(0);
        self.deleted.push(false);
        idx
    }

    /// Mark the current clause count as "original". Clauses added after this
    /// point are "learned" and eligible for LBD-based deletion.
    pub fn freeze_original(&mut self) {
        self.num_original = self.clauses.len();
    }

    /// Set the LBD score for a clause.
    pub fn set_lbd(&mut self, idx: usize, lbd: u16) {
        self.lbd[idx] = lbd;
    }

    /// Number of original (non-learned) clauses.
    pub fn num_original(&self) -> usize {
        self.num_original
    }

    /// Check if a clause has been deleted (tombstoned).
    pub fn is_deleted(&self, idx: usize) -> bool {
        self.deleted[idx]
    }

    /// Delete the worst learned clauses by LBD score.
    ///
    /// Keeps clauses that are:
    /// - Original (index < num_original)
    /// - Locked (currently a propagation reason on the trail)
    /// - "Glue" clauses: LBD ≤ 2 (binary learned clauses are always useful)
    ///
    /// Strategy: sort by LBD, keep the best half. This is the standard
    /// MiniSat/Glucose approach for balancing clause quality retention
    /// against watch list growth.
    ///
    /// Returns indices of deleted clauses (for watch list cleanup).
    pub fn reduce_learned(&mut self, locked: &[bool]) -> Vec<usize> {
        let mut candidates: Vec<(usize, u16)> = Vec::new();
        for i in self.num_original..self.clauses.len() {
            if self.deleted[i] {
                continue;
            }
            if i < locked.len() && locked[i] {
                continue;
            }
            // Protect glue clauses (LBD ≤ 2) — they bridge few decision levels
            // and provide critical implication chains
            if self.lbd[i] <= 2 {
                continue;
            }
            candidates.push((i, self.lbd[i]));
        }
        // Sort by LBD ascending (best/lowest first), delete the worst half
        candidates.sort_by_key(|&(_, lbd)| lbd);
        let keep = candidates.len() / 2;
        let to_delete: Vec<usize> = candidates[keep..].iter().map(|&(i, _)| i).collect();
        for &i in &to_delete {
            self.deleted[i] = true;
            self.clauses[i].literals.clear();
        }
        to_delete
    }

    /// Compact the database: remove tombstoned clauses and renumber.
    ///
    /// Returns a remap table: `remap[old_index] = Some(new_index)` for live
    /// clauses, `None` for deleted ones. Callers must update all stored clause
    /// indices (trail reasons, watch lists) using this table.
    pub fn compact(&mut self) -> Vec<Option<usize>> {
        let n = self.clauses.len();
        let mut remap: Vec<Option<usize>> = vec![None; n];
        let mut write = 0;

        for read in 0..n {
            if !self.deleted[read] {
                remap[read] = Some(write);
                if write != read {
                    self.clauses.swap(write, read);
                    self.lbd.swap(write, read);
                }
                write += 1;
            }
        }

        self.clauses.truncate(write);
        self.lbd.truncate(write);
        self.deleted.clear();
        self.deleted.resize(write, false);
        // num_original unchanged — original clauses are never deleted
        remap
    }

    /// Number of clauses (original + learned).
    pub fn len(&self) -> usize {
        self.clauses.len()
    }

    /// Whether the database contains no clauses.
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Get a clause by index.
    pub fn clause(&self, idx: usize) -> &Clause {
        &self.clauses[idx]
    }

    /// Highest variable index across all clauses. Returns 0 if empty.
    pub fn max_variable(&self) -> u32 {
        self.clauses
            .iter()
            .flat_map(|c| c.literals.iter())
            .map(|lit| lit.var())
            .max()
            .unwrap_or(0)
    }
}

impl Default for ClauseDb {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// BCP engine
// ============================================================================

/// Run BCP on the clause database, writing propagations through the trail.
///
/// Propagates unit clauses to fixpoint. On success, all propagated literals
/// are on the trail. On conflict, partial propagations from this BCP call
/// are also on the trail (so backtracking retracts them correctly).
///
/// Requires a `Propagate` phase proof — compile-time guarantee that BCP
/// only runs during the propagation phase of the CDCL loop.
pub fn run_bcp(
    db: &ClauseDb,
    trail: &mut Trail,
    _phase_proof: &crate::session::SolverSession<'_, Propagate>,
) -> BcpResult {
    trail.ensure_capacity(db.max_variable() as usize + 1);

    loop {
        let mut found_unit = false;

        // Evaluate non-tileable clauses directly (empty or >32 literals).
        for i in 0..db.len() {
            let clause = &db.clauses[i];
            if clause.literals.is_empty() || clause.literals.len() > 32 {
                match clause_tile::eval_clause_direct(&clause.literals, trail.assignments()) {
                    ClauseStatus::Conflict => {
                        return BcpResult::Conflict { clause_index: i };
                    }
                    ClauseStatus::Unit { propagate } => {
                        if trail.value(propagate.var()).is_none() {
                            trail.record_propagation(propagate, i);
                            found_unit = true;
                        }
                    }
                    ClauseStatus::Satisfied | ClauseStatus::Unresolved { .. } => {}
                }
            }
        }

        // Tile-based evaluation for clauses that fit (1-32 literals).
        let mut pool = ClausePool::new(db.len());
        let mut tiles: Vec<ClauseTile<Propagate>> = Vec::new();

        for i in 0..db.len() {
            let clause = &db.clauses[i];
            if clause.literals.is_empty() || clause.literals.len() > 32 {
                continue; // handled in direct evaluation above
            }
            let token = pool.acquire(i).unwrap();
            if let Some(tile) =
                clause_tile::make_clause_tile::<Propagate>(&clause.literals, token, i)
            {
                tiles.push(tile);
            }
        }

        let batches = clause_tile::pack_clauses(tiles);

        for batch in &batches {
            let results = batch.check_all(trail.assignments());
            for &(db_index, ref status) in &results {
                match status {
                    ClauseStatus::Conflict => {
                        return BcpResult::Conflict {
                            clause_index: db_index,
                        };
                    }
                    ClauseStatus::Unit { propagate } => {
                        if trail.value(propagate.var()).is_none() {
                            trail.record_propagation(*propagate, db_index);
                            found_unit = true;
                        }
                    }
                    ClauseStatus::Satisfied | ClauseStatus::Unresolved { .. } => {}
                }
            }
        }

        if !found_unit {
            break;
        }
    }

    BcpResult::Ok
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session;

    /// Helper: set up a trail with a decision, then run BCP.
    fn bcp_after_decision(db: &ClauseDb, trail: &mut Trail) -> BcpResult {
        session::with_session(|session| {
            let propagate = session.decide().propagate();
            run_bcp(db, trail, &propagate)
        })
    }

    #[test]
    fn simple_unit_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // ¬x0 ∨ x1
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]); // ¬x1 ∨ x2

        let mut trail = Trail::new(3);
        trail.new_decision(Lit::pos(0)); // x0=true
        let result = bcp_after_decision(&db, &mut trail);

        assert_eq!(result, BcpResult::Ok);
        assert_eq!(trail.value(1), Some(true));
        assert_eq!(trail.value(2), Some(true));
    }

    #[test]
    fn conflict_detection() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0)]); // ¬x0

        let mut trail = Trail::new(1);
        trail.new_decision(Lit::pos(0)); // x0=true → conflict
        let result = bcp_after_decision(&db, &mut trail);

        assert_eq!(result, BcpResult::Conflict { clause_index: 0 });
    }

    #[test]
    fn no_propagation_needed() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::pos(1), Lit::pos(2)]);

        let mut trail = Trail::new(3);
        trail.new_decision(Lit::pos(0));
        trail.record_propagation(Lit::pos(1), 0);
        trail.record_propagation(Lit::pos(2), 1);
        let result = bcp_after_decision(&db, &mut trail);

        assert_eq!(result, BcpResult::Ok);
    }

    #[test]
    fn chain_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // ¬x0 ∨ x1
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]); // ¬x1 ∨ x2
        db.add_clause(vec![Lit::neg(2), Lit::pos(3)]); // ¬x2 ∨ x3

        let mut trail = Trail::new(4);
        trail.new_decision(Lit::pos(0));
        let before = trail.len();
        let result = bcp_after_decision(&db, &mut trail);

        assert_eq!(result, BcpResult::Ok);
        assert_eq!(trail.len() - before, 3); // propagated 3 literals
        assert_eq!(trail.value(1), Some(true));
        assert_eq!(trail.value(2), Some(true));
        assert_eq!(trail.value(3), Some(true));
    }

    #[test]
    fn conflict_after_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // forces x1=true
        db.add_clause(vec![Lit::neg(0), Lit::neg(1)]); // forces x1=false

        let mut trail = Trail::new(2);
        trail.new_decision(Lit::pos(0));
        let result = bcp_after_decision(&db, &mut trail);

        match result {
            BcpResult::Conflict { .. } => {}
            other => panic!("expected Conflict, got {:?}", other),
        }
        // Partial propagation (x1=true) is on the trail — backtracking cleans it up
        assert!(trail.entry_for_var(1).is_some());
    }

    #[test]
    fn three_sat_instance() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::neg(0), Lit::pos(1), Lit::pos(3)]);
        db.add_clause(vec![Lit::neg(1), Lit::neg(2), Lit::pos(3)]);

        let mut trail = Trail::new(4);
        trail.new_decision(Lit::pos(0));
        trail.record_propagation(Lit::pos(1), 0);
        let result = bcp_after_decision(&db, &mut trail);

        assert_eq!(result, BcpResult::Ok);
    }

    #[test]
    fn batch_utilization() {
        let mut pool = ClausePool::new(8);
        let tiles: Vec<_> = (0..8)
            .filter_map(|i| {
                clause_tile::make_clause_tile::<Propagate>(
                    &[Lit::pos(0), Lit::pos(1)],
                    pool.acquire(i).unwrap(),
                    i,
                )
            })
            .collect();

        let batches = clause_tile::pack_clauses(tiles);
        assert_eq!(batches.len(), 1);
        assert!((batches[0].utilization() - 0.5).abs() < 0.01);
    }
}
