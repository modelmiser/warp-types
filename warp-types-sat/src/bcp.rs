//! Boolean Constraint Propagation (BCP) engine.
//!
//! Runs unit propagation using tile-local clause checking.
//! Each propagation step checks all clauses in parallel (via batched tiles),
//! identifies unit clauses, and propagates the forced assignments.
//!
//! # Architecture
//!
//! Currently CPU-only (SimWarp simulation). The ballot-based clause checking
//! pattern maps directly to GPU Tile<SIZE> when Rust nightly gets pred register
//! support for nvptx64 ballot. Until then, this validates the algorithm and
//! type-system design on CPU.
//!
//! The phase-typed session ensures BCP only runs during Propagate phase.

use crate::clause::ClausePool;
use crate::clause_tile::{self, ClauseStatus, ClauseTile};
use crate::literal::Lit;
use crate::phase::Propagate;

// ============================================================================
// BCP result
// ============================================================================

/// Result of running BCP to fixpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BcpResult {
    /// Propagation completed without conflict.
    /// Contains the list of propagated literals (the implication trail segment).
    Ok { propagated: Vec<Lit> },
    /// A conflict was found. Contains the conflicting clause index.
    Conflict { clause_index: usize },
}

// ============================================================================
// Clause database (simple, for testing)
// ============================================================================

/// A clause: a disjunction of literals.
#[derive(Debug, Clone)]
pub struct Clause {
    pub literals: Vec<Lit>,
}

/// Simple clause database for BCP testing.
pub struct ClauseDb {
    clauses: Vec<Clause>,
}

impl ClauseDb {
    pub fn new() -> Self {
        ClauseDb {
            clauses: Vec::new(),
        }
    }

    /// Add a clause, returns its index.
    pub fn add_clause(&mut self, literals: Vec<Lit>) -> usize {
        let idx = self.clauses.len();
        self.clauses.push(Clause { literals });
        idx
    }

    pub fn len(&self) -> usize {
        self.clauses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    pub fn clause(&self, idx: usize) -> &Clause {
        &self.clauses[idx]
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

/// Run BCP on the clause database with the given assignment.
///
/// Propagates unit clauses to fixpoint. Returns the list of propagated
/// literals or a conflict.
///
/// This function requires a `Propagate` phase proof — it can only be called
/// during the propagation phase of the CDCL loop. The phase token is not
/// consumed (BCP is an operation within Propagate, not a transition out of it).
pub fn run_bcp(
    db: &ClauseDb,
    assignments: &mut Vec<Option<bool>>,
    _phase_proof: &crate::session::SolverSession<'_, Propagate>,
) -> BcpResult {
    let mut propagated = Vec::new();

    // Propagate to fixpoint
    loop {
        let mut found_unit = false;

        // Build clause tiles and pack into batches
        let mut pool = ClausePool::new(db.len());
        let mut tiles: Vec<ClauseTile<Propagate>> = Vec::new();

        for i in 0..db.len() {
            let clause = &db.clauses[i];
            let token = pool.acquire(i).unwrap();
            tiles.push(clause_tile::make_clause_tile::<Propagate>(
                &clause.literals,
                token,
            ));
        }

        let batches = clause_tile::pack_clauses(tiles);

        // Check all batches
        for batch in &batches {
            let results = batch.check_all(assignments);
            for (tile_idx, status) in results.iter().enumerate() {
                match status {
                    ClauseStatus::Conflict => {
                        // Release all tokens before returning
                        return BcpResult::Conflict {
                            clause_index: tile_idx,
                        };
                    }
                    ClauseStatus::Unit { propagate } => {
                        // Propagate the forced literal
                        let var = propagate.var() as usize;
                        let value = !propagate.is_negated();
                        if assignments[var].is_none() {
                            assignments[var] = Some(value);
                            propagated.push(*propagate);
                            found_unit = true;
                        }
                    }
                    ClauseStatus::Satisfied | ClauseStatus::Unresolved { .. } => {
                        // Nothing to do
                    }
                }
            }
        }

        if !found_unit {
            break;
        }
    }

    BcpResult::Ok { propagated }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session;

    /// Helper: run BCP within a solver session.
    fn bcp_in_session(db: &ClauseDb, assignments: &mut Vec<Option<bool>>) -> BcpResult {
        session::with_session(|session| {
            let decide = session.decide();
            // We need a Propagate session to call run_bcp.
            // Use the generic transition since propagate() returns PropagationOutcome.
            let propagate: crate::session::SolverSession<'_, Propagate> =
                decide.transition();
            run_bcp(db, assignments, &propagate)
        })
    }

    #[test]
    fn simple_unit_propagation() {
        // x0 = true (decision)
        // Clause: (¬x0 ∨ x1)  → ¬x0 = false, so x1 must be true
        // Clause: (¬x1 ∨ x2)  → after x1 = true, x2 must be true
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);

        let mut assign = vec![Some(true), None, None]; // x0 = true
        let result = bcp_in_session(&db, &mut assign);

        assert_eq!(
            result,
            BcpResult::Ok {
                propagated: vec![Lit::pos(1), Lit::pos(2)]
            }
        );
        assert_eq!(assign, vec![Some(true), Some(true), Some(true)]);
    }

    #[test]
    fn conflict_detection() {
        // x0 = true (decision)
        // Clause: (¬x0)  → conflict immediately
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0)]);

        let mut assign = vec![Some(true)];
        let result = bcp_in_session(&db, &mut assign);

        match result {
            BcpResult::Conflict { .. } => {} // expected
            other => panic!("expected Conflict, got {:?}", other),
        }
    }

    #[test]
    fn no_propagation_needed() {
        // All clauses already satisfied
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::pos(1), Lit::pos(2)]);

        let mut assign = vec![Some(true), Some(true), Some(true)];
        let result = bcp_in_session(&db, &mut assign);

        assert_eq!(result, BcpResult::Ok { propagated: vec![] });
    }

    #[test]
    fn chain_propagation() {
        // Implication chain: x0 → x1 → x2 → x3
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // ¬x0 ∨ x1
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]); // ¬x1 ∨ x2
        db.add_clause(vec![Lit::neg(2), Lit::pos(3)]); // ¬x2 ∨ x3

        let mut assign = vec![Some(true), None, None, None];
        let result = bcp_in_session(&db, &mut assign);

        assert_eq!(
            result,
            BcpResult::Ok {
                propagated: vec![Lit::pos(1), Lit::pos(2), Lit::pos(3)]
            }
        );
        assert_eq!(
            assign,
            vec![Some(true), Some(true), Some(true), Some(true)]
        );
    }

    #[test]
    fn conflict_after_propagation() {
        // x0 → x1, x0 → ¬x1 (contradiction)
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);  // ¬x0 ∨ x1 (forces x1=true)
        db.add_clause(vec![Lit::neg(0), Lit::neg(1)]);  // ¬x0 ∨ ¬x1 (forces x1=false)

        let mut assign = vec![Some(true), None];
        let result = bcp_in_session(&db, &mut assign);

        match result {
            BcpResult::Conflict { .. } => {} // expected
            other => panic!("expected Conflict, got {:?}", other),
        }
    }

    #[test]
    fn three_sat_instance() {
        // A satisfiable 3-SAT instance:
        // (x0 ∨ x1 ∨ x2) ∧ (¬x0 ∨ x1 ∨ x3) ∧ (¬x1 ∨ ¬x2 ∨ x3)
        // With x0=true, x1=true: all clauses satisfied, no propagation needed.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::neg(0), Lit::pos(1), Lit::pos(3)]);
        db.add_clause(vec![Lit::neg(1), Lit::neg(2), Lit::pos(3)]);

        let mut assign = vec![Some(true), Some(true), None, None];
        let result = bcp_in_session(&db, &mut assign);

        assert_eq!(result, BcpResult::Ok { propagated: vec![] });
    }

    #[test]
    fn batch_utilization() {
        // 8 binary clauses → 8 tiles of 4 → 1 batch of 32 lanes
        // Utilization: 16 real literals / 32 lanes = 50%
        let mut pool = ClausePool::new(8);
        let tiles: Vec<_> = (0..8)
            .map(|i| {
                clause_tile::make_clause_tile::<Propagate>(
                    &[Lit::pos(0), Lit::pos(1)],
                    pool.acquire(i).unwrap(),
                )
            })
            .collect();

        let batches = clause_tile::pack_clauses(tiles);
        assert_eq!(batches.len(), 1);
        assert!((batches[0].utilization() - 0.5).abs() < 0.01);
    }
}
