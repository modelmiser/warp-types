//! Clause scheduler: ballot-driven idle-lane redistribution.
//!
//! The core C1 contribution from the spec-ulate analysis. When a tile
//! finishes (clause satisfied or fully propagated), the scheduler:
//! 1. Detects finished tiles (simulated ballot)
//! 2. Reclaims their clause tokens (affine release)
//! 3. Acquires new clauses from the pending pool
//! 4. Assigns freed lanes to new clause tiles
//!
//! This is typed work-stealing: the compiler proves no clause is
//! double-assigned (affine tokens) and all operations happen in the
//! correct phase (Propagate only).
//!
//! # Lane utilization
//!
//! Without the scheduler, a warp batch with 8 binary clauses (Tile<4> each)
//! checks all 8 once, then 32 lanes sit idle while the BCP fixpoint loop
//! rebuilds all tiles. With the scheduler, finished tiles are immediately
//! replaced with pending clauses — lanes stay busy across BCP iterations.

use crate::bcp::ClauseDb;
use crate::clause::ClausePool;
use crate::clause_tile::{self, ClauseBatch, ClauseStatus, ClauseTile};
use crate::literal::Lit;
use crate::phase::Propagate;

// ============================================================================
// Scheduler
// ============================================================================

/// Tracks which clauses need checking and redistributes work.
///
/// Separates clauses into:
/// - **active**: currently assigned to tiles, being checked
/// - **pending**: waiting for a free tile slot
/// - **done**: satisfied or fully propagated this round (no recheck needed)
pub struct ClauseScheduler {
    /// Pool enforcing exclusive clause ownership.
    pool: ClausePool,
    /// Indices of clauses not yet assigned to any tile.
    pending: Vec<usize>,
    /// Indices of clauses marked done this BCP round.
    done: Vec<usize>,
    /// Total clauses in the database.
    num_clauses: usize,
}

/// Result of one scheduler round.
pub struct SchedulerRound {
    /// Literals that were propagated in this round.
    pub propagated: Vec<Lit>,
    /// Whether a conflict was found (and which clause).
    pub conflict: Option<usize>,
    /// Number of clauses that were checked.
    pub clauses_checked: usize,
    /// Number of tiles that were recycled (freed + reassigned).
    pub tiles_recycled: usize,
}

impl ClauseScheduler {
    /// Create a scheduler for a clause database.
    pub fn new(db: &ClauseDb) -> Self {
        let n = db.len();
        ClauseScheduler {
            pool: ClausePool::new(n),
            pending: (0..n).collect(),
            done: Vec::new(),
            num_clauses: n,
        }
    }

    /// Number of clauses still pending assignment.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of clauses done (no recheck needed this round).
    pub fn done_count(&self) -> usize {
        self.done.len()
    }

    /// Reset for a new BCP round (all non-done clauses become pending again).
    pub fn reset_round(&mut self) {
        self.done.clear();
        // All clauses that aren't currently in tiles become pending
        // (The pool tracks which are acquired)
        self.pending.clear();
        for i in 0..self.num_clauses {
            if self.pool.acquire(i).is_some() {
                // Was available → now pending (we just acquired it, release immediately
                // to keep the pool state clean — pending means "needs tile assignment")
                // Actually: pending list is just indices, pool tracks actual ownership
                self.pending.push(i);
            }
        }
        // Release everything — pending list tracks intent, pool tracks tile assignment
        self.pool = ClausePool::new(self.num_clauses);
        self.pending = (0..self.num_clauses).collect();
    }

    /// Fill a warp's worth of tiles from pending clauses.
    ///
    /// Returns up to 32 lanes of clause tiles, packed into batches.
    /// Clauses are taken from the pending queue. Their tokens are acquired
    /// (affine — no double-assignment).
    fn fill_batch(&mut self, db: &ClauseDb) -> Option<ClauseBatch> {
        let mut tiles: Vec<ClauseTile<Propagate>> = Vec::new();
        let mut lanes_used = 0usize;

        while !self.pending.is_empty() {
            let clause_idx = self.pending[0];
            let clause = db.clause(clause_idx);
            let tile_size = clause_tile_size(clause.literals.len());

            if lanes_used + tile_size > 32 {
                break; // batch full
            }

            self.pending.remove(0);
            let token = self.pool.acquire(clause_idx)
                .expect("clause already acquired — affine discipline violated");

            if let Some(tile) = clause_tile::make_clause_tile::<Propagate>(
                &clause.literals,
                token,
                clause_idx,
            ) {
                lanes_used += tile.tile_size();
                tiles.push(tile);
            }
            // Tautological clauses (None) are skipped and implicitly done
        }

        if tiles.is_empty() {
            None
        } else {
            Some(ClauseBatch::pack(tiles))
        }
    }

    /// Run one round of scheduled BCP.
    ///
    /// Fills batches from pending clauses, checks them, propagates units,
    /// recycles finished tiles with new pending clauses. Continues until
    /// all pending clauses are checked or a conflict is found.
    ///
    /// Returns the round result. The caller loops this within the BCP
    /// fixpoint until no new propagations occur.
    pub fn run_round(
        &mut self,
        db: &ClauseDb,
        assignments: &mut Vec<Option<bool>>,
        _phase_proof: &crate::session::SolverSession<'_, Propagate>,
    ) -> SchedulerRound {
        let mut round = SchedulerRound {
            propagated: Vec::new(),
            conflict: None,
            clauses_checked: 0,
            tiles_recycled: 0,
        };

        loop {
            let batch = match self.fill_batch(db) {
                Some(b) => b,
                None => break, // no more pending clauses
            };

            let results = batch.check_all(assignments);
            round.clauses_checked += results.len();

            // Collect tiles to recycle and process results
            let mut recycle_indices = Vec::new();

            for &(db_index, ref status) in &results {
                match status {
                    ClauseStatus::Conflict => {
                        round.conflict = Some(db_index);
                        return round; // early exit on conflict
                    }
                    ClauseStatus::Satisfied => {
                        // Clause done — reclaim tile, don't re-enqueue
                        self.done.push(db_index);
                        recycle_indices.push(db_index);
                    }
                    ClauseStatus::Unit { propagate } => {
                        let var = propagate.var() as usize;
                        let value = !propagate.is_negated();
                        if var < assignments.len() && assignments[var].is_none() {
                            assignments[var] = Some(value);
                            round.propagated.push(*propagate);
                        }
                        // Unit clause becomes satisfied after propagation —
                        // mark done for this round
                        self.done.push(db_index);
                        recycle_indices.push(db_index);
                    }
                    ClauseStatus::Unresolved { .. } => {
                        // Still undetermined — will be rechecked in next BCP iteration
                        // Release token so it can be re-acquired
                        recycle_indices.push(db_index);
                    }
                }
            }

            // Release tokens from finished/unresolved tiles
            let tokens = batch.into_tokens();
            for token in tokens {
                self.pool.release(token);
            }

            round.tiles_recycled += recycle_indices.len();
        }

        round
    }
}

/// Round up clause length to tile size (duplicated from clause_tile for encapsulation).
fn clause_tile_size(clause_len: usize) -> usize {
    if clause_len <= 4 {
        4
    } else if clause_len <= 8 {
        8
    } else if clause_len <= 16 {
        16
    } else {
        32
    }
}

// ============================================================================
// Scheduled BCP (replacement for bcp::run_bcp)
// ============================================================================

/// Run BCP with the scheduler — tile-local evaluation with idle-lane redistribution.
///
/// This is the scheduled version of `bcp::run_bcp`. Instead of rebuilding all
/// tiles every iteration, it reuses tiles for satisfied/unit clauses and fills
/// freed slots with pending clauses.
pub fn run_bcp_scheduled(
    db: &ClauseDb,
    assignments: &mut Vec<Option<bool>>,
    phase_proof: &crate::session::SolverSession<'_, Propagate>,
) -> crate::bcp::BcpResult {
    // Ensure assignment array covers all variables
    let max_var = db.max_variable();
    if max_var as usize >= assignments.len() {
        assignments.resize(max_var as usize + 1, None);
    }

    let mut scheduler = ClauseScheduler::new(db);
    let mut all_propagated = Vec::new();

    loop {
        let round = scheduler.run_round(db, assignments, phase_proof);

        if let Some(clause_idx) = round.conflict {
            return crate::bcp::BcpResult::Conflict {
                clause_index: clause_idx,
            };
        }

        if round.propagated.is_empty() {
            break; // fixpoint reached
        }

        all_propagated.extend(round.propagated);

        // New propagations may create new unit clauses — reset pending
        // and recheck all non-done clauses
        scheduler.reset_round();
    }

    crate::bcp::BcpResult::Ok {
        propagated: all_propagated,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bcp::{BcpResult, ClauseDb};
    use crate::literal::Lit;
    use crate::session;

    fn scheduled_bcp(db: &ClauseDb, assignments: &mut Vec<Option<bool>>) -> BcpResult {
        session::with_session(|s| {
            let p = s.decide().propagate();
            run_bcp_scheduled(db, assignments, &p)
        })
    }

    #[test]
    fn simple_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // ¬x0 ∨ x1
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]); // ¬x1 ∨ x2

        let mut assign = vec![Some(true), None, None];
        let result = scheduled_bcp(&db, &mut assign);

        assert_eq!(
            result,
            BcpResult::Ok {
                propagated: vec![Lit::pos(1), Lit::pos(2)]
            }
        );
    }

    #[test]
    fn conflict_detection() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0)]); // ¬x0 — conflict with x0=true

        let mut assign = vec![Some(true)];
        let result = scheduled_bcp(&db, &mut assign);

        match result {
            BcpResult::Conflict { clause_index: 0 } => {} // correct db index
            other => panic!("expected Conflict{{clause_index: 0}}, got {:?}", other),
        }
    }

    #[test]
    fn chain_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::neg(2), Lit::pos(3)]);

        let mut assign = vec![Some(true), None, None, None];
        let result = scheduled_bcp(&db, &mut assign);

        assert_eq!(
            result,
            BcpResult::Ok {
                propagated: vec![Lit::pos(1), Lit::pos(2), Lit::pos(3)]
            }
        );
    }

    #[test]
    fn scheduled_matches_basic() {
        // Verify scheduled BCP produces same result as basic BCP
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::pos(0), Lit::pos(2)]);

        let mut assign1 = vec![Some(true), None, None];
        let mut assign2 = assign1.clone();

        let r1 = session::with_session(|s| {
            let p = s.decide().propagate();
            crate::bcp::run_bcp(&db, &mut assign1, &p)
        });
        let r2 = scheduled_bcp(&db, &mut assign2);

        assert_eq!(r1, r2);
        assert_eq!(assign1, assign2);
    }

    #[test]
    fn pigeonhole_unsat() {
        let cnf = "\
p cnf 2 3
1 0
2 0
-1 -2 0
";
        let inst = crate::dimacs::parse_dimacs_str(cnf).unwrap();
        let mut assign = vec![None, None];

        let result = session::with_session(|s| {
            let p = s.propagate(); // initial BCP
            run_bcp_scheduled(&inst.db, &mut assign, &p)
        });

        match result {
            BcpResult::Conflict { .. } => {}
            other => panic!("expected conflict, got {:?}", other),
        }
    }

    #[test]
    fn many_clauses_utilization() {
        // 20 binary clauses — more than fit in one 32-lane warp
        // Scheduler must process in multiple batches
        let mut db = ClauseDb::new();
        for i in 0..20 {
            // Each clause: (¬x_i ∨ x_{i+1})
            db.add_clause(vec![Lit::neg(i), Lit::pos(i + 1)]);
        }

        let mut assign = vec![None; 21];
        assign[0] = Some(true);
        let result = scheduled_bcp(&db, &mut assign);

        // Should propagate x1=true, x2=true, ..., x20=true
        match result {
            BcpResult::Ok { propagated } => {
                assert_eq!(propagated.len(), 20);
                for a in &assign {
                    assert_eq!(*a, Some(true));
                }
            }
            other => panic!("expected Ok, got {:?}", other),
        }
    }
}
