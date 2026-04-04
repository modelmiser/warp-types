//! Clause scheduler: ballot-driven idle-lane redistribution.
//!
//! When a tile finishes (clause satisfied or propagated), the scheduler
//! reclaims its token and assigns freed lanes to pending clauses.
//! All propagations go through the Trail — no ghost assignments.

use crate::bcp::{BcpResult, ClauseDb};
use crate::clause::ClausePool;
use crate::clause_tile::{self, ClauseBatch, ClauseStatus, ClauseTile};
use crate::phase::Propagate;
use crate::trail::Trail;

// ============================================================================
// Scheduler
// ============================================================================

/// Tracks which clauses need checking and redistributes work.
pub struct ClauseScheduler {
    pool: ClausePool,
    pending: Vec<usize>,
    done: Vec<usize>,
    num_clauses: usize,
}

/// Result of one scheduler round.
pub struct SchedulerRound {
    pub num_propagated: usize,
    pub conflict: Option<usize>,
    pub clauses_checked: usize,
    pub tiles_recycled: usize,
}

impl ClauseScheduler {
    pub fn new(db: &ClauseDb) -> Self {
        let n = db.len();
        ClauseScheduler {
            pool: ClausePool::new(n),
            pending: (0..n).collect(),
            done: Vec::new(),
            num_clauses: n,
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn done_count(&self) -> usize {
        self.done.len()
    }

    pub fn reset_round(&mut self) {
        self.done.clear();
        self.pool = ClausePool::new(self.num_clauses);
        self.pending = (0..self.num_clauses).collect();
    }

    fn fill_batch(&mut self, db: &ClauseDb) -> Option<ClauseBatch> {
        let mut tiles: Vec<ClauseTile<Propagate>> = Vec::new();
        let mut lanes_used = 0usize;

        while !self.pending.is_empty() {
            let clause_idx = self.pending[0];
            let clause = db.clause(clause_idx);
            let tile_size = clause_tile_size(clause.literals.len());

            if lanes_used + tile_size > 32 {
                break;
            }

            self.pending.remove(0);
            let token = self
                .pool
                .acquire(clause_idx)
                .expect("clause already acquired — affine discipline violated");

            if let Some(tile) =
                clause_tile::make_clause_tile::<Propagate>(&clause.literals, token, clause_idx)
            {
                lanes_used += tile.tile_size();
                tiles.push(tile);
            }
        }

        if tiles.is_empty() {
            None
        } else {
            Some(ClauseBatch::pack(tiles))
        }
    }

    /// Run one round. Propagations go through the trail.
    pub fn run_round(
        &mut self,
        db: &ClauseDb,
        trail: &mut Trail,
        _phase_proof: &crate::session::SolverSession<'_, Propagate>,
    ) -> SchedulerRound {
        let mut round = SchedulerRound {
            num_propagated: 0,
            conflict: None,
            clauses_checked: 0,
            tiles_recycled: 0,
        };

        // Evaluate non-tileable clauses directly (empty or >32 literals).
        let mut i = 0;
        while i < self.pending.len() {
            let clause_idx = self.pending[i];
            let clause = db.clause(clause_idx);
            if clause.literals.is_empty() || clause.literals.len() > 32 {
                self.pending.remove(i);
                match clause_tile::eval_clause_direct(&clause.literals, trail.assignments()) {
                    ClauseStatus::Conflict => {
                        round.conflict = Some(clause_idx);
                        return round;
                    }
                    ClauseStatus::Unit { propagate } => {
                        if trail.value(propagate.var()).is_none() {
                            trail.record_propagation(propagate, clause_idx);
                            round.num_propagated += 1;
                        }
                        self.done.push(clause_idx);
                    }
                    ClauseStatus::Satisfied => {
                        self.done.push(clause_idx);
                    }
                    ClauseStatus::Unresolved { .. } => {
                        // Still unresolved — will retry next round after reset
                    }
                }
                round.clauses_checked += 1;
            } else {
                i += 1;
            }
        }

        while let Some(batch) = self.fill_batch(db) {
            let results = batch.check_all(trail.assignments());
            round.clauses_checked += results.len();

            let mut recycle_indices = Vec::new();

            for &(db_index, ref status) in &results {
                match status {
                    ClauseStatus::Conflict => {
                        round.conflict = Some(db_index);
                        return round;
                    }
                    ClauseStatus::Satisfied => {
                        self.done.push(db_index);
                        recycle_indices.push(db_index);
                    }
                    ClauseStatus::Unit { propagate } => {
                        if trail.value(propagate.var()).is_none() {
                            trail.record_propagation(*propagate, db_index);
                            round.num_propagated += 1;
                        }
                        self.done.push(db_index);
                        recycle_indices.push(db_index);
                    }
                    ClauseStatus::Unresolved { .. } => {
                        recycle_indices.push(db_index);
                    }
                }
            }

            let tokens = batch.into_tokens();
            for token in tokens {
                self.pool.release(token);
            }

            round.tiles_recycled += recycle_indices.len();
        }

        round
    }
}

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
// Scheduled BCP
// ============================================================================

/// Run BCP with the scheduler, writing propagations through the trail.
pub fn run_bcp_scheduled(
    db: &ClauseDb,
    trail: &mut Trail,
    phase_proof: &crate::session::SolverSession<'_, Propagate>,
) -> BcpResult {
    trail.ensure_capacity(db.max_variable() as usize + 1);

    let mut scheduler = ClauseScheduler::new(db);

    loop {
        let round = scheduler.run_round(db, trail, phase_proof);

        if let Some(clause_idx) = round.conflict {
            return BcpResult::Conflict {
                clause_index: clause_idx,
            };
        }

        if round.num_propagated == 0 {
            break;
        }

        scheduler.reset_round();
    }

    BcpResult::Ok
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bcp::ClauseDb;
    use crate::literal::Lit;
    use crate::session;
    use crate::trail::Trail;

    fn scheduled_bcp(db: &ClauseDb, trail: &mut Trail) -> BcpResult {
        session::with_session(|s| {
            let p = s.decide().propagate();
            run_bcp_scheduled(db, trail, &p)
        })
    }

    #[test]
    fn simple_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);

        let mut trail = Trail::new(3);
        trail.new_decision(Lit::pos(0));
        let result = scheduled_bcp(&db, &mut trail);

        assert_eq!(result, BcpResult::Ok);
        assert_eq!(trail.value(1), Some(true));
        assert_eq!(trail.value(2), Some(true));
    }

    #[test]
    fn conflict_detection() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0)]);

        let mut trail = Trail::new(1);
        trail.new_decision(Lit::pos(0));
        let result = scheduled_bcp(&db, &mut trail);

        assert_eq!(result, BcpResult::Conflict { clause_index: 0 });
    }

    #[test]
    fn chain_propagation() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::neg(2), Lit::pos(3)]);

        let mut trail = Trail::new(4);
        trail.new_decision(Lit::pos(0));
        let before = trail.len();
        let result = scheduled_bcp(&db, &mut trail);

        assert_eq!(result, BcpResult::Ok);
        assert_eq!(trail.len() - before, 3);
        assert_eq!(trail.value(3), Some(true));
    }

    #[test]
    fn scheduled_matches_basic() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::pos(0), Lit::pos(2)]);

        let mut trail1 = Trail::new(3);
        trail1.new_decision(Lit::pos(0));
        let mut trail2 = Trail::new(3);
        trail2.new_decision(Lit::pos(0));

        let r1 = session::with_session(|s| {
            let p = s.decide().propagate();
            crate::bcp::run_bcp(&db, &mut trail1, &p)
        });
        let r2 = scheduled_bcp(&db, &mut trail2);

        assert_eq!(r1, r2);
        assert_eq!(trail1.assignments(), trail2.assignments());
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
        let mut trail = Trail::new(inst.num_vars as usize);

        let result = session::with_session(|s| {
            let p = s.propagate();
            run_bcp_scheduled(&inst.db, &mut trail, &p)
        });

        match result {
            BcpResult::Conflict { .. } => {}
            other => panic!("expected conflict, got {:?}", other),
        }
    }

    #[test]
    fn many_clauses_utilization() {
        let mut db = ClauseDb::new();
        for i in 0..20 {
            db.add_clause(vec![Lit::neg(i), Lit::pos(i + 1)]);
        }

        let mut trail = Trail::new(21);
        trail.new_decision(Lit::pos(0));
        let before = trail.len();
        let result = scheduled_bcp(&db, &mut trail);

        match result {
            BcpResult::Ok => {
                assert_eq!(trail.len() - before, 20);
                for i in 0..21 {
                    assert_eq!(trail.value(i), Some(true));
                }
            }
            other => panic!("expected Ok, got {:?}", other),
        }
    }
}
