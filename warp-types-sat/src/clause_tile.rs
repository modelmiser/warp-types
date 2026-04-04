//! Clause tiles: tile-local ballot-based clause checking.
//!
//! Maps each clause to a SimWarp tile segment. Each lane in the segment
//! holds one literal of the clause. Checking the clause is a single
//! ballot operation: O(1) per clause per propagation step, not O(literals).
//!
//! # Tile sizes
//!
//! Clauses are rounded up to the next power-of-2 tile size (4, 8, 16, 32).
//! A 3-literal clause → tile of 4 (1 lane padding).
//! A 5-literal clause → tile of 8 (3 lanes padding).
//! Padding lanes are treated as "satisfied" (don't affect clause status).
//!
//! # SimWarp simulation
//!
//! On CPU, we simulate tile-local operations using SimWarp segments.
//! On GPU, these map directly to `Tile<SIZE>` ballot/reduce operations.

use crate::clause::ClauseToken;
use crate::literal::Lit;
use crate::phase::Phase;
use core::marker::PhantomData;

// ============================================================================
// Clause status (result of tile-local evaluation)
// ============================================================================

/// Result of evaluating a clause against the current assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseStatus {
    /// At least one literal is true — clause is satisfied.
    Satisfied,
    /// Exactly one literal is unassigned, rest are false — unit clause.
    /// The unassigned literal must be propagated.
    Unit { propagate: Lit },
    /// All literals are false — conflict.
    Conflict,
    /// Multiple literals are unassigned — clause is not yet determined.
    Unresolved { unassigned_count: u32 },
}

// ============================================================================
// Clause tile
// ============================================================================

/// A clause mapped to a tile-sized segment for parallel evaluation.
///
/// `P: Phase` restricts which operations are available:
/// - `Propagate` phase: `check()` and `find_unit()` are available
/// - Other phases: clause tile exists but checking is gated
///
/// The `ClauseToken` proves exclusive ownership of this clause.
pub struct ClauseTile<P: Phase> {
    /// Literals in this clause (deduplicated, padded to tile size).
    literals: Vec<Lit>,
    /// Original clause length (after dedup, before padding).
    clause_len: usize,
    /// Tile size (4, 8, 16, or 32).
    tile_size: usize,
    /// Original clause database index (survives bin-packing reorder).
    db_index: usize,
    /// Ownership token (affine).
    _token: ClauseToken,
    /// Phase marker.
    _phase: PhantomData<P>,
}

impl<P: Phase> ClauseTile<P> {
    /// Number of real literals (not padding).
    pub fn clause_len(&self) -> usize {
        self.clause_len
    }

    /// Tile size (power of 2).
    pub fn tile_size(&self) -> usize {
        self.tile_size
    }

    /// Original clause database index.
    pub fn db_index(&self) -> usize {
        self.db_index
    }

    /// Recover the clause token (release ownership).
    pub fn into_token(self) -> ClauseToken {
        self._token
    }
}

// ============================================================================
// Clause tile construction
// ============================================================================

/// Round up to the next valid tile size (4, 8, 16, 32).
/// Clauses longer than 32 are not supported in a single tile.
fn tile_size_for(clause_len: usize) -> usize {
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

/// Create a clause tile from literals, a clause token, and the database index.
///
/// Deduplicates literals (removes exact duplicates). Detects tautologies
/// (clause containing both `x` and `¬x`) and returns `None` for them.
/// Pads to the next power-of-2 tile size.
pub fn make_clause_tile<P: Phase>(
    literals: &[Lit],
    token: ClauseToken,
    db_index: usize,
) -> Option<ClauseTile<P>> {
    if literals.is_empty() || literals.len() > 32 {
        // Empty clauses (trivially false) and oversize clauses (>32 literals)
        // can't be packed into a tile. Caller handles them via eval_clause_direct.
        // Token is consumed and dropped (pool is loop-local in BCP).
        return None;
    }

    // Deduplicate literals
    let mut deduped: Vec<Lit> = Vec::with_capacity(literals.len());
    for &lit in literals {
        // Check for tautology: literal and its complement both present
        if deduped.iter().any(|&l| l == lit.complement()) {
            return None; // tautological clause — always satisfied
        }
        if !deduped.contains(&lit) {
            deduped.push(lit);
        }
    }

    let clause_len = deduped.len();
    let ts = tile_size_for(clause_len);

    // Pad to tile size. Padding lanes are never evaluated: check() breaks at
    // clause_len, and eval_clause_direct is not used for tileable clauses.
    // Sentinel variable u32::MAX/2 is chosen to avoid collision with real variables
    // (any real variable this large would overflow Lit encoding, caught by debug_assert).
    deduped.resize(ts, Lit::pos(u32::MAX / 2));

    Some(ClauseTile {
        literals: deduped,
        clause_len,
        tile_size: ts,
        db_index,
        _token: token,
        _phase: PhantomData,
    })
}

// ============================================================================
// Direct clause evaluation (fallback for non-tileable clauses)
// ============================================================================

/// Evaluate a clause directly without tile packing.
///
/// Used for clauses that don't fit in a tile (empty or >32 literals).
/// Same logic as `ClauseTile::check()` but operates on raw literals.
pub fn eval_clause_direct(literals: &[Lit], assignments: &[Option<bool>]) -> ClauseStatus {
    let mut unassigned_count = 0u32;
    let mut last_unassigned = None;

    for &lit in literals {
        match lit.eval(assignments) {
            Some(true) => return ClauseStatus::Satisfied,
            Some(false) => {}
            None => {
                unassigned_count += 1;
                last_unassigned = Some(lit);
            }
        }
    }

    if unassigned_count == 0 {
        ClauseStatus::Conflict
    } else if unassigned_count == 1 {
        ClauseStatus::Unit {
            propagate: last_unassigned.unwrap(),
        }
    } else {
        ClauseStatus::Unresolved { unassigned_count }
    }
}

// ============================================================================
// Tile-local evaluation (SimWarp simulation)
// ============================================================================

impl ClauseTile<crate::phase::Propagate> {
    /// Evaluate the clause against the current assignment using tile-local ballot.
    ///
    /// This is the core BCP primitive. On GPU, this maps to:
    /// ```text
    /// ballot(literal_is_true)  → any() → Satisfied
    /// ballot(literal_is_unassigned) → popcount == 1 → Unit
    /// ballot(literal_is_false) → all() → Conflict
    /// ```
    ///
    /// On SimWarp (CPU simulation), we evaluate each lane and simulate the ballot.
    pub fn check(&self, assignments: &[Option<bool>]) -> ClauseStatus {
        // Evaluate each lane (literal) against the assignment.
        // Padding lanes beyond clause_len are treated as "satisfied".
        let mut any_satisfied = false;
        let mut unassigned_count = 0u32;
        let mut last_unassigned = None;

        for (i, lit) in self.literals.iter().enumerate() {
            if i >= self.clause_len {
                // Padding lane — doesn't affect clause status
                break;
            }
            match lit.eval(assignments) {
                Some(true) => {
                    any_satisfied = true;
                    // Early exit — at least one literal true
                    break;
                }
                Some(false) => {
                    // This literal is falsified — continue
                }
                None => {
                    unassigned_count += 1;
                    last_unassigned = Some(*lit);
                }
            }
        }

        if any_satisfied {
            ClauseStatus::Satisfied
        } else if unassigned_count == 1 {
            ClauseStatus::Unit {
                propagate: last_unassigned.unwrap(),
            }
        } else if unassigned_count == 0 {
            ClauseStatus::Conflict
        } else {
            ClauseStatus::Unresolved { unassigned_count }
        }
    }
}

// ============================================================================
// Batch evaluation: check multiple clauses across a warp
// ============================================================================

/// A batch of clause tiles packed into a single warp's worth of lanes.
///
/// Multiple small clauses can share one 32-lane warp. For example:
/// - 8 clauses of 4 literals each = 32 lanes = one full warp
/// - 4 clauses of 8 literals each = 32 lanes = one full warp
///
/// This is where the real lane utilization improvement comes from.
pub struct ClauseBatch {
    tiles: Vec<ClauseTile<crate::phase::Propagate>>,
    total_lanes: usize,
}

impl ClauseBatch {
    /// Pack clause tiles into a batch. Total lanes must not exceed 32.
    pub fn pack(tiles: Vec<ClauseTile<crate::phase::Propagate>>) -> Self {
        let total_lanes: usize = tiles.iter().map(|t| t.tile_size()).sum();
        assert!(
            total_lanes <= 32,
            "batch exceeds warp width: {total_lanes} lanes > 32"
        );
        ClauseBatch { tiles, total_lanes }
    }

    /// Number of clauses in this batch.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Total lanes used (including padding).
    pub fn total_lanes(&self) -> usize {
        self.total_lanes
    }

    /// Lane utilization: real literals / total lanes.
    pub fn utilization(&self) -> f64 {
        let real: usize = self.tiles.iter().map(|t| t.clause_len()).sum();
        if self.total_lanes == 0 {
            0.0
        } else {
            real as f64 / self.total_lanes as f64
        }
    }

    /// Check all clauses in the batch. Returns (db_index, status) for each.
    ///
    /// On GPU, this would be a single warp-wide ballot per check,
    /// with each tile segment computing independently.
    pub fn check_all(&self, assignments: &[Option<bool>]) -> Vec<(usize, ClauseStatus)> {
        self.tiles
            .iter()
            .map(|tile| (tile.db_index(), tile.check(assignments)))
            .collect()
    }

    /// Recover all clause tokens from the batch (release ownership).
    pub fn into_tokens(self) -> Vec<ClauseToken> {
        self.tiles.into_iter().map(|t| t.into_token()).collect()
    }
}

// ============================================================================
// Batch packing: greedy bin-packing of clauses into warps
// ============================================================================

/// Pack clauses into warp-sized batches using first-fit-decreasing bin packing.
///
/// Returns batches where each batch uses at most 32 lanes.
/// Larger clauses are packed first (better utilization).
pub fn pack_clauses(mut tiles: Vec<ClauseTile<crate::phase::Propagate>>) -> Vec<ClauseBatch> {
    // Sort by tile size descending (first-fit-decreasing)
    tiles.sort_by_key(|t| std::cmp::Reverse(t.tile_size()));

    let mut batches: Vec<Vec<ClauseTile<crate::phase::Propagate>>> = Vec::new();
    let mut batch_lanes: Vec<usize> = Vec::new();

    for tile in tiles {
        let ts = tile.tile_size();
        // Find first batch with enough room
        let slot = batch_lanes.iter().position(|&used| used + ts <= 32);
        match slot {
            Some(idx) => {
                batch_lanes[idx] += ts;
                batches[idx].push(tile);
            }
            None => {
                batch_lanes.push(ts);
                batches.push(vec![tile]);
            }
        }
    }

    batches.into_iter().map(ClauseBatch::pack).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause::ClausePool;
    use crate::literal::Lit;
    use crate::phase::Propagate;

    fn make_test_tile(lits: &[Lit], pool: &mut ClausePool, idx: usize) -> ClauseTile<Propagate> {
        let token = pool.acquire(idx).unwrap();
        make_clause_tile::<Propagate>(lits, token, idx).expect("tautological clause in test")
    }

    #[test]
    fn satisfied_clause() {
        let mut pool = ClausePool::new(1);
        // Clause: (x0 ∨ x1 ∨ x2)
        let tile = make_test_tile(&[Lit::pos(0), Lit::pos(1), Lit::pos(2)], &mut pool, 0);
        // x0 = true → clause satisfied
        let assign = vec![Some(true), Some(false), Some(false)];
        assert_eq!(tile.check(&assign), ClauseStatus::Satisfied);
    }

    #[test]
    fn unit_clause() {
        let mut pool = ClausePool::new(1);
        // Clause: (x0 ∨ x1 ∨ x2)
        let tile = make_test_tile(&[Lit::pos(0), Lit::pos(1), Lit::pos(2)], &mut pool, 0);
        // x0 = false, x1 = false, x2 = unassigned → unit (propagate x2)
        let assign = vec![Some(false), Some(false), None];
        assert_eq!(
            tile.check(&assign),
            ClauseStatus::Unit {
                propagate: Lit::pos(2)
            }
        );
    }

    #[test]
    fn conflict() {
        let mut pool = ClausePool::new(1);
        // Clause: (x0 ∨ x1)
        let tile = make_test_tile(&[Lit::pos(0), Lit::pos(1)], &mut pool, 0);
        // Both false → conflict
        let assign = vec![Some(false), Some(false)];
        assert_eq!(tile.check(&assign), ClauseStatus::Conflict);
    }

    #[test]
    fn unresolved() {
        let mut pool = ClausePool::new(1);
        // Clause: (x0 ∨ x1 ∨ x2)
        let tile = make_test_tile(&[Lit::pos(0), Lit::pos(1), Lit::pos(2)], &mut pool, 0);
        // x0 = false, x1 and x2 unassigned → unresolved
        let assign = vec![Some(false), None, None];
        assert_eq!(
            tile.check(&assign),
            ClauseStatus::Unresolved {
                unassigned_count: 2
            }
        );
    }

    #[test]
    fn negated_literals() {
        let mut pool = ClausePool::new(1);
        // Clause: (¬x0 ∨ ¬x1)
        let tile = make_test_tile(&[Lit::neg(0), Lit::neg(1)], &mut pool, 0);
        // x0 = true (so ¬x0 = false), x1 = false (so ¬x1 = true) → satisfied
        let assign = vec![Some(true), Some(false)];
        assert_eq!(tile.check(&assign), ClauseStatus::Satisfied);
    }

    #[test]
    fn tile_sizes() {
        assert_eq!(tile_size_for(1), 4);
        assert_eq!(tile_size_for(3), 4);
        assert_eq!(tile_size_for(4), 4);
        assert_eq!(tile_size_for(5), 8);
        assert_eq!(tile_size_for(8), 8);
        assert_eq!(tile_size_for(9), 16);
        assert_eq!(tile_size_for(16), 16);
        assert_eq!(tile_size_for(17), 32);
        assert_eq!(tile_size_for(32), 32);
    }

    #[test]
    fn batch_packing() {
        let mut pool = ClausePool::new(10);
        // 8 clauses of 3 literals each → tile_size 4 each → 32 lanes total
        let tiles: Vec<_> = (0..8)
            .map(|i| make_test_tile(&[Lit::pos(0), Lit::pos(1), Lit::pos(2)], &mut pool, i))
            .collect();

        let batch = ClauseBatch::pack(tiles);
        assert_eq!(batch.len(), 8);
        assert_eq!(batch.total_lanes(), 32);
        // 24 real literals / 32 lanes = 75%
        assert!((batch.utilization() - 0.75).abs() < 0.01);
    }

    #[test]
    fn batch_check_all() {
        let mut pool = ClausePool::new(2);
        let t0 = make_test_tile(&[Lit::pos(0), Lit::pos(1)], &mut pool, 0);
        let t1 = make_test_tile(&[Lit::neg(0), Lit::pos(2)], &mut pool, 1);

        let batch = ClauseBatch::pack(vec![t0, t1]);
        // x0 = true, x1 = false, x2 = unassigned
        let assign = vec![Some(true), Some(false), None];
        let results = batch.check_all(&assign);

        // Results carry db_index — find by index since bin-packing may reorder
        let r0 = results.iter().find(|(idx, _)| *idx == 0).unwrap();
        let r1 = results.iter().find(|(idx, _)| *idx == 1).unwrap();
        // Clause 0: (x0 ∨ x1) — x0 = true → Satisfied
        assert_eq!(r0.1, ClauseStatus::Satisfied);
        // Clause 1: (¬x0 ∨ x2) — ¬x0 = false, x2 = unassigned → Unit(x2)
        assert_eq!(
            r1.1,
            ClauseStatus::Unit {
                propagate: Lit::pos(2)
            }
        );
    }

    #[test]
    fn duplicate_literals_deduped() {
        let mut pool = ClausePool::new(1);
        // Clause: (x0 ∨ x0 ∨ x1) → deduped to (x0 ∨ x1)
        let token = pool.acquire(0).unwrap();
        let tile =
            make_clause_tile::<Propagate>(&[Lit::pos(0), Lit::pos(0), Lit::pos(1)], token, 0)
                .unwrap();
        assert_eq!(tile.clause_len(), 2); // deduped
                                          // x0 = false, x1 = unassigned → unit (not unresolved)
        let assign = vec![Some(false), None];
        assert_eq!(
            tile.check(&assign),
            ClauseStatus::Unit {
                propagate: Lit::pos(1)
            }
        );
    }

    #[test]
    fn tautological_clause_rejected() {
        let mut pool = ClausePool::new(1);
        // Clause: (x0 ∨ ¬x0) → tautology
        let token = pool.acquire(0).unwrap();
        let result = make_clause_tile::<Propagate>(&[Lit::pos(0), Lit::neg(0)], token, 0);
        assert!(result.is_none());
    }

    #[test]
    fn bin_packing_heterogeneous() {
        let mut pool = ClausePool::new(10);
        let mut tiles = Vec::new();

        // 1 clause of 16 literals → tile_size 16
        let big_lits: Vec<_> = (0..16).map(Lit::pos).collect();
        tiles.push(make_test_tile(&big_lits, &mut pool, 0));

        // 4 clauses of 3 literals → tile_size 4 each = 16
        for i in 1..5 {
            tiles.push(make_test_tile(
                &[Lit::pos(0), Lit::pos(1), Lit::pos(2)],
                &mut pool,
                i,
            ));
        }

        // Total: 16 + 4*4 = 32 → should fit in 1 batch
        let batches = pack_clauses(tiles);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 5);
        assert_eq!(batches[0].total_lanes(), 32);
    }

    #[test]
    fn bin_packing_overflow() {
        let mut pool = ClausePool::new(10);
        let mut tiles = Vec::new();

        // 2 clauses of 17 literals each → tile_size 32 each → 2 batches
        for i in 0..2 {
            let lits: Vec<_> = (0..17).map(Lit::pos).collect();
            tiles.push(make_test_tile(&lits, &mut pool, i));
        }

        let batches = pack_clauses(tiles);
        assert_eq!(batches.len(), 2); // Can't fit two 32-tile clauses in one warp
    }

    #[test]
    fn empty_clause_returns_none() {
        let mut pool = ClausePool::new(1);
        let token = pool.acquire(0).unwrap();
        assert!(make_clause_tile::<Propagate>(&[], token, 0).is_none());
    }

    #[test]
    fn oversize_clause_returns_none() {
        let mut pool = ClausePool::new(1);
        let token = pool.acquire(0).unwrap();
        let lits: Vec<_> = (0..33).map(Lit::pos).collect();
        assert!(make_clause_tile::<Propagate>(&lits, token, 0).is_none());
    }

    #[test]
    fn eval_direct_empty_is_conflict() {
        assert_eq!(eval_clause_direct(&[], &[]), ClauseStatus::Conflict);
    }

    #[test]
    fn eval_direct_large_clause() {
        // 64-literal clause, all false except one unassigned → Unit
        let lits: Vec<_> = (0..64).map(Lit::pos).collect();
        let mut assign: Vec<Option<bool>> = vec![Some(false); 64];
        assign[42] = None; // x42 unassigned
        assert_eq!(
            eval_clause_direct(&lits, &assign),
            ClauseStatus::Unit {
                propagate: Lit::pos(42)
            }
        );
    }

    #[test]
    fn eval_direct_satisfied() {
        let lits = vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)];
        let assign = vec![Some(false), Some(true), None];
        assert_eq!(eval_clause_direct(&lits, &assign), ClauseStatus::Satisfied);
    }
}
