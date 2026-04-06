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

use std::collections::{HashMap, HashSet};

use crate::clause::ClausePool;
use crate::clause_tile::{self, ClauseStatus, ClauseTile};
use crate::literal::Lit;
use crate::phase::Propagate;
use crate::trail::Trail;

// ============================================================================
// Clause reference (CRef) — arena offset
// ============================================================================

/// A clause reference: word offset into the clause arena.
///
/// Points to the inline header; literals follow at `cref + 1`.
/// Stored in `WatchEntry`, `Reason::Propagation`, and `BcpResult::Conflict`.
pub type CRef = u32;

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
    /// A conflict was found. Contains the conflicting clause's arena reference.
    Conflict { clause: CRef },
}

// ============================================================================
// Inline clause header
// ============================================================================
//
// Each clause occupies `1 + len` words in the arena:
//   [header][lit0][lit1]...[litN-1]
//
// Header layout (u32):
//   bit  31:      deleted flag
//   bits 30-20:   LBD (11 bits, max 2047)
//   bits 19-0:    length (20 bits, max 1_048_575 literals)

const HEADER_DELETED_BIT: u32 = 1 << 31;
const HEADER_LBD_SHIFT: u32 = 20;
const HEADER_LBD_MASK: u32 = 0x7FF;
const HEADER_LEN_MASK: u32 = 0xF_FFFF;

#[inline]
fn make_header(len: u32, lbd: u16, deleted: bool) -> u32 {
    let d = if deleted { HEADER_DELETED_BIT } else { 0 };
    d | ((lbd as u32 & HEADER_LBD_MASK) << HEADER_LBD_SHIFT) | (len & HEADER_LEN_MASK)
}

#[inline]
fn header_len(h: u32) -> u32 {
    h & HEADER_LEN_MASK
}

#[inline]
fn header_lbd(h: u32) -> u16 {
    ((h >> HEADER_LBD_SHIFT) & HEADER_LBD_MASK) as u16
}

#[inline]
fn header_is_deleted(h: u32) -> bool {
    h & HEADER_DELETED_BIT != 0
}

// ============================================================================
// Clause database — inline-header arena layout
// ============================================================================

/// A view into a clause stored in the arena.
///
/// Returned by `ClauseDb::clause()`. The `literals` field is a slice into the
/// contiguous arena — header + literals are adjacent cache-line neighbors.
pub struct ClauseRef<'a> {
    /// The literals in this clause (slice into the arena).
    pub literals: &'a [Lit],
}

/// Clause database storing original and learned clauses in a flat arena
/// with inline headers.
///
/// Layout: each clause occupies `1 + len` words in the arena:
///   `[header][lit0][lit1]...[litN-1]`
/// where `header` packs length, LBD, and deleted flag into a single u32.
///
/// A `CRef` is the arena index of the header word. Clause access reads the
/// header (1 word) then slices the adjacent literals — single contiguous
/// cache-line read. This eliminates the separate `offsets[]`, `lengths[]`,
/// `lbd[]`, and `deleted[]` arrays of the previous design.
///
/// Clauses are referenced by CRef (arena offset), not by sequential index.
/// The solver appends learned clauses during conflict analysis.
/// After `freeze_original()`, clauses before `original_limit` are
/// protected from deletion; all subsequent clauses are "learned" and
/// eligible for LBD-based garbage collection.
pub struct ClauseDb {
    /// Flat arena: inline headers + literal codes stored contiguously.
    arena: Vec<u32>,
    /// Number of clauses (including deleted/tombstoned).
    num_clauses: usize,
    /// Arena offset past the last original clause. Learned clauses start here.
    original_limit: u32,
    /// Number of original (input) clauses.
    num_original: usize,
    /// Highest variable index seen (tracked incrementally for O(1) access).
    max_var: u32,
    /// Optional per-clause resolution depth (for depth-weighted deletion).
    /// Only populated during instrumented solving.
    depth: HashMap<CRef, u16>,
}

impl ClauseDb {
    pub fn new() -> Self {
        ClauseDb {
            arena: Vec::new(),
            num_clauses: 0,
            original_limit: 0,
            num_original: 0,
            max_var: 0,
            depth: HashMap::new(),
        }
    }

    /// Add a clause, returns its CRef (arena offset of the header word).
    pub fn add_clause(&mut self, literals: Vec<Lit>) -> CRef {
        let cref = self.arena.len() as u32;
        let len = literals.len() as u32;
        self.arena.push(make_header(len, 0, false));
        for &lit in &literals {
            self.arena.push(lit.code());
            let v = lit.var();
            if v > self.max_var {
                self.max_var = v;
            }
        }
        self.num_clauses += 1;
        cref
    }

    /// Mark the current clause count as "original". Clauses added after this
    /// point are "learned" and eligible for LBD-based deletion.
    pub fn freeze_original(&mut self) {
        self.original_limit = self.arena.len() as u32;
        self.num_original = self.num_clauses;
    }

    /// Set the LBD score for a clause.
    pub fn set_lbd(&mut self, cref: CRef, lbd: u16) {
        let h = self.arena[cref as usize];
        let len = header_len(h);
        let deleted = header_is_deleted(h);
        self.arena[cref as usize] = make_header(len, lbd, deleted);
    }

    /// Number of original (non-learned) clauses.
    pub fn num_original(&self) -> usize {
        self.num_original
    }

    /// Check if a clause has been deleted (tombstoned).
    pub fn is_deleted(&self, cref: CRef) -> bool {
        header_is_deleted(self.arena[cref as usize])
    }

    /// Unchecked deleted check for the BCP hot loop.
    ///
    /// # Safety
    /// `cref` must point to a valid header in the arena.
    #[inline]
    pub unsafe fn is_deleted_unchecked(&self, cref: CRef) -> bool {
        header_is_deleted(*self.arena.get_unchecked(cref as usize))
    }

    /// Delete the worst learned clauses by LBD score.
    ///
    /// Keeps clauses that are:
    /// - Original (CRef < original_limit)
    /// - Locked (currently a propagation reason on the trail)
    /// - "Glue" clauses: LBD ≤ 2 (binary learned clauses are always useful)
    ///
    /// Strategy: sort by LBD, keep the best half. Standard MiniSat/Glucose.
    ///
    /// Returns CRefs of deleted clauses (for watch list cleanup).
    pub fn reduce_learned(&mut self, locked: &HashSet<CRef>) -> Vec<CRef> {
        let mut candidates: Vec<(CRef, u16)> = Vec::new();
        let mut pos = self.original_limit;
        while (pos as usize) < self.arena.len() {
            let h = self.arena[pos as usize];
            let len = header_len(h);
            if !header_is_deleted(h) && !locked.contains(&pos) && header_lbd(h) > 2 {
                candidates.push((pos, header_lbd(h)));
            }
            pos += 1 + len;
        }
        candidates.sort_by_key(|&(_, lbd)| lbd);
        let keep = candidates.len() / 2;
        let to_delete: Vec<CRef> = candidates[keep..].iter().map(|&(cref, _)| cref).collect();
        for &cref in &to_delete {
            let h = self.arena[cref as usize];
            let len = header_len(h);
            let lbd = header_lbd(h);
            self.arena[cref as usize] = make_header(len, lbd, true);
        }
        to_delete
    }

    /// Store the resolution depth for a learned clause.
    pub fn set_depth(&mut self, cref: CRef, depth: u16) {
        self.depth.insert(cref, depth);
    }

    /// Retrieve the resolution depth for a clause (0 if not set).
    pub fn get_depth(&self, cref: CRef) -> u16 {
        self.depth.get(&cref).copied().unwrap_or(0)
    }

    /// Depth-weighted learned clause deletion.
    ///
    /// Scores each candidate clause by `LBD + depth_weight * resolution_depth`
    /// (both as f64), then deletes the top half by score. `depth_weight = 0.0`
    /// is equivalent to `reduce_learned`.
    pub fn reduce_learned_weighted(
        &mut self,
        locked: &HashSet<CRef>,
        depth_weight: f64,
    ) -> Vec<CRef> {
        let mut candidates: Vec<(CRef, f64)> = Vec::new();
        let mut pos = self.original_limit;
        while (pos as usize) < self.arena.len() {
            let h = self.arena[pos as usize];
            let len = header_len(h);
            if !header_is_deleted(h) && !locked.contains(&pos) && header_lbd(h) > 2 {
                let lbd = header_lbd(h) as f64;
                let d = self.depth.get(&pos).copied().unwrap_or(0) as f64;
                let score = lbd + depth_weight * d;
                candidates.push((pos, score));
            }
            pos += 1 + len;
        }
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let keep = candidates.len() / 2;
        let to_delete: Vec<CRef> = candidates[keep..].iter().map(|&(cref, _)| cref).collect();
        for &cref in &to_delete {
            let h = self.arena[cref as usize];
            let len = header_len(h);
            let lbd = header_lbd(h);
            self.arena[cref as usize] = make_header(len, lbd, true);
            self.depth.remove(&cref);
        }
        to_delete
    }

    /// Compact the database: remove tombstoned clauses and rebuild the arena.
    ///
    /// Returns a remap table of `(old_cref, new_cref)` pairs for live clauses,
    /// sorted by old_cref. Callers must update all stored clause references
    /// (trail reasons, watch lists) using this table.
    pub fn compact(&mut self) -> Vec<(CRef, CRef)> {
        let mut new_arena = Vec::with_capacity(self.arena.len());
        let mut remap = Vec::new();
        let mut new_original_limit = 0u32;
        let mut new_num_clauses = 0usize;
        let mut past_original = false;

        let mut pos = 0u32;
        while (pos as usize) < self.arena.len() {
            if !past_original && pos >= self.original_limit {
                new_original_limit = new_arena.len() as u32;
                past_original = true;
            }
            let h = self.arena[pos as usize];
            let len = header_len(h) as usize;
            if !header_is_deleted(h) {
                let new_cref = new_arena.len() as u32;
                remap.push((pos, new_cref));
                // Push header with deleted flag cleared
                new_arena.push(make_header(len as u32, header_lbd(h), false));
                new_arena.extend_from_slice(&self.arena[pos as usize + 1..pos as usize + 1 + len]);
                new_num_clauses += 1;
            }
            pos += 1 + len as u32;
        }
        if !past_original {
            new_original_limit = new_arena.len() as u32;
        }

        // Remap depth table
        let mut new_depth = HashMap::new();
        for &(old, new_cref) in &remap {
            if let Some(&d) = self.depth.get(&old) {
                new_depth.insert(new_cref, d);
            }
        }

        self.arena = new_arena;
        self.num_clauses = new_num_clauses;
        self.original_limit = new_original_limit;
        self.depth = new_depth;
        // num_original unchanged — original clauses are never deleted
        remap
    }

    /// Number of clauses (original + learned, including tombstoned).
    pub fn len(&self) -> usize {
        self.num_clauses
    }

    /// Whether the database contains no clauses.
    pub fn is_empty(&self) -> bool {
        self.num_clauses == 0
    }

    /// Get a clause by CRef.
    ///
    /// Reads the inline header for length, then returns a slice into the
    /// adjacent literal words. Single contiguous cache-line access.
    pub fn clause(&self, cref: CRef) -> ClauseRef<'_> {
        let pos = cref as usize;
        let len = header_len(self.arena[pos]) as usize;
        // SAFETY: Lit is #[repr(transparent)] over u32.
        let literals = unsafe {
            std::slice::from_raw_parts(self.arena.as_ptr().add(pos + 1) as *const Lit, len)
        };
        ClauseRef { literals }
    }

    /// Unchecked clause access for the BCP hot loop.
    ///
    /// # Safety
    /// `cref` must point to a valid header in the arena.
    #[inline]
    pub unsafe fn clause_unchecked(&self, cref: CRef) -> ClauseRef<'_> {
        let pos = cref as usize;
        let len = header_len(*self.arena.get_unchecked(pos)) as usize;
        let raw_ptr = self.arena.as_ptr().add(pos + 1) as *const Lit;
        ClauseRef {
            literals: std::slice::from_raw_parts(raw_ptr, len),
        }
    }

    /// Swap two literal positions within a clause in-place.
    ///
    /// Used to maintain the MiniSat convention: `c[0]` and `c[1]` are always
    /// the watched literals. When a replacement watch is found at position `k`,
    /// swap it into the watched position.
    ///
    /// # Safety
    /// `cref` must be valid. `a` and `b` must be < clause length.
    #[inline]
    pub unsafe fn swap_literal_unchecked(&mut self, cref: CRef, a: usize, b: usize) {
        let pos = cref as usize;
        let ptr = self.arena.as_mut_ptr();
        std::ptr::swap(ptr.add(pos + 1 + a), ptr.add(pos + 1 + b));
    }

    /// Highest variable index across all clauses. Returns 0 if empty.
    pub fn max_variable(&self) -> u32 {
        self.max_var
    }

    /// Iterate all clause references in insertion order.
    ///
    /// Walks the arena linearly — O(n) in total. CRef iteration replaces
    /// the old `for i in 0..db.len()` pattern.
    pub fn iter_crefs(&self) -> CRefIter<'_> {
        CRefIter {
            arena: &self.arena,
            pos: 0,
        }
    }

    /// Collect all clause references into a Vec.
    ///
    /// Convenience method for code that needs random access by sequential
    /// index (gradient solver, SoA construction). Not needed on the BCP
    /// hot path, which accesses clauses by CRef from WatchEntry.
    pub fn crefs(&self) -> Vec<CRef> {
        self.iter_crefs().collect()
    }
}

impl Default for ClauseDb {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over clause references in the arena.
pub struct CRefIter<'a> {
    arena: &'a [u32],
    pos: u32,
}

impl<'a> Iterator for CRefIter<'a> {
    type Item = CRef;

    fn next(&mut self) -> Option<CRef> {
        if (self.pos as usize) >= self.arena.len() {
            return None;
        }
        let cref = self.pos;
        let len = header_len(self.arena[self.pos as usize]);
        self.pos += 1 + len;
        Some(cref)
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
    let crefs: Vec<CRef> = db.crefs();

    loop {
        let mut found_unit = false;

        // Evaluate non-tileable clauses directly (empty or >32 literals).
        for &cref in &crefs {
            let clause = db.clause(cref);
            if clause.literals.is_empty() || clause.literals.len() > 32 {
                match clause_tile::eval_clause_direct(clause.literals, trail.assignments()) {
                    ClauseStatus::Conflict => {
                        return BcpResult::Conflict { clause: cref };
                    }
                    ClauseStatus::Unit { propagate } => {
                        if trail.value(propagate.var()).is_none() {
                            trail.record_propagation(propagate, cref);
                            found_unit = true;
                        }
                    }
                    ClauseStatus::Satisfied | ClauseStatus::Unresolved { .. } => {}
                }
            }
        }

        // Tile-based evaluation for clauses that fit (1-32 literals).
        let mut pool = ClausePool::new(crefs.len());
        let mut tiles: Vec<ClauseTile<Propagate>> = Vec::new();

        for (seq_i, &cref) in crefs.iter().enumerate() {
            let clause = db.clause(cref);
            if clause.literals.is_empty() || clause.literals.len() > 32 {
                continue; // handled in direct evaluation above
            }
            let token = pool.acquire(seq_i).unwrap();
            if let Some(tile) =
                clause_tile::make_clause_tile::<Propagate>(clause.literals, token, cref)
            {
                tiles.push(tile);
            }
        }

        let batches = clause_tile::pack_clauses(tiles);

        for batch in &batches {
            let results = batch.check_all(trail.assignments());
            for &(db_ref, ref status) in &results {
                match status {
                    ClauseStatus::Conflict => {
                        return BcpResult::Conflict { clause: db_ref };
                    }
                    ClauseStatus::Unit { propagate } => {
                        if trail.value(propagate.var()).is_none() {
                            trail.record_propagation(*propagate, db_ref);
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

        assert_eq!(result, BcpResult::Conflict { clause: 0 });
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
                    i as CRef,
                )
            })
            .collect();

        let batches = clause_tile::pack_clauses(tiles);
        assert_eq!(batches.len(), 1);
        assert!((batches[0].utilization() - 0.5).abs() < 0.01);
    }
}
