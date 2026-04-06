//! Assignment trail — single source of truth for variable assignments.
//!
//! The trail owns the assignment array and is the sole writer. BCP writes
//! through the trail; backtracking retracts through the trail. This prevents
//! ghost assignments (values in the assignment array with no trail entry),
//! which is the #1 CDCL implementation bug.
//!
//! Each entry records:
//! - The assigned literal (variable + polarity)
//! - The decision level
//! - The reason (decision or propagation by a specific clause)

use crate::bcp::CRef;
use crate::literal::Lit;

/// Why a variable was assigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Chosen by the decide heuristic.
    Decision,
    /// Forced by unit propagation from the clause at this CRef.
    Propagation(CRef),
}

/// A single trail entry.
#[derive(Debug, Clone, Copy)]
pub struct TrailEntry {
    pub lit: Lit,
    pub level: u32,
    pub reason: Reason,
}

/// The assignment trail — single source of truth for variable assignments.
///
/// All assignment mutations go through `new_decision`, `record_propagation`,
/// or `backtrack_to`. This prevents ghost assignments (values in the assignment
/// array with no trail entry), which is the #1 CDCL implementation bug.
pub struct Trail {
    entries: Vec<TrailEntry>,
    /// `level_starts[i]` = index in `entries` where decision level `i` begins.
    level_starts: Vec<usize>,
    current_level: u32,
    /// The assignment array. `assignments[v] = Some(true/false)` if assigned.
    /// Trail is the sole writer — all mutations go through `new_decision`,
    /// `record_propagation`, or `backtrack_to`.
    assignments: Vec<Option<bool>>,
    /// Map from variable to position in `entries` (for O(1) entry lookup).
    var_position: Vec<Option<usize>>,
    /// Number of unassigned variables. Maintained incrementally for O(1) `all_assigned`.
    num_unassigned: usize,
    /// Literal-indexed assignment array for branch-free evaluation in BCP.
    /// `lit_values[lit.code()]` = Some(true) iff the literal is satisfied,
    /// Some(false) iff falsified, None iff unassigned.
    /// Eliminates the `if is_negated { !val }` conditional in eval_lit —
    /// single array lookup with no polarity branch.
    lit_values: Vec<Option<bool>>,
}

impl Trail {
    pub fn new(num_vars: usize) -> Self {
        Trail {
            entries: Vec::new(),
            level_starts: vec![0],
            current_level: 0,
            assignments: vec![None; num_vars],
            var_position: vec![None; num_vars],
            num_unassigned: num_vars,
            lit_values: vec![None; 2 * num_vars],
        }
    }

    /// Ensure the assignment array covers at least `num_vars` variables.
    pub fn ensure_capacity(&mut self, num_vars: usize) {
        if num_vars > self.assignments.len() {
            let added = num_vars - self.assignments.len();
            self.assignments.resize(num_vars, None);
            self.var_position.resize(num_vars, None);
            self.lit_values.resize(2 * num_vars, None);
            self.num_unassigned += added;
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

    pub fn num_vars(&self) -> usize {
        self.assignments.len()
    }

    /// Read-only access to the assignment array. Used by clause evaluation.
    pub fn assignments(&self) -> &[Option<bool>] {
        &self.assignments
    }

    /// Check if a variable is assigned.
    pub fn value(&self, var: u32) -> Option<bool> {
        self.assignments.get(var as usize).copied().flatten()
    }

    /// True when every variable has an assignment. O(1) via counter.
    pub fn all_assigned(&self) -> bool {
        self.num_unassigned == 0
    }

    /// Record a new decision: increments the decision level and assigns.
    ///
    /// # Panics
    /// Debug-panics if the variable is already assigned (would create a zombie trail entry).
    pub fn new_decision(&mut self, lit: Lit) {
        debug_assert!(
            self.assignments[lit.var() as usize].is_none(),
            "new_decision on already-assigned variable {}",
            lit.var()
        );
        self.current_level += 1;
        self.level_starts.push(self.entries.len());
        self.assignments[lit.var() as usize] = Some(!lit.is_negated());
        // Literal-indexed: the decided literal is true, its complement is false
        self.lit_values[lit.code() as usize] = Some(true);
        self.lit_values[lit.complement().code() as usize] = Some(false);
        self.var_position[lit.var() as usize] = Some(self.entries.len());
        self.num_unassigned -= 1;
        self.entries.push(TrailEntry {
            lit,
            level: self.current_level,
            reason: Reason::Decision,
        });
    }

    /// Record a propagated literal at the current decision level.
    ///
    /// # Panics
    /// Debug-panics if the variable is already assigned (would create a zombie trail entry).
    pub fn record_propagation(&mut self, lit: Lit, reason_clause: CRef) {
        debug_assert!(
            self.assignments[lit.var() as usize].is_none(),
            "record_propagation on already-assigned variable {}",
            lit.var()
        );
        self.assignments[lit.var() as usize] = Some(!lit.is_negated());
        self.lit_values[lit.code() as usize] = Some(true);
        self.lit_values[lit.complement().code() as usize] = Some(false);
        self.var_position[lit.var() as usize] = Some(self.entries.len());
        self.num_unassigned -= 1;
        self.entries.push(TrailEntry {
            lit,
            level: self.current_level,
            reason: Reason::Propagation(reason_clause),
        });
    }

    /// Entries at decision levels strictly above `level`, i.e. the entries that
    /// `backtrack_to(level)` will retract. Use for pre-backtrack iteration
    /// (phase saving, heap re-insertion) without scanning the full trail.
    pub fn entries_above(&self, level: u32) -> &[TrailEntry] {
        let start = self.level_starts[level as usize + 1];
        &self.entries[start..]
    }

    /// Backtrack to the given decision level, retracting all assignments above it.
    ///
    /// # Panics
    /// Debug-panics if `target_level >= current_level` (must backtrack to a strictly lower level).
    pub fn backtrack_to(&mut self, target_level: u32) {
        debug_assert!(
            target_level < self.current_level,
            "backtrack_to({}) but current_level is {}",
            target_level,
            self.current_level
        );
        let start = self.level_starts[target_level as usize + 1];
        let retracted = self.entries.len() - start;
        for entry in &self.entries[start..] {
            self.assignments[entry.lit.var() as usize] = None;
            self.lit_values[entry.lit.code() as usize] = None;
            self.lit_values[entry.lit.complement().code() as usize] = None;
            self.var_position[entry.lit.var() as usize] = None;
        }
        self.num_unassigned += retracted;
        self.entries.truncate(start);
        self.level_starts.truncate(target_level as usize + 1);
        self.current_level = target_level;
    }

    /// Get the trail entry for a variable (O(1) via position map).
    pub fn entry_for_var(&self, var: u32) -> Option<&TrailEntry> {
        self.var_position
            .get(var as usize)
            .and_then(|&pos| pos)
            .map(|pos| &self.entries[pos])
    }

    /// Unchecked entry lookup for analysis hot path.
    ///
    /// Eliminates two bounds checks vs `entry_for_var`: one on `var_position[var]`
    /// and one on `entries[pos]`. Returns None if the variable is unassigned.
    ///
    /// # Safety
    /// `var` must be < `self.num_vars()`. This is guaranteed for all variables
    /// from the clause DB (validated at solver startup: `max_variable() < num_vars`).
    #[inline]
    pub unsafe fn entry_for_var_unchecked(&self, var: u32) -> Option<&TrailEntry> {
        match *self.var_position.get_unchecked(var as usize) {
            Some(pos) => Some(self.entries.get_unchecked(pos)),
            None => None,
        }
    }

    /// Remap clause references in propagation reasons after database compaction.
    ///
    /// `remap` is a sorted list of `(old_cref, new_cref)` pairs produced by
    /// `ClauseDb::compact()`. Uses binary search for O(log n) per trail entry.
    pub fn remap_reasons(&mut self, remap: &[(CRef, CRef)]) {
        for entry in &mut self.entries {
            if let Reason::Propagation(ref mut cref) = entry.reason {
                match remap.binary_search_by_key(cref, |&(old, _)| old) {
                    Ok(i) => *cref = remap[i].1,
                    Err(_) => panic!(
                        "live trail reason CRef {} not found in compaction remap",
                        cref
                    ),
                }
            }
        }
    }

    /// Iterate entries in assignment order.
    pub fn entries(&self) -> &[TrailEntry] {
        &self.entries
    }

    /// Extract the full assignment as a `Vec<bool>` (for SAT result output).
    ///
    /// # Panics
    /// Panics if any variable is unassigned. Call `all_assigned()` first.
    pub fn assignment_vec(&self) -> Vec<bool> {
        self.assignments
            .iter()
            .enumerate()
            .map(|(i, a)| {
                a.unwrap_or_else(|| panic!("variable {i} is unassigned in assignment_vec()"))
            })
            .collect()
    }

    /// Split the trail for BCP: yields a mutable assigns slice (stable pointer)
    /// and a writer for trail entries. The compiler can prove these don't alias,
    /// so the assigns pointer stays in a register across propagations.
    pub fn bcp_split(&mut self) -> BcpTrail<'_> {
        BcpTrail {
            assigns: &mut self.assignments,
            lit_values: &mut self.lit_values,
            entries: &mut self.entries,
            var_position: &mut self.var_position,
            current_level: self.current_level,
            num_unassigned: &mut self.num_unassigned,
        }
    }
}

/// Split view of Trail for BCP — stable assigns pointer across propagations.
///
/// `assigns` is a `&mut [Option<bool>]` (a slice, not a Vec), so the compiler
/// knows its data pointer is stable. `record_propagation` writes to `entries`
/// (a different field) without invalidating the assigns pointer.
///
/// This eliminates the pointer re-derivation that occurs when BCP calls
/// `trail.record_propagation()` (which takes `&mut Trail`, invalidating all
/// references including the assignments slice).
pub struct BcpTrail<'a> {
    /// Mutable slice into the variable-indexed assignment array.
    /// Kept in sync for non-BCP consumers (value(), assignments()).
    pub assigns: &'a mut [Option<bool>],
    /// Literal-indexed values: `lit_values[lit.code()]` gives the literal's
    /// truth value directly — no polarity branch. This is the BCP hot-path array.
    pub lit_values: &'a mut [Option<bool>],
    entries: &'a mut Vec<TrailEntry>,
    var_position: &'a mut [Option<usize>],
    current_level: u32,
    num_unassigned: &'a mut usize,
}

impl<'a> BcpTrail<'a> {
    /// Number of entries on the trail.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the trail has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get trail entry at position `idx`.
    #[inline]
    pub fn entry_at(&self, idx: usize) -> &TrailEntry {
        &self.entries[idx]
    }

    /// Record a propagated literal. Writes to `assigns` (stable pointer)
    /// and pushes to `entries` (disjoint field).
    ///
    /// Uses unchecked indexing for `assigns` and `var_position` since all
    /// variables come from clauses validated at solver startup (var < num_vars).
    #[inline]
    pub fn record_propagation(&mut self, lit: Lit, reason_clause: CRef) {
        let var = lit.var() as usize;
        debug_assert!(
            var < self.assigns.len(),
            "BcpTrail::record_propagation variable {} out of bounds (len {})",
            var,
            self.assigns.len()
        );
        debug_assert!(
            self.assigns[var].is_none(),
            "BcpTrail::record_propagation on already-assigned variable {}",
            lit.var()
        );
        // SAFETY: var < assigns.len() — all literals come from clauses in the DB,
        // and solve_cdcl_core_inner asserts db.max_variable() < num_vars at startup.
        // assigns.len() == num_vars, lit_values.len() == 2 * num_vars.
        // lit.code() = 2*var + polarity <= 2*(num_vars-1) + 1 < 2*num_vars.
        let code = lit.code() as usize;
        unsafe {
            *self.assigns.get_unchecked_mut(var) = Some(!lit.is_negated());
            *self.lit_values.get_unchecked_mut(code) = Some(true);
            *self.lit_values.get_unchecked_mut(code ^ 1) = Some(false);
            *self.var_position.get_unchecked_mut(var) = Some(self.entries.len());
        }
        *self.num_unassigned -= 1;
        self.entries.push(TrailEntry {
            lit,
            level: self.current_level,
            reason: Reason::Propagation(reason_clause),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_and_propagation() {
        let mut trail = Trail::new(4);

        trail.record_propagation(Lit::pos(0), 0);
        assert_eq!(trail.current_level(), 0);
        assert_eq!(trail.value(0), Some(true));

        trail.new_decision(Lit::pos(1));
        assert_eq!(trail.current_level(), 1);
        trail.record_propagation(Lit::pos(2), 1);
        assert_eq!(trail.value(2), Some(true));

        trail.new_decision(Lit::neg(3));
        assert_eq!(trail.current_level(), 2);
        assert_eq!(trail.value(3), Some(false));

        assert_eq!(trail.len(), 4);
    }

    #[test]
    fn backtrack() {
        let mut trail = Trail::new(4);

        trail.record_propagation(Lit::pos(0), 0);
        trail.new_decision(Lit::pos(1));
        trail.record_propagation(Lit::pos(2), 1);
        trail.new_decision(Lit::neg(3));

        trail.backtrack_to(1);
        assert_eq!(trail.current_level(), 1);
        assert_eq!(trail.value(3), None);
        assert_eq!(trail.value(2), Some(true));
        assert_eq!(trail.value(0), Some(true));
        assert_eq!(trail.len(), 3);

        trail.backtrack_to(0);
        assert_eq!(trail.current_level(), 0);
        assert_eq!(trail.value(1), None);
        assert_eq!(trail.value(2), None);
        assert_eq!(trail.value(0), Some(true));
        assert_eq!(trail.len(), 1);
    }

    #[test]
    fn entry_lookup() {
        let mut trail = Trail::new(3);

        trail.new_decision(Lit::pos(0));
        trail.record_propagation(Lit::neg(1), 0);

        let e = trail.entry_for_var(1).unwrap();
        assert!(e.lit.is_negated());
        assert_eq!(e.level, 1);
        assert_eq!(e.reason, Reason::Propagation(0));

        assert!(trail.entry_for_var(2).is_none());
    }
}
