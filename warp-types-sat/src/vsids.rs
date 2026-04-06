//! VSIDS (Variable State Independent Decaying Sum) branching heuristic.
//!
//! Variables appearing in recent conflicts get higher activity scores and are
//! decided first. Combined with phase saving (remember last polarity) to avoid
//! re-exploring refuted subtrees.
//!
//! MiniSat optimization: instead of decaying all activities by 0.95 each conflict
//! (O(n)), increase the bump increment by 1/0.95 (O(1)). Mathematically equivalent.
//!
//! Variable selection uses a binary max-heap ordered by activity (O(log n) per
//! decision vs O(n) linear scan). Variables are removed from the heap when
//! popped by `pick()` and re-inserted via `notify_unassigned()` during backtrack.

use crate::bcp::ClauseDb;

/// Sentinel: variable is not in the heap.
const NOT_IN_HEAP: u32 = u32::MAX;

/// VSIDS branching heuristic with phase saving and priority heap.
pub struct Vsids {
    /// Per-variable activity score.
    activity: Vec<f64>,
    /// Current bump increment (grows to implement implicit decay).
    increment: f64,
    /// Decay factor. Standard value: 0.95 (MiniSat default).
    decay_factor: f64,
    /// Saved phase (polarity) for each variable.
    phase: Vec<bool>,
    /// Binary max-heap of variable indices, ordered by activity.
    heap: Vec<u32>,
    /// Position of each variable in `heap`. `NOT_IN_HEAP` if absent.
    heap_pos: Vec<u32>,
}

impl Vsids {
    /// Create with all activities at zero, all phases defaulting positive.
    /// All variables start in the heap, ordered by index (higher index = higher
    /// priority on ties, matching MiniSat convention).
    pub fn new(num_vars: u32) -> Self {
        let n = num_vars as usize;
        // Reverse order so higher-indexed vars are near the root (tie-breaking).
        let heap: Vec<u32> = (0..num_vars).rev().collect();
        let mut heap_pos = vec![0u32; n];
        for (i, &var) in heap.iter().enumerate() {
            heap_pos[var as usize] = i as u32;
        }
        Vsids {
            activity: vec![0.0; n],
            increment: 1.0,
            decay_factor: 0.95,
            phase: vec![true; n],
            heap,
            heap_pos,
        }
    }

    // ── Heap internals ──────────────────────────────────────────────

    /// Compare two variables by activity, breaking ties by index (higher wins).
    /// Tie-breaking by index ensures deterministic decision order and matches
    /// the old linear-scan behavior where `max_by` returned the last equal element.
    #[inline]
    fn activity_gt(&self, a: u32, b: u32) -> bool {
        let aa = self.activity[a as usize];
        let ab = self.activity[b as usize];
        aa > ab || (aa == ab && a > b)
    }

    #[inline]
    fn heap_swap(&mut self, i: usize, j: usize) {
        let vi = self.heap[i];
        let vj = self.heap[j];
        self.heap.swap(i, j);
        self.heap_pos[vi as usize] = j as u32;
        self.heap_pos[vj as usize] = i as u32;
    }

    fn sift_up(&mut self, mut pos: usize) {
        let var = self.heap[pos];
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if !self.activity_gt(var, self.heap[parent]) {
                break;
            }
            self.heap_swap(pos, parent);
            pos = parent;
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        let len = self.heap.len();
        loop {
            let left = 2 * pos + 1;
            if left >= len {
                break;
            }
            let right = left + 1;
            let child = if right < len && self.activity_gt(self.heap[right], self.heap[left]) {
                right
            } else {
                left
            };
            if !self.activity_gt(self.heap[child], self.heap[pos]) {
                break;
            }
            self.heap_swap(pos, child);
            pos = child;
        }
    }

    /// Build heap from scratch in O(n) using bottom-up heapify.
    fn rebuild_heap(&mut self) {
        for (i, &var) in self.heap.iter().enumerate() {
            self.heap_pos[var as usize] = i as u32;
        }
        let n = self.heap.len();
        if n > 1 {
            for i in (0..n / 2).rev() {
                self.sift_down(i);
            }
        }
    }

    fn heap_insert(&mut self, var: u32) {
        if self.heap_pos[var as usize] != NOT_IN_HEAP {
            return; // already in heap
        }
        let pos = self.heap.len();
        self.heap.push(var);
        self.heap_pos[var as usize] = pos as u32;
        self.sift_up(pos);
    }

    fn heap_pop(&mut self) -> Option<u32> {
        if self.heap.is_empty() {
            return None;
        }
        let var = self.heap[0];
        let last = self.heap.len() - 1;
        if last > 0 {
            self.heap_swap(0, last);
        }
        self.heap_pos[var as usize] = NOT_IN_HEAP;
        self.heap.pop();
        if !self.heap.is_empty() {
            self.sift_down(0);
        }
        Some(var)
    }

    // ── Public API ──────────────────────────────────────────────────

    /// Warm-start activities from clause occurrence counts.
    /// Variables appearing more often are more constrained — decide them first.
    /// Rebuilds the heap after updating all activities.
    pub fn initialize_from_clauses(&mut self, db: &ClauseDb) {
        for cref in db.iter_crefs() {
            for &lit in db.clause(cref).literals {
                self.activity[lit.var() as usize] += 1.0;
            }
        }
        self.rebuild_heap();
    }

    /// Bump activity of a variable (call for each var in the learned clause).
    pub fn bump(&mut self, var: u32) {
        self.activity[var as usize] += self.increment;
        // Rescale to prevent floating-point overflow.
        if self.activity[var as usize] > 1e100 {
            for a in &mut self.activity {
                *a *= 1e-100;
            }
            self.increment *= 1e-100;
            // Uniform rescale preserves relative order — heap invariant intact.
        }
        // Sift up if in heap (activity only increases).
        let pos = self.heap_pos[var as usize];
        if pos != NOT_IN_HEAP {
            self.sift_up(pos as usize);
        }
    }

    /// Apply decay after each conflict (O(1) via increment scaling).
    pub fn decay(&mut self) {
        self.increment /= self.decay_factor;
    }

    /// Pick the highest-activity unassigned variable and its saved phase.
    ///
    /// Pops from the max-heap, skipping assigned variables (they are re-inserted
    /// on backtrack via `notify_unassigned`). O(log n) amortized per decision.
    pub fn pick(&mut self, assignments: &[Option<bool>]) -> (u32, bool) {
        while let Some(var) = self.heap_pop() {
            if assignments[var as usize].is_none() {
                return (var, self.phase[var as usize]);
            }
        }
        panic!("pick called with all variables assigned");
    }

    /// Re-insert a variable into the heap after it becomes unassigned.
    /// Call during backtrack for each retracted variable.
    pub fn notify_unassigned(&mut self, var: u32) {
        self.heap_insert(var);
    }

    /// Read-only access to the per-variable activity scores.
    pub fn activities(&self) -> &[f64] {
        &self.activity
    }

    /// Save the phase (polarity) of a variable.
    pub fn save_phase(&mut self, var: u32, polarity: bool) {
        self.phase[var as usize] = polarity;
    }

    /// Set initial activity for a variable (e.g., from gradient confidence).
    /// Call `initialize_from_clauses` afterward to rebuild the heap.
    pub fn set_initial_activity(&mut self, var: u32, activity: f64) {
        self.activity[var as usize] = activity;
    }

    /// Set phase hint for a variable (e.g., from gradient polarity).
    pub fn set_phase(&mut self, var: u32, polarity: bool) {
        self.phase[var as usize] = polarity;
    }

    /// Apply trail-gradient signal to unassigned variables (seed-1a).
    ///
    /// For each unassigned variable: set phase to gradient-suggested polarity
    /// and give a small activity bump proportional to gradient magnitude.
    /// Assigned variables are left untouched.
    ///
    /// `boost_scale` controls how much the gradient magnitude affects activity.
    /// 0.0 = phase-only, no activity change. Higher = more aggressive reordering.
    pub fn apply_trail_gradient(
        &mut self,
        magnitudes: &[f64],
        polarities: &[bool],
        assignments: &[Option<bool>],
        boost_scale: f64,
    ) {
        let n = self.activity.len();
        debug_assert_eq!(magnitudes.len(), n);
        debug_assert_eq!(polarities.len(), n);
        debug_assert_eq!(assignments.len(), n);

        let mut needs_rebuild = false;
        for v in 0..n {
            if assignments[v].is_some() {
                continue; // skip assigned vars
            }
            // Phase hint from gradient direction
            self.phase[v] = polarities[v];

            // Activity boost proportional to gradient magnitude
            if boost_scale > 0.0 {
                let boost = magnitudes[v] * boost_scale;
                if boost > 0.0 {
                    self.activity[v] += boost;
                    needs_rebuild = true;
                }
            }
        }
        if needs_rebuild {
            self.rebuild_heap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Lit;

    #[test]
    fn picks_highest_activity() {
        let mut vsids = Vsids::new(3);
        vsids.bump(2);
        vsids.bump(2);
        vsids.bump(0);

        let assignments = vec![None, None, None];
        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 2);
    }

    #[test]
    fn skips_assigned() {
        let mut vsids = Vsids::new(3);
        vsids.bump(0);
        vsids.bump(0);
        vsids.bump(1);

        let assignments = vec![Some(true), None, None];
        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 1);
    }

    #[test]
    fn phase_saving() {
        let mut vsids = Vsids::new(3);
        vsids.save_phase(1, false);

        let assignments = vec![Some(true), None, Some(true)];
        let (var, pol) = vsids.pick(&assignments);
        assert_eq!(var, 1);
        assert!(!pol);
    }

    #[test]
    fn decay_increases_increment() {
        let mut vsids = Vsids::new(1);
        let before = vsids.increment;
        vsids.decay();
        assert!(vsids.increment > before);
    }

    #[test]
    fn overflow_protection() {
        let mut vsids = Vsids::new(2);
        for _ in 0..10000 {
            vsids.bump(0);
            vsids.decay();
        }
        assert!(vsids.activity[0].is_finite());
        assert!(vsids.activity[0] > 0.0);
    }

    #[test]
    fn clause_initialization() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::neg(1)]);
        db.add_clause(vec![Lit::pos(0), Lit::pos(2)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);

        let mut vsids = Vsids::new(3);
        vsids.initialize_from_clauses(&db);

        // var 0 appears 2x, var 1 appears 2x, var 2 appears 2x
        let assignments = vec![None, None, None];
        let (_, _) = vsids.pick(&assignments); // should not panic
    }

    #[test]
    fn notify_unassigned_reinserts() {
        let mut vsids = Vsids::new(3);
        vsids.bump(2);
        vsids.bump(2);

        let assignments = vec![None, None, None];
        // Pick var 2 (highest activity) — removes from heap
        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 2);

        // Re-insert on "backtrack"
        vsids.notify_unassigned(2);

        // Should pick var 2 again
        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 2);
    }

    #[test]
    fn heap_order_after_multiple_bumps() {
        let mut vsids = Vsids::new(5);
        vsids.bump(3); // 3→1
        vsids.bump(1); // 1→1
        vsids.bump(1); // 1→2
        vsids.bump(4); // 4→1
        vsids.bump(4); // 4→2
        vsids.bump(4); // 4→3
        // activities: [0.0, 2.0, 0.0, 1.0, 3.0]

        let assignments = vec![None, None, None, None, None];

        let (v1, _) = vsids.pick(&assignments);
        assert_eq!(v1, 4); // activity 3.0 (highest)

        let (v2, _) = vsids.pick(&assignments);
        assert_eq!(v2, 1); // activity 2.0

        let (v3, _) = vsids.pick(&assignments);
        assert_eq!(v3, 3); // activity 1.0
    }

    #[test]
    fn bump_sifts_up_in_heap() {
        let mut vsids = Vsids::new(3);
        // Initially all zero activity. Bump var 2.
        vsids.bump(2);

        let assignments = vec![None, None, None];
        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 2); // var 2 should be at top after bump
    }

    #[test]
    fn double_insert_is_idempotent() {
        let mut vsids = Vsids::new(2);
        vsids.bump(0);

        let assignments = vec![None, None];
        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 0);

        vsids.notify_unassigned(0);
        vsids.notify_unassigned(0); // duplicate — should be no-op

        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 0);
    }
}
