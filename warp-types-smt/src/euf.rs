//! Congruence closure engine for QF_EUF.
//!
//! Implements [`TheorySolver`](warp_types_sat::theory::TheorySolver) from
//! `warp_types_sat` to plug equality reasoning into the CDCL loop via DPLL(T).
//! The core algorithm is congruence closure over a backtrackable union-find:
//!
//! - **Union-find**: tracks equivalence classes of terms, with path halving
//!   and union by rank. Backtrackable via a trail-based undo stack.
//! - **Congruence table**: hash-based signature table mapping
//!   `(func, [repr(arg₁), ..., repr(argₙ)])` → `TermId`. When classes merge,
//!   function applications may become congruent.
//! - **Explanation**: records merge reasons (asserted equality or congruence)
//!   for lazy explanation during conflict analysis.
//!
//! # DPLL(T) contract
//!
//! - `check()`: called after each BCP fixpoint. Scans the trail for new
//!   equality/disequality assertions, updates the union-find, checks for
//!   conflicts (disequality between terms in the same equivalence class).
//! - `backtrack(level)`: undoes merges back to the given decision level.
//! - `explain(lit, key)`: lazily produces the reason clause for a theory
//!   propagation by tracing the congruence proof.

use std::collections::HashMap;

use crate::formula::{AtomId, AtomMap};
use crate::term::{FuncId, TermArena, TermId, TermKind};

use warp_types_sat::literal::Lit;
use warp_types_sat::theory::{TheoryContext, TheoryProp, TheoryResult, TheorySolver};

// ============================================================================
// Merge reasons (for explanation generation)
// ============================================================================

/// Why two terms were merged into the same equivalence class.
#[derive(Debug, Clone)]
enum MergeReason {
    /// An equality atom was asserted true on the SAT trail.
    Asserted(AtomId),
    /// Congruence: `f(a₁..aₙ)` and `f(b₁..bₙ)` merged because `aᵢ ~ bᵢ`.
    Congruence(TermId, TermId),
}

/// Record of a single union operation, for undo on backtrack.
#[derive(Debug)]
struct UndoEntry {
    /// The term whose parent was changed (the one that got reparented).
    child: TermId,
    /// Its old parent before the merge.
    old_parent: TermId,
    /// Old rank of the new root (may have been incremented).
    root: TermId,
    old_root_rank: u32,
}

/// Canonical signature for the congruence table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Signature {
    func: FuncId,
    arg_reprs: Vec<TermId>,
}

/// Data for lazy explanation of a theory propagation.
#[derive(Debug, Clone)]
struct PropRecord {
    /// The SAT literal that was propagated.
    lit: Lit,
    /// The equality atom: the two terms whose congruence was detected.
    t1: TermId,
    t2: TermId,
}

// ============================================================================
// EUF Solver
// ============================================================================

/// Congruence closure engine implementing the EUF theory.
///
/// Constructed by the solver after Boolean abstraction, then passed to
/// `warp_types_sat::solve_with_theory()` as the theory oracle.
pub struct EufSolver {
    // ── Atom map (theory ↔ SAT bridge) ──
    pub(crate) atom_map: AtomMap,

    // ── Union-find ──
    /// `parent[i]`: parent of term `i` in the UF forest.
    parent: Vec<TermId>,
    /// `rank[i]`: rank for union-by-rank.
    rank: Vec<u32>,

    // ── Congruence table ──
    /// Signature → representative term. When classes merge, re-signature
    /// function applications and check for new congruences.
    sig_table: HashMap<Signature, TermId>,
    /// `use_list[term_id]`: function application TermIds that have `term_id`
    /// as one of their arguments. Used to find terms that need re-signaturing
    /// when a class changes representative.
    use_list: Vec<Vec<TermId>>,
    /// All function application TermIds (for initial signature population).
    func_apps: Vec<TermId>,

    // ── Backtracking ──
    /// Undo stack: each entry records one union for reversal.
    undo_stack: Vec<UndoEntry>,
    /// `level_marks[level]`: undo_stack.len() at the start of each decision level.
    level_marks: Vec<usize>,

    // ── Merge reasons (for explain) ──
    /// Each merge records why it happened, parallel to undo_stack.
    merge_reasons: Vec<(TermId, TermId, MergeReason)>,
    /// level_marks for merge_reasons (parallel to level_marks).
    merge_level_marks: Vec<usize>,

    // ── Disequalities ──
    /// Active disequality assertions: (t1, t2, atom_id) where `(= t1 t2)` is false.
    disequalities: Vec<(TermId, TermId, AtomId)>,
    /// level_marks for disequalities.
    diseq_level_marks: Vec<usize>,

    // ── Incremental trail scanning ──
    /// How many trail entries we've already processed.
    trail_pos: usize,

    // ── Theory propagation records (for lazy explain) ──
    prop_records: Vec<PropRecord>,

    // ── Arena snapshot (read-only term structure) ──
    /// Cached term kinds for signature computation. Indexed by TermId.
    term_kinds: Vec<TermKind>,
}

impl EufSolver {
    /// Create a new EUF solver for the given term arena and atom map.
    pub fn new(arena: &TermArena, atom_map: AtomMap) -> Self {
        let n = arena.len();

        // Initialize union-find: each term is its own representative
        let parent: Vec<TermId> = (0..n as u32).map(TermId).collect();
        let rank = vec![0u32; n];

        // Build use-lists and func_apps from the arena
        let mut use_list: Vec<Vec<TermId>> = vec![Vec::new(); n];
        let mut func_apps = Vec::new();
        let mut term_kinds = Vec::with_capacity(n);

        for i in 0..n {
            let entry = arena.get(TermId(i as u32));
            term_kinds.push(entry.kind.clone());
            if let TermKind::Apply { args, .. } = &entry.kind {
                let tid = TermId(i as u32);
                func_apps.push(tid);
                for &arg in args {
                    use_list[arg.index()].push(tid);
                }
            }
        }

        // Populate initial signature table
        let mut sig_table = HashMap::new();
        for &tid in &func_apps {
            if let TermKind::Apply { func, ref args } = term_kinds[tid.index()] {
                let sig = Signature {
                    func,
                    arg_reprs: args.clone(), // initially, repr == self
                };
                sig_table.insert(sig, tid);
            }
        }

        EufSolver {
            atom_map,
            parent,
            rank,
            sig_table,
            use_list,
            func_apps,
            undo_stack: Vec::new(),
            level_marks: vec![0], // level 0 starts at undo position 0
            merge_reasons: Vec::new(),
            merge_level_marks: vec![0],
            disequalities: Vec::new(),
            diseq_level_marks: vec![0],
            trail_pos: 0,
            prop_records: Vec::new(),
            term_kinds,
        }
    }

    // ── Union-Find ──

    /// Find the representative of `x` (non-compressing path walk).
    ///
    /// Non-compressing find keeps `parent[]` stable for undo-based
    /// backtracking. Public within the crate so that [`CombiningSolver`]
    /// can check which terms are already in the same equivalence class
    /// before propagating module equalities.
    pub(crate) fn find(&self, mut x: TermId) -> TermId {
        while self.parent[x.index()] != x {
            x = self.parent[x.index()];
        }
        x
    }

    /// Merge the equivalence classes of `a` and `b`.
    /// Returns `true` if a merge actually happened (they were in different classes).
    fn merge(&mut self, a: TermId, b: TermId, reason: MergeReason) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false; // already same class
        }

        // Union by rank: attach smaller tree under larger
        let (root, child) = if self.rank[ra.index()] >= self.rank[rb.index()] {
            (ra, rb)
        } else {
            (rb, ra)
        };

        let old_root_rank = self.rank[root.index()];

        // Record undo BEFORE mutation
        self.undo_stack.push(UndoEntry {
            child,
            old_parent: child, // child was its own root
            root,
            old_root_rank,
        });
        self.merge_reasons.push((a, b, reason));

        // Mutate
        self.parent[child.index()] = root;
        if self.rank[root.index()] == self.rank[child.index()] {
            self.rank[root.index()] += 1;
        }

        // Propagate congruences: re-signature function apps that used `child` as repr
        self.propagate_congruences(root, child);

        true
    }

    /// After merging `child` into `root`, check function applications for
    /// new congruences. Any f-app that had `child` as an argument representative
    /// now has `root` — if the new signature matches an existing entry, those
    /// two f-apps are congruent and must be merged.
    fn propagate_congruences(&mut self, root: TermId, child: TermId) {
        // Collect f-apps to re-signature: those in child's use-list
        // We also need to include f-apps in root's use-list that might
        // now match. For correctness, iterate all f-apps from child's class.
        // After the merge, any f-app whose argument was `child` now has `root` as repr.

        // Gather affected terms: clone to avoid borrow issues
        let affected: Vec<TermId> = self.use_list[child.index()].clone();

        // Also move child's use-list entries to root
        let mut child_uses = std::mem::take(&mut self.use_list[child.index()]);
        self.use_list[root.index()].append(&mut child_uses);

        // Pending congruence merges (can't merge during iteration)
        let mut pending_merges: Vec<(TermId, TermId)> = Vec::new();

        for &tid in &affected {
            if let TermKind::Apply { func, ref args } = self.term_kinds[tid.index()] {
                // Remove old signature
                let old_sig = Signature {
                    func,
                    arg_reprs: args
                        .iter()
                        .map(|&a| {
                            // What was the old representative? For args that aren't child,
                            // their repr is unchanged. For child, old repr was child.
                            let r = self.find(a);
                            // find(a) now returns root if a was in child's class
                            r
                        })
                        .collect(),
                };
                // The old signature was computed with child as repr for child's class.
                // But now find() returns root. So old_sig already has the NEW reprs.
                // We need to check if this signature was already in the table under
                // a different term.
                if let Some(&existing) = self.sig_table.get(&old_sig) {
                    if self.find(existing) != self.find(tid) {
                        pending_merges.push((tid, existing));
                    }
                } else {
                    self.sig_table.insert(old_sig, tid);
                }
            }
        }

        // Process congruence merges
        for (t1, t2) in pending_merges {
            self.merge(t1, t2, MergeReason::Congruence(t1, t2));
        }
    }

    // ── Explanation ──

    /// Build an explanation clause for why `t1` and `t2` are in the same
    /// equivalence class. Returns the set of asserted equality atoms that
    /// form the proof chain.
    ///
    /// Uses BFS through the merge-reason graph to find a path from `t1` to
    /// `t2`, then recursively explains congruence steps.
    fn explain_equality(&self, t1: TermId, t2: TermId) -> Vec<AtomId> {
        if t1 == t2 {
            return Vec::new();
        }
        if self.find(t1) != self.find(t2) {
            return Vec::new(); // shouldn't happen if called correctly
        }

        let mut atoms = Vec::new();
        self.bfs_explain(t1, t2, &mut atoms);
        atoms
    }

    /// BFS-based explanation: build a graph from merge reasons and find
    /// the shortest path from t1 to t2, collecting atoms along the way.
    fn bfs_explain(&self, t1: TermId, t2: TermId, atoms: &mut Vec<AtomId>) {
        use std::collections::{HashSet, VecDeque};

        // Build adjacency: each merge reason (a, b, reason) creates an edge
        let mut adj: HashMap<TermId, Vec<(TermId, usize)>> = HashMap::new();
        for (idx, &(a, b, _)) in self.merge_reasons.iter().enumerate() {
            adj.entry(a).or_default().push((b, idx));
            adj.entry(b).or_default().push((a, idx));
        }

        // BFS from t1 to t2
        let mut visited: HashSet<TermId> = HashSet::new();
        let mut queue: VecDeque<(TermId, Vec<usize>)> = VecDeque::new();
        visited.insert(t1);
        queue.push_back((t1, Vec::new()));

        while let Some((current, path)) = queue.pop_front() {
            if current == t2 {
                // Found path — extract atoms from merge reasons along it
                for &reason_idx in &path {
                    self.extract_atoms_from_reason(reason_idx, atoms);
                }
                return;
            }
            if let Some(neighbors) = adj.get(&current) {
                for &(neighbor, reason_idx) in neighbors {
                    if visited.insert(neighbor) {
                        let mut new_path = path.clone();
                        new_path.push(reason_idx);
                        queue.push_back((neighbor, new_path));
                    }
                }
            }
        }
    }

    /// Extract the asserted atoms from a single merge reason.
    fn extract_atoms_from_reason(&self, reason_idx: usize, atoms: &mut Vec<AtomId>) {
        let (_, _, ref reason) = self.merge_reasons[reason_idx];
        match reason {
            MergeReason::Asserted(atom_id) => {
                atoms.push(*atom_id);
            }
            MergeReason::Congruence(fa, fb) => {
                if let (
                    TermKind::Apply { args: args_a, .. },
                    TermKind::Apply { args: args_b, .. },
                ) = (&self.term_kinds[fa.index()], &self.term_kinds[fb.index()])
                {
                    for (&ai, &bi) in args_a.iter().zip(args_b.iter()) {
                        if ai != bi {
                            self.bfs_explain(ai, bi, atoms);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// TheorySolver implementation
// ============================================================================

impl TheorySolver for EufSolver {
    fn check(&mut self, ctx: &TheoryContext<'_>) -> TheoryResult {
        let trail = ctx.trail;
        let trail_len = trail.len();

        // Ensure level marks cover the current decision level
        let current_level = trail.current_level() as usize;
        while self.level_marks.len() <= current_level + 1 {
            self.level_marks.push(self.undo_stack.len());
            self.merge_level_marks.push(self.merge_reasons.len());
            self.diseq_level_marks.push(self.disequalities.len());
        }

        // Process new trail entries incrementally
        let entries = trail.entries();
        for entry in entries.iter().take(trail_len).skip(self.trail_pos) {
            let var = entry.lit.var();
            let is_true = !entry.lit.is_negated();

            // Check if this variable corresponds to an equality atom
            if let Some((t1, t2)) = self.atom_map.atom_for_var(var) {
                if is_true {
                    // Equality asserted: merge t1 and t2
                    let atom_id =
                        self.atom_map.eq_to_atom[&if t1 <= t2 { (t1, t2) } else { (t2, t1) }];
                    self.merge(t1, t2, MergeReason::Asserted(atom_id));
                } else {
                    // Disequality asserted: record it
                    let atom_id =
                        self.atom_map.eq_to_atom[&if t1 <= t2 { (t1, t2) } else { (t2, t1) }];
                    self.disequalities.push((t1, t2, atom_id));
                }
            }
        }
        self.trail_pos = trail_len;

        // Check disequalities for violations
        for &(t1, t2, atom_id) in &self.disequalities {
            if self.find(t1) == self.find(t2) {
                // Conflict: t1 = t2 (by congruence closure) but (= t1 t2) is false
                // Theory lemma: the asserted equalities that imply t1 = t2, plus
                // the disequality literal, form a conflict clause.
                let eq_atoms = self.explain_equality(t1, t2);
                let eq_var = self.atom_map.var_for_atom(atom_id);

                // Conflict clause: (= t1 t2) ∨ ¬(= a1 b1) ∨ ¬(= a2 b2) ∨ ...
                // The disequality says ¬(= t1 t2) is on the trail (eq_var is false).
                // Each asserted equality (= ai bi) is on the trail (its var is true).
                // So the conflict clause has all these negated:
                let mut clause = Vec::new();
                clause.push(Lit::pos(eq_var)); // (= t1 t2) — currently false
                for &a in &eq_atoms {
                    let v = self.atom_map.var_for_atom(a);
                    clause.push(Lit::neg(v)); // ¬(= ai bi) — currently true, so negation is false
                }
                return TheoryResult::Conflict(clause);
            }
        }

        // Check for theory propagations: if find(t1) == find(t2) for some
        // unassigned atom (= t1 t2), propagate it.
        let num_atoms = self.atom_map.num_atoms();
        for atom_idx in 0..num_atoms {
            let atom_id = AtomId(atom_idx);
            let var = self.atom_map.var_for_atom(atom_id);

            // Skip if already assigned
            if ctx.trail.value(var).is_some() {
                continue;
            }

            let (t1, t2) = self.atom_map.atom_for_var(var).unwrap();
            if self.find(t1) == self.find(t2) {
                // Theory propagation: (= t1 t2) must be true
                let key = self.prop_records.len() as u32;
                self.prop_records.push(PropRecord {
                    lit: Lit::pos(var),
                    t1,
                    t2,
                });
                return TheoryResult::Propagate(vec![TheoryProp {
                    lit: Lit::pos(var),
                    key,
                }]);
            }
        }

        TheoryResult::Consistent
    }

    fn backtrack(&mut self, new_level: u32) {
        let target_level = new_level as usize + 1;

        // Undo union-find merges
        if target_level < self.level_marks.len() {
            let undo_target = self.level_marks[target_level];
            while self.undo_stack.len() > undo_target {
                let entry = self.undo_stack.pop().unwrap();
                // Restore parent
                self.parent[entry.child.index()] = entry.old_parent;
                // Restore rank
                self.rank[entry.root.index()] = entry.old_root_rank;
            }
            self.level_marks.truncate(target_level);
        }

        // Undo merge reasons
        if target_level < self.merge_level_marks.len() {
            let mr_target = self.merge_level_marks[target_level];
            self.merge_reasons.truncate(mr_target);
            self.merge_level_marks.truncate(target_level);
        }

        // Undo disequalities
        if target_level < self.diseq_level_marks.len() {
            let dq_target = self.diseq_level_marks[target_level];
            self.disequalities.truncate(dq_target);
            self.diseq_level_marks.truncate(target_level);
        }

        // Rebuild use-lists and signature table after backtrack
        // (simpler than incremental undo for use-lists)
        self.rebuild_use_lists_and_sig_table();

        // Reset trail position — the SAT solver will re-present existing
        // assignments through BCP, but we need to re-scan from the new position.
        // After backtrack, trail entries above new_level are removed by the SAT solver.
        // We don't know the exact new trail length, so reset to 0 and re-scan.
        self.trail_pos = 0;
    }

    fn explain(&mut self, _lit: Lit, key: u32) -> Vec<Lit> {
        let record = &self.prop_records[key as usize];
        let t1 = record.t1;
        let t2 = record.t2;

        // Build explanation: the asserted equalities that imply t1 = t2
        let eq_atoms = self.explain_equality(t1, t2);

        // Explanation clause: lit ∨ ¬(= a1 b1) ∨ ¬(= a2 b2) ∨ ...
        // The propagated literal + negation of the reasons
        let mut clause = Vec::new();
        clause.push(record.lit);
        for &a in &eq_atoms {
            let v = self.atom_map.var_for_atom(a);
            clause.push(Lit::neg(v));
        }
        clause
    }
}

impl EufSolver {
    /// Rebuild use-lists and signature table from scratch.
    /// Called after backtrack since incremental undo of use-lists is complex.
    fn rebuild_use_lists_and_sig_table(&mut self) {
        // Clear use-lists
        for list in self.use_list.iter_mut() {
            list.clear();
        }
        // Clear sig table
        self.sig_table.clear();

        // Rebuild from term_kinds
        for &tid in &self.func_apps {
            if let TermKind::Apply { func, ref args } = self.term_kinds[tid.index()] {
                // Add to use-lists using CURRENT representatives
                for &arg in args {
                    let repr = self.find(arg);
                    self.use_list[repr.index()].push(tid);
                }
                // Add to signature table using CURRENT representatives
                let sig = Signature {
                    func,
                    arg_reprs: args.iter().map(|&a| self.find(a)).collect(),
                };
                self.sig_table.entry(sig).or_insert(tid);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::{SortId, TermArena, TermKind};

    /// Helper: build a small arena with variables and function applications.
    fn test_arena() -> (TermArena, TermId, TermId, TermId, TermId, TermId) {
        let mut arena = TermArena::new();
        let s = SortId(0);
        let f = FuncId(0);

        let a = arena.intern(
            TermKind::Variable {
                name: "a".into(),
                sort: s,
            },
            s,
        ); // 0
        let b = arena.intern(
            TermKind::Variable {
                name: "b".into(),
                sort: s,
            },
            s,
        ); // 1
        let c = arena.intern(
            TermKind::Variable {
                name: "c".into(),
                sort: s,
            },
            s,
        ); // 2
        let fa = arena.intern(
            TermKind::Apply {
                func: f,
                args: vec![a],
            },
            s,
        ); // 3
        let fb = arena.intern(
            TermKind::Apply {
                func: f,
                args: vec![b],
            },
            s,
        ); // 4

        (arena, a, b, c, fa, fb)
    }

    #[test]
    fn union_find_basic() {
        let (arena, a, b, _c, _fa, _fb) = test_arena();
        let atom_map = AtomMap::new();
        let mut euf = EufSolver::new(&arena, atom_map);

        assert_ne!(euf.find(a), euf.find(b));
        euf.merge(a, b, MergeReason::Asserted(AtomId(0)));
        assert_eq!(euf.find(a), euf.find(b));
    }

    #[test]
    fn union_find_transitivity() {
        let (arena, a, b, c, _fa, _fb) = test_arena();
        let atom_map = AtomMap::new();
        let mut euf = EufSolver::new(&arena, atom_map);

        euf.merge(a, b, MergeReason::Asserted(AtomId(0)));
        euf.merge(b, c, MergeReason::Asserted(AtomId(1)));
        assert_eq!(euf.find(a), euf.find(c));
    }

    #[test]
    fn congruence_propagation() {
        let (arena, a, b, _c, fa, fb) = test_arena();
        let atom_map = AtomMap::new();
        let mut euf = EufSolver::new(&arena, atom_map);

        // Before merge: f(a) and f(b) are in different classes
        assert_ne!(euf.find(fa), euf.find(fb));

        // Merge a and b → f(a) and f(b) should become congruent
        euf.merge(a, b, MergeReason::Asserted(AtomId(0)));
        assert_eq!(euf.find(fa), euf.find(fb));
    }

    #[test]
    fn backtrack_restores_classes() {
        let (arena, a, b, _c, fa, fb) = test_arena();
        let atom_map = AtomMap::new();
        let mut euf = EufSolver::new(&arena, atom_map);

        // Level 0 mark
        assert_eq!(euf.level_marks.len(), 1);

        // Simulate entering level 1
        euf.level_marks.push(euf.undo_stack.len());
        euf.merge_level_marks.push(euf.merge_reasons.len());
        euf.diseq_level_marks.push(euf.disequalities.len());

        euf.merge(a, b, MergeReason::Asserted(AtomId(0)));
        assert_eq!(euf.find(a), euf.find(b));
        assert_eq!(euf.find(fa), euf.find(fb));

        // Backtrack to level 0
        euf.backtrack(0);
        assert_ne!(euf.find(a), euf.find(b));
        assert_ne!(euf.find(fa), euf.find(fb));
    }

    #[test]
    fn explanation_simple() {
        let (arena, a, b, _c, _fa, _fb) = test_arena();
        let mut atom_map = AtomMap::new();
        atom_map.get_or_create(a, b); // AtomId(0)
        let mut euf = EufSolver::new(&arena, atom_map);

        euf.merge(a, b, MergeReason::Asserted(AtomId(0)));
        let explanation = euf.explain_equality(a, b);
        assert_eq!(explanation, vec![AtomId(0)]);
    }

    #[test]
    fn explanation_transitive() {
        let (arena, a, b, c, _fa, _fb) = test_arena();
        let mut atom_map = AtomMap::new();
        atom_map.get_or_create(a, b); // AtomId(0)
        atom_map.get_or_create(b, c); // AtomId(1)
        let mut euf = EufSolver::new(&arena, atom_map);

        euf.merge(a, b, MergeReason::Asserted(AtomId(0)));
        euf.merge(b, c, MergeReason::Asserted(AtomId(1)));
        let explanation = euf.explain_equality(a, c);
        // Should contain both atoms
        assert!(explanation.contains(&AtomId(0)));
        assert!(explanation.contains(&AtomId(1)));
    }
}
