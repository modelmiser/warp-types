//! Bitvector theory module for Nelson-Oppen combination.
//!
//! Implements [`TheoryModule`] with constant propagation and ground
//! evaluation: when all arguments of a `BvOp` have known constant values,
//! the module evaluates the operation and checks for new equalities or
//! disequality violations.
//!
//! This is deliberately minimal — no bit-blasting, no word-level
//! simplification. It handles the cases where BV reasoning reduces to
//! arithmetic on concrete values, which is sufficient for GPU lane-index
//! formulas where most BV terms are constants or simple expressions
//! over constants.

use std::collections::HashMap;

use crate::combine::{ModuleEquality, ModuleResult, TheoryModule};
use crate::term::{BvOpKind, TermId, TermKind};

// ============================================================================
// BV evaluation
// ============================================================================

/// Mask for a `width`-bit unsigned value. Handles `width == 64` without
/// the `1u64 << 64` panic trap. Also used by the session's `bv_const` to
/// mask constants to their declared width at construction.
pub(crate) fn width_mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Evaluate a bitvector operation on concrete `(arg_width, arg_value)` pairs.
///
/// `result_width` is the output width (for `Extract` this is `hi - lo + 1`;
/// for `Concat` it is the sum of arg widths; for the rest it is the common
/// arg width). Per-arg widths are needed for `Concat`, which shifts the
/// first arg left by the *second* arg's width.
fn evaluate(op: BvOpKind, result_width: u32, args: &[(u32, u64)]) -> u64 {
    let out_mask = width_mask(result_width);
    let vals = || args.iter().map(|&(_, v)| v);
    let result = match op {
        // Wrapping fold: BV addition is modulo 2^width, and `.sum()` would
        // panic on u64 overflow in debug builds.
        BvOpKind::Add => vals().fold(0u64, |acc, v| acc.wrapping_add(v)),
        BvOpKind::And => vals().fold(out_mask, |a, b| a & b),
        BvOpKind::Or => vals().fold(0, |a, b| a | b),
        BvOpKind::Xor => vals().fold(0, |a, b| a ^ b),
        BvOpKind::Not => {
            debug_assert_eq!(args.len(), 1, "bvnot is unary");
            !args[0].1
        }
        BvOpKind::Sub => {
            debug_assert_eq!(args.len(), 2, "bvsub is binary");
            args[0].1.wrapping_sub(args[1].1)
        }
        BvOpKind::Extract { hi, lo } => {
            debug_assert_eq!(args.len(), 1, "extract is unary");
            debug_assert!(lo <= hi, "extract requires lo <= hi");
            args[0].1 >> lo
        }
        BvOpKind::Concat => {
            debug_assert_eq!(args.len(), 2, "concat is binary");
            let (_, hi_val) = args[0];
            let (rhs_width, lo_val) = args[1];
            // rhs_width ≤ 64 (validated at construction); guard the shift.
            let shifted = if rhs_width >= 64 {
                0
            } else {
                hi_val << rhs_width
            };
            shifted | (lo_val & width_mask(rhs_width))
        }
    };
    result & out_mask
}

// ============================================================================
// Value tracking
// ============================================================================

/// Why a term has its known constant value.
#[derive(Clone)]
enum ValueReason {
    /// Inherent BvConst — no trail dependency.
    Constant,
    /// Propagated through an asserted equality from `source`, which had a
    /// known value. `premises` are the trail equality atoms justifying the
    /// equality itself: for a trail atom that is the atom's own pair; for an
    /// EUF-derived equality it is the premise set the combiner supplied.
    Equality {
        source: TermId,
        premises: Vec<(TermId, TermId)>,
    },
    /// Computed by evaluating a BvOp whose args all had known values.
    /// Premises come from the args' reasons (recursive).
    Evaluation,
}

/// An equality the module has been told about, with the trail atoms that
/// justify it. Kept for the whole (level-scoped) lifetime of the assertion:
/// an equality between two not-yet-valued terms is value-relevant later,
/// when either side becomes valued, so `propagate()` re-examines the list
/// to fixpoint instead of dropping it (standard Nelson-Oppen treatment).
struct EqRecord {
    t1: TermId,
    t2: TermId,
    /// Trail equality atoms justifying `t1 = t2`.
    premises: Vec<(TermId, TermId)>,
}

/// Undo record for backtracking.
struct BvUndo {
    tid: TermId,
    old_value: Option<(u32, u64)>,
    old_reason: Option<ValueReason>,
}

// ============================================================================
// BV solver
// ============================================================================

/// Bitvector theory module.
///
/// Tracks known constant values for terms, evaluates ground `BvOp`
/// expressions, and detects equalities/conflicts through constant
/// propagation.
pub struct BvSolver {
    /// Read-only copy of term kinds from the arena.
    term_kinds: Vec<TermKind>,
    /// Known constant value for each term: `(width, value)`.
    known_value: Vec<Option<(u32, u64)>>,
    /// Why the term has its value (for premise collection).
    value_reasons: Vec<Option<ValueReason>>,
    /// `BvOp` term IDs (for re-evaluation).
    bv_ops: Vec<TermId>,
    /// Reverse map: `(width, value)` → term IDs with that constant value.
    value_to_terms: HashMap<(u32, u64), Vec<TermId>>,
    /// Active equalities (trail atoms and EUF-derived), with premises.
    /// Re-examined to fixpoint by `propagate()` — see [`EqRecord`].
    equalities: Vec<EqRecord>,
    /// Active disequalities from the trail.
    disequalities: Vec<(TermId, TermId)>,
    /// Whether re-evaluation is needed.
    dirty: bool,
    // ── Backtracking ──
    undo_stack: Vec<BvUndo>,
    level_marks: Vec<usize>,
    eq_level_marks: Vec<usize>,
    diseq_level_marks: Vec<usize>,
}

impl BvSolver {
    /// Create a BV solver from the arena's term kinds.
    pub fn new(term_kinds: &[TermKind]) -> Self {
        let n = term_kinds.len();
        let mut known_value = vec![None; n];
        let mut value_reasons = vec![None; n];
        let mut bv_ops = Vec::new();
        let mut value_to_terms: HashMap<(u32, u64), Vec<TermId>> = HashMap::new();

        for (i, kind) in term_kinds.iter().enumerate() {
            match kind {
                TermKind::BvConst { width, value } => {
                    known_value[i] = Some((*width, *value));
                    value_reasons[i] = Some(ValueReason::Constant);
                    value_to_terms
                        .entry((*width, *value))
                        .or_default()
                        .push(TermId(i as u32));
                }
                TermKind::BvOp { .. } => {
                    bv_ops.push(TermId(i as u32));
                }
                _ => {}
            }
        }

        BvSolver {
            term_kinds: term_kinds.to_vec(),
            known_value,
            value_reasons,
            bv_ops,
            value_to_terms,
            equalities: Vec::new(),
            disequalities: Vec::new(),
            dirty: false,
            undo_stack: Vec::new(),
            level_marks: vec![0],
            eq_level_marks: vec![0],
            diseq_level_marks: vec![0],
        }
    }

    /// Set a term's known value, recording undo info.
    fn set_value(&mut self, tid: TermId, width: u32, val: u64, reason: ValueReason) {
        if self.known_value[tid.index()] == Some((width, val)) {
            return; // Idempotent
        }
        self.undo_stack.push(BvUndo {
            tid,
            old_value: self.known_value[tid.index()],
            old_reason: self.value_reasons[tid.index()].clone(),
        });
        self.known_value[tid.index()] = Some((width, val));
        self.value_reasons[tid.index()] = Some(reason);
        self.value_to_terms
            .entry((width, val))
            .or_default()
            .push(tid);
        self.dirty = true;
    }

    /// Collect the trail equality premises that led to a term's value.
    fn collect_premises(&self, tid: TermId, out: &mut Vec<(TermId, TermId)>) {
        match &self.value_reasons[tid.index()] {
            Some(ValueReason::Constant) => {}
            Some(ValueReason::Equality { source, premises }) => {
                out.extend(premises.iter().copied());
                // Also collect premises from the source term
                self.collect_premises(*source, out);
            }
            Some(ValueReason::Evaluation) => {
                if let TermKind::BvOp { ref args, .. } = self.term_kinds[tid.index()] {
                    for &arg in args {
                        self.collect_premises(arg, out);
                    }
                }
            }
            None => {}
        }
    }

    /// Rebuild the reverse value→terms map from scratch.
    fn rebuild_value_to_terms(&mut self) {
        self.value_to_terms.clear();
        for (i, val) in self.known_value.iter().enumerate() {
            if let Some((w, v)) = val {
                self.value_to_terms
                    .entry((*w, *v))
                    .or_default()
                    .push(TermId(i as u32));
            }
        }
    }
}

impl TheoryModule for BvSolver {
    fn notify_equality(&mut self, t1: TermId, t2: TermId) {
        // A trail equality atom is its own premise.
        self.notify_derived_equality(t1, t2, vec![(t1, t2)]);
    }

    fn notify_derived_equality(&mut self, t1: TermId, t2: TermId, premises: Vec<(TermId, TermId)>) {
        // Record only — value propagation, mismatch detection, and the
        // (both-unvalued) case are all handled uniformly by the fixpoint in
        // `propagate()`. Dropping an equality between two not-yet-valued
        // terms here was the round-2 finding-A unsoundness.
        self.equalities.push(EqRecord { t1, t2, premises });
        self.dirty = true;
    }

    fn notify_disequality(&mut self, t1: TermId, t2: TermId) {
        self.disequalities.push((t1, t2));
        // A disequality is value-relevant on its own: both sides may already
        // be valued equal, and nothing else would re-trigger the check.
        self.dirty = true;
    }

    fn propagate(&mut self) -> ModuleResult {
        if !self.dirty {
            return ModuleResult::Consistent(Vec::new());
        }

        // Fixpoint: equality propagation can enable BvOp evaluation, whose
        // results can make further recorded equalities value-relevant.
        loop {
            self.dirty = false;

            // Propagate recorded equalities. Each is re-examined every pass:
            // an equality between two unvalued terms becomes actionable as
            // soon as either side gains a value.
            for i in 0..self.equalities.len() {
                let (t1, t2) = (self.equalities[i].t1, self.equalities[i].t2);
                let v1 = self.known_value[t1.index()];
                let v2 = self.known_value[t2.index()];
                match (v1, v2) {
                    (Some(val1), Some(val2)) => {
                        // Both known: a mismatch is a theory conflict. The
                        // equality's own premises plus both value-premise
                        // chains form the conflict clause.
                        if val1 != val2 {
                            let mut eq_premises = self.equalities[i].premises.clone();
                            self.collect_premises(t1, &mut eq_premises);
                            self.collect_premises(t2, &mut eq_premises);
                            eq_premises.sort();
                            eq_premises.dedup();
                            return ModuleResult::Conflict {
                                eq_premises,
                                diseq_premises: Vec::new(),
                            };
                        }
                    }
                    (Some((w, val)), None) => {
                        let premises = self.equalities[i].premises.clone();
                        self.set_value(
                            t2,
                            w,
                            val,
                            ValueReason::Equality {
                                source: t1,
                                premises,
                            },
                        );
                    }
                    (None, Some((w, val))) => {
                        let premises = self.equalities[i].premises.clone();
                        self.set_value(
                            t1,
                            w,
                            val,
                            ValueReason::Equality {
                                source: t2,
                                premises,
                            },
                        );
                    }
                    // Neither side valued yet — stays recorded; revisited when
                    // a later pass (or a later propagate() call) adds values.
                    (None, None) => {}
                }
            }

            // Evaluate BvOp terms whose args are all known
            for i in 0..self.bv_ops.len() {
                let op_tid = self.bv_ops[i];
                if let TermKind::BvOp {
                    op,
                    width,
                    ref args,
                } = self.term_kinds[op_tid.index()].clone()
                {
                    let arg_pairs: Option<Vec<(u32, u64)>> =
                        args.iter().map(|&a| self.known_value[a.index()]).collect();
                    if let Some(pairs) = arg_pairs {
                        let result = evaluate(op, width, &pairs);
                        match self.known_value[op_tid.index()] {
                            None => self.set_value(op_tid, width, result, ValueReason::Evaluation),
                            Some(recorded) => {
                                // The op already has a value (e.g. forced by a
                                // trail equality). If ground evaluation disagrees,
                                // that is a conflict — same class as the
                                // equality mismatch above.
                                if recorded != (width, result) {
                                    let mut eq_premises = Vec::new();
                                    self.collect_premises(op_tid, &mut eq_premises);
                                    for &arg in args {
                                        self.collect_premises(arg, &mut eq_premises);
                                    }
                                    eq_premises.sort();
                                    eq_premises.dedup();
                                    return ModuleResult::Conflict {
                                        eq_premises,
                                        diseq_premises: Vec::new(),
                                    };
                                }
                            }
                        }
                    }
                }
            }

            if !self.dirty {
                break;
            }
        }

        // Check disequalities for BV-level conflicts
        for &(d_t1, d_t2) in &self.disequalities {
            if let (Some((w1, v1)), Some((w2, v2))) = (
                self.known_value[d_t1.index()],
                self.known_value[d_t2.index()],
            ) {
                if w1 == w2 && v1 == v2 {
                    let mut eq_premises = Vec::new();
                    self.collect_premises(d_t1, &mut eq_premises);
                    self.collect_premises(d_t2, &mut eq_premises);
                    eq_premises.sort();
                    eq_premises.dedup();
                    return ModuleResult::Conflict {
                        eq_premises,
                        diseq_premises: vec![(d_t1, d_t2)],
                    };
                }
            }
        }

        // Report new equalities from constant evaluation
        let mut new_eqs = Vec::new();
        for i in 0..self.term_kinds.len() {
            if let Some((w, v)) = self.known_value[i] {
                if let Some(terms) = self.value_to_terms.get(&(w, v)) {
                    for &other in terms {
                        let tid = TermId(i as u32);
                        if other > tid {
                            let mut premises = Vec::new();
                            self.collect_premises(tid, &mut premises);
                            self.collect_premises(other, &mut premises);
                            premises.sort();
                            premises.dedup();
                            new_eqs.push(ModuleEquality {
                                t1: tid,
                                t2: other,
                                premises,
                            });
                        }
                    }
                }
            }
        }

        ModuleResult::Consistent(new_eqs)
    }

    fn push_level(&mut self) {
        self.level_marks.push(self.undo_stack.len());
        self.eq_level_marks.push(self.equalities.len());
        self.diseq_level_marks.push(self.disequalities.len());
    }

    fn backtrack(&mut self, level: u32) {
        let target = level as usize + 1;
        if target < self.level_marks.len() {
            let undo_target = self.level_marks[target];
            while self.undo_stack.len() > undo_target {
                let entry = self.undo_stack.pop().unwrap();
                self.known_value[entry.tid.index()] = entry.old_value;
                self.value_reasons[entry.tid.index()] = entry.old_reason;
            }
            self.level_marks.truncate(target);
            self.rebuild_value_to_terms();
        }
        if target < self.eq_level_marks.len() {
            let eq_target = self.eq_level_marks[target];
            self.equalities.truncate(eq_target);
            self.eq_level_marks.truncate(target);
        }
        if target < self.diseq_level_marks.len() {
            let dq_target = self.diseq_level_marks[target];
            self.disequalities.truncate(dq_target);
            self.diseq_level_marks.truncate(target);
        }
        // The combiner re-notifies the surviving trail, so any still-valid
        // conflict is rediscovered by the next propagate().
        self.dirty = true;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::{SortId, TermArena};

    fn make_arena() -> (TermArena, Vec<TermKind>) {
        let mut arena = TermArena::new();
        let s = SortId(0);
        // 0: x
        let x = arena.intern(
            TermKind::Variable {
                name: "x".into(),
                sort: s,
            },
            s,
        );
        // 1: bvconst(5, 3)
        let _three = arena.intern(TermKind::BvConst { width: 5, value: 3 }, s);
        // 2: bvconst(5, 1)
        let one = arena.intern(TermKind::BvConst { width: 5, value: 1 }, s);
        // 3: bvadd(5, [x, one])
        let _add = arena.intern(
            TermKind::BvOp {
                op: BvOpKind::Add,
                width: 5,
                args: vec![x, one],
            },
            s,
        );
        // 4: bvconst(5, 4)
        let _four = arena.intern(TermKind::BvConst { width: 5, value: 4 }, s);

        let kinds: Vec<TermKind> = (0..arena.len())
            .map(|i| arena.get(TermId(i as u32)).kind.clone())
            .collect();
        (arena, kinds)
    }

    #[test]
    fn constants_known_at_construction() {
        let (_, kinds) = make_arena();
        let bv = BvSolver::new(&kinds);
        assert_eq!(bv.known_value[1], Some((5, 3))); // bvconst(5, 3)
        assert_eq!(bv.known_value[2], Some((5, 1))); // bvconst(5, 1)
        assert_eq!(bv.known_value[4], Some((5, 4))); // bvconst(5, 4)
        assert_eq!(bv.known_value[0], None); // x: unknown
        assert_eq!(bv.known_value[3], None); // bvadd: not yet evaluated
    }

    #[test]
    fn equality_propagates_value() {
        let (_, kinds) = make_arena();
        let mut bv = BvSolver::new(&kinds);
        // Tell module: x = bvconst(5, 3)
        bv.notify_equality(TermId(0), TermId(1));
        let _ = bv.propagate();
        assert_eq!(bv.known_value[0], Some((5, 3)));
    }

    // ── Round-2 finding A: an equality between two not-yet-valued terms must
    //    be re-examined when either side becomes valued, not dropped. ──

    #[test]
    fn unvalued_equality_reexamined_on_later_value() {
        let (_, kinds) = make_arena();
        let mut bv = BvSolver::new(&kinds);
        // x = bvadd(x,1): both unvalued at notification time. (Uses the two
        // unvalued terms of the arena; the semantic content doesn't matter —
        // only that the link is kept.)
        bv.notify_equality(TermId(0), TermId(3));
        let _ = bv.propagate();
        assert_eq!(bv.known_value[3], None);
        // Later x = 3 arrives: the recorded x = bvadd(x,1) link must now
        // fire — bvadd gets forced to 3, evaluation says 3+1 = 4 → conflict.
        bv.notify_equality(TermId(0), TermId(1)); // x = 3
        let result = bv.propagate();
        match result {
            ModuleResult::Conflict { eq_premises, .. } => {
                assert!(eq_premises.contains(&(TermId(0), TermId(3))));
                assert!(eq_premises.contains(&(TermId(0), TermId(1))));
            }
            _ => panic!("dropped unvalued equality: x = bvadd(x,1) ∧ x = 3 must conflict"),
        }
    }

    #[test]
    fn derived_equality_carries_supplied_premises() {
        let (_, kinds) = make_arena();
        let mut bv = BvSolver::new(&kinds);
        // Combiner-style derived equality: bvadd(x,1) = bvconst(5,4) is not a
        // trail atom; its justification is some other atom pair, here (x, x+1)
        // standing in for a congruence explanation.
        bv.notify_derived_equality(TermId(3), TermId(4), vec![(TermId(0), TermId(3))]);
        bv.notify_equality(TermId(3), TermId(1)); // bvadd = 3, but derived says 4
        let result = bv.propagate();
        match result {
            ModuleResult::Conflict { eq_premises, .. } => {
                // The derived premises (not the non-atom pair itself) must
                // appear in the conflict.
                assert!(eq_premises.contains(&(TermId(0), TermId(3))));
                assert!(eq_premises.contains(&(TermId(3), TermId(1))));
                assert!(!eq_premises.contains(&(TermId(3), TermId(4))));
            }
            _ => panic!("conflicting derived equality must conflict"),
        }
    }

    #[test]
    fn bvop_evaluates_after_arg_known() {
        let (_, kinds) = make_arena();
        let mut bv = BvSolver::new(&kinds);
        bv.notify_equality(TermId(0), TermId(1)); // x = 3
        let _ = bv.propagate();
        // bvadd(x, 1) should now be evaluated: 3 + 1 = 4
        assert_eq!(bv.known_value[3], Some((5, 4)));
    }

    #[test]
    fn conflict_on_disequality_violation() {
        let (_, kinds) = make_arena();
        let mut bv = BvSolver::new(&kinds);
        bv.notify_equality(TermId(0), TermId(1)); // x = 3
        bv.notify_disequality(TermId(3), TermId(4)); // bvadd(x,1) ≠ 4
        let result = bv.propagate();
        assert!(matches!(result, ModuleResult::Conflict { .. }));
    }

    // ── Finding B: equality between two terms with different known values ──

    #[test]
    fn conflicting_known_values_conflict() {
        // x = bvconst(5,3) then x = bvconst(5,4): both sides known with
        // different values — must surface as a module conflict, not be
        // silently dropped.
        let (_, kinds) = make_arena();
        let mut bv = BvSolver::new(&kinds);
        bv.notify_equality(TermId(0), TermId(1)); // x = 3
        bv.notify_equality(TermId(0), TermId(4)); // x = 4 — contradiction
        let result = bv.propagate();
        match result {
            ModuleResult::Conflict { eq_premises, .. } => {
                // The clause must be built from the two trail equalities.
                assert!(eq_premises.contains(&(TermId(0), TermId(4))));
                assert!(eq_premises.contains(&(TermId(0), TermId(1))));
            }
            _ => panic!("expected Conflict for x = 3 ∧ x = 4"),
        }
    }

    #[test]
    fn forced_op_value_contradicts_evaluation() {
        // Same bug class at the evaluation site: bvadd(x,1) = 3 forced by an
        // equality, but x = 3 makes it evaluate to 4 — must conflict.
        let (_, kinds) = make_arena();
        let mut bv = BvSolver::new(&kinds);
        bv.notify_equality(TermId(0), TermId(1)); // x = 3
        bv.notify_equality(TermId(3), TermId(1)); // bvadd(x,1) = 3, but 3+1 = 4
        let result = bv.propagate();
        assert!(matches!(result, ModuleResult::Conflict { .. }));
    }

    // ── Finding G: bvadd must wrap, not panic, on u64 overflow ──

    #[test]
    fn evaluate_add_wraps_on_u64_overflow() {
        // u64::MAX + 2 wraps to 1 — must not panic in debug builds.
        assert_eq!(evaluate(BvOpKind::Add, 64, &[(64, u64::MAX), (64, 2)]), 1);
    }

    #[test]
    fn evaluate_bv_ops() {
        assert_eq!(evaluate(BvOpKind::Add, 5, &[(5, 3), (5, 1)]), 4);
        // overflow: (7+1) & 0b111 = 0
        assert_eq!(evaluate(BvOpKind::Add, 3, &[(3, 7), (3, 1)]), 0);
        assert_eq!(evaluate(BvOpKind::And, 8, &[(8, 0xFF), (8, 0x0F)]), 0x0F);
        assert_eq!(evaluate(BvOpKind::Or, 8, &[(8, 0xF0), (8, 0x0F)]), 0xFF);
        assert_eq!(evaluate(BvOpKind::Xor, 8, &[(8, 0xFF), (8, 0xFF)]), 0x00);
    }

    #[test]
    fn evaluate_bvnot() {
        // 5-bit: ~0b01010 & 0b11111 = 0b10101 = 21
        assert_eq!(evaluate(BvOpKind::Not, 5, &[(5, 0b01010)]), 0b10101);
        // 64-bit: ~0 = u64::MAX
        assert_eq!(evaluate(BvOpKind::Not, 64, &[(64, 0)]), u64::MAX);
    }

    #[test]
    fn evaluate_bvsub() {
        // 5 - 2 = 3 (no wrap)
        assert_eq!(evaluate(BvOpKind::Sub, 5, &[(5, 5), (5, 2)]), 3);
        // 0 - 1 = 31 in 5-bit (wrap): (-1 as u64) & 0b11111 = 0b11111 = 31
        assert_eq!(evaluate(BvOpKind::Sub, 5, &[(5, 0), (5, 1)]), 31);
    }

    #[test]
    fn evaluate_extract() {
        // extract [3:1] of 8-bit 0b11010110 = 0b011 = 3
        // (bit 7)1 1 0 1 0 1 1 0 (bit 0); bits 3..=1 are 0 1 1 → 0b011
        let v = 0b1101_0110;
        assert_eq!(
            evaluate(BvOpKind::Extract { hi: 3, lo: 1 }, 3, &[(8, v)]),
            0b011
        );
        // extract [7:7] of 8-bit 0b10000000 = 0b1
        assert_eq!(
            evaluate(BvOpKind::Extract { hi: 7, lo: 7 }, 1, &[(8, 0x80)]),
            1
        );
        // extract [7:0] = identity on 8 bits
        assert_eq!(
            evaluate(BvOpKind::Extract { hi: 7, lo: 0 }, 8, &[(8, 0xAB)]),
            0xAB
        );
    }

    #[test]
    fn evaluate_concat() {
        // concat(3-bit 0b101, 3-bit 0b010) = 6-bit 0b101_010 = 42
        assert_eq!(
            evaluate(BvOpKind::Concat, 6, &[(3, 0b101), (3, 0b010)]),
            0b101_010
        );
        // concat(4-bit 0xF, 4-bit 0x0) = 8-bit 0xF0
        assert_eq!(evaluate(BvOpKind::Concat, 8, &[(4, 0xF), (4, 0x0)]), 0xF0);
    }
}
