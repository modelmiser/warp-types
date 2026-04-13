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

/// Evaluate a bitvector operation on concrete values.
fn evaluate(op: BvOpKind, width: u32, args: &[u64]) -> u64 {
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let result = match op {
        BvOpKind::Add => args.iter().copied().sum::<u64>(),
        BvOpKind::And => args.iter().copied().fold(mask, |a, b| a & b),
        BvOpKind::Or => args.iter().copied().fold(0, |a, b| a | b),
        BvOpKind::Xor => args.iter().copied().fold(0, |a, b| a ^ b),
    };
    result & mask
}

// ============================================================================
// Value tracking
// ============================================================================

/// Why a term has its known constant value.
#[derive(Clone)]
enum ValueReason {
    /// Inherent BvConst — no trail dependency.
    Constant,
    /// Propagated from an equality notification: `(t1, t2)` was asserted
    /// on the trail, and the other side had a known value.
    Equality(TermId, TermId),
    /// Computed by evaluating a BvOp whose args all had known values.
    /// Premises come from the args' reasons (recursive).
    Evaluation,
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
    /// Active disequalities from the trail.
    disequalities: Vec<(TermId, TermId)>,
    /// Whether re-evaluation is needed.
    dirty: bool,
    // ── Backtracking ──
    undo_stack: Vec<BvUndo>,
    level_marks: Vec<usize>,
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
            disequalities: Vec::new(),
            dirty: false,
            undo_stack: Vec::new(),
            level_marks: vec![0],
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
            Some(ValueReason::Equality(t1, t2)) => {
                out.push((*t1, *t2));
                // Also collect premises from the source term
                let source = if *t1 == tid { *t2 } else { *t1 };
                self.collect_premises(source, out);
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
        let v1 = self.known_value[t1.index()];
        let v2 = self.known_value[t2.index()];
        match (v1, v2) {
            (Some(_), Some(_)) => {} // Both known — combiner handles mismatches
            (Some((w, val)), None) => {
                self.set_value(t2, w, val, ValueReason::Equality(t2, t1));
            }
            (None, Some((w, val))) => {
                self.set_value(t1, w, val, ValueReason::Equality(t1, t2));
            }
            (None, None) => {}
        }
    }

    fn notify_disequality(&mut self, t1: TermId, t2: TermId) {
        self.disequalities.push((t1, t2));
    }

    fn propagate(&mut self) -> ModuleResult {
        if !self.dirty {
            return ModuleResult::Consistent(Vec::new());
        }
        self.dirty = false;

        // Evaluate BvOp terms whose args are all known
        for i in 0..self.bv_ops.len() {
            let op_tid = self.bv_ops[i];
            if self.known_value[op_tid.index()].is_some() {
                continue;
            }
            if let TermKind::BvOp {
                op,
                width,
                ref args,
            } = self.term_kinds[op_tid.index()].clone()
            {
                let arg_values: Option<Vec<u64>> = args
                    .iter()
                    .map(|&a| self.known_value[a.index()].map(|(_, v)| v))
                    .collect();
                if let Some(vals) = arg_values {
                    let result = evaluate(op, width, &vals);
                    self.set_value(op_tid, width, result, ValueReason::Evaluation);
                }
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
        if target < self.diseq_level_marks.len() {
            let dq_target = self.diseq_level_marks[target];
            self.disequalities.truncate(dq_target);
            self.diseq_level_marks.truncate(target);
        }
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
        assert_eq!(bv.known_value[0], Some((5, 3)));
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

    #[test]
    fn evaluate_bv_ops() {
        assert_eq!(evaluate(BvOpKind::Add, 5, &[3, 1]), 4);
        assert_eq!(evaluate(BvOpKind::Add, 3, &[7, 1]), 0); // overflow: (7+1) & 0b111 = 0
        assert_eq!(evaluate(BvOpKind::And, 8, &[0xFF, 0x0F]), 0x0F);
        assert_eq!(evaluate(BvOpKind::Or, 8, &[0xF0, 0x0F]), 0xFF);
        assert_eq!(evaluate(BvOpKind::Xor, 8, &[0xFF, 0xFF]), 0x00);
    }
}
