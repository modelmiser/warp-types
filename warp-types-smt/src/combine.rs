//! Nelson-Oppen theory combination for DPLL(T).
//!
//! [`CombiningSolver`] wraps an EUF engine and a [`TheoryModule`], mediating
//! equality sharing between them. It implements
//! [`TheorySolver`](warp_types_sat::theory::TheorySolver) so the SAT backbone
//! sees a single theory oracle.
//!
//! # Architecture
//!
//! ```text
//! SAT solver (CDCL)
//!     │  check / backtrack / explain
//!     ▼
//! CombiningSolver ──── implements TheorySolver
//!     ├── EufSolver  ── congruence closure + trail scanning
//!     ├── TheoryModule ── pluggable domain theory (BV, LIA, ...)
//!     └── equality sharing ── Nelson-Oppen protocol
//! ```
//!
//! The equality sharing loop is driven by the SAT solver's theory-check
//! protocol — no internal fixpoint loop:
//!
//! 1. EUF processes the trail until fixpoint
//! 2. Once consistent, the combiner shares trail equalities to the module
//! 3. The module reports new equalities (with premises) or conflicts
//! 4. The combiner propagates equalities to the SAT solver, or returns
//!    conflict clauses constructed from the module's premises
//! 5. The SAT solver records propagations, re-runs BCP, calls `check()` again

use crate::euf::EufSolver;
use crate::term::TermId;

use warp_types_sat::literal::Lit;
use warp_types_sat::theory::{TheoryContext, TheoryProp, TheoryResult, TheorySolver};

// ============================================================================
// Theory module trait
// ============================================================================

/// An equality discovered by a theory module, with its premises.
///
/// The premises are the trail equalities that the module relied on
/// to derive this conclusion. The combining solver needs them to
/// construct sound conflict clauses and explanation clauses.
pub struct ModuleEquality {
    /// First term.
    pub t1: TermId,
    /// Second term.
    pub t2: TermId,
    /// Trail equality atoms that this deduction depends on.
    /// Each `(a, b)` means "the module used `a = b` from the trail."
    pub premises: Vec<(TermId, TermId)>,
}

/// Result of a theory module's consistency check.
pub enum ModuleResult {
    /// Module is consistent. May report new equalities with premises.
    Consistent(Vec<ModuleEquality>),
    /// Module detected an inconsistency.
    Conflict {
        /// Equality premises (asserted true on trail) contributing to the conflict.
        eq_premises: Vec<(TermId, TermId)>,
        /// Disequality premises (asserted false on trail) contributing to the conflict.
        diseq_premises: Vec<(TermId, TermId)>,
    },
}

/// A theory module for Nelson-Oppen combination.
///
/// Unlike [`TheorySolver`] (the SAT-facing interface that scans the trail),
/// a `TheoryModule` receives equality and disequality assertions directly
/// from the combining solver. It reasons within its own theory and reports
/// implied equalities (with premises) for sharing with other theories.
pub trait TheoryModule {
    /// The module is informed that `t1 = t2`.
    fn notify_equality(&mut self, t1: TermId, t2: TermId);

    /// The module is informed that `t1 ≠ t2`.
    fn notify_disequality(&mut self, t1: TermId, t2: TermId);

    /// Check consistency and report any new equalities discovered.
    fn propagate(&mut self) -> ModuleResult;

    /// Push a new backtrack level.
    fn push_level(&mut self);

    /// Backtrack to the given decision level (undo all state above it).
    fn backtrack(&mut self, level: u32);
}

// ============================================================================
// Null module (zero-cost pass-through)
// ============================================================================

/// Zero-cost null theory module.
///
/// Always consistent, never discovers equalities. When `CombiningSolver`
/// is monomorphized with `NullModule`, the compiler eliminates all
/// module-related code paths — performance identical to bare EUF.
pub struct NullModule;

impl TheoryModule for NullModule {
    #[inline(always)]
    fn notify_equality(&mut self, _t1: TermId, _t2: TermId) {}

    #[inline(always)]
    fn notify_disequality(&mut self, _t1: TermId, _t2: TermId) {}

    #[inline(always)]
    fn propagate(&mut self) -> ModuleResult {
        ModuleResult::Consistent(Vec::new())
    }

    #[inline(always)]
    fn push_level(&mut self) {}

    #[inline(always)]
    fn backtrack(&mut self, _level: u32) {}
}

// ============================================================================
// Combining solver
// ============================================================================

/// Key-space partition: EUF propagations use keys `0..MODULE_KEY_OFFSET`,
/// module propagations use `MODULE_KEY_OFFSET..`.
const MODULE_KEY_OFFSET: u32 = 1 << 24;

/// Record for lazily explaining a module-originated propagation.
struct ModulePropRecord {
    lit: Lit,
    /// Trail equality premises the module relied on for this deduction.
    premises: Vec<(TermId, TermId)>,
}

/// Nelson-Oppen combining solver.
///
/// Wraps an [`EufSolver`] and a [`TheoryModule`], implementing
/// [`TheorySolver`] for the SAT backbone.
pub struct CombiningSolver<M: TheoryModule> {
    euf: EufSolver,
    module: M,
    /// Trail entries already dispatched to the module.
    module_trail_pos: usize,
    /// Lazy-explanation records for module propagations.
    module_props: Vec<ModulePropRecord>,
}

impl<M: TheoryModule> CombiningSolver<M> {
    /// Create a combining solver wrapping EUF and a theory module.
    pub fn new(euf: EufSolver, module: M) -> Self {
        CombiningSolver {
            euf,
            module,
            module_trail_pos: 0,
            module_props: Vec::new(),
        }
    }

    /// Dispatch trail equalities/disequalities to the module.
    fn share_trail_to_module(&mut self, ctx: &TheoryContext<'_>) {
        let entries = ctx.trail.entries();
        let trail_len = ctx.trail.len();

        for entry in entries.iter().take(trail_len).skip(self.module_trail_pos) {
            let var = entry.lit.var();
            let is_true = !entry.lit.is_negated();

            if let Some((t1, t2)) = self.euf.atom_map.atom_for_var(var) {
                if is_true {
                    self.module.notify_equality(t1, t2);
                } else {
                    self.module.notify_disequality(t1, t2);
                }
            }
        }
        self.module_trail_pos = trail_len;
    }

    /// Build a conflict clause from module premises.
    ///
    /// Equality premises are negated (they're true on the trail),
    /// disequality premises become positive (they're false on the trail).
    fn build_conflict_clause(
        &self,
        eq_premises: &[(TermId, TermId)],
        diseq_premises: &[(TermId, TermId)],
    ) -> Vec<Lit> {
        let mut clause = Vec::new();
        for &(t1, t2) in eq_premises {
            let key = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            if let Some(&atom_id) = self.euf.atom_map.eq_to_atom.get(&key) {
                let var = self.euf.atom_map.var_for_atom(atom_id);
                clause.push(Lit::neg(var));
            }
        }
        for &(t1, t2) in diseq_premises {
            let key = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            if let Some(&atom_id) = self.euf.atom_map.eq_to_atom.get(&key) {
                let var = self.euf.atom_map.var_for_atom(atom_id);
                clause.push(Lit::pos(var));
            }
        }
        clause
    }
}

impl<M: TheoryModule> TheorySolver for CombiningSolver<M> {
    fn check(&mut self, ctx: &TheoryContext<'_>) -> TheoryResult {
        // ── Phase 1: EUF processes the trail ──
        let euf_result = self.euf.check(ctx);
        match &euf_result {
            TheoryResult::Conflict(_) | TheoryResult::Propagate(_) => {
                return euf_result;
            }
            TheoryResult::Consistent => {}
        }

        // ── Phase 2: Share trail to module ──
        self.share_trail_to_module(ctx);

        // ── Phase 3: Module consistency + equality sharing ──
        match self.module.propagate() {
            ModuleResult::Conflict {
                eq_premises,
                diseq_premises,
            } => {
                let clause = self.build_conflict_clause(&eq_premises, &diseq_premises);
                if clause.is_empty() {
                    TheoryResult::Consistent // Can't express conflict (missing atoms)
                } else {
                    TheoryResult::Conflict(clause)
                }
            }
            ModuleResult::Consistent(new_eqs) => {
                let mut props = Vec::new();

                for meq in new_eqs {
                    if self.euf.find(meq.t1) == self.euf.find(meq.t2) {
                        continue;
                    }

                    let canonical = if meq.t1 <= meq.t2 {
                        (meq.t1, meq.t2)
                    } else {
                        (meq.t2, meq.t1)
                    };
                    let atom_id = match self.euf.atom_map.eq_to_atom.get(&canonical) {
                        Some(&id) => id,
                        None => continue,
                    };
                    let var = self.euf.atom_map.var_for_atom(atom_id);

                    match ctx.trail.value(var) {
                        None => {
                            let key = self.module_props.len() as u32;
                            self.module_props.push(ModulePropRecord {
                                lit: Lit::pos(var),
                                premises: meq.premises,
                            });
                            props.push(TheoryProp {
                                lit: Lit::pos(var),
                                key: MODULE_KEY_OFFSET + key,
                            });
                        }
                        Some(true) => {}
                        Some(false) => {
                            // Trail says t1 ≠ t2, module says t1 = t2.
                            // Conflict clause: negate eq premises + assert diseq as eq.
                            let mut clause = Vec::new();
                            for &(pt1, pt2) in &meq.premises {
                                let k = if pt1 <= pt2 { (pt1, pt2) } else { (pt2, pt1) };
                                if let Some(&aid) = self.euf.atom_map.eq_to_atom.get(&k) {
                                    clause.push(Lit::neg(self.euf.atom_map.var_for_atom(aid)));
                                }
                            }
                            clause.push(Lit::pos(var)); // The equality must hold
                            return TheoryResult::Conflict(clause);
                        }
                    }
                }

                if props.is_empty() {
                    TheoryResult::Consistent
                } else {
                    TheoryResult::Propagate(props)
                }
            }
        }
    }

    fn backtrack(&mut self, new_level: u32) {
        self.euf.backtrack(new_level);
        self.module.backtrack(new_level);
        self.module_trail_pos = 0;
    }

    fn explain(&mut self, lit: Lit, key: u32) -> Vec<Lit> {
        if key >= MODULE_KEY_OFFSET {
            let idx = (key - MODULE_KEY_OFFSET) as usize;
            let record = &self.module_props[idx];
            let mut clause = vec![record.lit];
            for &(t1, t2) in &record.premises {
                let k = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
                if let Some(&atom_id) = self.euf.atom_map.eq_to_atom.get(&k) {
                    clause.push(Lit::neg(self.euf.atom_map.var_for_atom(atom_id)));
                }
            }
            clause
        } else {
            self.euf.explain(lit, key)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bv::BvSolver;
    use crate::formula::SmtFormula;
    use crate::session::SmtEnv;
    use crate::solver::{check_sat_combined, SmtResult};
    use crate::term::{BvOpKind, FuncDecl, FuncId, Sort, SortId, TermArena, TermKind};

    fn t(n: u32) -> TermId {
        TermId(n)
    }

    // ── EUF-only helpers (ConstantModule) ──

    struct ConstantModule {
        equalities: Vec<(TermId, TermId)>,
    }

    impl TheoryModule for ConstantModule {
        fn notify_equality(&mut self, _t1: TermId, _t2: TermId) {}
        fn notify_disequality(&mut self, _t1: TermId, _t2: TermId) {}

        fn propagate(&mut self) -> ModuleResult {
            ModuleResult::Consistent(
                self.equalities
                    .iter()
                    .map(|&(t1, t2)| ModuleEquality {
                        t1,
                        t2,
                        premises: Vec::new(), // Axiomatic
                    })
                    .collect(),
            )
        }

        fn push_level(&mut self) {}
        fn backtrack(&mut self, _level: u32) {}
    }

    /// EUF-only test environment: sort S, f: S → S, a=0, b=1, f(a)=2, f(b)=3.
    fn euf_env(assertions: Vec<SmtFormula>) -> SmtEnv {
        let mut arena = TermArena::new();
        let s = SortId(0);
        let f = FuncId(0);
        let a = arena.intern(
            TermKind::Variable {
                name: "a".into(),
                sort: s,
            },
            s,
        );
        let b = arena.intern(
            TermKind::Variable {
                name: "b".into(),
                sort: s,
            },
            s,
        );
        let _fa = arena.intern(
            TermKind::Apply {
                func: f,
                args: vec![a],
            },
            s,
        );
        let _fb = arena.intern(
            TermKind::Apply {
                func: f,
                args: vec![b],
            },
            s,
        );
        SmtEnv {
            arena,
            sorts: vec![Sort { name: "S".into() }],
            func_decls: vec![FuncDecl {
                name: "f".into(),
                arg_sorts: vec![s],
                ret_sort: s,
            }],
            assertions,
        }
    }

    // ── EUF pass-through tests ──

    #[test]
    fn null_module_passthrough_sat() {
        let env = euf_env(vec![SmtFormula::Eq(t(0), t(1))]);
        assert_eq!(check_sat_combined(env, NullModule), SmtResult::Sat);
    }

    #[test]
    fn null_module_passthrough_unsat() {
        let env = euf_env(vec![SmtFormula::And(vec![
            SmtFormula::Eq(t(0), t(1)),
            SmtFormula::Neq(t(2), t(3)),
        ])]);
        assert_eq!(check_sat_combined(env, NullModule), SmtResult::Unsat);
    }

    #[test]
    fn constant_module_forces_unsat() {
        let env = euf_env(vec![SmtFormula::And(vec![
            SmtFormula::Implies(
                Box::new(SmtFormula::Eq(t(0), t(1))),
                Box::new(SmtFormula::Eq(t(2), t(3))),
            ),
            SmtFormula::Neq(t(2), t(3)),
        ])]);
        let module = ConstantModule {
            equalities: vec![(t(0), t(1))],
        };
        assert_eq!(check_sat_combined(env, module), SmtResult::Unsat);
    }

    #[test]
    fn same_formula_sat_without_module() {
        let env = euf_env(vec![SmtFormula::And(vec![
            SmtFormula::Implies(
                Box::new(SmtFormula::Eq(t(0), t(1))),
                Box::new(SmtFormula::Eq(t(2), t(3))),
            ),
            SmtFormula::Neq(t(2), t(3)),
        ])]);
        assert_eq!(check_sat_combined(env, NullModule), SmtResult::Sat);
    }

    // ── BV cross-theory tests ──

    /// BV test environment:
    ///   0: x (Variable)
    ///   1: y (Variable)
    ///   2: bvconst(5, 3) "three"
    ///   3: bvconst(5, 4) "four"
    ///   4: bvconst(5, 1) "one"
    ///   5: bvadd(5, [x, one])
    fn bv_env(assertions: Vec<SmtFormula>) -> (SmtEnv, Vec<TermKind>) {
        let mut arena = TermArena::new();
        let s = SortId(0);
        let x = arena.intern(
            TermKind::Variable {
                name: "x".into(),
                sort: s,
            },
            s,
        );
        let _y = arena.intern(
            TermKind::Variable {
                name: "y".into(),
                sort: s,
            },
            s,
        );
        let _three = arena.intern(TermKind::BvConst { width: 5, value: 3 }, s);
        let _four = arena.intern(TermKind::BvConst { width: 5, value: 4 }, s);
        let one = arena.intern(TermKind::BvConst { width: 5, value: 1 }, s);
        let _add = arena.intern(
            TermKind::BvOp {
                op: BvOpKind::Add,
                width: 5,
                args: vec![x, one],
            },
            s,
        );
        // Collect term kinds for BvSolver construction
        let kinds: Vec<TermKind> = (0..arena.len())
            .map(|i| arena.get(TermId(i as u32)).kind.clone())
            .collect();
        let env = SmtEnv {
            arena,
            sorts: vec![Sort { name: "BV5".into() }],
            func_decls: Vec::new(),
            assertions,
        };
        (env, kinds)
    }

    /// BV+EUF test environment — adds f: S → S on top of bv_env.
    ///   6: f(bvadd(x, one)) = Apply(f, [5])
    ///   7: f(y) = Apply(f, [1])
    fn bv_euf_env(assertions: Vec<SmtFormula>) -> (SmtEnv, Vec<TermKind>) {
        let mut arena = TermArena::new();
        let s = SortId(0);
        let f = FuncId(0);
        let x = arena.intern(
            TermKind::Variable {
                name: "x".into(),
                sort: s,
            },
            s,
        );
        let y = arena.intern(
            TermKind::Variable {
                name: "y".into(),
                sort: s,
            },
            s,
        );
        let _three = arena.intern(TermKind::BvConst { width: 5, value: 3 }, s);
        let _four = arena.intern(TermKind::BvConst { width: 5, value: 4 }, s);
        let one = arena.intern(TermKind::BvConst { width: 5, value: 1 }, s);
        let add = arena.intern(
            TermKind::BvOp {
                op: BvOpKind::Add,
                width: 5,
                args: vec![x, one],
            },
            s,
        );
        let _f_add = arena.intern(
            TermKind::Apply {
                func: f,
                args: vec![add],
            },
            s,
        );
        let _f_y = arena.intern(
            TermKind::Apply {
                func: f,
                args: vec![y],
            },
            s,
        );
        let kinds: Vec<TermKind> = (0..arena.len())
            .map(|i| arena.get(TermId(i as u32)).kind.clone())
            .collect();
        let env = SmtEnv {
            arena,
            sorts: vec![Sort { name: "BV5".into() }],
            func_decls: vec![FuncDecl {
                name: "f".into(),
                arg_sorts: vec![s],
                ret_sort: s,
            }],
            assertions,
        };
        (env, kinds)
    }

    #[test]
    fn bv_constant_eval_unsat() {
        // x = 3 ∧ y = 4 ∧ bvadd(x, 1) ≠ y
        // Without BV: SAT (bvadd is uninterpreted)
        // With BV: bvadd(3, 1) = 4 = y → conflict with disequality → UNSAT
        let (env, kinds) = bv_env(vec![SmtFormula::And(vec![
            SmtFormula::Eq(t(0), t(2)),  // x = three
            SmtFormula::Eq(t(1), t(3)),  // y = four
            SmtFormula::Neq(t(5), t(1)), // bvadd(x,1) ≠ y
        ])]);
        let module = BvSolver::new(&kinds);
        assert_eq!(check_sat_combined(env, module), SmtResult::Unsat);
    }

    #[test]
    fn bv_same_formula_sat_without_module() {
        // Same formula — SAT when BV doesn't interpret bvadd
        let (env, _) = bv_env(vec![SmtFormula::And(vec![
            SmtFormula::Eq(t(0), t(2)),
            SmtFormula::Eq(t(1), t(3)),
            SmtFormula::Neq(t(5), t(1)),
        ])]);
        assert_eq!(check_sat_combined(env, NullModule), SmtResult::Sat);
    }

    #[test]
    fn bv_euf_congruence_unsat() {
        // x = 3 ∧ y = 4 ∧ f(bvadd(x,1)) ≠ f(y)
        // Purification creates atom (bvadd(x,1), y) from the f-application pair.
        // BV propagates bvadd(x,1) = y → EUF congruence: f(bvadd(x,1)) = f(y) → UNSAT
        let (env, kinds) = bv_euf_env(vec![SmtFormula::And(vec![
            SmtFormula::Eq(t(0), t(2)),  // x = three
            SmtFormula::Eq(t(1), t(3)),  // y = four
            SmtFormula::Neq(t(6), t(7)), // f(bvadd(x,1)) ≠ f(y)
        ])]);
        let module = BvSolver::new(&kinds);
        assert_eq!(check_sat_combined(env, module), SmtResult::Unsat);
    }

    #[test]
    fn bv_euf_same_formula_sat_without_module() {
        // Same formula — SAT without BV (bvadd uninterpreted, pick bvadd(x,1) ≠ y)
        let (env, _) = bv_euf_env(vec![SmtFormula::And(vec![
            SmtFormula::Eq(t(0), t(2)),
            SmtFormula::Eq(t(1), t(3)),
            SmtFormula::Neq(t(6), t(7)),
        ])]);
        assert_eq!(check_sat_combined(env, NullModule), SmtResult::Sat);
    }
}
