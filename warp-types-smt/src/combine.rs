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
//! 1. EUF processes the trail until fixpoint (conflict / propagation / consistent)
//! 2. Once consistent, the combiner shares trail equalities to the module
//! 3. The module reports any new equalities it discovered
//! 4. The combiner propagates them to the SAT solver as [`TheoryProp`]s
//! 5. The SAT solver records them on the trail, re-runs BCP, calls `check()` again
//! 6. EUF now sees the module's equalities on the trail and processes them
//!
//! This continues until both theories reach fixpoint or a conflict is found.
//!
//! # Limitations (v0.1)
//!
//! - Module equalities must correspond to existing atoms in the formula.
//!   Full Nelson-Oppen requires *purification* (introducing interface atoms
//!   for all shared variable pairs) — a formula-level transform.
//! - Module propagations are currently explained as axiomatic (unit clauses).
//!   A full implementation would trace through module-internal reasoning.
//! - Convex variant only: no case-split enumeration for non-convex theories
//!   like QF_BV (the non-convex extension enumerates disjunctions of equalities).

use crate::euf::EufSolver;
use crate::term::TermId;

use warp_types_sat::literal::Lit;
use warp_types_sat::theory::{TheoryContext, TheoryProp, TheoryResult, TheorySolver};

// ============================================================================
// Theory module trait
// ============================================================================

/// Result of a theory module's consistency check.
pub enum ModuleResult {
    /// Module is consistent. May report new equalities over shared variables
    /// to be communicated to other theories.
    Consistent(Vec<(TermId, TermId)>),
    /// Module detected an inconsistency in its current assertions.
    Conflict,
}

/// A theory module for Nelson-Oppen combination.
///
/// Unlike [`TheorySolver`] (the SAT-facing interface that scans the trail),
/// a `TheoryModule` receives equality and disequality assertions directly
/// from the combining solver. It reasons within its own theory and reports
/// implied equalities for sharing with other theories.
///
/// # Contract
///
/// - `notify_equality` / `notify_disequality`: inform the module of ground
///   truth from the SAT trail. Must be idempotent (re-asserting a known
///   fact after backtrack + replay is a no-op).
/// - `propagate`: check consistency and return new equalities. Returning
///   previously-reported equalities is allowed (the combiner deduplicates).
/// - `backtrack` + `push_level`: support level-based undo for CDCL.
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
/// module propagations use `MODULE_KEY_OFFSET..`. This avoids a tagged enum
/// in the hot path of conflict analysis.
const MODULE_KEY_OFFSET: u32 = 1 << 24;

/// Record for lazily explaining a module-originated propagation.
struct ModulePropRecord {
    /// The propagated literal.
    lit: Lit,
}

/// Nelson-Oppen combining solver.
///
/// Wraps an [`EufSolver`] and a [`TheoryModule`], implementing
/// [`TheorySolver`] for the SAT backbone. Equality sharing between EUF
/// and the module is mediated through SAT-level propagations: when the
/// module discovers `t1 = t2`, the combiner tells the SAT solver, which
/// records it on the trail. On the next `check()`, EUF picks it up
/// naturally — no cross-theory mutation needed.
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
    ///
    /// Scans trail entries the module hasn't seen yet and translates
    /// SAT assignments into theory-level notifications.
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
}

impl<M: TheoryModule> TheorySolver for CombiningSolver<M> {
    fn check(&mut self, ctx: &TheoryContext<'_>) -> TheoryResult {
        // ── Phase 1: EUF processes the trail ──
        //
        // EUF handles equality/disequality assertions from the trail,
        // updates congruence closure, checks for disequality violations,
        // and returns any implied equalities as propagations.
        let euf_result = self.euf.check(ctx);
        match &euf_result {
            TheoryResult::Conflict(_) | TheoryResult::Propagate(_) => {
                return euf_result;
            }
            TheoryResult::Consistent => {}
        }

        // ── Phase 2: Share trail to module ──
        //
        // EUF is at fixpoint. Notify the module of all equality/disequality
        // atoms from the trail that it hasn't processed yet.
        self.share_trail_to_module(ctx);

        // ── Phase 3: Module consistency + equality sharing ──
        //
        // Ask the module for new equalities. For each one:
        // - If already known to EUF → skip
        // - If no SAT atom exists → skip (needs purification)
        // - If SAT var unassigned → propagate to SAT solver
        // - If SAT var true → skip (already consistent)
        // - If SAT var false → conflict (module axiom contradicts trail)
        match self.module.propagate() {
            ModuleResult::Conflict => {
                // Module-only conflict. A full implementation would have
                // the module return involved atoms for clause construction.
                // Conservative: return Consistent (sound but incomplete).
                TheoryResult::Consistent
            }
            ModuleResult::Consistent(new_eqs) => {
                let mut props = Vec::new();

                for (t1, t2) in new_eqs {
                    // Already in the same equivalence class?
                    if self.euf.find(t1) == self.euf.find(t2) {
                        continue;
                    }

                    // Look up the SAT atom for this equality
                    let canonical = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
                    let atom_id = match self.euf.atom_map.eq_to_atom.get(&canonical) {
                        Some(&id) => id,
                        None => continue, // No atom — needs purification
                    };
                    let var = self.euf.atom_map.var_for_atom(atom_id);

                    match ctx.trail.value(var) {
                        None => {
                            // Unassigned — propagate to SAT solver.
                            // EUF will see it on the trail next check().
                            let key = self.module_props.len() as u32;
                            self.module_props
                                .push(ModulePropRecord { lit: Lit::pos(var) });
                            props.push(TheoryProp {
                                lit: Lit::pos(var),
                                key: MODULE_KEY_OFFSET + key,
                            });
                        }
                        Some(true) => {
                            // Already true — consistent, nothing to do.
                        }
                        Some(false) => {
                            // Trail says t1 ≠ t2, module says t1 = t2.
                            // Theory conflict: unit clause asserting the equality.
                            return TheoryResult::Conflict(vec![Lit::pos(var)]);
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
        // Reset trail position — module will be re-notified of surviving
        // equalities when share_trail_to_module runs on the next check().
        self.module_trail_pos = 0;
    }

    fn explain(&mut self, lit: Lit, key: u32) -> Vec<Lit> {
        if key >= MODULE_KEY_OFFSET {
            // Module propagation — currently axiomatic (unit clause).
            // The module asserted this equality unconditionally.
            // A full implementation would have the module provide premise
            // atoms for a proper explanation chain.
            let idx = (key - MODULE_KEY_OFFSET) as usize;
            vec![self.module_props[idx].lit]
        } else {
            // EUF propagation — delegate to EUF's BFS explanation.
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
    use crate::formula::SmtFormula;
    use crate::session::SmtEnv;
    use crate::solver::{check_sat_combined, SmtResult};
    use crate::term::{FuncDecl, FuncId, Sort, SortId, TermArena, TermKind};

    // ── Test module ──

    /// Theory module that unconditionally asserts pre-configured equalities.
    ///
    /// On every `propagate()` call, it reports the same equalities. The
    /// combining solver deduplicates (skips those already known to EUF or
    /// already assigned on the trail).
    struct ConstantModule {
        equalities: Vec<(TermId, TermId)>,
    }

    impl TheoryModule for ConstantModule {
        fn notify_equality(&mut self, _t1: TermId, _t2: TermId) {}
        fn notify_disequality(&mut self, _t1: TermId, _t2: TermId) {}

        fn propagate(&mut self) -> ModuleResult {
            ModuleResult::Consistent(self.equalities.clone())
        }

        fn push_level(&mut self) {}
        fn backtrack(&mut self, _level: u32) {}
    }

    // ── Helpers ──

    /// Build a test environment: sort S, function f: S → S.
    /// Terms interned in order: a=0, b=1, f(a)=2, f(b)=3.
    fn test_env(assertions: Vec<SmtFormula>) -> SmtEnv {
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

    fn t(n: u32) -> TermId {
        TermId(n)
    }

    // ── Pass-through tests (NullModule = bare EUF) ──

    #[test]
    fn null_module_passthrough_sat() {
        let env = test_env(vec![SmtFormula::Eq(t(0), t(1))]);
        assert_eq!(check_sat_combined(env, NullModule), SmtResult::Sat);
    }

    #[test]
    fn null_module_passthrough_unsat() {
        // a = b ∧ f(a) ≠ f(b) — UNSAT by congruence
        let env = test_env(vec![SmtFormula::And(vec![
            SmtFormula::Eq(t(0), t(1)),
            SmtFormula::Neq(t(2), t(3)),
        ])]);
        assert_eq!(check_sat_combined(env, NullModule), SmtResult::Unsat);
    }

    // ── Equality sharing tests ──

    #[test]
    fn module_equality_forces_unsat() {
        // Formula: (a = b → f(a) = f(b)) ∧ f(a) ≠ f(b)
        // Without module: SAT (a ≠ b satisfies implication vacuously)
        // With ConstantModule(a = b): UNSAT
        //   Module propagates a = b → EUF congruence gives f(a) = f(b)
        //   → conflicts with f(a) ≠ f(b)
        let env = test_env(vec![SmtFormula::And(vec![
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
        // Same formula — SAT when no module forces a = b
        let env = test_env(vec![SmtFormula::And(vec![
            SmtFormula::Implies(
                Box::new(SmtFormula::Eq(t(0), t(1))),
                Box::new(SmtFormula::Eq(t(2), t(3))),
            ),
            SmtFormula::Neq(t(2), t(3)),
        ])]);
        assert_eq!(check_sat_combined(env, NullModule), SmtResult::Sat);
    }

    #[test]
    fn module_equality_consistent_with_formula() {
        // Formula: a = b. Module: also says a = b. Redundant — still SAT.
        let env = test_env(vec![SmtFormula::Eq(t(0), t(1))]);
        let module = ConstantModule {
            equalities: vec![(t(0), t(1))],
        };
        assert_eq!(check_sat_combined(env, module), SmtResult::Sat);
    }

    #[test]
    fn module_equality_already_on_trail() {
        // Formula: a = b ∧ f(a) = f(b). Module: a = b (redundant).
        // SAT — module adds nothing new.
        let env = test_env(vec![SmtFormula::And(vec![
            SmtFormula::Eq(t(0), t(1)),
            SmtFormula::Eq(t(2), t(3)),
        ])]);
        let module = ConstantModule {
            equalities: vec![(t(0), t(1))],
        };
        assert_eq!(check_sat_combined(env, module), SmtResult::Sat);
    }
}
