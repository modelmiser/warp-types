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
//! protocol, with one bounded internal loop:
//!
//! 1. EUF processes the trail until fixpoint
//! 2. Once consistent, the combiner shares trail equalities to the module
//! 3. The module reports new equalities (with premises) or conflicts
//! 4. The combiner propagates equalities to the SAT solver, or returns
//!    conflict clauses constructed from the module's premises
//! 5. The SAT solver records propagations, re-runs BCP, calls `check()` again
//!
//! Module equalities whose pair has no SAT atom cannot ride this protocol
//! (they produce no trail activity), so `check()` asserts them into EUF
//! directly as derived merges and iterates steps 1–4 internally until no
//! atomless equality joins two distinct classes. The iteration is bounded
//! by the number of EUF equivalence classes.

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
    /// The module is informed that `t1 = t2`, asserted as a trail atom.
    fn notify_equality(&mut self, t1: TermId, t2: TermId);

    /// The module is informed that `t1 = t2` was *derived* by another theory
    /// (EUF congruence/transitivity over shared terms). The pair itself need
    /// not be a trail atom; `premises` are the trail equality atoms that
    /// justify it, and the module must use those (not the pair) when the
    /// equality contributes to a conflict or an explanation.
    fn notify_derived_equality(&mut self, t1: TermId, t2: TermId, premises: Vec<(TermId, TermId)>);

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
    fn notify_derived_equality(
        &mut self,
        _t1: TermId,
        _t2: TermId,
        _premises: Vec<(TermId, TermId)>,
    ) {
    }

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
///
/// The partition bounds both sides:
/// - EUF may hand out at most `2^24` propagation keys per solve
///   (`prop_records` grows monotonically and is never truncated). The EUF
///   side enforces this with a release assert at key handout — without it,
///   key `2^24` would silently alias into module key space and `explain`
///   would decode it against the wrong record table.
/// - The module side has `2^32 - 2^24` keys before `MODULE_KEY_OFFSET + key`
///   wraps. That bound is unreachable in practice (each record carries a
///   heap-allocated premise list, so memory exhausts orders of magnitude
///   earlier), which is why only the EUF side carries the assert.
pub(crate) const MODULE_KEY_OFFSET: u32 = 1 << 24;

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
    /// Decision level the module has been told about via `push_level`.
    /// Kept in lockstep with the trail entries dispatched to the module so
    /// that `backtrack` unwinds exactly the state recorded above the target
    /// level — otherwise retracted values persist and the module emits
    /// conflict/explanation clauses that are not currently falsified.
    module_level: u32,
    /// EUF merge records already forwarded to the module (cursor into
    /// `euf.merge_reasons`). Reset to 0 on backtrack, like `module_trail_pos`.
    module_merge_pos: usize,
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
            module_level: 0,
            module_merge_pos: 0,
            module_props: Vec::new(),
        }
    }

    /// Dispatch trail equalities/disequalities to the module.
    fn share_trail_to_module(&mut self, ctx: &TheoryContext<'_>) {
        let entries = ctx.trail.entries();
        let trail_len = ctx.trail.len();

        for entry in entries.iter().take(trail_len).skip(self.module_trail_pos) {
            // Push module levels so its undo marks mirror the trail's
            // decision levels (entries arrive in nondecreasing level order).
            while self.module_level < entry.level {
                self.module.push_level();
                self.module_level += 1;
            }

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

    /// Forward newly recorded EUF congruence merges to the module
    /// (Nelson-Oppen equality sharing in the EUF → module direction).
    ///
    /// Trail-asserted merges are skipped — `share_trail_to_module` already
    /// delivers those. Congruence merges have no trail atom, so without this
    /// the module never learns that e.g. `f(a) = f(b)` follows from `a = b`,
    /// and misses conflicts between the values it assigns the two terms.
    /// Each forwarded merge carries its premises (the asserted equality atoms
    /// from the EUF explanation) so the module can build valid clauses.
    fn share_euf_merges_to_module(&mut self) {
        let merge_count = self.euf.merge_count();
        while self.module_merge_pos < merge_count {
            let idx = self.module_merge_pos;
            self.module_merge_pos += 1;
            let Some((t1, t2)) = self.euf.congruence_merge_at(idx) else {
                continue;
            };
            let premises: Vec<(TermId, TermId)> = self
                .euf
                .explain_equality(t1, t2)
                .into_iter()
                .map(|atom| {
                    let var = self.euf.atom_map.var_for_atom(atom);
                    self.euf
                        .atom_map
                        .atom_for_var(var)
                        .expect("equality atom's SAT variable must map back to its term pair")
                })
                .collect();
            self.module.notify_derived_equality(t1, t2, premises);
        }
    }

    /// Equality atom for a premise pair. Panics if the pair has no atom:
    /// every premise a module reports must be a trail equality atom —
    /// silently dropping it would strengthen the clause into a
    /// theory-invalid lemma (and a fully dropped conflict would read as
    /// consistent). This is a `TheoryModule` contract requirement.
    fn atom_for_premise(&self, t1: TermId, t2: TermId) -> crate::formula::AtomId {
        let key = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        match self.euf.atom_map.eq_to_atom.get(&key) {
            Some(&atom_id) => atom_id,
            None => panic!(
                "TheoryModule premise ({t1:?}, {t2:?}) has no SAT atom — every premise in a \
                 module conflict/equality must be a trail equality atom (derived equalities \
                 must be reported through their own atom-level premises); dropping it would \
                 produce an invalid stronger lemma"
            ),
        }
    }

    /// SAT variable for a premise pair (see [`Self::atom_for_premise`]).
    fn var_for_premise(&self, t1: TermId, t2: TermId) -> u32 {
        self.euf
            .atom_map
            .var_for_atom(self.atom_for_premise(t1, t2))
    }

    /// Build a conflict clause from module premises.
    ///
    /// Equality premises are negated (they're true on the trail),
    /// disequality premises become positive (they're false on the trail).
    ///
    /// # Panics
    /// Panics if any premise has no SAT atom, or if the module reported a
    /// conflict with no premises at all (see [`Self::var_for_premise`]).
    fn build_conflict_clause(
        &self,
        eq_premises: &[(TermId, TermId)],
        diseq_premises: &[(TermId, TermId)],
    ) -> Vec<Lit> {
        // Sort + dedup at the clause construction boundary (cold path): the
        // TheoryModule contract does not guarantee premise lists free of
        // duplicates, and a duplicate literal in an installed clause puts
        // both watch slots on the same literal.
        let mut eq_premises = eq_premises.to_vec();
        eq_premises.sort();
        eq_premises.dedup();
        let mut diseq_premises = diseq_premises.to_vec();
        diseq_premises.sort();
        diseq_premises.dedup();
        let mut clause = Vec::new();
        for &(t1, t2) in &eq_premises {
            clause.push(Lit::neg(self.var_for_premise(t1, t2)));
        }
        for &(t1, t2) in &diseq_premises {
            clause.push(Lit::pos(self.var_for_premise(t1, t2)));
        }
        assert!(
            !clause.is_empty(),
            "TheoryModule reported a conflict with no premises — a conflict must cite the \
             trail assertions it depends on, or it cannot be expressed as a falsified lemma"
        );
        clause
    }
}

impl<M: TheoryModule> TheorySolver for CombiningSolver<M> {
    fn check(&mut self, ctx: &TheoryContext<'_>) -> TheoryResult {
        // Equality-sharing loop. Module equalities whose pair HAS a SAT atom
        // are routed through SAT propagation (the SAT solver records them,
        // re-runs BCP, and calls check() again — no loop needed here). But a
        // module equality whose pair has no atom produces no trail activity
        // at all, so nothing would re-trigger check(): it must be asserted
        // into EUF directly (as a derived merge carrying its premise chain)
        // and EUF re-checked within THIS call. The loop is bounded: it
        // repeats only when a derived merge actually joined two distinct EUF
        // classes, and the class count is finite (≤ term count).
        loop {
            // ── Phase 1: EUF processes the trail (and any derived merges) ──
            let euf_result = self.euf.check(ctx);
            match &euf_result {
                TheoryResult::Conflict(_) | TheoryResult::Propagate(_) => {
                    return euf_result;
                }
                TheoryResult::Consistent => {}
            }

            // ── Phase 2: Share trail + EUF-derived merges to module ──
            self.share_trail_to_module(ctx);
            self.share_euf_merges_to_module();

            // ── Phase 3: Module consistency + equality sharing ──
            match self.module.propagate() {
                ModuleResult::Conflict {
                    eq_premises,
                    diseq_premises,
                } => {
                    return TheoryResult::Conflict(
                        self.build_conflict_clause(&eq_premises, &diseq_premises),
                    )
                }
                ModuleResult::Consistent(new_eqs) => {
                    let mut props = Vec::new();
                    let mut derived_merge = false;

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
                            None => {
                                // No SAT atom for this pair (purification only
                                // creates atoms for argument pairs of matching
                                // applications). Assert it into EUF as a
                                // derived merge — dropping it here was the
                                // round-3 finding-A unsoundness (EUF never
                                // learned module equalities like u = v, so
                                // congruence f(u) ~ f(v) and the downstream
                                // conflict were missed). Premises must be
                                // trail atoms (atom_for_premise panics
                                // otherwise), so explanations crossing this
                                // merge stay expressible.
                                let premise_atoms = meq
                                    .premises
                                    .iter()
                                    .map(|&(pt1, pt2)| self.atom_for_premise(pt1, pt2))
                                    .collect();
                                self.euf.merge_derived(meq.t1, meq.t2, premise_atoms);
                                derived_merge = true;
                                continue;
                            }
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
                                // Premise atoms are mandatory (var_for_premise
                                // panics on a missing one — silently dropping a
                                // premise would strengthen the lemma unsoundly).
                                let mut premises = meq.premises.clone();
                                premises.sort();
                                premises.dedup();
                                let mut clause: Vec<Lit> = premises
                                    .iter()
                                    .map(|&(pt1, pt2)| Lit::neg(self.var_for_premise(pt1, pt2)))
                                    .collect();
                                clause.push(Lit::pos(var)); // The equality must hold
                                return TheoryResult::Conflict(clause);
                            }
                        }
                    }

                    if !props.is_empty() {
                        // Derived merges (if any) are already recorded in EUF;
                        // the next check() call continues from them.
                        return TheoryResult::Propagate(props);
                    }
                    if !derived_merge {
                        return TheoryResult::Consistent;
                    }
                    // A derived merge joined two EUF classes without any SAT
                    // activity — re-run EUF (conflict/propagation scan over
                    // the new merges) and re-share before concluding.
                }
            }
        }
    }

    fn backtrack(&mut self, new_level: u32) {
        self.euf.backtrack(new_level);
        self.module.backtrack(new_level);
        if self.module_level > new_level {
            self.module_level = new_level;
        }
        self.module_trail_pos = 0;
        // EUF truncated its merge records; re-forward the survivors after the
        // next trail re-scan (same rationale as module_trail_pos).
        self.module_merge_pos = 0;
    }

    fn explain(&mut self, lit: Lit, key: u32) -> Vec<Lit> {
        if key >= MODULE_KEY_OFFSET {
            let idx = (key - MODULE_KEY_OFFSET) as usize;
            let record = &self.module_props[idx];
            // Sort + dedup at the explanation construction boundary — same
            // rationale as `build_conflict_clause`.
            let mut premises = record.premises.clone();
            premises.sort();
            premises.dedup();
            let mut clause = vec![record.lit];
            for (t1, t2) in premises {
                // Missing premise atoms panic (see var_for_premise): dropping
                // one would yield an explanation clause that was never unit.
                clause.push(Lit::neg(self.var_for_premise(t1, t2)));
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
        fn notify_derived_equality(
            &mut self,
            _t1: TermId,
            _t2: TermId,
            _premises: Vec<(TermId, TermId)>,
        ) {
        }
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

    // ── Finding C: module state must unwind on backtrack ──

    /// Drives the TheorySolver interface directly through a decision +
    /// backtrack cycle. A value learned from a level-1 equality must be
    /// retracted when the solver backtracks to level 0; otherwise the module
    /// keeps proposing equalities whose premises are no longer on the trail
    /// (conflict/explanation clauses that are not currently falsified —
    /// a TheorySolver contract violation).
    #[test]
    fn module_state_unwinds_on_backtrack() {
        use crate::formula::AtomMap;
        use warp_types_sat::bcp::ClauseDb;
        use warp_types_sat::trail::Trail;

        let mut arena = TermArena::new();
        let s = SortId(0);
        let x = arena.intern(
            TermKind::Variable {
                name: "x".into(),
                sort: s,
            },
            s,
        );
        let three = arena.intern(TermKind::BvConst { width: 5, value: 3 }, s);
        let four = arena.intern(TermKind::BvConst { width: 5, value: 4 }, s);
        let one = arena.intern(TermKind::BvConst { width: 5, value: 1 }, s);
        let add = arena.intern(
            TermKind::BvOp {
                op: BvOpKind::Add,
                width: 5,
                args: vec![x, one],
            },
            s,
        );
        let kinds: Vec<TermKind> = (0..arena.len())
            .map(|i| arena.get(TermId(i as u32)).kind.clone())
            .collect();

        let mut atom_map = AtomMap::new();
        let (_, v_x_three) = atom_map.get_or_create(x, three); // var 0
        let (_, _v_add_four) = atom_map.get_or_create(add, four); // var 1

        let euf = EufSolver::new(&arena, atom_map);
        let bv = BvSolver::new(&kinds);
        let mut comb = CombiningSolver::new(euf, bv);

        let db = ClauseDb::new();
        let mut trail = Trail::new(2);

        // Decision level 1: x = 3. BV evaluates bvadd(x,1) = 4 and proposes
        // the equality (add, four).
        trail.new_decision(Lit::pos(v_x_three));
        let ctx = TheoryContext {
            trail: &trail,
            db: &db,
            num_vars: 2,
        };
        let r = comb.check(&ctx);
        assert!(
            matches!(r, TheoryResult::Propagate(_)),
            "level-1 check should propagate (add = four)"
        );

        // Backtrack to level 0: the decision is retracted.
        trail.backtrack_to(0);
        comb.backtrack(0);

        // With module state correctly unwound nothing is known any more:
        // the check must be Consistent with no propagations. (Pre-fix, the
        // stale x=3 / add=4 values survive and the module proposes x = three
        // out of thin air.)
        let ctx = TheoryContext {
            trail: &trail,
            db: &db,
            num_vars: 2,
        };
        match comb.check(&ctx) {
            TheoryResult::Consistent => {}
            TheoryResult::Propagate(props) => panic!(
                "spurious propagation after backtrack: {:?} — stale module state",
                props.iter().map(|p| p.lit).collect::<Vec<_>>()
            ),
            TheoryResult::Conflict(c) => {
                panic!("spurious conflict after backtrack: {c:?}")
            }
        }
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
