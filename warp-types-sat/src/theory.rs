//! Theory solver integration for DPLL(T).
//!
//! Provides the [`TheorySolver`] trait for plugging theory-specific reasoning
//! into the CDCL loop. The SAT solver calls [`TheorySolver::check`] after each
//! BCP fixpoint; the theory can propagate implied literals, report conflicts,
//! or confirm consistency. Conflict analysis calls [`TheorySolver::explain`]
//! lazily — only for theory-propagated literals that participate in conflicts.
//!
//! [`NoTheory`] is a zero-cost no-op implementation for pure SAT solving.
//! When monomorphized with `NoTheory`, the compiler eliminates all theory
//! code paths — no branches, no vtable, no overhead.
//!
//! # DPLL(T) Integration Points
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │  CDCL Loop                                  │
//! │                                             │
//! │  Decision ─► BCP ─► fixpoint ──┐            │
//! │                                │            │
//! │         ┌──────────────────────▼──────────┐ │
//! │         │  theory.check(trail, db)        │ │
//! │         │                                 │ │
//! │         │  Consistent ─► next decision    │ │
//! │         │  Propagate  ─► record on trail  │ │
//! │         │                ─► re-run BCP    │ │
//! │         │  Conflict   ─► add clause to db │ │
//! │         │                ─► resolve       │ │
//! │         └─────────────────────────────────┘ │
//! │                                             │
//! │  Backtrack ─► theory.backtrack(level)        │
//! │                                             │
//! │  Analysis hits TheoryPropagation(key) ──┐   │
//! │         ┌───────────────────────────────▼──┐ │
//! │         │  theory.explain(lit, key)        │ │
//! │         │  → reason clause for resolution  │ │
//! │         └─────────────────────────────────┘ │
//! └─────────────────────────────────────────────┘
//! ```

use crate::bcp::ClauseDb;
use crate::literal::Lit;
use crate::trail::Trail;

/// Read-only view of solver state for theory consistency checks.
///
/// The theory receives this during [`TheorySolver::check`] to inspect the
/// current partial assignment and clause database without mutating solver state.
pub struct TheoryContext<'a> {
    /// The assignment trail — query `trail.value(var)` for assignments,
    /// `trail.current_level()` for the decision level.
    pub trail: &'a Trail,
    /// The clause database (original + learned clauses).
    pub db: &'a ClauseDb,
    /// Number of variables in the problem.
    pub num_vars: u32,
}

/// A theory-implied literal propagation.
pub struct TheoryProp {
    /// The literal implied by the theory.
    pub lit: Lit,
    /// Opaque key identifying this propagation's reason. Passed back to
    /// [`TheorySolver::explain`] during conflict analysis to lazily retrieve
    /// the explanation clause. Theory solvers assign keys to track which
    /// deduction produced each propagation.
    pub key: u32,
}

/// Result of a theory consistency check.
pub enum TheoryResult {
    /// Current partial assignment is consistent with the theory.
    /// The solver proceeds to the next decision.
    Consistent,

    /// Theory can propagate additional literals. Each propagation carries
    /// an opaque key for lazy explanation during conflict analysis.
    ///
    /// After recording these on the trail, the solver re-runs BCP to
    /// propagate consequences, then calls `check` again.
    Propagate(Vec<TheoryProp>),

    /// Current partial assignment is inconsistent with the theory.
    ///
    /// The returned clause must be a valid theory lemma: a disjunction of
    /// literals that is true in every model of the theory but false under
    /// the current assignment. The solver adds it to the clause database
    /// and enters conflict resolution.
    Conflict(Vec<Lit>),
}

/// A theory solver for DPLL(T) integration.
///
/// Implement this trait to plug domain-specific reasoning (equality,
/// arithmetic, bitvectors, arrays, etc.) into the CDCL SAT solver.
///
/// # Contract
///
/// - **`check`** is called after each BCP fixpoint. It must be sound:
///   - `Consistent` only if the partial assignment doesn't violate the theory.
///   - `Conflict(lits)` only if `lits` is a valid theory lemma (true in all
///     theory models) that is currently falsified.
///   - `Propagate(props)` only if each `prop.lit` is theory-implied by the
///     current assignment.
///
/// - **`explain`** is called lazily during conflict analysis. The returned
///   clause must contain `lit` and be unit under the assignment at the time
///   `lit` was propagated (all other literals false, `lit` unresolved).
///
/// - **`backtrack`** is called whenever the solver retracts assignments
///   (conflict-driven backjumping or restart). The theory must retract
///   its internal state to match.
pub trait TheorySolver {
    /// Check consistency of the current partial assignment against the theory.
    fn check(&mut self, ctx: &TheoryContext<'_>) -> TheoryResult;

    /// Retract theory state to match a backtrack to `new_level`.
    ///
    /// All assignments at levels > `new_level` have been retracted from the
    /// trail. The theory must retract any internal state that depended on them.
    fn backtrack(&mut self, new_level: u32);

    /// Produce a reason clause for a theory-propagated literal.
    ///
    /// Called lazily during conflict analysis — only for theory propagations
    /// that participate in the conflict derivation. The returned clause must:
    /// - Contain `lit`
    /// - Be unit under the assignment when `lit` was propagated
    /// - Be a valid theory lemma (true in all models of the theory)
    ///
    /// `key` is the opaque identifier from the original [`TheoryProp`].
    fn explain(&mut self, lit: Lit, key: u32) -> Vec<Lit>;
}

/// Zero-cost no-op theory solver for pure SAT solving.
///
/// All methods are trivial: `check` always returns `Consistent`, `backtrack`
/// is a no-op, and `explain` is unreachable (no theory propagations exist).
///
/// When the CDCL loop is monomorphized with `NoTheory`, the compiler
/// eliminates all theory-related branches — the generated code is identical
/// to the pre-theory solver.
pub struct NoTheory;

impl TheorySolver for NoTheory {
    #[inline(always)]
    fn check(&mut self, _ctx: &TheoryContext<'_>) -> TheoryResult {
        TheoryResult::Consistent
    }

    #[inline(always)]
    fn backtrack(&mut self, _new_level: u32) {}

    #[inline(always)]
    fn explain(&mut self, _lit: Lit, _key: u32) -> Vec<Lit> {
        unreachable!("NoTheory::explain called — pure SAT has no theory propagations")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_theory_is_always_consistent() {
        let mut t = NoTheory;
        let trail = Trail::new(4);
        let db = ClauseDb::new();
        let ctx = TheoryContext {
            trail: &trail,
            db: &db,
            num_vars: 4,
        };
        assert!(matches!(t.check(&ctx), TheoryResult::Consistent));
    }

    #[test]
    fn no_theory_backtrack_is_noop() {
        let mut t = NoTheory;
        t.backtrack(0);
        t.backtrack(42);
    }

    #[test]
    #[should_panic(expected = "NoTheory::explain")]
    fn no_theory_explain_panics() {
        let mut t = NoTheory;
        t.explain(Lit::pos(0), 0);
    }
}
