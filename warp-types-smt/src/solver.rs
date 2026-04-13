//! Top-level SMT solver: wires formula abstraction, EUF theory, and SAT together.
//!
//! The solving pipeline:
//! 1. Tseitin-transform SMT formulas into propositional CNF + atom map
//! 2. Build EUF congruence closure engine with the atom map
//! 3. Call `solve_with_theory()` from `warp_types_sat` (DPLL(T) loop)
//! 4. Interpret the SAT result as an SMT result

use crate::combine::{CombiningSolver, NullModule, TheoryModule};
use crate::euf::EufSolver;
use crate::formula;
use crate::session::SmtEnv;

use warp_types_sat::solver::{solve_with_theory, SolveResult};

/// Result of an SMT satisfiability check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtResult {
    /// Satisfiable — a model exists.
    Sat,
    /// Unsatisfiable — no model exists.
    Unsat,
    /// Solver budget exhausted without conclusive result.
    Unknown,
}

/// Run the full SMT solving pipeline on the accumulated environment.
///
/// Called by `SmtSession<'s, Asserted>::check_sat()`. Uses [`NullModule`]
/// as the second theory (zero overhead — compiles identical to bare EUF).
pub(crate) fn check_sat(env: SmtEnv) -> SmtResult {
    check_sat_combined(env, NullModule)
}

/// Run the SMT pipeline with a custom theory module for Nelson-Oppen combination.
///
/// The module receives equality/disequality notifications from the trail
/// and can propagate new equalities back through the combining solver.
/// See [`CombiningSolver`] for the full protocol.
pub(crate) fn check_sat_combined<M: TheoryModule>(env: SmtEnv, module: M) -> SmtResult {
    if env.assertions.is_empty() {
        return SmtResult::Sat;
    }

    // Step 1: Boolean abstraction — Tseitin-transform formulas to CNF
    let abstraction = formula::abstract_formulas(&env.assertions);

    // Step 2: Build theory solvers
    let euf = EufSolver::new(&env.arena, abstraction.atom_map);
    let mut combiner = CombiningSolver::new(euf, module);

    // Step 3: DPLL(T) — the SAT solver drives the combining solver,
    // which mediates between EUF and the theory module via equality sharing.
    let result = solve_with_theory(abstraction.db, abstraction.num_vars, &mut combiner);

    match result {
        SolveResult::Sat(_) => SmtResult::Sat,
        SolveResult::Unsat => SmtResult::Unsat,
        SolveResult::Unknown => SmtResult::Unknown,
    }
}
