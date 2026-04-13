//! Top-level SMT solver: wires formula abstraction, EUF theory, and SAT together.
//!
//! The solving pipeline:
//! 1. Tseitin-transform SMT formulas into propositional CNF + atom map
//! 2. Build EUF congruence closure engine with the atom map
//! 3. Call `solve_with_theory()` from `warp_types_sat` (DPLL(T) loop)
//! 4. Interpret the SAT result as an SMT result

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
/// Called by `SmtSession<'s, Asserted>::check_sat()`.
pub(crate) fn check_sat(env: SmtEnv) -> SmtResult {
    // Handle empty assertions: trivially satisfiable
    if env.assertions.is_empty() {
        return SmtResult::Sat;
    }

    // Step 1: Boolean abstraction — Tseitin-transform formulas to CNF
    let abstraction = formula::abstract_formulas(&env.assertions);

    // Step 2: Build EUF theory solver with the atom map
    let mut euf = EufSolver::new(&env.arena, abstraction.atom_map);

    // Step 3: DPLL(T) — the SAT solver drives decisions and BCP,
    // calling euf.check() after each fixpoint, euf.backtrack() on
    // conflict-driven backjumping, and euf.explain() lazily during
    // conflict analysis.
    let result = solve_with_theory(abstraction.db, abstraction.num_vars, &mut euf);

    // Step 4: Interpret
    match result {
        SolveResult::Sat(_) => SmtResult::Sat,
        SolveResult::Unsat => SmtResult::Unsat,
        SolveResult::Unknown => SmtResult::Unknown,
    }
}
