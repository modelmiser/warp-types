//! SMT session with lifetime-branded phase tracking.
//!
//! `SmtSession<'s, P>` follows the pattern from `warp_types_sat::SolverSession`
//! and `warp_types_bmc::BmcSession`:
//! - `'s` is an invariant lifetime brand (prevents cross-session mixing)
//! - `P: Phase` tracks the current SMT workflow phase
//! - Transitions consume the session and produce a new phase
//! - Terminal states (`Sat`, `Unsat`, `Unknown`) have no outgoing transitions
//!
//! Unlike the BMC/SAT sessions which are zero-sized, the SMT session carries
//! mutable state (`SmtEnv`) through phases: sorts accumulate in `Init`,
//! formulas accumulate in `Declared`.

use core::marker::PhantomData;

use crate::phase::*;
use crate::term::{FuncDecl, FuncId, Sort, SortId, TermArena, TermId, TermKind};

// ============================================================================
// Environment (accumulated state across phases)
// ============================================================================

/// Internal environment carried through SMT session phases.
///
/// Accumulates sort/function declarations during `Init` and formulas during
/// `Declared`. Consumed by the solver in `Asserted → check_sat()`.
pub(crate) struct SmtEnv {
    pub(crate) arena: TermArena,
    pub(crate) sorts: Vec<Sort>,
    pub(crate) func_decls: Vec<FuncDecl>,
    pub(crate) assertions: Vec<crate::formula::SmtFormula>,
}

impl SmtEnv {
    fn new() -> Self {
        SmtEnv {
            arena: TermArena::new(),
            sorts: Vec::new(),
            func_decls: Vec::new(),
            assertions: Vec::new(),
        }
    }
}

// ============================================================================
// SMT session
// ============================================================================

/// An SMT session branded with lifetime `'s` and phase `P`.
///
/// Carries the environment (sorts, functions, assertions) through phases.
/// Phase transitions consume the session and produce a new one — invalid
/// transitions are compile errors.
#[must_use = "dropping an SmtSession loses phase tracking — use a transition or terminal"]
pub struct SmtSession<'s, P: Phase> {
    _brand: PhantomData<fn(&'s ()) -> &'s ()>,
    _phase: PhantomData<P>,
    env: SmtEnv,
}

impl<'s, P: Phase> SmtSession<'s, P> {
    pub(crate) fn new(env: SmtEnv) -> Self {
        SmtSession {
            _brand: PhantomData,
            _phase: PhantomData,
            env,
        }
    }

    /// Current phase name.
    pub fn phase_name(&self) -> &'static str {
        P::NAME
    }
}

// ============================================================================
// Phase transitions: Init
// ============================================================================

impl<'s> SmtSession<'s, Init> {
    /// Declare a new uninterpreted sort. Returns the session and the sort ID.
    ///
    /// Affine: consumes and returns the session to maintain move semantics.
    pub fn declare_sort(mut self, name: &str) -> (SmtSession<'s, Init>, SortId) {
        let id = SortId(self.env.sorts.len() as u32);
        self.env.sorts.push(Sort {
            name: name.to_string(),
        });
        (self, id)
    }

    /// Declare an uninterpreted function symbol. Returns the session and the function ID.
    ///
    /// # Arguments
    /// - `name`: function name (e.g. "f")
    /// - `arg_sorts`: argument sort signature
    /// - `ret_sort`: return sort
    pub fn declare_fun(
        mut self,
        name: &str,
        arg_sorts: &[SortId],
        ret_sort: SortId,
    ) -> (SmtSession<'s, Init>, FuncId) {
        let id = FuncId(self.env.func_decls.len() as u32);
        self.env.func_decls.push(FuncDecl {
            name: name.to_string(),
            arg_sorts: arg_sorts.to_vec(),
            ret_sort,
        });
        (self, id)
    }

    /// Create a variable (named constant) in the term arena.
    /// Returns the session and the term ID.
    pub fn var(mut self, name: &str, sort: SortId) -> (SmtSession<'s, Init>, TermId) {
        let id = self.env.arena.intern(
            TermKind::Variable {
                name: name.to_string(),
                sort,
            },
            sort,
        );
        (self, id)
    }

    /// Create a function application term in the arena.
    /// Returns the session and the term ID.
    pub fn apply(mut self, func: FuncId, args: &[TermId]) -> (SmtSession<'s, Init>, TermId) {
        let ret_sort = self.env.func_decls[func.0 as usize].ret_sort;
        let id = self.env.arena.intern(
            TermKind::Apply {
                func,
                args: args.to_vec(),
            },
            ret_sort,
        );
        (self, id)
    }

    /// Finish declarations and move to the assertion phase.
    pub fn finish_declarations(self) -> SmtSession<'s, Declared> {
        SmtSession::new(self.env)
    }
}

// ============================================================================
// Phase transitions: Declared
// ============================================================================

impl<'s> SmtSession<'s, Declared> {
    /// Assert an SMT formula. Consumes and returns the session.
    pub fn assert_formula(
        mut self,
        formula: crate::formula::SmtFormula,
    ) -> SmtSession<'s, Declared> {
        self.env.assertions.push(formula);
        SmtSession::new(self.env)
    }

    /// Finish assertions and move to the solving phase.
    pub fn finish_assertions(self) -> SmtSession<'s, Asserted> {
        SmtSession::new(self.env)
    }
}

// ============================================================================
// Phase transitions: Asserted
// ============================================================================

impl<'s> SmtSession<'s, Asserted> {
    /// Check satisfiability. Consumes the session and returns the result.
    ///
    /// Wires the Boolean abstraction layer, EUF theory solver, and
    /// warp-types-sat's DPLL(T) engine together.
    pub fn check_sat(self) -> crate::solver::SmtResult {
        crate::solver::check_sat(self.env)
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Create an SMT session with a fresh lifetime brand.
///
/// The closure receives an `SmtSession<'s, Init>` with a unique lifetime
/// brand. The brand prevents terms and sessions from different `with_session`
/// calls from being mixed — a compile-time safety net.
///
/// # Example
///
/// ```
/// use warp_types_smt::*;
///
/// let result = with_session(|session| {
///     let (session, s) = session.declare_sort("S");
///     let (session, f) = session.declare_fun("f", &[s], s);
///     let (session, a) = session.var("a", s);
///     let (session, b) = session.var("b", s);
///     let declared = session.finish_declarations();
///     let asserted = declared
///         .assert_formula(SmtFormula::Eq(a, b))
///         .finish_assertions();
///     asserted.check_sat()
/// });
/// ```
pub fn with_session<R>(f: impl for<'s> FnOnce(SmtSession<'s, Init>) -> R) -> R {
    f(SmtSession::new(SmtEnv::new()))
}
