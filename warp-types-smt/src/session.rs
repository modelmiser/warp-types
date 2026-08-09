//! SMT session with lifetime-branded phase tracking.
//!
//! `SmtSession<'s, P>` follows the pattern from `warp_types_sat::SolverSession`
//! and `warp_types_bmc::BmcSession`:
//! - `'s` is an invariant lifetime brand (prevents cross-session mixing of
//!   *session values*; `TermId`/`SortId`/`FuncId` are unbranded — see
//!   [`with_session`] for what is and is not caught)
//! - `P: Phase` tracks the current SMT workflow phase
//! - Transitions consume the session and produce a new phase
//! - Terminal states (`Sat`, `Unsat`, `Unknown`) have no outgoing transitions
//!
//! Unlike the BMC/SAT sessions which are zero-sized, the SMT session carries
//! mutable state (`SmtEnv`) through phases: sorts accumulate in `Init`,
//! formulas accumulate in `Declared`.

use core::marker::PhantomData;

use crate::phase::*;
use crate::term::{BvOpKind, FuncDecl, FuncId, Sort, SortId, TermArena, TermId, TermKind};

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

    /// Bounds-check `TermId` arguments against this session's arena.
    ///
    /// Ids carry no lifetime brand (see [`with_session`]), so an id from a
    /// different session is caught only when it is out of range — but then
    /// it must fail loudly here rather than panic deep in the solver or
    /// silently alias another term.
    fn check_term_ids(&self, entry_point: &str, args: &[TermId]) {
        for &t in args {
            assert!(
                t.index() < self.env.arena.len(),
                "{entry_point}: TermId({}) out of range for this session's arena \
                 (len {}) — TermIds must come from this session",
                t.index(),
                self.env.arena.len()
            );
        }
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
    ///
    /// # Panics
    /// Panics if `func` or any of `args` is out of range for this session
    /// (ids are unbranded — see [`with_session`]).
    pub fn apply(mut self, func: FuncId, args: &[TermId]) -> (SmtSession<'s, Init>, TermId) {
        assert!(
            (func.0 as usize) < self.env.func_decls.len(),
            "apply: FuncId({}) out of range for this session's function table \
             (len {}) — FuncIds must come from this session",
            func.0,
            self.env.func_decls.len()
        );
        self.check_term_ids("apply", args);
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

    /// Create a bitvector constant in the term arena.
    /// Returns the session and the term ID.
    ///
    /// `value` is masked to `width` bits, so `bv_const(5, 34)` and
    /// `bv_const(5, 2)` intern the same term. (Unmasked constants would be
    /// keyed by their raw value in the BV module while evaluated operations
    /// are masked, silently missing equalities and conflicts.)
    ///
    /// # Panics
    /// Panics if `width` is not in `1..=64` (values are `u64`-backed).
    pub fn bv_const(
        mut self,
        width: u32,
        value: u64,
        sort: SortId,
    ) -> (SmtSession<'s, Init>, TermId) {
        assert!(
            (1..=64).contains(&width),
            "bv_const: width ({width}) must be in 1..=64"
        );
        let value = value & crate::bv::width_mask(width);
        let id = self
            .env
            .arena
            .intern(TermKind::BvConst { width, value }, sort);
        (self, id)
    }

    /// Create a bitvector operation term in the arena.
    /// Returns the session and the term ID.
    ///
    /// # Panics
    /// Panics if:
    /// - any of `args` is out of range for this session;
    /// - the arity is wrong for `op` (`Not`/`Extract` are unary, `Sub`/
    ///   `Concat` are binary, `Add`/`And`/`Or`/`Xor` need at least one arg);
    /// - `op` is `Extract { hi, lo }` with `lo > hi`, `hi >= 64` (values are
    ///   `u64`-backed), or `width != hi - lo + 1`.
    pub fn bv_op(
        mut self,
        op: BvOpKind,
        width: u32,
        args: &[TermId],
        sort: SortId,
    ) -> (SmtSession<'s, Init>, TermId) {
        self.check_term_ids("bv_op", args);
        match op {
            BvOpKind::Not => assert!(
                args.len() == 1,
                "bv_op: bvnot is unary (got {} args)",
                args.len()
            ),
            BvOpKind::Sub => assert!(
                args.len() == 2,
                "bv_op: bvsub is binary (got {} args)",
                args.len()
            ),
            BvOpKind::Concat => assert!(
                args.len() == 2,
                "bv_op: concat is binary (got {} args)",
                args.len()
            ),
            BvOpKind::Extract { hi, lo } => {
                assert!(
                    args.len() == 1,
                    "bv_op: Extract is unary (got {} args)",
                    args.len()
                );
                assert!(lo <= hi, "bv_op: Extract lo ({lo}) must be <= hi ({hi})");
                assert!(
                    hi < 64,
                    "bv_op: Extract hi ({hi}) must be < 64 (values are u64-backed)"
                );
                assert!(
                    width == hi - lo + 1,
                    "bv_op: Extract{{hi: {hi}, lo: {lo}}} result width must be {} (got {width})",
                    hi - lo + 1
                );
            }
            BvOpKind::Add | BvOpKind::And | BvOpKind::Or | BvOpKind::Xor => assert!(
                !args.is_empty(),
                "bv_op: {op:?} needs at least one argument"
            ),
        }
        let id = self.env.arena.intern(
            TermKind::BvOp {
                op,
                width,
                args: args.to_vec(),
            },
            sort,
        );
        (self, id)
    }

    /// Extract bits `[hi:lo]` inclusive from a BV term.
    /// Result width is `hi - lo + 1`.
    ///
    /// The caller is responsible for ensuring `t` has enough bits to cover
    /// `hi` (matches the contract of [`bv_op`](Self::bv_op), which trusts
    /// the caller-supplied width).
    ///
    /// # Panics
    /// Panics if `lo > hi`, if `hi >= 64` (values are `u64`-backed, so the
    /// evaluator shifts by `lo`, which must stay below 64), or if `t` is out
    /// of range for this session.
    pub fn bv_extract(
        mut self,
        hi: u32,
        lo: u32,
        t: TermId,
        sort: SortId,
    ) -> (SmtSession<'s, Init>, TermId) {
        assert!(lo <= hi, "bv_extract: lo ({lo}) must be <= hi ({hi})");
        assert!(
            hi < 64,
            "bv_extract: hi ({hi}) must be < 64 (values are u64-backed)"
        );
        self.check_term_ids("bv_extract", &[t]);
        let result_width = hi - lo + 1;
        let id = self.env.arena.intern(
            TermKind::BvOp {
                op: BvOpKind::Extract { hi, lo },
                width: result_width,
                args: vec![t],
            },
            sort,
        );
        (self, id)
    }

    /// Concatenate two BV terms; the first arg becomes the high bits.
    /// Result width is `hi_w + lo_w`. Widths are explicit because the
    /// arena does not carry BV widths for `Variable` terms.
    ///
    /// # Panics
    /// Panics if `hi_w + lo_w > 64` (this implementation uses `u64`
    /// for values), or if either argument is out of range for this session.
    pub fn bv_concat(
        mut self,
        hi_arg: TermId,
        hi_w: u32,
        lo_arg: TermId,
        lo_w: u32,
        sort: SortId,
    ) -> (SmtSession<'s, Init>, TermId) {
        self.check_term_ids("bv_concat", &[hi_arg, lo_arg]);
        let result_width = hi_w + lo_w;
        assert!(
            result_width <= 64,
            "bv_concat: combined width ({result_width}) exceeds 64"
        );
        let id = self.env.arena.intern(
            TermKind::BvOp {
                op: BvOpKind::Concat,
                width: result_width,
                args: vec![hi_arg, lo_arg],
            },
            sort,
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
    /// Check satisfiability with EUF only. Consumes the session.
    ///
    /// Bitvector operations (`BvOp`) are treated as uninterpreted, and so are
    /// `BvConst` terms: EUF sees distinct constants as ordinary distinct
    /// terms and will merge them when equalities force it (`3 = 4` is
    /// EUF-satisfiable). `Sat` is a complete verdict only in EUF semantics;
    /// for formulas containing `BvConst`, use
    /// [`check_sat_bv`](Self::check_sat_bv).
    pub fn check_sat(self) -> crate::solver::SmtResult {
        crate::solver::check_sat(self.env)
    }

    /// Check satisfiability with EUF + bitvector reasoning. Consumes the session.
    ///
    /// The BV theory module evaluates ground bitvector operations and shares
    /// discovered equalities with EUF via Nelson-Oppen combination. Use this
    /// when the formula contains `BvConst` / `BvOp` terms.
    ///
    /// # Completeness boundary (ground-only BV)
    ///
    /// BV reasoning is ground-only: the module evaluates operations once all
    /// their arguments have concrete values, but performs no bit-blasting and
    /// no word-level solving. Consequently:
    ///
    /// - [`SmtResult::Unsat`](crate::SmtResult::Unsat) is always sound.
    /// - [`SmtResult::Sat`](crate::SmtResult::Sat) means **no ground conflict
    ///   was found**; it is a complete "a model exists" verdict only for
    ///   formulas whose BV terms all become ground under the found
    ///   assignment. Canonical incompleteness witness: at width 1,
    ///   `x ≠ 0 ∧ x ≠ 1` returns `Sat` although it is BV-unsatisfiable,
    ///   because nothing forces `x` to a concrete value.
    pub fn check_sat_bv(self) -> crate::solver::SmtResult {
        let kinds: Vec<TermKind> = (0..self.env.arena.len())
            .map(|i| self.env.arena.get(TermId(i as u32)).kind.clone())
            .collect();
        let module = crate::bv::BvSolver::new(&kinds);
        crate::solver::check_sat_combined(self.env, module)
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Create an SMT session with a fresh lifetime brand.
///
/// The closure receives an `SmtSession<'s, Init>` with a unique invariant
/// lifetime brand. The brand prevents *session values* from different
/// `with_session` calls from being mixed (e.g. smuggling one session into
/// another call's closure) — that is a compile error.
///
/// # What the brand does NOT prevent
///
/// [`TermId`], [`SortId`], and [`FuncId`] are plain indices and carry no
/// brand: the type system does not stop you from using an id created in one
/// session inside another. Session entry points that index into the arena or
/// the function table ([`apply`](SmtSession::apply),
/// [`bv_op`](SmtSession::bv_op), [`bv_extract`](SmtSession::bv_extract),
/// [`bv_concat`](SmtSession::bv_concat)) bounds-check their id arguments and
/// panic with a descriptive message when an id is out of range. An id that
/// happens to be in range for both sessions cannot be detected — it silently
/// denotes whatever term the *current* session holds at that index. Do not
/// move ids across sessions.
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
