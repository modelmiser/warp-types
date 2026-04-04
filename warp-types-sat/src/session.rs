//! Solver session with lifetime-branded phase tracking.
//!
//! `SolverSession<'s, P>` is the solver-level analog of
//! `warp_types::fence::GlobalRegion<'r, S>`:
//!
//! - `'s` is an invariant lifetime brand (prevents cross-session mixing)
//! - `P: Phase` tracks the current CDCL phase
//! - Transitions consume the session and produce a new phase
//! - Terminal states (`Sat`, `Unsat`) have no outgoing transitions

use core::marker::PhantomData;
use crate::phase::*;

// ============================================================================
// Solver session
// ============================================================================

/// A CDCL solver session branded with lifetime `'s` and phase `P`.
///
/// The lifetime `'s` is an identity brand: each `with_session` call introduces
/// a fresh, unnameable lifetime. Sessions from different `with_session` calls
/// have different lifetimes and cannot interact — the compiler rejects it.
///
/// The phase `P` determines which operations are available. Phase transitions
/// consume the session and produce one in the target phase.
///
/// # Zero-sized
///
/// Like `Warp<S>`, this carries no runtime data. It exists purely in the
/// type system. All phase-tracking overhead is compile-time only.
#[must_use = "dropping a SolverSession loses phase tracking — use a transition or terminal"]
pub struct SolverSession<'s, P: Phase> {
    // fn(&'s ()) -> &'s () makes 's invariant (same as GlobalRegion in fence.rs).
    // Cannot be widened or narrowed — prevents cross-session unification.
    _brand: PhantomData<fn(&'s ()) -> &'s ()>,
    _phase: PhantomData<P>,
}

impl<'s, P: Phase> SolverSession<'s, P> {
    /// Create a new session (crate-internal only — prevents forgery).
    pub(crate) fn new() -> Self {
        SolverSession {
            _brand: PhantomData,
            _phase: PhantomData,
        }
    }

    /// Current phase name (for diagnostics).
    pub fn phase_name(&self) -> &'static str {
        P::NAME
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Create a solver session with a fresh lifetime brand.
///
/// This is the only way to obtain a `SolverSession`. The higher-ranked
/// lifetime `for<'s>` ensures `'s` is fresh and unnameable outside the closure.
///
/// ```
/// use warp_types_sat::*;
///
/// let is_sat = with_session(|session| {
///     // session: SolverSession<'s, Idle>
///     let session = session.decide();
///     // session: SolverSession<'s, Decide>
///     let outcome = session.propagate();
///     // outcome: PropagationOutcome<'s>
///     outcome.handle(
///         |idle| {
///             // No conflict — declare SAT
///             idle.sat().is_satisfiable()
///         },
///         |conflict| {
///             // Conflict path — analyze, backtrack, then declare SAT
///             let analyzed = conflict.analyze();
///             let backtracked = analyzed.backtrack();
///             backtracked.resume().sat().is_satisfiable()
///         },
///     )
/// });
/// assert!(is_sat);
/// ```
pub fn with_session<R>(f: impl for<'s> FnOnce(SolverSession<'s, Idle>) -> R) -> R {
    f(SolverSession::new())
}

// ============================================================================
// Phase transitions
// ============================================================================

// --- Idle transitions ---

impl<'s> SolverSession<'s, Idle> {
    /// Begin the decision phase: choose a variable and polarity.
    ///
    /// Consumes Idle, produces Decide.
    pub fn decide(self) -> SolverSession<'s, Decide> {
        SolverSession::new()
    }

    /// Declare satisfiable (all variables assigned, no conflict).
    ///
    /// Terminal state — no further transitions available.
    pub fn sat(self) -> SolverSession<'s, Sat> {
        SolverSession::new()
    }

    /// Declare unsatisfiable (empty clause derived at decision level 0).
    ///
    /// Terminal state — no further transitions available.
    pub fn unsat(self) -> SolverSession<'s, Unsat> {
        SolverSession::new()
    }
}

// --- Decide transitions ---

impl<'s> SolverSession<'s, Decide> {
    /// Begin propagation (BCP) after making a decision.
    ///
    /// Consumes Decide, produces PropagationOutcome — the data-dependent
    /// branch between "no conflict" (→ Idle) and "conflict found" (→ Conflict).
    ///
    /// This is the typed equivalent of DynDiverge for solver phases:
    /// the caller must handle both outcomes.
    pub fn propagate(self) -> PropagationOutcome<'s> {
        // In a real solver, BCP runs here. The outcome determines the phase.
        // For now, default to no-conflict. Real implementations will override
        // this with actual propagation logic.
        PropagationOutcome::Done(SolverSession::new())
    }
}

// --- Conflict transitions ---

impl<'s> SolverSession<'s, Conflict> {
    /// Begin conflict analysis: traverse implication graph, compute 1-UIP,
    /// build learned clause, bump variable activity.
    ///
    /// Consumes Conflict, produces Analyze.
    pub fn analyze(self) -> SolverSession<'s, Analyze> {
        SolverSession::new()
    }
}

// --- Analyze transitions ---

impl<'s> SolverSession<'s, Analyze> {
    /// Backtrack: unwind the trail to the appropriate decision level
    /// and assert the learned clause.
    ///
    /// Consumes Analyze, produces Backtrack.
    pub fn backtrack(self) -> SolverSession<'s, Backtrack> {
        SolverSession::new()
    }
}

// --- Backtrack transitions ---

impl<'s> SolverSession<'s, Backtrack> {
    /// Resume solving after backtracking. Returns to Idle for the next decision.
    ///
    /// Consumes Backtrack, produces Idle.
    pub fn resume(self) -> SolverSession<'s, Idle> {
        SolverSession::new()
    }

    /// Declare unsatisfiable (backtracked to level 0 with no remaining decisions).
    ///
    /// Terminal state — no further transitions available.
    pub fn unsat(self) -> SolverSession<'s, Unsat> {
        SolverSession::new()
    }
}

// ============================================================================
// Terminal states (no outgoing transitions)
// ============================================================================

impl<'s> SolverSession<'s, Sat> {
    /// Extract the result. The session is consumed.
    pub fn is_satisfiable(&self) -> bool {
        true
    }
}

impl<'s> SolverSession<'s, Unsat> {
    /// Extract the result. The session is consumed.
    pub fn is_unsatisfiable(&self) -> bool {
        true
    }
}

// ============================================================================
// Generic transition (for substrate-agnostic code)
// ============================================================================

impl<'s, P: Phase> SolverSession<'s, P> {
    /// Transition to any valid target phase.
    ///
    /// This is the generic version of the specific transition methods.
    /// Useful when writing substrate-agnostic code that parameterizes
    /// over the transition.
    pub fn transition<To: Phase>(self) -> SolverSession<'s, To>
    where
        P: CanTransition<To>,
    {
        SolverSession::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_sat() {
        let result = with_session(|session| {
            let session = session.decide();
            let outcome = session.propagate();
            outcome.handle(
                |idle| idle.sat().is_satisfiable(),
                |conflict| {
                    let analyzed = conflict.analyze();
                    let bt = analyzed.backtrack();
                    bt.resume().sat().is_satisfiable()
                },
            )
        });
        assert!(result);
    }

    #[test]
    fn conflict_path() {
        // Manually construct a conflict outcome to test the conflict path
        let result = with_session(|session| {
            let session = session.decide();
            // Use generic transition to test CanTransition bounds
            let _propagate: SolverSession<'_, Propagate> = session.transition();
            // Simulate conflict found
            let conflict: SolverSession<'_, Conflict> = SolverSession::new();
            let analyzed = conflict.analyze();
            let bt = analyzed.backtrack();
            bt.unsat().is_unsatisfiable()
        });
        assert!(result);
    }

    #[test]
    fn full_cdcl_loop() {
        let result = with_session(|session| {
            // Iteration 1: decide → propagate → no conflict → idle
            let session = session.decide();
            let outcome = session.propagate();
            let idle = outcome.handle(
                |idle| idle,
                |conflict| conflict.analyze().backtrack().resume(),
            );

            // Iteration 2: decide → propagate → (assume no conflict) → sat
            let session = idle.decide();
            let outcome = session.propagate();
            outcome.handle(
                |idle| idle.sat().is_satisfiable(),
                |conflict| {
                    conflict.analyze().backtrack().resume().sat().is_satisfiable()
                },
            )
        });
        assert!(result);
    }

    #[test]
    fn phase_names() {
        with_session(|session| {
            assert_eq!(session.phase_name(), "idle");
            let session = session.decide();
            assert_eq!(session.phase_name(), "decide");
        });
    }

    #[test]
    fn generic_transition() {
        with_session(|session| {
            // All of these use the generic transition method
            let d: SolverSession<'_, Decide> = session.transition();
            let p: SolverSession<'_, Propagate> = d.transition();
            let c: SolverSession<'_, Conflict> = p.transition();
            let a: SolverSession<'_, Analyze> = c.transition();
            let b: SolverSession<'_, Backtrack> = a.transition();
            let _idle: SolverSession<'_, Idle> = b.transition();
        });
    }
}
