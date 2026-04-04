//! Solver session with lifetime-branded phase tracking.
//!
//! `SolverSession<'s, P>` is the solver-level analog of
//! `warp_types::fence::GlobalRegion<'r, S>`:
//!
//! - `'s` is an invariant lifetime brand (prevents cross-session mixing)
//! - `P: Phase` tracks the current CDCL phase
//! - Transitions consume the session and produce a new phase
//! - Terminal states (`Sat`, `Unsat`) have no outgoing transitions

use crate::phase::*;
use core::marker::PhantomData;

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
///     let propagate = session.propagate();
///     // propagate: SolverSession<'s, Propagate>
///     // ... run BCP here, then finish ...
///     let outcome = propagate.finish_no_conflict();
///     // outcome: SolverSession<'s, Idle>
///     outcome.sat()
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

    /// Begin propagation directly from Idle (initial BCP or post-restart).
    ///
    /// Use this before the first decision to propagate unit clauses in the
    /// initial clause database, or after a restart to re-propagate.
    pub fn propagate(self) -> SolverSession<'s, Propagate> {
        SolverSession::new()
    }

    /// Declare satisfiable (all variables assigned, no conflict).
    ///
    /// Terminal state — session consumed.
    pub fn sat(self) -> bool {
        true
    }

    /// Declare unsatisfiable (empty clause derived at decision level 0).
    ///
    /// Terminal state — session consumed.
    pub fn unsat(self) -> bool {
        false
    }
}

// --- Decide transitions ---

impl<'s> SolverSession<'s, Decide> {
    /// Enter propagation phase (BCP) after making a decision.
    ///
    /// Returns a `SolverSession<Propagate>` — the caller runs BCP using
    /// `run_bcp()` with this session as the phase proof, then calls
    /// `finish_no_conflict()` or `finish_conflict()` based on the result.
    pub fn propagate(self) -> SolverSession<'s, Propagate> {
        SolverSession::new()
    }
}

// --- Propagate transitions ---

impl<'s> SolverSession<'s, Propagate> {
    /// BCP completed without conflict. Return to Idle for next decision.
    pub fn finish_no_conflict(self) -> SolverSession<'s, Idle> {
        SolverSession::new()
    }

    /// BCP found a conflict. Enter Conflict phase.
    pub fn finish_conflict(self) -> SolverSession<'s, Conflict> {
        SolverSession::new()
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
    /// Re-propagate after backtracking. The learned clause is now unit
    /// at the backtrack level and must be immediately propagated.
    ///
    /// This is NOT optional — CDCL correctness requires propagating
    /// the asserting literal before any new decision.
    pub fn propagate(self) -> SolverSession<'s, Propagate> {
        SolverSession::new()
    }

    /// Declare unsatisfiable (backtracked to level 0 with no remaining decisions).
    ///
    /// Terminal state — session consumed.
    pub fn unsat(self) -> bool {
        false
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
    use crate::phase;

    #[test]
    fn happy_path_sat() {
        let result = with_session(|session| {
            let decide = session.decide();
            let propagate = decide.propagate();
            // BCP finds no conflict
            let idle = propagate.finish_no_conflict();
            idle.sat()
        });
        assert!(result);
    }

    #[test]
    fn conflict_path() {
        let result = with_session(|session| {
            let decide = session.decide();
            let propagate = decide.propagate();
            // BCP finds conflict
            let conflict = propagate.finish_conflict();
            let analyzed = conflict.analyze();
            let bt = analyzed.backtrack();
            // Re-propagate learned clause (CDCL requirement)
            let propagate = bt.propagate();
            let idle = propagate.finish_no_conflict();
            idle.sat()
        });
        assert!(result);
    }

    #[test]
    fn initial_propagation() {
        // BCP before first decision (unit clauses in initial database)
        let result = with_session(|session| {
            let propagate = session.propagate();
            let idle = propagate.finish_no_conflict();
            idle.decide().propagate().finish_no_conflict().sat()
        });
        assert!(result);
    }

    #[test]
    fn backtrack_to_unsat() {
        let result = with_session(|session| {
            let decide = session.decide();
            let propagate = decide.propagate();
            let conflict = propagate.finish_conflict();
            let analyzed = conflict.analyze();
            let bt = analyzed.backtrack();
            // Backtracked to level 0 — UNSAT
            bt.unsat()
        });
        assert!(!result); // unsat returns false
    }

    #[test]
    fn full_cdcl_loop() {
        let result = with_session(|session| {
            // Initial BCP
            let propagate = session.propagate();
            let idle = propagate.finish_no_conflict();

            // Iteration 1: decide → propagate → conflict → analyze → backtrack → re-propagate
            let propagate = idle.decide().propagate();
            let conflict = propagate.finish_conflict();
            let bt = conflict.analyze().backtrack();
            let propagate = bt.propagate(); // re-propagate learned clause
            let idle = propagate.finish_no_conflict();

            // Iteration 2: decide → propagate → no conflict → SAT
            idle.decide().propagate().finish_no_conflict().sat()
        });
        assert!(result);
    }

    #[test]
    fn phase_names() {
        with_session(|session| {
            assert_eq!(session.phase_name(), "idle");
            let session = session.decide();
            assert_eq!(session.phase_name(), "decide");
            let session = session.propagate();
            assert_eq!(session.phase_name(), "propagate");
        });
    }

    #[test]
    fn generic_transition() {
        with_session(|session| {
            let d: SolverSession<'_, phase::Decide> = session.transition();
            let p: SolverSession<'_, phase::Propagate> = d.transition();
            let c: SolverSession<'_, phase::Conflict> = p.transition();
            let a: SolverSession<'_, phase::Analyze> = c.transition();
            let b: SolverSession<'_, phase::Backtrack> = a.transition();
            // Backtrack → Propagate (re-propagate learned clause)
            let p2: SolverSession<'_, phase::Propagate> = b.transition();
            let _idle: SolverSession<'_, phase::Idle> = p2.transition();
        });
    }
}
