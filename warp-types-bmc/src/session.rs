//! BMC session with lifetime-branded phase tracking.
//!
//! `BmcSession<'s, P>` follows the same pattern as `warp_types_sat::SolverSession`:
//! - `'s` is an invariant lifetime brand (prevents cross-session mixing)
//! - `P: Phase` tracks the current BMC phase
//! - Transitions consume the session and produce a new phase
//! - Terminal states (`Counterexample`, `Exhausted`) have no outgoing transitions

use crate::phase::*;
use core::marker::PhantomData;

// ============================================================================
// BMC session
// ============================================================================

/// A BMC session branded with lifetime `'s` and phase `P`.
///
/// Zero-sized — all phase tracking is compile-time only.
#[must_use = "dropping a BmcSession loses phase tracking — use a transition or terminal"]
pub struct BmcSession<'s, P: Phase> {
    _brand: PhantomData<fn(&'s ()) -> &'s ()>,
    _phase: PhantomData<P>,
    depth: u32,
}

impl<'s, P: Phase> BmcSession<'s, P> {
    pub(crate) fn new(depth: u32) -> Self {
        BmcSession {
            _brand: PhantomData,
            _phase: PhantomData,
            depth,
        }
    }

    /// Current phase name.
    pub fn phase_name(&self) -> &'static str {
        P::NAME
    }

    /// Current unroll depth.
    pub fn depth(&self) -> u32 {
        self.depth
    }
}

// ============================================================================
// Phase transitions
// ============================================================================

impl<'s> BmcSession<'s, Init> {
    /// Load a transition system model. Consumes Init, produces Modeled.
    pub fn build_model(self) -> BmcSession<'s, Modeled> {
        BmcSession::new(0)
    }
}

impl<'s> BmcSession<'s, Modeled> {
    /// Unroll the transition relation one step (depth 0 = initial state only).
    pub fn unroll(self) -> BmcSession<'s, Unrolled> {
        BmcSession::new(0)
    }
}

impl<'s> BmcSession<'s, Unrolled> {
    /// Encode the safety property at the current depth.
    /// Adds "bad state reachable at frame k?" as a SAT query.
    pub fn encode_property(self) -> BmcSession<'s, Encoded> {
        BmcSession::new(self.depth)
    }
}

impl<'s> BmcSession<'s, Encoded> {
    /// Check the encoded property via SAT. Returns the check result.
    ///
    /// This is where warp-types-sat is called as an oracle.
    pub fn check_sat(self) -> BmcSession<'s, Safe> {
        // Placeholder — real implementation dispatches to SAT solver
        BmcSession::new(self.depth)
    }

    /// Check and return counterexample if SAT.
    pub fn check_counterexample(self) -> BmcSession<'s, Counterexample> {
        BmcSession::new(self.depth)
    }

    /// Check and return exhausted if budget exceeded.
    pub fn check_exhausted(self) -> BmcSession<'s, Exhausted> {
        BmcSession::new(self.depth)
    }
}

impl<'s> BmcSession<'s, Safe> {
    /// Deepen: unroll one more step and try again.
    pub fn deepen(self) -> BmcSession<'s, Unrolled> {
        BmcSession::new(self.depth + 1)
    }

    /// Accept bounded safety at the current depth.
    pub fn accept(self) -> u32 {
        self.depth
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Create a BMC session with a fresh lifetime brand.
pub fn with_session<R>(f: impl for<'s> FnOnce(BmcSession<'s, Init>) -> R) -> R {
    f(BmcSession::new(0))
}
