//! PDR session with lifetime-branded phase tracking.
//!
//! `PdrSession<'s, P>` follows the pattern from `warp_types_bmc::BmcSession`:
//! - `'s` is an invariant lifetime brand (prevents cross-session mixing)
//! - `P: Phase` tracks the current PDR workflow phase
//! - Transitions consume the session and produce a new phase
//! - Terminal states (`Safe`, `CounterexampleFound`, `Exhausted`) have no outgoing transitions
//!
//! Zero-sized — all PDR state (frames, obligations) lives inside `checker::check()`.

use core::marker::PhantomData;
use crate::phase::*;

// ============================================================================
// PDR session
// ============================================================================

/// A PDR session branded with lifetime `'s` and phase `P`.
///
/// Zero-sized — all phase tracking is compile-time only.
#[must_use = "dropping a PdrSession loses phase tracking — use a transition or terminal"]
pub struct PdrSession<'s, P: Phase> {
    _brand: PhantomData<fn(&'s ()) -> &'s ()>,
    _phase: PhantomData<P>,
}

impl<'s, P: Phase> PdrSession<'s, P> {
    pub(crate) fn new() -> Self {
        PdrSession {
            _brand: PhantomData,
            _phase: PhantomData,
        }
    }

    /// Current phase name.
    pub fn phase_name(&self) -> &'static str {
        P::NAME
    }
}

// ============================================================================
// Phase transitions
// ============================================================================

impl<'s> PdrSession<'s, Init> {
    /// Load a transition system model. Consumes Init, produces Modeled.
    pub fn build_model(self) -> PdrSession<'s, Modeled> {
        PdrSession::new()
    }
}

impl<'s> PdrSession<'s, Modeled> {
    /// Check result: property is safe (inductive invariant found).
    pub fn check_safe(self) -> PdrSession<'s, Safe> {
        PdrSession::new()
    }

    /// Check result: counterexample found.
    pub fn check_counterexample(self) -> PdrSession<'s, CounterexampleFound> {
        PdrSession::new()
    }

    /// Check result: frame budget exhausted.
    pub fn check_exhausted(self) -> PdrSession<'s, Exhausted> {
        PdrSession::new()
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Create a PDR session with a fresh lifetime brand.
pub fn with_session<R>(f: impl for<'s> FnOnce(PdrSession<'s, Init>) -> R) -> R {
    f(PdrSession::new())
}
