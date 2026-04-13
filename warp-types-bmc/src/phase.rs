//! BMC phase markers.
//!
//! Zero-sized types encoding bounded model checking phases. Sealed —
//! external crates cannot forge phases or create invalid transitions.
//!
//! # Phase machine
//!
//! ```text
//!   Init
//!     → build_model()          → Modeled
//!   Modeled
//!     → unroll()               → Unrolled { depth: 0 }
//!   Unrolled { depth: k }
//!     → encode_property()      → Encoded { depth: k }
//!   Encoded { depth: k }
//!     → check()                → CheckResult
//!       → Counterexample(k)      terminal: bug found at depth k
//!       → Safe(k)                → deepen() → Unrolled { depth: k+1 }
//!       → Exhausted              terminal: SAT budget exceeded
//! ```

// ============================================================================
// Sealed trait
// ============================================================================

pub(crate) mod sealed {
    pub(crate) struct SealToken;

    #[allow(private_interfaces)]
    pub trait Sealed {
        #[doc(hidden)]
        fn _sealed() -> SealToken;
    }
}

// ============================================================================
// Phase trait + markers
// ============================================================================

/// Marker trait for BMC phases. Sealed.
pub trait Phase: sealed::Sealed + 'static {
    /// Human-readable phase name.
    const NAME: &'static str;
}

/// Initial state — no model loaded yet.
#[derive(Debug)]
pub struct Init;

/// Model built — transition system is loaded, ready for unrolling.
#[derive(Debug)]
pub struct Modeled;

/// Transition relation unrolled to depth `k`.
/// Each unroll appends one time-frame of state variables and transition clauses.
#[derive(Debug)]
pub struct Unrolled;

/// Property encoded at the current unroll depth.
/// The SAT instance is ready to be checked.
#[derive(Debug)]
pub struct Encoded;

/// SAT returned UNSAT — no counterexample exists at this depth.
/// Can deepen (unroll further) or declare bounded safety.
#[derive(Debug)]
pub struct Safe;

/// SAT returned SAT — counterexample trace found.
/// Terminal phase: the model has a reachable bad state.
#[derive(Debug)]
pub struct Counterexample;

/// SAT budget exhausted without a conclusive result.
/// Terminal phase.
#[derive(Debug)]
pub struct Exhausted;

// ============================================================================
// Sealed + Phase impls
// ============================================================================

macro_rules! impl_phase {
    ($ty:ty, $name:literal) => {
        #[allow(private_interfaces)]
        impl sealed::Sealed for $ty {
            fn _sealed() -> sealed::SealToken {
                sealed::SealToken
            }
        }
        impl Phase for $ty {
            const NAME: &'static str = $name;
        }
    };
}

impl_phase!(Init, "init");
impl_phase!(Modeled, "modeled");
impl_phase!(Unrolled, "unrolled");
impl_phase!(Encoded, "encoded");
impl_phase!(Safe, "safe");
impl_phase!(Counterexample, "counterexample");
impl_phase!(Exhausted, "exhausted");
