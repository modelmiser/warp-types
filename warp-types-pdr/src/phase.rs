//! PDR phase markers.
//!
//! Zero-sized types encoding the IC3/PDR workflow. Sealed —
//! external crates cannot forge phases or create invalid transitions.
//!
//! # Phase machine
//!
//! ```text
//!   Init
//!     → build_model()       → Modeled
//!   Modeled
//!     → check()             → PdrResult
//!       → Safe                terminal: inductive invariant found
//!       → CounterexampleFound terminal: concrete trace to bad state
//!       → Exhausted           terminal: frame budget exceeded
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

/// Marker trait for PDR phases. Sealed.
pub trait Phase: sealed::Sealed + 'static {
    /// Human-readable phase name.
    const NAME: &'static str;
}

/// Initial state — no model loaded yet.
#[derive(Debug)]
pub struct Init;

/// Model built — transition system loaded, ready for PDR.
#[derive(Debug)]
pub struct Modeled;

/// Inductive invariant found — property is safe at all depths.
/// Terminal phase.
#[derive(Debug)]
pub struct Safe;

/// Counterexample trace found — property is violated.
/// Terminal phase.
#[derive(Debug)]
pub struct CounterexampleFound;

/// Frame budget exhausted without conclusive result.
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
impl_phase!(Safe, "safe");
impl_phase!(CounterexampleFound, "counterexample-found");
impl_phase!(Exhausted, "exhausted");
