//! SMT solver phase markers.
//!
//! Zero-sized types encoding the SMT solving workflow. Sealed —
//! external crates cannot forge phases or create invalid transitions.
//!
//! # Phase machine
//!
//! ```text
//!   Init
//!     → declare_sort(), declare_fun()   stays Init (accumulates declarations)
//!     → finish_declarations()           → Declared
//!   Declared
//!     → assert_formula()                stays Declared (accumulates assertions)
//!     → finish_assertions()             → Asserted
//!   Asserted
//!     → check_sat()                     → SmtResult
//!       → Sat                             terminal: satisfiable model exists
//!       → Unsat                           terminal: no model exists
//!       → Unknown                         terminal: solver budget exhausted
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

/// Marker trait for SMT phases. Sealed.
pub trait Phase: sealed::Sealed + 'static {
    /// Human-readable phase name.
    const NAME: &'static str;
}

/// Initial state — declare sorts and function symbols.
#[derive(Debug)]
pub struct Init;

/// Declarations finalized — assert formulas.
#[derive(Debug)]
pub struct Declared;

/// Assertions finalized — ready to solve.
#[derive(Debug)]
pub struct Asserted;

/// Satisfiable — a model exists.
/// Terminal phase.
#[derive(Debug)]
pub struct Sat;

/// Unsatisfiable — no model exists.
/// Terminal phase.
#[derive(Debug)]
pub struct Unsat;

/// Solver budget exhausted without conclusive result.
/// Terminal phase.
#[derive(Debug)]
pub struct Unknown;

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
impl_phase!(Declared, "declared");
impl_phase!(Asserted, "asserted");
impl_phase!(Sat, "sat");
impl_phase!(Unsat, "unsat");
impl_phase!(Unknown, "unknown");
