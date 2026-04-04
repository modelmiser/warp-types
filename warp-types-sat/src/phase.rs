//! CDCL solver phase markers.
//!
//! Zero-sized types encoding solver phases. Sealed — external crates
//! cannot forge phases or create invalid transitions.
//!
//! Follows the `warp_types::fence::WriteState` pattern:
//! - Marker types carry no runtime data
//! - Sealed trait prevents external implementation
//! - Phase transitions are separate functions that consume old → produce new

// ============================================================================
// Sealed trait (own copy — warp_types' sealed is pub(crate))
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

/// Marker trait for CDCL solver phases. Sealed.
pub trait Phase: sealed::Sealed + Copy + 'static {
    /// Human-readable phase name (for diagnostics).
    const NAME: &'static str;
}

/// Solver is idle — ready for next decision or termination check.
#[derive(Debug, Clone, Copy)]
pub struct Idle;

/// Decision phase — choosing the next variable and polarity.
/// Available operations: variable scoring, decision heuristics (VSIDS sort).
#[derive(Debug, Clone, Copy)]
pub struct Decide;

/// Propagation phase — running BCP after a decision.
/// Available operations: clause checking (tile ballot), unit propagation.
#[derive(Debug, Clone, Copy)]
pub struct Propagate;

/// Conflict detected during propagation.
/// Available operations: conflict analysis, clause learning, activity bumping.
#[derive(Debug, Clone, Copy)]
pub struct Conflict;

/// Analyzing a conflict — building the learned clause.
/// Available operations: implication graph traversal, 1-UIP computation.
#[derive(Debug, Clone, Copy)]
pub struct Analyze;

/// Backtracking after conflict analysis.
/// Available operations: trail unwinding, assignment retraction.
#[derive(Debug, Clone, Copy)]
pub struct Backtrack;

/// Terminal: satisfiable (solution found).
#[derive(Debug, Clone, Copy)]
pub struct Sat;

/// Terminal: unsatisfiable (exhausted search space).
#[derive(Debug, Clone, Copy)]
pub struct Unsat;

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

impl_phase!(Idle, "idle");
impl_phase!(Decide, "decide");
impl_phase!(Propagate, "propagate");
impl_phase!(Conflict, "conflict");
impl_phase!(Analyze, "analyze");
impl_phase!(Backtrack, "backtrack");
impl_phase!(Sat, "sat");
impl_phase!(Unsat, "unsat");

// ============================================================================
// Phase transition proofs (sealed marker traits)
// ============================================================================

/// Proof that phase `From` can transition to phase `To`.
///
/// Only implemented for valid CDCL transitions. Attempting an invalid
/// transition (e.g., Idle → Analyze) is a compile error.
pub trait CanTransition<To: Phase>: sealed::Sealed + Phase {}

// Valid CDCL transitions:
//
//   Idle → Decide        (start decision)
//   Idle → Propagate     (initial BCP before first decision, or after restart)
//   Idle → Sat           (all variables assigned, no conflict)
//   Idle → Unsat         (empty clause derived at level 0)
//   Decide → Propagate   (decision made, run BCP)
//   Propagate → Idle     (BCP complete, no conflict — next decision)
//   Propagate → Conflict (BCP found conflict)
//   Conflict → Analyze   (start conflict analysis)
//   Analyze → Backtrack  (learned clause, now backtrack)
//   Backtrack → Propagate (re-propagate learned clause immediately)
//   Backtrack → Unsat    (backtracked to level 0 — unsatisfiable)

impl CanTransition<Decide> for Idle {}
impl CanTransition<Propagate> for Idle {}
impl CanTransition<Sat> for Idle {}
impl CanTransition<Unsat> for Idle {}
impl CanTransition<Propagate> for Decide {}
impl CanTransition<Idle> for Propagate {}
impl CanTransition<Conflict> for Propagate {}
impl CanTransition<Analyze> for Conflict {}
impl CanTransition<Backtrack> for Analyze {}
impl CanTransition<Propagate> for Backtrack {}
impl CanTransition<Unsat> for Backtrack {}

// PropagationOutcome removed — the caller now explicitly calls
// finish_no_conflict() or finish_conflict() on SolverSession<Propagate>
// based on the BcpResult. This keeps the phase proof connected to the
// actual BCP execution rather than being a pre-determined stub.
