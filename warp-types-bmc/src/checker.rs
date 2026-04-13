//! BMC checker: phase-typed bounded model checking loop.
//!
//! Connects the transition system model, unrolling engine, and SAT solver
//! through the phase-typed BMC session. Each phase transition is enforced
//! at compile time — you cannot check a property before encoding it, cannot
//! encode before unrolling, cannot deepen after finding a counterexample.

use crate::model::TransitionSystem;
use crate::phase::*;
use crate::session::{self, BmcSession};
use crate::unroll;

use warp_types_sat::solver::{solve_watched_budget, SolveResult};

/// Result of a BMC run.
#[derive(Debug)]
pub enum BmcResult {
    /// Counterexample found at depth `depth`. Contains the state trace.
    CounterexampleFound { depth: u32, trace: Vec<Vec<bool>> },
    /// No counterexample found up to `max_depth`. Bounded safety.
    BoundedSafe { max_depth: u32 },
    /// SAT solver budget exhausted at depth `depth`.
    Exhausted { depth: u32 },
}

/// Run bounded model checking on a transition system.
///
/// Incrementally unrolls the transition relation from depth 0 to `max_depth`,
/// checking the safety property at each depth. The phase-typed session ensures
/// the correct ordering: build → unroll → encode → check → (deepen | stop).
///
/// `conflict_budget` limits the SAT solver's work per depth. 0 = unlimited.
pub fn check(sys: &TransitionSystem, max_depth: u32, conflict_budget: u64) -> BmcResult {
    session::with_session(|init: BmcSession<'_, Init>| {
        let modeled = init.build_model();
        let mut unrolled = modeled.unroll();

        for depth in 0..=max_depth {
            // Encode property at current depth
            let encoded = unrolled.encode_property();

            // Build SAT instance
            let (db, num_vars) = unroll::encode_bmc(sys, depth);

            // Call SAT oracle
            let (result, _stats) = solve_watched_budget(db, num_vars, conflict_budget);

            match result {
                SolveResult::Sat(assignment) => {
                    // Counterexample found — extract trace
                    let trace = unroll::extract_trace(&assignment, sys.num_state_vars, depth);
                    let _cex = encoded.check_counterexample();
                    return BmcResult::CounterexampleFound { depth, trace };
                }
                SolveResult::Unsat => {
                    // Safe at this depth — deepen if not at max
                    let safe = encoded.check_sat();
                    if depth < max_depth {
                        unrolled = safe.deepen();
                    } else {
                        let max = safe.accept();
                        return BmcResult::BoundedSafe { max_depth: max };
                    }
                }
                SolveResult::Unknown => {
                    let _exhausted = encoded.check_exhausted();
                    return BmcResult::Exhausted { depth };
                }
            }
        }

        // Should not reach here — loop covers 0..=max_depth
        unreachable!()
    })
}
