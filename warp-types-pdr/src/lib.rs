//! Phase-typed Property-Directed Reachability (IC3).
//!
//! Proves unbounded safety properties or finds concrete counterexamples
//! by maintaining an inductive frame sequence. Unlike bounded model checking
//! (which only proves safety up to a depth bound), PDR can prove that a
//! property holds at *all* depths by finding an inductive invariant.
//!
//! # PDR Phase Machine
//!
//! ```text
//! with_session(|session| {
//!   // session: PdrSession<'s, Init>
//!   let modeled = session.build_model();   // → Modeled
//!   // Internally: strengthen → propagate → converge (or counterexample)
//!   //   → check_safe()            → Safe     (inductive invariant found)
//!   //   → check_counterexample()  → CounterexampleFound (trace found)
//!   //   → check_exhausted()       → Exhausted (frame budget exceeded)
//! })
//! ```
//!
//! # Built on warp-types-sat and warp-types-bmc
//!
//! Uses the CDCL SAT solver from `warp-types-sat` as the decision oracle
//! (multiple queries per frame: consecution, predecessor, generalization).
//! Shares the `TransitionSystem` model type from `warp-types-bmc`.

pub mod checker;
pub mod cube;
pub mod frames;
pub mod phase;
pub mod session;

pub use checker::{check, PdrResult};
pub use cube::Cube;
pub use frames::{Frame, FrameSequence};
pub use phase::{CounterexampleFound, Exhausted, Init, Modeled, Phase, Safe};
pub use session::{with_session, PdrSession};
