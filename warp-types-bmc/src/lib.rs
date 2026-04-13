//! Phase-typed Bounded Model Checker.
//!
//! Encodes the BMC workflow in Rust's type system following the typestate
//! pattern from `warp_types`. Phase transitions consume the old state and
//! produce the new — invalid transitions are compile errors.
//!
//! # BMC Phase Machine
//!
//! ```text
//! with_session(|session| {
//!   // session: BmcSession<'s, Init>
//!   let modeled = session.build_model();      // → Modeled
//!   let unrolled = modeled.unroll();           // → Unrolled (depth 0)
//!   let encoded = unrolled.encode_property();  // → Encoded
//!   // Check via SAT oracle:
//!   //   → check_sat()            → Safe     (UNSAT — no bug at this depth)
//!   //   → check_counterexample() → Counterexample (SAT — bug found)
//!   //   → check_exhausted()      → Exhausted (budget exceeded)
//!   // From Safe:
//!   //   → deepen()  → Unrolled (depth k+1)
//!   //   → accept()  → bounded safety result
//! })
//! ```
//!
//! # Built on warp-types-sat
//!
//! Uses the CDCL SAT solver from `warp-types-sat` as the decision oracle.
//! The BMC encoding (initial state, transition unrolling, property negation
//! via Tseitin) is handled by the `unroll` module; the phase-typed session
//! ensures the correct sequencing of model → unroll → encode → check.

pub mod checker;
pub mod model;
pub mod phase;
pub mod session;
pub mod unroll;

pub use checker::{check, BmcResult};
pub use model::TransitionSystem;
pub use phase::{Counterexample, Encoded, Exhausted, Init, Modeled, Phase, Safe, Unrolled};
pub use session::{with_session, BmcSession};
