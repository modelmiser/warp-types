//! Phase-typed CDCL SAT solver scaffold.
//!
//! Encodes the CDCL state machine in Rust's type system following the
//! typestate pattern from `warp_types::fence`. Phase transitions consume
//! the old state and produce the new — invalid transitions are compile errors.
//!
//! # CDCL Phase Machine
//!
//! ```text
//! with_session(|session| {
//!   // session: SolverSession<'s, Idle>
//!   //   → propagate()           → SolverSession<'s, Propagate>  (initial BCP)
//!   //   → decide()              → SolverSession<'s, Decide>
//!   //     → propagate()         → SolverSession<'s, Propagate>
//!   //       → finish_no_conflict() → SolverSession<'s, Idle>
//!   //       → finish_conflict()    → SolverSession<'s, Conflict>
//!   //         → analyze()       → SolverSession<'s, Analyze>
//!   //           → backtrack()   → SolverSession<'s, Backtrack>
//!   //             → propagate() → SolverSession<'s, Propagate>  (re-propagate learned clause)
//!   //             → unsat()     → bool
//!   //   → sat()                 → bool
//!   //   → unsat()               → bool
//! })
//! ```
//!
//! # Substrate-agnostic
//!
//! Phase types encode the CDCL contract, not the implementation.
//! GPU, FPGA, and CPU backends implement phases differently but satisfy
//! the same typed contract — preventing cross-substrate phase-ordering bugs.
//!
//! # Affine clause tokens
//!
//! `ClauseToken` is non-Copy, non-Clone. Prevents the #1 parallel SAT bug:
//! two threads propagating the same clause, producing contradictory learned clauses.

pub mod phase;
pub mod session;
pub mod clause;
pub mod literal;
pub mod clause_tile;
pub mod bcp;
pub mod dimacs;
pub mod scheduler;
pub mod bench;

pub use phase::*;
pub use session::*;
pub use clause::*;
