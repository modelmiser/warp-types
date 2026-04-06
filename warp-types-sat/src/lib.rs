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

pub mod analyze;
pub mod bcp;
pub(crate) mod bench;
pub mod clause;
pub mod clause_tile;
pub mod dimacs;
pub mod gpu_gradient;
pub mod gradient;
pub mod literal;
pub mod phase;
pub mod restart;
pub mod scheduler;
pub mod session;
pub mod solver;
pub mod trail;
pub mod vsids;
pub mod watch;

// Core types re-exported at crate root.
pub use clause::{ClausePool, ClauseToken};
pub use dimacs::parse_dimacs_str;
pub use phase::{
    Analyze, Backtrack, CanTransition, Conflict, Decide, Idle, Phase, Propagate, Sat, Unsat,
};
pub use session::{with_session, SolverSession};
pub use analyze::{
    ConflictProfile, Correlation, DagSummary, ProofDag, ResolutionStep,
    clause_reuse_frequency, correlate_centrality_vs_bump_freq,
    correlate_depth_vs_clause_reuse, correlate_depth_vs_next_bcp,
    correlate_pivot_vs_gradient, pearson_r, pivot_frequency,
    working_width_profile,
};
pub use bcp::CRef;
pub use solver::{solve, solve_watched, solve_watched_stats, SolveResult, SolveStats};
