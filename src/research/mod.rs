//! Research explorations — prototypes, not production API.
//!
//! These modules contain prototypes, proofs-of-concept, and explorations
//! from the development of the warp type system. They are preserved for
//! reference and are not re-exported from the crate root.
//!
//! They ARE public, though: this said "not part of the public API" until a
//! 2026-09-05 review pointed out that is false. Under the non-default
//! `research` feature any downstream crate can reach
//! `warp_types::research::coalescing::*`, unsafe constructors included, so
//! the safety contracts here are load-bearing for people outside this repo.
//! Not re-exported at the root is not the same as not public.
//!
//! Note: Many research modules define their own local `Warp<S>`, `ActiveSet`,
//! etc. rather than importing the core types. These local definitions may have
//! different semantics (e.g., `Copy` vs affine). The research demos illustrate
//! concepts but do not prove properties of the core type system — the Lean
//! formalization in `lean/` does that.

pub mod arbitrary_predicates;
pub mod inactive_lanes;
pub mod lane;
pub mod nested_diverge;
pub mod shared;
pub mod shuffle_duality;
pub mod static_verify;
pub mod varying_loops;
// soundness.rs removed — canonical proof lives in src/proof.rs (gated behind formal-proof feature)
pub mod borrowing;
pub mod coalescing;
pub mod cpu_gpu_session;
pub mod crossbar_protocol;
pub mod divergent_values;
pub mod early_exit;
pub mod implicit_merge;
pub mod inter_block;
pub mod predicate_expressiveness;
pub mod protocol_inference;
pub mod recursive_protocols;
pub mod register_pressure;
pub mod scalability;
pub mod uniformity_inference;
pub mod warp_size;
pub mod work_stealing;
