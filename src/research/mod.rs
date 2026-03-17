//! Research explorations — compiled but not part of the public API.
//!
//! These modules contain prototypes, proofs-of-concept, and explorations
//! from the development of the warp type system. They are preserved for
//! reference but are not re-exported from the crate root.
//!
//! Note: Many research modules define their own local `Warp<S>`, `ActiveSet`,
//! etc. rather than importing the core types. These local definitions may have
//! different semantics (e.g., `Copy` vs affine). The research demos illustrate
//! concepts but do not prove properties of the core type system — the Lean
//! formalization in `lean/` does that.


pub mod static_verify;
pub mod lane;
pub mod shared;
pub mod nested_diverge;
pub mod varying_loops;
pub mod shuffle_duality;
pub mod inactive_lanes;
pub mod arbitrary_predicates;
// soundness.rs removed — canonical proof lives in src/proof.rs (gated behind formal-proof feature)
pub mod inter_block;
pub mod recursive_protocols;
pub mod protocol_inference;
pub mod uniformity_inference;
pub mod early_exit;
pub mod divergent_values;
pub mod work_stealing;
pub mod implicit_merge;
pub mod borrowing;
pub mod coalescing;
pub mod register_pressure;
pub mod scalability;
pub mod cpu_gpu_session;
pub mod predicate_expressiveness;
pub mod crossbar_protocol;
pub mod warp_size;
