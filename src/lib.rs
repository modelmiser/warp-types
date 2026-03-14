//! Warp Types: Session-typed GPU warp programming.
//!
//! Prevents warp divergence bugs at compile time using session types.
//! A diverged warp literally cannot call shuffle — the method doesn't exist.
//!
//! # Core Idea
//!
//! ```
//! use warp_types::*;
//!
//! let warp: Warp<All> = Warp::kernel_entry();
//! let data = data::PerLane::new(42i32);
//!
//! // OK: shuffle on full warp
//! let _shuffled = warp.shuffle_xor(data, 1);
//!
//! // After diverge, shuffle is gone from the type:
//! let (evens, odds) = warp.diverge_even_odd();
//! // evens.shuffle_xor(data, 1);  // COMPILE ERROR — method not found
//! let merged: Warp<All> = merge(evens, odds);
//! ```
//!
//! # Module Overview
//!
//! - [`active_set`] — Lane subset types (`All`, `Even`, `Odd`, ...) and complement proofs
//! - [`warp`] — `Warp<S>` type parameterized by active set
//! - [`data`] — Value categories: `PerLane<T>`, `Uniform<T>`, `SingleLane<T, N>`
//! - [`diverge`] — Split warps by predicate (produces complementary sub-warps)
//! - [`merge`] — Rejoin complementary sub-warps (compile-time verified)
//! - [`shuffle`] — Shuffle/ballot/reduce (restricted to `Warp<All>`) + permutation algebra
//! - [`fence`] — Fence-divergence interactions (§5.6) — type-state write tracking
//! - [`block`] — Block-level: shared memory ownership, inter-block sessions, reductions
//! - [`proof`] — Soundness proof sketch (progress + preservation)
//! - [`platform`] — CPU/GPU platform trait for dual-mode algorithms
//! - [`warp_size`] — Const-generic warp size portability
//! - [`gradual`] — `DynWarp` ↔ `Warp<S>` bridge for gradual typing (§9.4)

#![allow(dead_code)]

// ============================================================================
// Core modules (public API)
// ============================================================================

pub mod active_set;
pub mod warp;
pub mod data;
pub mod diverge;
pub mod merge;
pub mod shuffle;
pub mod fence;
pub mod block;
#[cfg(any(test, feature = "formal-proof"))]
pub mod proof;
pub mod platform;
pub mod warp_size;
pub mod gradual;

// ============================================================================
// Research explorations (compiled, not re-exported)
// ============================================================================

pub mod research;

// ============================================================================
// Zero-overhead verification: inspectable functions for LLVM IR comparison
// ============================================================================

/// Butterfly reduction: diverge → merge → shuffle pattern.
///
/// In optimized LLVM IR, this function contains NO traces of `Warp<S>`,
/// `PhantomData`, or active-set types. The type system is fully erased.
/// Inspect with: `cargo rustc --release --lib -- --emit=llvm-ir`
/// then search for `zero_overhead_butterfly` in the .ll file.
#[no_mangle]
#[inline(never)]
pub fn zero_overhead_butterfly(data: data::PerLane<i32>) -> i32 {
    let warp: Warp<All> = Warp::kernel_entry();
    // Shuffle XOR 16: exchange with partner 16 lanes away
    let step1 = warp.shuffle_xor(data, 16);
    // Shuffle XOR 8
    let step2 = warp.shuffle_xor(step1, 8);
    // Shuffle XOR 4
    let step3 = warp.shuffle_xor(step2, 4);
    // Shuffle XOR 2
    let step4 = warp.shuffle_xor(step3, 2);
    // Shuffle XOR 1
    let step5 = warp.shuffle_xor(step4, 1);
    // Final reduction
    warp.reduce_sum(step5)
}

/// Diverge-merge round trip: the type system's core mechanism.
///
/// In optimized LLVM IR, this compiles to a no-op (returns input unchanged).
/// The diverge, merge, and all warp handles are completely erased.
#[no_mangle]
#[inline(never)]
pub fn zero_overhead_diverge_merge(data: data::PerLane<i32>) -> data::PerLane<i32> {
    let warp: Warp<All> = Warp::kernel_entry();
    let (evens, odds) = warp.diverge_even_odd();
    let _merged: Warp<All> = merge(evens, odds);
    data // diverge/merge is pure type-level — data passes through unchanged
}

// ============================================================================
// GpuValue trait
// ============================================================================

/// Marker trait for types that can live in GPU registers.
///
/// Requires `Copy` (registers are value types), `Send + Sync` (cross-lane),
/// `Default` (inactive lanes need a value), and `'static` (no borrows).
pub trait GpuValue: Copy + Send + Sync + Default + 'static {}

impl GpuValue for i32 {}
impl GpuValue for u32 {}
impl GpuValue for f32 {}
impl GpuValue for i64 {}
impl GpuValue for u64 {}
impl GpuValue for f64 {}
impl GpuValue for bool {}

// ============================================================================
// Re-exports — flat access to the most-used types
// ============================================================================

pub use active_set::{
    ActiveSet, ComplementOf, ComplementWithin, CanDiverge,
    All, None, Even, Odd, LowHalf, HighHalf,
    Lane0, NotLane0, EvenLow, EvenHigh, OddLow, OddHigh,
};
pub use warp::Warp;
pub use data::{LaneId, WarpId, Uniform, PerLane, SingleLane, Role};
pub use merge::{merge, merge_within};
pub use shuffle::{
    Shuffle, Ballot, Vote, Reduce, BallotResult,
    Permutation, HasDual, Xor, RotateDown, RotateUp, Identity, Compose,
};
pub use fence::{GlobalRegion, Unwritten, PartialWrite, FullWrite, Fenced, WriteState};
pub use block::{SharedRegion, BlockId, ThreadId};
pub use platform::{Platform, CpuSimd, GpuWarp32, SimdVector};
