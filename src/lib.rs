//! Warp Types: Type-safe GPU warp programming via linear typestate.
//!
//! Prevents warp divergence bugs at compile time using linear typestate.
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
//! - [mod@merge] — Rejoin complementary sub-warps (compile-time verified)
//! - [`shuffle`] — Shuffle/ballot/reduce (restricted to `Warp<All>`) + permutation algebra
//! - [`fence`] — Fence-divergence interactions (§5.6) — type-state write tracking
//! - [`block`] — Block-level: shared memory ownership, inter-block sessions, reductions
//! - `proof` — Soundness proof sketch (progress + preservation)
//! - [`platform`] — CPU/GPU platform trait for dual-mode algorithms
//! - [`gradual`] — `DynWarp` ↔ `Warp<S>` bridge for gradual typing (§9.4)
//! - [`gpu`] — GPU intrinsics for nvptx64 and amdgpu targets
//! - [`cub`] — Typed CUB-equivalent warp primitives (reduce, scan, broadcast)
//! - [`sort`] — Typed warp-level bitonic sort
//! - [`tile`] — Cooperative Groups: thread block tiles with typed shuffle safety
//! - [`dynamic`] — Data-dependent divergence with structural complement guarantees
//! - [`simwarp`] — Multi-lane warp simulator with real shuffle semantics (testing)

#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", no_main)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx, asm_experimental_arch))]
// dead_code is allowed only in the research module (experimental prototypes).
// Core modules should not have dead code — if it's unused, remove it or
// mark it #[allow(dead_code)] individually with a justification comment.

// ============================================================================
// Warp size configuration
// ============================================================================

/// Number of lanes per warp/wavefront.
///
/// - NVIDIA: 32 lanes (default)
/// - AMD: 64 lanes (enable `warp64` feature)
#[cfg(not(feature = "warp64"))]
pub const WARP_SIZE: u32 = 32;

/// Number of lanes per warp/wavefront (AMD 64-lane mode).
#[cfg(feature = "warp64")]
pub const WARP_SIZE: u32 = 64;

// ============================================================================
// Core modules (public API)
// ============================================================================

pub mod active_set;
pub mod block;
pub mod cub;
pub mod data;
pub mod diverge;
pub mod dynamic;
pub mod fence;
pub mod gpu;
pub mod gradual;
pub mod merge;
pub mod platform;
#[cfg(any(test, feature = "formal-proof"))]
pub mod proof;
pub mod shuffle;
pub mod simwarp;
pub mod sort;
pub mod tile;
pub mod warp;

// ============================================================================
// Research explorations (compiled, not re-exported)
// ============================================================================

#[cfg(not(target_arch = "nvptx64"))]
#[allow(dead_code)]
// Research modules contain experimental prototypes with unused code
// Research modules: exploratory demos, not production API.
// Suppress clippy lints inappropriate for proof-of-concept code.
#[allow(
    clippy::new_without_default,
    clippy::needless_range_loop,
    clippy::module_inception,
    clippy::doc_markdown,
    clippy::empty_line_after_doc_comments,
    clippy::items_after_test_module,
    clippy::approx_constant,
    rustdoc::invalid_html_tags,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_rust_codeblocks
)]
pub mod research;

// ============================================================================
// Zero-overhead verification: inspectable functions for LLVM IR comparison
// ============================================================================

/// Zero-overhead benchmark: 5 shuffle permutations + butterfly reduction.
///
/// This function exercises shuffle and reduce to verify type erasure.
/// The 5 `shuffle_xor` calls permute data; `reduce_sum` does the actual
/// butterfly reduction (5 more shuffle-XOR + add steps). Total: 10 shuffles.
///
/// In optimized LLVM IR, this function contains NO traces of `Warp<S>`,
/// `PhantomData`, or active-set types. The type system is fully erased.
/// Inspect with: `cargo rustc --release --lib -- --emit=llvm-ir`
/// then search for `zero_overhead_butterfly` in the .ll file.
#[export_name = "warp_types_zero_overhead_butterfly"]
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
    warp.reduce_sum(step5).get()
}

/// Diverge-merge round trip: the type system's core mechanism.
///
/// In optimized LLVM IR, this compiles to a no-op (returns input unchanged).
/// The diverge, merge, and all warp handles are completely erased.
#[export_name = "warp_types_zero_overhead_diverge_merge"]
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
///
/// Sealed: only primitive GPU types implement this trait. External crates
/// cannot add implementations, ensuring `PerLane<T>` and `Uniform<T>`
/// only wrap types with known GPU register semantics.
pub trait GpuValue: active_set::sealed::Sealed + Copy + Send + Sync + Default + 'static {}

#[allow(private_interfaces)]
impl active_set::sealed::Sealed for i32 {
    fn _sealed() -> active_set::sealed::SealToken {
        active_set::sealed::SealToken
    }
}
impl GpuValue for i32 {}
#[allow(private_interfaces)]
impl active_set::sealed::Sealed for u32 {
    fn _sealed() -> active_set::sealed::SealToken {
        active_set::sealed::SealToken
    }
}
impl GpuValue for u32 {}
#[allow(private_interfaces)]
impl active_set::sealed::Sealed for f32 {
    fn _sealed() -> active_set::sealed::SealToken {
        active_set::sealed::SealToken
    }
}
impl GpuValue for f32 {}
#[allow(private_interfaces)]
impl active_set::sealed::Sealed for i64 {
    fn _sealed() -> active_set::sealed::SealToken {
        active_set::sealed::SealToken
    }
}
impl GpuValue for i64 {}
#[allow(private_interfaces)]
impl active_set::sealed::Sealed for u64 {
    fn _sealed() -> active_set::sealed::SealToken {
        active_set::sealed::SealToken
    }
}
impl GpuValue for u64 {}
#[allow(private_interfaces)]
impl active_set::sealed::Sealed for f64 {
    fn _sealed() -> active_set::sealed::SealToken {
        active_set::sealed::SealToken
    }
}
impl GpuValue for f64 {}
#[allow(private_interfaces)]
impl active_set::sealed::Sealed for bool {
    fn _sealed() -> active_set::sealed::SealToken {
        active_set::sealed::SealToken
    }
}
impl GpuValue for bool {}

// ============================================================================
// Re-exports — flat access to the most-used types
// ============================================================================

pub use active_set::{
    ActiveSet, All, CanDiverge, ComplementOf, ComplementWithin, Empty, Even, EvenHigh, EvenLow,
    HighHalf, Lane0, LowHalf, NotLane0, Odd, OddHigh, OddLow,
};
pub use block::{BlockId, SharedRegion, ThreadId};
pub use data::{LaneId, PerLane, Role, SingleLane, Uniform, WarpId};
pub use dynamic::DynDiverge;
pub use fence::{Fenced, FullWrite, GlobalRegion, PartialWrite, Unwritten, WriteState};
pub use gradual::DynWarp;
pub use merge::{merge, merge_within};
pub use platform::{CpuSimd, GpuWarp32, GpuWarp64, Platform, SimdVector};
pub use shuffle::{
    BallotResult, Compose, HasDual, Identity, Permutation, RotateDown, RotateUp, ShuffleSafe, Xor,
};
pub use tile::Tile;
pub use warp::Warp;
pub use warp_types_kernel::warp_kernel;

/// Convenience prelude — import everything needed for typical usage.
///
/// ```rust
/// use warp_types::prelude::*;
///
/// let warp: Warp<All> = Warp::kernel_entry();
/// let (evens, odds) = warp.diverge_even_odd();
/// let merged: Warp<All> = merge(evens, odds);
/// ```
pub mod prelude {
    pub use crate::data;
    pub use crate::gpu::GpuShuffle;
    pub use crate::{
        merge, merge_within, ActiveSet, All, CanDiverge, ComplementOf, ComplementWithin,
        DynDiverge, DynWarp, Empty, Even, EvenHigh, EvenLow, GpuValue, HighHalf, Lane0, LowHalf,
        NotLane0, Odd, OddHigh, OddLow, PerLane, SingleLane, Tile, Uniform, Warp,
    };
}
