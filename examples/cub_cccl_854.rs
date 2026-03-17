//! # CUB/CCCL Issue #854: Compiler Predicates Off Mask Initialization
//!
//! Demonstrates how warp typestate prevents a bug class where the
//! compiler optimizes away mask initialization, producing wrong PTX.
//!
//! ## The Real Bug
//!
//! In CUB's `WarpScanShfl`, with sub-warp logical warp sizes:
//!
//! ```cuda
//! // LOGICAL_WARP_THREADS = 16 (sub-warp scan)
//! unsigned member_mask = 0xFFFF;  // mask for 16-thread logical warp
//!
//! for (int i = 1; i < LOGICAL_WARP_THREADS; i <<= 1) {
//!     int val = __shfl_up_sync(member_mask, partial, i);
//!     if (lane_id >= i) partial += val;
//!     // BUG: compiler may predicate off member_mask initialization
//!     // when threads exit this loop early. PTX shfl.sync.idx.b32
//!     // executes unconditionally with uninitialized mask.
//! }
//! ```
//!
//! **Failure mode:** Silent wrong scan. Only with `LOGICAL_WARP_THREADS < 32`
//! in early-exit loops. Found by `cuda-memcheck --tool synccheck`, not by
//! observing wrong output.
//!
//! **Source:** <https://github.com/NVIDIA/cccl/issues/854>
//!
//! ## Why `__shfl_sync` Doesn't Help
//!
//! The source code is correct — `member_mask` is properly initialized. The
//! compiler generates wrong PTX: it predicates off the mask initialization,
//! so the PTX shuffle instruction reads an uninitialized register as its mask.
//! `__shfl_sync`'s mask parameter becomes garbage at the instruction level
//! despite being correct at the source level. No runtime API can fix a
//! compiler optimization bug.
//!
//! ## Why Warp Typestate Catches It
//!
//! In our type system, the mask is not a runtime variable that can be
//! optimized away — it's a type parameter. `Warp<SubWarp16>` is a zero-sized
//! phantom type. There is no register to predicate off, no initialization
//! to elide. The compiler cannot remove something that doesn't exist at
//! runtime.
//!
//! Run: `cargo test --example cub_cccl_854`

#![allow(clippy::needless_range_loop, clippy::new_without_default)]

use std::marker::PhantomData;

// ============================================================================
// MINIMAL TYPE SYSTEM (self-contained for the example)
// ============================================================================

pub trait ActiveSet: Copy + 'static {
    const MASK: u32;
    const SIZE: u32;
    const NAME: &'static str;
}

#[derive(Copy, Clone)]
pub struct Warp<S: ActiveSet> {
    _phantom: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self {
        Warp {
            _phantom: PhantomData,
        }
    }
    pub fn active_mask(&self) -> u32 {
        S::MASK
    }
    pub fn size(&self) -> u32 {
        S::SIZE
    }
}

// Active set types
#[derive(Copy, Clone)]
pub struct All;
#[derive(Copy, Clone)]
pub struct SubWarp16;

impl ActiveSet for All {
    const MASK: u32 = 0xFFFFFFFF;
    const SIZE: u32 = 32;
    const NAME: &'static str = "All";
}
impl ActiveSet for SubWarp16 {
    const MASK: u32 = 0x0000FFFF;
    const SIZE: u32 = 16;
    const NAME: &'static str = "SubWarp16";
}

// ============================================================================
// GATED OPERATIONS
// ============================================================================

/// Shuffle-up: only on Warp<All>.
/// Sub-warp scans must first establish full-warp context.
impl Warp<All> {
    pub fn shuffle_up(&self, data: &[i32; 32], delta: u32) -> [i32; 32] {
        let mut result = [0i32; 32];
        for lane in 0..32u32 {
            result[lane as usize] = if lane >= delta {
                data[(lane - delta) as usize]
            } else {
                0
            };
        }
        result
    }

    pub fn extract_subwarp16(self) -> Warp<SubWarp16> {
        Warp::new()
    }
}

/// Sub-warp operations: scan within the logical warp.
/// No shuffle — uses only sub-warp-safe operations.
impl Warp<SubWarp16> {
    /// Inclusive scan within the 16-lane sub-warp.
    /// Does NOT use shuffle — uses the type-safe sub-warp protocol.
    pub fn inclusive_scan_sum(&self, data: &[i32; 16]) -> [i32; 16] {
        let mut result = *data;
        for i in 0..16 {
            if i > 0 {
                result[i] += result[i - 1];
            }
        }
        result
    }
}

// ============================================================================
// THE BUG (CUB/CCCL #854) — why source-level correctness isn't enough
// ============================================================================

/// What the CUDA code does (translated to our type system).
///
/// The source-level pattern: initialize member_mask, use it in shfl_up_sync
/// inside a loop. Compiler predicates off the mask initialization.
///
/// In our type system, this pattern is impossible because the mask is a type,
/// not a runtime variable. There is nothing for the compiler to optimize away.
///
/// ```compile_fail
/// # use cub_cccl_854::*;
/// fn buggy_subwarp_scan(warp: Warp<All>) -> [i32; 32] {
///     let subwarp = warp.extract_subwarp16();
///
///     // BUG: Try to shuffle on the sub-warp handle
///     // In CUDA, this works at source level but the compiler may produce
///     // wrong PTX for the mask initialization
///     let data = [1i32; 32];
///     let shifted = subwarp.shuffle_up(&data, 1);
///     //            ^^^^^^^^^^ ERROR: no method `shuffle_up` found for `Warp<SubWarp16>`
///     shifted
/// }
/// ```
fn _buggy_version_for_doctest() {}

// ============================================================================
// THE FIX
// ============================================================================

/// Correct approach: use sub-warp-safe operations that don't rely on
/// a runtime mask the compiler can optimize away.
fn correct_subwarp_scan(warp: Warp<All>) -> [i32; 16] {
    let subwarp = warp.extract_subwarp16();

    // Sub-warp scan using type-safe operations
    let data = [1i32; 16]; // each lane contributes 1
    subwarp.inclusive_scan_sum(&data)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_subwarp_scan() {
        let warp: Warp<All> = Warp::new();
        let result = correct_subwarp_scan(warp);

        // Inclusive sum of [1,1,1,...,1] = [1,2,3,...,16]
        for i in 0..16 {
            assert_eq!(result[i], (i + 1) as i32);
        }
    }

    #[test]
    fn test_type_prevents_shuffle_on_subwarp() {
        let warp: Warp<All> = Warp::new();
        let subwarp = warp.extract_subwarp16();

        // Verify: subwarp is Warp<SubWarp16>, which has NO shuffle_up method
        assert_eq!(subwarp.active_mask(), 0x0000FFFF);
        assert_eq!(subwarp.size(), 16);

        // The following would not compile:
        // subwarp.shuffle_up(&[0i32; 32], 1);
        // error[E0599]: no method named `shuffle_up` found for `Warp<SubWarp16>`
    }

    #[test]
    fn test_why_compiler_cant_break_types() {
        // In CUB's bug, the compiler optimized away a runtime mask variable.
        // In our system, the mask is a type parameter — PhantomData<SubWarp16>.
        // PhantomData is zero-sized: no register, no initialization, nothing
        // to predicate off.
        let warp: Warp<All> = Warp::new();
        let subwarp = warp.extract_subwarp16();

        // The "mask" is structural, not a value
        assert_eq!(std::mem::size_of::<Warp<SubWarp16>>(), 0);
        assert_eq!(std::mem::size_of::<Warp<All>>(), 0);

        // Same zero-size — the distinction exists only in the type system
        assert_eq!(subwarp.active_mask(), SubWarp16::MASK);
    }

    #[test]
    fn test_full_warp_shuffle_works() {
        let warp: Warp<All> = Warp::new();
        let data = [0i32; 32];

        // shuffle_up IS available on Warp<All> — no sub-warp ambiguity
        let _shifted = warp.shuffle_up(&data, 1);
    }
}

fn main() {
    println!("CUB/CCCL #854: Compiler Predicates Off Mask Initialization");
    println!("============================================================\n");

    println!("The Bug (CUDA):");
    println!("  CUB's WarpScanShfl: member_mask initialized at source level.");
    println!("  Compiler predicates off the initialization in early-exit loops.");
    println!("  PTX shfl.sync reads uninitialized mask register. Silent wrong scan.\n");

    println!("Why __shfl_sync Doesn't Help:");
    println!("  Source code is correct! The compiler produces wrong PTX.");
    println!("  No runtime API can fix a compiler optimization bug.\n");

    println!("Why Warp Typestate Catches It:");
    println!("  The mask is a type (PhantomData), not a register.");
    println!("  Zero-sized: nothing to initialize, nothing to optimize away.");
    println!("  Warp<SubWarp16> has no shuffle_up — sub-warp ops are separate.\n");

    let warp: Warp<All> = Warp::new();
    let result = correct_subwarp_scan(warp);
    println!("Correct sub-warp scan: {:?}", &result[..]);
    println!("Expected: [1, 2, 3, ..., 16]");

    assert_eq!(std::mem::size_of::<Warp<SubWarp16>>(), 0);
    println!("\nWarp<SubWarp16> size: 0 bytes (phantom type — nothing to optimize away)");
}
