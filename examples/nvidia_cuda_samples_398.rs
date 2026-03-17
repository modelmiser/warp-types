//! # NVIDIA cuda-samples Issue #398: Reduction Shuffle-Mask Bug
//!
//! Demonstrates how warp typestate catches a real bug from NVIDIA's
//! own cuda-samples repository.
//!
//! ## The Real Bug
//!
//! In `reduce7`, NVIDIA's official parallel reduction sample:
//!
//! ```cuda
//! // Block-level tree reduction narrows to one warp...
//! if (tid < warpSize) {
//!     // Warp-level reduction using shuffle
//!     unsigned mask = __ballot_sync(0xFFFFFFFF, tid < blockDim.x / warpSize);
//!     // BUG: When blockDim.x == 32, only tid 0 enters.
//!     // mask = 1, so shfl_down reads from lane 16 which is NOT in mask.
//!     sdata[tid] += __shfl_down_sync(mask, sdata[tid], 16);
//!     sdata[tid] += __shfl_down_sync(mask, sdata[tid], 8);
//!     // ... etc
//! }
//! ```
//!
//! **Failure mode:** Silent wrong reduction sum. Only at block_size=32 with
//! one block. No crash, no error — undefined values folded into the result.
//!
//! **Source:** <https://github.com/NVIDIA/cuda-samples/issues/398>
//!
//! ## Why Session Types Catch It
//!
//! After the block-level reduction, only a subset of lanes are active.
//! In our type system, you have a `Warp<Lane0>` (or similar subset).
//! `shuffle_down` simply doesn't exist on `Warp<Lane0>` — compile error.
//!
//! Run: `cargo test --example nvidia_cuda_samples_398`

#![allow(clippy::needless_range_loop, clippy::new_without_default)]

use std::marker::PhantomData;

// ============================================================================
// MINIMAL TYPE SYSTEM (self-contained for the example)
// ============================================================================

pub trait ActiveSet: Copy + 'static {
    const MASK: u32;
    const NAME: &'static str;
}

pub trait ComplementOf<Other: ActiveSet>: ActiveSet {}

#[derive(Copy, Clone)]
pub struct Warp<S: ActiveSet> {
    _phantom: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self { Warp { _phantom: PhantomData } }
    pub fn active_mask(&self) -> u32 { S::MASK }
}

#[derive(Copy, Clone, Debug)]
pub struct PerLane<T>(pub [T; 32]);

impl PerLane<i32> {
    pub fn new(arr: [i32; 32]) -> Self { PerLane(arr) }
}

// Active set types
#[derive(Copy, Clone)] pub struct All;
#[derive(Copy, Clone)] pub struct Lane0;
#[derive(Copy, Clone)] pub struct NotLane0;

impl ActiveSet for All      { const MASK: u32 = 0xFFFFFFFF; const NAME: &'static str = "All"; }
impl ActiveSet for Lane0    { const MASK: u32 = 0x00000001; const NAME: &'static str = "Lane0"; }
impl ActiveSet for NotLane0 { const MASK: u32 = 0xFFFFFFFE; const NAME: &'static str = "NotLane0"; }

impl ComplementOf<NotLane0> for Lane0 {}
impl ComplementOf<Lane0> for NotLane0 {}

// ============================================================================
// GATED OPERATIONS: Only Warp<All> has shuffle_down
// ============================================================================

impl Warp<All> {
    /// Diverge: model the `if (tid < blockDim.x / warpSize)` conditional.
    /// When block_size=32, only lane 0 satisfies this.
    pub fn extract_lane0(self) -> (Warp<Lane0>, Warp<NotLane0>) {
        (Warp::new(), Warp::new())
    }

    /// Shuffle down — __shfl_down_sync with full mask.
    /// ONLY available on Warp<All>.
    pub fn shuffle_down(&self, data: PerLane<i32>, delta: u32) -> PerLane<i32> {
        let mut result = [0i32; 32];
        for lane in 0..32u32 {
            let src = lane + delta;
            result[lane as usize] = if src < 32 {
                data.0[src as usize]
            } else {
                0 // out of range
            };
        }
        PerLane(result)
    }

    /// Tree reduction via shuffle_down (correct: requires Warp<All>).
    pub fn warp_reduce_sum(&self, mut data: PerLane<i32>) -> i32 {
        // Standard butterfly reduction
        for delta in [16, 8, 4, 2, 1] {
            let shifted = self.shuffle_down(data, delta);
            let mut sum = [0i32; 32];
            for i in 0..32 {
                sum[i] = data.0[i] + shifted.0[i];
            }
            data = PerLane(sum);
        }
        data.0[0] // lane 0 has the sum
    }
}

pub fn merge<S1, S2>(_left: Warp<S1>, _right: Warp<S2>) -> Warp<All>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
{
    Warp::new()
}

// ============================================================================
// THE BUG (cuda-samples #398)
// ============================================================================

/// What the CUDA code does (translated to our type system).
///
/// This function models reduce7's warp-level reduction after block-level
/// tree reduction. With block_size=32, only tid 0 entered the block reduction,
/// so only lane 0 is "active" in the final warp reduction.
///
/// UNCOMMENT TO SEE COMPILE ERROR:
///
/// ```compile_fail
/// # use nvidia_cuda_samples_398::*;
/// fn buggy_final_warp_reduce(warp: Warp<All>, sdata: PerLane<i32>) -> i32 {
///     // Block reduction narrowed us to just lane 0
///     let (lane0, _rest) = warp.extract_lane0();
///
///     // BUG: Try to shuffle_down on Warp<Lane0>
///     // In CUDA: __shfl_down_sync(ballot_mask, sdata[tid], 16)
///     // where ballot_mask = 1 (only lane 0 active)
///     let shifted = lane0.shuffle_down(sdata, 16);
///     //            ^^^^^ ERROR: no method `shuffle_down` found for `Warp<Lane0>`
///     shifted.0[0]
/// }
/// ```
fn _buggy_version_for_doctest() {}

// ============================================================================
// THE FIX
// ============================================================================

/// Correct approach: check whether reduction is needed.
///
/// If only one lane is active, the reduction is already done — lane 0 has
/// the final value. No shuffle needed.
fn correct_final_reduce(warp: Warp<All>, sdata: PerLane<i32>, active_count: u32) -> i32 {
    if active_count == 1 {
        // Only lane 0 has data. Nothing to reduce.
        sdata.0[0]
    } else {
        // Multiple lanes active — safe to shuffle because warp is Warp<All>.
        warp.warp_reduce_sum(sdata)
    }
}

/// Alternative fix: ensure all lanes participate with zeroed inactive data.
///
/// Instead of reducing on a subset, make inactive lanes contribute zero,
/// then reduce the full warp.
fn correct_reduce_with_identity(warp: Warp<All>, sdata: PerLane<i32>, active_mask: u32) -> i32 {
    // Zero out inactive lanes (identity element for addition)
    let mut cleaned = [0i32; 32];
    for lane in 0..32u32 {
        if active_mask & (1 << lane) != 0 {
            cleaned[lane as usize] = sdata.0[lane as usize];
        }
    }

    // Now safe to reduce — warp is Warp<All>, all lanes participate
    warp.warp_reduce_sum(PerLane(cleaned))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_reduce_single_lane() {
        let warp: Warp<All> = Warp::new();
        let mut sdata = [0i32; 32];
        sdata[0] = 42; // only lane 0 has data

        let result = correct_final_reduce(warp, PerLane(sdata), 1);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_correct_reduce_full_warp() {
        let warp: Warp<All> = Warp::new();
        let mut sdata = [0i32; 32];
        for i in 0..32 {
            sdata[i] = 1; // each lane contributes 1
        }

        let result = correct_final_reduce(warp, PerLane(sdata), 32);
        assert_eq!(result, 32);
    }

    #[test]
    fn test_correct_reduce_with_identity() {
        let warp: Warp<All> = Warp::new();
        let mut sdata = [0i32; 32];
        sdata[0] = 10;
        sdata[1] = 20;
        sdata[2] = 30;

        // Only first 3 lanes active
        let result = correct_reduce_with_identity(warp, PerLane(sdata), 0b111);
        assert_eq!(result, 60);
    }

    #[test]
    fn test_type_system_prevents_bug() {
        let warp: Warp<All> = Warp::new();
        let (lane0, _rest) = warp.extract_lane0();

        // Verify: lane0 is Warp<Lane0>, which has NO shuffle_down method.
        assert_eq!(lane0.active_mask(), 0x00000001);

        // The following would not compile:
        // lane0.shuffle_down(PerLane([0i32; 32]), 16);
        // error[E0599]: no method named `shuffle_down` found for struct `Warp<Lane0>`
    }

    #[test]
    fn test_diverge_merge_roundtrip() {
        let warp: Warp<All> = Warp::new();
        let (lane0, rest) = warp.extract_lane0();

        // Can merge back because Lane0: ComplementOf<NotLane0>
        let restored: Warp<All> = merge(lane0, rest);
        assert_eq!(restored.active_mask(), 0xFFFFFFFF);

        // Now shuffle is available again
        let data = PerLane([1i32; 32]);
        let _shifted = restored.shuffle_down(data, 1);
    }
}

// ============================================================================
// MAIN: Paper-ready demonstration
// ============================================================================

fn main() {
    println!("NVIDIA cuda-samples #398: Reduction Shuffle-Mask Bug");
    println!("====================================================\n");

    println!("The Bug (CUDA):");
    println!("  After block-level tree reduction with block_size=32,");
    println!("  only lane 0 is active. __shfl_down_sync(mask=1, val, 16)");
    println!("  reads from lane 16, which is inactive. Undefined result.\n");

    println!("warp-types' Type System:");
    println!("  After extract_lane0(), you have Warp<Lane0>.");
    println!("  shuffle_down() does not exist on Warp<Lane0>.");
    println!("  Compile error: no method named `shuffle_down`\n");

    println!("The Fix:");
    println!("  Option A: Check active_count == 1, skip reduction.");
    println!("  Option B: Zero inactive lanes, reduce full warp.\n");

    // Demonstrate both fixes
    let warp: Warp<All> = Warp::new();
    let mut sdata = [0i32; 32];
    sdata[0] = 42;

    let result_a = correct_final_reduce(warp, PerLane(sdata), 1);
    println!("Fix A (single-lane check): sum = {}", result_a);

    let result_b = correct_reduce_with_identity(warp, PerLane(sdata), 0x1);
    println!("Fix B (zero-inactive):     sum = {}", result_b);

    assert_eq!(result_a, 42);
    assert_eq!(result_b, 42);
    println!("\nBoth fixes produce correct result: 42");
}
