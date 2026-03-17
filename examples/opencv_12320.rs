//! # OpenCV Issue #12320: Hardcoded Full Mask in warpScanInclusive
//!
//! Demonstrates how warp typestate catches a real deadlock bug from
//! OpenCV's GPU module.
//!
//! ## The Real Bug
//!
//! In OpenCV's `warpScanInclusive` (used by GPU-accelerated reduce, histogram,
//! stereo matching, etc.):
//!
//! ```cuda
//! __device__ unsigned int warpScanInclusive(unsigned int idata, ...) {
//!     unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (OPENCV_CUDA_WARP_SIZE - 1));
//!     s_Data[pos] = 0;
//!     pos += OPENCV_CUDA_WARP_SIZE;
//!     s_Data[pos] = idata;
//!
//!     for (unsigned int offset = 1; offset < OPENCV_CUDA_WARP_SIZE; offset <<= 1)
//!         s_Data[pos] += s_Data[pos - offset];
//!
//!     return s_Data[pos];
//! }
//!
//! // Later rewritten to use shuffle:
//! __device__ unsigned int warpScanInclusive(unsigned int idata) {
//!     unsigned int result = idata;
//!     for (int offset = 1; offset < 32; offset <<= 1) {
//!         unsigned int val = __shfl_up_sync(0xFFFFFFFF, result, offset);
//!         //                               ^^^^^^^^^^^ BUG: hardcoded full mask
//!         if (threadIdx.x % 32 >= offset)
//!             result += val;
//!     }
//!     return result;
//! }
//! ```
//!
//! This function is called from divergent branches throughout OpenCV's GPU code.
//! On Volta+ (independent thread scheduling), threads in the other branch never
//! reach the `__shfl_up_sync` call. The hardcoded `0xFFFFFFFF` mask tells the
//! hardware to wait for all 32 lanes — but some lanes will never arrive.
//! **Result: deadlock (infinite hang).**
//!
//! **Failure mode:** Hard deadlock on Volta, Turing, Ampere, Ada GPUs.
//! Works fine on Pascal and earlier (lockstep masking hides the bug).
//! Multiple reports: #12320, #13014, #13761.
//!
//! **Fix:** Replace `0xFFFFFFFF` with `__activemask()`.
//!
//! **Source:** <https://github.com/opencv/opencv/issues/12320>
//!
//! ## Why `__shfl_up_sync` Doesn't Help
//!
//! The `_sync` suffix is meant to enforce synchronization — all lanes named in
//! the mask must converge before the shuffle executes. But the mask is a runtime
//! `u32`. The programmer wrote `0xFFFFFFFF` (all lanes), even though the
//! function is called from a divergent branch where only a subset of lanes are
//! active. The hardware dutifully waits for all 32 lanes. Some never come.
//! Deadlock.
//!
//! `__activemask()` would return the correct subset, but the programmer used a
//! constant instead. This is the same class of bug as PIConGPU #2514 — but with
//! a worse failure mode (deadlock rather than silent wrong results).
//!
//! ## Why Session Types Catch It
//!
//! After divergence, the warp handle is `Warp<Active>`, not `Warp<All>`.
//! `shuffle_up` only exists on `Warp<All>`. Calling `warpScanInclusive` on a
//! diverged warp is a compile error — the function requires `Warp<All>` and you
//! have `Warp<Active>`. The type IS the mask. There is no `0xFFFFFFFF` to
//! hardcode wrong.
//!
//! Run: `cargo test --example opencv_12320`

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

// Active set types
#[derive(Copy, Clone)] pub struct All;
#[derive(Copy, Clone)] pub struct Active;    // threads that entered the divergent branch
#[derive(Copy, Clone)] pub struct Inactive;  // threads that took the other path

impl ActiveSet for All      { const MASK: u32 = 0xFFFFFFFF; const NAME: &'static str = "All"; }
impl ActiveSet for Active   { const MASK: u32 = 0x0000FFFF; const NAME: &'static str = "Active"; }
impl ActiveSet for Inactive { const MASK: u32 = 0xFFFF0000; const NAME: &'static str = "Inactive"; }

impl ComplementOf<Inactive> for Active {}
impl ComplementOf<Active> for Inactive {}

// ============================================================================
// GATED OPERATIONS: Only Warp<All> has shuffle_up
// ============================================================================

impl Warp<All> {
    /// Diverge: model a conditional branch where only some threads enter.
    /// E.g., `if (tid < threshold)` in OpenCV's scan callers.
    pub fn diverge_on_condition(self) -> (Warp<Active>, Warp<Inactive>) {
        (Warp::new(), Warp::new())
    }

    /// Shuffle up — `__shfl_up_sync` with full mask. ONLY on Warp<All>.
    pub fn shuffle_up(&self, data: &PerLane<u32>, delta: u32) -> PerLane<u32> {
        let mut result = [0u32; 32];
        for lane in 0..32u32 {
            result[lane as usize] = if lane >= delta {
                data.0[(lane - delta) as usize]
            } else {
                data.0[lane as usize] // shfl_up: lane below delta returns own value
            };
        }
        PerLane(result)
    }

    /// Inclusive prefix sum using shuffle — the corrected warpScanInclusive.
    /// Requires Warp<All> because shuffle_up needs all lanes converged.
    pub fn warp_scan_inclusive(&self, idata: &PerLane<u32>) -> PerLane<u32> {
        let mut result = *idata;
        let mut offset = 1u32;
        while offset < 32 {
            let shifted = self.shuffle_up(&result, offset);
            let mut sum = [0u32; 32];
            for lane in 0..32u32 {
                sum[lane as usize] = if lane >= offset {
                    result.0[lane as usize] + shifted.0[lane as usize]
                } else {
                    result.0[lane as usize]
                };
            }
            result = PerLane(sum);
            offset <<= 1;
        }
        result
    }
}

/// Sub-warp inclusive scan: operates only on the active lanes.
/// Does not use shuffle — no risk of deadlock with inactive lanes.
impl Warp<Active> {
    pub fn sub_warp_scan_inclusive(&self, data: &[u32]) -> Vec<u32> {
        let mut result: Vec<u32> = data.to_vec();
        for i in 1..result.len() {
            result[i] += result[i - 1];
        }
        result
    }
}

pub fn merge<S1, S2>(_left: Warp<S1>, _right: Warp<S2>) -> Warp<All>
where S1: ComplementOf<S2>, S2: ActiveSet {
    Warp::new()
}

/// Merge per-lane data from two complementary active sets.
pub fn merge_data(
    active_data: &PerLane<u32>,
    inactive_data: &PerLane<u32>,
    active_mask: u32,
) -> PerLane<u32> {
    let mut result = [0u32; 32];
    for lane in 0..32u32 {
        result[lane as usize] = if active_mask & (1 << lane) != 0 {
            active_data.0[lane as usize]
        } else {
            inactive_data.0[lane as usize]
        };
    }
    PerLane(result)
}

// ============================================================================
// THE BUG (OpenCV #12320)
// ============================================================================

/// What the CUDA code does (translated to our type system).
///
/// `warpScanInclusive` uses `__shfl_up_sync(0xFFFFFFFF, ...)` — hardcoded full
/// mask. When called from a divergent branch, inactive threads never reach the
/// shuffle. On Volta+ (independent thread scheduling), the hardware waits for
/// all 32 lanes. Deadlock.
///
/// UNCOMMENT TO SEE COMPILE ERROR:
///
/// ```compile_fail
/// # use opencv_12320::*;
/// fn buggy_scan_from_divergent_branch(warp: Warp<All>, data: PerLane<u32>) -> PerLane<u32> {
///     // Some threads enter this branch, others don't
///     let (active, _inactive) = warp.diverge_on_condition();
///
///     // BUG: warpScanInclusive uses shuffle_up with 0xFFFFFFFF mask
///     // but we're inside a divergent branch — only Active lanes are here.
///     // On Volta+: deadlock. Hardware waits for Inactive lanes that never arrive.
///     let result = active.warp_scan_inclusive(&data);
///     //           ^^^^^^ ERROR: no method `warp_scan_inclusive` found for `Warp<Active>`
///     result
/// }
/// ```
fn _buggy_version_for_doctest() {}

// ============================================================================
// THE FIX (two approaches)
// ============================================================================

/// Fix A: Merge back to `Warp<All>` before scanning.
///
/// Inactive lanes contribute identity (0 for addition). Then scan the full
/// warp. Active lanes get correct prefix sums; inactive lanes get zeroes.
fn fix_merge_then_scan(
    warp: Warp<All>,
    active_data: PerLane<u32>,
) -> PerLane<u32> {
    let (active, inactive) = warp.diverge_on_condition();

    // Inactive lanes contribute identity element (0)
    let identity = PerLane([0u32; 32]);
    let combined = merge_data(&active_data, &identity, Active::MASK);

    // Merge back to Warp<All> — type system verifies complement
    let full: Warp<All> = merge(active, inactive);

    // Now warp_scan_inclusive is safe: all 32 lanes converge.
    // This is like using __activemask() at the type level —
    // except the type system guarantees correctness, not a runtime query.
    full.warp_scan_inclusive(&combined)
}

/// Fix B: Use sub-warp scan without shuffle.
///
/// Instead of forcing all lanes to converge, use a sub-warp operation that
/// only involves active lanes. No shuffle, no convergence requirement, no
/// deadlock risk. This matches OpenCV's fix: `__activemask()` restricts the
/// operation to participating threads.
fn fix_subwarp_scan(
    warp: Warp<All>,
    active_data: &[u32],
) -> Vec<u32> {
    let (active, _inactive) = warp.diverge_on_condition();

    // Scan only within active lanes — no shuffle, no deadlock
    active.sub_warp_scan_inclusive(active_data)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_warp_scan_inclusive() {
        // Baseline: warp_scan_inclusive works correctly on Warp<All>
        let warp: Warp<All> = Warp::new();
        let data = PerLane([1u32; 32]);

        let result = warp.warp_scan_inclusive(&data);

        // Inclusive prefix sum of [1,1,1,...,1] = [1,2,3,...,32]
        for lane in 0..32 {
            assert_eq!(result.0[lane], (lane + 1) as u32,
                "lane {} expected {} got {}", lane, lane + 1, result.0[lane]);
        }
    }

    #[test]
    fn test_full_warp_scan_with_varying_data() {
        let warp: Warp<All> = Warp::new();
        let mut data_arr = [0u32; 32];
        for i in 0..32 {
            data_arr[i] = (i + 1) as u32; // [1, 2, 3, ..., 32]
        }

        let result = warp.warp_scan_inclusive(&PerLane(data_arr));

        // Inclusive prefix sum of [1,2,3,...,32] = [1, 3, 6, 10, ...]
        let mut expected = 0u32;
        for lane in 0..32 {
            expected += (lane + 1) as u32;
            assert_eq!(result.0[lane], expected,
                "lane {} expected {} got {}", lane, expected, result.0[lane]);
        }
    }

    #[test]
    fn test_type_prevents_scan_on_diverged_warp() {
        let warp: Warp<All> = Warp::new();
        let (active, _inactive) = warp.diverge_on_condition();

        // Verify: active is Warp<Active>, which has NO warp_scan_inclusive
        // and NO shuffle_up method.
        assert_eq!(active.active_mask(), 0x0000FFFF);

        // The following would not compile:
        // active.warp_scan_inclusive(&PerLane([1u32; 32]));
        // error[E0599]: no method named `warp_scan_inclusive` found for `Warp<Active>`
        //
        // active.shuffle_up(&PerLane([1u32; 32]), 1);
        // error[E0599]: no method named `shuffle_up` found for `Warp<Active>`
    }

    #[test]
    fn test_fix_a_merge_then_scan() {
        let warp: Warp<All> = Warp::new();

        // Active lanes (lower 16) each contribute 1
        let mut active_data = [0u32; 32];
        for lane in 0..16 {
            active_data[lane] = 1;
        }

        let result = fix_merge_then_scan(warp, PerLane(active_data));

        // Active lanes (0-15) get prefix sum [1, 2, 3, ..., 16]
        for lane in 0..16 {
            assert_eq!(result.0[lane], (lane + 1) as u32,
                "active lane {} expected {} got {}", lane, lane + 1, result.0[lane]);
        }

        // Inactive lanes (16-31) have 0 data, so their prefix sums carry
        // the total from active lanes (16) forward
        for lane in 16..32 {
            assert_eq!(result.0[lane], 16,
                "inactive lane {} expected 16 got {}", lane, result.0[lane]);
        }
    }

    #[test]
    fn test_fix_b_subwarp_scan() {
        let warp: Warp<All> = Warp::new();

        // Only active lanes' data
        let active_data: Vec<u32> = vec![1; 16];

        let result = fix_subwarp_scan(warp, &active_data);

        // Sub-warp inclusive scan of [1,1,...,1] = [1,2,3,...,16]
        assert_eq!(result.len(), 16);
        for i in 0..16 {
            assert_eq!(result[i], (i + 1) as u32,
                "sub-warp lane {} expected {} got {}", i, i + 1, result[i]);
        }
    }

    #[test]
    fn test_merge_restores_scan_access() {
        let warp: Warp<All> = Warp::new();
        let (active, inactive) = warp.diverge_on_condition();

        // Can merge back because Active: ComplementOf<Inactive>
        let restored: Warp<All> = merge(active, inactive);
        assert_eq!(restored.active_mask(), 0xFFFFFFFF);

        // warp_scan_inclusive is available again after merge
        let data = PerLane([1u32; 32]);
        let result = restored.warp_scan_inclusive(&data);
        assert_eq!(result.0[31], 32); // last lane has full sum
    }

    #[test]
    fn test_why_hardcoded_mask_deadlocks() {
        // The OpenCV bug explained through our type system:
        //
        // 1. warpScanInclusive uses __shfl_up_sync(0xFFFFFFFF, ...).
        //    In our system: shuffle_up requires Warp<All> (mask = 0xFFFFFFFF).
        //
        // 2. But it's called from a divergent branch where only Active lanes
        //    are present. In our system: you have Warp<Active> (mask = 0x0000FFFF).
        //
        // 3. On pre-Volta GPUs, lockstep execution means inactive lanes still
        //    "execute" the instruction (result discarded). No deadlock.
        //    On Volta+, inactive lanes genuinely aren't there. Deadlock.
        //
        // 4. Our type system catches this at compile time regardless of GPU
        //    architecture. The bug is the bug, whether or not hardware masks it.

        let warp: Warp<All> = Warp::new();
        let (active, _) = warp.diverge_on_condition();

        // The type IS the mask — no runtime value to get wrong
        assert_eq!(active.active_mask(), Active::MASK);
        assert_ne!(active.active_mask(), All::MASK);

        // 0xFFFFFFFF (All) vs 0x0000FFFF (Active) — the mismatch is visible
        // in the type system, not buried in a runtime constant.
    }

    #[test]
    fn test_both_fixes_agree() {
        // Both fixes should produce the same prefix sums for active lanes
        let warp_a: Warp<All> = Warp::new();
        let warp_b: Warp<All> = Warp::new();

        let mut active_data_arr = [0u32; 32];
        let active_data_vec: Vec<u32> = (1..=16).collect();
        for i in 0..16 {
            active_data_arr[i] = (i + 1) as u32;
        }

        let result_a = fix_merge_then_scan(warp_a, PerLane(active_data_arr));
        let result_b = fix_subwarp_scan(warp_b, &active_data_vec);

        // Active lanes should match between both fixes
        for i in 0..16 {
            assert_eq!(result_a.0[i], result_b[i],
                "lane {} disagrees: fix_a={}, fix_b={}", i, result_a.0[i], result_b[i]);
        }
    }

    #[test]
    fn test_shuffle_up_semantics() {
        let warp: Warp<All> = Warp::new();
        let mut data = [0u32; 32];
        for i in 0..32 {
            data[i] = i as u32;
        }

        // shuffle_up by 1: each lane gets value from lane - 1
        let shifted = warp.shuffle_up(&PerLane(data), 1);

        // Lane 0 keeps its own value (no lane -1)
        assert_eq!(shifted.0[0], 0);
        // Lane 1 gets lane 0's value
        assert_eq!(shifted.0[1], 0);
        // Lane 2 gets lane 1's value
        assert_eq!(shifted.0[2], 1);
        // Lane 31 gets lane 30's value
        assert_eq!(shifted.0[31], 30);
    }
}

// ============================================================================
// MAIN: Paper-ready demonstration
// ============================================================================

fn main() {
    println!("OpenCV #12320: Hardcoded Full Mask in warpScanInclusive");
    println!("=======================================================\n");

    println!("The Bug (CUDA):");
    println!("  warpScanInclusive uses __shfl_up_sync(0xFFFFFFFF, ...).");
    println!("  Called from divergent branches throughout OpenCV's GPU code.");
    println!("  On Volta+: inactive threads never reach the shuffle.");
    println!("  Hardware waits for all 32 lanes. Deadlock. Infinite hang.\n");

    println!("Why __shfl_up_sync Doesn't Help:");
    println!("  The mask is a runtime u32. Programmer wrote 0xFFFFFFFF.");
    println!("  Hardware trusts the mask — waits for all named lanes.");
    println!("  __activemask() would return the correct subset, but");
    println!("  the programmer used a constant instead.\n");

    println!("Why Session Types Catch It:");
    println!("  After divergence: Warp<Active>, not Warp<All>.");
    println!("  warp_scan_inclusive() requires Warp<All>. Compile error.");
    println!("  No runtime mask to hardcode wrong.\n");

    // Demonstrate both fixes
    let warp: Warp<All> = Warp::new();
    let mut active_data = [0u32; 32];
    for i in 0..16 {
        active_data[i] = 1;
    }

    println!("Fix A (merge, then full-warp scan):");
    let result_a = fix_merge_then_scan(warp, PerLane(active_data));
    print!("  Active lanes:   ");
    for i in 0..16 {
        print!("{} ", result_a.0[i]);
    }
    println!();

    println!("\nFix B (sub-warp scan, no shuffle):");
    let result_b = fix_subwarp_scan(Warp::<All>::new(), &vec![1u32; 16]);
    print!("  Active lanes:   ");
    for v in &result_b {
        print!("{} ", v);
    }
    println!();

    println!("\nBoth produce correct inclusive prefix sum: [1, 2, 3, ..., 16]");
    println!("Pre-Volta: bug hidden by lockstep. Volta+: deadlock. Types: compile error.");
}
