//! # Apache TVM PR #17307: LowerThreadAllreduce Warp Reduction Mask
//!
//! Demonstrates how warp typestate prevents a bug where a compiler
//! pass computes a group-based sub-mask for `shfl_down_sync` that triggers
//! "CUDA illegal instruction" at runtime.
//!
//! ## The Real Bug
//!
//! TVM's `LowerThreadAllreduce` performs warp-level reductions using
//! `tvm_warp_shuffle_down`. When multiple groups share a warp (e.g., 4 groups
//! of 8 threads in one 32-thread warp), the pass computed a sub-group mask:
//!
//! ```text
//! // C++ from lower_thread_allreduce.cc:
//! mask = activemask & (((1 << reduce_extent) - 1) << (reduce_extent * group_index));
//!
//! // Example: 4 groups of 8 threads in a 32-thread warp
//! //   group 0: mask = activemask & 0x000000FF  (lanes 0-7)
//! //   group 1: mask = activemask & 0x0000FF00  (lanes 8-15)
//! //   group 2: mask = activemask & 0x00FF0000  (lanes 16-23)
//! //   group 3: mask = activemask & 0xFF000000  (lanes 24-31)
//! ```
//!
//! **Failure mode:** "CUDA illegal instruction" on NVIDIA H100. Also confirmed
//! broken on RTX 4090, AMD Radeon 7900 XTX, and Apple M2 Ultra. The computed
//! sub-group mask does not match what the hardware expects — the shuffle
//! requires the full warp mask, not a narrowed sub-group mask.
//!
//! **Source:** <https://github.com/apache/tvm/pull/17307>
//!
//! ## Why the Computed Mask Fails
//!
//! `__shfl_down_sync(mask, val, delta)` requires that all threads identified
//! in `mask` actually execute the instruction. The sub-group mask claims only
//! 8 of 32 lanes participate, but all 32 lanes in the warp are live. The
//! hardware sees a mask that disagrees with the actual execution state and
//! faults. The fix: remove the mask narrowing entirely, use the full active
//! mask, and let `shfl_down`'s `width` parameter handle the sub-group
//! boundary (values don't cross group boundaries because the shuffle wraps
//! at `width`).
//!
//! ## Why Session Types Catch It
//!
//! Computing a sub-group mask produces `Warp<SubGroup>`, not `Warp<All>`.
//! `shuffle_down` only exists on `Warp<All>`. The type system rejects the
//! narrowed mask at compile time — there is no way to call `shuffle_down`
//! on a sub-group handle. The correct approach uses `Warp<All>` with a
//! `width` parameter that confines the reduction to group boundaries without
//! narrowing the mask.
//!
//! Run: `cargo test --example tvm_17307`

use std::marker::PhantomData;

// ============================================================================
// MINIMAL TYPE SYSTEM (self-contained for the example)
// ============================================================================

pub trait ActiveSet: Copy + 'static {
    const MASK: u32;
    const NAME: &'static str;
}

#[derive(Copy, Clone)]
pub struct Warp<S: ActiveSet> {
    _phantom: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self { Warp { _phantom: PhantomData } }
    pub fn active_mask(&self) -> u32 { S::MASK }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerLane<T>(pub [T; 32]);

// Active set types
#[derive(Copy, Clone)] pub struct All;
#[derive(Copy, Clone)] pub struct SubGroup0; // lanes 0-7
#[derive(Copy, Clone)] pub struct SubGroup1; // lanes 8-15
#[derive(Copy, Clone)] pub struct SubGroup2; // lanes 16-23
#[derive(Copy, Clone)] pub struct SubGroup3; // lanes 24-31

impl ActiveSet for All       { const MASK: u32 = 0xFFFFFFFF; const NAME: &'static str = "All"; }
impl ActiveSet for SubGroup0 { const MASK: u32 = 0x000000FF; const NAME: &'static str = "SubGroup0"; }
impl ActiveSet for SubGroup1 { const MASK: u32 = 0x0000FF00; const NAME: &'static str = "SubGroup1"; }
impl ActiveSet for SubGroup2 { const MASK: u32 = 0x00FF0000; const NAME: &'static str = "SubGroup2"; }
impl ActiveSet for SubGroup3 { const MASK: u32 = 0xFF000000; const NAME: &'static str = "SubGroup3"; }

// ============================================================================
// GATED OPERATIONS: Only Warp<All> has shuffle_down
// ============================================================================

impl Warp<All> {
    /// Narrow the warp to a computed sub-group mask.
    /// This is what TVM's buggy code did: compute a mask for each group.
    pub fn narrow_to_subgroup0(self) -> Warp<SubGroup0> { Warp::new() }
    pub fn narrow_to_subgroup1(self) -> Warp<SubGroup1> { Warp::new() }

    /// Shuffle down with width — the correct primitive.
    ///
    /// `width` confines the shuffle to sub-group boundaries: lane `i`
    /// reads from lane `i + delta` only if both are in the same width-sized
    /// group. Values don't cross group boundaries. The MASK remains full
    /// (all lanes participate in the instruction).
    pub fn shuffle_down(&self, data: &PerLane<f32>, delta: u32, width: u32) -> PerLane<f32> {
        let mut result = [0.0f32; 32];
        for lane in 0..32u32 {
            // Sub-group boundaries: lane's group starts at (lane / width) * width
            let group_start = (lane / width) * width;
            let src = lane + delta;
            result[lane as usize] = if src < group_start + width {
                data.0[src as usize]
            } else {
                0.0 // out of sub-group range
            };
        }
        PerLane(result)
    }

    /// Full-warp allreduce (sum) using shuffle_down with width.
    ///
    /// Each width-sized group is reduced independently. All 32 lanes execute
    /// every shuffle instruction — the `width` parameter handles the boundary,
    /// not the mask.
    pub fn allreduce_sum(&self, data: &PerLane<f32>, group_size: u32) -> PerLane<f32> {
        let mut acc = *data;
        let mut delta = group_size / 2;
        while delta >= 1 {
            let shifted = self.shuffle_down(&acc, delta, group_size);
            for i in 0..32 {
                acc.0[i] += shifted.0[i];
            }
            delta /= 2;
        }
        acc
    }
}

// ============================================================================
// THE BUG (TVM PR #17307)
// ============================================================================

/// What TVM's LowerThreadAllreduce did (translated to our type system).
///
/// The pass computed a sub-group mask to restrict the shuffle:
/// ```text
/// mask = activemask & (((1 << reduce_extent) - 1) << (reduce_extent * group_index));
/// ```
/// This narrows the mask from 0xFFFFFFFF to e.g. 0x000000FF for group 0.
/// The narrowed mask produces `Warp<SubGroup0>`, not `Warp<All>`.
///
/// UNCOMMENT TO SEE COMPILE ERROR:
///
/// ```compile_fail
/// # use tvm_17307::*;
/// fn buggy_allreduce(warp: Warp<All>, data: &PerLane<f32>) -> PerLane<f32> {
///     // TVM computed a sub-group mask for each group
///     let subgroup = warp.narrow_to_subgroup0();
///
///     // BUG: Try to shuffle_down on the sub-group handle
///     // This is exactly what triggered "CUDA illegal instruction" on H100
///     let shifted = subgroup.shuffle_down(data, 4, 8);
///     //            ^^^^^^^^ ERROR: no method `shuffle_down` found for `Warp<SubGroup0>`
///     *data
/// }
/// ```
fn _buggy_version_for_doctest() {}

/// Simulates the buggy mask computation that TVM performed.
///
/// This function shows what happens at the VALUE level when you compute
/// the sub-group mask. The mask narrows from 0xFFFFFFFF to a subset,
/// but all 32 lanes are actually executing. The hardware faults because
/// the mask disagrees with reality.
fn buggy_compute_mask(group_index: u32, reduce_extent: u32) -> u32 {
    let activemask: u32 = 0xFFFFFFFF; // all 32 lanes are alive
    // TVM's mask computation (from lower_thread_allreduce.cc):
    activemask & (((1u32 << reduce_extent) - 1) << (reduce_extent * group_index))
}

// ============================================================================
// THE FIX
// ============================================================================

/// Correct approach: use Warp<All> with `width` to confine the reduction.
///
/// This is what TVM PR #17307 fixed: remove the mask narrowing and use the
/// full active mask. The `width` parameter on `shuffle_down` handles the
/// sub-group boundary — values don't cross group boundaries because the
/// shuffle wraps at `width`.
fn correct_allreduce(
    warp: Warp<All>,
    data: &PerLane<f32>,
    group_size: u32,
) -> PerLane<f32> {
    // No mask narrowing. Warp<All> means all lanes participate.
    // The `group_size` parameter confines each group's reduction.
    warp.allreduce_sum(data, group_size)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buggy_mask_narrows_incorrectly() {
        // TVM's mask computation for 4 groups of 8 in a 32-thread warp
        let reduce_extent = 8u32;

        let mask0 = buggy_compute_mask(0, reduce_extent);
        let mask1 = buggy_compute_mask(1, reduce_extent);
        let mask2 = buggy_compute_mask(2, reduce_extent);
        let mask3 = buggy_compute_mask(3, reduce_extent);

        // Each mask covers only 8 of 32 lanes
        assert_eq!(mask0, 0x000000FF); // lanes 0-7
        assert_eq!(mask1, 0x0000FF00); // lanes 8-15
        assert_eq!(mask2, 0x00FF0000); // lanes 16-23
        assert_eq!(mask3, 0xFF000000); // lanes 24-31

        // But all 32 lanes are alive! The mask lies about participation.
        // Hardware expects mask == activemask when all lanes execute the
        // instruction. Mismatch triggers "CUDA illegal instruction" on H100.
        assert_eq!(mask0.count_ones(), 8);  // claims 8 participate
        assert_ne!(mask0, 0xFFFFFFFF);       // but 32 actually do
    }

    #[test]
    fn test_correct_allreduce_groups_of_8() {
        // 4 groups of 8 threads in one 32-thread warp
        // Each group sums its own lanes independently
        let warp: Warp<All> = Warp::new();
        let mut data = [0.0f32; 32];
        for i in 0..32 {
            data[i] = 1.0; // each lane contributes 1.0
        }

        let result = correct_allreduce(warp, &PerLane(data), 8);

        // Lane 0 of each group has the group sum (8.0)
        assert_eq!(result.0[0], 8.0);   // group 0 leader
        assert_eq!(result.0[8], 8.0);   // group 1 leader
        assert_eq!(result.0[16], 8.0);  // group 2 leader
        assert_eq!(result.0[24], 8.0);  // group 3 leader
    }

    #[test]
    fn test_correct_allreduce_groups_of_16() {
        // 2 groups of 16 threads — another case from TVM's test suite
        let warp: Warp<All> = Warp::new();
        let mut data = [0.0f32; 32];
        for i in 0..32 {
            data[i] = (i + 1) as f32; // lanes 1..32
        }

        let result = correct_allreduce(warp, &PerLane(data), 16);

        // Group 0 (lanes 0-15): sum of 1+2+...+16 = 136
        assert_eq!(result.0[0], 136.0);
        // Group 1 (lanes 16-31): sum of 17+18+...+32 = 392
        assert_eq!(result.0[16], 392.0);
    }

    #[test]
    fn test_correct_allreduce_full_warp() {
        // Single group of 32 — no sub-groups
        let warp: Warp<All> = Warp::new();
        let data = [1.0f32; 32];

        let result = correct_allreduce(warp, &PerLane(data), 32);

        assert_eq!(result.0[0], 32.0);
    }

    #[test]
    fn test_type_prevents_subgroup_shuffle() {
        let warp: Warp<All> = Warp::new();
        let subgroup = warp.narrow_to_subgroup0();

        // Verify: subgroup is Warp<SubGroup0>, which has NO shuffle_down method
        assert_eq!(subgroup.active_mask(), 0x000000FF);

        // The following would not compile:
        // subgroup.shuffle_down(&PerLane([0.0f32; 32]), 4, 8);
        // error[E0599]: no method named `shuffle_down` found for `Warp<SubGroup0>`
    }

    #[test]
    fn test_fix_uses_full_mask() {
        // The fix: use Warp<All> (full active mask) instead of a narrowed mask.
        // The width parameter handles sub-group boundaries.
        let warp: Warp<All> = Warp::new();

        // Full mask — all 32 lanes participate in the instruction
        assert_eq!(warp.active_mask(), 0xFFFFFFFF);

        // Width = 8 confines the reduction to 8-lane groups
        // without narrowing the mask
        let data = PerLane([2.0f32; 32]);
        let result = warp.shuffle_down(&data, 4, 8);

        // Lane 0: reads from lane 4 (same group, 0+4 < 8) → 2.0
        assert_eq!(result.0[0], 2.0);
        // Lane 4: reads from lane 8 (crosses group boundary, 4+4 >= 8) → 0.0
        assert_eq!(result.0[4], 0.0);
        // Lane 8: reads from lane 12 (same group, 8+4 < 16) → 2.0
        assert_eq!(result.0[8], 2.0);
    }

    #[test]
    fn test_subgroup_mask_vs_full_mask_semantics() {
        // This test shows WHY the sub-group mask is wrong at the hardware level.
        //
        // When TVM computed mask = 0x000000FF for group 0:
        //   - It told the hardware "only lanes 0-7 are participating"
        //   - But lanes 8-31 are also executing the shfl_down instruction
        //   - Hardware sees 32 lanes execute with a mask claiming 8
        //   - Result: "CUDA illegal instruction"
        //
        // When the fix uses mask = 0xFFFFFFFF:
        //   - It tells the hardware "all 32 lanes participate"
        //   - All 32 lanes do execute the instruction
        //   - Mask matches reality — no fault
        //   - The `width` parameter (not the mask) handles group boundaries

        let buggy_mask = buggy_compute_mask(0, 8);
        let correct_mask = All::MASK;

        // Buggy: only 8 bits set, but 32 lanes execute
        assert_eq!(buggy_mask.count_ones(), 8);
        // Correct: 32 bits set, 32 lanes execute
        assert_eq!(correct_mask.count_ones(), 32);
    }

    #[test]
    fn test_width_confines_groups_correctly() {
        // The width parameter is what actually handles sub-group boundaries.
        // Each group reduces independently without mask narrowing.
        let warp: Warp<All> = Warp::new();

        // 4 groups, each with values [10, 20, 30, 40, 50, 60, 70, 80]
        let mut data = [0.0f32; 32];
        for group in 0..4u32 {
            for lane in 0..8u32 {
                data[(group * 8 + lane) as usize] = ((lane + 1) * 10) as f32;
            }
        }

        let result = correct_allreduce(warp, &PerLane(data), 8);

        // Each group leader (lane 0, 8, 16, 24) has sum = 10+20+...+80 = 360
        let expected_sum = (1..=8).map(|x| x as f32 * 10.0).sum::<f32>();
        assert_eq!(expected_sum, 360.0);
        assert_eq!(result.0[0], expected_sum);
        assert_eq!(result.0[8], expected_sum);
        assert_eq!(result.0[16], expected_sum);
        assert_eq!(result.0[24], expected_sum);
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("Apache TVM PR #17307: LowerThreadAllreduce Warp Reduction Mask");
    println!("===============================================================\n");

    println!("The Bug (TVM compiler pass):");
    println!("  LowerThreadAllreduce computed a sub-group mask for shfl_down:");
    println!("    mask = activemask & (((1 << extent) - 1) << (extent * group_idx))");
    println!("  Example: 4 groups of 8 in a 32-thread warp:");

    for group in 0..4u32 {
        let mask = buggy_compute_mask(group, 8);
        println!("    group {}: mask = 0x{:08X} ({} of 32 lanes)", group, mask, mask.count_ones());
    }

    println!("\n  All 32 lanes execute the shfl_down instruction, but the mask");
    println!("  claims only 8 participate. Hardware faults on H100, RTX 4090,");
    println!("  AMD 7900 XTX, Apple M2 Ultra.\n");

    println!("Why Session Types Catch It:");
    println!("  Narrowing the mask produces Warp<SubGroup0>, not Warp<All>.");
    println!("  shuffle_down() only exists on Warp<All>. Compile error.");
    println!("  No runtime mask to get wrong — the type IS the mask.\n");

    println!("The Fix (PR #17307):");
    println!("  Remove mask computation. Use full active mask (Warp<All>).");
    println!("  The `width` parameter on shuffle_down confines groups.\n");

    let warp: Warp<All> = Warp::new();
    let data = PerLane([1.0f32; 32]);

    let result = correct_allreduce(warp, &data, 8);
    println!("Correct allreduce (4 groups of 8, each lane = 1.0):");
    println!("  Group 0 sum (lane 0):  {}", result.0[0]);
    println!("  Group 1 sum (lane 8):  {}", result.0[8]);
    println!("  Group 2 sum (lane 16): {}", result.0[16]);
    println!("  Group 3 sum (lane 24): {}", result.0[24]);
}
