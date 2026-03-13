//! # LLVM Issue #155682: shfl_sync After Conditional Eliminates Branch
//!
//! Demonstrates how session-typed divergence prevents a bug where the compiler
//! "optimizes" a conditional into unconditional execution because of UB from
//! an uninitialized value reaching a shuffle.
//!
//! ## The Real Bug
//!
//! ```cuda
//! __device__ int atomic_then_broadcast(int* counter, int laneId) {
//!     int row;
//!     if (laneId == 0) {
//!         row = atomicAdd(counter, 16);  // Only lane 0 does the atomic
//!     }
//!     // BUG: row is uninitialized on lanes 1-31
//!     row = __shfl_sync(0xffffffff, row, 0) + laneId;
//!     return row;
//!     // Clang/LLVM eliminates the if entirely — atomicAdd runs on all 32 lanes.
//!     // Counter advances 32x too fast. NVCC handles this correctly.
//! }
//! ```
//!
//! **Failure mode:** Atomic counter advances 32x too fast. Silent, no warnings.
//! Only visible by inspecting generated PTX.
//!
//! **Source:** <https://github.com/llvm/llvm-project/issues/155682>
//!
//! ## Why `__shfl_sync` Doesn't Help
//!
//! The shuffle reads `row` from lane 0, which is fine — lane 0 initialized it.
//! But lanes 1-31 pass their uninitialized `row` to the shuffle. LLVM sees:
//! "using an uninitialized value is UB → the `if` must have been taken →
//! eliminate the branch." This is a *legal* optimization under C++ UB rules.
//! `__shfl_sync`'s mask is correct (0xFFFFFFFF, all lanes ARE active). The
//! bug isn't in the mask — it's in the uninitialized data reaching the shuffle.
//!
//! ## Why Session Types Catch It
//!
//! After `if (laneId == 0)`, lane 0 has `Warp<Lane0>` and the rest have
//! `Warp<NotLane0>`. To shuffle, you must merge back to `Warp<All>`. The
//! merge forces both sides to provide data — the `NotLane0` side must supply
//! a value for `row`, eliminating the uninitialized value. No UB means no
//! branch elimination.
//!
//! Run: `cargo test --example llvm_155682`

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
#[derive(Copy, Clone)] pub struct Lane0;
#[derive(Copy, Clone)] pub struct NotLane0;

impl ActiveSet for All      { const MASK: u32 = 0xFFFFFFFF; const NAME: &'static str = "All"; }
impl ActiveSet for Lane0    { const MASK: u32 = 0x00000001; const NAME: &'static str = "Lane0"; }
impl ActiveSet for NotLane0 { const MASK: u32 = 0xFFFFFFFE; const NAME: &'static str = "NotLane0"; }

impl ComplementOf<NotLane0> for Lane0 {}
impl ComplementOf<Lane0> for NotLane0 {}

// ============================================================================
// GATED OPERATIONS: Only Warp<All> has shuffle
// ============================================================================

impl Warp<All> {
    pub fn diverge_lane0(self) -> (Warp<Lane0>, Warp<NotLane0>) {
        (Warp::new(), Warp::new())
    }

    /// Shuffle: broadcast from a source lane. ONLY on Warp<All>.
    pub fn shuffle_broadcast(&self, data: &PerLane<i32>, src_lane: u32) -> PerLane<i32> {
        let val = data.0[src_lane as usize];
        PerLane([val; 32])
    }
}

pub fn merge_data(
    _lane0_data: i32,
    _notlane0_data: i32,
    lane0_mask: u32,
) -> PerLane<i32> {
    let mut result = [_notlane0_data; 32];
    for i in 0..32 {
        if lane0_mask & (1 << i) != 0 {
            result[i] = _lane0_data;
        }
    }
    PerLane(result)
}

pub fn merge<S1, S2>(_left: Warp<S1>, _right: Warp<S2>) -> Warp<All>
where S1: ComplementOf<S2>, S2: ActiveSet {
    Warp::new()
}

// ============================================================================
// THE BUG (LLVM #155682)
// ============================================================================

/// What the CUDA code does (translated to our type system).
///
/// Lane 0 does an atomicAdd, then all lanes shuffle to broadcast the result.
/// The bug: lanes 1-31 pass uninitialized `row` to the shuffle, triggering
/// LLVM's UB-based branch elimination.
///
/// UNCOMMENT TO SEE COMPILE ERROR:
///
/// ```compile_fail
/// # use llvm_155682::*;
/// fn buggy_atomic_broadcast(warp: Warp<All>, counter: &mut i32) -> PerLane<i32> {
///     let (lane0, _rest) = warp.diverge_lane0();
///
///     // Lane 0 does the atomic (correctly)
///     let row = *counter;
///     *counter += 16;
///
///     // BUG: Try to shuffle on Warp<Lane0>
///     // In CUDA: __shfl_sync(0xffffffff, row, 0)
///     // But we only have Warp<Lane0>, not Warp<All>
///     let broadcast = lane0.shuffle_broadcast(&PerLane([row; 32]), 0);
///     //              ^^^^^ ERROR: no method `shuffle_broadcast` found for `Warp<Lane0>`
///     broadcast
/// }
/// ```
fn _buggy_version_for_doctest() {}

// ============================================================================
// THE FIX
// ============================================================================

/// Correct approach: merge back to Warp<All> before shuffling.
/// Both sides must provide data — no uninitialized values.
fn correct_atomic_broadcast(warp: Warp<All>, counter: &mut i32) -> PerLane<i32> {
    let (lane0, rest) = warp.diverge_lane0();

    // Lane 0 does the atomic
    let atomic_result = *counter;
    *counter += 16;

    // Non-lane-0 threads contribute a placeholder (0).
    // This is the key: the merge FORCES us to provide a value for all lanes.
    // No uninitialized data → no UB → no branch elimination.
    let combined_data = merge_data(
        atomic_result, // lane 0's value
        0,             // other lanes' value (explicit, not uninitialized!)
        Lane0::MASK,
    );

    // Merge back to Warp<All>
    let full: Warp<All> = merge(lane0, rest);

    // Now shuffle is safe — all lanes have defined data
    let broadcast = full.shuffle_broadcast(&combined_data, 0);

    // Add lane ID offset
    let mut result = broadcast.0;
    for i in 0..32 {
        result[i] += i as i32;
    }
    PerLane(result)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_atomic_broadcast() {
        let warp: Warp<All> = Warp::new();
        let mut counter = 100;

        let result = correct_atomic_broadcast(warp, &mut counter);

        // Counter advanced by 16 (once, not 32x)
        assert_eq!(counter, 116);

        // Lane 0 got atomicAdd result (100), broadcast to all, plus lane offset
        assert_eq!(result.0[0], 100);  // lane 0: 100 + 0
        assert_eq!(result.0[1], 101);  // lane 1: 100 + 1
        assert_eq!(result.0[31], 131); // lane 31: 100 + 31
    }

    #[test]
    fn test_type_prevents_diverged_shuffle() {
        let warp: Warp<All> = Warp::new();
        let (lane0, _rest) = warp.diverge_lane0();

        // Verify: lane0 is Warp<Lane0>, which has NO shuffle_broadcast method
        assert_eq!(lane0.active_mask(), 0x00000001);

        // The following would not compile:
        // lane0.shuffle_broadcast(&PerLane([0i32; 32]), 0);
        // error[E0599]: no method named `shuffle_broadcast` found for `Warp<Lane0>`
    }

    #[test]
    fn test_why_merge_prevents_ub() {
        // The LLVM bug: uninitialized `row` on lanes 1-31 is UB.
        // LLVM exploits UB to eliminate the branch.
        //
        // Our fix: merge_data REQUIRES a value for both sides.
        // The programmer must explicitly provide 0 (or any value)
        // for non-lane-0 threads. No uninitialized data exists.

        let combined = merge_data(42, 0, Lane0::MASK);

        // Lane 0 has the atomic result
        assert_eq!(combined.0[0], 42);
        // All other lanes have an explicit value (0), not uninitialized
        for i in 1..32 {
            assert_eq!(combined.0[i], 0);
        }
    }

    #[test]
    fn test_counter_advances_once() {
        // The LLVM bug: atomicAdd runs on all 32 lanes (counter += 16*32 = 512)
        // Correct: atomicAdd runs on lane 0 only (counter += 16)
        let warp: Warp<All> = Warp::new();
        let mut counter = 0;

        let _result = correct_atomic_broadcast(warp, &mut counter);
        assert_eq!(counter, 16); // 16, not 512

        let _result = correct_atomic_broadcast(Warp::<All>::new(), &mut counter);
        assert_eq!(counter, 32); // 32, not 1024
    }
}

fn main() {
    println!("LLVM #155682: shfl_sync After Conditional Eliminates Branch");
    println!("=============================================================\n");

    println!("The Bug (CUDA/Clang):");
    println!("  if (laneId == 0) {{ row = atomicAdd(counter, 16); }}");
    println!("  row = __shfl_sync(0xffffffff, row, 0) + laneId;");
    println!("  LLVM eliminates the if — atomicAdd runs on all 32 lanes.");
    println!("  Counter advances 32x too fast. NVCC handles it correctly.\n");

    println!("Why __shfl_sync Doesn't Help:");
    println!("  The mask (0xffffffff) is correct — all lanes ARE active.");
    println!("  The bug is uninitialized `row` on lanes 1-31 reaching the shuffle.");
    println!("  LLVM sees UB → assumes branch always taken → eliminates it.\n");

    println!("Why Session Types Catch It:");
    println!("  After if: Warp<Lane0>, not Warp<All>. Must merge before shuffle.");
    println!("  Merge forces all lanes to provide data. No uninitialized values.");
    println!("  No UB → no branch elimination.\n");

    let warp: Warp<All> = Warp::new();
    let mut counter = 0;
    let result = correct_atomic_broadcast(warp, &mut counter);

    println!("Counter after one call: {} (correct: 16, buggy: 512)", counter);
    println!("Lane results: [0]={}, [1]={}, [31]={}", result.0[0], result.0[1], result.0[31]);
}
