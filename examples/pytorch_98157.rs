//! # PyTorch Issue #98157: `__activemask()` in `countRadixUsingMask`
//!
//! Demonstrates how warp typestate catches a real bug in PyTorch's
//! radix sort used by `topk`, `kthvalue`, and `median`.
//!
//! ## The Real Bug
//!
//! In `aten/src/ATen/native/cuda/Sort.cuh`, `countRadixUsingMask()`:
//!
//! ```cuda
//! __device__ void countRadixUsingMask(
//!     unsigned* counts, unsigned* desired,
//!     unsigned desiredMask, int radixDigitPos,
//!     int sliceSize, IndexType withinSliceStride,
//!     ...) {
//!   for (int i = threadIdx.x; i < sliceSize; i += blockDim.x) {
//!     unsigned val = getBitfield(doLdg(&data[i * withinSliceStride]),
//!                                radixDigitPos, RADIX_BITS);
//!     #pragma unroll
//!     for (int j = 0; j < RADIX_SIZE; ++j) {
//!       bool vote = (val == j);
//!       // BUG: __activemask() captures HARDWARE state, not convergence
//!       counts[j] += __popc(__ballot_sync(__activemask(), vote));
//!     }
//!   }
//! }
//! ```
//!
//! **Failure mode:** `__activemask()` does not guarantee that all threads on the
//! same execution path are in the same active group. The NVIDIA PTX ISA spec
//! warns that a thread executing `activemask` "may find that other threads
//! that it was converged with have diverged" (paraphrased from PTX ISA §9.7.12). On Volta+ with
//! independent thread scheduling, the hardware scheduler can split threads that
//! are logically converged across different scheduling groups. The ballot counts
//! will be **wrong** — some votes are missing because logically-converged threads
//! are in different hardware groups. This causes wrong distribution counts in
//! radix select, making `topk`, `kthvalue`, and `median` return wrong results.
//!
//! A second consequence: `findPattern()` uses the wrong counts to select the
//! k-th element, producing a **data race** when multiple threads disagree about
//! which radix bucket to descend into.
//!
//! **Source:** <https://github.com/pytorch/pytorch/issues/98157>
//!
//! ## Why `__activemask()` Is Unsafe
//!
//! `__activemask()` is a hardware query: "which threads happen to be executing
//! right now?" This is NOT the same as "which threads are on this code path."
//! On Volta+ GPUs with independent thread scheduling, two threads on the same
//! code path can be in different scheduling groups, giving different activemask
//! results. Using `__activemask()` as the ballot mask under-counts votes.
//!
//! ## Why Session Types Catch It
//!
//! In our type system, there is no `activemask()` function that returns a runtime
//! value. The active set is the type parameter `S` in `Warp<S>`. You cannot
//! construct an active set from hardware state — you can only refine it through
//! `diverge()` (which produces complementary typed sub-warps) or prove convergence
//! through `merge()` (which recovers `Warp<All>`).
//!
//! - `ballot()` requires `Warp<All>` — compile error on any subset.
//! - The fix: either prove convergence (`Warp<All>`) or use `DynWarp` with
//!   runtime checking.
//!
//! Run: `cargo test --example pytorch_98157`

#![allow(dead_code)]

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
#[derive(Copy, Clone)] pub struct LoopActive;    // threads still in the loop
#[derive(Copy, Clone)] pub struct LoopExited;    // threads that exited early

impl ActiveSet for All        { const MASK: u32 = 0xFFFFFFFF; const NAME: &'static str = "All"; }
impl ActiveSet for LoopActive { const MASK: u32 = 0x00FFFFFF; const NAME: &'static str = "LoopActive"; }
impl ActiveSet for LoopExited { const MASK: u32 = 0xFF000000; const NAME: &'static str = "LoopExited"; }

impl ComplementOf<LoopExited> for LoopActive {}
impl ComplementOf<LoopActive> for LoopExited {}

// ============================================================================
// RADIX HELPERS
// ============================================================================

const RADIX_BITS: u32 = 2;
const RADIX_SIZE: usize = 1 << RADIX_BITS; // 4 buckets

/// Extract a radix digit from a value.
fn get_bitfield(val: u32, pos: u32, bits: u32) -> u32 {
    (val >> pos) & ((1 << bits) - 1)
}

// ============================================================================
// GATED OPERATIONS: Only Warp<All> has ballot
// ============================================================================

impl Warp<All> {
    /// Diverge: model loop iteration where some threads exit early
    /// (sliceSize not a multiple of blockDim.x, or data-dependent exit).
    pub fn diverge_loop(self) -> (Warp<LoopActive>, Warp<LoopExited>) {
        (Warp::new(), Warp::new())
    }

    /// Ballot — __ballot_sync with provably-converged mask. ONLY on Warp<All>.
    ///
    /// This is the key: ballot requires ALL lanes to participate.
    /// The mask is the type `All`, not a runtime value from `__activemask()`.
    pub fn ballot(&self, predicate: &[bool; 32]) -> u32 {
        let mut result = 0u32;
        for i in 0..32 {
            if predicate[i] { result |= 1 << i; }
        }
        result
    }

    /// Shuffle broadcast from a source lane. ONLY on Warp<All>.
    pub fn shuffle_broadcast(&self, data: &[u32; 32], src_lane: u32) -> [u32; 32] {
        [data[src_lane as usize]; 32]
    }
}

pub fn merge<S1, S2>(_left: Warp<S1>, _right: Warp<S2>) -> Warp<All>
where S1: ComplementOf<S2>, S2: ActiveSet {
    Warp::new()
}

// ============================================================================
// DYNAMICALLY-CHECKED WARP (models __activemask() pattern)
// ============================================================================

/// DynWarp — runtime-checked warp, analogous to using __activemask().
///
/// Unlike `Warp<S>`, the mask is a runtime value. Every collective operation
/// checks the mask at runtime. This catches the PyTorch bug at runtime
/// instead of at compile time — a stepping stone toward full static typing.
#[derive(Clone, Debug)]
pub struct DynWarp {
    active_mask: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarpError {
    pub operation: &'static str,
    pub expected_mask: u32,
    pub actual_mask: u32,
}

impl std::fmt::Display for WarpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: expected mask 0x{:08X}, got 0x{:08X}",
               self.operation, self.expected_mask, self.actual_mask)
    }
}

impl DynWarp {
    pub fn all() -> Self { DynWarp { active_mask: 0xFFFFFFFF } }
    pub fn from_mask(mask: u32) -> Self { DynWarp { active_mask: mask } }
    pub fn active_mask(&self) -> u32 { self.active_mask }
    pub fn population(&self) -> u32 { self.active_mask.count_ones() }

    /// Ballot — runtime check for all-active.
    pub fn ballot(&self, predicate: &[bool; 32]) -> Result<u32, WarpError> {
        if self.active_mask != 0xFFFFFFFF {
            return Err(WarpError {
                operation: "ballot",
                expected_mask: 0xFFFFFFFF,
                actual_mask: self.active_mask,
            });
        }
        let mut mask = 0u32;
        for (i, &p) in predicate.iter().enumerate() {
            if p { mask |= 1 << i; }
        }
        Ok(mask)
    }

    /// Diverge by predicate mask.
    pub fn diverge(self, predicate_mask: u32) -> (DynWarp, DynWarp) {
        let true_mask = self.active_mask & predicate_mask;
        let false_mask = self.active_mask & !predicate_mask;
        (DynWarp::from_mask(true_mask), DynWarp::from_mask(false_mask))
    }

    /// Merge two disjoint DynWarps.
    pub fn merge(self, other: DynWarp) -> Result<DynWarp, WarpError> {
        let overlap = self.active_mask & other.active_mask;
        if overlap != 0 {
            return Err(WarpError {
                operation: "merge",
                expected_mask: 0,
                actual_mask: overlap,
            });
        }
        Ok(DynWarp::from_mask(self.active_mask | other.active_mask))
    }
}

// ============================================================================
// THE BUG (PyTorch #98157)
// ============================================================================

/// Simulates the hardware scheduler splitting logically-converged threads.
///
/// On Volta+ with independent thread scheduling, `__activemask()` can return
/// a SUBSET of the threads on the current code path. Two threads on the same
/// branch may be in different scheduling groups.
///
/// This function models what happens when we use `__activemask()` for ballot:
/// the ballot only counts votes from the threads in the CURRENT scheduling
/// group, not all threads on this code path.
fn activemask_ballot_buggy(
    data: &[u32; 32],
    radix_pos: u32,
    hardware_group_mask: u32, // what __activemask() returns (< 0xFFFFFFFF)
) -> [u32; RADIX_SIZE] {
    let mut counts = [0u32; RADIX_SIZE];

    // This is what PyTorch's countRadixUsingMask does:
    // __ballot_sync(__activemask(), vote)
    // The ballot only counts threads in the hardware group, NOT all converged threads.
    for bucket in 0..RADIX_SIZE {
        let mut vote_mask = 0u32;
        for lane in 0..32u32 {
            let in_group = hardware_group_mask & (1 << lane) != 0;
            let digit = get_bitfield(data[lane as usize], radix_pos, RADIX_BITS);
            if in_group && digit == bucket as u32 {
                vote_mask |= 1 << lane;
            }
        }
        counts[bucket] = vote_mask.count_ones();
    }

    counts
}

/// The correct ballot: all 32 lanes participate.
///
/// Uses `Warp<All>` — the type system guarantees convergence.
fn ballot_correct(
    warp: &Warp<All>,
    data: &[u32; 32],
    radix_pos: u32,
) -> [u32; RADIX_SIZE] {
    let mut counts = [0u32; RADIX_SIZE];

    for bucket in 0..RADIX_SIZE {
        let mut pred = [false; 32];
        for lane in 0..32 {
            pred[lane] = get_bitfield(data[lane], radix_pos, RADIX_BITS) == bucket as u32;
        }
        // This compiles because warp is Warp<All>.
        // No hardware query — the type IS the proof of convergence.
        let vote_mask = warp.ballot(&pred);
        counts[bucket] = vote_mask.count_ones();
    }

    counts
}

// ============================================================================
// THE BUG: compile_fail demonstration
// ============================================================================

/// What the PyTorch code looks like in our type system.
///
/// After divergence (loop exit), some threads are inactive. The type system
/// prevents calling ballot on the remaining subset.
///
/// UNCOMMENT TO SEE COMPILE ERROR:
///
/// ```compile_fail
/// # use pytorch_98157::*;
/// fn buggy_count_radix(warp: Warp<All>, data: &[u32; 32]) -> [u32; 4] {
///     // Some threads exit the loop early
///     let (active, _exited) = warp.diverge_loop();
///
///     // BUG: Try to ballot on the loop-active subset
///     // In CUDA: __ballot_sync(__activemask(), vote)
///     let pred = [true; 32];
///     let mask = active.ballot(&pred);
///     //         ^^^^^^ ERROR: no method `ballot` found for `Warp<LoopActive>`
///     [mask; 4]
/// }
/// ```
fn _buggy_version_for_doctest() {}

// ============================================================================
// THE FIX (two approaches)
// ============================================================================

/// Fix 1: Prove convergence via the type system.
///
/// Merge back to `Warp<All>` before ballot. Exited lanes contribute
/// zero-votes (their data doesn't affect the radix count).
fn fix_static_convergence(
    warp: Warp<All>,
    data: &[u32; 32],
    radix_pos: u32,
) -> [u32; RADIX_SIZE] {
    // Model the loop body where some threads exit early
    let (active, exited) = warp.diverge_loop();

    // Before ballot: merge back to Warp<All>.
    // Exited threads contribute false to all predicates (zero votes).
    let full: Warp<All> = merge(active, exited);

    // Now ballot is safe — all 32 lanes participate.
    // Active lanes vote based on their data, exited lanes vote false.
    let mut counts = [0u32; RADIX_SIZE];
    for bucket in 0..RADIX_SIZE {
        let mut pred = [false; 32];
        for lane in 0..32 {
            // Only active lanes vote; exited lanes stay false
            if LoopActive::MASK & (1 << lane) != 0 {
                pred[lane] = get_bitfield(data[lane], radix_pos, RADIX_BITS) == bucket as u32;
            }
        }
        let vote_mask = full.ballot(&pred);
        counts[bucket] = vote_mask.count_ones();
    }
    counts
}

/// Fix 2: Use DynWarp with runtime checking.
///
/// For code being migrated incrementally: DynWarp catches the
/// `__activemask()` bug at runtime instead of compile time.
fn fix_dynwarp_checked(
    data: &[u32; 32],
    radix_pos: u32,
    loop_active_mask: u32,
) -> Result<[u32; RADIX_SIZE], WarpError> {
    let dyn_warp = DynWarp::from_mask(loop_active_mask);

    // This fails at runtime if loop_active_mask != 0xFFFFFFFF
    let mut counts = [0u32; RADIX_SIZE];
    for bucket in 0..RADIX_SIZE {
        let mut pred = [false; 32];
        for lane in 0..32 {
            pred[lane] = get_bitfield(data[lane], radix_pos, RADIX_BITS) == bucket as u32;
        }
        // Runtime check: ballot requires all-active
        let vote_mask = dyn_warp.ballot(&pred)?;
        counts[bucket] = vote_mask.count_ones();
    }
    Ok(counts)
}

/// Correct DynWarp workflow: diverge, merge, then ballot.
fn fix_dynwarp_merge_then_ballot(
    data: &[u32; 32],
    radix_pos: u32,
) -> Result<[u32; RADIX_SIZE], WarpError> {
    let full = DynWarp::all();
    let (active, exited) = full.diverge(LoopActive::MASK);

    // Merge back before ballot
    let restored = active.merge(exited)?;

    // Now ballot succeeds — merged mask is 0xFFFFFFFF
    let mut counts = [0u32; RADIX_SIZE];
    for bucket in 0..RADIX_SIZE {
        let mut pred = [false; 32];
        for lane in 0..32 {
            if LoopActive::MASK & (1 << lane) != 0 {
                pred[lane] = get_bitfield(data[lane], radix_pos, RADIX_BITS) == bucket as u32;
            }
        }
        let vote_mask = restored.ballot(&pred)?;
        counts[bucket] = vote_mask.count_ones();
    }
    Ok(counts)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create test data where each lane has a known radix digit.
    /// Lanes 0-7: digit 0, lanes 8-15: digit 1, lanes 16-23: digit 2, lanes 24-31: digit 3.
    fn test_data() -> [u32; 32] {
        let mut data = [0u32; 32];
        for i in 0..32 {
            // At radix_pos=0 with RADIX_BITS=2, the digit is the low 2 bits.
            data[i] = (i / 8) as u32; // 0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3
        }
        data
    }

    #[test]
    fn test_correct_ballot_full_warp() {
        let warp: Warp<All> = Warp::new();
        let data = test_data();
        let counts = ballot_correct(&warp, &data, 0);

        // 8 lanes per bucket
        assert_eq!(counts, [8, 8, 8, 8]);
    }

    #[test]
    fn test_buggy_activemask_undercounts() {
        let data = test_data();

        // Simulate hardware splitting: only lower 16 lanes in this group
        let hardware_group = 0x0000FFFF;
        let buggy_counts = activemask_ballot_buggy(&data, 0, hardware_group);

        // BUG: only sees lanes 0-15 → digits 0 and 1 only
        // Lanes 16-31 (digits 2 and 3) are in a different scheduling group
        assert_eq!(buggy_counts[0], 8); // lanes 0-7: digit 0 (all in group)
        assert_eq!(buggy_counts[1], 8); // lanes 8-15: digit 1 (all in group)
        assert_eq!(buggy_counts[2], 0); // lanes 16-23: digit 2 (MISSING!)
        assert_eq!(buggy_counts[3], 0); // lanes 24-31: digit 3 (MISSING!)

        // Total votes: 16, not 32. Distribution is wrong.
        let total: u32 = buggy_counts.iter().sum();
        assert_eq!(total, 16); // should be 32!
    }

    #[test]
    fn test_correct_counts_vs_buggy() {
        let warp: Warp<All> = Warp::new();
        let data = test_data();

        // Correct: all lanes participate
        let correct = ballot_correct(&warp, &data, 0);

        // Buggy: hardware split loses half the votes
        let hardware_group = 0x0000FFFF;
        let buggy = activemask_ballot_buggy(&data, 0, hardware_group);

        // The correct counts are uniform
        assert_eq!(correct, [8, 8, 8, 8]);

        // The buggy counts are skewed — radix select will pick the wrong bucket
        assert_ne!(correct, buggy);

        // This is how topk/kthvalue/median return wrong results:
        // the radix select descends into the wrong bucket because the
        // distribution counts don't reflect all the data.
    }

    #[test]
    fn test_fix_static_convergence() {
        let warp: Warp<All> = Warp::new();
        let data = test_data();

        let counts = fix_static_convergence(warp, &data, 0);

        // Only LoopActive lanes (0-23) vote. Lanes 24-31 are exited.
        // Lanes 0-7: digit 0 (8 votes), lanes 8-15: digit 1 (8 votes),
        // lanes 16-23: digit 2 (8 votes), lanes 24-31: exited (0 votes for digit 3)
        assert_eq!(counts[0], 8);
        assert_eq!(counts[1], 8);
        assert_eq!(counts[2], 8);
        assert_eq!(counts[3], 0); // exited lanes don't vote

        // Total = 24 active lanes, all accounted for
        let total: u32 = counts.iter().sum();
        assert_eq!(total, 24);
    }

    #[test]
    fn test_fix_dynwarp_catches_partial_ballot() {
        let data = test_data();

        // DynWarp with partial mask — ballot FAILS at runtime
        let result = fix_dynwarp_checked(&data, 0, LoopActive::MASK);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.operation, "ballot");
        assert_eq!(err.actual_mask, LoopActive::MASK);
    }

    #[test]
    fn test_fix_dynwarp_merge_then_ballot() {
        let data = test_data();

        // Correct DynWarp workflow: diverge → merge → ballot
        let counts = fix_dynwarp_merge_then_ballot(&data, 0).unwrap();

        // Same result as static convergence fix
        assert_eq!(counts[0], 8);
        assert_eq!(counts[1], 8);
        assert_eq!(counts[2], 8);
        assert_eq!(counts[3], 0);
    }

    #[test]
    fn test_type_prevents_diverged_ballot() {
        let warp: Warp<All> = Warp::new();
        let (active, _exited) = warp.diverge_loop();

        // Verify: active is Warp<LoopActive>, which has NO ballot method
        assert_eq!(active.active_mask(), LoopActive::MASK);
        assert_ne!(active.active_mask(), All::MASK);

        // The following would not compile:
        // active.ballot(&[true; 32]);
        // error[E0599]: no method named `ballot` found for `Warp<LoopActive>`
    }

    #[test]
    fn test_merge_restores_ballot() {
        let warp: Warp<All> = Warp::new();
        let (active, exited) = warp.diverge_loop();

        // Merge back — type system verifies complement
        let full: Warp<All> = merge(active, exited);
        assert_eq!(full.active_mask(), 0xFFFFFFFF);

        // Ballot now available
        let pred = [true; 32];
        let mask = full.ballot(&pred);
        assert_eq!(mask, 0xFFFFFFFF);
    }

    #[test]
    fn test_activemask_is_not_convergence() {
        // The core insight: __activemask() is a hardware query, not a convergence proof.
        // Two threads on the SAME code path can get DIFFERENT activemask results.
        //
        // Scenario: 32 threads all in the same loop iteration, but the hardware
        // scheduler splits them into two groups of 16.

        let data = test_data();

        // Group A sees lanes 0-15
        let group_a_counts = activemask_ballot_buggy(&data, 0, 0x0000FFFF);
        // Group B sees lanes 16-31
        let group_b_counts = activemask_ballot_buggy(&data, 0, 0xFFFF0000);

        // Each group has a partial view of the distribution
        assert_eq!(group_a_counts, [8, 8, 0, 0]); // missing buckets 2,3
        assert_eq!(group_b_counts, [0, 0, 8, 8]); // missing buckets 0,1

        // Neither group has the correct distribution [8, 8, 8, 8].
        // If countRadixUsingMask uses either group's counts, the subsequent
        // findPattern will select the wrong bucket for the k-th element.
    }

    #[test]
    fn test_data_race_in_findpattern() {
        // The second consequence: when threads in different hardware groups
        // compute different radix counts, they may disagree about which bucket
        // contains the k-th element. This is a data race in findPattern.

        let data = test_data();

        // Group A thinks: buckets = [8, 8, 0, 0] → k=20 is in bucket 1? No, overflow!
        let group_a = activemask_ballot_buggy(&data, 0, 0x0000FFFF);
        // Group B thinks: buckets = [0, 0, 8, 8] → k=20 is in bucket 3
        let group_b = activemask_ballot_buggy(&data, 0, 0xFFFF0000);

        // Simulate findPattern: which bucket contains the k-th element?
        let k = 20u32;

        let bucket_a = find_bucket(&group_a, k);
        let bucket_b = find_bucket(&group_b, k);

        // Different groups pick different buckets — DATA RACE
        // Group A: total=16, can't even find k=20 (returns None)
        // Group B: 0+0+8+8=16, also can't find k=20 (returns None)
        // But even with k values in range, they'll disagree.
        // Let's use k=5:
        let k_small = 5u32;
        let bucket_a_small = find_bucket(&group_a, k_small);
        let bucket_b_small = find_bucket(&group_b, k_small);

        // Group A: cumulative [8, 16, 16, 16], k=5 is in bucket 0
        assert_eq!(bucket_a_small, Some(0));
        // Group B: cumulative [0, 0, 8, 16], k=5 is in bucket 2
        assert_eq!(bucket_b_small, Some(2));

        // DISAGREEMENT: threads in group A descend into bucket 0,
        // threads in group B descend into bucket 2. This is the data race.
        assert_ne!(bucket_a_small, bucket_b_small);

        // The correct answer uses all 32 lanes:
        let warp: Warp<All> = Warp::new();
        let correct = ballot_correct(&warp, &data, 0);
        let bucket_correct = find_bucket(&correct, k_small);
        // Cumulative [8, 16, 24, 32], k=5 is in bucket 0
        assert_eq!(bucket_correct, Some(0));

        // Suppress unused variable warnings
        let _ = (bucket_a, bucket_b);
    }

    #[test]
    fn test_warp_is_zero_sized() {
        // The active set is a type, not a runtime value.
        // No register to query, no hardware state to depend on.
        assert_eq!(std::mem::size_of::<Warp<All>>(), 0);
        assert_eq!(std::mem::size_of::<Warp<LoopActive>>(), 0);
        assert_eq!(std::mem::size_of::<Warp<LoopExited>>(), 0);
    }

    #[test]
    fn test_dynwarp_full_workflow() {
        let data = test_data();

        // Phase 1: DynWarp catches the __activemask() pattern at runtime
        let partial = DynWarp::from_mask(0x0000FFFF); // simulates __activemask()
        let err = partial.ballot(&[true; 32]).unwrap_err();
        assert_eq!(err.operation, "ballot");

        // Phase 2: Correct workflow with DynWarp
        let full = DynWarp::all();
        let (active, exited) = full.diverge(LoopActive::MASK);

        // Can't ballot on subset
        assert!(active.ballot(&[true; 32]).is_err());

        // Merge back
        let restored = active.merge(exited).unwrap();
        assert_eq!(restored.active_mask(), 0xFFFFFFFF);

        // Now ballot works
        let mut pred = [false; 32];
        for lane in 0..32 {
            if LoopActive::MASK & (1 << lane) != 0 {
                pred[lane] = get_bitfield(data[lane], 0, RADIX_BITS) == 0;
            }
        }
        let mask = restored.ballot(&pred).unwrap();
        // 8 lanes with digit 0, all in the active set
        assert_eq!(mask.count_ones(), 8);
    }
}

// ============================================================================
// HELPER: findPattern bucket selection
// ============================================================================

/// Simulate findPattern's bucket selection: given radix counts, which bucket
/// contains the k-th element (0-indexed)?
fn find_bucket(counts: &[u32; RADIX_SIZE], k: u32) -> Option<u32> {
    let mut cumulative = 0u32;
    for (bucket, &count) in counts.iter().enumerate() {
        cumulative += count;
        if k < cumulative {
            return Some(bucket as u32);
        }
    }
    None // k exceeds total count
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("PyTorch #98157: __activemask() in countRadixUsingMask");
    println!("=====================================================\n");

    println!("The Bug (CUDA):");
    println!("  countRadixUsingMask uses __ballot_sync(__activemask(), vote).");
    println!("  __activemask() captures HARDWARE scheduling state, not convergence.");
    println!("  On Volta+ with independent thread scheduling, logically-converged");
    println!("  threads can be in different hardware groups. The ballot under-counts");
    println!("  votes, producing wrong distribution counts.\n");

    println!("Consequences:");
    println!("  1. Wrong radix distribution → topk/kthvalue/median return wrong results");
    println!("  2. Threads disagree on bucket selection → data race in findPattern\n");

    println!("Why __activemask() Is Unsafe:");
    println!("  It answers 'which threads are executing NOW' (hardware state),");
    println!("  not 'which threads are on this code path' (program convergence).");
    println!("  The PTX ISA spec explicitly warns against using it for synchronization.\n");

    println!("Why Session Types Catch It:");
    println!("  No activemask() function exists. The active set is a type parameter.");
    println!("  ballot() requires Warp<All>. After diverge(), you have Warp<LoopActive>.");
    println!("  Compile error: no method `ballot` found for `Warp<LoopActive>`.\n");

    // Demonstrate the bug
    let data = {
        let mut d = [0u32; 32];
        for i in 0..32 { d[i] = (i / 8) as u32; }
        d
    };

    let warp: Warp<All> = Warp::new();
    let correct = ballot_correct(&warp, &data, 0);
    println!("Correct counts (all 32 lanes): {:?}", correct);

    let buggy = activemask_ballot_buggy(&data, 0, 0x0000FFFF);
    println!("Buggy counts (16-lane group):  {:?}", buggy);

    println!("\nThe buggy counts are wrong — buckets 2 and 3 show 0 instead of 8.");
    println!("findPattern would select the wrong bucket for the k-th element.");

    // Demonstrate the fix
    let fixed = fix_static_convergence(Warp::new(), &data, 0);
    println!("\nFixed counts (merge → ballot): {:?}", fixed);
    println!("Active lanes (0-23) correctly counted; exited lanes (24-31) contribute 0.");

    println!("\nRun `cargo test --example pytorch_98157` for full test suite.");
}
