//! # PIConGPU Issue #2514: Hardcoded Full Mask in Divergent Ballot
//!
//! Demonstrates how session-typed divergence catches a real bug from the
//! PIConGPU plasma physics simulation.
//!
//! ## The Real Bug
//!
//! In PMacc's `atomicAllInc()`:
//!
//! ```cuda
//! __device__ int atomicAllInc(int* addr) {
//!     // Some threads reach here, others took a different branch
//!     int mask = __ballot_sync(0xFFFFFFFF, 1);  // BUG!
//!     // 0xFFFFFFFF claims all 32 lanes participate,
//!     // but we're inside a divergent branch — some lanes are inactive.
//!     int leader = __ffs(mask) - 1;
//!     int count = __popc(mask);
//!     if (threadIdx.x % 32 == leader) {
//!         old = atomicAdd(addr, count);
//!     }
//!     old = __shfl_sync(mask, old, leader);
//!     return old + __popc(mask & ((1 << (threadIdx.x % 32)) - 1));
//! }
//! ```
//!
//! **Failure mode:** Undefined behavior. Ran for months on K80 GPUs (pre-Volta)
//! where lockstep execution masked the bug. Discovered during CUDA 9 migration.
//! Produces plausible but mathematically wrong plasma simulation results.
//!
//! **Source:** <https://github.com/ComputationalRadiationPhysics/picongpu/issues/2514>
//!
//! ## Why `__shfl_sync` Doesn't Help
//!
//! The bug uses `__ballot_sync(0xFFFFFFFF, 1)` — the mask is hardcoded.
//! `__shfl_sync` requires a correct mask, but the mask is a runtime `u32`.
//! The programmer passed `0xFFFFFFFF` when fewer lanes were active. The
//! hardware accepts any mask value — it doesn't verify against the actual
//! active set. `__activemask()` would return the correct active set, but
//! the programmer used a hardcoded constant instead.
//!
//! ## Why Session Types Catch It
//!
//! After divergence, the warp handle is `Warp<Active>`, not `Warp<All>`.
//! `ballot()` only exists on `Warp<All>`. There is no runtime mask to get
//! wrong — the type *is* the mask.
//!
//! Run: `cargo test --example picongpu_2514`

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

// Active set types
#[derive(Copy, Clone)] pub struct All;
#[derive(Copy, Clone)] pub struct Active;     // threads that entered the branch
#[derive(Copy, Clone)] pub struct Inactive;   // threads that took the other path

impl ActiveSet for All      { const MASK: u32 = 0xFFFFFFFF; const NAME: &'static str = "All"; }
impl ActiveSet for Active   { const MASK: u32 = 0x0000FFFF; const NAME: &'static str = "Active"; }
impl ActiveSet for Inactive { const MASK: u32 = 0xFFFF0000; const NAME: &'static str = "Inactive"; }

impl ComplementOf<Inactive> for Active {}
impl ComplementOf<Active> for Inactive {}

// ============================================================================
// GATED OPERATIONS: Only Warp<All> has ballot
// ============================================================================

impl Warp<All> {
    /// Diverge: model the conditional branch that leads to atomicAllInc.
    pub fn diverge_on_condition(self) -> (Warp<Active>, Warp<Inactive>) {
        (Warp::new(), Warp::new())
    }

    /// Ballot — __ballot_sync with full mask. ONLY on Warp<All>.
    pub fn ballot(&self, predicate: &[bool; 32]) -> u32 {
        let mut result = 0u32;
        for i in 0..32 {
            if predicate[i] { result |= 1 << i; }
        }
        result
    }

    /// Shuffle — ONLY on Warp<All>.
    pub fn shuffle(&self, data: &[i32; 32], src_lane: u32) -> [i32; 32] {
        [data[src_lane as usize]; 32]
    }
}

pub fn merge<S1, S2>(_left: Warp<S1>, _right: Warp<S2>) -> Warp<All>
where S1: ComplementOf<S2>, S2: ActiveSet {
    Warp::new()
}

// ============================================================================
// THE BUG (PIConGPU #2514)
// ============================================================================

/// What the CUDA code does (translated to our type system).
///
/// atomicAllInc is called inside a divergent branch. The ballot uses
/// 0xFFFFFFFF claiming all lanes participate — but they don't.
///
/// UNCOMMENT TO SEE COMPILE ERROR:
///
/// ```compile_fail
/// # use picongpu_2514::*;
/// fn buggy_atomic_all_inc(warp: Warp<All>, addr: &mut i32) -> i32 {
///     // Some threads enter this branch, others don't
///     let (active, _inactive) = warp.diverge_on_condition();
///
///     // BUG: ballot on diverged warp — claims all 32 lanes participate
///     // In CUDA: __ballot_sync(0xFFFFFFFF, 1)
///     let mask = active.ballot(&[true; 32]);
///     //         ^^^^^^ ERROR: no method `ballot` found for `Warp<Active>`
///     mask as i32
/// }
/// ```
fn _buggy_version_for_doctest() {}

// ============================================================================
// THE FIX
// ============================================================================

/// Correct approach: merge back to Warp<All> before ballot,
/// with inactive lanes contributing a 0 predicate.
fn correct_atomic_all_inc(warp: Warp<All>) -> (u32, u32) {
    let (active, inactive) = warp.diverge_on_condition();

    // Merge back — type system verifies complement
    let full: Warp<All> = merge(active, inactive);

    // Now ballot is safe: all lanes participate.
    // Active lanes vote 1, inactive lanes vote 0.
    let mut pred = [false; 32];
    for i in 0..32 {
        pred[i] = (Active::MASK & (1 << i)) != 0;
    }
    let mask = full.ballot(&pred);
    let count = mask.count_ones();

    (mask, count)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_ballot_counts_active_lanes() {
        let warp: Warp<All> = Warp::new();
        let (mask, count) = correct_atomic_all_inc(warp);

        // Active has lower 16 lanes
        assert_eq!(mask, 0x0000FFFF);
        assert_eq!(count, 16);
    }

    #[test]
    fn test_type_prevents_diverged_ballot() {
        let warp: Warp<All> = Warp::new();
        let (active, _inactive) = warp.diverge_on_condition();

        // Verify: active is Warp<Active>, which has NO ballot method
        assert_eq!(active.active_mask(), 0x0000FFFF);

        // The following would not compile:
        // active.ballot(&[true; 32]);
        // error[E0599]: no method named `ballot` found for `Warp<Active>`
    }

    #[test]
    fn test_merge_restores_ballot_access() {
        let warp: Warp<All> = Warp::new();
        let (active, inactive) = warp.diverge_on_condition();
        let restored: Warp<All> = merge(active, inactive);

        // After merge, ballot works again
        let mask = restored.ballot(&[true; 32]);
        assert_eq!(mask, 0xFFFFFFFF);
    }

    #[test]
    fn test_why_shfl_sync_fails() {
        // __shfl_sync requires a mask parameter, but the mask is just a u32.
        // The programmer can pass any value — hardware doesn't verify.
        // PIConGPU passed 0xFFFFFFFF when only some lanes were active.
        //
        // Our type system has no runtime mask to get wrong:
        // Warp<Active> structurally cannot call ballot(), regardless of
        // what mask value the programmer might have intended.
        let warp: Warp<All> = Warp::new();
        let (active, _) = warp.diverge_on_condition();

        // The mask IS the type, not a runtime value
        assert_eq!(active.active_mask(), Active::MASK);
        assert_ne!(active.active_mask(), All::MASK); // NOT 0xFFFFFFFF
    }
}

fn main() {
    println!("PIConGPU #2514: Hardcoded Full Mask in Divergent Ballot");
    println!("=======================================================\n");

    println!("The Bug (CUDA):");
    println!("  __ballot_sync(0xFFFFFFFF, 1) inside a divergent branch.");
    println!("  Hardcoded mask claims all 32 lanes, but some are inactive.");
    println!("  Ran for months on K80 GPUs — lockstep execution masked the UB.\n");

    println!("Why __shfl_sync Doesn't Help:");
    println!("  The mask is a runtime u32. Hardware accepts any value.");
    println!("  The programmer used 0xFFFFFFFF instead of __activemask().\n");

    println!("Why Session Types Catch It:");
    println!("  After divergence: Warp<Active>, not Warp<All>.");
    println!("  ballot() doesn't exist on Warp<Active>. Compile error.\n");

    let warp: Warp<All> = Warp::new();
    let (mask, count) = correct_atomic_all_inc(warp);
    println!("Correct ballot mask: 0x{:08X} ({} active lanes)", mask, count);
}
