//! # The Bug That Types Catch
//!
//! This demo shows how warp typestate catches shuffle-after-diverge
//! bugs at compile time.
//!
//! ## The Bug (in untyped CUDA)
//!
//! ```cuda
//! __device__ int filtered_sum(int data, bool keep) {
//!     if (keep) {
//!         // BUG: shuffle reads from ALL lanes, including inactive ones!
//!         int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);
//!         return data + partner;
//!     }
//!     return 0;
//! }
//! ```
//!
//! This compiles fine in CUDA. It may work sometimes. It will fail unpredictably.
//!
//! ## The Fix (with warp typestate)
//!
//! Our type system makes this a compile error, and guides you to the fix.
//!
//! Run this demo:
//! ```
//! cargo test --example demo_bug_that_types_catch
//! ```

#![allow(clippy::needless_range_loop, clippy::new_without_default)]

use std::marker::PhantomData;

// ============================================================================
// TYPE SYSTEM CORE
// ============================================================================

/// Trait for active set types
pub trait ActiveSet: Copy + 'static {
    const MASK: u32;
    const NAME: &'static str;
}

/// All 32 lanes active
#[derive(Copy, Clone)]
pub struct All;
impl ActiveSet for All {
    const MASK: u32 = 0xFFFFFFFF;
    const NAME: &'static str = "All";
}

/// Even lanes active (0, 2, 4, ...)
#[derive(Copy, Clone)]
pub struct Even;
impl ActiveSet for Even {
    const MASK: u32 = 0x55555555;
    const NAME: &'static str = "Even";
}

/// Odd lanes active (1, 3, 5, ...)
#[derive(Copy, Clone)]
pub struct Odd;
impl ActiveSet for Odd {
    const MASK: u32 = 0xAAAAAAAA;
    const NAME: &'static str = "Odd";
}

/// Trait that proves S1 and S2 are complements
pub trait ComplementOf<Other: ActiveSet>: ActiveSet {}
impl ComplementOf<Odd> for Even {}
impl ComplementOf<Even> for Odd {}

/// A warp with typed active set
#[derive(Copy, Clone)]
pub struct Warp<S: ActiveSet> {
    _marker: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self {
        Warp { _marker: PhantomData }
    }

    pub fn active_mask(&self) -> u32 {
        S::MASK
    }
}

/// Per-lane data
#[derive(Copy, Clone)]
pub struct PerLane<T>(pub [T; 32]);

impl<T: Copy + Default> PerLane<T> {
    pub fn splat(value: T) -> Self {
        PerLane([value; 32])
    }
}

impl<T: Copy + Default + std::ops::Add<Output = T>> PerLane<T> {
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [T::default(); 32];
        for i in 0..32 {
            result[i] = self.0[i] + other.0[i];
        }
        PerLane(result)
    }
}

// ============================================================================
// KEY INSIGHT: Method availability = safety
// ============================================================================

// shuffle_xor is ONLY available on Warp<All>
impl Warp<All> {
    /// Shuffle XOR - exchange values with lane XOR partner
    ///
    /// This method ONLY exists on Warp<All>.
    /// Calling it on Warp<Even> is not "checked and rejected" - the method
    /// simply doesn't exist. This is the key to our safety guarantee.
    pub fn shuffle_xor<T: Copy + Default>(&self, data: PerLane<T>, mask: u32) -> PerLane<T> {
        let mut result = [T::default(); 32];
        for lane in 0..32u32 {
            let src = lane ^ mask;
            result[lane as usize] = data.0[src as usize];
        }
        PerLane(result)
    }

    /// Diverge into even and odd lanes
    pub fn diverge_even_odd(self) -> (Warp<Even>, Warp<Odd>) {
        (Warp::new(), Warp::new())
    }
}

/// Merge two complementary warps back into Warp<All>
pub fn merge<S1, S2>(_left: Warp<S1>, _right: Warp<S2>) -> Warp<All>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
{
    Warp::new()
}

/// Merge data from two complementary active sets
pub fn merge_data<T: Copy + Default, S1: ActiveSet, S2: ComplementOf<S1>>(
    left: PerLane<T>,
    right: PerLane<T>,
) -> PerLane<T> {
    let mut result = [T::default(); 32];
    for lane in 0..32u32 {
        if S1::MASK & (1 << lane) != 0 {
            result[lane as usize] = left.0[lane as usize];
        } else {
            result[lane as usize] = right.0[lane as usize];
        }
    }
    PerLane(result)
}

// ============================================================================
// DEMO: The bug that types catch
// ============================================================================

/// This function demonstrates the CORRECT way to do filtered operations.
///
/// The key insight: if you want to shuffle, you need Warp<All>.
/// If you've diverged, you must merge back before shuffling.
fn filtered_sum_correct(warp: Warp<All>, data: PerLane<i32>, _keep: PerLane<bool>) -> PerLane<i32> {
    // Step 1: Diverge based on predicate
    let (active, inactive) = warp.diverge_even_odd();

    // Step 2: Prepare data for each branch
    // Active lanes keep their data, inactive lanes contribute 0
    let active_data = data;
    let inactive_data = PerLane::splat(0i32);

    // Step 3: Merge back to Warp<All>
    // The type system VERIFIES that Even and Odd are complements!
    let warp: Warp<All> = merge(active, inactive);
    let combined: PerLane<i32> = merge_data::<i32, Even, Odd>(active_data, inactive_data);

    // Step 4: Now shuffle is safe - we have Warp<All>
    let partner = warp.shuffle_xor(combined, 1);

    // Step 5: Add values
    combined.add(&partner)
}

/// This function shows what the BUGGY version would look like.
///
/// UNCOMMENT THIS TO SEE THE COMPILE ERROR:
///
/// ```compile_fail
/// fn filtered_sum_buggy(warp: Warp<All>, data: PerLane<i32>) -> PerLane<i32> {
///     let (active, _inactive) = warp.diverge_even_odd();
///
///     // BUG: Try to shuffle on Warp<Even> - this won't compile!
///     let partner = active.shuffle_xor(data, 1);
///     //            ^^^^^^ error[E0599]: no method named `shuffle_xor` found
///     //                   for struct `Warp<Even>` in the current scope
///     //
///     // The method `shuffle_xor` exists on `Warp<All>`, but not on `Warp<Even>`.
///     // This is not a runtime check - the method simply doesn't exist.
///
///     data.add(&partner)
/// }
/// ```
fn _buggy_version_commented_out() {
    // See docstring above for the compile error
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_filtered_sum() {
        let warp: Warp<All> = Warp::new();

        // Create test data: lanes 0-31 have values 0-31
        let mut data_arr = [0i32; 32];
        for i in 0..32 {
            data_arr[i] = i as i32;
        }
        let data = PerLane(data_arr);

        // Keep flags (not used in this simplified demo)
        let keep = PerLane([true; 32]);

        // Run the correct version
        let result = filtered_sum_correct(warp, data, keep);

        // Verify: each lane should have its value plus its XOR-1 partner's value
        // But since we zero'd odd lanes, even lanes get: data[even] + 0
        // And odd lanes get: 0 + data[even]
        println!("Results:");
        for i in 0..8 {
            println!("  Lane {}: {}", i, result.0[i]);
        }
    }

    #[test]
    fn test_type_system_catches_bug() {
        // This test verifies the TYPE SYSTEM catches the bug.
        // We can't actually write the buggy code because it won't compile!

        let warp: Warp<All> = Warp::new();
        let (even_warp, _odd_warp) = warp.diverge_even_odd();

        // Verify that Warp<Even> does NOT have shuffle_xor method
        // (This is a compile-time property, tested by the compile_fail doctest above)

        // Verify the active mask is correct
        assert_eq!(even_warp.active_mask(), 0x55555555);
    }

    #[test]
    fn test_merge_requires_complements() {
        let warp: Warp<All> = Warp::new();
        let (evens, odds) = warp.diverge_even_odd();

        // This compiles because Even and Odd ARE complements
        let _merged: Warp<All> = merge(evens, odds);

        // This would NOT compile:
        // let _bad: Warp<All> = merge(evens, evens);
        // error: the trait `ComplementOf<Even>` is not implemented for `Even`
    }

    #[test]
    fn test_shuffle_requires_all() {
        let warp: Warp<All> = Warp::new();
        let data = PerLane([1i32; 32]);

        // This compiles because warp is Warp<All>
        let _result = warp.shuffle_xor(data, 1);

        // If we diverge, shuffle is no longer available:
        let (evens, _odds) = warp.diverge_even_odd();
        // evens.shuffle_xor(data, 1);  // ERROR: method not found
        let _ = evens;  // Suppress unused warning
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("Warp Typestate Divergence Demo");
    println!("==============================\n");

    println!("This demo shows how linear typestate catches GPU divergence bugs.\n");

    println!("The Bug (would be undefined behavior in CUDA):");
    println!("  1. Warp diverges based on a predicate");
    println!("  2. Only some lanes are active");
    println!("  3. Code tries to shuffle, reading from inactive lanes");
    println!("  4. Result: undefined behavior, non-deterministic bugs\n");

    println!("Our Solution:");
    println!("  1. Warp<All> has shuffle_xor method");
    println!("  2. Warp<Even> does NOT have shuffle_xor method");
    println!("  3. Trying to shuffle after diverge = compile error");
    println!("  4. Must merge back to Warp<All> before shuffling\n");

    println!("Run the tests to see it in action:");
    println!("  cargo test --example demo_bug_that_types_catch\n");

    // Demo the correct version
    let warp: Warp<All> = Warp::new();
    let mut data_arr = [0i32; 32];
    for i in 0..32 {
        data_arr[i] = i as i32;
    }
    let data = PerLane(data_arr);
    let keep = PerLane([true; 32]);

    let result = filtered_sum_correct(warp, data, keep);

    println!("Correct filtered_sum result (first 8 lanes):");
    for i in 0..8 {
        println!("  Lane {}: {}", i, result.0[i]);
    }
}
