//! Merge operations: reconverging diverged warps.
//!
//! Merge is the second half of the warp typestate pattern.
//! The key safety property: `merge()` only compiles if the two warps
//! have complementary active sets.
//!
//! # Compile-Time Safety
//!
//! ## Bug: Merge non-complements (caught)
//!
//! ```compile_fail
//! use warp_types::*;
//!
//! fn buggy_merge() {
//!     let warp: Warp<All> = Warp::kernel_entry();
//!     let (evens, _odds) = warp.diverge_even_odd();
//!     let warp2: Warp<All> = Warp::kernel_entry();
//!     let (low, _high) = warp2.diverge_halves();
//!     // BUG: Even and LowHalf are not complements (they overlap)
//!     let _ = merge(evens, low);
//! }
//! ```
//!
//! ## Bug: Merge same set (caught)
//!
//! ```compile_fail
//! use warp_types::*;
//!
//! fn buggy_merge_same() {
//!     let w1: Warp<All> = Warp::kernel_entry();
//!     let (evens1, _odds1) = w1.diverge_even_odd();
//!     let w2: Warp<All> = Warp::kernel_entry();
//!     let (evens2, _odds2) = w2.diverge_even_odd();
//!     // BUG: Can't merge Even with Even
//!     let _ = merge(evens1, evens2);
//! }
//! ```
//!
//! ## Bug: Merge nested complements as top-level (caught)
//!
//! EvenLow + EvenHigh = Even (16 lanes), NOT All (32 lanes).
//! `merge()` returns `Warp<All>`, so this would be unsound.
//! Use `merge_within()` instead, which returns `Warp<Even>`.
//!
//! ```compile_fail
//! use warp_types::*;
//!
//! fn buggy_nested_merge() {
//!     let w: Warp<All> = Warp::kernel_entry();
//!     let (evens, _odds) = w.diverge_even_odd();
//!     let (even_low, even_high) = evens.diverge_halves();
//!     // BUG: EvenLow and EvenHigh are NOT ComplementOf each other
//!     // (they cover Even, not All). This is a compile error.
//!     let _fake_all: Warp<All> = merge(even_low, even_high);
//! }
//! ```

use crate::active_set::*;
use crate::warp::Warp;

/// Merge two warps with complementary active sets.
///
/// **COMPILE-TIME VERIFIED**: Only works if `S1` and `S2` are complements.
/// You cannot merge overlapping sets. You cannot forget lanes.
///
/// # Examples
///
/// ```
/// use warp_types::*;
///
/// let all: Warp<All> = Warp::kernel_entry();
/// let (evens, odds) = all.diverge_even_odd();
/// let merged: Warp<All> = merge(evens, odds);
/// assert_eq!(merged.active_mask(), 0xFFFFFFFF);
/// ```
pub fn merge<S1, S2>(_left: Warp<S1>, _right: Warp<S2>) -> Warp<All>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
{
    Warp::new()
}

/// Merge two warps back to their parent active set (for nested divergence).
///
/// Unlike `merge()` which always returns `Warp<All>`, this returns the
/// parent set `P` where `S1` and `S2` are complements within `P`.
pub fn merge_within<S1, S2, P>(_left: Warp<S1>, _right: Warp<S2>) -> Warp<P>
where
    S1: ComplementWithin<S2, P>,
    S2: ActiveSet,
    P: ActiveSet,
{
    Warp::new()
}

/// Diverge, process each branch, automatically merge.
///
/// This combinator provides convenient syntax for the common
/// "diverge → do work → merge" pattern. The merge is explicit in the
/// type signature but automatic in the control flow.
///
/// # Implementation note
///
/// Sub-warps are manufactured via `Warp::new()` without an actual diverge
/// step. This is correct because all `Warp<S>` are zero-sized phantom types —
/// there is no runtime state to split. The `ComplementOf` bound ensures only
/// valid complement pairs can be specified (sealed trait, no external impls).
pub fn with_diverged<S1, S2, A, F1, F2>(
    _warp: Warp<All>,
    then_fn: F1,
    else_fn: F2,
) -> (A, A, Warp<All>)
where
    S1: ActiveSet + ComplementOf<S2>,
    S2: ActiveSet + ComplementOf<S1>,
    F1: FnOnce(Warp<S1>) -> A,
    F2: FnOnce(Warp<S2>) -> A,
{
    let then_result = then_fn(Warp::new());
    let else_result = else_fn(Warp::new());
    (then_result, else_result, Warp::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_even_odd() {
        let all: Warp<All> = Warp::new();
        let (evens, odds) = all.diverge_even_odd();
        let merged: Warp<All> = merge(evens, odds);
        assert_eq!(merged.active_set_name(), "All");
    }

    #[test]
    fn test_merge_halves() {
        let all: Warp<All> = Warp::new();
        let (low, high) = all.diverge_halves();
        let merged = merge(low, high);
        assert_eq!(merged.active_mask(), 0xFFFFFFFF);
    }

    #[test]
    fn test_merge_lane0() {
        let all: Warp<All> = Warp::new();
        let (lane0, rest) = all.extract_lane0();
        let merged = merge(lane0, rest);
        assert_eq!(merged.population(), 32);
    }

    #[test]
    fn test_nested_merge() {
        let all: Warp<All> = Warp::new();
        let (evens, odds) = all.diverge_even_odd();
        let (even_low, even_high) = evens.diverge_halves();

        // Merge nested: EvenLow + EvenHigh → Even
        let evens_restored: Warp<Even> = merge_within(even_low, even_high);
        assert_eq!(evens_restored.active_set_name(), "Even");

        // Merge top-level: Even + Odd → All
        let all_restored = merge(evens_restored, odds);
        assert_eq!(all_restored.active_set_name(), "All");
    }

    #[test]
    fn test_merge_ordering_equivalence() {
        // Tree merge via Even/Odd
        let el1: Warp<EvenLow> = Warp::new();
        let eh1: Warp<EvenHigh> = Warp::new();
        let ol1: Warp<OddLow> = Warp::new();
        let oh1: Warp<OddHigh> = Warp::new();

        let even: Warp<Even> = merge_within(el1, eh1);
        let odd: Warp<Odd> = merge_within(ol1, oh1);
        let result1 = merge(even, odd);

        // Tree merge via Low/High
        let el2: Warp<EvenLow> = Warp::new();
        let eh2: Warp<EvenHigh> = Warp::new();
        let ol2: Warp<OddLow> = Warp::new();
        let oh2: Warp<OddHigh> = Warp::new();

        let low: Warp<LowHalf> = merge_within(el2, ol2);
        let high: Warp<HighHalf> = merge_within(eh2, oh2);
        let result2 = merge(low, high);

        // Both produce Warp<All>
        assert_eq!(result1.population(), 32);
        assert_eq!(result2.population(), 32);
    }

    #[test]
    fn test_full_pipeline_nested_diverge_to_shuffle() {
        let warp: Warp<All> = Warp::kernel_entry();
        let (evens, odds) = warp.diverge_even_odd();
        let (even_low, even_high) = evens.diverge_halves();

        // Nested merge
        let evens_restored: Warp<Even> = merge_within(even_low, even_high);
        let all_restored: Warp<All> = merge(evens_restored, odds);

        // Now shuffle works
        let data = crate::data::PerLane::new(1i32);
        let result = all_restored.shuffle_xor(data, 1);
        assert_eq!(result.get(), 1); // CPU identity
        let sum = all_restored.reduce_sum(data);
        assert_eq!(sum.get(), 32);
    }

    #[test]
    fn test_with_diverged() {
        let warp: Warp<All> = Warp::new();
        let (a, b, merged) = with_diverged::<Even, Odd, i32, _, _>(warp, |_| 100, |_| 200);
        assert_eq!(a, 100);
        assert_eq!(b, 200);
        assert_eq!(merged.population(), 32);
    }
}
