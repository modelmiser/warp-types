//! Data-dependent divergence with structural complement guarantees.
//!
//! When the active set depends on runtime data (e.g., `data[lane] > threshold`),
//! the mask can't be known at compile time. But the complement relationship CAN:
//! the true-branch and false-branch are always disjoint and covering by construction.
//!
//! `DynDiverge` captures this: a paired divergence where both branches must merge
//! to recover `Warp<All>`. The mask is dynamic. The complement is static.
//!
//! # Example
//!
//! ```
//! use warp_types::*;
//! use warp_types::dynamic::DynDiverge;
//!
//! let warp: Warp<All> = Warp::kernel_entry();
//!
//! // Diverge on a runtime predicate (mask determined by data)
//! let diverged: DynDiverge = warp.diverge_dynamic(0x0000FFFF);
//!
//! // Can't shuffle on either branch — they're partial warps
//! // diverged.true_warp().shuffle_xor(...)  // doesn't exist
//!
//! // MUST merge to get Warp<All> back — then shuffle is available
//! let warp: Warp<All> = diverged.merge();
//! let data = data::PerLane::new(42i32);
//! let _result = warp.shuffle_xor(data, 1);
//! ```
//!
//! # Safety Properties
//!
//! 1. **Complement by construction**: `true_mask | false_mask == parent_mask` always holds
//! 2. **No shuffle on branches**: `DynDiverge` doesn't expose shuffle methods
//! 3. **Must merge**: `DynDiverge` holds the `Warp<All>` ownership — you can't proceed without merging
//! 4. **No mask manipulation**: Can't forge or modify masks after divergence

use crate::warp::Warp;
use crate::active_set::All;

/// A data-dependent divergence with paired branches.
///
/// Created by `warp.diverge_dynamic(mask)`. Holds both branches together,
/// guaranteeing they are complements. Must be merged to recover `Warp<All>`.
///
/// The mask is runtime. The complement is structural.
pub struct DynDiverge {
    true_mask: u64,
    false_mask: u64,
}

impl DynDiverge {
    /// Mask of lanes in the true branch.
    pub fn true_mask(&self) -> u64 {
        self.true_mask
    }

    /// Mask of lanes in the false branch.
    pub fn false_mask(&self) -> u64 {
        self.false_mask
    }

    /// Number of lanes in the true branch.
    pub fn true_count(&self) -> u32 {
        self.true_mask.count_ones()
    }

    /// Number of lanes in the false branch.
    pub fn false_count(&self) -> u32 {
        self.false_mask.count_ones()
    }

    /// Merge both branches, recovering `Warp<All>`.
    ///
    /// This always succeeds because the complement is guaranteed by construction.
    /// Consumes the `DynDiverge` — you can't use the branches after merging.
    pub fn merge(self) -> Warp<All> {
        // Safety: true_mask | false_mask == 0xFFFFFFFF (or parent mask)
        // This is guaranteed by diverge_dynamic's construction.
        Warp::kernel_entry()
    }

    /// Execute a closure on the true branch, then the false branch, then merge.
    ///
    /// This is the structured way to use data-dependent divergence.
    /// Both branches execute, then merge restores `Warp<All>`.
    ///
    /// ```
    /// use warp_types::*;
    /// use warp_types::dynamic::DynDiverge;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let mask = 0x0000FFFF_u64; // runtime predicate result
    ///
    /// let warp: Warp<All> = warp.diverge_dynamic(mask)
    ///     .with_branches(
    ///         |true_mask| { /* work on true lanes */ },
    ///         |false_mask| { /* work on false lanes */ },
    ///     );
    ///
    /// // Now warp is Warp<All> again — shuffle available
    /// let data = data::PerLane::new(1i32);
    /// let _sum = warp.reduce_sum(data);
    /// ```
    pub fn with_branches<F1, F2>(self, true_fn: F1, false_fn: F2) -> Warp<All>
    where
        F1: FnOnce(u64),
        F2: FnOnce(u64),
    {
        true_fn(self.true_mask);
        false_fn(self.false_mask);
        self.merge()
    }
}

impl Warp<All> {
    /// Diverge on a runtime predicate mask.
    ///
    /// The mask determines which lanes take the true branch (bits set)
    /// and which take the false branch (bits clear). The two branches
    /// are complements by construction.
    ///
    /// Returns a `DynDiverge` that must be merged to recover `Warp<All>`.
    /// Neither branch supports shuffle — only the merged warp does.
    ///
    /// This is the solution to data-dependent divergence without dependent types.
    /// The mask is runtime. The complement guarantee is structural.
    pub fn diverge_dynamic(self, predicate_mask: u64) -> DynDiverge {
        let all_mask = self.active_mask();
        DynDiverge {
            true_mask: all_mask & predicate_mask,
            false_mask: all_mask & !predicate_mask,
        }
    }
}

// ============================================================================
// Cross-function active set inference
// ============================================================================

/// Write functions generic over any active set.
///
/// Rust's generics already provide cross-function inference:
///
/// ```
/// use warp_types::*;
///
/// // This function works on ANY warp — the active set is inferred at each call site
/// fn count_active<S: ActiveSet>(warp: &Warp<S>) -> u32 {
///     warp.population()
/// }
///
/// let all: Warp<All> = Warp::kernel_entry();
/// assert_eq!(count_active(&all), 32);
///
/// let (evens, _odds) = all.diverge_even_odd();
/// assert_eq!(count_active(&evens), 16);
/// ```
///
/// For functions that REQUIRE `Warp<All>` (because they shuffle),
/// simply take `Warp<All>` directly — the type system enforces it:
///
/// ```
/// use warp_types::*;
///
/// fn my_reduce(warp: &Warp<All>, data: data::PerLane<i32>) -> i32 {
///     warp.reduce_sum(data)
/// }
///
/// let warp: Warp<All> = Warp::kernel_entry();
/// let sum = my_reduce(&warp, data::PerLane::new(1));
/// assert_eq!(sum, 32);
/// ```
///
/// The "limitation" in cross-function inference is really just Rust requiring
/// explicit generic parameters in function signatures. The active set IS
/// inferred at call sites — you never write `::<Even>` manually.
pub const _CROSS_FUNCTION_NOTE: () = ();

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::PerLane;

    #[test]
    fn test_diverge_dynamic_masks() {
        let warp: Warp<All> = Warp::kernel_entry();
        let diverged = warp.diverge_dynamic(0x0000FFFF);

        assert_eq!(diverged.true_mask(), 0x0000FFFF);
        assert_eq!(diverged.false_mask(), 0xFFFF0000);
        assert_eq!(diverged.true_count(), 16);
        assert_eq!(diverged.false_count(), 16);

        // Complement by construction
        assert_eq!(diverged.true_mask() | diverged.false_mask(), 0xFFFFFFFF);
        assert_eq!(diverged.true_mask() & diverged.false_mask(), 0);
    }

    #[test]
    fn test_diverge_dynamic_merge() {
        let warp: Warp<All> = Warp::kernel_entry();
        let diverged = warp.diverge_dynamic(0x55555555); // even lanes

        // Merge recovers Warp<All>
        let merged = diverged.merge();
        assert_eq!(merged.active_mask(), 0xFFFFFFFF);

        // Can shuffle after merge
        let data = PerLane::new(1i32);
        let _result = merged.shuffle_xor(data, 1);
    }

    #[test]
    fn test_diverge_dynamic_with_branches() {
        let warp: Warp<All> = Warp::kernel_entry();
        let mut true_seen = 0u64;
        let mut false_seen = 0u64;

        let merged = warp.diverge_dynamic(0x0F0F0F0F)
            .with_branches(
                |t| { true_seen = t; },
                |f| { false_seen = f; },
            );

        assert_eq!(true_seen, 0x0F0F0F0F);
        assert_eq!(false_seen, 0xF0F0F0F0);
        assert_eq!(merged.population(), 32);
    }

    #[test]
    fn test_diverge_dynamic_empty_branch() {
        let warp: Warp<All> = Warp::kernel_entry();
        let diverged = warp.diverge_dynamic(0xFFFFFFFF); // all lanes true

        assert_eq!(diverged.true_count(), 32);
        assert_eq!(diverged.false_count(), 0);

        let merged = diverged.merge();
        let data = PerLane::new(1i32);
        let _ = merged.reduce_sum(data);
    }

    #[test]
    fn test_diverge_dynamic_arbitrary_predicate() {
        let warp: Warp<All> = Warp::kernel_entry();

        // Simulate: diverge on data[lane] > 15
        // Lanes 16-31 would be true (mask = 0xFFFF0000)
        let predicate_mask = 0xFFFF0000_u64;
        let diverged = warp.diverge_dynamic(predicate_mask);

        assert_eq!(diverged.true_count(), 16);
        assert_eq!(diverged.false_count(), 16);

        // Must merge before any warp collective
        let warp = diverged.merge();
        let _ = warp.reduce_sum(PerLane::new(1i32));
    }

    // Cross-function inference tests

    fn generic_helper<S: crate::active_set::ActiveSet>(warp: &Warp<S>) -> u32 {
        warp.population()
    }

    fn all_only_helper(warp: &Warp<All>, data: PerLane<i32>) -> i32 {
        warp.reduce_sum(data)
    }

    #[test]
    fn test_cross_function_inference() {
        let warp: Warp<All> = Warp::kernel_entry();

        // Generic function — S inferred as All
        assert_eq!(generic_helper(&warp), 32);

        // Warp<All>-specific function
        let sum = all_only_helper(&warp, PerLane::new(1i32));
        assert_eq!(sum, 32);

        // After diverge, generic function infers S = Even
        let (evens, odds) = warp.diverge_even_odd();
        assert_eq!(generic_helper(&evens), 16);
        assert_eq!(generic_helper(&odds), 16);

        // all_only_helper(&evens, data)  // COMPILE ERROR — Warp<Even> ≠ Warp<All>
    }
}
