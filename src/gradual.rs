//! Gradual typing: `DynWarp` ↔ `Warp<S>` bridge.
//!
//! The migration path from untyped to typed warp programming:
//!
//! 1. **Start**: `DynWarp::all()` — all operations runtime-checked
//! 2. **Boundary**: `dyn_warp.ascribe::<All>()?` — validate at function edges
//! 3. **End**: `Warp<S>` everywhere — fully compile-time checked
//!
//! `DynWarp` provides a subset of the `Warp<S>` API with scalar (i32) shuffle
//! operations. Generic shuffles require the static type system. It carries its
//! active mask at runtime instead of in the type system. Every operation that
//! would be a type error on `Warp<S>` becomes a `Result::Err` on `DynWarp`.
//!
//! # Cost
//!
//! `DynWarp` is NOT zero-overhead: it carries a `u64` mask (8 bytes).
//! `Warp<S>` is zero-sized. Migrating from `DynWarp` to `Warp<S>` is both
//! a safety upgrade (compile-time vs runtime) and a performance upgrade
//! (zero-sized vs 8 bytes + branch per operation).
//!
//! # Example: Gradual Migration
//!
//! ```
//! use warp_types::gradual::DynWarp;
//! use warp_types::{Warp, All, Even, Odd, ActiveSet};
//!
//! // Phase 1: All dynamic — catches bugs at runtime
//! let dyn_warp = DynWarp::all();
//! let (evens, odds) = dyn_warp.diverge(Even::MASK);
//! assert!(evens.shuffle_xor_scalar(42, 1).is_err()); // Caught!
//! let merged = evens.merge(odds).unwrap();
//! assert!(merged.shuffle_xor_scalar(42, 1).is_ok());
//!
//! // Phase 2: Ascribe to static type at boundary
//! let merged = DynWarp::all();
//! let warp: Warp<All> = merged.ascribe::<All>().unwrap();
//! assert_eq!(warp.active_mask(), 0xFFFFFFFF);
//!
//! // Phase 3: Erase back to dynamic for interop with untyped code
//! let dyn_again = DynWarp::from_static(warp);
//! assert_eq!(dyn_again.active_mask(), 0xFFFFFFFF);
//! ```

use crate::active_set::ActiveSet;
use crate::warp::Warp;

/// Error when a runtime warp operation violates a safety invariant.
///
/// Contains enough information to diagnose the failure:
/// which operation, what mask was expected, what mask was present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarpError {
    pub operation: &'static str,
    pub expected_mask: u64,
    pub actual_mask: u64,
}

impl core::fmt::Display for WarpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}: expected mask 0x{:08X}, got 0x{:08X}",
            self.operation, self.expected_mask, self.actual_mask
        )
    }
}

/// Error when ascribing a `DynWarp` to a specific `Warp<S>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AscribeError {
    pub expected_name: &'static str,
    pub expected_mask: u64,
    pub actual_mask: u64,
}

impl core::fmt::Display for AscribeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ascribe to {}: expected mask 0x{:08X}, got 0x{:08X}",
            self.expected_name, self.expected_mask, self.actual_mask
        )
    }
}

#[cfg(not(target_arch = "nvptx64"))]
impl std::error::Error for WarpError {}

#[cfg(not(target_arch = "nvptx64"))]
impl std::error::Error for AscribeError {}

/// A dynamically-checked warp — runtime equivalent of `Warp<S>`.
///
/// Every safety check that `Warp<S>` enforces at compile time, `DynWarp`
/// enforces at runtime. This makes it suitable for:
///
/// - **Prototyping**: Write warp code quickly, add types later
/// - **Migration**: Convert existing untyped GPU code incrementally
/// - **Testing**: Verify mask logic with runtime assertions before committing to types
/// - **Dynamic predicates**: When the active set depends on runtime data
///
/// Use `ascribe::<S>()` to promote to `Warp<S>` when the mask is known.
/// Use `DynWarp::from_static()` to demote `Warp<S>` to `DynWarp`.
#[derive(Clone, Debug)]
pub struct DynWarp {
    active_mask: u64,
    /// The full mask representing "all lanes active" for this warp width.
    /// 0xFFFFFFFF for 32-lane (NVIDIA), 0xFFFFFFFFFFFFFFFF for 64-lane (AMD).
    full_mask: u64,
}

impl DynWarp {
    /// All 32 lanes active (NVIDIA warp size).
    ///
    /// For AMD 64-lane wavefronts, use `DynWarp::all_64()`.
    pub fn all() -> Self {
        DynWarp { active_mask: 0xFFFFFFFF, full_mask: 0xFFFFFFFF }
    }

    /// All 64 lanes active (AMD wavefront size).
    pub fn all_64() -> Self {
        DynWarp { active_mask: 0xFFFFFFFFFFFFFFFF, full_mask: 0xFFFFFFFFFFFFFFFF }
    }

    /// Create from a specific mask within a 32-lane warp.
    ///
    /// Useful for testing or constructing `DynWarp`s with known masks.
    /// For production code, prefer `DynWarp::all()` or `DynWarp::from_static()`.
    pub fn from_mask(mask: u64) -> Self {
        DynWarp { active_mask: mask, full_mask: 0xFFFFFFFF }
    }

    /// Erase a static `Warp<S>` into a dynamic warp (always succeeds).
    ///
    /// This is the "forget" direction: we discard compile-time information
    /// and move to runtime tracking. Always safe — going from more
    /// information to less.
    pub fn from_static<S: ActiveSet>(_warp: Warp<S>) -> Self {
        // Determine warp width from mask: if it fits in 32 bits, use 32-lane
        let full = if S::MASK <= 0xFFFFFFFF { 0xFFFFFFFF } else { 0xFFFFFFFFFFFFFFFF };
        DynWarp { active_mask: S::MASK, full_mask: full }
    }

    /// Promote this `DynWarp` to a compile-time typed `Warp<S>`.
    ///
    /// Succeeds only if the runtime mask matches `S::MASK` exactly.
    /// This is the gradual typing boundary: the point where runtime
    /// evidence becomes compile-time proof.
    ///
    /// ```
    /// use warp_types::gradual::DynWarp;
    /// use warp_types::{Warp, All, Even};
    ///
    /// let dyn_warp = DynWarp::all();
    /// let warp: Warp<All> = dyn_warp.ascribe::<All>().unwrap();
    ///
    /// // Wrong ascription fails:
    /// let dyn_warp = DynWarp::all();
    /// assert!(dyn_warp.ascribe::<Even>().is_err());
    /// ```
    pub fn ascribe<S: ActiveSet>(self) -> Result<Warp<S>, AscribeError> {
        if self.active_mask == S::MASK {
            Ok(Warp::new())
        } else {
            Err(AscribeError {
                expected_name: S::NAME,
                expected_mask: S::MASK,
                actual_mask: self.active_mask,
            })
        }
    }

    /// Current active lane mask.
    pub fn active_mask(&self) -> u64 {
        self.active_mask
    }

    /// Number of active lanes.
    pub fn population(&self) -> u32 {
        self.active_mask.count_ones()
    }

    // ========================================================================
    // Operations that require All lanes (runtime-checked)
    // ========================================================================

    /// Shuffle XOR on a single scalar — runtime check for all-active.
    ///
    /// The `Warp<S>` equivalent only exists on `Warp<All>`.
    /// `DynWarp` checks at runtime instead.
    pub fn shuffle_xor_scalar(&self, value: i32, _xor_mask: u32) -> Result<i32, WarpError> {
        let full = self.full_mask;
        if self.active_mask != full {
            return Err(WarpError {
                operation: "shuffle_xor",
                expected_mask: full,
                actual_mask: self.active_mask,
            });
        }
        // In a real implementation: __shfl_xor_sync(0xFFFFFFFF, value, xor_mask)
        // For the type system prototype, we model the XOR partner selection:
        Ok(value) // placeholder — real shuffle reads from partner lane
    }

    /// Shuffle down on a single scalar — runtime check for all-active.
    pub fn shuffle_down_scalar(&self, value: i32, _delta: u32) -> Result<i32, WarpError> {
        let full = self.full_mask;
        if self.active_mask != full {
            return Err(WarpError {
                operation: "shuffle_down",
                expected_mask: full,
                actual_mask: self.active_mask,
            });
        }
        Ok(value) // placeholder — real shuffle reads from partner lane
    }

    /// Sum reduction — runtime check for all-active.
    pub fn reduce_sum_scalar(&self, value: i32) -> Result<i32, WarpError> {
        let full = self.full_mask;
        if self.active_mask != full {
            return Err(WarpError {
                operation: "reduce_sum",
                expected_mask: full,
                actual_mask: self.active_mask,
            });
        }
        // CPU single-thread: butterfly doubling gives value * warp_width
        let warp_width = full.count_ones() as i32;
        Ok(value.wrapping_mul(warp_width))
    }

    /// Broadcast — runtime check for all-active.
    pub fn broadcast_scalar(&self, value: i32) -> Result<i32, WarpError> {
        let full = self.full_mask;
        if self.active_mask != full {
            return Err(WarpError {
                operation: "broadcast",
                expected_mask: full,
                actual_mask: self.active_mask,
            });
        }
        Ok(value)
    }

    /// Ballot — runtime check for all-active.
    ///
    /// Only supports 32-lane warps. Returns an error for 64-lane warps
    /// because the `[bool; 32]` input and `u32` return type cannot represent
    /// 64 lanes. Use the static `Warp<All>` API for 64-lane ballot.
    pub fn ballot(&self, predicate: &[bool; 32]) -> Result<u32, WarpError> {
        let full = self.full_mask;
        if full > 0xFFFFFFFF {
            return Err(WarpError {
                operation: "ballot (64-lane warp incompatible with u32 result)",
                expected_mask: 0xFFFFFFFF,
                actual_mask: full,
            });
        }
        if self.active_mask != full {
            return Err(WarpError {
                operation: "ballot",
                expected_mask: full,
                actual_mask: self.active_mask,
            });
        }
        let mut mask = 0u32;
        for (i, &p) in predicate.iter().enumerate() {
            if p {
                mask |= 1 << i;
            }
        }
        Ok(mask)
    }

    // ========================================================================
    // Diverge / Merge (runtime-checked)
    // ========================================================================

    /// Split by predicate mask. Always succeeds.
    ///
    /// Returns two `DynWarp`s with disjoint masks that together cover
    /// the original. This is the runtime equivalent of `warp.diverge()`.
    pub fn diverge(self, predicate_mask: u64) -> (DynWarp, DynWarp) {
        let true_mask = self.active_mask & predicate_mask;
        let false_mask = self.active_mask & !predicate_mask;
        (
            DynWarp { active_mask: true_mask, full_mask: self.full_mask },
            DynWarp { active_mask: false_mask, full_mask: self.full_mask },
        )
    }

    /// Merge with another `DynWarp`. Runtime check: masks must be disjoint.
    ///
    /// This is the runtime equivalent of `merge(a, b)` which requires
    /// `ComplementOf` at compile time. Here we check disjointness at runtime.
    ///
    /// **Invariant difference from static path**: The static `merge()` requires
    /// `ComplementOf<S1, S2>` which checks BOTH disjointness AND covering
    /// (S1 ∪ S2 = All). `DynWarp::merge` only checks disjointness — it cannot
    /// check covering because it doesn't track the parent set. Two small disjoint
    /// DynWarps can merge without recovering All. Use `ascribe::<All>()` after
    /// merge to verify the result covers all lanes.
    pub fn merge(self, other: DynWarp) -> Result<DynWarp, WarpError> {
        if self.full_mask != other.full_mask {
            return Err(WarpError {
                operation: "merge (full_mask mismatch)",
                expected_mask: self.full_mask,
                actual_mask: other.full_mask,
            });
        }
        let overlap = self.active_mask & other.active_mask;
        if overlap != 0 {
            return Err(WarpError {
                operation: "merge",
                expected_mask: 0, // expected no overlap
                actual_mask: overlap,
            });
        }
        Ok(DynWarp {
            active_mask: self.active_mask | other.active_mask,
            full_mask: self.full_mask,
        })
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_set::*;

    // --- Construction and ascription ---

    #[test]
    fn dyn_warp_all() {
        let w = DynWarp::all();
        assert_eq!(w.active_mask(), 0xFFFFFFFF);
        assert_eq!(w.population(), 32);
    }

    #[test]
    fn ascribe_all_succeeds() {
        let w = DynWarp::all();
        let warp: Warp<All> = w.ascribe().unwrap();
        assert_eq!(warp.active_mask(), 0xFFFFFFFF);
    }

    #[test]
    fn ascribe_wrong_type_fails() {
        let w = DynWarp::all();
        let err = w.ascribe::<Even>().unwrap_err();
        assert_eq!(err.expected_name, "Even");
        assert_eq!(err.expected_mask, 0x55555555);
        assert_eq!(err.actual_mask, 0xFFFFFFFF);
    }

    #[test]
    fn from_static_roundtrip() {
        let warp: Warp<Even> = Warp::new();
        let dyn_warp = DynWarp::from_static(warp);
        assert_eq!(dyn_warp.active_mask(), Even::MASK);

        // Ascribe back to the same type
        let _: Warp<Even> = dyn_warp.ascribe().unwrap();
    }

    // --- Shuffle safety ---

    #[test]
    fn shuffle_all_succeeds() {
        let w = DynWarp::all();
        assert!(w.shuffle_xor_scalar(42, 1).is_ok());
    }

    #[test]
    fn shuffle_partial_fails() {
        let w = DynWarp::from_mask(Even::MASK);
        let err = w.shuffle_xor_scalar(42, 1).unwrap_err();
        assert_eq!(err.operation, "shuffle_xor");
        assert_eq!(err.actual_mask, Even::MASK);
    }

    // --- Ballot safety ---

    #[test]
    fn ballot_all_succeeds() {
        let w = DynWarp::all();
        let pred = [true; 32];
        assert_eq!(w.ballot(&pred).unwrap(), 0xFFFFFFFF);
    }

    #[test]
    fn ballot_partial_fails() {
        let w = DynWarp::from_mask(LowHalf::MASK);
        let pred = [true; 32];
        assert!(w.ballot(&pred).is_err());
    }

    // --- Diverge / Merge ---

    #[test]
    fn diverge_produces_disjoint_masks() {
        let w = DynWarp::all();
        let (evens, odds) = w.diverge(Even::MASK);
        assert_eq!(evens.active_mask(), Even::MASK);
        assert_eq!(odds.active_mask(), Odd::MASK);
        assert_eq!(evens.active_mask() & odds.active_mask(), 0);
    }

    #[test]
    fn merge_disjoint_succeeds() {
        let evens = DynWarp::from_mask(Even::MASK);
        let odds = DynWarp::from_mask(Odd::MASK);
        let merged = evens.merge(odds).unwrap();
        assert_eq!(merged.active_mask(), 0xFFFFFFFF);
    }

    #[test]
    fn merge_overlapping_fails() {
        let a = DynWarp::from_mask(0x0000FFFF); // LowHalf
        let b = DynWarp::from_mask(0x55555555); // Even — overlaps with LowHalf
        assert!(a.merge(b).is_err());
    }

    // --- Full migration workflow ---

    #[test]
    fn gradual_migration_workflow() {
        // Phase 1: Dynamic — discover the bug at runtime
        let w = DynWarp::all();
        let (evens, odds) = w.diverge(Even::MASK);
        assert!(evens.shuffle_xor_scalar(42, 1).is_err()); // Bug caught!
        let merged = evens.merge(odds).unwrap();
        assert!(merged.shuffle_xor_scalar(42, 1).is_ok()); // After merge: safe

        // Phase 2: Ascribe to static type
        let all = DynWarp::all();
        let warp: Warp<All> = all.ascribe().unwrap();
        // Now we have compile-time safety: warp.shuffle_xor exists on Warp<All>
        assert_eq!(warp.population(), 32);

        // Phase 3: Can go back to dynamic for interop
        let dyn_again = DynWarp::from_static(warp);
        assert_eq!(dyn_again.active_mask(), 0xFFFFFFFF);
    }

    #[test]
    fn nested_diverge_merge_dynamic() {
        let w = DynWarp::all();

        // Diverge into halves
        let (low, high) = w.diverge(LowHalf::MASK);
        assert_eq!(low.population(), 16);
        assert_eq!(high.population(), 16);

        // Diverge low half further
        let (even_low, odd_low) = low.diverge(Even::MASK);
        assert_eq!(even_low.active_mask(), EvenLow::MASK);
        assert_eq!(odd_low.active_mask(), OddLow::MASK);

        // Can't shuffle on any subset
        assert!(even_low.shuffle_xor_scalar(1, 1).is_err());

        // Merge back: even_low + odd_low = low_half
        let low_restored = even_low.merge(odd_low).unwrap();
        assert_eq!(low_restored.active_mask(), LowHalf::MASK);

        // Merge back: low + high = all
        let all = low_restored.merge(high).unwrap();
        assert_eq!(all.active_mask(), 0xFFFFFFFF);

        // Now ascribe to static type
        let _warp: Warp<All> = all.ascribe().unwrap();
    }

    // --- 64-lane (AMD wavefront) ---

    #[test]
    fn dyn_warp_all_64() {
        let w = DynWarp::all_64();
        assert_eq!(w.active_mask(), 0xFFFFFFFFFFFFFFFF);
        assert_eq!(w.population(), 64);
    }

    #[test]
    fn reduce_sum_64_lane() {
        let w = DynWarp::all_64();
        let result = w.reduce_sum_scalar(1).unwrap();
        assert_eq!(result, 64); // Not 32!
    }

    #[test]
    fn ballot_64_lane_errors() {
        let w = DynWarp::all_64();
        let pred = [true; 32];
        let err = w.ballot(&pred).unwrap_err();
        assert!(err.operation.contains("64-lane"));
    }

    #[test]
    fn shuffle_64_lane_succeeds() {
        let w = DynWarp::all_64();
        assert!(w.shuffle_xor_scalar(42, 1).is_ok());
    }

    #[test]
    fn merge_mismatched_width_fails() {
        let a = DynWarp::all();       // 32-lane
        let b = DynWarp::all_64();    // 64-lane
        let (a1, a2) = a.diverge(Even::MASK);
        let (b1, _b2) = b.diverge(Even::MASK);
        // Can't merge 32-lane and 64-lane halves
        assert!(a1.clone().merge(b1).is_err());
        // Can merge same-width halves
        assert!(a1.merge(a2).is_ok());
    }

    #[test]
    fn error_messages_are_clear() {
        let w = DynWarp::from_mask(Even::MASK);
        let err = w.shuffle_xor_scalar(42, 1).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("shuffle_xor"));
        assert!(msg.contains("FFFFFFFF")); // expected
        assert!(msg.contains("55555555")); // actual

        let err = DynWarp::from_mask(0x1234).ascribe::<All>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("All"));
        assert!(msg.contains("00001234"));
    }
}
