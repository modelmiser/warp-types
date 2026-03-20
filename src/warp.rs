//! The core `Warp<S>` type: a warp with compile-time tracked active lanes.
//!
//! `Warp<S>` is a zero-cost wrapper that carries active-set information in
//! the type parameter `S`. Operations that require specific lane configurations
//! (like shuffle requiring all lanes) are only implemented for the matching type.
//!
//! ## Linearity: use-after-diverge is a compile error
//!
//! ```compile_fail
//! use warp_types::*;
//!
//! fn use_after_diverge() {
//!     let warp: Warp<All> = Warp::kernel_entry();
//!     let data = data::PerLane::new(42i32);
//!     let (_evens, _odds) = warp.diverge_even_odd(); // consumes warp
//!     let _ = warp.shuffle_xor(data, 1); // ERROR: use of moved value: `warp`
//! }
//! ```
//!
//! ## No duplication: Warp cannot be cloned (Lemma 4.8)
//!
//! ```compile_fail
//! use warp_types::*;
//!
//! fn warp_clone() {
//!     let warp: Warp<All> = Warp::kernel_entry();
//!     let warp2 = warp.clone(); // ERROR: no method named `clone` found for `Warp<All>`
//! }
//! ```

use crate::active_set::ActiveSet;
use core::marker::PhantomData;

/// A warp with compile-time tracked active lanes.
///
/// The type parameter `S` records which lanes are active. This enables:
/// - `Warp<All>` provides shuffle/reduce (all lanes needed)
/// - `Warp<Even>` cannot shuffle (would read inactive odd lanes)
/// - `merge(Warp<Even>, Warp<Odd>)` → `Warp<All>` (complement verified)
///
/// Zero-cost: `Warp<S>` is zero-sized. The active set exists only in the type system.
///
/// Warp handles are affine — diverge consumes the parent warp via Rust's move
/// semantics, preventing use-after-diverge. Dropping a warp without merging is
/// permitted by the compiler (Rust is affine, not linear), but `#[must_use]`
/// emits a warning. The Lean formalization models linear semantics; the Rust
/// implementation approximates this via move + must_use.
#[must_use = "dropping a Warp without merging may indicate a missing merge — \
              both branches of a diverge should be merged before shuffling"]
pub struct Warp<S: ActiveSet> {
    _phantom: PhantomData<S>,
}

impl Warp<crate::active_set::All> {
    /// The only public entry point: get a warp with all lanes active.
    ///
    /// In real GPU code, this corresponds to kernel entry — the point
    /// where all 32 lanes are guaranteed active. Sub-warps are created
    /// only via `diverge_*()` methods, which consume the parent.
    ///
    /// **Limitation:** This can be called multiple times, creating independent
    /// `Warp<All>` handles. Rust's affine type system cannot enforce that only
    /// one exists. In real GPU code, call this once at kernel entry. Calling it
    /// again after a diverge would bypass the typestate discipline — don't do
    /// that. The Lean formalization models linear semantics where this is
    /// enforced; the Rust implementation trusts the programmer here.
    pub fn kernel_entry() -> Self {
        Warp {
            _phantom: PhantomData,
        }
    }
}

impl<S: ActiveSet> Warp<S> {
    /// Create a warp with any active set (crate-internal only).
    ///
    /// External code must use `Warp::kernel_entry()` to get a `Warp<All>`,
    /// then `diverge_*()` to get sub-warps. This prevents forging arbitrary
    /// warp handles that don't correspond to actual lane states.
    ///
    /// External code cannot construct arbitrary warps:
    ///
    /// ```compile_fail
    /// use warp_types::Warp;
    /// use warp_types::active_set::Even;
    /// // Warp::new() is pub(crate), not accessible externally
    /// let fake: Warp<Even> = Warp::new();
    /// ```
    pub(crate) fn new() -> Self {
        Warp {
            _phantom: PhantomData,
        }
    }

    /// Human-readable name of the active set.
    pub fn active_set_name(&self) -> &'static str {
        S::NAME
    }

    /// Bitmask of active lanes (u64 for AMD 64-lane wavefront support).
    pub fn active_mask(&self) -> u64 {
        S::MASK
    }

    /// Number of active lanes.
    pub fn population(&self) -> u32 {
        S::MASK.count_ones()
    }

    /// Barrier synchronization for active lanes.
    ///
    /// Safe because we only sync the lanes that are actually active.
    /// In a real implementation: `__syncwarp(S::MASK)`.
    pub fn sync(&self) {
        // In real implementation: __syncwarp(S::MASK)
    }
}

// NOTE: No Default impl for Warp<All>. Warp handles should only be created
// via kernel_entry() (public) or diverge (which consumes the parent).
// A Default impl would allow silent construction in generic contexts
// (unwrap_or_default, etc.), weakening the entry-point discipline.

impl<S: ActiveSet> core::fmt::Debug for Warp<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Warp<{}>(mask={:016X})", S::NAME, S::MASK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_set::*;

    #[test]
    fn test_warp_all() {
        let w: Warp<All> = Warp::new();
        assert_eq!(w.active_set_name(), "All");
        assert_eq!(w.active_mask(), 0xFFFFFFFF);
        assert_eq!(w.population(), 32);
    }

    #[test]
    fn test_warp_even() {
        let w: Warp<Even> = Warp::new();
        assert_eq!(w.active_set_name(), "Even");
        assert_eq!(w.population(), 16);
    }

    #[test]
    fn test_warp_kernel_entry() {
        let w: Warp<All> = Warp::kernel_entry();
        assert_eq!(w.population(), 32);
    }

    #[test]
    fn test_warp_debug() {
        let w: Warp<Even> = Warp::new();
        let s = format!("{:?}", w);
        assert!(s.contains("Even"));
    }

    #[test]
    fn test_sync_any_active_set() {
        let all: Warp<All> = Warp::new();
        let even: Warp<Even> = Warp::new();
        let odd: Warp<Odd> = Warp::new();
        // All compile — sync works on any active set
        all.sync();
        even.sync();
        odd.sync();
    }
}
