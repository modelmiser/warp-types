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
/// - `Warp<Even>` has no full-warp shuffle (would read inactive odd lanes);
///   `shuffle_xor_within` allows masks that stay within the active set
/// - `merge(Warp<Even>, Warp<Odd>)` → `Warp<All>` (complement verified)
///
/// Zero-cost: `Warp<S>` is zero-sized. The active set exists only in the type system.
///
/// Warp handles are affine — diverge consumes the parent warp via Rust's move
/// semantics, preventing use-after-diverge. Dropping a warp without merging is
/// permitted by the compiler (Rust is affine, not linear), but `#[must_use]`
/// emits a warning. The Lean calculus is affine for warps as well (paper §4.8); the Rust
/// implementation approximates this via move + must_use.
#[must_use = "dropping a Warp without merging may indicate a missing merge — \
              both branches of a diverge should be merged before shuffling"]
pub struct Warp<S: ActiveSet> {
    _phantom: PhantomData<S>,
}

impl Warp<crate::active_set::All> {
    /// Get a warp with all lanes active, at kernel entry.
    ///
    /// In real GPU code, this corresponds to kernel entry — the point
    /// where all 32 lanes are guaranteed active. Sub-warps are created
    /// via `diverge_*()` methods, which consume the parent.
    ///
    /// # Trusted entry points
    ///
    /// A `Warp<S>` is a zero-sized witness that the lanes in `S` are the
    /// converged lanes of this hardware warp. No constructor inspects
    /// hardware state, so the witness is trusted, not unforgeable. Every
    /// `unsafe` block in the typed shuffles relies on it, and these are the
    /// routes that produce one without hardware evidence:
    ///
    /// 1. `Warp::kernel_entry()` itself: it can be called anywhere, including
    ///    after a diverge or inside a raw branch. Call it once, at kernel entry.
    /// 2. Gradual ascription: `DynWarp::{all, all_64, from_mask_32,
    ///    from_mask_64, from_mask}(…).ascribe::<S>()` checks `S` only against
    ///    the caller's own mask, so it can produce any `Warp<S>` anywhere.
    ///    `DynWarp::from_static(warp)` consumes a real handle and is sound.
    /// 3. Merge checks set shape, not provenance: `Even` from one
    ///    `kernel_entry()` handle and `Odd` from another merge without error.
    /// 4. A raw data-dependent branch around a `&self` call
    ///    (`if tid < 16 { warp.shuffle_xor(..) }`) is ordinary Rust and is
    ///    exactly the hardware hazard; the guarantee covers programs that
    ///    express divergence through `diverge`/`merge` only.
    /// 5. The `research` feature (host-only, `cfg(not(target_arch = "nvptx64"))`)
    ///    compiles design explorations whose `diverge`/`merge` mint handles
    ///    freely. Never enable it for device code.
    ///
    /// Everything else that yields a handle — `diverge_*`, `merge`, `Warp::tile`,
    /// `Tile::partition_*`, `DynDiverge::merge`, the fence witnesses — consumes
    /// or borrows an existing witness, so it is derived, not a trust root.
    /// `merge(Warp<All>, Warp<Empty>)` is accepted (`ComplementOf<Empty> for
    /// All`); it is the identity. Inside the crate, `Warp::new()` is
    /// `pub(crate)`. The Lean calculus has the same gap: its `warpVal` literals
    /// are unrestricted (paper §4.8).
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

    /// Intended barrier synchronization for active lanes.
    ///
    /// **Currently a no-op on every target:** it does not emit `__syncwarp`.
    /// The intended lowering is `__syncwarp(S::MASK)`, naming only active lanes.
    pub fn sync(&self) {
        // In real implementation: __syncwarp(S::MASK)
    }
}

// NOTE: No Default impl for Warp<All>. Warp handles should only be created
// via kernel_entry() (public) or diverge (which consumes the parent); the
// other routes are listed under "Trusted entry points" on kernel_entry.
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
        assert_eq!(w.active_mask(), All::MASK);
        assert_eq!(w.population(), crate::WARP_SIZE);
    }

    #[test]
    fn test_warp_even() {
        let w: Warp<Even> = Warp::new();
        assert_eq!(w.active_set_name(), "Even");
        assert_eq!(w.population(), crate::WARP_SIZE / 2);
    }

    #[test]
    fn test_warp_kernel_entry() {
        let w: Warp<All> = Warp::kernel_entry();
        assert_eq!(w.population(), crate::WARP_SIZE);
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
