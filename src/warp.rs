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
    /// `unsafe` block in the typed shuffles relies on it, as does the one in
    /// [`Warp::sync`], and these are the
    /// routes that produce one without hardware evidence:
    ///
    /// 1. `Warp::kernel_entry()` itself: it can be called anywhere, including
    ///    after a diverge or inside a raw branch. Call it once, at kernel entry.
    /// 2. Gradual ascription: `DynWarp::{all, all_64, from_mask_32,
    ///    from_mask_64, from_mask}(…).ascribe::<S>()` checks `S` only against
    ///    the caller's own mask, so it can produce any `Warp<S>` anywhere; and
    ///    `DynWarp`'s own collectives (`shuffle_xor_scalar` …) trust the same
    ///    claim after their runtime check (`require_all`: stored masks only).
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
    /// 6. Launch geometry: `kernel_entry()` claims all 32 lanes of the warp. A
    ///    block whose thread count is not a multiple of 32 has a last warp
    ///    with fewer live lanes, and the claim is false there. The collectives
    ///    differ in what follows. `shfl.sync` and `vote.sync` wait only for
    ///    the *non-exited* lanes named (PTX ISA 9.5); the ISA does not say in
    ///    so many words that a lane which was never launched counts as exited,
    ///    but the devices measured treat it so and the shuffles do not hang. A
    ///    value sourced from a missing lane is undefined. `bar.warp.sync`
    ///    has no such exemption in the ISA's text — it waits for "all threads
    ///    corresponding to membermask" — so [`Warp::sync`] does not pass
    ///    `S::MASK` to it: it names only the lanes a full-mask ballot reports
    ///    as live (see its documentation for what that rests on). Prefer
    ///    full warps all the same; the shuffles' data hazard is unchanged.
    /// 7. A live lane that never arrives: a lane that spins in the panic
    ///    handler, or sits in a raw branch that skips the call, while the
    ///    others execute a collective that names it, leaves them waiting. A
    ///    lane that has EXITED is different: `shfl.sync` and `vote.sync` do
    ///    not wait for it (item 6); the ISA does not say that of
    ///    `bar.warp.sync`, which is why `sync` names only lanes its ballot
    ///    reports.
    ///
    /// Everything else that yields a handle — `diverge_*`, `merge`, `Warp::tile`,
    /// `Tile::partition_*`, `DynDiverge::merge`, the fence witnesses — consumes
    /// or borrows an existing witness, so it is derived, not a trust root.
    /// `merge(Warp<All>, Warp<Empty>)` is accepted (`ComplementOf<Empty> for
    /// All`); it is the identity. Inside the crate, `Warp::new()` is
    /// `pub(crate)`. The Lean calculus has the same gap: its `warpVal` literals
    /// are unrestricted (paper §4.8).
    #[inline]
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
    #[inline]
    pub fn active_set_name(&self) -> &'static str {
        S::NAME
    }

    /// Bitmask of active lanes (u64 for AMD 64-lane wavefront support).
    #[inline]
    pub fn active_mask(&self) -> u64 {
        S::MASK
    }

    /// Number of active lanes.
    #[inline]
    pub fn population(&self) -> u32 {
        S::MASK.count_ones()
    }

    /// Barrier synchronization for the active lanes: `__syncwarp(S::MASK)`.
    ///
    /// On `nvptx64` every lane of the warp first takes a full-mask ballot of
    /// `true`, which reports the warp's live lanes. Each live lane in `S` then
    /// executes `bar.warp.sync` with membermask `S::MASK & live`; a lane
    /// outside `S` skips the barrier (`diverge_*` emits no branch, so every
    /// lane of the warp reaches this call, and the barrier is undefined for an
    /// executing lane its membermask does not name). On every other target —
    /// the CPU model, `amdgpu` — it emits nothing. `Warp<Empty>::sync()` emits
    /// the ballot and no barrier (no lane passes the guard).
    ///
    /// Why the ballot: the barrier waits for every lane it names, and the PTX
    /// ISA (9.5) does not exempt exited or missing lanes for `bar.warp.sync`
    /// as it does for `shfl.sync` and `vote.sync`. In the last warp of a block
    /// whose size is not a multiple of 32, `S::MASK` names lanes that do not
    /// exist. The ballot is a `vote.sync`, which waits only for non-exited
    /// lanes, and it hands every live lane the same mask, so they all pass
    /// the barrier the same membermask. (`activemask` would not do: it is not
    /// a rendezvous, so two lanes could read different masks.) What this
    /// rests on beyond the ISA's text: that a lane which does not exist
    /// contributes 0 to `vote.sync.ballot`. The ISA says so for `activemask`
    /// and for the older `vote.ballot`, not in so many words for
    /// `vote.sync.ballot`; an RTX 4000 Ada reports exactly the live lanes
    /// (block sizes 1, 16, 20, 33, 48, 63). If a device reported a missing
    /// lane as 1, `sync` would name it, as it did before the ballot.
    ///
    /// The barrier also orders memory among the lanes that take part: a lane
    /// may store, `sync()`, and then read what the other lanes of `S` stored
    /// (PTX ISA 9.5, `bar.warp.sync`).
    ///
    /// It was a no-op on every target through 0.4.3.
    ///
    /// On `.target sm_6x` and below the ISA also requires that only lanes
    /// named in some membermask be active at the barrier, so a subset barrier
    /// is undefined there; the crate is built for sm_70 and later.
    ///
    /// **It relies on the `Warp<S>` witness, like the typed shuffles.** Every
    /// live lane of the warp must reach this call (the ballot waits for them),
    /// and every live lane in `S` must reach it with the same `S` (the barrier
    /// waits for them). A forged or stale handle, or a lane that never
    /// arrives, makes this call hang or leaves it undefined — see items 1–4
    /// and 7 of "Trusted entry points" on [`Warp::kernel_entry`].
    #[inline(always)]
    pub fn sync(&self) {
        #[cfg(target_arch = "nvptx64")]
        {
            // SAFETY (ballot): `ballot_sync` needs the membermask to name the
            // executing lane — the full mask names every lane — and waits for
            // the non-exited lanes it names. That all of those reach this
            // call is NOT established here; it is the `Warp<S>` witness, taken
            // on trust (`diverge_*` emits no branch, so lanes outside `S`
            // arrive too).
            let live = unsafe { crate::gpu::ballot_sync(0xFFFF_FFFF, true) };
            // `as u32`: nvptx64 warps are 32-lane (`warp64` + nvptx64 is a
            // compile error, lib.rs).
            let members = (S::MASK as u32) & live;
            if (members >> crate::gpu::lane_id()) & 1 == 1 {
                // SAFETY (barrier): `syncwarp` asks two things. (1) The
                // membermask names the executing lane: the guard above tests
                // this lane's bit of the very mask passed. (2) Every lane the
                // membermask names executes a barrier with the same mask:
                // `members` is the same in every live lane (one ballot result
                // and a constant), and it names only lanes the ballot reports
                // live; that each of them is in fact here, behind the same
                // handle, is again the witness, taken on trust. Where the
                // witness is false the barrier hangs or is undefined.
                unsafe { crate::gpu::syncwarp(members) }
            }
        }
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
