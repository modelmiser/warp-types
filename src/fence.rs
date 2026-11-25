//! Fence-divergence interaction types (§5.6).
//!
//! Global memory writes from diverged warps must be carefully tracked.
//! A fence is only valid after ALL lanes have written — which, in this model,
//! means every lane's write STEP has been recorded, by the same complement
//! proof used for merge. The states below record steps taken in the type;
//! they do not observe a store (see "in the model", below).
//!
//! # Type-state protocol
//!
//! ```text
//! GlobalRegion::with_region(|region| {
//!   // region: GlobalRegion<'r, Unwritten>
//!   //   → Warp<S1>.global_store() → GlobalRegion<'r, PartialWrite<S1>>
//!   //   → merge_writes(PartialWrite<S1>, PartialWrite<S2>) → GlobalRegion<'r, FullWrite>
//!   //     (requires S1: ComplementOf<S2>, same 'r)
//!   //   → threadfence(FullWrite) → GlobalRegion<'r, Fenced>
//! })
//! ```
//!
//! The lifetime `'r` ties all partial writes to the region that created them.
//! Two partial writes from *different* `GlobalRegion::with_region` calls have
//! different lifetimes and **cannot** be merged — the compiler rejects it.
//!
//! This turns both memory ordering bugs and cross-region confusion into type errors
//! **in the model**. What the module actually does is narrower:
//!
//! - `GlobalRegion` is phantom: zero-sized, bound to no address. `global_store`
//!   and `global_store_complement` store nothing; they record, in the type,
//!   that a set of lanes has done its writes. `read` loads nothing.
//! - On `nvptx64`, [`threadfence`] is the one step that emits code: a
//!   `membar.gl`, which makes the executing lane's own earlier stores visible
//!   at global level before its later operations. It does not wait for other
//!   lanes, so a reader cannot know from it that another lane has stored yet.
//!   Lanes of one warp that read each other's data need a barrier
//!   ([`Warp::sync`]), and for them the barrier alone is enough; the fence
//!   matters for observers outside the warp.
//! - Nothing ties a `GlobalRegion` to the pointer your kernel really writes
//!   through. The protocol is a discipline the types help you follow, not a
//!   check on your memory traffic.

use crate::active_set::{ActiveSet, ComplementOf, ComplementWithin};
use crate::warp::Warp;
use core::marker::PhantomData;

// ============================================================================
// Write state markers
// ============================================================================

/// Marker trait for global region write states.
///
/// Sealed — external crates cannot implement this trait, preventing
/// forgery of write-state transitions.
pub trait WriteState: crate::active_set::sealed::Sealed {}

/// No write step has been recorded.
#[derive(Debug, Clone, Copy)]
pub struct Unwritten;
#[allow(private_interfaces)]
impl crate::active_set::sealed::Sealed for Unwritten {
    fn _sealed() -> crate::active_set::sealed::SealToken {
        crate::active_set::sealed::SealToken
    }
}
impl WriteState for Unwritten {}

/// Partial write: the write step has been recorded for the lanes in `S` only.
#[derive(Debug, Clone, Copy)]
pub struct PartialWrite<S: ActiveSet> {
    _phantom: PhantomData<S>,
}
#[allow(private_interfaces)]
impl<S: ActiveSet> crate::active_set::sealed::Sealed for PartialWrite<S> {
    fn _sealed() -> crate::active_set::sealed::SealToken {
        crate::active_set::sealed::SealToken
    }
}
impl<S: ActiveSet> WriteState for PartialWrite<S> {}

/// The write step has been recorded for all lanes (complement-verified).
#[derive(Debug, Clone, Copy)]
pub struct FullWrite;
#[allow(private_interfaces)]
impl crate::active_set::sealed::Sealed for FullWrite {
    fn _sealed() -> crate::active_set::sealed::SealToken {
        crate::active_set::sealed::SealToken
    }
}
impl WriteState for FullWrite {}

/// Fence has been issued after full write.
#[derive(Debug, Clone, Copy)]
pub struct Fenced;
#[allow(private_interfaces)]
impl crate::active_set::sealed::Sealed for Fenced {
    fn _sealed() -> crate::active_set::sealed::SealToken {
        crate::active_set::sealed::SealToken
    }
}
impl WriteState for Fenced {}

// ============================================================================
// Global region with write tracking + region identity
// ============================================================================

/// A global memory region with type-state tracked write progress.
///
/// The lifetime `'r` is an *identity brand*: every call to
/// [`GlobalRegion::with_region`] introduces a fresh, unnameable lifetime,
/// so partial writes from different regions cannot be mixed.
///
/// The type parameter `S` tracks which write state the region is in.
/// Operations are only available in the correct state:
/// - `global_store()` requires a warp (any active set)
/// - `merge_writes()` requires complementary partial writes **from the same region**
/// - `threadfence()` requires full write
/// - Reading requires fenced state
///
/// # Region identity
///
/// ```compile_fail
/// use warp_types::prelude::*;
/// use warp_types::fence::*;
/// // Two different regions — lifetimes differ, merge is rejected.
/// GlobalRegion::with_region(|region1| {
///     GlobalRegion::with_region(|region2| {
///         let warp1 = Warp::kernel_entry();
///         let (evens, _odds) = warp1.diverge_even_odd();
///         let warp2 = Warp::kernel_entry();
///         let (_odds2, odds2b) = warp2.diverge_even_odd();
///         let (_evens, partial_even) = evens.global_store(region1);
///         let (_odds2b, partial_odd) = odds2b.global_store(region2);
///         // Cross-region merge — compile error: lifetime mismatch
///         let _full = merge_writes(partial_even, partial_odd);
///     });
/// });
/// ```
#[must_use = "GlobalRegion tracks write progress — dropping it loses the write-state proof"]
pub struct GlobalRegion<'r, S: WriteState> {
    // fn(&'r ()) -> &'r () makes 'r invariant (cannot be widened or narrowed).
    // This is critical: covariant 'r would let the compiler unify distinct
    // region lifetimes by widening one to match the other.
    _brand: PhantomData<fn(&'r ()) -> &'r ()>,
    _state: PhantomData<S>,
}

impl GlobalRegion<'_, Unwritten> {
    /// Enter a region scope. The callback receives a fresh `GlobalRegion`
    /// whose lifetime `'r` is unique and unnameable — partial writes
    /// derived from it cannot be merged with writes from any other region.
    ///
    /// # Examples
    ///
    /// ```
    /// use warp_types::prelude::*;
    /// use warp_types::fence::*;
    /// GlobalRegion::with_region(|region| {
    ///     let warp = Warp::kernel_entry();
    ///     let (evens, odds) = warp.diverge_even_odd();
    ///     let (evens, partial_even) = evens.global_store(region);
    ///     let (odds, full) = odds.global_store_complement(partial_even);
    ///     let _fenced = threadfence(full);
    /// });
    /// ```
    #[inline]
    pub fn with_region<R>(f: impl for<'r> FnOnce(GlobalRegion<'r, Unwritten>) -> R) -> R {
        f(GlobalRegion {
            _brand: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<'r> GlobalRegion<'r, Unwritten> {
    /// Split an unwritten region into two halves sharing the same
    /// lifetime brand. Each half can be stored to independently,
    /// then the resulting partial writes can be merged (they share `'r`).
    ///
    /// This is the safe way to create two partial writes from one region
    /// when using `merge_writes` or `merge_writes_within` instead of
    /// the sequential `global_store` / `global_store_complement` path.
    ///
    /// **Note:** This is a type-level model — `GlobalRegion` is phantom
    /// (zero-sized, no real memory). The lifetime brand `'r` ties the
    /// halves together but does not track which memory addresses each
    /// half covers. A real GPU memory model would need address-level
    /// tracking to prevent combining writes to disjoint address ranges.
    #[inline]
    pub fn split(self) -> (GlobalRegion<'r, Unwritten>, GlobalRegion<'r, Unwritten>) {
        (
            GlobalRegion {
                _brand: PhantomData,
                _state: PhantomData,
            },
            GlobalRegion {
                _brand: PhantomData,
                _state: PhantomData,
            },
        )
    }
}

/// Warp writes to a global region, producing a partial write.
impl<S: ActiveSet> Warp<S> {
    /// Record that the lanes of `S` have written the region. **Stores nothing**
    /// — the region is phantom; do the real store through your own pointer.
    ///
    /// Returns the warp (unchanged) and a partially-written region
    /// that tracks which lanes have written. The lifetime `'r` is
    /// preserved, tying the partial write to its origin region.
    ///
    /// **Note:** Even `Warp<All>` produces `PartialWrite<All>`, not `FullWrite`.
    /// Use `global_store_complement` with the complement's partial write to
    /// advance to `FullWrite`. The sequential path is:
    /// `warp.global_store(region)` → `warp.global_store_complement(partial)` → `FullWrite`.
    #[inline]
    pub fn global_store<'r>(
        self,
        _region: GlobalRegion<'r, Unwritten>,
    ) -> (Self, GlobalRegion<'r, PartialWrite<S>>) {
        (
            self,
            GlobalRegion {
                _brand: PhantomData,
                _state: PhantomData,
            },
        )
    }

    /// Record a write to a region that already has a partial write from
    /// complementary lanes, producing a full write. **Stores nothing**, as
    /// [`Warp::global_store`].
    ///
    /// Returns the warp (unchanged) so it can still be merged.
    /// The lifetime `'r` must match — both writes must target the same region.
    #[inline]
    pub fn global_store_complement<'r, S2: ActiveSet>(
        self,
        _region: GlobalRegion<'r, PartialWrite<S2>>,
    ) -> (Self, GlobalRegion<'r, FullWrite>)
    where
        S: ComplementOf<S2>,
    {
        (
            self,
            GlobalRegion {
                _brand: PhantomData,
                _state: PhantomData,
            },
        )
    }
}

/// Merge writes from complementary partial writes (top-level: covers All).
///
/// Requires the same `ComplementOf` proof as warp merge, AND the same
/// region lifetime `'r` — preventing cross-region merging.
///
/// Writing with wrong complement type fails:
///
/// ```compile_fail
/// use warp_types::prelude::*;
/// use warp_types::fence::*;
/// GlobalRegion::with_region(|region1| {
///     GlobalRegion::with_region(|region2| {
///         let warp1 = Warp::kernel_entry();
///         let (evens, _odds) = warp1.diverge_even_odd();
///         let warp2 = Warp::kernel_entry();
///         let (low, _high) = warp2.diverge_halves();
///         let (_evens, partial_even) = evens.global_store(region1);
///         let (_low, partial_low) = low.global_store(region2);
///         // Even and LowHalf are not complements (they overlap) — compile error
///         let _full = merge_writes(partial_even, partial_low);
///     });
/// });
/// ```
#[inline]
pub fn merge_writes<'r, S1, S2>(
    _a: GlobalRegion<'r, PartialWrite<S1>>,
    _b: GlobalRegion<'r, PartialWrite<S2>>,
) -> GlobalRegion<'r, FullWrite>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
{
    GlobalRegion {
        _brand: PhantomData,
        _state: PhantomData,
    }
}

/// Merge writes from partial writes that are complements within a parent set.
///
/// For nested divergence: e.g., EvenLow + EvenHigh within Even.
/// Returns a partial write for the parent set, not a full write.
/// The lifetime `'r` must match — both writes must target the same region.
#[inline]
pub fn merge_writes_within<'r, S1, S2, P>(
    _a: GlobalRegion<'r, PartialWrite<S1>>,
    _b: GlobalRegion<'r, PartialWrite<S2>>,
) -> GlobalRegion<'r, PartialWrite<P>>
where
    S1: ComplementWithin<S2, P>,
    S2: ActiveSet,
    P: ActiveSet,
{
    GlobalRegion {
        _brand: PhantomData,
        _state: PhantomData,
    }
}

/// Issue a thread fence after all writes are complete.
///
/// Only callable on `GlobalRegion<FullWrite>` — the types record that every
/// lane's write step has been taken (complement proof) before the fence. They
/// do not observe the writes themselves, and nothing here waits for a lane.
///
/// On `nvptx64` this emits `membar.gl` (`__threadfence()`), executed by every
/// lane that reaches the call: each lane's own earlier stores become visible
/// at global level before its later operations (PTX ISA 9.5, `membar`). On
/// every other target it emits nothing; it emitted nothing on any target
/// through 0.4.3.
#[inline]
pub fn threadfence(_proof: GlobalRegion<FullWrite>) -> GlobalRegion<Fenced> {
    #[cfg(target_arch = "nvptx64")]
    crate::gpu::threadfence();
    GlobalRegion {
        _brand: PhantomData,
        _state: PhantomData,
    }
}

impl GlobalRegion<'_, Fenced> {
    /// The read step of the protocol. **A model only: it reads no memory and
    /// returns `T::default()`.** `GlobalRegion` is phantom — zero-sized, with
    /// no address — so there is nothing here to load from. What the type
    /// records is that a read is permitted at this point: all lanes have
    /// written (`FullWrite`) and the fence has been issued (`Fenced`).
    ///
    /// For a real load through your own pointer, of data other lanes of the
    /// warp stored: store, `Warp::sync()` on a handle covering the lanes
    /// involved, then load — the barrier orders memory among the lanes that
    /// take part (PTX ISA 9.5, `bar.warp.sync`). [`threadfence`] does not wait
    /// for anyone and is not what makes that load safe; it is for observers
    /// outside the warp.
    #[inline]
    pub fn read<T: Default>(&self) -> T {
        T::default() // model placeholder: no memory behind a GlobalRegion
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_set::*;

    #[test]
    fn test_full_fence_protocol() {
        GlobalRegion::with_region(|region| {
            let warp: Warp<All> = Warp::new();
            let (evens, odds) = warp.diverge_even_odd();

            // Each half writes to global memory
            let (evens, partial_even) = evens.global_store(region);
            // evens still usable — global_store returns the warp

            // Second half completes the write — warp returned for merge
            let (odds, full) = odds.global_store_complement(partial_even);

            // Warps can still be merged after fence operations
            let _merged: Warp<All> = crate::merge(evens, odds);

            // Fence after full write
            let fenced = threadfence(full);
            let _val: i32 = fenced.read();
        });
    }

    #[test]
    fn test_merge_writes_same_region() {
        // merge_writes with split: both partials share the region lifetime
        GlobalRegion::with_region(|region| {
            let (r1, r2) = region.split();

            let warp: Warp<All> = Warp::new();
            let (evens, odds) = warp.diverge_even_odd();

            let (_evens, partial_even) = evens.global_store(r1);
            let (_odds, partial_odd) = odds.global_store(r2);

            let full = merge_writes(partial_even, partial_odd);
            let _fenced = threadfence(full);
        });
    }

    #[test]
    fn test_nested_fence_protocol() {
        // Nested divergence: EvenLow + EvenHigh are complements within Even.
        // merge_writes_within returns PartialWrite<Even>, which can then be
        // merged with PartialWrite<Odd> to get FullWrite.
        GlobalRegion::with_region(|region| {
            let (r_odd, r_nested) = region.split();
            let (r_el, r_eh) = r_nested.split();

            let warp: Warp<All> = Warp::new();
            let (evens, odds) = warp.diverge_even_odd();
            let (even_low, even_high) = evens.diverge_halves();

            let (_odds, partial_odd) = odds.global_store(r_odd);
            let (_el, partial_el) = even_low.global_store(r_el);
            let (_eh, partial_eh) = even_high.global_store(r_eh);

            // Nested merge: EvenLow + EvenHigh → Even (partial)
            let even_partial: GlobalRegion<PartialWrite<Even>> =
                merge_writes_within(partial_el, partial_eh);

            // Top-level merge: Even + Odd → FullWrite
            let full = merge_writes(even_partial, partial_odd);
            let _fenced = threadfence(full);
        });
    }

    #[test]
    fn test_global_store_complement_same_region() {
        // Sequential path: store then store_complement on the same region
        GlobalRegion::with_region(|region| {
            let warp: Warp<All> = Warp::new();
            let (evens, odds) = warp.diverge_even_odd();

            let (_evens, partial) = evens.global_store(region);
            let (_odds, full) = odds.global_store_complement(partial);
            let _fenced = threadfence(full);
        });
    }

    #[test]
    fn test_with_region_returns_value() {
        let result = GlobalRegion::with_region(|region| {
            let warp: Warp<All> = Warp::new();
            let (evens, odds) = warp.diverge_even_odd();
            let (_evens, partial) = evens.global_store(region);
            let (_odds, full) = odds.global_store_complement(partial);
            let fenced = threadfence(full);
            fenced.read::<i32>()
        });
        assert_eq!(result, 0); // Default for i32
    }

    #[test]
    fn test_split_preserves_region_identity() {
        // Splitting preserves the region lifetime — all descendants
        // can be merged because they share 'r.
        GlobalRegion::with_region(|region| {
            let (a, b) = region.split();
            let (a1, a2) = a.split();

            let warp: Warp<All> = Warp::new();
            let (evens, odds) = warp.diverge_even_odd();
            let (even_low, even_high) = evens.diverge_halves();

            let (_el, p_el) = even_low.global_store(a1);
            let (_eh, p_eh) = even_high.global_store(a2);
            let (_odds, p_odd) = odds.global_store(b);

            let even_partial: GlobalRegion<PartialWrite<Even>> = merge_writes_within(p_el, p_eh);
            let full = merge_writes(even_partial, p_odd);
            let _fenced = threadfence(full);
        });
    }
}
