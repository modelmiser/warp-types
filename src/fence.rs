//! Fence-divergence interaction types (§5.6).
//!
//! Global memory writes from diverged warps must be carefully tracked.
//! A fence is only valid after ALL lanes have written — which requires
//! the same complement proof used for merge.
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
//! This turns both memory ordering bugs and cross-region confusion into type errors.

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

/// No writes have occurred.
#[derive(Debug, Clone, Copy)]
pub struct Unwritten;
#[allow(private_interfaces)]
impl crate::active_set::sealed::Sealed for Unwritten {
    fn _sealed() -> crate::active_set::sealed::SealToken {
        crate::active_set::sealed::SealToken
    }
}
impl WriteState for Unwritten {}

/// Partial write: only lanes in `S` have written.
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

/// All lanes have written (complement-verified).
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
    /// Store values to global memory.
    ///
    /// Returns the warp (unchanged) and a partially-written region
    /// that tracks which lanes have written. The lifetime `'r` is
    /// preserved, tying the partial write to its origin region.
    ///
    /// **Note:** Even `Warp<All>` produces `PartialWrite<All>`, not `FullWrite`.
    /// Use `global_store_complement` with the complement's partial write to
    /// advance to `FullWrite`, or call this then `merge_writes` with a
    /// `PartialWrite<Empty>` (which `Empty: ComplementOf<All>` satisfies).
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

    /// Store values to a region that already has a partial write from
    /// complementary lanes, producing a full write.
    ///
    /// Returns the warp (unchanged) so it can still be merged.
    /// The lifetime `'r` must match — both writes must target the same region.
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
/// Only callable on `GlobalRegion<FullWrite>` — the type system ensures
/// all lanes have written before the fence.
pub fn threadfence(_proof: GlobalRegion<FullWrite>) -> GlobalRegion<Fenced> {
    // In real implementation: __threadfence()
    GlobalRegion {
        _brand: PhantomData,
        _state: PhantomData,
    }
}

impl GlobalRegion<'_, Fenced> {
    /// Read from a fenced global region. Safe because:
    /// 1. All lanes have written (FullWrite)
    /// 2. Fence ensures visibility (Fenced)
    pub fn read<T: Default>(&self) -> T {
        T::default() // placeholder
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

            let even_partial: GlobalRegion<PartialWrite<Even>> =
                merge_writes_within(p_el, p_eh);
            let full = merge_writes(even_partial, p_odd);
            let _fenced = threadfence(full);
        });
    }
}
