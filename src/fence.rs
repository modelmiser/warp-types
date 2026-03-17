//! Fence-divergence interaction types (§5.6).
//!
//! Global memory writes from diverged warps must be carefully tracked.
//! A fence is only valid after ALL lanes have written — which requires
//! the same complement proof used for merge.
//!
//! # Type-state protocol
//!
//! ```text
//! GlobalRegion<Unwritten>
//!   → Warp<S1>.global_store() → GlobalRegion<PartialWrite<S1>>
//!   → merge_writes(PartialWrite<S1>, PartialWrite<S2>) → GlobalRegion<FullWrite>
//!     (requires S1: ComplementOf<S2>)
//!   → threadfence(FullWrite) → GlobalRegion<Fenced>
//! ```
//!
//! This turns a memory ordering bug into a type error.

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
// Global region with write tracking
// ============================================================================

/// A global memory region with type-state tracked write progress.
///
/// The type parameter `S` tracks which write state the region is in.
/// Operations are only available in the correct state:
/// - `global_store()` requires a warp (any active set)
/// - `merge_writes()` requires complementary partial writes
/// - `threadfence()` requires full write
/// - Reading requires fenced state
#[must_use = "GlobalRegion tracks write progress — dropping it loses the write-state proof"]
pub struct GlobalRegion<S: WriteState> {
    _phantom: PhantomData<S>,
}

impl GlobalRegion<Unwritten> {
    /// Create a new unwritten global region.
    pub fn new() -> Self {
        GlobalRegion {
            _phantom: PhantomData,
        }
    }
}

impl Default for GlobalRegion<Unwritten> {
    fn default() -> Self {
        Self::new()
    }
}

/// Warp writes to a global region, producing a partial write.
impl<S: ActiveSet> Warp<S> {
    /// Store values to global memory.
    ///
    /// Returns the warp (unchanged) and a partially-written region
    /// that tracks which lanes have written.
    ///
    /// **Note:** Even `Warp<All>` produces `PartialWrite<All>`, not `FullWrite`.
    /// Use `global_store_complement` with the complement's partial write to
    /// advance to `FullWrite`, or call this then `merge_writes` with a
    /// `PartialWrite<Empty>` (which `Empty: ComplementOf<All>` satisfies).
    pub fn global_store(
        self,
        _region: GlobalRegion<Unwritten>,
    ) -> (Self, GlobalRegion<PartialWrite<S>>) {
        (
            self,
            GlobalRegion {
                _phantom: PhantomData,
            },
        )
    }

    /// Store values to a region that already has a partial write from
    /// complementary lanes, producing a full write.
    ///
    /// Returns the warp (unchanged) so it can still be merged.
    pub fn global_store_complement<S2: ActiveSet>(
        self,
        _region: GlobalRegion<PartialWrite<S2>>,
    ) -> (Self, GlobalRegion<FullWrite>)
    where
        S: ComplementOf<S2>,
    {
        (
            self,
            GlobalRegion {
                _phantom: PhantomData,
            },
        )
    }
}

/// Merge writes from complementary partial writes (top-level: covers All).
///
/// Requires the same `ComplementOf` proof as warp merge.
///
/// Writing with wrong complement type fails:
///
/// ```compile_fail
/// use warp_types::prelude::*;
/// use warp_types::fence::*;
/// let warp1 = Warp::kernel_entry();
/// let (evens, _odds) = warp1.diverge_even_odd();
/// let warp2 = Warp::kernel_entry();
/// let (low, _high) = warp2.diverge_halves();
/// let region1 = GlobalRegion::new();
/// let region2 = GlobalRegion::new();
/// let (_evens, partial_even) = evens.global_store(region1);
/// let (_low, partial_low) = low.global_store(region2);
/// // Even and LowHalf are not complements (they overlap) — compile error
/// let _full = merge_writes(partial_even, partial_low);
/// ```
pub fn merge_writes<S1, S2>(
    _a: GlobalRegion<PartialWrite<S1>>,
    _b: GlobalRegion<PartialWrite<S2>>,
) -> GlobalRegion<FullWrite>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
{
    GlobalRegion {
        _phantom: PhantomData,
    }
}

/// Merge writes from partial writes that are complements within a parent set.
///
/// For nested divergence: e.g., EvenLow + EvenHigh within Even.
/// Returns a partial write for the parent set, not a full write.
pub fn merge_writes_within<S1, S2, P>(
    _a: GlobalRegion<PartialWrite<S1>>,
    _b: GlobalRegion<PartialWrite<S2>>,
) -> GlobalRegion<PartialWrite<P>>
where
    S1: ComplementWithin<S2, P>,
    S2: ActiveSet,
    P: ActiveSet,
{
    GlobalRegion {
        _phantom: PhantomData,
    }
}

/// Issue a thread fence after all writes are complete.
///
/// Only callable on `GlobalRegion<FullWrite>` — the type system ensures
/// all lanes have written before the fence.
pub fn threadfence(_proof: GlobalRegion<FullWrite>) -> GlobalRegion<Fenced> {
    // In real implementation: __threadfence()
    GlobalRegion {
        _phantom: PhantomData,
    }
}

impl GlobalRegion<Fenced> {
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
        let warp: Warp<All> = Warp::new();
        let (evens, odds) = warp.diverge_even_odd();

        // Each half writes to global memory
        let region = GlobalRegion::new();
        let (evens, partial_even) = evens.global_store(region);
        // evens still usable — global_store returns the warp

        // Second half completes the write — warp returned for merge
        let (odds, full) = odds.global_store_complement(partial_even);

        // Warps can still be merged after fence operations
        let _merged: Warp<All> = crate::merge(evens, odds);

        // Fence after full write
        let fenced = threadfence(full);
        let _val: i32 = fenced.read();
    }

    #[test]
    fn test_merge_writes() {
        let region1 = GlobalRegion::<PartialWrite<Even>> {
            _phantom: PhantomData,
        };
        let region2 = GlobalRegion::<PartialWrite<Odd>> {
            _phantom: PhantomData,
        };

        let full = merge_writes(region1, region2);
        let _fenced = threadfence(full);
    }

    #[test]
    fn test_nested_fence_protocol() {
        // Nested divergence: EvenLow + EvenHigh are complements within Even (not All).
        // merge_writes_within returns PartialWrite<Even>, which can then be merged
        // with PartialWrite<Odd> to get FullWrite.
        let el = GlobalRegion::<PartialWrite<EvenLow>> {
            _phantom: PhantomData,
        };
        let eh = GlobalRegion::<PartialWrite<EvenHigh>> {
            _phantom: PhantomData,
        };

        // Nested merge: EvenLow + EvenHigh → Even (partial)
        let even_partial: GlobalRegion<PartialWrite<Even>> = merge_writes_within(el, eh);

        // Top-level merge: Even + Odd → FullWrite
        let odd = GlobalRegion::<PartialWrite<Odd>> {
            _phantom: PhantomData,
        };
        let full = merge_writes(even_partial, odd);
        let _fenced = threadfence(full);
    }
}
