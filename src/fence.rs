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

use std::marker::PhantomData;
use crate::active_set::{ActiveSet, ComplementOf};
use crate::warp::Warp;

// ============================================================================
// Write state markers
// ============================================================================

/// Marker trait for global region write states.
pub trait WriteState {}

/// No writes have occurred.
pub struct Unwritten;
impl WriteState for Unwritten {}

/// Partial write: only lanes in `S` have written.
pub struct PartialWrite<S: ActiveSet> {
    _phantom: PhantomData<S>,
}
impl<S: ActiveSet> WriteState for PartialWrite<S> {}

/// All lanes have written (complement-verified).
pub struct FullWrite;
impl WriteState for FullWrite {}

/// Fence has been issued after full write.
pub struct Fenced;
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
pub struct GlobalRegion<S: WriteState> {
    _phantom: PhantomData<S>,
}

impl GlobalRegion<Unwritten> {
    /// Create a new unwritten global region.
    pub fn new() -> Self {
        GlobalRegion { _phantom: PhantomData }
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
    pub fn global_store(self, _region: GlobalRegion<Unwritten>) -> (Self, GlobalRegion<PartialWrite<S>>) {
        (self, GlobalRegion { _phantom: PhantomData })
    }

    /// Store values to a region that already has a partial write from
    /// complementary lanes, producing a full write.
    pub fn global_store_complement<S2: ActiveSet>(
        self,
        _region: GlobalRegion<PartialWrite<S2>>,
    ) -> GlobalRegion<FullWrite>
    where
        S: ComplementOf<S2>,
    {
        GlobalRegion { _phantom: PhantomData }
    }
}

/// Merge writes from complementary partial writes.
///
/// Requires the same `ComplementOf` proof as warp merge.
pub fn merge_writes<S1, S2>(
    _a: GlobalRegion<PartialWrite<S1>>,
    _b: GlobalRegion<PartialWrite<S2>>,
) -> GlobalRegion<FullWrite>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
{
    GlobalRegion { _phantom: PhantomData }
}

/// Issue a thread fence after all writes are complete.
///
/// Only callable on `GlobalRegion<FullWrite>` — the type system ensures
/// all lanes have written before the fence.
pub fn threadfence(_proof: GlobalRegion<FullWrite>) -> GlobalRegion<Fenced> {
    // In real implementation: __threadfence()
    GlobalRegion { _phantom: PhantomData }
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
        let _ = evens; // warp still usable

        // Second half completes the write
        let full = odds.global_store_complement(partial_even);

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
        // Nested divergence: four partial writes merged pairwise
        let el = GlobalRegion::<PartialWrite<EvenLow>> { _phantom: PhantomData };
        let eh = GlobalRegion::<PartialWrite<EvenHigh>> { _phantom: PhantomData };

        // EvenLow + EvenHigh are complements
        let _full = merge_writes(el, eh);
    }
}
