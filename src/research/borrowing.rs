//! Borrowing Patterns for GPU Shared Memory
//!
//! **STATUS: Research** — Exploratory prototype, not promoted to main API.
//!
//! Research question: "How to handle temporary shared access (borrowing)?"
//!
//! # Background
//!
//! GPU shared memory is a scarce resource. We want to:
//! 1. Prevent race conditions (multiple writers)
//! 2. Allow temporary access (borrowing)
//! 3. Track ownership precisely (linear types)
//!
//! Rust's borrow checker works for CPU memory. Can we adapt it to GPU?
//!
//! # Key Insight
//!
//! GPU borrowing differs from CPU borrowing:
//! - Multiple LANES may access simultaneously (within a warp)
//! - Different WARPS need explicit synchronization
//! - SIMT means all lanes do the same operation (or are masked)
//!
//! # Patterns Explored
//!
//! 1. **Lane-Parallel Borrows**: All lanes borrow in lockstep
//! 2. **Split Borrows**: Different lanes borrow different indices
//! 3. **Scoped Borrows**: RAII-style with sync at scope end
//! 4. **Lease/Return**: Explicit checkout and return

use std::marker::PhantomData;

// ============================================================================
// BASIC TYPES
// ============================================================================

pub trait ActiveSet: Copy + 'static {
    const MASK: u32;
}

#[derive(Copy, Clone)]
pub struct All;
impl ActiveSet for All {
    const MASK: u32 = 0xFFFFFFFF;
}

#[derive(Copy, Clone)]
pub struct Warp<S: ActiveSet> {
    _marker: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self {
        Warp {
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// SHARED MEMORY REGION
// ============================================================================

/// A region of shared memory with explicit owner
#[derive(Debug)]
pub struct SharedMem<T: Copy, const SIZE: usize> {
    data: [T; SIZE],
}

impl<T: Copy + Default, const SIZE: usize> SharedMem<T, SIZE> {
    pub fn new() -> Self {
        SharedMem {
            data: [T::default(); SIZE],
        }
    }

    pub fn from_slice(values: &[T]) -> Self
    where
        [T; SIZE]: Default,
    {
        let mut data = <[T; SIZE]>::default();
        for (i, v) in values.iter().take(SIZE).enumerate() {
            data[i] = *v;
        }
        SharedMem { data }
    }
}

// ============================================================================
// PATTERN 1: LANE-PARALLEL BORROWS
// ============================================================================

/// Lane-parallel borrowing: All lanes borrow the same index
///
/// In SIMT, all lanes execute the same instruction. When borrowing,
/// all lanes borrow the SAME logical slot but from their own perspective.
pub mod lane_parallel {
    use super::*;

    /// Immutable borrow - all lanes read the same value
    pub struct SharedRef<'a, T: Copy, const SIZE: usize> {
        mem: &'a SharedMem<T, SIZE>,
    }

    impl<'a, T: Copy, const SIZE: usize> SharedRef<'a, T, SIZE> {
        pub fn read(&self, index: usize) -> T {
            assert!(index < SIZE, "index out of bounds");
            self.mem.data[index]
        }
    }

    /// Mutable borrow - exclusive write access
    pub struct SharedMut<'a, T: Copy, const SIZE: usize> {
        mem: &'a mut SharedMem<T, SIZE>,
    }

    impl<'a, T: Copy, const SIZE: usize> SharedMut<'a, T, SIZE> {
        pub fn read(&self, index: usize) -> T {
            assert!(index < SIZE, "index out of bounds");
            self.mem.data[index]
        }

        pub fn write(&mut self, index: usize, value: T) {
            assert!(index < SIZE, "index out of bounds");
            self.mem.data[index] = value;
        }
    }

    /// Borrow shared memory for reading
    pub fn borrow_shared<'a, T: Copy, const SIZE: usize>(
        _warp: &Warp<All>,
        mem: &'a SharedMem<T, SIZE>,
    ) -> SharedRef<'a, T, SIZE> {
        SharedRef { mem }
    }

    /// Borrow shared memory for writing (exclusive)
    pub fn borrow_mut<'a, T: Copy, const SIZE: usize>(
        _warp: &Warp<All>,
        mem: &'a mut SharedMem<T, SIZE>,
    ) -> SharedMut<'a, T, SIZE> {
        SharedMut { mem }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_shared_borrow() {
            let warp: Warp<All> = Warp::new();
            let mut mem: SharedMem<i32, 32> = SharedMem::new();

            // Write via mutable borrow
            {
                let mut borrow = borrow_mut(&warp, &mut mem);
                borrow.write(0, 42);
            }

            // Read via shared borrow
            {
                let borrow = borrow_shared(&warp, &mem);
                assert_eq!(borrow.read(0), 42);
            }
        }

        #[test]
        fn test_multiple_shared_borrows() {
            let warp: Warp<All> = Warp::new();
            let mem: SharedMem<i32, 32> = SharedMem::new();

            // Multiple shared borrows allowed
            let b1 = borrow_shared(&warp, &mem);
            let b2 = borrow_shared(&warp, &mem);

            assert_eq!(b1.read(0), b2.read(0));
        }
    }
}

// ============================================================================
// PATTERN 2: SPLIT BORROWS (Disjoint Index Sets)
// ============================================================================

/// Split borrows: Different lanes access different indices
///
/// Key insight: If lanes access DISJOINT indices, no race is possible.
/// We can grant "parallel exclusive" access where each lane owns its slot.
pub mod split {
    use super::*;

    /// Per-lane exclusive access to a single index
    pub struct LaneSlot<'a, T: Copy, const SIZE: usize> {
        mem: &'a mut SharedMem<T, SIZE>,
        owned_index: usize,
    }

    impl<'a, T: Copy, const SIZE: usize> LaneSlot<'a, T, SIZE> {
        pub fn read(&self) -> T {
            self.mem.data[self.owned_index]
        }

        pub fn write(&mut self, value: T) {
            self.mem.data[self.owned_index] = value;
        }

        pub fn index(&self) -> usize {
            self.owned_index
        }
    }

    /// Grant each lane exclusive access to its own slot
    ///
    /// Lane i gets exclusive access to index i.
    /// This is safe because indices are disjoint.
    pub fn split_by_lane<'a, T: Copy, const SIZE: usize>(
        _warp: &Warp<All>,
        mem: &'a mut SharedMem<T, SIZE>,
        lane: usize,
    ) -> LaneSlot<'a, T, SIZE> {
        assert!(lane < SIZE, "lane must be within memory size");
        LaneSlot {
            mem,
            owned_index: lane,
        }
    }

    /// Per-lane view of disjoint regions (immutable version)
    pub struct DisjointView<'a, T: Copy, const SIZE: usize> {
        mem: &'a SharedMem<T, SIZE>,
        owned_indices: Vec<usize>,
    }

    impl<'a, T: Copy, const SIZE: usize> DisjointView<'a, T, SIZE> {
        pub fn read(&self, local_index: usize) -> Option<T> {
            self.owned_indices
                .get(local_index)
                .map(|&i| self.mem.data[i])
        }
    }

    /// Split memory into disjoint views based on predicate
    pub fn split_by_predicate<'a, T: Copy, const SIZE: usize, F>(
        _warp: &Warp<All>,
        mem: &'a SharedMem<T, SIZE>,
        pred: F,
    ) -> (DisjointView<'a, T, SIZE>, DisjointView<'a, T, SIZE>)
    where
        F: Fn(usize) -> bool,
    {
        let mut true_indices = Vec::new();
        let mut false_indices = Vec::new();

        for i in 0..SIZE {
            if pred(i) {
                true_indices.push(i);
            } else {
                false_indices.push(i);
            }
        }

        (
            DisjointView {
                mem,
                owned_indices: true_indices,
            },
            DisjointView {
                mem,
                owned_indices: false_indices,
            },
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_lane_exclusive_access() {
            let warp: Warp<All> = Warp::new();
            let mut mem: SharedMem<i32, 32> = SharedMem::new();

            // Each lane writes to its own slot
            for lane in 0..32 {
                let mut slot = split_by_lane(&warp, &mut mem, lane);
                slot.write(lane as i32 * 10);
            }

            // Verify each lane's value
            for lane in 0..32 {
                let slot = split_by_lane(&warp, &mut mem, lane);
                assert_eq!(slot.read(), lane as i32 * 10);
            }
        }

        #[test]
        fn test_disjoint_views() {
            let warp: Warp<All> = Warp::new();
            let mem: SharedMem<i32, 8> = SharedMem::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);

            let (evens, odds) = split_by_predicate(&warp, &mem, |i| i % 2 == 0);

            // Even indices: 0, 2, 4, 6
            assert_eq!(evens.read(0), Some(0));
            assert_eq!(evens.read(1), Some(2));
            assert_eq!(evens.read(2), Some(4));
            assert_eq!(evens.read(3), Some(6));

            // Odd indices: 1, 3, 5, 7
            assert_eq!(odds.read(0), Some(1));
            assert_eq!(odds.read(1), Some(3));
        }
    }
}

// ============================================================================
// PATTERN 3: SCOPED BORROWS (RAII-style)
// ============================================================================

/// Scoped borrows: Borrow with automatic sync at scope end
///
/// In GPU programming, sync barriers are critical but easy to forget.
/// This pattern ties sync to scope exit.
pub mod scoped {
    use super::*;

    /// A scoped mutable borrow that syncs on drop
    pub struct ScopedMut<'a, T: Copy, const SIZE: usize> {
        mem: &'a mut SharedMem<T, SIZE>,
        needs_sync: bool,
    }

    impl<'a, T: Copy, const SIZE: usize> ScopedMut<'a, T, SIZE> {
        pub fn read(&self, index: usize) -> T {
            self.mem.data[index]
        }

        pub fn write(&mut self, index: usize, value: T) {
            self.mem.data[index] = value;
            self.needs_sync = true;
        }
    }

    impl<'a, T: Copy, const SIZE: usize> Drop for ScopedMut<'a, T, SIZE> {
        fn drop(&mut self) {
            if self.needs_sync {
                // In real GPU code: __syncthreads() or __threadfence_block()
                // Here we just mark that sync happened
                // println!("Sync barrier at scope exit");
            }
        }
    }

    /// Create a scoped mutable borrow
    pub fn scoped_borrow<'a, T: Copy, const SIZE: usize>(
        _warp: &Warp<All>,
        mem: &'a mut SharedMem<T, SIZE>,
    ) -> ScopedMut<'a, T, SIZE> {
        ScopedMut {
            mem,
            needs_sync: false,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_scoped_sync() {
            let warp: Warp<All> = Warp::new();
            let mut mem: SharedMem<i32, 32> = SharedMem::new();

            // Sync happens at end of scope
            {
                let mut borrow = scoped_borrow(&warp, &mut mem);
                borrow.write(0, 100);
                // Drop at end of scope triggers sync
            }

            assert_eq!(mem.data[0], 100);
        }
    }
}

// ============================================================================
// PATTERN 4: LEASE/RETURN
// ============================================================================

/// Lease/Return: Explicit checkout and return of memory regions
///
/// This is a more explicit form of borrowing where the return is
/// a separate operation (not tied to scope).
pub mod lease {

    /// A leased memory region (must be returned)
    pub struct Lease<T: Copy, const SIZE: usize> {
        data: [T; SIZE],
        source_id: usize, // Identifies which pool this came from
    }

    impl<T: Copy, const SIZE: usize> Lease<T, SIZE> {
        pub fn read(&self, index: usize) -> T {
            self.data[index]
        }

        pub fn write(&mut self, index: usize, value: T) {
            self.data[index] = value;
        }

        pub fn source_id(&self) -> usize {
            self.source_id
        }
    }

    /// A pool that manages leases
    pub struct LeasePool<T: Copy + Default, const SIZE: usize, const SLOTS: usize> {
        slots: [[T; SIZE]; SLOTS],
        leased: [bool; SLOTS],
    }

    impl<T: Copy + Default, const SIZE: usize, const SLOTS: usize> LeasePool<T, SIZE, SLOTS> {
        pub fn new() -> Self {
            LeasePool {
                slots: [[T::default(); SIZE]; SLOTS],
                leased: [false; SLOTS],
            }
        }

        /// Checkout a slot (returns None if all leased)
        pub fn checkout(&mut self) -> Option<Lease<T, SIZE>> {
            for (i, is_leased) in self.leased.iter_mut().enumerate() {
                if !*is_leased {
                    *is_leased = true;
                    return Some(Lease {
                        data: self.slots[i],
                        source_id: i,
                    });
                }
            }
            None
        }

        /// Return a lease
        pub fn return_lease(&mut self, lease: Lease<T, SIZE>) {
            let id = lease.source_id;
            assert!(self.leased[id], "Returning non-leased slot");
            self.slots[id] = lease.data;
            self.leased[id] = false;
        }

        /// Count available slots
        pub fn available(&self) -> usize {
            self.leased.iter().filter(|&&l| !l).count()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_lease_pool() {
            let mut pool: LeasePool<i32, 32, 4> = LeasePool::new();

            assert_eq!(pool.available(), 4);

            // Checkout a lease
            let mut lease = pool.checkout().unwrap();
            assert_eq!(pool.available(), 3);

            // Modify the lease
            lease.write(0, 42);

            // Return the lease
            pool.return_lease(lease);
            assert_eq!(pool.available(), 4);
        }

        #[test]
        fn test_lease_exhaustion() {
            let mut pool: LeasePool<i32, 8, 2> = LeasePool::new();

            let _l1 = pool.checkout().unwrap();
            let _l2 = pool.checkout().unwrap();

            // Pool exhausted
            assert!(pool.checkout().is_none());
        }
    }
}

// ============================================================================
// KEY FINDINGS
// ============================================================================

/// Summary: How to handle temporary shared access (borrowing)?
///
/// ## Patterns
///
/// | Pattern | Use Case | GPU Analog |
/// |---------|----------|------------|
/// | Lane-Parallel | All lanes access same index | Uniform load/store |
/// | Split | Each lane accesses own index | Coalesced access |
/// | Scoped | Sync needed at end | __syncthreads() |
/// | Lease/Return | Explicit lifetime management | Memory pools |
///
/// ## Key Insights
///
/// 1. **SIMT changes borrowing semantics**: In SIMT, all lanes do the same
///    operation. "Mutable borrow" means all lanes write - which is safe
///    if they write to DIFFERENT indices or the SAME value.
///
/// 2. **Disjoint access is key**: If we can prove indices are disjoint,
///    parallel mutable access is safe. This is like Rust's split_at_mut.
///
/// 3. **Sync ties to scope**: GPU sync barriers are critical. Tying them
///    to scope exit (RAII) prevents forgetting them.
///
/// 4. **Pools for dynamic allocation**: GPU shared memory is fixed size.
///    Lease pools provide dynamic-ish allocation with explicit return.
///
/// ## Comparison with CPU Rust
///
/// | CPU Rust | GPU Warp |
/// |----------|----------|
/// | Single owner | All lanes "own" together |
/// | &T / &mut T | SharedRef / SharedMut (per-warp) |
/// | split_at_mut | split_by_lane (per-index) |
/// | Drop | Drop + __syncthreads() |
///
/// The key difference: GPU "borrow" is always 32-way parallel. The type
/// system must track WHICH INDICES each lane can access, not just
/// read/write permissions.
pub const _FINDINGS: () = ();

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_borrow_cycle() {
        let warp: Warp<All> = Warp::new();
        let mut mem: SharedMem<i32, 32> = SharedMem::new();

        // Phase 1: Lane-parallel write (all lanes write same index)
        {
            let mut borrow = lane_parallel::borrow_mut(&warp, &mut mem);
            borrow.write(0, 100); // All lanes write index 0
        }

        // Phase 2: Split access (each lane writes own index)
        for lane in 0..32 {
            let mut slot = split::split_by_lane(&warp, &mut mem, lane);
            slot.write(lane as i32);
        }

        // Verify
        for lane in 0..32 {
            let slot = split::split_by_lane(&warp, &mut mem, lane);
            if lane == 0 {
                // Index 0 was overwritten in phase 2
                assert_eq!(slot.read(), 0);
            } else {
                assert_eq!(slot.read(), lane as i32);
            }
        }
    }
}
