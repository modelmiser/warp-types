//! GPU data types: uniform vs per-lane value distinction.
//!
//! The fundamental insight: GPU values are either uniform across all lanes
//! or vary per-lane. Making this distinction in the type system prevents
//! a large class of bugs (reading reduction results from wrong lanes,
//! passing divergent data where uniform is expected, etc.).

use core::marker::PhantomData;
use crate::GpuValue;

/// A lane identifier (0..31 for NVIDIA, 0..63 for AMD).
///
/// Type-safe: you can't accidentally use an arbitrary int as a lane id.
/// Supports up to 64 lanes to accommodate AMD wavefronts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaneId(u8);

impl LaneId {
    pub const fn new(id: u8) -> Self {
        assert!(id < 64, "Lane ID must be < 64 (supports NVIDIA 32-lane and AMD 64-lane)");
        LaneId(id)
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A warp identifier within a thread block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WarpId(u16);

impl WarpId {
    pub const fn new(id: u16) -> Self {
        WarpId(id)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A value guaranteed to be the same across all lanes in a warp.
///
/// You can only create `Uniform` values through operations that guarantee
/// uniformity (broadcasts, constants, ballot results). This prevents the
/// common bug of assuming a value is uniform when it isn't.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform<T: GpuValue> {
    value: T,
}

impl<T: GpuValue> Uniform<T> {
    /// Create a uniform value from a compile-time constant.
    pub const fn from_const(value: T) -> Self {
        Uniform { value }
    }

    /// Get the value. Safe because it's the same in all lanes.
    pub fn get(self) -> T {
        self.value
    }

    /// Broadcast: convert uniform to per-lane (identity, but changes type).
    pub fn broadcast(self) -> PerLane<T> {
        PerLane { value: self.value }
    }
}

/// A value that MAY DIFFER across lanes in a warp.
///
/// This is the default for most GPU computations. Each lane has its own
/// value, and you can only access other lanes' values through explicit
/// shuffle operations.
#[must_use = "PerLane values carry per-lane GPU data — dropping discards computation"]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerLane<T: GpuValue> {
    value: T,
}

impl<T: GpuValue> PerLane<T> {
    pub fn new(value: T) -> Self {
        PerLane { value }
    }

    pub fn get(self) -> T {
        self.value
    }

    /// Assert this value is actually uniform.
    ///
    /// # Safety
    /// Caller must ensure all lanes hold the same value.
    pub unsafe fn assume_uniform(self) -> Uniform<T> {
        Uniform { value: self.value }
    }
}

/// A value that exists ONLY in a specific lane.
///
/// Models the result of a reduction — only one lane has the answer.
/// Prevents the common bug of reading a reduction result from all lanes
/// (undefined behavior in CUDA).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleLane<T: GpuValue, const LANE: u8> {
    value: T,
    _phantom: PhantomData<()>,
}

impl<T: GpuValue, const LANE: u8> SingleLane<T, LANE> {
    pub fn new(value: T) -> Self {
        SingleLane { value, _phantom: PhantomData }
    }

    /// Read the value. Only valid in the owning lane.
    pub fn get(self) -> T {
        self.value
    }

    /// Broadcast to all lanes — the ONLY safe way to share with other lanes.
    pub fn broadcast(self) -> Uniform<T> {
        Uniform::from_const(self.value)
    }
}

/// A role within a warp (e.g., coordinator vs worker lanes).
///
/// Roles enable modeling warp-level protocols where different lanes
/// have different responsibilities. Uses `u64` mask to match
/// `ActiveSet::MASK` width (supporting AMD 64-lane wavefronts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Role {
    pub mask: u64,
    pub name: &'static str,
}

impl Role {
    pub const fn lanes(start: u8, end: u8, name: &'static str) -> Self {
        assert!(start < 64 && end <= 64 && start < end);
        let mask = ((1u64 << (end - start)) - 1) << start;
        Role { mask, name }
    }

    pub const fn lane(id: u8, name: &'static str) -> Self {
        assert!(id < 64);
        Role { mask: 1u64 << id, name }
    }

    pub const fn contains(self, lane: LaneId) -> bool {
        (self.mask & (1u64 << lane.0)) != 0
    }

    pub const fn count(self) -> u32 {
        self.mask.count_ones()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lane_id() {
        let lane = LaneId::new(15);
        assert_eq!(lane.get(), 15);
        assert_eq!(lane.index(), 15);
    }

    #[test]
    fn test_lane_id_boundary_31() {
        let lane = LaneId::new(31);
        assert_eq!(lane.get(), 31);
        assert_eq!(lane.index(), 31);
    }

    #[test]
    fn test_lane_id_boundary_63() {
        let lane = LaneId::new(63);
        assert_eq!(lane.get(), 63);
    }

    #[test]
    #[should_panic]
    fn test_lane_id_out_of_range() {
        LaneId::new(64);
    }

    #[test]
    fn test_uniform_broadcast() {
        let u: Uniform<i32> = Uniform::from_const(42);
        let p: PerLane<i32> = u.broadcast();
        assert_eq!(p.get(), 42);
    }

    #[test]
    fn test_single_lane_broadcast() {
        let reduced: SingleLane<i32, 0> = SingleLane::new(42);
        let uniform: Uniform<i32> = reduced.broadcast();
        assert_eq!(uniform.get(), 42);
    }

    #[test]
    fn test_role_coverage() {
        let coordinator = Role::lanes(0, 4, "coordinator");
        let worker = Role::lanes(4, 32, "worker");

        assert!(coordinator.contains(LaneId::new(0)));
        assert!(coordinator.contains(LaneId::new(3)));
        assert!(!coordinator.contains(LaneId::new(4)));

        assert!(!worker.contains(LaneId::new(3)));
        assert!(worker.contains(LaneId::new(4)));
        assert!(worker.contains(LaneId::new(31)));

        assert_eq!(coordinator.count(), 4);
        assert_eq!(worker.count(), 28);
    }
}
