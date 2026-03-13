//! Lane and warp topology types
//!
//! The fundamental insight: GPU values are either uniform across all lanes
//! or vary per-lane. Making this distinction in the type system prevents
//! a large class of bugs.

use std::marker::PhantomData;
use crate::GpuValue;

/// A lane identifier (0..31 for NVIDIA, 0..63 for AMD)
///
/// Type-safe: you can't accidentally use an arbitrary int as a lane id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneId(u8);

impl LaneId {
    /// Create a lane id. Panics if out of range.
    pub const fn new(id: u8) -> Self {
        assert!(id < 32, "Lane ID must be < 32");
        LaneId(id)
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    /// The lane's position within its warp
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A warp identifier within a thread block
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarpId(u16);

impl WarpId {
    pub const fn new(id: u16) -> Self {
        WarpId(id)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A value that is GUARANTEED to be the same across all lanes in a warp.
///
/// The compiler enforces this: you can only create Uniform values through
/// operations that guarantee uniformity (broadcasts, reductions to all lanes).
///
/// # Why this matters
///
/// In CUDA, you write `int x = ...` and hope it's uniform. If it's not,
/// you get divergence bugs. With `Uniform<T>`, the type system tracks it.
#[derive(Clone, Copy, Debug)]
pub struct Uniform<T: GpuValue> {
    value: T,
}

impl<T: GpuValue> Uniform<T> {
    /// Create a uniform value from a compile-time constant.
    /// This is always safe - constants are inherently uniform.
    pub const fn from_const(value: T) -> Self {
        Uniform { value }
    }

    /// Get the value. Safe because it's the same in all lanes.
    pub fn get(self) -> T {
        self.value
    }

    /// Broadcast: convert uniform to per-lane (identity, but changes type)
    pub fn broadcast(self) -> PerLane<T> {
        PerLane { value: self.value }
    }
}

/// A value that MAY DIFFER across lanes in a warp.
///
/// This is the default for most GPU computations. Each lane has its own
/// value, and you can only access other lanes' values through explicit
/// shuffle operations.
///
/// # Why this matters
///
/// `PerLane<T>` makes it clear when you're working with divergent data.
/// Shuffle operations return `PerLane<T>` because even though you're
/// reading from another lane, the result varies based on your lane id.
#[derive(Clone, Copy, Debug)]
pub struct PerLane<T: GpuValue> {
    value: T,
}

impl<T: GpuValue> PerLane<T> {
    /// Create a per-lane value. Each lane may have a different value.
    pub fn new(value: T) -> Self {
        PerLane { value }
    }

    /// Get this lane's value.
    pub fn get(self) -> T {
        self.value
    }

    /// UNSAFE: Assert this value is actually uniform.
    ///
    /// Use only when you KNOW all lanes have the same value
    /// (e.g., after a broadcast or uniform initialization).
    ///
    /// # Safety
    /// Caller must ensure all lanes hold the same value.
    pub unsafe fn assume_uniform(self) -> Uniform<T> {
        Uniform { value: self.value }
    }
}

/// A role within a warp (e.g., coordinator vs worker lanes)
///
/// Roles enable modeling warp-level protocols where different lanes
/// have different responsibilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Role {
    /// Which lanes belong to this role (bitmask)
    pub mask: u32,
    /// Human-readable name
    pub name: &'static str,
}

impl Role {
    /// Create a role from a lane range
    pub const fn lanes(start: u8, end: u8, name: &'static str) -> Self {
        assert!(start < 32 && end <= 32 && start < end);
        let mask = ((1u32 << (end - start)) - 1) << start;
        Role { mask, name }
    }

    /// Create a role from a single lane
    pub const fn lane(id: u8, name: &'static str) -> Self {
        assert!(id < 32);
        Role { mask: 1u32 << id, name }
    }

    /// Check if a lane belongs to this role
    pub const fn contains(self, lane: LaneId) -> bool {
        (self.mask & (1u32 << lane.0)) != 0
    }

    /// Number of lanes in this role
    pub const fn count(self) -> u32 {
        self.mask.count_ones()
    }
}

/// A warp with assigned roles
///
/// This is the foundation for warp-level session types. Each warp
/// can have lanes assigned to different roles, and the type system
/// can verify that communication between roles is well-formed.
pub struct Warp<const N: usize> {
    roles: [Role; N],
    _phantom: PhantomData<()>,
}

impl<const N: usize> Warp<N> {
    /// Create a warp with the given roles.
    ///
    /// # Panics
    /// Panics if roles overlap or don't cover all 32 lanes.
    pub const fn new(roles: [Role; N]) -> Self {
        // Verify roles cover all lanes exactly once
        let mut coverage = 0u32;
        let mut i = 0;
        while i < N {
            // Check no overlap
            assert!(
                coverage & roles[i].mask == 0,
                "Roles must not overlap"
            );
            coverage |= roles[i].mask;
            i += 1;
        }
        // All 32 lanes must be covered
        assert!(coverage == 0xFFFFFFFF, "Roles must cover all 32 lanes");

        Warp {
            roles,
            _phantom: PhantomData,
        }
    }

    /// Get the roles
    pub const fn roles(&self) -> &[Role; N] {
        &self.roles
    }
}

// Example: A coordinator-worker warp configuration
pub const COORDINATOR_WORKER: Warp<2> = Warp::new([
    Role::lanes(0, 4, "coordinator"),   // Lanes 0-3
    Role::lanes(4, 32, "worker"),       // Lanes 4-31
]);

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
    #[should_panic]
    fn test_lane_id_out_of_range() {
        LaneId::new(32);
    }

    #[test]
    fn test_uniform_broadcast() {
        let u: Uniform<i32> = Uniform::from_const(42);
        let p: PerLane<i32> = u.broadcast();
        assert_eq!(p.get(), 42);
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

    #[test]
    fn test_warp_roles() {
        // This compiles: roles cover all 32 lanes exactly once
        let _warp = Warp::new([
            Role::lanes(0, 16, "left"),
            Role::lanes(16, 32, "right"),
        ]);
    }
}
