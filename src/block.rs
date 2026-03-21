//! Block-level types: shared memory ownership and inter-block sessions.
//!
//! GPU parallelism has three levels:
//! - **Warp** (32/64 lanes): shuffles, lockstep, linear typestate
//! - **Block** (multiple warps): shared memory, `__syncthreads()`
//! - **Grid** (multiple blocks): global memory, cooperative groups
//!
//! This module provides typed abstractions for the block and grid levels.

use crate::data::Role;
use crate::GpuValue;
use core::marker::PhantomData;

// ============================================================================
// Shared memory with ownership semantics
// ============================================================================

/// A region of shared memory owned by a specific role.
///
/// The key insight: shared memory races happen because ownership is implicit.
/// By making ownership explicit, we prevent races at the type level.
///
/// `OWNER` is a type-level tag (u8 discriminator) that prevents cross-type access
/// at compile time. The `owner` field carries the runtime lane range metadata
/// (which lanes belong to this role). These encode different concerns: OWNER
/// prevents mixing regions at the type level; Role describes the lane geometry.
pub struct SharedRegion<T: GpuValue, const OWNER: u8> {
    data: [T; 32],
    owner: Role,
    _phantom: PhantomData<()>,
}

impl<T: GpuValue + Default, const OWNER: u8> SharedRegion<T, OWNER> {
    pub fn new(owner: Role) -> Self {
        SharedRegion {
            data: [T::default(); 32],
            owner,
            _phantom: PhantomData,
        }
    }
}

impl<T: GpuValue, const OWNER: u8> SharedRegion<T, OWNER> {
    pub fn write(&mut self, index: usize, value: T) {
        assert!(index < 32, "Index out of bounds");
        self.data[index] = value;
    }

    pub fn read(&self, index: usize) -> T {
        assert!(index < 32, "Index out of bounds");
        self.data[index]
    }

    pub fn grant_read(&self) -> SharedView<'_, T, OWNER> {
        SharedView {
            region: self,
            _phantom: PhantomData,
        }
    }

    pub fn owner(&self) -> Role {
        self.owner
    }
}

/// A read-only view of a shared region (for non-owning roles).
pub struct SharedView<'a, T: GpuValue, const OWNER: u8> {
    region: &'a SharedRegion<T, OWNER>,
    _phantom: PhantomData<()>,
}

impl<'a, T: GpuValue, const OWNER: u8> SharedView<'a, T, OWNER> {
    pub fn read(&self, index: usize) -> T {
        self.region.read(index)
    }
}

/// A work queue in shared memory with typed producer/consumer roles.
///
/// Uses a circular buffer with 32 slots and one sentinel for full detection,
/// giving an effective capacity of 31 items.
pub struct WorkQueue<T: GpuValue, const PRODUCER: u8, const CONSUMER: u8> {
    tasks: SharedRegion<T, PRODUCER>,
    head: usize,
    tail: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct QueueFull;

impl<T: GpuValue + Default, const PRODUCER: u8, const CONSUMER: u8>
    WorkQueue<T, PRODUCER, CONSUMER>
{
    pub fn new(producer_role: Role, _consumer_role: Role) -> Self {
        WorkQueue {
            tasks: SharedRegion::new(producer_role),
            head: 0,
            tail: 0,
        }
    }

    pub fn push(&mut self, task: T) -> Result<(), QueueFull> {
        let next = (self.head + 1) % 32;
        if next == self.tail {
            return Err(QueueFull);
        }
        self.tasks.write(self.head, task);
        self.head = next;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.tail == self.head {
            return None;
        }
        let task = self.tasks.read(self.tail);
        self.tail = (self.tail + 1) % 32;
        Some(task)
    }

    pub fn is_empty(&self) -> bool {
        self.tail == self.head
    }
    pub fn is_full(&self) -> bool {
        (self.head + 1) % 32 == self.tail
    }
}

// ============================================================================
// GPU hierarchy types
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(u32);

impl BlockId {
    pub const fn new(id: u32) -> Self {
        BlockId(id)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ThreadId {
    block: BlockId,
    warp: crate::data::WarpId,
    lane: crate::data::LaneId,
}

impl ThreadId {
    pub const fn new(block: BlockId, warp: crate::data::WarpId, lane: crate::data::LaneId) -> Self {
        ThreadId { block, warp, lane }
    }

    pub const fn block(self) -> BlockId {
        self.block
    }

    pub const fn warp(self) -> crate::data::WarpId {
        self.warp
    }

    pub const fn lane(self) -> crate::data::LaneId {
        self.lane
    }
}

// ============================================================================
// Inter-block protocol types
// ============================================================================

pub trait BlockRole {
    const NAME: &'static str;
}

pub struct Leader;
impl BlockRole for Leader {
    const NAME: &'static str = "Leader";
}

pub struct Worker;
impl BlockRole for Worker {
    const NAME: &'static str = "Worker";
}

pub trait ProtocolState {}

pub struct Initial;
impl ProtocolState for Initial {}

pub struct WorkDistributed;
impl ProtocolState for WorkDistributed {}

pub struct WorkComplete;
impl ProtocolState for WorkComplete {}

/// A session between blocks, parameterized by role, state, and block count.
pub struct BlockSession<R: BlockRole, S: ProtocolState, const N: usize> {
    block_id: BlockId,
    _role: PhantomData<R>,
    _state: PhantomData<S>,
}

impl<R: BlockRole, S: ProtocolState, const N: usize> BlockSession<R, S, N> {
    #[allow(dead_code)] // Constructor for future block-level API usage
    pub(crate) fn new(block_id: BlockId) -> Self {
        BlockSession {
            block_id,
            _role: PhantomData,
            _state: PhantomData,
        }
    }

    pub fn block_id(&self) -> BlockId {
        self.block_id
    }
}

// ============================================================================
// Hierarchical reduction (type-state machine)
// ============================================================================

pub struct WarpPhase;
pub struct BlockPhase;
pub struct GridPhase;
pub struct Complete;

pub struct ReductionSession<Phase> {
    value: u32,
    _phase: PhantomData<Phase>,
}

impl ReductionSession<WarpPhase> {
    #[allow(dead_code)] // Constructor for future reduction pipeline usage
    pub(crate) fn new(value: u32) -> Self {
        ReductionSession {
            value,
            _phase: PhantomData,
        }
    }

    pub fn warp_reduce(self) -> (u32, ReductionSession<BlockPhase>) {
        (
            self.value,
            ReductionSession {
                value: self.value,
                _phase: PhantomData,
            },
        )
    }
}

impl ReductionSession<BlockPhase> {
    pub fn block_reduce(self) -> (u32, ReductionSession<GridPhase>) {
        (
            self.value,
            ReductionSession {
                value: self.value,
                _phase: PhantomData,
            },
        )
    }
}

impl ReductionSession<GridPhase> {
    pub fn grid_reduce(self) -> (u32, ReductionSession<Complete>) {
        (
            self.value,
            ReductionSession {
                value: self.value,
                _phase: PhantomData,
            },
        )
    }
}

impl ReductionSession<Complete> {
    pub fn result(self) -> u32 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COORDINATOR: u8 = 0;
    const WORKER_ROLE: u8 = 1;

    #[test]
    fn test_shared_region_ownership() {
        let coordinator = Role::lanes(0, 4, "coordinator");
        let mut region: SharedRegion<i32, COORDINATOR> = SharedRegion::new(coordinator);
        region.write(0, 42);
        assert_eq!(region.read(0), 42);
        let view = region.grant_read();
        assert_eq!(view.read(0), 42);
    }

    #[test]
    fn test_work_queue() {
        let coordinator = Role::lanes(0, 4, "coordinator");
        let worker = Role::lanes(4, 32, "worker");
        let mut queue: WorkQueue<i32, COORDINATOR, WORKER_ROLE> =
            WorkQueue::new(coordinator, worker);

        assert!(queue.is_empty());
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        queue.push(3).unwrap();
        assert!(!queue.is_empty());
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_work_queue_full() {
        let coordinator = Role::lanes(0, 4, "coordinator");
        let worker = Role::lanes(4, 32, "worker");
        let mut queue: WorkQueue<i32, COORDINATOR, WORKER_ROLE> =
            WorkQueue::new(coordinator, worker);

        // Ring buffer of size 32 has capacity 31 (one slot reserved for full detection)
        for i in 0..31 {
            assert!(queue.push(i).is_ok());
        }
        assert!(queue.is_full());
        assert!(queue.push(31).is_err());
    }

    #[test]
    fn test_hierarchical_reduction() {
        let session = ReductionSession::<WarpPhase>::new(42);
        let (warp_result, session) = session.warp_reduce();
        assert_eq!(warp_result, 42);
        let (block_result, session) = session.block_reduce();
        assert_eq!(block_result, 42);
        let (grid_result, session) = session.grid_reduce();
        assert_eq!(grid_result, 42);
        assert_eq!(session.result(), 42);
    }

    #[test]
    fn test_block_session() {
        let leader: BlockSession<Leader, Initial, 4> = BlockSession::new(BlockId::new(0));
        assert_eq!(leader.block_id().0, 0);
        let worker: BlockSession<Worker, Initial, 4> = BlockSession::new(BlockId::new(1));
        assert_eq!(worker.block_id().0, 1);
    }
}
