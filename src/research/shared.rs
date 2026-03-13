//! Shared memory with ownership semantics
//!
//! GPU shared memory is a common source of race conditions.
//! This module explores how ownership/borrowing can prevent them.

use std::marker::PhantomData;
use crate::{GpuValue, data::Role};

/// A region of shared memory owned by a specific role
///
/// The key insight: shared memory races happen because ownership is implicit.
/// By making ownership explicit in the type system, we can prevent races.
///
/// # Ownership Model
///
/// - Each SharedRegion is owned by exactly one Role
/// - The owning role has read/write access
/// - Other roles can be granted read access via `grant_read()`
/// - Write access is never shared (single-writer principle)
///
/// # Example
///
/// ```ignore
/// // Coordinator owns the task queue
/// let mut queue: SharedRegion<Task, COORDINATOR> = SharedRegion::new();
///
/// // Coordinator writes tasks
/// queue.write(lane, task);  // Only coordinator can call this
///
/// // Grant read access to workers
/// let worker_view: SharedView<Task, WORKER> = queue.grant_read();
///
/// // Workers can read but not write
/// let task = worker_view.read(lane);
/// // worker_view.write(lane, task);  // ERROR: SharedView is read-only
/// ```
pub struct SharedRegion<T: GpuValue, const OWNER: u8> {
    /// The actual storage (in real impl, this would be __shared__)
    data: [T; 32],
    /// Which role owns this region
    owner: Role,
    _phantom: PhantomData<()>,
}

impl<T: GpuValue + Default, const OWNER: u8> SharedRegion<T, OWNER> {
    /// Create a new shared region owned by the specified role
    pub fn new(owner: Role) -> Self {
        SharedRegion {
            data: [T::default(); 32],
            owner,
            _phantom: PhantomData,
        }
    }
}

impl<T: GpuValue, const OWNER: u8> SharedRegion<T, OWNER> {
    /// Write to a slot. Only the owning role can call this.
    ///
    /// In a real implementation, this would be enforced by requiring
    /// proof that the caller is in the owning role.
    pub fn write(&mut self, index: usize, value: T) {
        assert!(index < 32, "Index out of bounds");
        self.data[index] = value;
    }

    /// Read from a slot. Owner always has read access.
    pub fn read(&self, index: usize) -> T {
        assert!(index < 32, "Index out of bounds");
        self.data[index]
    }

    /// Grant read-only access to another role
    ///
    /// This returns a SharedView that can only read, not write.
    /// The lifetime is tied to the SharedRegion borrow.
    pub fn grant_read(&self) -> SharedView<'_, T, OWNER> {
        SharedView {
            region: self,
            _phantom: PhantomData,
        }
    }

    /// Get the owning role
    pub fn owner(&self) -> Role {
        self.owner
    }
}

/// A read-only view of a shared region
///
/// This is what non-owning roles receive when granted access.
/// They can read but not write.
pub struct SharedView<'a, T: GpuValue, const OWNER: u8> {
    region: &'a SharedRegion<T, OWNER>,
    _phantom: PhantomData<()>,
}

impl<'a, T: GpuValue, const OWNER: u8> SharedView<'a, T, OWNER> {
    /// Read from a slot
    pub fn read(&self, index: usize) -> T {
        self.region.read(index)
    }

    // Note: no write() method - this is read-only!
}

/// A work queue in shared memory
///
/// This is a common pattern in persistent thread programs.
/// The type system enforces the producer-consumer relationship.
pub struct WorkQueue<T: GpuValue, const PRODUCER: u8, const CONSUMER: u8> {
    /// The task storage
    tasks: SharedRegion<T, PRODUCER>,
    /// Head index (producer writes)
    head: usize,
    /// Tail index (consumer reads, producer reads for "full" check)
    tail: usize,
    _phantom: PhantomData<()>,
}

impl<T: GpuValue + Default, const PRODUCER: u8, const CONSUMER: u8> WorkQueue<T, PRODUCER, CONSUMER> {
    /// Create a new work queue
    pub fn new(producer_role: Role, _consumer_role: Role) -> Self {
        WorkQueue {
            tasks: SharedRegion::new(producer_role),
            head: 0,
            tail: 0,
            _phantom: PhantomData,
        }
    }

    /// Push a task (only producer can call)
    pub fn push(&mut self, task: T) -> Result<(), QueueFull> {
        let next = (self.head + 1) % 32;
        if next == self.tail {
            return Err(QueueFull);
        }
        self.tasks.write(self.head, task);
        self.head = next;
        Ok(())
    }

    /// Pop a task (only consumer can call)
    ///
    /// In a real implementation, this would require proof of consumer role.
    pub fn pop(&mut self) -> Option<T> {
        if self.tail == self.head {
            return None;
        }
        let task = self.tasks.read(self.tail);
        self.tail = (self.tail + 1) % 32;
        Some(task)
    }

    /// Check if queue is empty (both roles can call)
    pub fn is_empty(&self) -> bool {
        self.tail == self.head
    }

    /// Check if queue is full (both roles can call)
    pub fn is_full(&self) -> bool {
        (self.head + 1) % 32 == self.tail
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QueueFull;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Role;

    // Role IDs (in a real impl, these would be const generics)
    const COORDINATOR: u8 = 0;
    const WORKER: u8 = 1;

    #[test]
    fn test_shared_region_ownership() {
        let coordinator = Role::lanes(0, 4, "coordinator");
        let mut region: SharedRegion<i32, COORDINATOR> = SharedRegion::new(coordinator);

        // Owner can write and read
        region.write(0, 42);
        assert_eq!(region.read(0), 42);

        // Grant read access to others
        let view = region.grant_read();
        assert_eq!(view.read(0), 42);
        // view.write(0, 100);  // This would be a compile error!
    }

    #[test]
    fn test_work_queue() {
        let coordinator = Role::lanes(0, 4, "coordinator");
        let worker = Role::lanes(4, 32, "worker");

        let mut queue: WorkQueue<i32, COORDINATOR, WORKER> =
            WorkQueue::new(coordinator, worker);

        assert!(queue.is_empty());

        // Producer pushes
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        queue.push(3).unwrap();

        assert!(!queue.is_empty());

        // Consumer pops
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), None);

        assert!(queue.is_empty());
    }
}
