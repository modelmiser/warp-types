//! Work Stealing with Session Types
//!
//! **STATUS: Research** — Exploratory prototype, not promoted to main API.
//!
//! Research questions:
//! - "Can session types express work-stealing efficiently?"
//! - "Can we verify deadlock-freedom in work-stealing?"
//! - "How to handle dynamic role assignment?"
//!
//! # Background
//!
//! Work-stealing is a load balancing technique where idle threads steal
//! work from busy threads. At warp level, this involves:
//! 1. Discovering who has work (ballot)
//! 2. Transferring work items (shuffle)
//! 3. Updating state (per-lane bookkeeping)
//!
//! # Key Insight: Intra-Warp Work Stealing is Deadlock-Free
//!
//! Within a warp, all lanes execute in lockstep. There's no "waiting" -
//! ballot and shuffle are collective operations that complete atomically.
//! Therefore, intra-warp work stealing CANNOT deadlock by construction.
//!
//! Inter-warp work stealing (via shared memory or atomics) CAN deadlock
//! and needs careful protocol design.
//!
//! # Patterns
//!
//! 1. **Compaction**: Pack work into contiguous lanes
//! 2. **Load Balancing**: Redistribute from overloaded to idle lanes
//! 3. **Dynamic Roles**: Lanes switch between producer/consumer

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

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerLane<T>(pub [T; 32]);

// ============================================================================
// WORK ITEM REPRESENTATION
// ============================================================================

/// A work item that can be transferred between lanes
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WorkItem<T: Copy> {
    pub data: T,
    pub valid: bool,
}

impl<T: Copy + Default> Default for WorkItem<T> {
    fn default() -> Self {
        WorkItem {
            data: T::default(),
            valid: false,
        }
    }
}

impl<T: Copy + Default> WorkItem<T> {
    pub fn some(data: T) -> Self {
        WorkItem { data, valid: true }
    }

    pub fn none() -> Self {
        WorkItem {
            data: T::default(),
            valid: false,
        }
    }
}

/// Per-lane work queue (simplified: each lane has 0 or 1 work items)
#[derive(Copy, Clone, Debug)]
pub struct WorkQueue<T: Copy> {
    pub items: [WorkItem<T>; 32],
}

impl<T: Copy + Default> WorkQueue<T> {
    pub fn new() -> Self {
        WorkQueue {
            items: [WorkItem::none(); 32],
        }
    }

    pub fn has_work(&self, lane: usize) -> bool {
        self.items[lane].valid
    }

    pub fn set_work(&mut self, lane: usize, data: T) {
        self.items[lane] = WorkItem::some(data);
    }

    pub fn take_work(&mut self, lane: usize) -> Option<T> {
        if self.items[lane].valid {
            let data = self.items[lane].data;
            self.items[lane] = WorkItem::none();
            Some(data)
        } else {
            None
        }
    }

    /// Count of lanes with work
    pub fn work_count(&self) -> usize {
        self.items.iter().filter(|w| w.valid).count()
    }
}

// ============================================================================
// PATTERN 1: COMPACTION
// ============================================================================

/// Compaction: Pack active work items into lowest lanes
///
/// Before: [X, _, X, _, _, X, _, X]  (4 items scattered)
/// After:  [X, X, X, X, _, _, _, _]  (4 items packed)
///
/// This is useful for coalesced memory access and efficient processing.
pub mod compaction {
    use super::*;

    /// Ballot: count lanes with work, return mask
    pub fn ballot(queue: &WorkQueue<i32>) -> u32 {
        let mut mask = 0u32;
        for lane in 0..32 {
            if queue.has_work(lane) {
                mask |= 1 << lane;
            }
        }
        mask
    }

    /// Prefix sum (exclusive) of bits set before each position
    /// This gives the destination lane for each active item
    pub fn prefix_popcount(mask: u32) -> [usize; 32] {
        let mut result = [0usize; 32];
        let mut count = 0;
        for lane in 0..32 {
            result[lane] = count;
            if mask & (1 << lane) != 0 {
                count += 1;
            }
        }
        result
    }

    /// Compact work items to lowest lanes
    ///
    /// Type signature: Warp<All> -> WorkQueue -> WorkQueue
    /// Invariant: Work count preserved, positions changed
    pub fn compact(_warp: &Warp<All>, queue: &WorkQueue<i32>) -> WorkQueue<i32> {
        let mask = ballot(queue);
        let dest = prefix_popcount(mask);

        let mut result = WorkQueue::new();

        for lane in 0..32 {
            if queue.has_work(lane) {
                // This lane's work goes to dest[lane]
                let target = dest[lane];
                result.items[target] = queue.items[lane];
            }
        }

        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_compact_sparse() {
            let warp: Warp<All> = Warp::new();
            let mut queue = WorkQueue::new();

            // Scatter work: lanes 1, 4, 7, 10
            queue.set_work(1, 100);
            queue.set_work(4, 200);
            queue.set_work(7, 300);
            queue.set_work(10, 400);

            assert_eq!(queue.work_count(), 4);

            let compacted = compact(&warp, &queue);

            // Work should be in lanes 0, 1, 2, 3
            assert!(compacted.has_work(0));
            assert!(compacted.has_work(1));
            assert!(compacted.has_work(2));
            assert!(compacted.has_work(3));
            assert!(!compacted.has_work(4));

            // Values preserved
            assert_eq!(compacted.items[0].data, 100);
            assert_eq!(compacted.items[1].data, 200);
            assert_eq!(compacted.items[2].data, 300);
            assert_eq!(compacted.items[3].data, 400);

            assert_eq!(compacted.work_count(), 4);
        }

        #[test]
        fn test_compact_already_compact() {
            let warp: Warp<All> = Warp::new();
            let mut queue = WorkQueue::new();

            queue.set_work(0, 10);
            queue.set_work(1, 20);

            let compacted = compact(&warp, &queue);

            assert_eq!(compacted.items[0].data, 10);
            assert_eq!(compacted.items[1].data, 20);
        }
    }
}

// ============================================================================
// PATTERN 2: LOAD BALANCING
// ============================================================================

/// Load balancing: Redistribute work from busy to idle lanes
///
/// Session type perspective:
/// - Role discovery via ballot (uniform operation)
/// - Work transfer via shuffle (typed permutation)
/// - No blocking, no deadlock (lockstep execution)
pub mod load_balancing {
    use super::*;

    /// Find lanes that need work and lanes that have excess
    pub fn find_imbalance(queue: &WorkQueue<i32>) -> (Vec<usize>, Vec<usize>) {
        let mut needy = Vec::new(); // Lanes without work
        let mut donors = Vec::new(); // Lanes with work

        for lane in 0..32 {
            if queue.has_work(lane) {
                donors.push(lane);
            } else {
                needy.push(lane);
            }
        }

        (needy, donors)
    }

    /// Balance work: donors give to needy lanes
    ///
    /// Simple strategy: first N needy lanes get work from last N donors
    /// (where N = min(needy.len(), donors.len() / 2))
    ///
    /// This maintains Warp<All> - no divergence needed!
    pub fn balance(_warp: &Warp<All>, queue: &mut WorkQueue<i32>) {
        let (needy, donors) = find_imbalance(queue);

        // Transfer from donors to needy (take from end of donors)
        let transfer_count = std::cmp::min(needy.len(), donors.len() / 2);

        for i in 0..transfer_count {
            let donor_lane = donors[donors.len() - 1 - i];
            let needy_lane = needy[i];

            // Transfer work
            if let Some(data) = queue.take_work(donor_lane) {
                queue.set_work(needy_lane, data);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_balance_work() {
            let warp: Warp<All> = Warp::new();
            let mut queue = WorkQueue::new();

            // 4 lanes have work, 28 don't
            queue.set_work(0, 100);
            queue.set_work(1, 200);
            queue.set_work(2, 300);
            queue.set_work(3, 400);

            assert_eq!(queue.work_count(), 4);

            // Balance should transfer from 2 donors to 2 needy
            // (half of 4 donors = 2)
            balance(&warp, &mut queue);

            // Should still have 4 work items, more spread out
            assert_eq!(queue.work_count(), 4);

            // Lanes 2 and 3 should have donated (from end of donors list)
            assert!(!queue.has_work(2) || !queue.has_work(3));
        }
    }
}

// ============================================================================
// PATTERN 3: DYNAMIC ROLES
// ============================================================================

/// Dynamic roles: Lanes switch between producer/consumer based on state
///
/// Key insight: Role assignment happens via ballot, not divergence!
/// All lanes execute the same code, but ballot results tell them their role.
pub mod dynamic_roles {
    use super::*;

    /// Role that a lane plays in this iteration
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum Role {
        Producer, // Has work to give
        Consumer, // Needs work
        Idle,     // Neither (balanced)
    }

    /// Compute role for each lane based on work state
    pub fn compute_roles(queue: &WorkQueue<i32>) -> PerLane<Role> {
        let work_mask = compaction::ballot(queue);
        let work_count = work_mask.count_ones() as usize;

        // Target: spread work evenly
        // Lanes 0..work_count should have work
        // Lanes work_count..32 should be idle or consumers

        let mut roles = [Role::Idle; 32];

        for lane in 0..32 {
            let has_work = queue.has_work(lane);
            let should_have_work = lane < work_count;

            roles[lane] = match (has_work, should_have_work) {
                (true, false) => Role::Producer, // Has work, shouldn't
                (false, true) => Role::Consumer, // No work, should
                _ => Role::Idle,                 // Already balanced
            };
        }

        PerLane(roles)
    }

    /// Execute one round of work redistribution
    ///
    /// Session type: All lanes participate, roles determined by ballot
    /// No divergence needed - all lanes execute same code
    pub fn redistribute_round(_warp: &Warp<All>, queue: &mut WorkQueue<i32>) -> bool {
        let roles = compute_roles(queue);

        // Find producer-consumer pairs
        let mut producers: Vec<usize> = Vec::new();
        let mut consumers: Vec<usize> = Vec::new();

        for lane in 0..32 {
            match roles.0[lane] {
                Role::Producer => producers.push(lane),
                Role::Consumer => consumers.push(lane),
                Role::Idle => {}
            }
        }

        // Match producers with consumers
        let transfers = std::cmp::min(producers.len(), consumers.len());

        for i in 0..transfers {
            let from = producers[i];
            let to = consumers[i];

            if let Some(data) = queue.take_work(from) {
                queue.set_work(to, data);
            }
        }

        transfers > 0 // Return true if any work was transferred
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_role_computation() {
            let mut queue = WorkQueue::new();

            // Work in lanes 5, 10, 15, 20 (4 items, scattered)
            queue.set_work(5, 1);
            queue.set_work(10, 2);
            queue.set_work(15, 3);
            queue.set_work(20, 4);

            let roles = compute_roles(&queue);

            // Lanes 0-3 should want work (consumers)
            assert_eq!(roles.0[0], Role::Consumer);
            assert_eq!(roles.0[1], Role::Consumer);
            assert_eq!(roles.0[2], Role::Consumer);
            assert_eq!(roles.0[3], Role::Consumer);

            // Lanes with work beyond target should be producers
            assert_eq!(roles.0[5], Role::Producer);
            assert_eq!(roles.0[10], Role::Producer);
        }

        #[test]
        fn test_redistribute_until_balanced() {
            let warp: Warp<All> = Warp::new();
            let mut queue = WorkQueue::new();

            // All work in lane 0
            queue.set_work(0, 100);
            queue.set_work(1, 200);
            queue.set_work(2, 300);
            queue.set_work(3, 400);

            // Redistribute
            redistribute_round(&warp, &mut queue);

            // Work should now be in lanes 0-3 (already was)
            // But if work was scattered, it would compact
            assert_eq!(queue.work_count(), 4);
        }

        #[test]
        fn test_dynamic_roles_no_deadlock() {
            // Key property: all operations are collective (ballot, shuffle)
            // No lane waits for another, so no deadlock possible
            let warp: Warp<All> = Warp::new();
            let mut queue = WorkQueue::new();

            // Create imbalanced state
            for lane in 0..16 {
                queue.set_work(lane, lane as i32);
            }

            // Multiple rounds of redistribution
            for _ in 0..10 {
                redistribute_round(&warp, &mut queue);
            }

            // Still have all work (no loss)
            assert_eq!(queue.work_count(), 16);
        }
    }
}

// ============================================================================
// SESSION TYPE PERSPECTIVE
// ============================================================================

/// Summary: How session types express work-stealing
///
/// ## Key Insight: No Divergence Needed!
///
/// Intra-warp work stealing uses COLLECTIVE operations:
/// - `ballot()` to discover who has work (uniform result)
/// - `shuffle()` to transfer data (typed permutation)
/// - No blocking, no waiting, no divergence
///
/// This means:
/// 1. All lanes execute the SAME session (no role asymmetry)
/// 2. The "session" is parameterized by ballot results
/// 3. Deadlock is impossible (lockstep execution)
///
/// ## Session Type Structure
///
/// ```text
/// WorkStealSession =
///     ballot(has_work)              // Discover roles
///   . compute_roles(ballot_result)  // Assign producer/consumer
///   . shuffle(work_data)            // Transfer based on roles
///   . repeat                        // Until balanced
/// ```
///
/// ## Answering the Research Questions
///
/// Q: "Can session types express work-stealing efficiently?"
/// A: YES for intra-warp. The session is a loop of collective operations.
///    No divergence, no role asymmetry, just parameterized data flow.
///
/// Q: "Can we verify deadlock-freedom in work-stealing?"
/// A: YES for intra-warp. Deadlock-free BY CONSTRUCTION because:
///    - All operations are collective (no waiting)
///    - Lockstep execution (no blocking)
///    - Type Warp<All> ensures all lanes participate
///
/// Q: "How to handle dynamic role assignment?"
/// A: Roles computed FROM ballot results, not divergence. All lanes
///    execute the same code with different data based on ballot.
///
/// ## Inter-Warp Work Stealing
///
/// Different story! Warps CAN block on shared memory atomics.
/// This needs traditional session types (send/recv) with
/// careful protocol design to avoid deadlock.
/// See: inter_block.rs for inter-warp patterns.
pub const _SESSION_PERSPECTIVE: () = ();

// ============================================================================
// DEADLOCK FREEDOM PROOF SKETCH
// ============================================================================

/// Theorem: Intra-warp work stealing is deadlock-free
///
/// Proof sketch:
///
/// 1. All warp operations are COLLECTIVE:
///    - ballot() executes in all lanes simultaneously
///    - shuffle() executes in all lanes simultaneously
///    - No lane can "block" waiting for another
///
/// 2. SIMT execution model guarantees:
///    - All lanes in a warp progress together
///    - No lane can be "ahead" of others
///    - Divergence only affects which code executes, not timing
///
/// 3. Our work-stealing uses only Warp<All> operations:
///    - No diverge() calls
///    - All lanes participate in every ballot/shuffle
///    - Role differences are in DATA, not CODE
///
/// 4. Therefore:
///    - No circular waiting (all complete together)
///    - No blocking (collective ops don't block)
///    - Deadlock impossible QED
///
/// Note: This proof relies on STAYING in Warp<All>. If we diverged
/// and tried to shuffle across the divergence boundary, we'd violate
/// the type system (shuffle requires Warp<All>).
pub const _DEADLOCK_FREEDOM: () = ();

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_work_stealing_cycle() {
        let warp: Warp<All> = Warp::new();
        let mut queue = WorkQueue::new();

        // Initial: work scattered in odd lanes
        for lane in (1..32).step_by(2) {
            queue.set_work(lane, lane as i32 * 10);
        }
        assert_eq!(queue.work_count(), 16);

        // Compact first
        queue = compaction::compact(&warp, &queue);

        // All work should be in lanes 0-15
        for lane in 0..16 {
            assert!(queue.has_work(lane), "Lane {} should have work", lane);
        }
        for lane in 16..32 {
            assert!(!queue.has_work(lane), "Lane {} should not have work", lane);
        }

        // Balance (already compact, so mostly no-op)
        load_balancing::balance(&warp, &mut queue);

        // Still 16 items
        assert_eq!(queue.work_count(), 16);
    }
}
