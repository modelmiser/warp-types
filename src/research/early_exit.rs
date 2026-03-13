//! Early Exit Patterns
//!
//! Research question: "What about asymmetric merge (some lanes early-exit)?"
//!
//! # Background
//!
//! In many GPU algorithms, some lanes find their answer early and want to exit:
//!
//! ```text
//! for each lane:
//!     while not_found[lane]:
//!         search_step()
//!         if found[lane]:
//!             break  // This lane exits early!
//! ```
//!
//! This creates "asymmetric" divergence - lanes exit at different times.
//! How do we type this?
//!
//! # Patterns Explored
//!
//! 1. **Ballot-based exit**: All lanes check, exit together when all done
//! 2. **Progressive reduction**: Shrinking active set, no warp ops
//! 3. **Work redistribution**: Lanes that finish help others
//! 4. **Existential exit**: Track "at most N lanes active"
//!
//! # Key Insight
//!
//! Early exit is a special case of REDUCING active set (from varying_loops.rs).
//! The active set shrinks monotonically. Two approaches:
//!
//! 1. **No warp ops after exit**: Body can't use shuffle/reduce
//! 2. **Ballot coordination**: Use ballot to check if all done, exit together

use std::marker::PhantomData;

// ============================================================================
// BASIC TYPES (reusing from static_verify)
// ============================================================================

pub trait ActiveSet: Copy + 'static {
    const MASK: u32;
}

#[derive(Copy, Clone)]
pub struct All;
impl ActiveSet for All { const MASK: u32 = 0xFFFFFFFF; }

#[derive(Copy, Clone)]
pub struct Warp<S: ActiveSet> {
    _marker: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self {
        Warp { _marker: PhantomData }
    }
}

#[derive(Copy, Clone)]
pub struct PerLane<T>(pub [T; 32]);

// ============================================================================
// PATTERN 1: BALLOT-BASED EXIT
// ============================================================================

/// All lanes check condition, exit together when all lanes are done.
/// This maintains Warp<All> throughout.
pub mod ballot_exit {
    use super::*;

    /// Ballot: check which lanes have a condition true
    pub fn ballot(_warp: &Warp<All>, pred: PerLane<bool>) -> u32 {
        let mut mask = 0u32;
        for lane in 0..32 {
            if pred.0[lane] {
                mask |= 1 << lane;
            }
        }
        mask
    }

    /// All lanes done?
    pub fn all_done(warp: &Warp<All>, done: PerLane<bool>) -> bool {
        ballot(warp, done) == 0xFFFFFFFF
    }

    /// Any lane done?
    pub fn any_done(warp: &Warp<All>, done: PerLane<bool>) -> bool {
        ballot(warp, done) != 0
    }

    /// Search with ballot-based exit
    ///
    /// All lanes execute until ALL are done.
    /// Lanes that finish early just spin (waste work but safe).
    pub fn search_ballot<F>(warp: Warp<All>, mut step: F) -> (Warp<All>, PerLane<bool>)
    where
        F: FnMut(usize) -> PerLane<bool>,  // Returns "found" per lane
    {
        let mut found = PerLane([false; 32]);
        let mut iter = 0;

        // All lanes execute together
        while !all_done(&warp, found) {
            let new_found = step(iter);
            // Update found status
            for lane in 0..32 {
                found.0[lane] = found.0[lane] || new_found.0[lane];
            }
            iter += 1;
        }

        (warp, found)  // Still Warp<All>!
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_ballot_exit() {
            let warp: Warp<All> = Warp::new();

            // Simulate: lane N finds answer at iteration N
            let (warp_out, found) = search_ballot(warp, |iter| {
                let mut result = [false; 32];
                for lane in 0..32 {
                    result[lane] = iter >= lane;  // Lane 0 done at iter 0, lane 31 at iter 31
                }
                PerLane(result)
            });

            // All lanes should have found
            assert!(found.0.iter().all(|&f| f));

            // Warp is still All
            let _: Warp<All> = warp_out;
        }
    }
}

// ============================================================================
// PATTERN 2: REDUCING ACTIVE SET
// ============================================================================

/// Active set shrinks as lanes exit. No warp ops allowed in body.
pub mod reducing_exit {
    use super::*;

    /// A warp with runtime-tracked active set (shrinking)
    pub struct ReducingWarp {
        active_mask: u32,
    }

    impl ReducingWarp {
        pub fn new() -> Self {
            ReducingWarp { active_mask: 0xFFFFFFFF }
        }

        pub fn active_mask(&self) -> u32 {
            self.active_mask
        }

        pub fn any_active(&self) -> bool {
            self.active_mask != 0
        }

        pub fn all_exited(&self) -> bool {
            self.active_mask == 0
        }

        /// Mark lanes as exited
        pub fn exit_lanes(&mut self, exiting: u32) {
            self.active_mask &= !exiting;
        }
    }

    /// Search with reducing active set
    ///
    /// Lanes that find their answer exit. No shuffle/reduce in body!
    pub fn search_reducing<F>(mut step: F) -> PerLane<bool>
    where
        F: FnMut(usize, u32) -> u32,  // Returns mask of lanes that found
    {
        let mut warp = ReducingWarp::new();
        let mut found = PerLane([false; 32]);
        let mut iter = 0;

        while warp.any_active() {
            let newly_found = step(iter, warp.active_mask());

            // Update found status
            for lane in 0..32 {
                if newly_found & (1 << lane) != 0 {
                    found.0[lane] = true;
                }
            }

            // Exit those lanes
            warp.exit_lanes(newly_found);
            iter += 1;
        }

        found
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_reducing_exit() {
            // Lane N exits at iteration N
            let found = search_reducing(|iter, active| {
                let mut exiting = 0u32;
                for lane in 0..32 {
                    if active & (1 << lane) != 0 && iter == lane {
                        exiting |= 1 << lane;
                    }
                }
                exiting
            });

            assert!(found.0.iter().all(|&f| f));
        }
    }
}

// ============================================================================
// PATTERN 3: WORK REDISTRIBUTION
// ============================================================================

/// Lanes that finish early help others (work stealing within warp).
pub mod work_redistribution {
    use super::*;

    /// Work item for redistribution
    #[derive(Copy, Clone, Debug)]
    pub struct WorkItem {
        pub lane: usize,
        pub data: i32,
    }

    /// Redistribute work from finished lanes to busy ones
    ///
    /// This is complex but enables better utilization.
    /// Requires coordination via shuffle.
    pub fn redistribute_work(
        _warp: &Warp<All>,
        _done: PerLane<bool>,
        work: PerLane<Option<WorkItem>>,
    ) -> PerLane<Option<WorkItem>> {
        // Find lanes that are done but still have work
        // Find lanes that need more work
        // Use shuffle to redistribute

        // Placeholder - real implementation would use ballot + shuffle
        work
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_redistribute_placeholder() {
            let warp: Warp<All> = Warp::new();
            let done = PerLane([false; 32]);
            let work = PerLane([None; 32]);

            let _redistributed = redistribute_work(&warp, done, work);
        }
    }
}

// ============================================================================
// PATTERN 4: EXISTENTIAL EXIT TRACKING
// ============================================================================

/// Track "at most N lanes still active" without knowing which ones.
///
/// Note: Full implementation of bounds tracking would need dependent types.
/// This is a sketch showing the concept.
pub mod existential_exit {
    /// A warp where we track active count at runtime
    pub struct BoundedWarp {
        active_mask: u32,
        max_active: usize,  // Upper bound (shrinks over time)
    }

    impl BoundedWarp {
        pub fn new() -> Self {
            BoundedWarp {
                active_mask: 0xFFFFFFFF,
                max_active: 32,
            }
        }

        pub fn max_active(&self) -> usize {
            self.max_active
        }

        /// Exit some lanes, updating the bound
        pub fn exit_lanes(&mut self, exiting: u32) {
            let count = exiting.count_ones() as usize;
            self.active_mask &= !exiting;
            self.max_active = self.max_active.saturating_sub(count);
        }

        pub fn all_exited(&self) -> bool {
            self.max_active == 0
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_bounded_warp() {
            let mut warp = BoundedWarp::new();
            assert_eq!(warp.max_active(), 32);

            warp.exit_lanes(0x0000FFFF);  // Exit 16 lanes
            assert_eq!(warp.max_active(), 16);

            warp.exit_lanes(0xFFFF0000);  // Exit remaining 16
            assert!(warp.all_exited());
        }
    }
}

// ============================================================================
// KEY INSIGHT: EARLY EXIT = REDUCING LOOP
// ============================================================================

/// Summary of findings:
///
/// Early exit is a special case of the "reducing" loop pattern from varying_loops.rs.
///
/// | Pattern | Warp Ops? | Complexity | Use Case |
/// |---------|-----------|------------|----------|
/// | Ballot-based | Yes | Low | When most lanes finish together |
/// | Reducing | No | Low | Independent per-lane work |
/// | Work redistribution | Yes | High | Load balancing critical |
/// | Existential | Partial | Medium | When exact set unknown |
///
/// Recommendation: Use ballot-based for most cases. It maintains Warp<All>
/// and allows full warp operations. Lanes that finish early just skip work
/// (predicated execution, natural on GPU).

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_ballot_preserves_warp_all() {
        let warp: Warp<All> = Warp::new();

        let (warp_out, _) = ballot_exit::search_ballot(warp, |iter| {
            PerLane([iter > 10; 32])
        });

        // Type system confirms: still Warp<All>
        let _: Warp<All> = warp_out;
    }

    #[test]
    fn test_reducing_no_warp_type() {
        // Reducing pattern doesn't return a typed Warp
        // because the active set is runtime-dependent
        let found = reducing_exit::search_reducing(|iter, _| {
            if iter < 32 { 1u32 << iter } else { 0 }
        });

        assert!(found.0.iter().all(|&f| f));
    }
}
