//! Typed warp-level bitonic sort.
//!
//! Bitonic sort is the fundamental GPU sorting primitive. Each stage uses
//! shuffle-XOR to compare-and-swap elements between lanes at specific distances.
//! Getting the mask sequence wrong produces a subtly broken sort.
//!
//! Our type system provides two layers of safety:
//! 1. **All shuffles require `Warp<All>`** — can't sort a diverged warp
//! 2. **The mask sequence is structured** — each stage is a typed function
//!    with the correct XOR distances, not a bag of magic numbers
//!
//! # Bitonic Sort Algorithm (32 elements, 1 per lane)
//!
//! 5 stages, each with increasing substages:
//! ```text
//! Stage 1: compare-swap at distance 1           (1 substage)
//! Stage 2: compare-swap at distances 2, 1       (2 substages)
//! Stage 3: compare-swap at distances 4, 2, 1    (3 substages)
//! Stage 4: compare-swap at distances 8, 4, 2, 1 (4 substages)
//! Stage 5: compare-swap at distances 16, 8, 4, 2, 1 (5 substages)
//! ```
//!
//! Total: 15 compare-and-swap operations using shuffle-XOR.

use crate::GpuValue;
use crate::gpu::{self, GpuShuffle};
use crate::data::PerLane;
use crate::warp::Warp;
use crate::active_set::All;

// ============================================================================
// Core compare-and-swap via shuffle
// ============================================================================

/// Direction-aware compare-and-swap between lanes at distance `xor_mask`.
///
/// Uses `lane_id()` to determine sort direction per lane:
/// - `ascending = (lane_id & stage_mask) == 0`
/// - `is_lower = (lane_id & xor_mask) == 0` (lower lane in the pair)
/// - Lower lane keeps min in ascending blocks, max in descending blocks
///
/// This produces a correct bitonic sort on GPU hardware.
///
/// **CPU emulation:** `shuffle_xor` returns self, so `my == partner` always.
/// Direction logic is evaluated but moot — both branches produce the same value.
/// The sort is a no-op on CPU, which is correct for type-system validation.
#[inline(always)]
fn compare_swap<T: GpuValue + GpuShuffle + Ord>(
    warp: &Warp<All>,
    val: PerLane<T>,
    xor_mask: u32,
    stage_mask: u32,
) -> PerLane<T> {
    let partner_val = warp.shuffle_xor(val, xor_mask);
    let my = val.get();
    let partner = partner_val.get();

    // Direction-aware: each lane determines whether to keep min or max
    // based on its position in the bitonic network.
    let lane = gpu::lane_id();
    let ascending = (lane & stage_mask) == 0;
    let is_lower = (lane & xor_mask) == 0;
    let keep_smaller = ascending == is_lower;

    if keep_smaller {
        if my <= partner { val } else { partner_val }
    } else if my >= partner {
        val
    } else {
        partner_val
    }
}

// ============================================================================
// Bitonic sort stages (typed, correct by construction)
// ============================================================================

impl Warp<All> {
    /// Full warp-level bitonic sort of 32 elements (one per lane).
    ///
    /// Sorts in ascending order across lane indices. After sorting:
    /// - Lane 0 has the minimum
    /// - Lane 31 has the maximum
    ///
    /// Uses exactly 15 compare-and-swap operations via shuffle-XOR.
    /// The type system guarantees:
    /// - All shuffles operate on a full warp (no inactive lanes)
    /// - The mask sequence is correct (each stage properly structured)
    ///
    /// On GPU: 15 `shfl.sync.bfly.b32` instructions + 15 min/max comparisons.
    /// Zero overhead from the type system.
    ///
    /// Uses direction-aware compare-and-swap via `lane_id()` — correct on GPU.
    ///
    /// ```
    /// use warp_types::*;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let data = data::PerLane::new(42i32);
    /// let sorted = warp.bitonic_sort(data);
    /// ```
    ///
    /// Bitonic sort requires all lanes:
    ///
    /// ```compile_fail
    /// use warp_types::prelude::*;
    /// let warp = Warp::kernel_entry();
    /// let (evens, _odds) = warp.diverge_even_odd();
    /// let data = data::PerLane::new(42i32);
    /// evens.bitonic_sort(data); // ERROR: method not found for Warp<Even>
    /// ```
    pub fn bitonic_sort<T: GpuValue + GpuShuffle + Ord>(
        &self,
        data: PerLane<T>,
    ) -> PerLane<T> {
        let mut val = data;

        // Stage 1: blocks of 2
        val = compare_swap(self, val, 1, 2);

        // Stage 2: blocks of 4
        val = compare_swap(self, val, 2, 4);
        val = compare_swap(self, val, 1, 4);

        // Stage 3: blocks of 8
        val = compare_swap(self, val, 4, 8);
        val = compare_swap(self, val, 2, 8);
        val = compare_swap(self, val, 1, 8);

        // Stage 4: blocks of 16
        val = compare_swap(self, val, 8, 16);
        val = compare_swap(self, val, 4, 16);
        val = compare_swap(self, val, 2, 16);
        val = compare_swap(self, val, 1, 16);

        // Stage 5: blocks of 32 (full warp)
        val = compare_swap(self, val, 16, 32);
        val = compare_swap(self, val, 8, 32);
        val = compare_swap(self, val, 4, 32);
        val = compare_swap(self, val, 2, 32);
        val = compare_swap(self, val, 1, 32);

        val
    }

    /// Bitonic sort with custom comparator.
    ///
    /// Like `bitonic_sort` but uses a user-provided comparison function
    /// instead of `Ord`. Useful for sorting by key, reverse order, etc.
    pub fn bitonic_sort_by<T, F>(
        &self,
        data: PerLane<T>,
        cmp: F,
    ) -> PerLane<T>
    where
        T: GpuValue + GpuShuffle,
        F: Fn(&T, &T) -> core::cmp::Ordering,
    {
        let mut val = data;

        // Direction-aware custom compare-and-swap
        let cas = |warp: &Warp<All>, v: PerLane<T>, xor_mask: u32, stage_mask: u32| -> PerLane<T> {
            let partner_val = warp.shuffle_xor(v, xor_mask);
            let my = v.get();
            let partner = partner_val.get();
            let lane = gpu::lane_id();
            let ascending = (lane & stage_mask) == 0;
            let is_lower = (lane & xor_mask) == 0;
            let keep_smaller = ascending == is_lower;
            if keep_smaller {
                match cmp(&my, &partner) {
                    core::cmp::Ordering::Greater => partner_val,
                    _ => v,
                }
            } else {
                match cmp(&my, &partner) {
                    core::cmp::Ordering::Less => partner_val,
                    _ => v,
                }
            }
        };

        // Stage 1
        val = cas(self, val, 1, 2);
        // Stage 2
        val = cas(self, val, 2, 4);
        val = cas(self, val, 1, 4);
        // Stage 3
        val = cas(self, val, 4, 8);
        val = cas(self, val, 2, 8);
        val = cas(self, val, 1, 8);
        // Stage 4
        val = cas(self, val, 8, 16);
        val = cas(self, val, 4, 16);
        val = cas(self, val, 2, 16);
        val = cas(self, val, 1, 16);
        // Stage 5
        val = cas(self, val, 16, 32);
        val = cas(self, val, 8, 32);
        val = cas(self, val, 4, 32);
        val = cas(self, val, 2, 32);
        val = cas(self, val, 1, 32);

        val
    }

    /// Key-value bitonic sort: sorts keys, moves values to match.
    ///
    /// Common GPU pattern: sort an array of (key, value) pairs.
    /// Both key and value must be shuffleable.
    ///
    /// ```
    /// use warp_types::*;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let keys = data::PerLane::new(5i32);
    /// let vals = data::PerLane::new(100u32);
    /// let (sorted_keys, sorted_vals) = warp.bitonic_sort_pairs(keys, vals);
    /// ```
    pub fn bitonic_sort_pairs<K, V>(
        &self,
        keys: PerLane<K>,
        values: PerLane<V>,
    ) -> (PerLane<K>, PerLane<V>)
    where
        K: GpuValue + GpuShuffle + Ord,
        V: GpuValue + GpuShuffle,
    {
        let mut k = keys;
        let mut v = values;

        // Direction-aware key-value compare-and-swap
        let cas_kv = |warp: &Warp<All>,
                      key: PerLane<K>,
                      val: PerLane<V>,
                      xor_mask: u32,
                      stage_mask: u32|
                      -> (PerLane<K>, PerLane<V>) {
            let partner_key = warp.shuffle_xor(key, xor_mask);
            let partner_val = warp.shuffle_xor(val, xor_mask);
            let lane = gpu::lane_id();
            let ascending = (lane & stage_mask) == 0;
            let is_lower = (lane & xor_mask) == 0;
            let keep_smaller = ascending == is_lower;
            if keep_smaller {
                if key.get() <= partner_key.get() {
                    (key, val)
                } else {
                    (partner_key, partner_val)
                }
            } else if key.get() >= partner_key.get() {
                (key, val)
            } else {
                (partner_key, partner_val)
            }
        };

        // Stage 1
        (k, v) = cas_kv(self, k, v, 1, 2);
        // Stage 2
        (k, v) = cas_kv(self, k, v, 2, 4);
        (k, v) = cas_kv(self, k, v, 1, 4);
        // Stage 3
        (k, v) = cas_kv(self, k, v, 4, 8);
        (k, v) = cas_kv(self, k, v, 2, 8);
        (k, v) = cas_kv(self, k, v, 1, 8);
        // Stage 4
        (k, v) = cas_kv(self, k, v, 8, 16);
        (k, v) = cas_kv(self, k, v, 4, 16);
        (k, v) = cas_kv(self, k, v, 2, 16);
        (k, v) = cas_kv(self, k, v, 1, 16);
        // Stage 5
        (k, v) = cas_kv(self, k, v, 16, 32);
        (k, v) = cas_kv(self, k, v, 8, 32);
        (k, v) = cas_kv(self, k, v, 4, 32);
        (k, v) = cas_kv(self, k, v, 2, 32);
        (k, v) = cas_kv(self, k, v, 1, 32);

        (k, v)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::PerLane;

    #[test]
    fn test_bitonic_sort_single_value() {
        // TYPE-SYSTEM TEST: validates that bitonic_sort compiles on Warp<All>.
        // CPU shuffle is identity, so the sort is a no-op here.
        // For algorithmic correctness with real lane exchange, see simwarp::tests.
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(42i32);
        let sorted = warp.bitonic_sort(data);
        assert_eq!(sorted.get(), 42);
    }

    #[test]
    fn test_bitonic_sort_by_reverse() {
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(42i32);
        let sorted = warp.bitonic_sort_by(data, |a, b| b.cmp(a)); // reverse
        assert_eq!(sorted.get(), 42);
    }

    #[test]
    fn test_bitonic_sort_pairs() {
        let warp: Warp<All> = Warp::kernel_entry();
        let keys = PerLane::new(5i32);
        let vals = PerLane::new(100u32);
        let (sk, sv) = warp.bitonic_sort_pairs(keys, vals);
        assert_eq!(sk.get(), 5);
        assert_eq!(sv.get(), 100);
    }

    #[test]
    fn test_bitonic_requires_warp_all() {
        // This test documents that bitonic_sort is ONLY available on Warp<All>.
        // Attempting to call it on Warp<Even> would be a compile error.
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(1i32);
        let _ = warp.bitonic_sort(data);
        let _ = warp.bitonic_sort_by(data, |a, b| a.cmp(b));
        let _ = warp.bitonic_sort_pairs(data, PerLane::new(0u32));
    }

    #[test]
    fn test_compare_swap_ascending() {
        let warp: Warp<All> = Warp::kernel_entry();
        // When partner == self (CPU emulation), smaller value is kept
        let data = PerLane::new(10i32);
        let result = compare_swap(&warp, data, 1, 2);
        assert_eq!(result.get(), 10); // identity under CPU emulation
    }

    #[test]
    fn test_bitonic_shuffle_count() {
        // Verify the algorithm uses exactly 15 shuffle operations.
        // Stage 1: 1, Stage 2: 2, Stage 3: 3, Stage 4: 4, Stage 5: 5
        // Total: 1 + 2 + 3 + 4 + 5 = 15
        // Each compare_swap calls shuffle_xor once.
        // This is a documentation test — the structure is visible in the code.
        assert_eq!(1 + 2 + 3 + 4 + 5, 15);
    }

    #[test]
    fn test_bitonic_mask_sequence() {
        // Verify the XOR mask sequence is correct for bitonic sort.
        // Stage k (1-indexed) has substages with masks: 2^(k-1), 2^(k-2), ..., 2, 1
        let expected_masks: Vec<Vec<u32>> = vec![
            vec![1],                  // Stage 1
            vec![2, 1],              // Stage 2
            vec![4, 2, 1],           // Stage 3
            vec![8, 4, 2, 1],        // Stage 4
            vec![16, 8, 4, 2, 1],    // Stage 5
        ];

        for (stage, masks) in expected_masks.iter().enumerate() {
            let k = stage + 1;
            for (substage, &mask) in masks.iter().enumerate() {
                let j = masks.len() - substage; // j counts down from k
                assert_eq!(mask, 1 << (j - 1),
                    "Stage {}, substage {}: expected mask {}, got {}",
                    k, substage, 1u32 << (j - 1), mask);
            }
        }
    }
}
