// Lane-indexed loops are intentional: shuffle operations use the index for
// arithmetic (XOR, addition, modular segment math), not just array access.
#![allow(clippy::needless_range_loop)]

//! SimWarp: multi-lane warp simulator with real shuffle semantics.
//!
//! Unlike the single-lane `GpuShuffle` (identity on CPU), `SimWarp` holds
//! all lane values and performs real data exchange. Use this to validate
//! algorithms that depend on non-identity shuffle behavior.
//!
//! # Relationship to `GpuShuffle`
//!
//! `GpuShuffle` operates on a single scalar — it can't do real exchange
//! because it doesn't know the other lanes' values. SimWarp fills this gap
//! by holding `[T; WIDTH]` and implementing the actual GPU shuffle semantics:
//!
//! - `shuffle_xor(mask)`: lane[i] reads from lane[i ^ mask]
//! - `shuffle_down(delta)`: lane[i] reads from lane[i + delta] (clamps at WIDTH)
//! - `shuffle_up(delta)`: lane[i] reads from lane[i - delta] (clamps at 0)
//! - Width-confined variants partition into segments of `width` lanes

/// Multi-lane warp simulator with real shuffle semantics.
///
/// `WIDTH` defaults to 32 (NVIDIA). Use 64 for AMD wavefront testing.
#[derive(Clone, Debug)]
pub struct SimWarp<T: Copy, const WIDTH: usize = 32> {
    pub lanes: [T; WIDTH],
}

impl<T: Copy + Default, const WIDTH: usize> SimWarp<T, WIDTH> {
    /// Create a SimWarp with per-lane initialization.
    ///
    /// ```ignore
    /// let sw = SimWarp::<i32>::new(|lane| lane as i32 * 10);
    /// assert_eq!(sw.lanes[3], 30);
    /// ```
    pub fn new(init: impl Fn(u32) -> T) -> Self {
        let mut lanes = [T::default(); WIDTH];
        for i in 0..WIDTH {
            lanes[i] = init(i as u32);
        }
        SimWarp { lanes }
    }

    /// Read a single lane's value.
    pub fn lane(&self, id: usize) -> T {
        self.lanes[id]
    }
}

impl<T: Copy, const WIDTH: usize> SimWarp<T, WIDTH> {
    /// Create from an existing array.
    pub fn from_array(lanes: [T; WIDTH]) -> Self {
        SimWarp { lanes }
    }

    // ========================================================================
    // Shuffle operations — real GPU semantics
    // ========================================================================

    /// Butterfly shuffle: lane[i] reads from lane[i ^ mask].
    ///
    /// GPU: `shfl.sync.bfly.b32`. XOR is its own inverse — applying twice
    /// returns to the original arrangement.
    pub fn shuffle_xor(&self, mask: u32) -> Self {
        let mut out = self.lanes;
        for i in 0..WIDTH {
            let src = (i as u32 ^ mask) as usize;
            out[i] = if src < WIDTH { self.lanes[src] } else { self.lanes[i] };
        }
        SimWarp { lanes: out }
    }

    /// Shuffle down: lane[i] reads from lane[i + delta].
    /// Out-of-range lanes read their own value (GPU clamp behavior).
    pub fn shuffle_down(&self, delta: u32) -> Self {
        let mut out = self.lanes;
        for i in 0..WIDTH {
            // Use u64 arithmetic to prevent usize overflow on 32-bit platforms.
            let src = i as u64 + delta as u64;
            out[i] = if src < WIDTH as u64 { self.lanes[src as usize] } else { self.lanes[i] };
        }
        SimWarp { lanes: out }
    }

    /// Shuffle up: lane[i] reads from lane[i - delta].
    /// Lanes below delta read their own value (GPU clamp behavior).
    pub fn shuffle_up(&self, delta: u32) -> Self {
        let mut out = self.lanes;
        for i in 0..WIDTH {
            out[i] = if (i as u32) >= delta {
                self.lanes[i - delta as usize]
            } else {
                self.lanes[i]
            };
        }
        SimWarp { lanes: out }
    }

    /// Indexed shuffle: all lanes read from lane[src_lane].
    pub fn shuffle_idx(&self, src_lane: u32) -> Self {
        let src = src_lane as usize;
        let val = if src < WIDTH { self.lanes[src] } else { self.lanes[0] };
        SimWarp { lanes: [val; WIDTH] }
    }

    /// Butterfly shuffle confined to segments of `width` lanes.
    ///
    /// Each segment of `width` consecutive lanes shuffles independently.
    /// XOR stays within the segment: if `(i ^ mask)` would leave the
    /// segment, the lane reads its own value.
    pub fn shuffle_xor_width(&self, mask: u32, width: u32) -> Self {
        let w = width as usize;
        assert!(w > 0 && w.is_power_of_two() && w <= WIDTH, "width must be power-of-2 in 1..={WIDTH}");
        let mut out = self.lanes;
        for i in 0..WIDTH {
            let seg_base = (i / w) * w;
            let within = i % w;
            let partner_within = within ^ (mask as usize);
            out[i] = if partner_within < w {
                self.lanes[seg_base + partner_within]
            } else {
                self.lanes[i]
            };
        }
        SimWarp { lanes: out }
    }

    /// Shuffle down confined to segments of `width` lanes.
    pub fn shuffle_down_width(&self, delta: u32, width: u32) -> Self {
        let w = width as usize;
        assert!(w > 0 && w.is_power_of_two() && w <= WIDTH, "width must be power-of-2 in 1..={WIDTH}");
        let mut out = self.lanes;
        for i in 0..WIDTH {
            let seg_base = (i / w) * w;
            let within = i % w;
            // Use u64 arithmetic to prevent usize overflow on 32-bit platforms.
            let src_within = within as u64 + delta as u64;
            out[i] = if src_within < w as u64 {
                self.lanes[seg_base + src_within as usize]
            } else {
                self.lanes[i]
            };
        }
        SimWarp { lanes: out }
    }

    /// Shuffle up confined to segments of `width` lanes.
    pub fn shuffle_up_width(&self, delta: u32, width: u32) -> Self {
        let w = width as usize;
        assert!(w > 0 && w.is_power_of_two() && w <= WIDTH, "width must be power-of-2 in 1..={WIDTH}");
        let mut out = self.lanes;
        for i in 0..WIDTH {
            let seg_base = (i / w) * w;
            let within = i % w;
            out[i] = if within >= delta as usize {
                self.lanes[seg_base + within - delta as usize]
            } else {
                self.lanes[i]
            };
        }
        SimWarp { lanes: out }
    }

    // ========================================================================
    // Combinators
    // ========================================================================

    /// Element-wise binary operation (add, min, max, etc.)
    pub fn zip_with(&self, other: &Self, f: impl Fn(T, T) -> T) -> Self {
        let mut out = self.lanes;
        for i in 0..WIDTH {
            out[i] = f(self.lanes[i], other.lanes[i]);
        }
        SimWarp { lanes: out }
    }

    /// Per-lane transform.
    pub fn map(&self, f: impl Fn(usize, T) -> T) -> Self {
        let mut out = self.lanes;
        for i in 0..WIDTH {
            out[i] = f(i, self.lanes[i]);
        }
        SimWarp { lanes: out }
    }
}

// ============================================================================
// Algorithm reference implementations — same shuffle sequences as the
// actual code in cub.rs, sort.rs, tile.rs, but with real lane exchange.
// ============================================================================

/// Butterfly reduction (matches cub.rs `Warp<All>::reduce` for 32-lane warps).
///
/// Dynamically computes log2(WIDTH) XOR stages, so works for any power-of-2 WIDTH.
pub fn butterfly_reduce<T: Copy + Default, const WIDTH: usize>(
    sw: &SimWarp<T, WIDTH>,
    op: impl Fn(T, T) -> T,
) -> SimWarp<T, WIDTH> {
    let mut v = sw.clone();
    let mut mask = WIDTH as u32 / 2;
    while mask >= 1 {
        let shuffled = v.shuffle_xor(mask);
        v = v.zip_with(&shuffled, &op);
        mask /= 2;
    }
    v
}

/// Tile-confined butterfly reduction (matches tile.rs `Tile<SIZE>::reduce_sum`).
pub fn tile_reduce<T: Copy + Default>(
    sw: &SimWarp<T>,
    tile_size: u32,
    op: impl Fn(T, T) -> T,
) -> SimWarp<T> {
    let mut v = sw.clone();
    let mut stride = 1u32;
    while stride < tile_size {
        let shuffled = v.shuffle_xor_width(stride, tile_size);
        v = v.zip_with(&shuffled, &op);
        stride *= 2;
    }
    v
}

/// Bitonic sort (matches sort.rs `Warp<All>::bitonic_sort`).
///
/// Direction-aware compare-and-swap using lane_id, identical to the
/// actual GPU algorithm. Returns sorted SimWarp.
pub fn bitonic_sort(sw: &SimWarp<i32>) -> SimWarp<i32> {
    let mut v = sw.clone();

    let cas = |v: &SimWarp<i32>, xor_mask: u32, stage_mask: u32| -> SimWarp<i32> {
        let partner = v.shuffle_xor(xor_mask);
        v.map(|lane_id, my| {
            let p = partner.lane(lane_id);
            let ascending = (lane_id as u32 & stage_mask) == 0;
            let is_lower = (lane_id as u32 & xor_mask) == 0;
            let keep_smaller = ascending == is_lower;
            if keep_smaller {
                if my <= p { my } else { p }
            } else if my >= p {
                my
            } else {
                p
            }
        })
    };

    // Exact same 15-step sequence as sort.rs
    v = cas(&v, 1, 2);

    v = cas(&v, 2, 4);
    v = cas(&v, 1, 4);

    v = cas(&v, 4, 8);
    v = cas(&v, 2, 8);
    v = cas(&v, 1, 8);

    v = cas(&v, 8, 16);
    v = cas(&v, 4, 16);
    v = cas(&v, 2, 16);
    v = cas(&v, 1, 16);

    v = cas(&v, 16, 32);
    v = cas(&v, 8, 32);
    v = cas(&v, 4, 32);
    v = cas(&v, 2, 32);
    v = cas(&v, 1, 32);

    v
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Shuffle correctness ---

    #[test]
    fn shuffle_xor_swaps_pairs() {
        let sw = SimWarp::<i32>::new(|i| i as i32);
        let result = sw.shuffle_xor(1);
        // XOR 1 swaps adjacent pairs: [1,0,3,2,5,4,...]
        assert_eq!(result.lane(0), 1);
        assert_eq!(result.lane(1), 0);
        assert_eq!(result.lane(2), 3);
        assert_eq!(result.lane(3), 2);
    }

    #[test]
    fn shuffle_xor_is_involution() {
        let sw = SimWarp::<i32>::new(|i| i as i32 * 7 + 3);
        for mask in [1, 2, 4, 8, 16] {
            let double = sw.shuffle_xor(mask).shuffle_xor(mask);
            assert_eq!(sw.lanes, double.lanes, "XOR {mask} should be self-inverse");
        }
    }

    #[test]
    fn shuffle_down_clamps() {
        let sw = SimWarp::<i32>::new(|i| i as i32);
        let result = sw.shuffle_down(1);
        // Lane 0 reads from lane 1, ..., lane 30 reads from lane 31
        assert_eq!(result.lane(0), 1);
        assert_eq!(result.lane(30), 31);
        // Lane 31 clamps to own value
        assert_eq!(result.lane(31), 31);
    }

    #[test]
    fn shuffle_up_clamps() {
        let sw = SimWarp::<i32>::new(|i| i as i32);
        let result = sw.shuffle_up(1);
        // Lane 0 clamps to own value
        assert_eq!(result.lane(0), 0);
        // Lane 1 reads from lane 0, etc.
        assert_eq!(result.lane(1), 0);
        assert_eq!(result.lane(31), 30);
    }

    // --- Tile confinement ---

    #[test]
    fn tile_shuffle_xor_confined() {
        let sw = SimWarp::<i32>::new(|i| i as i32);
        // Tile size 8: lanes 0-7 are one tile, 8-15 another, etc.
        let result = sw.shuffle_xor_width(1, 8);

        // Within tile 0: lane 0 gets lane 1's value, lane 1 gets lane 0's
        assert_eq!(result.lane(0), 1);
        assert_eq!(result.lane(1), 0);

        // Within tile 1: lane 8 gets lane 9's value
        assert_eq!(result.lane(8), 9);
        assert_eq!(result.lane(9), 8);

        // Cross-tile XOR: mask=8 in an 8-wide tile exceeds tile boundary
        let cross = sw.shuffle_xor_width(8, 8);
        // Should clamp to own value (can't leave tile)
        assert_eq!(cross.lane(0), 0);
        assert_eq!(cross.lane(7), 7);
    }

    // --- Butterfly reduce with distinct values ---

    #[test]
    fn butterfly_reduce_sum_distinct() {
        let sw = SimWarp::<i32>::new(|i| i as i32 + 1); // [1, 2, ..., 32]
        let result = butterfly_reduce(&sw, |a, b| a + b);
        let expected = 32 * 33 / 2; // 528
        for i in 0..32 {
            assert_eq!(result.lane(i), expected, "lane {i} should have sum {expected}");
        }
    }

    #[test]
    fn butterfly_reduce_max_distinct() {
        let sw = SimWarp::<i32>::new(|i| (i as i32 * 7 + 13) % 100);
        let result = butterfly_reduce(&sw, |a, b| a.max(b));
        let expected = (0..32).map(|i| (i * 7 + 13) % 100).max().unwrap();
        assert_eq!(result.lane(0), expected);
    }

    // --- Tile reduce with distinct values ---

    #[test]
    fn tile_reduce_sum_per_tile() {
        let sw = SimWarp::<i32>::new(|i| i as i32 + 1); // [1..=32]
        let result = tile_reduce(&sw, 8, |a, b| a + b);

        // Tile 0 (lanes 0-7): sum of 1+2+...+8 = 36
        for i in 0..8 {
            assert_eq!(result.lane(i), 36, "tile 0, lane {i}");
        }
        // Tile 1 (lanes 8-15): sum of 9+10+...+16 = 100
        for i in 8..16 {
            assert_eq!(result.lane(i), 100, "tile 1, lane {i}");
        }
        // Tile 2 (lanes 16-23): sum of 17+18+...+24 = 164
        for i in 16..24 {
            assert_eq!(result.lane(i), 164, "tile 2, lane {i}");
        }
        // Tile 3 (lanes 24-31): sum of 25+26+...+32 = 228
        for i in 24..32 {
            assert_eq!(result.lane(i), 228, "tile 3, lane {i}");
        }
    }

    // --- Bitonic sort with real data exchange ---

    #[test]
    fn bitonic_sort_ascending() {
        // Reverse order: lane 0 has 31, lane 31 has 0
        let sw = SimWarp::<i32>::new(|i| 31 - i as i32);
        let sorted = bitonic_sort(&sw);
        for i in 0..32 {
            assert_eq!(sorted.lane(i), i as i32, "lane {i} should have {i}");
        }
    }

    #[test]
    fn bitonic_sort_already_sorted() {
        let sw = SimWarp::<i32>::new(|i| i as i32);
        let sorted = bitonic_sort(&sw);
        for i in 0..32 {
            assert_eq!(sorted.lane(i), i as i32);
        }
    }

    #[test]
    fn bitonic_sort_all_same() {
        let sw = SimWarp::<i32>::new(|_| 42);
        let sorted = bitonic_sort(&sw);
        for i in 0..32 {
            assert_eq!(sorted.lane(i), 42);
        }
    }

    #[test]
    fn bitonic_sort_random_pattern() {
        // Pseudo-random pattern: (i * 17 + 5) % 32
        let sw = SimWarp::<i32>::new(|i| ((i as i32 * 17 + 5) % 32));
        let sorted = bitonic_sort(&sw);
        // Verify monotonically non-decreasing
        for i in 1..32 {
            assert!(
                sorted.lane(i) >= sorted.lane(i - 1),
                "lane {} ({}) < lane {} ({})",
                i, sorted.lane(i), i - 1, sorted.lane(i - 1)
            );
        }
    }

    // --- Hillis-Steele scan (demonstrate the cub.rs bug) ---

    #[test]
    fn hillis_steele_correct_with_guard() {
        // Correct scan: only add when lane_id >= stride
        let sw = SimWarp::<i32>::new(|_| 1); // all ones
        let mut v = sw;
        for stride in [1u32, 2, 4, 8, 16] {
            let shifted = v.shuffle_up(stride);
            v = v.map(|lane_id, val| {
                if lane_id as u32 >= stride {
                    val + shifted.lane(lane_id)
                } else {
                    val
                }
            });
        }
        // Lane i should have i+1 (inclusive prefix sum of all 1s)
        for i in 0..32 {
            assert_eq!(v.lane(i), (i + 1) as i32, "lane {i}");
        }
    }

    #[test]
    fn hillis_steele_broken_without_guard() {
        // This reproduces the cub.rs bug: unconditional add after shfl_up
        let sw = SimWarp::<i32>::new(|_| 1);
        let mut v = sw;
        for stride in [1u32, 2, 4, 8, 16] {
            let shifted = v.shuffle_up(stride);
            // Bug: no guard — lanes < stride get clamped value (own) and double
            v = v.zip_with(&shifted, |a, b| a + b);
        }
        // Lane 0 should be 1 but gets doubled every stage: 1→2→4→8→16→32
        assert_eq!(v.lane(0), 32, "lane 0 should be 32 (doubled 5 times)");
        // Lane 31 should be 32 (correct inclusive sum)
        assert_eq!(v.lane(31), 32, "lane 31 happens to be correct");
        // Lane 1 should be 2 but gets doubled from stage 2 onward
        assert_ne!(v.lane(1), 2, "lane 1 is wrong — scan bug");
    }

    // --- 64-lane AMD wavefront ---

    #[test]
    fn simwarp_64_lane_reduce() {
        let sw = SimWarp::<i32, 64>::new(|i| i as i32 + 1);
        let mut v = sw;
        for &mask in &[32, 16, 8, 4, 2, 1] {
            let shuffled = v.shuffle_xor(mask);
            v = v.zip_with(&shuffled, |a, b| a + b);
        }
        let expected = 64 * 65 / 2; // 2080
        assert_eq!(v.lane(0), expected);
    }
}
