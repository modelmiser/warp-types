//! Typed CUB-equivalent warp primitives.
//!
//! Provides warp-collective operations matching NVIDIA CUB's warp-level API,
//! but with compile-time safety: all operations require `Warp<All>`.
//!
//! # CUB Equivalents
//!
//! | CUB (C++)               | warp-types (Rust)                    |
//! |--------------------------|--------------------------------------|
//! | `cub::WarpReduce<T>`     | `warp.reduce(data, op)`              |
//! | `cub::WarpScan<T>`       | `warp.inclusive_sum(data)`            |
//! | `cub::WarpExchange<T>`   | *(not yet implemented)*              |
//!
//! # Safety
//!
//! All methods are on `Warp<All>` — the compiler prevents calling them
//! on diverged sub-warps. CUB's C++ API has no such protection.

use crate::active_set::All;
use crate::data::PerLane;
use crate::gpu::GpuShuffle;
use crate::warp::Warp;
use crate::GpuValue;

// ============================================================================
// WarpReduce — reduction with arbitrary operator
// ============================================================================

impl Warp<All> {
    /// Reduce with a custom binary operator across all lanes.
    ///
    /// Equivalent to `cub::WarpReduce<T>::Reduce(val, op)`.
    /// Uses butterfly reduction (5 stages for 32 lanes).
    ///
    /// Every lane gets the result (unlike CUB where only lane 0 does).
    ///
    /// ```
    /// use warp_types::*;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let data = data::PerLane::new(3i32);
    /// let max = warp.reduce(data, |a, b| if a > b { a } else { b });
    /// ```
    pub fn reduce<T, F>(&self, data: PerLane<T>, op: F) -> T
    where
        T: GpuValue + GpuShuffle,
        F: Fn(T, T) -> T,
    {
        let mut val = data.get();
        val = op(val, val.gpu_shfl_xor(16));
        val = op(val, val.gpu_shfl_xor(8));
        val = op(val, val.gpu_shfl_xor(4));
        val = op(val, val.gpu_shfl_xor(2));
        val = op(val, val.gpu_shfl_xor(1));
        val
    }

    /// Inclusive prefix sum across all lanes.
    ///
    /// Equivalent to `cub::WarpScan<T>::InclusiveSum(val, &sum)`.
    /// Uses Hillis-Steele parallel scan (5 stages for 32 lanes).
    ///
    /// Lane i should get sum of lanes 0..=i.
    ///
    /// **WARNING:** This function does not produce a correct inclusive scan on
    /// any target. On CPU, `shfl_up` is identity, so each stage doubles the
    /// value (result: val × 32). On GPU, lanes where `lane_id < stride` get
    /// their own value back from `shfl_up` (clamped), causing them to double
    /// instead of preserving their partial sum. A correct Hillis-Steele scan
    /// requires `if lane_id >= stride { val = val + s; }`, which needs
    /// per-lane branching not available in the single-thread CPU emulation
    /// (the `gpu::lane_id()` intrinsic exists but returns 0 on CPU).
    ///
    /// Retained to demonstrate the type-system contract (requires `Warp<All>`).
    ///
    /// ```
    /// use warp_types::*;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let data = data::PerLane::new(1i32);
    /// let prefix = warp.inclusive_sum(data);
    /// // CPU emulation: NOT a correct inclusive scan (see doc)
    /// ```
    #[deprecated(
        note = "Not correct on any target — Hillis-Steele without lane_id guard. Use SimWarp for tested scan."
    )]
    pub fn inclusive_sum<T>(&self, data: PerLane<T>) -> PerLane<T>
    where
        T: GpuValue + GpuShuffle + core::ops::Add<Output = T>,
    {
        let mut val = data.get();
        // Hillis-Steele: add from lane (id - delta), not (id + delta)
        // Using shuffle-up: lane[i] reads from lane[i - delta]
        let s1 = val.gpu_shfl_up(1);
        val = val + s1;
        let s2 = val.gpu_shfl_up(2);
        val = val + s2;
        let s4 = val.gpu_shfl_up(4);
        val = val + s4;
        let s8 = val.gpu_shfl_up(8);
        val = val + s8;
        let s16 = val.gpu_shfl_up(16);
        val = val + s16;
        PerLane::new(val)
    }

    /// Exclusive prefix sum across all lanes (STUB -- not yet correct).
    ///
    /// Equivalent to `cub::WarpScan<T>::ExclusiveSum(val, &sum)`.
    /// Lane i should get sum of lanes 0..i (lane 0 gets `identity`).
    ///
    /// **WARNING:** This function does not produce a correct exclusive scan on
    /// any target. Lane 0 should receive `identity` but gets its own value
    /// instead, because the implementation lacks `lane_id()`. On CPU,
    /// `shfl_up` returns self, so the result is the inclusive sum. On GPU,
    /// lane 0's `shfl_up(1)` returns its own value, not the identity.
    /// Use `inclusive_sum` and implement the shift manually in GPU kernel code.
    ///
    /// Retained to demonstrate the type-system contract (requires `Warp<All>`).
    ///
    /// ```
    /// use warp_types::*;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let data = data::PerLane::new(1i32);
    /// #[allow(deprecated)]
    /// let prefix = warp.exclusive_sum(data, 0);
    /// // CPU emulation: NOT a correct exclusive scan (see doc)
    /// ```
    #[deprecated(note = "produces incorrect results — use inclusive_sum and manual shift instead")]
    pub fn exclusive_sum<T>(&self, data: PerLane<T>, identity: T) -> PerLane<T>
    where
        T: GpuValue + GpuShuffle + core::ops::Add<Output = T>,
    {
        // Inclusive scan then shift down by 1
        #[allow(deprecated)]
        let inclusive = self.inclusive_sum(data);
        let shifted = inclusive.get().gpu_shfl_up(1);
        // TODO: lane 0 should get `identity`, but we lack lane_id() on CPU.
        // On real GPU, use: if lane_id() == 0 { identity } else { shifted }
        let _ = identity;
        PerLane::new(shifted)
    }

    /// Reduce with addition (convenience wrapper).
    ///
    /// Equivalent to `cub::WarpReduce<T>::Sum(val)`.
    pub fn reduce_add<T>(&self, data: PerLane<T>) -> T
    where
        T: GpuValue + GpuShuffle + core::ops::Add<Output = T>,
    {
        self.reduce_sum(data).get()
    }

    /// Reduce with maximum.
    ///
    /// Equivalent to `cub::WarpReduce<T>::Reduce(val, cub::Max())`.
    pub fn reduce_max<T>(&self, data: PerLane<T>) -> T
    where
        T: GpuValue + GpuShuffle + Ord,
    {
        self.reduce(data, |a, b| if a >= b { a } else { b })
    }

    /// Reduce with minimum.
    ///
    /// Equivalent to `cub::WarpReduce<T>::Reduce(val, cub::Min())`.
    pub fn reduce_min<T>(&self, data: PerLane<T>) -> T
    where
        T: GpuValue + GpuShuffle + Ord,
    {
        self.reduce(data, |a, b| if a <= b { a } else { b })
    }

    /// Warp-level broadcast from a specific lane.
    ///
    /// Equivalent to `__shfl_sync(0xFFFFFFFF, val, src_lane)`.
    /// All lanes receive the value from `src_lane`.
    pub fn broadcast_lane<T: GpuValue + GpuShuffle>(
        &self,
        data: PerLane<T>,
        src_lane: u32,
    ) -> PerLane<T> {
        debug_assert!(src_lane < 32, "broadcast_lane: src_lane {src_lane} >= 32");
        PerLane::new(data.get().gpu_shfl_idx(src_lane))
    }

    /// Warp-level shuffle up: lane\[i\] reads from lane\[i - delta\].
    ///
    /// Useful for scan-like operations. Lanes below delta get undefined values.
    pub fn shuffle_up<T: GpuValue + GpuShuffle>(&self, data: PerLane<T>, delta: u32) -> PerLane<T> {
        PerLane::new(data.get().gpu_shfl_up(delta))
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
    fn test_reduce_custom_op() {
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(5i32);
        // Custom op: max
        let result = warp.reduce(data, |a, b| if a > b { a } else { b });
        // CPU emulation: shfl_xor returns self, so max(5, 5) = 5
        assert_eq!(result, 5);
    }

    #[test]
    fn test_reduce_add() {
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(1i32);
        let result = warp.reduce_add(data);
        // CPU emulation: each shfl_xor returns self
        // 1 + 1 = 2, 2 + 2 = 4, ..., 16 + 16 = 32
        assert_eq!(result, 32);
    }

    #[test]
    fn test_reduce_max() {
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(42i32);
        let result = warp.reduce_max(data);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_reduce_min() {
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(7i32);
        let result = warp.reduce_min(data);
        assert_eq!(result, 7);
    }

    #[test]
    #[allow(deprecated)]
    fn test_inclusive_sum() {
        // TYPE-SYSTEM TEST: validates inclusive_sum compiles. The result (32) is
        // the INCORRECT CPU identity behavior (doubling per stage), not a correct
        // prefix sum. This function is #[deprecated] and documented broken.
        // For correct scan with real lane exchange, see simwarp::hillis_steele_*.
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(1i32);
        let result = warp.inclusive_sum(data);
        assert_eq!(result.get(), 32);
    }

    #[test]
    fn test_broadcast_lane() {
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(99i32);
        let result = warp.broadcast_lane(data, 0);
        // CPU emulation: shfl_idx returns self
        assert_eq!(result.get(), 99);
    }

    #[test]
    fn test_shuffle_up() {
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(10i32);
        let result = warp.shuffle_up(data, 1);
        assert_eq!(result.get(), 10);
    }

    // Verify CUB-equivalent methods are ONLY on Warp<All>
    #[test]
    #[allow(deprecated)]
    fn test_cub_requires_all() {
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(1i32);
        // All these compile because warp is Warp<All>
        let _ = warp.reduce_add(data);
        let _ = warp.reduce_max(data);
        let _ = warp.reduce_min(data);
        let _ = warp.inclusive_sum(data);
        let _ = warp.broadcast_lane(data, 0);
        let _ = warp.shuffle_up(data, 1);
    }
}
