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
//! | `cub::WarpScan<T>`       | `warp.inclusive_sum(data)`, `warp.exclusive_sum(data, id)` |
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

// SAFETY (every `unsafe { ….gpu_shfl_*(…) }` in this impl): `&self` is a
// `Warp<All>`, the typed witness that every lane is converged, which is the
// `GpuShuffle` contract. The witness is trusted ("Trusted entry points" on
// `Warp::kernel_entry`).
impl Warp<All> {
    /// Reduce with a custom binary operator across all lanes.
    ///
    /// Equivalent to `cub::WarpReduce<T>::Reduce(val, op)`.
    /// Uses butterfly reduction (log₂(WARP_SIZE) stages).
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
    #[inline]
    pub fn reduce<T, F>(&self, data: PerLane<T>, op: F) -> T
    where
        T: GpuValue + GpuShuffle,
        F: Fn(T, T) -> T,
    {
        let mut val = data.get();
        #[cfg(feature = "warp64")]
        {
            val = op(val, unsafe { val.gpu_shfl_xor(32) });
        }
        val = op(val, unsafe { val.gpu_shfl_xor(16) });
        val = op(val, unsafe { val.gpu_shfl_xor(8) });
        val = op(val, unsafe { val.gpu_shfl_xor(4) });
        val = op(val, unsafe { val.gpu_shfl_xor(2) });
        val = op(val, unsafe { val.gpu_shfl_xor(1) });
        val
    }

    /// Inclusive prefix sum across all lanes: lane `i` gets the sum of lanes
    /// `0..=i`.
    ///
    /// Equivalent to `cub::WarpScan<T>::InclusiveSum(val, &sum)`.
    /// Hillis–Steele on `shuffle_up`, log₂(WARP_SIZE) stages, WITH the lane
    /// guard: every lane issues every shuffle (they are collective), and a lane
    /// adds the value it received only when `lane_id() >= stride` — below
    /// that, `shfl.sync.up` hands a lane its own value back, which an
    /// unguarded scan (this method through 0.4.3) added, doubling those lanes.
    ///
    /// On the CPU path the calling thread is lane 0 (`gpu::lane_id()` returns
    /// 0), so the result is the lane's own value — lane 0's inclusive sum.
    /// Not available on `amdgpu`, which has no `lane_id` yet.
    ///
    /// ```
    /// use warp_types::*;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let prefix = warp.inclusive_sum(data::PerLane::new(5i32));
    /// assert_eq!(prefix.get(), 5); // CPU: lane 0 sums lanes 0..=0
    /// ```
    #[cfg(not(target_arch = "amdgpu"))]
    #[inline]
    pub fn inclusive_sum<T>(&self, data: PerLane<T>) -> PerLane<T>
    where
        T: GpuValue + GpuShuffle + core::ops::Add<Output = T>,
    {
        let lane = crate::gpu::lane_id();
        let mut val = data.get();
        let mut stride = 1u32;
        while stride < crate::WARP_SIZE {
            let s = unsafe { val.gpu_shfl_up(stride) };
            if lane >= stride {
                val = val + s;
            }
            stride *= 2;
        }
        PerLane::new(val)
    }

    /// Exclusive prefix sum across all lanes: lane `i` gets the sum of lanes
    /// `0..i`, and lane 0 gets `identity`.
    ///
    /// Equivalent to `cub::WarpScan<T>::ExclusiveSum(val, &sum)` with an
    /// explicit identity. The inclusive scan shifted up one lane; lane 0,
    /// whose `shfl.sync.up` returns its own value, takes `identity` instead.
    ///
    /// On the CPU path the calling thread is lane 0, so the result is
    /// `identity`. Not available on `amdgpu`, which has no `lane_id` yet.
    ///
    /// ```
    /// use warp_types::*;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let prefix = warp.exclusive_sum(data::PerLane::new(5i32), 0);
    /// assert_eq!(prefix.get(), 0); // CPU: lane 0 gets the identity
    /// ```
    #[cfg(not(target_arch = "amdgpu"))]
    #[inline]
    pub fn exclusive_sum<T>(&self, data: PerLane<T>, identity: T) -> PerLane<T>
    where
        T: GpuValue + GpuShuffle + core::ops::Add<Output = T>,
    {
        let inclusive = self.inclusive_sum(data).get();
        let shifted = unsafe { inclusive.gpu_shfl_up(1) };
        PerLane::new(if crate::gpu::lane_id() == 0 {
            identity
        } else {
            shifted
        })
    }

    /// Reduce with addition (convenience wrapper).
    ///
    /// Equivalent to `cub::WarpReduce<T>::Sum(val)`.
    #[inline]
    pub fn reduce_add<T>(&self, data: PerLane<T>) -> T
    where
        T: GpuValue + GpuShuffle + core::ops::Add<Output = T>,
    {
        self.reduce_sum(data).get()
    }

    /// Reduce with maximum.
    ///
    /// Equivalent to `cub::WarpReduce<T>::Reduce(val, cub::Max())`.
    #[inline]
    pub fn reduce_max<T>(&self, data: PerLane<T>) -> T
    where
        T: GpuValue + GpuShuffle + Ord,
    {
        self.reduce(data, |a, b| if a >= b { a } else { b })
    }

    /// Reduce with minimum.
    ///
    /// Equivalent to `cub::WarpReduce<T>::Reduce(val, cub::Min())`.
    #[inline]
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
    ///
    /// # Panics
    ///
    /// If `src_lane >= 32`. **In a device kernel, keep `src_lane` provably in
    /// range** (`lane & 31`, a constant, or a value the compiler can bound):
    /// the check below is an `assert!`, so when it cannot be proven dead the
    /// panic path stays in the PTX and pulls `core::panicking::panic_fmt` and
    /// `Display::fmt` in as undefined externs — and a module with undefined
    /// externs fails to *load*, before any kernel runs. A computed lane such
    /// as `ballot.trailing_zeros()` (32 when the ballot is zero) is the usual
    /// way in. Found on an H200 on 2026-11-02; see
    /// `research/gpu-lrat-checker.md`.
    #[inline]
    pub fn broadcast_lane<T: GpuValue + GpuShuffle>(
        &self,
        data: PerLane<T>,
        src_lane: u32,
    ) -> PerLane<T> {
        assert!(
            src_lane < crate::WARP_SIZE,
            "broadcast_lane: src_lane {src_lane} >= {}",
            crate::WARP_SIZE
        );
        PerLane::new(unsafe { data.get().gpu_shfl_idx(src_lane) })
    }

    /// Warp-level shuffle up: lane\[i\] reads from lane\[i - delta\].
    ///
    /// Useful for scan-like operations. Lanes below delta get undefined values.
    #[inline]
    pub fn shuffle_up<T: GpuValue + GpuShuffle>(&self, data: PerLane<T>, delta: u32) -> PerLane<T> {
        PerLane::new(unsafe { data.get().gpu_shfl_up(delta) })
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
        // CPU emulation: each shfl_xor returns self, so value doubles per stage
        // 5 stages (32-lane): 1→2→4→8→16→32. 6 stages (64-lane): →64.
        assert_eq!(result, crate::WARP_SIZE as i32);
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
    fn test_inclusive_sum_cpu_is_lane_zero() {
        // CPU path: the thread is lane 0, so its inclusive sum is its own
        // value (the unguarded scan through 0.4.3 returned 32 here). The
        // multi-lane result is checked on the device: examples/gpu-project
        // Test 9.
        let warp: Warp<All> = Warp::kernel_entry();
        assert_eq!(warp.inclusive_sum(PerLane::new(1i32)).get(), 1);
        assert_eq!(warp.inclusive_sum(PerLane::new(7u64)).get(), 7);
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

    #[test]
    fn test_exclusive_sum_cpu_is_identity() {
        // CPU path: lane 0 gets the identity (the unguarded scan through 0.4.3
        // returned 3 * 32 here and ignored `identity`).
        let warp: Warp<All> = Warp::kernel_entry();
        assert_eq!(warp.exclusive_sum(PerLane::new(3i32), 0).get(), 0);
        assert_eq!(warp.exclusive_sum(PerLane::new(3i32), -9).get(), -9);
    }

    #[test]
    #[should_panic(expected = "broadcast_lane")]
    fn test_broadcast_lane_rejects_out_of_range_lane() {
        let warp: Warp<All> = Warp::kernel_entry();
        let _ = warp.broadcast_lane(PerLane::new(1i32), crate::WARP_SIZE);
    }

    // Verify CUB-equivalent methods are ONLY on Warp<All>
    #[test]
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
