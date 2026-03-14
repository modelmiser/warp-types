//! Platform abstraction for CPU/GPU unified targeting
//!
//! This module defines the `Platform` trait that allows the same algorithm
//! code to run on either CPU SIMD or GPU warps.
//!
//! # Key Insight
//!
//! CPU SIMD (AVX-512, NEON) and GPU warps are structurally similar:
//! - Both have "lanes" executing in lockstep
//! - Both have shuffle/permute operations
//! - Both benefit from uniform/varying tracking
//!
//! The `Platform` trait abstracts over these, enabling:
//! - Same algorithm code for both targets
//! - Fat binaries with CPU and GPU implementations
//! - Runtime dispatch based on data size or availability

use crate::GpuValue;

/// Platform-specific SIMD vector type
pub trait SimdVector<T: GpuValue>: Copy {
    /// Number of lanes in this vector
    const WIDTH: usize;

    /// Create a vector with all lanes set to the same value
    fn splat(value: T) -> Self;

    /// Extract value from a specific lane
    fn extract(self, lane: usize) -> T;

    /// Insert value into a specific lane
    fn insert(self, lane: usize, value: T) -> Self;
}

/// A computation platform (CPU SIMD or GPU warp)
///
/// This trait abstracts over the execution model, allowing algorithms
/// to be written once and compiled for multiple targets.
pub trait Platform: Copy + 'static {
    /// Number of parallel lanes
    const WIDTH: usize;

    /// Platform name for debugging
    const NAME: &'static str;

    /// The vector type for this platform
    type Vector<T: GpuValue>: SimdVector<T>;

    /// Mask type for predicated operations
    type Mask: Copy;

    // === Core Operations ===

    /// Broadcast a scalar to all lanes
    fn broadcast<T: GpuValue>(value: T) -> Self::Vector<T> {
        Self::Vector::splat(value)
    }

    /// Shuffle: each lane reads from source[indices[lane]]
    fn shuffle<T: GpuValue>(
        source: Self::Vector<T>,
        indices: Self::Vector<u32>,
    ) -> Self::Vector<T>;

    /// Shuffle down: lane i reads from lane i+delta (with wrapping or clamping)
    fn shuffle_down<T: GpuValue>(source: Self::Vector<T>, delta: usize) -> Self::Vector<T>;

    /// Shuffle XOR: lane i reads from lane i^mask
    fn shuffle_xor<T: GpuValue>(source: Self::Vector<T>, mask: usize) -> Self::Vector<T>;

    // === Reductions ===

    /// Sum all lanes, result available in all lanes (uniform)
    fn reduce_sum<T: GpuValue + core::ops::Add<Output = T>>(values: Self::Vector<T>) -> T;

    /// Maximum across all lanes
    fn reduce_max<T: GpuValue + Ord>(values: Self::Vector<T>) -> T;

    /// Minimum across all lanes
    fn reduce_min<T: GpuValue + Ord>(values: Self::Vector<T>) -> T;

    // === Predicates ===

    /// Ballot: collect per-lane bools into a mask
    fn ballot(predicates: Self::Vector<bool>) -> Self::Mask;

    /// All lanes true?
    fn all(predicates: Self::Vector<bool>) -> bool;

    /// Any lane true?
    fn any(predicates: Self::Vector<bool>) -> bool;

    /// Population count of a mask
    fn mask_popcount(mask: Self::Mask) -> u32;
}

// ============================================================================
// CPU SIMD Implementation (Portable)
// ============================================================================

/// Portable CPU SIMD platform
///
/// This uses scalar emulation for portability. In a real implementation,
/// this would use std::simd or platform-specific intrinsics (AVX-512, NEON).
#[derive(Copy, Clone, Debug)]
pub struct CpuSimd<const WIDTH: usize>;

/// Portable vector type (array-based)
#[derive(Copy, Clone, Debug)]
pub struct PortableVector<T: GpuValue, const WIDTH: usize> {
    data: [T; WIDTH],
}

impl<T: GpuValue, const WIDTH: usize> SimdVector<T> for PortableVector<T, WIDTH> {
    const WIDTH: usize = WIDTH;

    fn splat(value: T) -> Self {
        PortableVector { data: [value; WIDTH] }
    }

    fn extract(self, lane: usize) -> T {
        self.data[lane % WIDTH]
    }

    fn insert(self, lane: usize, value: T) -> Self {
        let mut result = self;
        result.data[lane % WIDTH] = value;
        result
    }
}

impl<T: GpuValue, const WIDTH: usize> Default for PortableVector<T, WIDTH> {
    fn default() -> Self {
        PortableVector { data: [T::default(); WIDTH] }
    }
}

impl<const WIDTH: usize> Platform for CpuSimd<WIDTH>
where
    [(); WIDTH]: Sized,
{
    const WIDTH: usize = WIDTH;
    const NAME: &'static str = "CpuSimd";

    type Vector<T: GpuValue> = PortableVector<T, WIDTH>;
    type Mask = u64;

    fn shuffle<T: GpuValue>(
        source: Self::Vector<T>,
        indices: Self::Vector<u32>,
    ) -> Self::Vector<T> {
        let mut result = PortableVector::default();
        for i in 0..WIDTH {
            let src_idx = indices.data[i] as usize % WIDTH;
            result.data[i] = source.data[src_idx];
        }
        result
    }

    fn shuffle_down<T: GpuValue>(source: Self::Vector<T>, delta: usize) -> Self::Vector<T> {
        let mut result = PortableVector::default();
        for i in 0..WIDTH {
            let src_idx = (i + delta) % WIDTH;
            result.data[i] = source.data[src_idx];
        }
        result
    }

    fn shuffle_xor<T: GpuValue>(source: Self::Vector<T>, mask: usize) -> Self::Vector<T> {
        let mut result = PortableVector::default();
        for i in 0..WIDTH {
            let src_idx = (i ^ mask) % WIDTH;
            result.data[i] = source.data[src_idx];
        }
        result
    }

    fn reduce_sum<T: GpuValue + core::ops::Add<Output = T>>(values: Self::Vector<T>) -> T {
        values.data.into_iter().reduce(|a, b| a + b).unwrap()
    }

    fn reduce_max<T: GpuValue + Ord>(values: Self::Vector<T>) -> T {
        values.data.into_iter().max().unwrap()
    }

    fn reduce_min<T: GpuValue + Ord>(values: Self::Vector<T>) -> T {
        values.data.into_iter().min().unwrap()
    }

    fn ballot(predicates: Self::Vector<bool>) -> Self::Mask {
        let mut mask = 0u64;
        for i in 0..WIDTH {
            if predicates.data[i] {
                mask |= 1 << i;
            }
        }
        mask
    }

    fn all(predicates: Self::Vector<bool>) -> bool {
        predicates.data.iter().all(|&b| b)
    }

    fn any(predicates: Self::Vector<bool>) -> bool {
        predicates.data.iter().any(|&b| b)
    }

    fn mask_popcount(mask: Self::Mask) -> u32 {
        mask.count_ones()
    }
}

// ============================================================================
// GPU Warp Implementation (Placeholder)
// ============================================================================

/// GPU warp platform (32 lanes for NVIDIA)
///
/// In a real implementation, this would lower to PTX intrinsics:
/// - shuffle → __shfl_sync
/// - reduce_sum → __reduce_add_sync or butterfly reduction
/// - ballot → __ballot_sync
#[derive(Copy, Clone, Debug)]
pub struct GpuWarp32;

/// GPU warp platform (64 lanes for AMD)
#[derive(Copy, Clone, Debug)]
pub struct GpuWarp64;

// For now, GPU platforms are just CPU emulation
// In a real compiler, these would emit different IR

impl Platform for GpuWarp32 {
    const WIDTH: usize = 32;
    const NAME: &'static str = "GpuWarp32";

    type Vector<T: GpuValue> = PortableVector<T, 32>;
    type Mask = u32;

    fn shuffle<T: GpuValue>(
        source: Self::Vector<T>,
        indices: Self::Vector<u32>,
    ) -> Self::Vector<T> {
        CpuSimd::<32>::shuffle(source, indices)
    }

    fn shuffle_down<T: GpuValue>(source: Self::Vector<T>, delta: usize) -> Self::Vector<T> {
        CpuSimd::<32>::shuffle_down(source, delta)
    }

    fn shuffle_xor<T: GpuValue>(source: Self::Vector<T>, mask: usize) -> Self::Vector<T> {
        CpuSimd::<32>::shuffle_xor(source, mask)
    }

    fn reduce_sum<T: GpuValue + core::ops::Add<Output = T>>(values: Self::Vector<T>) -> T {
        CpuSimd::<32>::reduce_sum(values)
    }

    fn reduce_max<T: GpuValue + Ord>(values: Self::Vector<T>) -> T {
        CpuSimd::<32>::reduce_max(values)
    }

    fn reduce_min<T: GpuValue + Ord>(values: Self::Vector<T>) -> T {
        CpuSimd::<32>::reduce_min(values)
    }

    fn ballot(predicates: Self::Vector<bool>) -> Self::Mask {
        CpuSimd::<32>::ballot(predicates) as u32
    }

    fn all(predicates: Self::Vector<bool>) -> bool {
        CpuSimd::<32>::all(predicates)
    }

    fn any(predicates: Self::Vector<bool>) -> bool {
        CpuSimd::<32>::any(predicates)
    }

    fn mask_popcount(mask: Self::Mask) -> u32 {
        mask.count_ones()
    }
}

// ============================================================================
// Generic Algorithms (using PortableVector)
// ============================================================================

/// Parallel reduction using butterfly pattern
///
/// This shows the key insight: the same algorithm works for CPU SIMD and GPU warps
/// because both support shuffle_xor. The actual implementation would be
/// specialized per platform for optimal codegen.
pub fn butterfly_reduce_sum<const WIDTH: usize, T>(
    values: PortableVector<T, WIDTH>,
) -> T
where
    T: GpuValue + core::ops::Add<Output = T>,
{
    let mut v = values;
    let mut stride = 1;
    while stride < WIDTH {
        // XOR shuffle swaps elements at distance `stride`
        let mut shuffled: PortableVector<T, WIDTH> = PortableVector::default();
        for i in 0..WIDTH {
            shuffled.data[i] = v.data[i ^ stride];
        }
        // Add corresponding elements
        for i in 0..WIDTH {
            v.data[i] = v.data[i] + shuffled.data[i];
        }
        stride *= 2;
    }
    v.data[0]
}

/// Prefix sum (inclusive scan)
pub fn prefix_sum<const WIDTH: usize, T>(
    values: PortableVector<T, WIDTH>,
) -> PortableVector<T, WIDTH>
where
    T: GpuValue + core::ops::Add<Output = T>,
{
    let mut v = values;
    let mut stride = 1;
    while stride < WIDTH {
        let mut result = v;
        for i in stride..WIDTH {
            result.data[i] = v.data[i] + v.data[i - stride];
        }
        v = result;
        stride *= 2;
    }
    v
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_simd_broadcast() {
        let v = CpuSimd::<8>::broadcast(42i32);
        for i in 0..8 {
            assert_eq!(v.extract(i), 42);
        }
    }

    #[test]
    fn test_cpu_simd_shuffle_xor() {
        // Create vector [0, 1, 2, 3, 4, 5, 6, 7]
        let mut v = PortableVector::<i32, 8>::default();
        for i in 0..8 {
            v = v.insert(i, i as i32);
        }

        // XOR with 1 swaps adjacent pairs: [1, 0, 3, 2, 5, 4, 7, 6]
        let shuffled = CpuSimd::<8>::shuffle_xor(v, 1);
        assert_eq!(shuffled.extract(0), 1);
        assert_eq!(shuffled.extract(1), 0);
        assert_eq!(shuffled.extract(2), 3);
        assert_eq!(shuffled.extract(3), 2);
    }

    #[test]
    fn test_cpu_simd_reduce_sum() {
        let mut v = PortableVector::<i32, 8>::default();
        for i in 0..8 {
            v = v.insert(i, (i + 1) as i32);
        }
        // Sum of 1..=8 = 36
        assert_eq!(CpuSimd::<8>::reduce_sum(v), 36);
    }

    #[test]
    fn test_butterfly_reduce() {
        let mut v = PortableVector::<i32, 8>::default();
        for i in 0..8 {
            v = v.insert(i, (i + 1) as i32);
        }
        let sum = butterfly_reduce_sum::<8, i32>(v);
        assert_eq!(sum, 36);
    }

    #[test]
    fn test_ballot() {
        // Odd lanes are true
        let mut predicates = PortableVector::<bool, 8>::default();
        for i in 0..8 {
            predicates = predicates.insert(i, i % 2 == 1);
        }
        let mask = CpuSimd::<8>::ballot(predicates);
        // Binary: 10101010 = 0xAA
        assert_eq!(mask, 0b10101010);
        assert_eq!(CpuSimd::<8>::mask_popcount(mask), 4);
    }

    #[test]
    fn test_gpu_warp32_emulation() {
        let v = GpuWarp32::broadcast(7i32);
        assert_eq!(v.extract(0), 7);
        assert_eq!(v.extract(31), 7);

        let mut values = PortableVector::<i32, 32>::default();
        for i in 0..32 {
            values = values.insert(i, 1);
        }
        assert_eq!(GpuWarp32::reduce_sum(values), 32);
    }

    #[test]
    fn test_prefix_sum() {
        let mut v = PortableVector::<i32, 8>::default();
        for i in 0..8 {
            v = v.insert(i, 1); // All ones
        }
        let result = prefix_sum::<8, i32>(v);
        // Should be [1, 2, 3, 4, 5, 6, 7, 8]
        for i in 0..8 {
            assert_eq!(result.extract(i), (i + 1) as i32);
        }
    }
}
