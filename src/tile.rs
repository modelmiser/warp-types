//! Cooperative Groups: thread block tiles with typed shuffle safety.
//!
//! NVIDIA Cooperative Groups (CUDA 9.0+) partition warps into tiles of
//! 4, 8, 16, or 32 threads. Each tile supports collective operations
//! (shuffle, ballot, reduce) within its own lanes.
//!
//! # Key Difference from Divergence
//!
//! Diverged sub-warps (`Warp<Even>`) have inactive lanes — shuffle is unsafe.
//! Tiles are *partitions* — ALL threads within a tile are active by construction.
//! Shuffle within a tile is always safe because every lane participates.
//!
//! ```text
//! Warp (32 lanes)
//! ├── Tile<16> (lanes 0-15)   ← shuffle safe within tile
//! └── Tile<16> (lanes 16-31)  ← shuffle safe within tile
//!     ├── Tile<8> (lanes 16-23)
//!     └── Tile<8> (lanes 24-31)
//! ```
//!
//! # Type System Guarantee
//!
//! `Tile<N>` has `shuffle_xor` for any N — because all N lanes participate.
//! This is unlike `Warp<S>` where only `Warp<All>` has shuffle.
//! The safety comes from the partition structure, not the active set.

use core::marker::PhantomData;
use crate::GpuValue;
use crate::gpu::GpuShuffle;
use crate::data::PerLane;
use crate::warp::Warp;
use crate::active_set::{All, sealed};

/// A thread block tile of `SIZE` threads.
///
/// All threads within a tile are guaranteed active — shuffle is always safe.
/// Created by partitioning a `Warp<All>` via `warp.tile::<N>()`.
///
/// # Supported Sizes
///
/// 4, 8, 16, 32 — matching NVIDIA's cooperative groups API.
/// Only power-of-two sizes that divide 32 are valid.
pub struct Tile<const SIZE: usize> {
    _phantom: PhantomData<()>,
}

/// Marker trait for valid tile sizes (powers of 2 that divide 32).
///
/// Sealed — only implemented for Tile<4>, Tile<8>, Tile<16>, Tile<32>.
/// External crates cannot implement this for arbitrary sizes.
pub trait ValidTileSize: sealed::Sealed {
    /// Mask for this tile within a warp (based on thread position).
    const TILE_MASK: u32;
}

impl sealed::Sealed for Tile<4> {}
impl sealed::Sealed for Tile<8> {}
impl sealed::Sealed for Tile<16> {}
impl sealed::Sealed for Tile<32> {}

impl ValidTileSize for Tile<4> {
    const TILE_MASK: u32 = 0xF; // 4 lanes
}
impl ValidTileSize for Tile<8> {
    const TILE_MASK: u32 = 0xFF; // 8 lanes
}
impl ValidTileSize for Tile<16> {
    const TILE_MASK: u32 = 0xFFFF; // 16 lanes
}
impl ValidTileSize for Tile<32> {
    const TILE_MASK: u32 = 0xFFFFFFFF; // 32 lanes = full warp
}

impl Warp<All> {
    /// Partition the warp into tiles of `SIZE` threads.
    ///
    /// Equivalent to `cg::tiled_partition<SIZE>(cg::this_thread_block())`.
    ///
    /// Each thread gets a `Tile<SIZE>` representing its local tile.
    /// All tiles have exactly `SIZE` active lanes — shuffle is safe.
    ///
    /// ```
    /// use warp_types::*;
    /// use warp_types::tile::Tile;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let tile: Tile<16> = warp.tile();
    /// // tile.shuffle_xor is available — all 16 lanes participate
    /// let data = data::PerLane::new(42i32);
    /// let _partner = tile.shuffle_xor(data, 1);
    /// ```
    ///
    /// Tiles can only be created from `Warp<All>`:
    ///
    /// ```compile_fail
    /// use warp_types::prelude::*;
    /// let warp = Warp::kernel_entry();
    /// let (evens, _odds) = warp.diverge_even_odd();
    /// let _tile: Tile<16> = evens.tile(); // ERROR: method not found
    /// ```
    pub fn tile<const SIZE: usize>(&self) -> Tile<SIZE>
    where
        Tile<SIZE>: ValidTileSize,
    {
        Tile { _phantom: PhantomData }
    }
}

impl<const SIZE: usize> Tile<SIZE>
where
    Tile<SIZE>: ValidTileSize,
{
    /// Shuffle XOR within the tile.
    ///
    /// Each thread exchanges with the thread at `(thread_rank XOR mask)` within
    /// the tile. Caller must ensure mask < SIZE (no automatic clamping).
    ///
    /// **Always safe**: all `SIZE` threads in the tile participate.
    ///
    /// On GPU: emits `shfl.sync.bfly.b32` with `c = ((32-SIZE)<<8)|0x1F`,
    /// confining the shuffle to SIZE-lane segments.
    pub fn shuffle_xor<T: GpuValue + GpuShuffle>(
        &self, data: PerLane<T>, mask: u32,
    ) -> PerLane<T> {
        PerLane::new(data.get().gpu_shfl_xor_width(mask, SIZE as u32))
    }

    /// Shuffle down within the tile (confined to tile-sized segments).
    pub fn shuffle_down<T: GpuValue + GpuShuffle>(
        &self, data: PerLane<T>, delta: u32,
    ) -> PerLane<T> {
        PerLane::new(data.get().gpu_shfl_down_width(delta, SIZE as u32))
    }

    /// Sum reduction across all tile lanes.
    ///
    /// Uses butterfly reduction with `log2(SIZE)` shuffle-XOR steps.
    pub fn reduce_sum<T: GpuValue + GpuShuffle + core::ops::Add<Output = T>>(
        &self, data: PerLane<T>,
    ) -> T {
        let mut val = data.get();
        let mut stride = 1u32;
        while stride < SIZE as u32 {
            val = val + val.gpu_shfl_xor_width(stride, SIZE as u32);
            stride *= 2;
        }
        val
    }

    /// Inclusive prefix sum within the tile.
    ///
    /// **WARNING:** Not correct on any target. On CPU, `shfl_up` is identity,
    /// so each stage doubles (result: val × SIZE). On GPU, lanes where
    /// `lane_id < stride` get clamped (own value), doubling instead of
    /// preserving. Needs `if lane_id >= stride` guard (requires `lane_id()`).
    /// Retained for type-system demonstration.
    pub fn inclusive_sum<T: GpuValue + GpuShuffle + core::ops::Add<Output = T>>(
        &self, data: PerLane<T>,
    ) -> PerLane<T> {
        let mut val = data.get();
        let mut stride = 1u32;
        while stride < SIZE as u32 {
            let s = val.gpu_shfl_up_width(stride, SIZE as u32);
            val = val + s;
            stride *= 2;
        }
        PerLane::new(val)
    }

    /// Number of threads in this tile.
    pub const fn size(&self) -> usize {
        SIZE
    }
}

// ============================================================================
// Sub-partitioning: Tile<N> → Tile<N/2>, Tile<N/4>, etc.
// ============================================================================

impl Tile<32> {
    /// Sub-partition into tiles of 16.
    pub fn partition_16(&self) -> Tile<16> { Tile { _phantom: PhantomData } }
    /// Sub-partition into tiles of 8.
    pub fn partition_8(&self) -> Tile<8> { Tile { _phantom: PhantomData } }
    /// Sub-partition into tiles of 4.
    pub fn partition_4(&self) -> Tile<4> { Tile { _phantom: PhantomData } }
}

impl Tile<16> {
    /// Sub-partition into tiles of 8.
    pub fn partition_8(&self) -> Tile<8> { Tile { _phantom: PhantomData } }
    /// Sub-partition into tiles of 4.
    pub fn partition_4(&self) -> Tile<4> { Tile { _phantom: PhantomData } }
}

impl Tile<8> {
    /// Sub-partition into tiles of 4.
    pub fn partition_4(&self) -> Tile<4> { Tile { _phantom: PhantomData } }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::PerLane;

    #[test]
    fn test_tile_from_warp() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile32: Tile<32> = warp.tile();
        let tile16: Tile<16> = warp.tile();
        let tile8: Tile<8> = warp.tile();
        let tile4: Tile<4> = warp.tile();

        assert_eq!(tile32.size(), 32);
        assert_eq!(tile16.size(), 16);
        assert_eq!(tile8.size(), 8);
        assert_eq!(tile4.size(), 4);
    }

    #[test]
    fn test_tile_shuffle() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<16> = warp.tile();
        let data = PerLane::new(42i32);

        // Shuffle within tile — always safe
        let result = tile.shuffle_xor(data, 1);
        assert_eq!(result.get(), 42); // CPU identity
    }

    #[test]
    fn test_tile_reduce() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<8> = warp.tile();
        let data = PerLane::new(1i32);

        // Reduce: 1 + 1 = 2, 2 + 2 = 4, 4 + 4 = 8 (3 stages for tile<8>)
        let sum = tile.reduce_sum(data);
        assert_eq!(sum, 8);
    }

    #[test]
    fn test_tile_reduce_32() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<32> = warp.tile();
        let data = PerLane::new(1i32);
        let sum = tile.reduce_sum(data);
        assert_eq!(sum, 32);
    }

    #[test]
    fn test_tile_reduce_4() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<4> = warp.tile();
        let data = PerLane::new(1i32);
        let sum = tile.reduce_sum(data);
        assert_eq!(sum, 4);
    }

    #[test]
    fn test_tile_sub_partition() {
        let warp: Warp<All> = Warp::kernel_entry();
        let t32: Tile<32> = warp.tile();
        let t16 = t32.partition_16();
        let t8 = t16.partition_8();
        let t4 = t8.partition_4();

        assert_eq!(t16.size(), 16);
        assert_eq!(t8.size(), 8);
        assert_eq!(t4.size(), 4);
    }

    #[test]
    fn test_tile_shuffle_64bit() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<16> = warp.tile();
        let data = PerLane::new(123456789_i64);

        // 64-bit shuffle within tile — two-pass on GPU, identity on CPU
        let result = tile.shuffle_xor(data, 1);
        assert_eq!(result.get(), 123456789_i64);
    }

    #[test]
    fn test_tile_inclusive_sum() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<8> = warp.tile();
        let data = PerLane::new(1i32);
        let result = tile.inclusive_sum(data);
        // CPU emulation: 1 + 1 = 2, 2 + 2 = 4, 4 + 4 = 8
        assert_eq!(result.get(), 8);
    }
}
