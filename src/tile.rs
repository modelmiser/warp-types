//! Cooperative Groups: thread block tiles with typed shuffle safety.
//!
//! NVIDIA Cooperative Groups (CUDA 9.0+) partition warps into tiles of
//! 4, 8, 16, or 32 threads. Each tile supports collective operations
//! (shuffle, ballot, reduce) within its own lanes.
//!
//! # Key Difference from Divergence
//!
//! Diverged sub-warps (`Warp<Even>`) have inactive lanes — a full-warp shuffle
//! is unavailable on them. Tiles are *partitions* of a fully active warp: a
//! `Tile` borrows `&Warp<All>`, so the warp cannot be diverged while the tile
//! lives, and a tile shuffle is lowered as a full-warp `shfl.sync` (membermask
//! `0xFFFFFFFF`) whose width operand confines the exchange to the tile.
//! **All 32 lanes of the warp must execute a tile shuffle**, not only the N
//! lanes of one tile: per-tile divergent control flow around a tile
//! collective is the raw-branch hazard listed under "Trusted entry points"
//! on `Warp::kernel_entry`, exactly as for `Warp<All>`.
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
//! `Tile<N>` has `shuffle_xor` for any N because a tile can only be made from
//! a `&Warp<All>` and keeps that borrow alive. The safety comes from the
//! full-warp borrow, not from the partition structure: the width operand
//! decides which lanes exchange, the membermask stays the whole warp.

use crate::active_set::{sealed, All};
use crate::data::PerLane;
use crate::gpu::GpuShuffle;
use crate::warp::Warp;
use crate::GpuValue;
use core::marker::PhantomData;

/// A thread block tile of `SIZE` threads.
///
/// All threads within a tile are guaranteed active — shuffle is always safe.
/// Created by partitioning a `Warp<All>` via `warp.tile::<N>()`.
///
/// The lifetime `'w` brands the tile with a borrow of the warp it was
/// partitioned from (same pattern as `GlobalRegion<'r, S>`): the warp cannot
/// be diverged (moved) while any of its tiles is live, so a tile can never
/// shuffle while its warp is diverged.
///
/// # Supported Sizes
///
/// 4, 8, 16, 32 — matching NVIDIA's cooperative groups API.
/// Only power-of-two sizes that divide 32 are valid.
#[must_use = "a Tile represents a partitioned warp — dropping it silently discards the partition"]
pub struct Tile<'w, const SIZE: usize> {
    _phantom: PhantomData<&'w ()>,
}

/// Marker trait for valid tile sizes (powers of 2 that divide 32).
///
/// Sealed — only implemented for Tile<4>, Tile<8>, Tile<16>, Tile<32>.
/// External crates cannot implement this for arbitrary sizes.
pub trait ValidTileSize: sealed::Sealed {
    /// The lanes a tile at position 0 covers. Informational: tile shuffles
    /// pass the full-warp membermask and confine the exchange with the PTX
    /// width operand, so this constant is not used as a membermask.
    const TILE_MASK: u32;
}

#[allow(private_interfaces)]
impl sealed::Sealed for Tile<'_, 4> {
    fn _sealed() -> sealed::SealToken {
        sealed::SealToken
    }
}
#[allow(private_interfaces)]
impl sealed::Sealed for Tile<'_, 8> {
    fn _sealed() -> sealed::SealToken {
        sealed::SealToken
    }
}
#[allow(private_interfaces)]
impl sealed::Sealed for Tile<'_, 16> {
    fn _sealed() -> sealed::SealToken {
        sealed::SealToken
    }
}
#[allow(private_interfaces)]
impl sealed::Sealed for Tile<'_, 32> {
    fn _sealed() -> sealed::SealToken {
        sealed::SealToken
    }
}

impl ValidTileSize for Tile<'_, 4> {
    const TILE_MASK: u32 = 0xF; // 4 lanes
}
impl ValidTileSize for Tile<'_, 8> {
    const TILE_MASK: u32 = 0xFF; // 8 lanes
}
impl ValidTileSize for Tile<'_, 16> {
    const TILE_MASK: u32 = 0xFFFF; // 16 lanes
}
impl ValidTileSize for Tile<'_, 32> {
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
    ///
    /// A tile borrows its warp — the warp cannot be diverged (moved) while
    /// a tile is still live, so a tile can never shuffle across a diverged warp:
    ///
    /// ```compile_fail
    /// use warp_types::prelude::*;
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let tile: Tile<16> = warp.tile();
    /// let (_evens, _odds) = warp.diverge_even_odd(); // ERROR: move of borrowed `warp`
    /// let _ = tile.shuffle_xor(data::PerLane::new(42i32), 1);
    /// ```
    #[inline]
    pub fn tile<'w, const SIZE: usize>(&'w self) -> Tile<'w, SIZE>
    where
        Tile<'w, SIZE>: ValidTileSize,
    {
        Tile {
            _phantom: PhantomData,
        }
    }
}

// SAFETY (every `unsafe { ….gpu_shfl_*_width(…) }` in this impl): a `Tile`
// can only be made from `&Warp<All>` (`Warp::tile`), so every lane of the warp
// is converged, which is the `GpuShuffle` contract. The witness is trusted
// ("Trusted entry points" on `Warp::kernel_entry`).
impl<'w, const SIZE: usize> Tile<'w, SIZE>
where
    Tile<'w, SIZE>: ValidTileSize,
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
    #[inline]
    pub fn shuffle_xor<T: GpuValue + GpuShuffle>(&self, data: PerLane<T>, mask: u32) -> PerLane<T> {
        assert!(
            mask < SIZE as u32,
            "shuffle_xor: mask {mask} >= tile SIZE {SIZE}"
        );
        PerLane::new(unsafe { data.get().gpu_shfl_xor_width(mask, SIZE as u32) })
    }

    /// Shuffle down within the tile (confined to tile-sized segments).
    #[inline]
    pub fn shuffle_down<T: GpuValue + GpuShuffle>(
        &self,
        data: PerLane<T>,
        delta: u32,
    ) -> PerLane<T> {
        assert!(
            delta < SIZE as u32,
            "shuffle_down: delta {delta} >= tile SIZE {SIZE}"
        );
        PerLane::new(unsafe { data.get().gpu_shfl_down_width(delta, SIZE as u32) })
    }

    /// Sum reduction across all tile lanes.
    ///
    /// Uses butterfly reduction with `log2(SIZE)` shuffle-XOR steps.
    #[inline]
    pub fn reduce_sum<T: GpuValue + GpuShuffle + core::ops::Add<Output = T>>(
        &self,
        data: PerLane<T>,
    ) -> T {
        let mut val = data.get();
        let mut stride = 1u32;
        while stride < SIZE as u32 {
            val = val + unsafe { val.gpu_shfl_xor_width(stride, SIZE as u32) };
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
    #[deprecated(
        note = "Not correct on any target — Hillis-Steele without lane_id guard. Use SimWarp for tested scan."
    )]
    #[inline]
    pub fn inclusive_sum<T: GpuValue + GpuShuffle + core::ops::Add<Output = T>>(
        &self,
        data: PerLane<T>,
    ) -> PerLane<T> {
        let mut val = data.get();
        let mut stride = 1u32;
        while stride < SIZE as u32 {
            let s = unsafe { val.gpu_shfl_up_width(stride, SIZE as u32) };
            val = val + s;
            stride *= 2;
        }
        PerLane::new(val)
    }

    /// Number of threads in this tile.
    #[inline]
    pub const fn size(&self) -> usize {
        SIZE
    }
}

// ============================================================================
// Sub-partitioning: Tile<N> → Tile<N/2>, Tile<N/4>, etc.
// ============================================================================

impl<'w> Tile<'w, 32> {
    /// Sub-partition into tiles of 16.
    #[inline]
    pub fn partition_16(&self) -> Tile<'w, 16> {
        Tile {
            _phantom: PhantomData,
        }
    }
    /// Sub-partition into tiles of 8.
    #[inline]
    pub fn partition_8(&self) -> Tile<'w, 8> {
        Tile {
            _phantom: PhantomData,
        }
    }
    /// Sub-partition into tiles of 4.
    #[inline]
    pub fn partition_4(&self) -> Tile<'w, 4> {
        Tile {
            _phantom: PhantomData,
        }
    }
}

impl<'w> Tile<'w, 16> {
    /// Sub-partition into tiles of 8.
    #[inline]
    pub fn partition_8(&self) -> Tile<'w, 8> {
        Tile {
            _phantom: PhantomData,
        }
    }
    /// Sub-partition into tiles of 4.
    #[inline]
    pub fn partition_4(&self) -> Tile<'w, 4> {
        Tile {
            _phantom: PhantomData,
        }
    }
}

impl<'w> Tile<'w, 8> {
    /// Sub-partition into tiles of 4.
    #[inline]
    pub fn partition_4(&self) -> Tile<'w, 4> {
        Tile {
            _phantom: PhantomData,
        }
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
    fn test_tile_shuffle_down_cpu_identity() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<8> = warp.tile();
        assert_eq!(tile.shuffle_down(PerLane::new(5i32), 1).get(), 5);
    }

    #[test]
    #[should_panic(expected = "shuffle_xor")]
    fn test_tile_shuffle_xor_rejects_mask_at_or_above_size() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<8> = warp.tile();
        let _ = tile.shuffle_xor(PerLane::new(1i32), 8);
    }

    #[test]
    #[should_panic(expected = "shuffle_down")]
    fn test_tile_shuffle_down_rejects_delta_at_or_above_size() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<4> = warp.tile();
        let _ = tile.shuffle_down(PerLane::new(1i32), 4);
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
    #[allow(deprecated)]
    fn test_tile_inclusive_sum() {
        let warp: Warp<All> = Warp::kernel_entry();
        let tile: Tile<8> = warp.tile();
        let data = PerLane::new(1i32);
        let result = tile.inclusive_sum(data);
        // CPU emulation: 1 + 1 = 2, 2 + 2 = 4, 4 + 4 = 8
        assert_eq!(result.get(), 8);
    }
}
