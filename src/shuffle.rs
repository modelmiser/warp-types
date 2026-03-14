//! Shuffle operations and permutation algebra.
//!
//! Shuffles let lanes exchange values within a warp. This module provides:
//!
//! 1. **Type-safe shuffle traits** — enforce correct return types
//!    (shuffle → PerLane, ballot → Uniform, reduce → SingleLane)
//! 2. **Warp<All>-restricted shuffles** — shuffle methods only on full warps
//! 3. **Permutation algebra** — XOR/Rotate/Compose with group-theoretic properties

use core::marker::PhantomData;
use crate::GpuValue;
use crate::data::{LaneId, Uniform, PerLane, SingleLane};
use crate::warp::Warp;
use crate::active_set::All;

// ============================================================================
// Shuffle traits (type-safe operation signatures)
// ============================================================================

/// Shuffle trait for warp-level data exchange.
pub trait Shuffle<T: GpuValue> {
    fn shfl_down(self, delta: u32) -> PerLane<T>;
    fn shfl_up(self, delta: u32) -> PerLane<T>;
    fn shfl_xor(self, mask: u32) -> PerLane<T>;
    fn shfl(self, source_lane: PerLane<u32>) -> PerLane<T>;
}

impl<T: GpuValue> Shuffle<T> for PerLane<T> {
    fn shfl_down(self, _delta: u32) -> PerLane<T> { self }
    fn shfl_up(self, _delta: u32) -> PerLane<T> { self }
    fn shfl_xor(self, _mask: u32) -> PerLane<T> { self }
    fn shfl(self, _source_lane: PerLane<u32>) -> PerLane<T> { self }
}

/// Result of a warp ballot operation.
///
/// A ballot collects a predicate from all lanes into a bitmask.
/// The result is Uniform because every lane gets the same bitmask.
#[derive(Clone, Copy, Debug)]
pub struct BallotResult {
    mask: Uniform<u32>,
}

impl BallotResult {
    pub fn mask(self) -> Uniform<u32> { self.mask }

    pub fn lane_voted(self, lane: LaneId) -> Uniform<bool> {
        Uniform::from_const((self.mask.get() & (1 << lane.get())) != 0)
    }

    pub fn popcount(self) -> Uniform<u32> {
        Uniform::from_const(self.mask.get().count_ones())
    }

    pub fn first_lane(self) -> Option<LaneId> {
        let tz = self.mask.get().trailing_zeros();
        if tz < 32 { Some(LaneId::new(tz as u8)) } else { None }
    }
}

/// Ballot: collect a per-lane predicate into a uniform mask.
pub trait Ballot {
    fn ballot(predicate: PerLane<bool>) -> BallotResult;
}

/// Vote operations (warp-level predicates).
pub trait Vote {
    fn all(predicate: PerLane<bool>) -> Uniform<bool>;
    fn any(predicate: PerLane<bool>) -> Uniform<bool>;
    fn unanimous(predicate: PerLane<bool>) -> Uniform<bool>;
}

/// Warp-level reduction.
pub trait Reduce<T: GpuValue> {
    fn reduce_sum(values: PerLane<T>) -> SingleLane<T, 0>;
    fn reduce_max(values: PerLane<T>) -> SingleLane<T, 0>;
    fn reduce_min(values: PerLane<T>) -> SingleLane<T, 0>;
}

// ============================================================================
// Shuffle operations restricted to Warp<All>
// ============================================================================

impl Warp<All> {
    /// Shuffle XOR: each lane exchanges with lane (id ^ mask).
    ///
    /// **ONLY AVAILABLE ON Warp<All>** — diverged warps cannot shuffle.
    ///
    /// On GPU: emits `shfl.sync.bfly.b32` via inline assembly.
    /// On CPU: returns the input value (single-thread identity).
    pub fn shuffle_xor<T: GpuValue + crate::gpu::GpuShuffle>(
        &self, data: PerLane<T>, mask: u32,
    ) -> PerLane<T> {
        PerLane::new(data.get().gpu_shfl_xor(mask))
    }

    /// Shuffle down: lane[i] reads from lane[i+delta].
    ///
    /// On GPU: emits `shfl.sync.down.b32`.
    /// On CPU: returns input (identity).
    pub fn shuffle_down<T: GpuValue + crate::gpu::GpuShuffle>(
        &self, data: PerLane<T>, delta: u32,
    ) -> PerLane<T> {
        PerLane::new(data.get().gpu_shfl_down(delta))
    }

    /// Sum reduction across all lanes.
    ///
    /// On GPU: butterfly reduction using 5 shuffle-XOR + add steps.
    /// On CPU: returns the single thread's value.
    pub fn reduce_sum<T: GpuValue + crate::gpu::GpuShuffle + core::ops::Add<Output = T>>(
        &self, data: PerLane<T>,
    ) -> T {
        let mut val = data.get();
        val = val + val.gpu_shfl_xor(16);
        val = val + val.gpu_shfl_xor(8);
        val = val + val.gpu_shfl_xor(4);
        val = val + val.gpu_shfl_xor(2);
        val = val + val.gpu_shfl_xor(1);
        val
    }

    /// Broadcast: all lanes get the same value.
    pub fn broadcast<T: GpuValue>(&self, value: T) -> PerLane<T> {
        PerLane::new(value)
    }
}

// ============================================================================
// Permutation algebra (from shuffle duality research)
// ============================================================================

/// A permutation on lane indices [0, 32).
pub trait Permutation: Copy + Clone {
    /// Where does lane `i` send its value?
    fn forward(i: u32) -> u32;
    /// Where does lane `i` receive from? Invariant: `inverse(forward(i)) == i`.
    fn inverse(i: u32) -> u32;
    /// Is this permutation its own inverse (involution)?
    fn is_self_dual() -> bool {
        (0..32).all(|i| Self::forward(i) == Self::inverse(i))
    }
}

/// The dual (inverse) of a permutation.
pub trait HasDual: Permutation {
    type Dual: Permutation;
}

/// XOR shuffle: lane i exchanges with lane i ⊕ mask.
///
/// XOR shuffles are involutions (self-dual) and form the group (Z₂)⁵.
#[derive(Copy, Clone, Debug)]
pub struct Xor<const MASK: u32>;

impl<const MASK: u32> Permutation for Xor<MASK> {
    fn forward(i: u32) -> u32 { (i ^ MASK) & 0x1F }
    fn inverse(i: u32) -> u32 { (i ^ MASK) & 0x1F }
    fn is_self_dual() -> bool { true }
}

impl<const MASK: u32> HasDual for Xor<MASK> {
    type Dual = Xor<MASK>;
}

/// Rotate down: lane i receives from lane (i + delta) mod 32.
#[derive(Copy, Clone, Debug)]
pub struct RotateDown<const DELTA: u32>;

/// Rotate up: lane i receives from lane (i - delta) mod 32.
#[derive(Copy, Clone, Debug)]
pub struct RotateUp<const DELTA: u32>;

impl<const DELTA: u32> Permutation for RotateDown<DELTA> {
    fn forward(i: u32) -> u32 { (i + 32 - (DELTA & 0x1F)) & 0x1F }
    fn inverse(i: u32) -> u32 { (i + (DELTA & 0x1F)) & 0x1F }
    fn is_self_dual() -> bool { DELTA == 0 || DELTA == 16 }
}

impl<const DELTA: u32> Permutation for RotateUp<DELTA> {
    fn forward(i: u32) -> u32 { (i + (DELTA & 0x1F)) & 0x1F }
    fn inverse(i: u32) -> u32 { (i + 32 - (DELTA & 0x1F)) & 0x1F }
    fn is_self_dual() -> bool { DELTA == 0 || DELTA == 16 }
}

impl<const DELTA: u32> HasDual for RotateDown<DELTA> {
    type Dual = RotateUp<DELTA>;
}

impl<const DELTA: u32> HasDual for RotateUp<DELTA> {
    type Dual = RotateDown<DELTA>;
}

/// Identity permutation.
#[derive(Copy, Clone, Debug)]
pub struct Identity;

impl Permutation for Identity {
    fn forward(i: u32) -> u32 { i }
    fn inverse(i: u32) -> u32 { i }
    fn is_self_dual() -> bool { true }
}

impl HasDual for Identity {
    type Dual = Identity;
}

/// Composition of two permutations: apply P1 then P2.
#[derive(Copy, Clone, Debug)]
pub struct Compose<P1: Permutation, P2: Permutation>(PhantomData<(P1, P2)>);

impl<P1: Permutation, P2: Permutation> Permutation for Compose<P1, P2> {
    fn forward(i: u32) -> u32 { P2::forward(P1::forward(i)) }
    fn inverse(i: u32) -> u32 { P1::inverse(P2::inverse(i)) }
}

// Butterfly network type aliases
pub type ButterflyStage0 = Xor<1>;
pub type ButterflyStage1 = Xor<2>;
pub type ButterflyStage2 = Xor<4>;
pub type ButterflyStage3 = Xor<8>;
pub type ButterflyStage4 = Xor<16>;

pub type FullButterfly = Compose<
    Compose<Compose<Compose<ButterflyStage0, ButterflyStage1>, ButterflyStage2>, ButterflyStage3>,
    ButterflyStage4,
>;

/// Apply a permutation to an array of values.
pub fn shuffle_by<T: Copy, P: Permutation>(values: [T; 32], _perm: P) -> [T; 32] {
    let mut result = values;
    for i in 0..32 {
        let src = P::inverse(i as u32) as usize;
        result[i] = values[src];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ballot_result() {
        let result = BallotResult {
            mask: Uniform::from_const(0b1010_1010),
        };
        assert!(!result.lane_voted(LaneId::new(0)).get());
        assert!(result.lane_voted(LaneId::new(1)).get());
        assert_eq!(result.popcount().get(), 4);
        assert_eq!(result.first_lane(), Some(LaneId::new(1)));
    }

    #[test]
    fn test_shuffle_only_on_all() {
        let all: Warp<All> = Warp::new();
        let data = PerLane::new(42i32);
        let _shuffled = all.shuffle_xor(data, 1);
        let _reduced = all.reduce_sum(data);
    }

    #[test]
    fn test_shuffle_64bit_types() {
        let all: Warp<All> = Warp::new();

        // i64: two-pass shuffle on GPU, identity on CPU
        let data_i64 = PerLane::new(0x0000_0001_0000_0002_i64);
        let shuffled_i64 = all.shuffle_xor(data_i64, 1);
        assert_eq!(shuffled_i64.get(), 0x0000_0001_0000_0002_i64);

        // u64
        let data_u64 = PerLane::new(u64::MAX);
        let shuffled_u64 = all.shuffle_xor(data_u64, 1);
        assert_eq!(shuffled_u64.get(), u64::MAX);

        // f64: bit-preserving two-pass
        let data_f64 = PerLane::new(3.14159_f64);
        let shuffled_f64 = all.shuffle_xor(data_f64, 1);
        assert_eq!(shuffled_f64.get(), 3.14159_f64);

        // Reduction works on 64-bit
        let ones_i64 = PerLane::new(1_i64);
        let sum = all.reduce_sum(ones_i64);
        assert_eq!(sum, 32_i64); // 1 + 1 + ... (5 XOR stages)
    }

    #[test]
    fn test_xor_self_dual() {
        assert!(Xor::<5>::is_self_dual());
        for mask in 0..32u32 {
            for lane in 0..32u32 {
                let after_two = (((lane ^ mask) & 0x1F) ^ mask) & 0x1F;
                assert_eq!(after_two, lane);
            }
        }
    }

    #[test]
    fn test_rotate_duality() {
        for lane in 0..32u32 {
            let down_then_up = RotateUp::<1>::forward(RotateDown::<1>::forward(lane));
            assert_eq!(down_then_up, lane);
        }
    }

    #[test]
    fn test_shuffle_roundtrip() {
        let original: [i32; 32] = core::array::from_fn(|i| i as i32);
        let shuffled = shuffle_by(original, Xor::<5>);
        let unshuffled = shuffle_by(shuffled, Xor::<5>);
        assert_eq!(unshuffled, original);
    }

    #[test]
    fn test_butterfly_permutation() {
        // Full butterfly: XOR with 1|2|4|8|16 = 31, so maps i → i ^ 31
        for i in 0..32u32 {
            assert_eq!(FullButterfly::forward(i), i ^ 31);
        }
    }

    #[test]
    fn test_compose_associative() {
        for i in 0..32u32 {
            let ab_c = Compose::<Compose<Xor<3>, Xor<5>>, Xor<7>>::forward(i);
            let a_bc = Compose::<Xor<3>, Compose<Xor<5>, Xor<7>>>::forward(i);
            assert_eq!(ab_c, a_bc);
        }
    }
}
