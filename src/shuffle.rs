//! Shuffle operations and permutation algebra.
//!
//! Shuffles let lanes exchange values within a warp. This module provides:
//!
//! 1. **Type-safe shuffle traits** — enforce correct return types
//!    (shuffle → PerLane, ballot → Uniform, reduce → T)
//! 2. **`Warp<All>`-restricted shuffles** — shuffle methods only on full warps
//! 3. **Permutation algebra** — XOR/Rotate/Compose with group-theoretic properties

use crate::active_set::All;
use crate::data::{LaneId, PerLane, Uniform};
use crate::warp::Warp;
use crate::GpuValue;
use core::marker::PhantomData;

/// Result of a warp ballot operation.
///
/// A ballot collects a predicate from all lanes into a bitmask.
/// The result is Uniform because every lane gets the same bitmask.
///
/// **Note:** The mask is `u32`, matching NVIDIA's `__ballot_sync` return type.
/// For AMD 64-lane wavefronts, a `BallotResult64` variant would be needed.
/// Lanes >= 32 always return `false` from `lane_voted()`.
#[derive(Clone, Copy, Debug)]
pub struct BallotResult {
    mask: Uniform<u32>,
}

impl BallotResult {
    /// Create a ballot result from a uniform mask.
    pub fn from_mask(mask: Uniform<u32>) -> Self {
        BallotResult { mask }
    }

    pub fn mask(self) -> Uniform<u32> {
        self.mask
    }

    pub fn lane_voted(self, lane: LaneId) -> Uniform<bool> {
        let id = lane.get();
        // Guard: u32 mask only covers lanes 0..31. Lanes >= 32 are outside
        // the ballot scope (would need BallotResult64 for AMD wavefronts).
        if id >= 32 {
            return Uniform::from_const(false);
        }
        Uniform::from_const((self.mask.get() & (1u32 << id)) != 0)
    }

    pub fn popcount(self) -> Uniform<u32> {
        Uniform::from_const(self.mask.get().count_ones())
    }

    pub fn first_lane(self) -> Option<LaneId> {
        let tz = self.mask.get().trailing_zeros();
        if tz < 32 {
            Some(LaneId::new(tz as u8))
        } else {
            None
        }
    }
}

// ============================================================================
// Shuffle safety marker (for error message improvement)
// ============================================================================

/// Marker trait for warp types that support shuffle operations.
///
/// Currently only `Warp<All>` implements this. `Tile<N>` has its own shuffle
/// methods (all tile lanes are active by construction) but does not implement
/// this trait. If you get an error mentioning this trait, it means you're
/// trying to shuffle on a diverged warp — merge back to `Warp<All>` first.
#[diagnostic::on_unimplemented(
    message = "shuffle requires all lanes active, but `{Self}` may have inactive lanes",
    label = "this warp may be diverged — shuffle needs Warp<All>",
    note = "after diverge_even_odd(), call merge(evens, odds) to get Warp<All> back, then shuffle"
)]
pub trait ShuffleSafe {}

impl ShuffleSafe for Warp<All> {}

// ============================================================================
// Shuffle operations restricted to Warp<All>
// ============================================================================

impl Warp<All> {
    /// Shuffle XOR: each lane exchanges with lane (id ^ mask).
    ///
    /// **ONLY AVAILABLE ON `Warp<All>`** — diverged warps cannot shuffle.
    ///
    /// On GPU: emits `shfl.sync.bfly.b32` via inline assembly.
    /// On CPU: returns the input value (single-thread identity).
    pub fn shuffle_xor<T: GpuValue + crate::gpu::GpuShuffle>(
        &self,
        data: PerLane<T>,
        mask: u32,
    ) -> PerLane<T> {
        PerLane::new(data.get().gpu_shfl_xor(mask))
    }

    /// Shuffle down: lane\[i\] reads from lane\[i+delta\].
    ///
    /// On GPU: emits `shfl.sync.down.b32`.
    /// On CPU: returns input (identity).
    pub fn shuffle_down<T: GpuValue + crate::gpu::GpuShuffle>(
        &self,
        data: PerLane<T>,
        delta: u32,
    ) -> PerLane<T> {
        PerLane::new(data.get().gpu_shfl_down(delta))
    }

    /// Sum reduction across all lanes.
    ///
    /// Returns `Uniform<T>` because a full-warp reduction produces the same
    /// result in every lane.
    ///
    /// On GPU: butterfly reduction using 5 shuffle-XOR + add steps.
    /// On CPU: returns val × 32 (butterfly doubling via identity shuffle).
    pub fn reduce_sum<T: GpuValue + crate::gpu::GpuShuffle + core::ops::Add<Output = T>>(
        &self,
        data: PerLane<T>,
    ) -> Uniform<T> {
        let mut val = data.get();
        val = val + val.gpu_shfl_xor(16);
        val = val + val.gpu_shfl_xor(8);
        val = val + val.gpu_shfl_xor(4);
        val = val + val.gpu_shfl_xor(2);
        val = val + val.gpu_shfl_xor(1);
        Uniform::from_const(val)
    }

    /// Warp ballot: collect a predicate from all lanes into a bitmask.
    ///
    /// Every lane gets the same bitmask — the result is `Uniform<u32>`.
    /// Requires `Warp<All>` because reading predicates from inactive lanes
    /// is undefined behavior.
    ///
    /// **Note:** Currently CPU-emulation only on all targets. A GPU codepath
    /// using `gpu::ballot_sync` requires `#[cfg(target_arch = "nvptx64")]` gating.
    /// On CPU: returns mask with bit 0 set if predicate is true (single-thread identity).
    pub fn ballot(&self, predicate: PerLane<bool>) -> BallotResult {
        // CPU emulation: single thread, so ballot = predicate in lane 0
        let mask = if predicate.get() { 1u32 } else { 0u32 };
        BallotResult::from_mask(Uniform::from_const(mask))
    }

    /// Broadcast: all lanes get the same value.
    pub fn broadcast<T: GpuValue>(&self, value: T) -> PerLane<T> {
        PerLane::new(value)
    }

    /// Shuffle XOR on a raw scalar — convenience that skips PerLane wrapping.
    ///
    /// Equivalent to `self.shuffle_xor(PerLane::new(val), mask).get()` but
    /// avoids the verbosity of wrapping/unwrapping for the common case.
    pub fn shuffle_xor_raw<T: GpuValue + crate::gpu::GpuShuffle>(&self, val: T, mask: u32) -> T {
        val.gpu_shfl_xor(mask)
    }

    /// Shuffle down on a raw scalar — convenience that skips PerLane wrapping.
    pub fn shuffle_down_raw<T: GpuValue + crate::gpu::GpuShuffle>(&self, val: T, delta: u32) -> T {
        val.gpu_shfl_down(delta)
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
    fn forward(i: u32) -> u32 {
        (i ^ MASK) & 0x1F
    }
    fn inverse(i: u32) -> u32 {
        (i ^ MASK) & 0x1F
    }
    fn is_self_dual() -> bool {
        true
    }
}

impl<const MASK: u32> HasDual for Xor<MASK> {
    type Dual = Xor<MASK>;
}

/// Rotate down: lane i receives from lane (i + delta) mod 32.
///
/// Consistent with CUDA `__shfl_down_sync`: data flows from higher-numbered
/// lanes to lower. `forward(i)` returns the *destination* of lane i's value
/// (lane i - delta), while `inverse(i)` returns lane i's *source* (lane i + delta).
#[derive(Copy, Clone, Debug)]
pub struct RotateDown<const DELTA: u32>;

/// Rotate up: lane i receives from lane (i - delta) mod 32.
///
/// Dual of `RotateDown`. Data flows from lower-numbered lanes to higher.
#[derive(Copy, Clone, Debug)]
pub struct RotateUp<const DELTA: u32>;

impl<const DELTA: u32> Permutation for RotateDown<DELTA> {
    fn forward(i: u32) -> u32 {
        (i + 32 - (DELTA & 0x1F)) & 0x1F
    }
    fn inverse(i: u32) -> u32 {
        (i + (DELTA & 0x1F)) & 0x1F
    }
    fn is_self_dual() -> bool {
        (DELTA & 0x1F) == 0 || (DELTA & 0x1F) == 16
    }
}

impl<const DELTA: u32> Permutation for RotateUp<DELTA> {
    fn forward(i: u32) -> u32 {
        (i + (DELTA & 0x1F)) & 0x1F
    }
    fn inverse(i: u32) -> u32 {
        (i + 32 - (DELTA & 0x1F)) & 0x1F
    }
    fn is_self_dual() -> bool {
        (DELTA & 0x1F) == 0 || (DELTA & 0x1F) == 16
    }
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
    fn forward(i: u32) -> u32 {
        i & 0x1F
    }
    fn inverse(i: u32) -> u32 {
        i & 0x1F
    }
    fn is_self_dual() -> bool {
        true
    }
}

impl HasDual for Identity {
    type Dual = Identity;
}

/// Composition of two permutations: apply P1 then P2.
#[derive(Copy, Clone, Debug)]
pub struct Compose<P1: Permutation, P2: Permutation>(PhantomData<(P1, P2)>);

impl<P1: Permutation, P2: Permutation> Permutation for Compose<P1, P2> {
    fn forward(i: u32) -> u32 {
        P2::forward(P1::forward(i))
    }
    fn inverse(i: u32) -> u32 {
        P1::inverse(P2::inverse(i))
    }
}

impl<P1: Permutation + HasDual, P2: Permutation + HasDual> HasDual for Compose<P1, P2> {
    type Dual = Compose<P2::Dual, P1::Dual>;
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
    for (i, slot) in result.iter_mut().enumerate() {
        let src = P::inverse(i as u32) as usize;
        *slot = values[src];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ballot_result_empty_mask() {
        let result = BallotResult {
            mask: Uniform::from_const(0),
        };
        assert_eq!(result.first_lane(), None);
        assert_eq!(result.popcount().get(), 0);
        assert!(!result.lane_voted(LaneId::new(0)).get());
        assert!(!result.lane_voted(LaneId::new(31)).get());
    }

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
        #[allow(clippy::approx_constant)]
        let data_f64 = PerLane::new(3.14159_f64);
        let shuffled_f64 = all.shuffle_xor(data_f64, 1);
        #[allow(clippy::approx_constant)]
        let expected_f64 = 3.14159_f64;
        assert_eq!(shuffled_f64.get(), expected_f64);

        // Reduction works on 64-bit
        let ones_i64 = PerLane::new(1_i64);
        let sum = all.reduce_sum(ones_i64);
        assert_eq!(sum.get(), 32_i64); // 1 + 1 + ... (5 XOR stages)
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
