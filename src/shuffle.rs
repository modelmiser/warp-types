//! Shuffle operations and permutation algebra.
//!
//! Shuffles let lanes exchange values within a warp. This module provides:
//!
//! 1. **Type-safe shuffle traits** — enforce correct return types
//!    (shuffle → PerLane, ballot → Uniform, reduce → T)
//! 2. **`Warp<All>`-restricted shuffles** — shuffle methods only on full warps
//! 3. **Permutation algebra** — XOR/Rotate/Compose with group-theoretic properties

use crate::active_set::{ActiveSet, All};
use crate::data::{LaneId, PerLane, Uniform};
use crate::gpu::GpuShuffle;
use crate::warp::Warp;
use crate::GpuValue;
use core::marker::PhantomData;

/// Result of a warp ballot operation.
///
/// A ballot collects a predicate from all lanes into a bitmask.
/// The result is Uniform because every lane gets the same bitmask.
///
/// The mask is `u64` — covers both NVIDIA 32-lane (upper 32 bits zero)
/// and AMD 64-lane wavefronts.
#[must_use = "BallotResult carries lane vote data — dropping discards the ballot"]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct BallotResult {
    mask: Uniform<u64>,
}

impl BallotResult {
    /// Create a ballot result from a uniform mask.
    pub fn from_mask(mask: Uniform<u64>) -> Self {
        BallotResult { mask }
    }

    /// Create from a 32-bit mask (NVIDIA compatibility).
    pub fn from_mask_u32(mask: Uniform<u32>) -> Self {
        BallotResult {
            mask: Uniform::from_const(mask.get() as u64),
        }
    }

    pub fn mask(self) -> Uniform<u64> {
        self.mask
    }

    /// Get the lower 32 bits (NVIDIA compatibility).
    pub fn mask_u32(self) -> Uniform<u32> {
        Uniform::from_const(self.mask.get() as u32)
    }

    pub fn lane_voted(self, lane: LaneId) -> Uniform<bool> {
        let id = lane.get();
        if id >= crate::WARP_SIZE as u8 {
            return Uniform::from_const(false);
        }
        Uniform::from_const((self.mask.get() & (1u64 << id)) != 0)
    }

    pub fn popcount(self) -> Uniform<u32> {
        Uniform::from_const(self.mask.get().count_ones())
    }

    pub fn first_lane(self) -> Option<LaneId> {
        let tz = self.mask.get().trailing_zeros();
        if tz < crate::WARP_SIZE {
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
// Shuffle XOR within a sub-warp (§3.3 SHUFFLE-WITHIN typing rule)
// ============================================================================

/// Check that an XOR shuffle mask preserves an active set.
///
/// Returns `true` if for every active lane `i` (bit set in `active_mask`),
/// lane `i ^ xor_mask` is also active. This means the XOR permutation
/// maps the active set to itself — no lane reads from an inactive partner.
///
/// The check works by computing the XOR-permuted bitmask and verifying it
/// equals the original: bit `j` of the permuted mask is set iff bit
/// `(j ^ xor_mask)` is set in the original.
#[inline]
fn xor_mask_preserves_active_set(active_mask: u64, xor_mask: u32) -> bool {
    let ws = crate::WARP_SIZE;
    let xor = xor_mask & (ws - 1); // 5 bits for 32-lane, 6 bits for 64-lane
    let mut permuted = 0u64;
    let mut j = 0u32;
    while j < ws {
        if active_mask & (1u64 << (j ^ xor)) != 0 {
            permuted |= 1u64 << j;
        }
        j += 1;
    }
    permuted == active_mask
}

impl<S: ActiveSet> Warp<S> {
    /// Shuffle XOR within a sub-warp, when the mask preserves the active set.
    ///
    /// This implements the §3.3 SHUFFLE-WITHIN typing rule: an XOR shuffle
    /// is safe on `Warp<S>` (not just `Warp<All>`) when the XOR mask maps
    /// every active lane to another active lane. Formally: for every lane `i`
    /// in `S`, lane `(i ^ mask)` is also in `S`.
    ///
    /// # Examples
    ///
    /// ```
    /// use warp_types::*;
    ///
    /// let warp: Warp<All> = Warp::kernel_entry();
    /// let (evens, odds) = warp.diverge_even_odd();
    ///
    /// let data = PerLane::new(42i32);
    ///
    /// // XOR mask 2 on Even lanes: lane 0↔2, 4↔6, etc. — stays within Even.
    /// let _shuffled = evens.shuffle_xor_within(data, 2);
    ///
    /// // XOR mask 1 would map even→odd — panics!
    /// // evens.shuffle_xor_within(data, 1); // panic
    /// #
    /// # drop(odds); // suppress must_use
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `mask` does not preserve `S`, i.e., there exists an active
    /// lane `i` where lane `(i ^ mask)` is not active.
    pub fn shuffle_xor_within<T: GpuValue + GpuShuffle>(
        &self,
        data: PerLane<T>,
        mask: u32,
    ) -> PerLane<T> {
        assert!(
            xor_mask_preserves_active_set(S::MASK, mask),
            "shuffle_xor_within: XOR mask {} does not preserve active set {} (mask={:#018X})",
            mask,
            S::NAME,
            S::MASK,
        );
        PerLane::new(data.get().gpu_shfl_xor(mask))
    }
}

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
    /// On GPU: butterfly reduction using log2(WARP_SIZE) shuffle-XOR + add steps.
    /// On CPU: returns val × WARP_SIZE (butterfly doubling via identity shuffle).
    ///
    /// **Overflow note:** GPU hardware wraps on integer overflow (two's complement,
    /// verified on RTX 4000 Ada — see `reproduce/gpu_semantics_test.cu`). This
    /// function uses `+` (Rust's `Add` trait), which panics in debug on overflow.
    /// For GPU-faithful wrapping semantics, use [`reduce_sum_wrapping`](Self::reduce_sum_wrapping).
    pub fn reduce_sum<T: GpuValue + crate::gpu::GpuShuffle + core::ops::Add<Output = T>>(
        &self,
        data: PerLane<T>,
    ) -> Uniform<T> {
        let mut val = data.get();
        #[cfg(feature = "warp64")]
        {
            val = val + val.gpu_shfl_xor(32);
        }
        val = val + val.gpu_shfl_xor(16);
        val = val + val.gpu_shfl_xor(8);
        val = val + val.gpu_shfl_xor(4);
        val = val + val.gpu_shfl_xor(2);
        val = val + val.gpu_shfl_xor(1);
        Uniform::from_const(val)
    }

    /// Wrapping butterfly reduce-sum matching GPU hardware overflow semantics.
    ///
    /// GPU integer arithmetic wraps on overflow (two's complement, no trap).
    /// This variant uses `wrapping_add` to match that behavior exactly.
    /// Hardware-verified on RTX 4000 Ada — see `reproduce/gpu_semantics_test.cu`.
    pub fn reduce_sum_wrapping_i32(&self, data: PerLane<i32>) -> Uniform<i32> {
        let mut val = data.get();
        #[cfg(feature = "warp64")]
        {
            val = val.wrapping_add(val.gpu_shfl_xor(32));
        }
        val = val.wrapping_add(val.gpu_shfl_xor(16));
        val = val.wrapping_add(val.gpu_shfl_xor(8));
        val = val.wrapping_add(val.gpu_shfl_xor(4));
        val = val.wrapping_add(val.gpu_shfl_xor(2));
        val = val.wrapping_add(val.gpu_shfl_xor(1));
        Uniform::from_const(val)
    }

    /// Wrapping reduce-sum for `u32` — GPU hardware overflow semantics.
    pub fn reduce_sum_wrapping_u32(&self, data: PerLane<u32>) -> Uniform<u32> {
        let mut val = data.get();
        #[cfg(feature = "warp64")]
        {
            val = val.wrapping_add(val.gpu_shfl_xor(32));
        }
        val = val.wrapping_add(val.gpu_shfl_xor(16));
        val = val.wrapping_add(val.gpu_shfl_xor(8));
        val = val.wrapping_add(val.gpu_shfl_xor(4));
        val = val.wrapping_add(val.gpu_shfl_xor(2));
        val = val.wrapping_add(val.gpu_shfl_xor(1));
        Uniform::from_const(val)
    }

    /// Wrapping reduce-sum for `i64` — GPU hardware overflow semantics.
    pub fn reduce_sum_wrapping_i64(&self, data: PerLane<i64>) -> Uniform<i64> {
        let mut val = data.get();
        #[cfg(feature = "warp64")]
        {
            val = val.wrapping_add(val.gpu_shfl_xor(32));
        }
        val = val.wrapping_add(val.gpu_shfl_xor(16));
        val = val.wrapping_add(val.gpu_shfl_xor(8));
        val = val.wrapping_add(val.gpu_shfl_xor(4));
        val = val.wrapping_add(val.gpu_shfl_xor(2));
        val = val.wrapping_add(val.gpu_shfl_xor(1));
        Uniform::from_const(val)
    }

    /// Wrapping reduce-sum for `u64` — GPU hardware overflow semantics.
    pub fn reduce_sum_wrapping_u64(&self, data: PerLane<u64>) -> Uniform<u64> {
        let mut val = data.get();
        #[cfg(feature = "warp64")]
        {
            val = val.wrapping_add(val.gpu_shfl_xor(32));
        }
        val = val.wrapping_add(val.gpu_shfl_xor(16));
        val = val.wrapping_add(val.gpu_shfl_xor(8));
        val = val.wrapping_add(val.gpu_shfl_xor(4));
        val = val.wrapping_add(val.gpu_shfl_xor(2));
        val = val.wrapping_add(val.gpu_shfl_xor(1));
        Uniform::from_const(val)
    }

    /// Warp ballot: collect a predicate from all lanes into a bitmask.
    ///
    /// Every lane gets the same bitmask — the result is `Uniform<u64>`.
    /// Requires `Warp<All>` because reading predicates from inactive lanes
    /// is undefined behavior.
    ///
    /// On GPU (nvptx64): calls `vote.sync.ballot.b32` via PTX inline asm.
    /// On CPU: returns mask with bit 0 set if predicate is true (single-thread identity).
    pub fn ballot(&self, predicate: PerLane<bool>) -> BallotResult {
        #[cfg(target_arch = "nvptx64")]
        {
            let mask = crate::gpu::ballot_sync(0xFFFFFFFF, predicate.get()) as u64;
            BallotResult::from_mask(Uniform::from_const(mask))
        }
        #[cfg(not(target_arch = "nvptx64"))]
        {
            // CPU emulation: single thread, so ballot = predicate in lane 0
            let mask = if predicate.get() { 1u64 } else { 0u64 };
            BallotResult::from_mask(Uniform::from_const(mask))
        }
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

/// A permutation on lane indices [0, WARP_SIZE).
pub trait Permutation: Copy + Clone {
    /// Where does lane `i` send its value?
    fn forward(i: u32) -> u32;
    /// Where does lane `i` receive from? Invariant: `inverse(forward(i)) == i`.
    fn inverse(i: u32) -> u32;
    /// Is this permutation its own inverse (involution)?
    fn is_self_dual() -> bool {
        (0..crate::WARP_SIZE).all(|i| Self::forward(i) == Self::inverse(i))
    }
}

/// The dual (inverse) of a permutation.
pub trait HasDual: Permutation {
    type Dual: Permutation;
}

/// XOR shuffle: lane i exchanges with lane i ⊕ mask.
///
/// XOR shuffles are involutions (self-dual) and form the group (Z₂)^log₂(WARP_SIZE).
#[derive(Copy, Clone, Debug)]
pub struct Xor<const MASK: u32>;

impl<const MASK: u32> Permutation for Xor<MASK> {
    fn forward(i: u32) -> u32 {
        (i ^ MASK) & (crate::WARP_SIZE - 1)
    }
    fn inverse(i: u32) -> u32 {
        (i ^ MASK) & (crate::WARP_SIZE - 1)
    }
    fn is_self_dual() -> bool {
        true
    }
}

impl<const MASK: u32> HasDual for Xor<MASK> {
    type Dual = Xor<MASK>;
}

/// Rotate down: lane i receives from lane (i + delta) mod WARP_SIZE.
///
/// **Not the same as CUDA `__shfl_down_sync`**: this is a modular rotation
/// (wraps around), whereas CUDA's `__shfl_down_sync` clamps (out-of-range
/// lanes read their own value). Data flows from higher-numbered lanes to
/// lower. `forward(i)` returns the *destination* of lane i's value
/// (lane i - delta), while `inverse(i)` returns lane i's *source* (lane i + delta).
#[derive(Copy, Clone, Debug)]
pub struct RotateDown<const DELTA: u32>;

/// Rotate up: lane i receives from lane (i - delta) mod WARP_SIZE.
///
/// Dual of `RotateDown`. Data flows from lower-numbered lanes to higher.
#[derive(Copy, Clone, Debug)]
pub struct RotateUp<const DELTA: u32>;

impl<const DELTA: u32> Permutation for RotateDown<DELTA> {
    fn forward(i: u32) -> u32 {
        let mask = crate::WARP_SIZE - 1;
        (i + crate::WARP_SIZE - (DELTA & mask)) & mask
    }
    fn inverse(i: u32) -> u32 {
        let mask = crate::WARP_SIZE - 1;
        (i + (DELTA & mask)) & mask
    }
    fn is_self_dual() -> bool {
        let mask = crate::WARP_SIZE - 1;
        (DELTA & mask) == 0 || (DELTA & mask) == crate::WARP_SIZE / 2
    }
}

impl<const DELTA: u32> Permutation for RotateUp<DELTA> {
    fn forward(i: u32) -> u32 {
        let mask = crate::WARP_SIZE - 1;
        (i + (DELTA & mask)) & mask
    }
    fn inverse(i: u32) -> u32 {
        let mask = crate::WARP_SIZE - 1;
        (i + crate::WARP_SIZE - (DELTA & mask)) & mask
    }
    fn is_self_dual() -> bool {
        let mask = crate::WARP_SIZE - 1;
        (DELTA & mask) == 0 || (DELTA & mask) == crate::WARP_SIZE / 2
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
        i & (crate::WARP_SIZE - 1)
    }
    fn inverse(i: u32) -> u32 {
        i & (crate::WARP_SIZE - 1)
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

/// 32-lane full butterfly: XOR stages 1|2|4|8|16.
#[cfg(not(feature = "warp64"))]
pub type FullButterfly = Compose<
    Compose<Compose<Compose<ButterflyStage0, ButterflyStage1>, ButterflyStage2>, ButterflyStage3>,
    ButterflyStage4,
>;

/// 64-lane butterfly adds XOR<32> as the 6th stage.
#[cfg(feature = "warp64")]
pub type ButterflyStage5 = Xor<32>;

/// 64-lane full butterfly: XOR stages 1|2|4|8|16|32.
#[cfg(feature = "warp64")]
pub type FullButterfly = Compose<
    Compose<
        Compose<
            Compose<Compose<ButterflyStage0, ButterflyStage1>, ButterflyStage2>,
            ButterflyStage3,
        >,
        ButterflyStage4,
    >,
    ButterflyStage5,
>;

/// Apply a permutation to an array of values.
#[cfg(not(feature = "warp64"))]
pub fn shuffle_by<T: Copy, P: Permutation>(values: [T; 32], _perm: P) -> [T; 32] {
    let mut result = values;
    for (i, slot) in result.iter_mut().enumerate() {
        let src = (P::inverse(i as u32) & (crate::WARP_SIZE - 1)) as usize;
        *slot = values[src];
    }
    result
}

/// Apply a permutation to an array of values.
#[cfg(feature = "warp64")]
pub fn shuffle_by<T: Copy, P: Permutation>(values: [T; 64], _perm: P) -> [T; 64] {
    let mut result = values;
    for (i, slot) in result.iter_mut().enumerate() {
        let src = (P::inverse(i as u32) & (crate::WARP_SIZE - 1)) as usize;
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
        assert_eq!(sum.get(), crate::WARP_SIZE as i64);
    }

    #[test]
    fn test_reduce_sum_wrapping_i32() {
        let all: Warp<All> = Warp::new();
        let data = PerLane::new(i32::MAX);
        let result = all.reduce_sum_wrapping_i32(data);
        let mut expected = i32::MAX;
        let stages = crate::WARP_SIZE.trailing_zeros();
        for _ in 0..stages {
            expected = expected.wrapping_add(expected);
        }
        assert_eq!(result.get(), expected);
    }

    #[test]
    fn test_reduce_sum_wrapping_u32() {
        let all: Warp<All> = Warp::new();
        let data = PerLane::new(u32::MAX);
        let result = all.reduce_sum_wrapping_u32(data);
        let mut expected = u32::MAX;
        let stages = crate::WARP_SIZE.trailing_zeros();
        for _ in 0..stages {
            expected = expected.wrapping_add(expected);
        }
        assert_eq!(result.get(), expected);
    }

    #[test]
    fn test_xor_self_dual() {
        assert!(Xor::<5>::is_self_dual());
        let ws = crate::WARP_SIZE;
        let mask_bits = ws - 1;
        for mask in 0..ws {
            for lane in 0..ws {
                let after_two = (((lane ^ mask) & mask_bits) ^ mask) & mask_bits;
                assert_eq!(after_two, lane);
            }
        }
    }

    #[test]
    fn test_rotate_duality() {
        for lane in 0..crate::WARP_SIZE {
            let down_then_up = RotateUp::<1>::forward(RotateDown::<1>::forward(lane));
            assert_eq!(down_then_up, lane);
        }
    }

    #[test]
    fn test_shuffle_roundtrip() {
        let original: [i32; crate::WARP_SIZE as usize] = core::array::from_fn(|i| i as i32);
        let shuffled = shuffle_by(original, Xor::<5>);
        let unshuffled = shuffle_by(shuffled, Xor::<5>);
        assert_eq!(unshuffled, original);
    }

    #[test]
    fn test_butterfly_permutation() {
        // Full butterfly: XOR with all stages = WARP_SIZE-1, so maps i → i ^ (WARP_SIZE-1)
        let ws = crate::WARP_SIZE;
        for i in 0..ws {
            assert_eq!(FullButterfly::forward(i), i ^ (ws - 1));
        }
    }

    #[test]
    fn test_compose_associative() {
        for i in 0..crate::WARP_SIZE {
            let ab_c = Compose::<Compose<Xor<3>, Xor<5>>, Xor<7>>::forward(i);
            let a_bc = Compose::<Xor<3>, Compose<Xor<5>, Xor<7>>>::forward(i);
            assert_eq!(ab_c, a_bc);
        }
    }

    // ========================================================================
    // shuffle_xor_within tests (§3.3 SHUFFLE-WITHIN typing rule)
    // ========================================================================

    #[test]
    fn test_xor_mask_preserves_active_set_all() {
        // All lanes active: any mask preserves the set.
        for mask in 0..crate::WARP_SIZE {
            assert!(
                xor_mask_preserves_active_set(crate::active_set::All::MASK, mask),
                "All should accept mask {mask}"
            );
        }
    }

    #[test]
    fn test_xor_mask_preserves_even() {
        use crate::active_set::Even;
        // Even: mask 2 maps 0↔2, 4↔6, etc. — stays within Even.
        assert!(xor_mask_preserves_active_set(Even::MASK, 2));
        assert!(xor_mask_preserves_active_set(Even::MASK, 4));
        assert!(xor_mask_preserves_active_set(Even::MASK, 6));
        // Even: mask 0 is identity — always preserves.
        assert!(xor_mask_preserves_active_set(Even::MASK, 0));
        // Even: mask 1 maps even→odd — does NOT preserve.
        assert!(!xor_mask_preserves_active_set(Even::MASK, 1));
        assert!(!xor_mask_preserves_active_set(Even::MASK, 3));
        assert!(!xor_mask_preserves_active_set(Even::MASK, 5));
    }

    #[test]
    fn test_xor_mask_preserves_odd() {
        use crate::active_set::Odd;
        // Odd: even masks preserve (same group structure as Even).
        assert!(xor_mask_preserves_active_set(Odd::MASK, 2));
        assert!(xor_mask_preserves_active_set(Odd::MASK, 4));
        // Odd: odd masks do NOT preserve.
        assert!(!xor_mask_preserves_active_set(Odd::MASK, 1));
        assert!(!xor_mask_preserves_active_set(Odd::MASK, 3));
    }

    #[test]
    fn test_xor_mask_preserves_low_half() {
        use crate::active_set::LowHalf;
        let half = crate::WARP_SIZE / 2;
        // LowHalf: lanes 0..half-1. Masks < half stay within LowHalf.
        for mask in 0..half {
            assert!(
                xor_mask_preserves_active_set(LowHalf::MASK, mask),
                "LowHalf should accept mask {mask}"
            );
        }
        // Mask = half maps lane 0 outside LowHalf — does NOT preserve.
        assert!(!xor_mask_preserves_active_set(LowHalf::MASK, half));
        assert!(!xor_mask_preserves_active_set(LowHalf::MASK, half + 1));
    }

    #[test]
    fn test_xor_mask_preserves_high_half() {
        use crate::active_set::HighHalf;
        let half = crate::WARP_SIZE / 2;
        // HighHalf: lanes half..WARP_SIZE-1. Masks < half stay within HighHalf.
        for mask in 0..half {
            assert!(
                xor_mask_preserves_active_set(HighHalf::MASK, mask),
                "HighHalf should accept mask {mask}"
            );
        }
        // Mask = half maps lane half→0 (outside HighHalf) — does NOT preserve.
        assert!(!xor_mask_preserves_active_set(HighHalf::MASK, half));
    }

    #[test]
    fn test_xor_mask_preserves_even_low() {
        use crate::active_set::EvenLow;
        let half = crate::WARP_SIZE / 2;
        // EvenLow: even lanes in 0..half-1. Must be even AND < half.
        assert!(xor_mask_preserves_active_set(EvenLow::MASK, 2));
        assert!(xor_mask_preserves_active_set(EvenLow::MASK, 4));
        assert!(xor_mask_preserves_active_set(EvenLow::MASK, 6));
        // Mask 1 would go even→odd — fails.
        assert!(!xor_mask_preserves_active_set(EvenLow::MASK, 1));
        // Mask = half would go low→high — fails.
        assert!(!xor_mask_preserves_active_set(EvenLow::MASK, half));
    }

    #[test]
    fn test_shuffle_xor_within_on_warp_all() {
        // Warp<All> should accept any mask via shuffle_xor_within.
        let warp: Warp<All> = Warp::new();
        let data = PerLane::new(42i32);
        for mask in [0, 1, 2, 5, 16, 31] {
            let result = warp.shuffle_xor_within(data, mask);
            // CPU identity: shuffle returns input.
            assert_eq!(result.get(), 42);
        }
    }

    #[test]
    fn test_shuffle_xor_within_on_even() {
        use crate::active_set::Even;
        let warp: Warp<Even> = Warp::new();
        let data = PerLane::new(99i32);
        // Even masks (2, 4, 6) should succeed.
        let r = warp.shuffle_xor_within(data, 2);
        assert_eq!(r.get(), 99); // CPU identity
        let r = warp.shuffle_xor_within(data, 4);
        assert_eq!(r.get(), 99);
    }

    #[test]
    #[should_panic(expected = "does not preserve active set")]
    fn test_shuffle_xor_within_even_rejects_odd_mask() {
        use crate::active_set::Even;
        let warp: Warp<Even> = Warp::new();
        let data = PerLane::new(99i32);
        // Mask 1 maps even→odd — should panic.
        let _ = warp.shuffle_xor_within(data, 1);
    }

    #[test]
    #[should_panic(expected = "does not preserve active set")]
    fn test_shuffle_xor_within_low_half_rejects_high_mask() {
        use crate::active_set::LowHalf;
        let warp: Warp<LowHalf> = Warp::new();
        let data = PerLane::new(7i32);
        // Mask = half maps low→high — should panic.
        let _ = warp.shuffle_xor_within(data, crate::WARP_SIZE / 2);
    }

    #[test]
    fn test_shuffle_xor_within_simwarp_even_mask2() {
        // Verify real lane exchange using SimWarp.
        // Even lanes: {0, 2, 4, 6, 8, ...}. XOR mask 2: 0↔2, 4↔6, 8↔10, ...
        use crate::simwarp::SimWarp;

        let sw = SimWarp::<i32>::new(|i| i as i32 * 10);
        let shuffled = sw.shuffle_xor(2);

        // Lane 0 gets lane 2's value, lane 2 gets lane 0's value.
        assert_eq!(shuffled.lane(0), 20); // was 0, now 2*10
        assert_eq!(shuffled.lane(2), 0); // was 20, now 0*10
        assert_eq!(shuffled.lane(4), 60); // was 40, now 6*10
        assert_eq!(shuffled.lane(6), 40); // was 60, now 4*10

        // Verify the preservation property: for Even lanes and mask 2,
        // every even lane's partner (lane ^ 2) is also even.
        for lane in (0..crate::WARP_SIZE).step_by(2) {
            let partner = lane ^ 2;
            assert_eq!(
                partner % 2,
                0,
                "lane {lane}'s partner {partner} should be even"
            );
        }
    }

    #[test]
    fn test_shuffle_xor_within_simwarp_odd_mask2() {
        // Odd lanes: {1, 3, 5, 7, ...}. XOR mask 2: 1↔3, 5↔7, 9↔11, ...
        use crate::simwarp::SimWarp;

        let sw = SimWarp::<i32>::new(|i| i as i32);
        let shuffled = sw.shuffle_xor(2);

        // Lane 1 gets lane 3's value, lane 3 gets lane 1's value.
        assert_eq!(shuffled.lane(1), 3);
        assert_eq!(shuffled.lane(3), 1);
        assert_eq!(shuffled.lane(5), 7);
        assert_eq!(shuffled.lane(7), 5);

        // Verify: for Odd lanes and mask 2, every odd partner is odd.
        for lane in (1..crate::WARP_SIZE).step_by(2) {
            let partner = lane ^ 2;
            assert_ne!(
                partner % 2,
                0,
                "lane {lane}'s partner {partner} should be odd"
            );
        }
    }

    #[test]
    fn test_shuffle_xor_within_simwarp_low_half() {
        // LowHalf: lanes 0..15. XOR mask 8: 0↔8, 1↔9, 2↔10, ..., 7↔15.
        use crate::simwarp::SimWarp;

        let sw = SimWarp::<i32>::new(|i| i as i32 * 3);
        let shuffled = sw.shuffle_xor(8);

        // All partners stay within 0..15.
        assert_eq!(shuffled.lane(0), 24); // lane 8's value: 8*3
        assert_eq!(shuffled.lane(8), 0); // lane 0's value: 0*3
        assert_eq!(shuffled.lane(7), 45); // lane 15's value: 15*3
        assert_eq!(shuffled.lane(15), 21); // lane 7's value: 7*3

        for lane in 0..16u32 {
            let partner = lane ^ 8;
            assert!(
                partner < 16,
                "lane {lane}'s partner {partner} should be in LowHalf"
            );
        }
    }

    #[test]
    fn test_shuffle_xor_within_after_diverge() {
        // End-to-end: diverge, shuffle within sub-warp, merge.
        let warp: Warp<All> = Warp::kernel_entry();
        let data = PerLane::new(42i32);

        let (evens, odds) = warp.diverge_even_odd();

        // Both sub-warps can shuffle with even masks.
        let _even_shuffled = evens.shuffle_xor_within(data, 2);
        let _odd_shuffled = odds.shuffle_xor_within(data, 4);

        // Merge back to All.
        let _merged: Warp<All> = crate::merge::merge(evens, odds);
    }
}
