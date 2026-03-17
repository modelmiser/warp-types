//! Shuffle Duality: Permutation-Group View of Warp Shuffles
//!
//! **STATUS: Superseded** — Promoted to `src/shuffle.rs` (Permutation algebra). Retained as research artifact.
//!
//! Key insight: Shuffle duality isn't session-type duality (send/recv).
//! It's PERMUTATION duality (forward/inverse).
//!
//! In session types: Alice sends, Bob receives (role asymmetry)
//! In SIMT shuffles: All lanes do same thing (no role asymmetry)
//!
//! The duality lives in the PERMUTATION STRUCTURE:
//! - shuffle(P) then shuffle(P⁻¹) = identity
//! - XOR shuffles are self-dual (involutions)
//! - Down/Up shuffles are mutual duals
//!
//! This connects to GROUP THEORY, not session type duality.

use std::marker::PhantomData;

// ============================================================================
// CORE TRAIT: Permutation
// ============================================================================

/// A permutation on lane indices [0, 32)
pub trait Permutation: Copy + Clone {
    /// Apply permutation: where does lane `i` send its value?
    fn forward(i: u32) -> u32;

    /// Inverse permutation: where does lane `i` receive from?
    ///
    /// Invariant: inverse(forward(i)) == i for all i
    fn inverse(i: u32) -> u32;

    /// Is this permutation its own inverse? (involution)
    fn is_self_dual() -> bool {
        // Default: check if forward == inverse for all lanes
        // Override for compile-time knowledge
        (0..32).all(|i| Self::forward(i) == Self::inverse(i))
    }
}

/// The dual (inverse) of a permutation
pub trait HasDual: Permutation {
    type Dual: Permutation;
}

// ============================================================================
// XOR SHUFFLE: Self-Dual Involutions
// ============================================================================

/// XOR shuffle: lane i exchanges with lane i ⊕ mask
///
/// Mathematical properties:
/// - Involution: XOR(m) ∘ XOR(m) = Identity
/// - Abelian group: XOR(a) ∘ XOR(b) = XOR(a ⊕ b) = XOR(b) ∘ XOR(a)
/// - Identity element: XOR(0)
///
/// These form the group (Z₂)⁵ under XOR.
#[derive(Copy, Clone, Debug)]
pub struct Xor<const MASK: u32>;

impl<const MASK: u32> Permutation for Xor<MASK> {
    fn forward(i: u32) -> u32 {
        (i ^ MASK) & 0x1F // Keep in [0, 32)
    }

    fn inverse(i: u32) -> u32 {
        // XOR is self-inverse!
        (i ^ MASK) & 0x1F
    }

    fn is_self_dual() -> bool {
        true // All XOR shuffles are involutions
    }
}

impl<const MASK: u32> HasDual for Xor<MASK> {
    type Dual = Xor<MASK>; // Self-dual!
}

// ============================================================================
// ROTATE SHUFFLE: Down/Up Duality
// ============================================================================

/// Rotate down: lane i receives from lane (i + delta) mod 32
///
/// Lane i's value goes to lane (i - delta) mod 32
#[derive(Copy, Clone, Debug)]
pub struct RotateDown<const DELTA: u32>;

/// Rotate up: lane i receives from lane (i - delta) mod 32
///
/// Lane i's value goes to lane (i + delta) mod 32
#[derive(Copy, Clone, Debug)]
pub struct RotateUp<const DELTA: u32>;

impl<const DELTA: u32> Permutation for RotateDown<DELTA> {
    fn forward(i: u32) -> u32 {
        // I send to lane (i - delta)
        (i + 32 - (DELTA & 0x1F)) & 0x1F
    }

    fn inverse(i: u32) -> u32 {
        // I receive from lane (i + delta)
        (i + (DELTA & 0x1F)) & 0x1F
    }

    fn is_self_dual() -> bool {
        DELTA == 0 || DELTA == 16 // Only 0 and half-rotation are self-dual
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
        DELTA == 0 || DELTA == 16
    }
}

impl<const DELTA: u32> HasDual for RotateDown<DELTA> {
    type Dual = RotateUp<DELTA>; // Down and Up are duals
}

impl<const DELTA: u32> HasDual for RotateUp<DELTA> {
    type Dual = RotateDown<DELTA>;
}

// ============================================================================
// IDENTITY: Trivial Self-Dual
// ============================================================================

/// Identity permutation: no shuffle, everyone keeps their value
#[derive(Copy, Clone, Debug)]
pub struct Identity;

impl Permutation for Identity {
    fn forward(i: u32) -> u32 {
        i
    }
    fn inverse(i: u32) -> u32 {
        i
    }
    fn is_self_dual() -> bool {
        true
    }
}

impl HasDual for Identity {
    type Dual = Identity;
}

// ============================================================================
// COMPOSITION: Permutation Group Structure
// ============================================================================

/// Composition of two permutations: first P1, then P2
///
/// (P2 ∘ P1)(i) = P2(P1(i))
///
/// This forms a GROUP:
/// - Associative: (P3 ∘ P2) ∘ P1 = P3 ∘ (P2 ∘ P1)
/// - Identity: Id ∘ P = P ∘ Id = P
/// - Inverse: P ∘ P⁻¹ = P⁻¹ ∘ P = Id
#[derive(Copy, Clone, Debug)]
pub struct Compose<P1: Permutation, P2: Permutation>(PhantomData<(P1, P2)>);

impl<P1: Permutation, P2: Permutation> Permutation for Compose<P1, P2> {
    fn forward(i: u32) -> u32 {
        P2::forward(P1::forward(i))
    }

    fn inverse(i: u32) -> u32 {
        // (P2 ∘ P1)⁻¹ = P1⁻¹ ∘ P2⁻¹
        P1::inverse(P2::inverse(i))
    }
}

// ============================================================================
// BUTTERFLY NETWORK: Composed XOR Shuffles
// ============================================================================

/// Butterfly network stage: XOR with 2^stage
///
/// Full butterfly = Xor<1> ∘ Xor<2> ∘ Xor<4> ∘ Xor<8> ∘ Xor<16>
///
/// This is used in parallel prefix/reduction algorithms.
/// Each stage is self-dual, but the composition is NOT self-dual
/// (unless you reverse the order).
pub type ButterflyStage0 = Xor<1>;
pub type ButterflyStage1 = Xor<2>;
pub type ButterflyStage2 = Xor<4>;
pub type ButterflyStage3 = Xor<8>;
pub type ButterflyStage4 = Xor<16>;

/// Full butterfly: all 5 stages composed
pub type FullButterfly = Compose<
    Compose<Compose<Compose<ButterflyStage0, ButterflyStage1>, ButterflyStage2>, ButterflyStage3>,
    ButterflyStage4,
>;

/// Inverse butterfly: reverse order (since each stage is self-dual)
pub type InverseButterfly = Compose<
    Compose<Compose<Compose<ButterflyStage4, ButterflyStage3>, ButterflyStage2>, ButterflyStage1>,
    ButterflyStage0,
>;

// ============================================================================
// SHUFFLE OPERATION WITH TYPE-LEVEL PERMUTATION
// ============================================================================

/// A value that has been shuffled by permutation P
///
/// Tracks the permutation in the type system.
/// To "unshuffle", apply the dual permutation.
#[derive(Copy, Clone, Debug)]
pub struct Shuffled<T, P: Permutation> {
    pub data: T,
    _perm: PhantomData<P>,
}

impl<T, P: Permutation> Shuffled<T, P> {
    pub fn new(data: T) -> Self {
        Shuffled {
            data,
            _perm: PhantomData,
        }
    }
}

/// Shuffle by permutation P
pub fn shuffle<T: Copy, P: Permutation>(values: [T; 32], _perm: P) -> Shuffled<[T; 32], P> {
    let mut result = values;
    for i in 0..32 {
        // Lane i receives from lane P.inverse(i)
        let src = P::inverse(i as u32) as usize;
        result[i] = values[src];
    }
    Shuffled::new(result)
}

/// Unshuffle: apply dual permutation to recover original order
pub fn unshuffle<T: Copy, P: HasDual>(shuffled: Shuffled<[T; 32], P>) -> Shuffled<[T; 32], P::Dual>
where
    P::Dual: Permutation,
{
    let mut result = shuffled.data;
    for i in 0..32 {
        let src = <P::Dual as Permutation>::inverse(i as u32) as usize;
        result[i] = shuffled.data[src];
    }
    Shuffled::new(result)
}

/// For self-dual permutations, shuffle and unshuffle are the same
pub fn shuffle_involution<T: Copy, P: Permutation + HasDual<Dual = P>>(
    values: [T; 32],
    _perm: P,
) -> [T; 32] {
    let mut result = values;
    for i in 0..32 {
        let src = P::inverse(i as u32) as usize;
        result[i] = values[src];
    }
    result
}

// ============================================================================
// LANE-LOCAL VIEW: What Each Lane Sees
// ============================================================================

/// Each lane's view of a shuffle operation
///
/// In SIMT, all lanes execute the same instruction, but each lane
/// has a different SOURCE (where it receives from) and DESTINATION
/// (where it sends to).
#[derive(Copy, Clone, Debug)]
pub struct LaneView {
    pub my_lane: u32,
    pub i_send_to: u32,
    pub i_receive_from: u32,
}

impl LaneView {
    pub fn for_lane<P: Permutation>(lane: u32) -> Self {
        LaneView {
            my_lane: lane,
            i_send_to: P::forward(lane),
            i_receive_from: P::inverse(lane),
        }
    }

    /// Is this a "symmetric" exchange? (send and receive same partner)
    pub fn is_symmetric(&self) -> bool {
        self.i_send_to == self.i_receive_from
    }
}

/// For self-dual permutations, every lane has symmetric exchange
pub fn all_symmetric<P: Permutation>() -> bool {
    (0..32).all(|i| {
        let view = LaneView::for_lane::<P>(i);
        view.is_symmetric()
    })
}

// ============================================================================
// CONNECTION TO SESSION TYPES
// ============================================================================

/// Session type interpretation of shuffles
///
/// Classical: Alice: !T.end,  Bob: ?T.end  (asymmetric roles)
///
/// SIMT: All lanes: Exchange<P, T>.end
///       where Exchange<P, T> means "send to P(me), receive from P⁻¹(me)"
///
/// The "duality" is in the permutation, not the session structure.
/// All lanes have the SAME session type, parameterized by lane ID.
pub mod session_view {
    use super::*;

    /// A session type for lane `i` under permutation `P`
    ///
    /// Semantics:
    /// - Send my value to lane P.forward(i)
    /// - Receive value from lane P.inverse(i)
    /// - Both happen "simultaneously" (single shuffle instruction)
    #[derive(Copy, Clone, Debug)]
    pub struct Exchange<P: Permutation, T> {
        _marker: PhantomData<(P, T)>,
    }

    /// The "projection" of a warp shuffle onto a single lane
    ///
    /// In MPST terms: global protocol is "everyone shuffles by P"
    /// Each lane's projection is "I exchange with my P-partners"
    pub fn project_to_lane<P: Permutation>(lane: u32) -> LaneView {
        LaneView::for_lane::<P>(lane)
    }

    /// Key insight: Unlike classical session types, all projections
    /// have the SAME STRUCTURE. The difference is only in which
    /// concrete lanes are partners.
    ///
    /// This is a "parameterized" or "indexed" session type:
    /// Session(i) = Exchange<P, T> for all i, but partners differ.
    pub fn projections_isomorphic<P: Permutation>() -> bool {
        // All lanes do: send to one lane, receive from one lane
        // Structure is identical, only partner IDs differ
        true
    }
}

// ============================================================================
// ALGEBRAIC PROPERTIES
// ============================================================================

pub mod algebra {

    /// XOR shuffles form an abelian group isomorphic to (Z₂)⁵
    ///
    /// - Closure: Xor<a> ∘ Xor<b> = Xor<a^b>
    /// - Associativity: (Xor<a> ∘ Xor<b>) ∘ Xor<c> = Xor<a> ∘ (Xor<b> ∘ Xor<c>)
    /// - Identity: Xor<0>
    /// - Inverse: Xor<a>⁻¹ = Xor<a> (self-inverse)
    /// - Commutativity: Xor<a> ∘ Xor<b> = Xor<b> ∘ Xor<a>
    pub fn xor_group_composition(a: u32, b: u32) -> u32 {
        a ^ b
    }

    /// Rotations form a cyclic group Z₃₂
    ///
    /// - Closure: Rotate<a> ∘ Rotate<b> = Rotate<(a+b) mod 32>
    /// - Identity: Rotate<0>
    /// - Inverse: Rotate<a>⁻¹ = Rotate<32-a>
    pub fn rotate_group_composition(a: u32, b: u32) -> u32 {
        (a + b) & 0x1F
    }

    /// All permutations on 32 elements form the symmetric group S₃₂
    ///
    /// |S₃₂| = 32! ≈ 2.6 × 10³⁵
    ///
    /// XOR shuffles are a tiny subgroup: |{Xor<m>}| = 32
    /// Rotations are another subgroup: |{Rotate<d>}| = 32
    ///
    /// Butterfly networks use the XOR subgroup.
    /// Shift-based algorithms use the rotation subgroup.
    pub const XOR_SUBGROUP_SIZE: u64 = 32;
    pub const ROTATE_SUBGROUP_SIZE: u64 = 32;
    // S_32 is astronomically large, but we only use small subgroups
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_self_dual() {
        // XOR shuffles are involutions: applying twice = identity
        for mask in 0..32u32 {
            for lane in 0..32u32 {
                let after_one = (lane ^ mask) & 0x1F;
                let after_two = (after_one ^ mask) & 0x1F;
                assert_eq!(after_two, lane, "XOR<{}> is not involution", mask);
            }
        }
    }

    #[test]
    fn test_xor_is_self_dual_type() {
        assert!(Xor::<5>::is_self_dual());
        assert!(Xor::<0>::is_self_dual());
        assert!(Xor::<31>::is_self_dual());
    }

    #[test]
    fn test_rotate_duality() {
        // RotateDown<1> and RotateUp<1> are inverses
        for lane in 0..32u32 {
            let down_then_up = RotateUp::<1>::forward(RotateDown::<1>::forward(lane));
            assert_eq!(down_then_up, lane);
        }
    }

    #[test]
    fn test_shuffle_unshuffle_roundtrip() {
        let original: [i32; 32] = core::array::from_fn(|i| i as i32);

        // Shuffle by XOR<5>
        let shuffled = shuffle(original, Xor::<5>);

        // XOR is self-dual, so unshuffle = shuffle again
        let unshuffled = unshuffle(shuffled);

        assert_eq!(unshuffled.data, original);
    }

    #[test]
    fn test_lane_view_symmetric_for_xor() {
        // For XOR shuffles, every lane has symmetric exchange
        assert!(all_symmetric::<Xor<5>>());
        assert!(all_symmetric::<Xor<16>>());
        assert!(all_symmetric::<Xor<0>>());
    }

    #[test]
    fn test_lane_view_asymmetric_for_rotate() {
        // For rotation (except 0 and 16), lanes have asymmetric exchange
        let view = LaneView::for_lane::<RotateDown<5>>(0);
        assert_ne!(view.i_send_to, view.i_receive_from);

        // Lane 0 sends to lane 27 (0-5 mod 32), receives from lane 5
        assert_eq!(view.i_send_to, 27);
        assert_eq!(view.i_receive_from, 5);
    }

    #[test]
    fn test_xor_group_structure() {
        // XOR forms abelian group
        // Closure
        assert_eq!(algebra::xor_group_composition(5, 3), 6);

        // Identity
        assert_eq!(algebra::xor_group_composition(5, 0), 5);

        // Self-inverse
        assert_eq!(algebra::xor_group_composition(5, 5), 0);

        // Commutativity
        assert_eq!(
            algebra::xor_group_composition(5, 3),
            algebra::xor_group_composition(3, 5)
        );

        // Associativity
        assert_eq!(
            algebra::xor_group_composition(algebra::xor_group_composition(5, 3), 7),
            algebra::xor_group_composition(5, algebra::xor_group_composition(3, 7))
        );
    }

    #[test]
    fn test_butterfly_permutation() {
        // Full butterfly should be a specific permutation
        // Let's trace lane 0 through all stages
        let mut lane = 0u32;
        lane = Xor::<1>::forward(lane); // 0 -> 1
        lane = Xor::<2>::forward(lane); // 1 -> 3
        lane = Xor::<4>::forward(lane); // 3 -> 7
        lane = Xor::<8>::forward(lane); // 7 -> 15
        lane = Xor::<16>::forward(lane); // 15 -> 31

        assert_eq!(lane, 31, "Full butterfly maps 0 -> 31");

        // In general: full butterfly maps i -> 31 - i (bit reversal adjacent)
        // Actually: XOR with 11111 = 31, so maps i -> i ^ 31
        for i in 0..32u32 {
            assert_eq!(FullButterfly::forward(i), i ^ 31);
        }
    }

    #[test]
    fn test_butterfly_inverse() {
        // Inverse butterfly should undo full butterfly
        for i in 0..32u32 {
            let through = InverseButterfly::forward(FullButterfly::forward(i));
            assert_eq!(through, i);
        }
    }
}
