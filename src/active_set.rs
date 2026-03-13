//! Active set types: compile-time lane subset tracking.
//!
//! Active sets are zero-sized marker types that represent subsets of warp lanes.
//! The type system tracks which lanes are active through diverge/merge operations,
//! preventing shuffle-from-inactive-lane bugs at compile time.
//!
//! # Lattice structure
//!
//! Active sets form a Boolean lattice under subset ordering:
//!
//! ```text
//!                    All (32 lanes)
//!                   /    \
//!            Even (16)   Odd (16)     LowHalf (16)   HighHalf (16)
//!             / \         / \            / \              / \
//!        EvenLow EvenHigh OddLow OddHigh EvenLow OddLow EvenHigh OddHigh
//!          (8)    (8)      (8)    (8)      (8)    (8)     (8)     (8)
//! ```
//!
//! Note: `EvenLow` appears under both `Even` and `LowHalf` — same set,
//! reached by different diverge paths. Path independence is a key property.

/// Marker trait for active lane set types.
///
/// Each implementor is a zero-sized type encoding a specific bitmask of lanes.
/// The `MASK` constant enables runtime debugging; the type itself provides
/// compile-time tracking.
pub trait ActiveSet: Copy + 'static {
    /// Bitmask of active lanes (for runtime debugging/verification).
    const MASK: u32;
    /// Human-readable name.
    const NAME: &'static str;
}

/// Proof that `Self` and `Other` are complements: disjoint AND covering all 32 lanes.
///
/// This is THE key safety trait. `merge(a, b)` requires `A: ComplementOf<B>`.
/// Only implemented for valid complement pairs — the compiler rejects invalid merges.
pub trait ComplementOf<Other: ActiveSet>: ActiveSet {}

/// Proof that `Self` and `Other` are complements within a parent set `P`.
///
/// `S1 ∪ S2 = P` and `S1 ∩ S2 = ∅`. Used for nested divergence where
/// merge returns to a parent set rather than `All`.
pub trait ComplementWithin<Other: ActiveSet, Parent: ActiveSet>: ActiveSet {}

/// Proof that an active set can be split into two disjoint subsets.
///
/// Implemented for each valid diverge pattern (e.g., `All` → `Even` + `Odd`).
pub trait CanDiverge<TrueBranch: ActiveSet, FalseBranch: ActiveSet>: ActiveSet + Sized {
    fn diverge(warp: crate::warp::Warp<Self>) -> (crate::warp::Warp<TrueBranch>, crate::warp::Warp<FalseBranch>);
}

// ============================================================================
// Concrete active set types
// ============================================================================

/// All 32 lanes active.
#[derive(Copy, Clone, Debug, Default)]
pub struct All;
impl ActiveSet for All {
    const MASK: u32 = 0xFFFFFFFF;
    const NAME: &'static str = "All";
}

/// No lanes active (degenerate).
#[derive(Copy, Clone, Debug, Default)]
pub struct None;
impl ActiveSet for None {
    const MASK: u32 = 0x00000000;
    const NAME: &'static str = "None";
}

/// Even lanes: 0, 2, 4, ..., 30.
#[derive(Copy, Clone, Debug, Default)]
pub struct Even;
impl ActiveSet for Even {
    const MASK: u32 = 0x55555555;
    const NAME: &'static str = "Even";
}

/// Odd lanes: 1, 3, 5, ..., 31.
#[derive(Copy, Clone, Debug, Default)]
pub struct Odd;
impl ActiveSet for Odd {
    const MASK: u32 = 0xAAAAAAAA;
    const NAME: &'static str = "Odd";
}

/// Lower half: lanes 0–15.
#[derive(Copy, Clone, Debug, Default)]
pub struct LowHalf;
impl ActiveSet for LowHalf {
    const MASK: u32 = 0x0000FFFF;
    const NAME: &'static str = "LowHalf";
}

/// Upper half: lanes 16–31.
#[derive(Copy, Clone, Debug, Default)]
pub struct HighHalf;
impl ActiveSet for HighHalf {
    const MASK: u32 = 0xFFFF0000;
    const NAME: &'static str = "HighHalf";
}

/// Lane 0 only.
#[derive(Copy, Clone, Debug, Default)]
pub struct Lane0;
impl ActiveSet for Lane0 {
    const MASK: u32 = 0x00000001;
    const NAME: &'static str = "Lane0";
}

/// All lanes except lane 0.
#[derive(Copy, Clone, Debug, Default)]
pub struct NotLane0;
impl ActiveSet for NotLane0 {
    const MASK: u32 = 0xFFFFFFFE;
    const NAME: &'static str = "NotLane0";
}

// Intersection types for nested divergence

/// Even ∩ LowHalf = lanes 0, 2, 4, 6, 8, 10, 12, 14.
#[derive(Copy, Clone, Debug, Default)]
pub struct EvenLow;
impl ActiveSet for EvenLow {
    const MASK: u32 = Even::MASK & LowHalf::MASK;
    const NAME: &'static str = "EvenLow";
}

/// Even ∩ HighHalf = lanes 16, 18, 20, 22, 24, 26, 28, 30.
#[derive(Copy, Clone, Debug, Default)]
pub struct EvenHigh;
impl ActiveSet for EvenHigh {
    const MASK: u32 = Even::MASK & HighHalf::MASK;
    const NAME: &'static str = "EvenHigh";
}

/// Odd ∩ LowHalf = lanes 1, 3, 5, 7, 9, 11, 13, 15.
#[derive(Copy, Clone, Debug, Default)]
pub struct OddLow;
impl ActiveSet for OddLow {
    const MASK: u32 = Odd::MASK & LowHalf::MASK;
    const NAME: &'static str = "OddLow";
}

/// Odd ∩ HighHalf = lanes 17, 19, 21, 23, 25, 27, 29, 31.
#[derive(Copy, Clone, Debug, Default)]
pub struct OddHigh;
impl ActiveSet for OddHigh {
    const MASK: u32 = Odd::MASK & HighHalf::MASK;
    const NAME: &'static str = "OddHigh";
}

// ============================================================================
// Complement relationships
// ============================================================================

// Top-level complements (within All)
impl ComplementOf<Odd> for Even {}
impl ComplementOf<Even> for Odd {}
impl ComplementOf<HighHalf> for LowHalf {}
impl ComplementOf<LowHalf> for HighHalf {}
impl ComplementOf<NotLane0> for Lane0 {}
impl ComplementOf<Lane0> for NotLane0 {}
impl ComplementOf<None> for All {}
impl ComplementOf<All> for None {}
impl ComplementOf<EvenHigh> for EvenLow {}
impl ComplementOf<EvenLow> for EvenHigh {}

// ComplementWithin relationships (for nested merges)

// Within All
impl ComplementWithin<Odd, All> for Even {}
impl ComplementWithin<Even, All> for Odd {}
impl ComplementWithin<HighHalf, All> for LowHalf {}
impl ComplementWithin<LowHalf, All> for HighHalf {}

// Within Even
impl ComplementWithin<EvenHigh, Even> for EvenLow {}
impl ComplementWithin<EvenLow, Even> for EvenHigh {}

// Within Odd
impl ComplementWithin<OddHigh, Odd> for OddLow {}
impl ComplementWithin<OddLow, Odd> for OddHigh {}

// Within LowHalf
impl ComplementWithin<OddLow, LowHalf> for EvenLow {}
impl ComplementWithin<EvenLow, LowHalf> for OddLow {}

// Within HighHalf
impl ComplementWithin<OddHigh, HighHalf> for EvenHigh {}
impl ComplementWithin<EvenHigh, HighHalf> for OddHigh {}

// ============================================================================
// CanDiverge implementations
// ============================================================================

impl CanDiverge<Even, Odd> for All {
    fn diverge(_warp: crate::warp::Warp<Self>) -> (crate::warp::Warp<Even>, crate::warp::Warp<Odd>) {
        (crate::warp::Warp::new(), crate::warp::Warp::new())
    }
}

impl CanDiverge<LowHalf, HighHalf> for All {
    fn diverge(_warp: crate::warp::Warp<Self>) -> (crate::warp::Warp<LowHalf>, crate::warp::Warp<HighHalf>) {
        (crate::warp::Warp::new(), crate::warp::Warp::new())
    }
}

impl CanDiverge<EvenLow, EvenHigh> for Even {
    fn diverge(_warp: crate::warp::Warp<Self>) -> (crate::warp::Warp<EvenLow>, crate::warp::Warp<EvenHigh>) {
        (crate::warp::Warp::new(), crate::warp::Warp::new())
    }
}

impl CanDiverge<OddLow, OddHigh> for Odd {
    fn diverge(_warp: crate::warp::Warp<Self>) -> (crate::warp::Warp<OddLow>, crate::warp::Warp<OddHigh>) {
        (crate::warp::Warp::new(), crate::warp::Warp::new())
    }
}

impl CanDiverge<EvenLow, OddLow> for LowHalf {
    fn diverge(_warp: crate::warp::Warp<Self>) -> (crate::warp::Warp<EvenLow>, crate::warp::Warp<OddLow>) {
        (crate::warp::Warp::new(), crate::warp::Warp::new())
    }
}

impl CanDiverge<EvenHigh, OddHigh> for HighHalf {
    fn diverge(_warp: crate::warp::Warp<Self>) -> (crate::warp::Warp<EvenHigh>, crate::warp::Warp<OddHigh>) {
        (crate::warp::Warp::new(), crate::warp::Warp::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_values() {
        assert_eq!(All::MASK, 0xFFFFFFFF);
        assert_eq!(None::MASK, 0x00000000);
        assert_eq!(Even::MASK, 0x55555555);
        assert_eq!(Odd::MASK, 0xAAAAAAAA);
        assert_eq!(LowHalf::MASK, 0x0000FFFF);
        assert_eq!(HighHalf::MASK, 0xFFFF0000);
        assert_eq!(Lane0::MASK, 0x00000001);
        assert_eq!(NotLane0::MASK, 0xFFFFFFFE);
        assert_eq!(EvenLow::MASK, 0x00005555);
        assert_eq!(EvenHigh::MASK, 0x55550000);
        assert_eq!(OddLow::MASK, 0x0000AAAA);
        assert_eq!(OddHigh::MASK, 0xAAAA0000);
    }

    #[test]
    fn test_intersection_properties() {
        assert_eq!(Even::MASK & LowHalf::MASK, EvenLow::MASK);
        assert_eq!(Even::MASK & HighHalf::MASK, EvenHigh::MASK);
        assert_eq!(Odd::MASK & LowHalf::MASK, OddLow::MASK);
        assert_eq!(Odd::MASK & HighHalf::MASK, OddHigh::MASK);
    }

    #[test]
    fn test_union_properties() {
        assert_eq!(EvenLow::MASK | EvenHigh::MASK, Even::MASK);
        assert_eq!(OddLow::MASK | OddHigh::MASK, Odd::MASK);
        assert_eq!(EvenLow::MASK | OddLow::MASK, LowHalf::MASK);
        assert_eq!(EvenHigh::MASK | OddHigh::MASK, HighHalf::MASK);
        assert_eq!(
            EvenLow::MASK | EvenHigh::MASK | OddLow::MASK | OddHigh::MASK,
            All::MASK
        );
    }

    #[test]
    fn test_pairwise_disjoint() {
        let sets = [EvenLow::MASK, EvenHigh::MASK, OddLow::MASK, OddHigh::MASK];
        for i in 0..sets.len() {
            for j in (i + 1)..sets.len() {
                assert_eq!(sets[i] & sets[j], 0, "sets {} and {} overlap", i, j);
            }
        }
    }

    #[test]
    fn test_complement_symmetry() {
        assert_eq!(Even::MASK | Odd::MASK, All::MASK);
        assert_eq!(Even::MASK & Odd::MASK, 0);
        assert_eq!(LowHalf::MASK | HighHalf::MASK, All::MASK);
        assert_eq!(LowHalf::MASK & HighHalf::MASK, 0);
        assert_eq!(Lane0::MASK | NotLane0::MASK, All::MASK);
        assert_eq!(Lane0::MASK & NotLane0::MASK, 0);
    }
}
