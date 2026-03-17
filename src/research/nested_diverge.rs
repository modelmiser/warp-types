//! Nested Divergence: Recursive Splitting of Warps
//!
//! **STATUS: Superseded** — Promoted to main's `warp_sets!` macro and `merge_within`. Retained as research artifact.
//!
//! When you diverge within already-divergent code, you create a TREE of active sets.
//! The type system must track:
//! 1. Intersection types: EvenLow = Even ∩ LowHalf
//! 2. Complement relationships at each level
//! 3. Valid merge orderings
//!
//! Key insight: Nested divergence creates a BINARY TREE of active sets.
//! Each leaf is a conjunction of predicates from root to leaf.
//! Merging works bottom-up, respecting the tree structure.

use std::marker::PhantomData;

// ============================================================================
// ACTIVE SET HIERARCHY
// ============================================================================

/// Marker trait for active lane sets
pub trait ActiveSet {
    /// Bitmask of active lanes (for runtime, but type encodes it)
    const MASK: u32;

    /// Human-readable name for debugging
    fn name() -> &'static str;
}

/// All 32 lanes active
pub struct All;
impl ActiveSet for All {
    const MASK: u32 = 0xFFFFFFFF;
    fn name() -> &'static str { "All" }
}

/// Even lanes: 0, 2, 4, ..., 30
pub struct Even;
impl ActiveSet for Even {
    const MASK: u32 = 0x55555555;
    fn name() -> &'static str { "Even" }
}

/// Odd lanes: 1, 3, 5, ..., 31
pub struct Odd;
impl ActiveSet for Odd {
    const MASK: u32 = 0xAAAAAAAA;
    fn name() -> &'static str { "Odd" }
}

/// Low half: lanes 0-15
pub struct LowHalf;
impl ActiveSet for LowHalf {
    const MASK: u32 = 0x0000FFFF;
    fn name() -> &'static str { "LowHalf" }
}

/// High half: lanes 16-31
pub struct HighHalf;
impl ActiveSet for HighHalf {
    const MASK: u32 = 0xFFFF0000;
    fn name() -> &'static str { "HighHalf" }
}

// ============================================================================
// INTERSECTION TYPES (Nested Divergence Results)
// ============================================================================

/// Even ∩ LowHalf = lanes 0, 2, 4, 6, 8, 10, 12, 14
pub struct EvenLow;
impl ActiveSet for EvenLow {
    const MASK: u32 = Even::MASK & LowHalf::MASK;  // 0x00005555
    fn name() -> &'static str { "EvenLow" }
}

/// Even ∩ HighHalf = lanes 16, 18, 20, 22, 24, 26, 28, 30
pub struct EvenHigh;
impl ActiveSet for EvenHigh {
    const MASK: u32 = Even::MASK & HighHalf::MASK;  // 0x55550000
    fn name() -> &'static str { "EvenHigh" }
}

/// Odd ∩ LowHalf = lanes 1, 3, 5, 7, 9, 11, 13, 15
pub struct OddLow;
impl ActiveSet for OddLow {
    const MASK: u32 = Odd::MASK & LowHalf::MASK;  // 0x0000AAAA
    fn name() -> &'static str { "OddLow" }
}

/// Odd ∩ HighHalf = lanes 17, 19, 21, 23, 25, 27, 29, 31
pub struct OddHigh;
impl ActiveSet for OddHigh {
    const MASK: u32 = Odd::MASK & HighHalf::MASK;  // 0xAAAA0000
    fn name() -> &'static str { "OddHigh" }
}

// ============================================================================
// COMPLEMENT RELATIONSHIPS
// ============================================================================

/// S1 and S2 are complements within parent P
/// Meaning: S1 ∪ S2 = P and S1 ∩ S2 = ∅
pub trait ComplementWithin<S2: ActiveSet, Parent: ActiveSet>: ActiveSet {}

// Top-level complements (within All)
impl ComplementWithin<Odd, All> for Even {}
impl ComplementWithin<Even, All> for Odd {}
impl ComplementWithin<HighHalf, All> for LowHalf {}
impl ComplementWithin<LowHalf, All> for HighHalf {}

// Nested complements (within Even)
impl ComplementWithin<EvenHigh, Even> for EvenLow {}
impl ComplementWithin<EvenLow, Even> for EvenHigh {}

// Nested complements (within Odd)
impl ComplementWithin<OddHigh, Odd> for OddLow {}
impl ComplementWithin<OddLow, Odd> for OddHigh {}

// Nested complements (within LowHalf)
impl ComplementWithin<OddLow, LowHalf> for EvenLow {}
impl ComplementWithin<EvenLow, LowHalf> for OddLow {}

// Nested complements (within HighHalf)
impl ComplementWithin<OddHigh, HighHalf> for EvenHigh {}
impl ComplementWithin<EvenHigh, HighHalf> for OddHigh {}

// ============================================================================
// WARP TYPE WITH ACTIVE SET
// ============================================================================

/// A warp with a statically-known active set
#[derive(Debug)]
pub struct Warp<S: ActiveSet> {
    _marker: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self {
        Warp { _marker: PhantomData }
    }

    pub fn active_set_name(&self) -> &'static str {
        S::name()
    }

    pub fn active_mask(&self) -> u32 {
        S::MASK
    }

    pub fn population(&self) -> u32 {
        S::MASK.count_ones()
    }
}

impl<S: ActiveSet> Default for Warp<S> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DIVERGE OPERATIONS
// ============================================================================

/// Trait for types that can be split by a predicate
pub trait CanDiverge<TrueBranch: ActiveSet, FalseBranch: ActiveSet>: ActiveSet + Sized {
    fn diverge(warp: Warp<Self>) -> (Warp<TrueBranch>, Warp<FalseBranch>);
}

// All → Even, Odd
impl CanDiverge<Even, Odd> for All {
    fn diverge(_warp: Warp<Self>) -> (Warp<Even>, Warp<Odd>) {
        (Warp::new(), Warp::new())
    }
}

// All → LowHalf, HighHalf
impl CanDiverge<LowHalf, HighHalf> for All {
    fn diverge(_warp: Warp<Self>) -> (Warp<LowHalf>, Warp<HighHalf>) {
        (Warp::new(), Warp::new())
    }
}

// Even → EvenLow, EvenHigh (NESTED!)
impl CanDiverge<EvenLow, EvenHigh> for Even {
    fn diverge(_warp: Warp<Self>) -> (Warp<EvenLow>, Warp<EvenHigh>) {
        (Warp::new(), Warp::new())
    }
}

// Odd → OddLow, OddHigh (NESTED!)
impl CanDiverge<OddLow, OddHigh> for Odd {
    fn diverge(_warp: Warp<Self>) -> (Warp<OddLow>, Warp<OddHigh>) {
        (Warp::new(), Warp::new())
    }
}

// LowHalf → EvenLow, OddLow (alternative nesting)
impl CanDiverge<EvenLow, OddLow> for LowHalf {
    fn diverge(_warp: Warp<Self>) -> (Warp<EvenLow>, Warp<OddLow>) {
        (Warp::new(), Warp::new())
    }
}

// HighHalf → EvenHigh, OddHigh (alternative nesting)
impl CanDiverge<EvenHigh, OddHigh> for HighHalf {
    fn diverge(_warp: Warp<Self>) -> (Warp<EvenHigh>, Warp<OddHigh>) {
        (Warp::new(), Warp::new())
    }
}

// ============================================================================
// MERGE OPERATIONS
// ============================================================================

/// Merge two complementary sub-warps back into their parent
pub fn merge<S1, S2, P>(
    _left: Warp<S1>,
    _right: Warp<S2>,
) -> Warp<P>
where
    S1: ComplementWithin<S2, P>,
    S2: ActiveSet,
    P: ActiveSet,
{
    // Type system has verified S1 ∪ S2 = P
    Warp::new()
}

// ============================================================================
// KEY INSIGHT: MERGE ORDERING FLEXIBILITY
// ============================================================================

/// Can we merge in different orders and get the same result?
///
/// Consider 4 leaf sets from double-nested divergence:
///   EvenLow, EvenHigh, OddLow, OddHigh
///
/// Tree-structured merge (respects diverge order):
///   merge(EvenLow, EvenHigh) → Even
///   merge(OddLow, OddHigh) → Odd
///   merge(Even, Odd) → All
///
/// Alternative merge (different grouping):
///   merge(EvenLow, OddLow) → LowHalf
///   merge(EvenHigh, OddHigh) → HighHalf
///   merge(LowHalf, HighHalf) → All
///
/// BOTH are valid! The final result is All either way.
/// The type system allows both orderings.
pub mod merge_ordering {
    use super::*;

    /// Standard tree merge: Even/Odd split first
    pub fn tree_merge_even_odd(
        even_low: Warp<EvenLow>,
        even_high: Warp<EvenHigh>,
        odd_low: Warp<OddLow>,
        odd_high: Warp<OddHigh>,
    ) -> Warp<All> {
        // Merge within Even
        let even: Warp<Even> = merge(even_low, even_high);

        // Merge within Odd
        let odd: Warp<Odd> = merge(odd_low, odd_high);

        // Merge Even and Odd
        merge(even, odd)
    }

    /// Alternative merge: Low/High split first
    pub fn tree_merge_low_high(
        even_low: Warp<EvenLow>,
        even_high: Warp<EvenHigh>,
        odd_low: Warp<OddLow>,
        odd_high: Warp<OddHigh>,
    ) -> Warp<All> {
        // Merge within LowHalf
        let low: Warp<LowHalf> = merge(even_low, odd_low);

        // Merge within HighHalf
        let high: Warp<HighHalf> = merge(even_high, odd_high);

        // Merge Low and High
        merge(low, high)
    }

    // Both functions produce Warp<All> - the type system proves equivalence!
}

// ============================================================================
// WHAT ABOUT INVALID MERGES?
// ============================================================================

/// These would be compile errors:
///
/// ```compile_fail
/// // Can't merge EvenLow with OddHigh - not complements within any parent!
/// let bad: Warp<???> = merge(even_low, odd_high);
/// ```
///
/// EvenLow (0,2,4,6,8,10,12,14) and OddHigh (17,19,21,...,31) are:
/// - Disjoint: EvenLow ∩ OddHigh = ∅ ✓
/// - But NOT complementary within any standard parent
/// - EvenLow ∪ OddHigh ≠ Even, Odd, LowHalf, HighHalf, or All
///
/// To merge them, you'd need a custom parent type:
/// struct EvenLowOrOddHigh;  // Mask = 0x0000555 | 0xAAAA0000
///
/// But this isn't a "natural" split from any single diverge!
pub mod invalid_merges {
    // This module is documentation - the invalid code won't compile
}

// ============================================================================
// CROSS-MERGE: When Different Paths Produce Same Leaves
// ============================================================================

/// Key insight: EvenLow can be reached two ways:
/// 1. All → Even → EvenLow
/// 2. All → LowHalf → EvenLow
///
/// The resulting Warp<EvenLow> is the same type either way!
/// This enables flexible merge orderings.
pub mod path_independence {
    use super::*;

    pub fn via_even() -> Warp<EvenLow> {
        let all: Warp<All> = Warp::new();
        let (even, _odd): (Warp<Even>, Warp<Odd>) = All::diverge(all);
        let (even_low, _even_high): (Warp<EvenLow>, Warp<EvenHigh>) = Even::diverge(even);
        even_low
    }

    pub fn via_low() -> Warp<EvenLow> {
        let all: Warp<All> = Warp::new();
        let (low, _high): (Warp<LowHalf>, Warp<HighHalf>) = All::diverge(all);
        let (even_low, _odd_low): (Warp<EvenLow>, Warp<OddLow>) = LowHalf::diverge(low);
        even_low
    }

    /// Both produce Warp<EvenLow> - same type, regardless of path taken!
    pub fn paths_are_equivalent() -> bool {
        // The type system guarantees via_even() and via_low() return the same type
        true
    }
}

// ============================================================================
// LATTICE STRUCTURE OF ACTIVE SETS
// ============================================================================

/// Active sets form a LATTICE under subset ordering:
///
/// ```text
///                    All (32 lanes)
///                   /    \
///                  /      \
///          Even (16)    Odd (16)      LowHalf (16)    HighHalf (16)
///            / \          / \             / \              / \
///           /   \        /   \           /   \            /   \
///     EvenLow  EvenHigh OddLow OddHigh  EvenLow OddLow  EvenHigh OddHigh
///       (8)      (8)     (8)    (8)       (8)    (8)      (8)     (8)
/// ```
///
/// Note: EvenLow appears in TWO places! It's the same set reached different ways.
///
/// Lattice operations:
/// - Meet (∩): Intersection of two sets
/// - Join (∪): Union of two sets
/// - Complement: Within a parent, the "other half"
///
/// The ComplementWithin trait encodes the complement relationship.
/// The CanDiverge trait encodes which splits are valid.
pub mod lattice {
    use super::*;

    /// Verify the lattice relationships hold
    #[allow(dead_code)]
    fn verify_lattice() {
        // Even ∩ LowHalf = EvenLow
        assert_eq!(Even::MASK & LowHalf::MASK, EvenLow::MASK);

        // Even ∩ HighHalf = EvenHigh
        assert_eq!(Even::MASK & HighHalf::MASK, EvenHigh::MASK);

        // EvenLow ∪ EvenHigh = Even
        assert_eq!(EvenLow::MASK | EvenHigh::MASK, Even::MASK);

        // EvenLow ∪ OddLow = LowHalf
        assert_eq!(EvenLow::MASK | OddLow::MASK, LowHalf::MASK);

        // All four leaves union to All
        assert_eq!(
            EvenLow::MASK | EvenHigh::MASK | OddLow::MASK | OddHigh::MASK,
            All::MASK
        );
    }
}

// ============================================================================
// DEPTH-3 NESTING: QUADRANTS OF QUADRANTS
// ============================================================================

/// Can we go deeper? Yes, but we need more intersection types.
///
/// Example: Split EvenLow by "very low" (lanes 0-7) vs "mid-low" (lanes 8-15)
///
/// EvenLow = {0, 2, 4, 6, 8, 10, 12, 14}
///   → EvenVeryLow = {0, 2, 4, 6}
///   → EvenMidLow = {8, 10, 12, 14}
pub mod depth_3 {
    use super::*;

    pub struct VeryLow;  // Lanes 0-7
    impl ActiveSet for VeryLow {
        const MASK: u32 = 0x000000FF;
        fn name() -> &'static str { "VeryLow" }
    }

    pub struct MidLow;  // Lanes 8-15
    impl ActiveSet for MidLow {
        const MASK: u32 = 0x0000FF00;
        fn name() -> &'static str { "MidLow" }
    }

    /// Even ∩ VeryLow = lanes 0, 2, 4, 6
    pub struct EvenVeryLow;
    impl ActiveSet for EvenVeryLow {
        const MASK: u32 = Even::MASK & VeryLow::MASK;  // 0x00000055
        fn name() -> &'static str { "EvenVeryLow" }
    }

    /// Even ∩ MidLow = lanes 8, 10, 12, 14
    pub struct EvenMidLow;
    impl ActiveSet for EvenMidLow {
        const MASK: u32 = Even::MASK & MidLow::MASK;  // 0x00005500
        fn name() -> &'static str { "EvenMidLow" }
    }

    // Complements within EvenLow
    impl ComplementWithin<EvenMidLow, EvenLow> for EvenVeryLow {}
    impl ComplementWithin<EvenVeryLow, EvenLow> for EvenMidLow {}

    // Can diverge EvenLow further
    impl CanDiverge<EvenVeryLow, EvenMidLow> for EvenLow {
        fn diverge(_warp: Warp<Self>) -> (Warp<EvenVeryLow>, Warp<EvenMidLow>) {
            (Warp::new(), Warp::new())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_depth_3_diverge() {
            let all: Warp<All> = Warp::new();

            // Level 1: All → Even, Odd
            let (even, _odd) = All::diverge(all);

            // Level 2: Even → EvenLow, EvenHigh
            let (even_low, _even_high) = Even::diverge(even);

            // Level 3: EvenLow → EvenVeryLow, EvenMidLow
            let (even_very_low, even_mid_low) = EvenLow::diverge(even_low);

            assert_eq!(even_very_low.population(), 4);  // Lanes 0,2,4,6
            assert_eq!(even_mid_low.population(), 4);   // Lanes 8,10,12,14
        }

        #[test]
        fn test_depth_3_merge() {
            let even_very_low: Warp<EvenVeryLow> = Warp::new();
            let even_mid_low: Warp<EvenMidLow> = Warp::new();

            // Merge back to EvenLow
            let even_low: Warp<EvenLow> = merge(even_very_low, even_mid_low);
            assert_eq!(even_low.population(), 8);
        }
    }
}

// ============================================================================
// THE EXPONENTIAL PROBLEM
// ============================================================================

/// With N diverge predicates, we could have 2^N leaf types!
///
/// Example: 3 predicates (even/odd, low/high, verylow/midlow)
/// → 8 possible leaf types (only some are reachable from valid paths)
///
/// For a real type system:
/// - Can't enumerate all 2^32 possible subsets
/// - Must use compositional types: Intersect<Even, LowHalf, VeryLow>
/// - Or: runtime masks with static "shape" verification
///
/// Our marker-type approach works for common patterns but doesn't scale
/// to arbitrary predicates. This is the Q1 limitation.
pub mod exponential {
    // The number of potential active sets grows exponentially with predicates.
    // Marker types handle common cases; dependent types would be needed for full generality.
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::merge_ordering::*;

    #[test]
    fn test_mask_values() {
        assert_eq!(All::MASK, 0xFFFFFFFF);
        assert_eq!(Even::MASK, 0x55555555);
        assert_eq!(Odd::MASK, 0xAAAAAAAA);
        assert_eq!(LowHalf::MASK, 0x0000FFFF);
        assert_eq!(HighHalf::MASK, 0xFFFF0000);

        // Intersection types
        assert_eq!(EvenLow::MASK, 0x00005555);
        assert_eq!(EvenHigh::MASK, 0x55550000);
        assert_eq!(OddLow::MASK, 0x0000AAAA);
        assert_eq!(OddHigh::MASK, 0xAAAA0000);
    }

    #[test]
    fn test_population_counts() {
        assert_eq!(Warp::<All>::new().population(), 32);
        assert_eq!(Warp::<Even>::new().population(), 16);
        assert_eq!(Warp::<Odd>::new().population(), 16);
        assert_eq!(Warp::<EvenLow>::new().population(), 8);
        assert_eq!(Warp::<EvenHigh>::new().population(), 8);
    }

    #[test]
    fn test_basic_diverge_merge() {
        let all: Warp<All> = Warp::new();

        // Diverge
        let (even, odd): (Warp<Even>, Warp<Odd>) = All::diverge(all);
        assert_eq!(even.population(), 16);
        assert_eq!(odd.population(), 16);

        // Merge
        let reunited: Warp<All> = merge(even, odd);
        assert_eq!(reunited.population(), 32);
    }

    #[test]
    fn test_nested_diverge() {
        let all: Warp<All> = Warp::new();

        // Level 1
        let (even, odd) = All::diverge(all);

        // Level 2 (within Even)
        let (even_low, even_high) = Even::diverge(even);

        // Level 2 (within Odd)
        let (odd_low, odd_high) = Odd::diverge(odd);

        assert_eq!(even_low.population(), 8);
        assert_eq!(even_high.population(), 8);
        assert_eq!(odd_low.population(), 8);
        assert_eq!(odd_high.population(), 8);
    }

    #[test]
    fn test_merge_ordering_equivalence() {
        let _even_low: Warp<EvenLow> = Warp::new();
        let _even_high: Warp<EvenHigh> = Warp::new();
        let _odd_low: Warp<OddLow> = Warp::new();
        let _odd_high: Warp<OddHigh> = Warp::new();

        // Both orderings produce Warp<All>
        let result1 = tree_merge_even_odd(
            Warp::new(), Warp::new(), Warp::new(), Warp::new()
        );
        let result2 = tree_merge_low_high(
            Warp::new(), Warp::new(), Warp::new(), Warp::new()
        );

        assert_eq!(result1.population(), 32);
        assert_eq!(result2.population(), 32);
    }

    #[test]
    fn test_union_properties() {
        // EvenLow ∪ EvenHigh = Even
        assert_eq!(EvenLow::MASK | EvenHigh::MASK, Even::MASK);

        // EvenLow ∪ OddLow = LowHalf
        assert_eq!(EvenLow::MASK | OddLow::MASK, LowHalf::MASK);

        // All four = All
        let all_four = EvenLow::MASK | EvenHigh::MASK | OddLow::MASK | OddHigh::MASK;
        assert_eq!(all_four, All::MASK);
    }

    #[test]
    fn test_disjoint_properties() {
        // All intersection types are pairwise disjoint
        assert_eq!(EvenLow::MASK & EvenHigh::MASK, 0);
        assert_eq!(EvenLow::MASK & OddLow::MASK, 0);
        assert_eq!(EvenLow::MASK & OddHigh::MASK, 0);
        assert_eq!(EvenHigh::MASK & OddLow::MASK, 0);
        assert_eq!(EvenHigh::MASK & OddHigh::MASK, 0);
        assert_eq!(OddLow::MASK & OddHigh::MASK, 0);
    }
}
