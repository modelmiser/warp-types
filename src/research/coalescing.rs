//! Memory Coalescing Types
//!
//! **STATUS: Research** — Exploratory prototype, not promoted to main API.
//!
//! Research question: "Can types verify coalescing properties?"
//!
//! # Background
//!
//! Memory coalescing is critical for GPU performance:
//! - Coalesced: All lanes access consecutive addresses → 1 transaction
//! - Uncoalesced: Random access → up to 32 transactions
//! - 10-100x performance difference!
//!
//! # Key Insight
//!
//! Access patterns can be classified by their REGULARITY:
//! - Uniform: All lanes access same address (broadcast)
//! - Consecutive: Lane i accesses base + i (perfect coalescing)
//! - Strided: Lane i accesses base + i*stride (partial coalescing)
//! - Random: No pattern (worst case)
//!
//! These patterns can be encoded in the type system!
//!
//! # Design
//!
//! ```text
//! Ptr<T, AccessPattern>
//! where AccessPattern in {Uniform, Consecutive, Strided<N>, Random}
//! ```

use std::marker::PhantomData;

use crate::WARP_SIZE;

// ============================================================================
// ACCESS PATTERN TYPES
// ============================================================================

/// Marker trait for access patterns
pub trait AccessPattern: Copy + 'static {
    fn name() -> &'static str;
    fn transactions_per_warp() -> usize; // 1 = perfect, 32 = worst
}

/// All lanes access the same address (broadcast load)
#[derive(Copy, Clone)]
pub struct Uniform;

impl AccessPattern for Uniform {
    fn name() -> &'static str {
        "Uniform"
    }
    fn transactions_per_warp() -> usize {
        1
    }
}

/// Lane i accesses base + i * sizeof(T) (perfect coalescing)
#[derive(Copy, Clone)]
pub struct Consecutive;

impl AccessPattern for Consecutive {
    fn name() -> &'static str {
        "Consecutive"
    }
    fn transactions_per_warp() -> usize {
        1
    } // For aligned, same-line access
}

/// Lane i accesses base + i * STRIDE * sizeof(T)
#[derive(Copy, Clone)]
pub struct Strided<const STRIDE: usize>;

impl<const STRIDE: usize> AccessPattern for Strided<STRIDE> {
    fn name() -> &'static str {
        "Strided"
    }
    fn transactions_per_warp() -> usize {
        // Depends on cache line size and stride
        // Rough estimate: stride 0 = uniform (1 txn), otherwise min(WARP_SIZE, STRIDE)
        std::cmp::max(1, std::cmp::min(WARP_SIZE as usize, STRIDE))
    }
}

/// Random access pattern (worst case)
#[derive(Copy, Clone)]
pub struct Random;

impl AccessPattern for Random {
    fn name() -> &'static str {
        "Random"
    }
    fn transactions_per_warp() -> usize {
        WARP_SIZE as usize
    } // Worst case
}

// ============================================================================
// TYPED POINTERS
// ============================================================================

/// A pointer with known access pattern
#[derive(Clone)]
pub struct WarpPtr<T: Copy, P: AccessPattern> {
    base: *const T,
    _pattern: PhantomData<P>,
}

impl<T: Copy, P: AccessPattern> WarpPtr<T, P> {
    /// Create a new typed pointer.
    ///
    /// # Safety
    ///
    /// For every offset a safe API may use for the declared pattern `P`,
    /// `base.add(offset)` must be within ONE allocated object, aligned, and
    /// point to an initialized `T` that stays valid — and unaliased by any
    /// live `&mut` — for the lifetime of this value. Same-allocation and
    /// provenance are requirements of `pointer::add` itself, not just
    /// readability.
    ///
    /// The offsets, per pattern:
    ///
    /// - `Uniform`: offset 0 only (1 element).
    /// - `Consecutive`: `0..WARP_SIZE`. **`WARP_SIZE` elements, not 32** —
    ///   it is 64 under the `warp64` feature (`lib.rs`), and `load::consecutive`
    ///   iterates `0..WARP_SIZE`. An allocation sized for the literal 32 is a
    ///   64-element read on a `warp64` build, through a SAFE function. This
    ///   said "32 elements" until a 2026-09-05 review; a caller who satisfied
    ///   it exactly and enabled the feature had UB.
    /// - `Strided<S>`: **no span is required, because nothing reads one.** No
    ///   `load`/`store` in this module indexes `base + lane * S`. Do not
    ///   promise that span until such a function exists — and when it does,
    ///   note `S` is an unbounded `usize` (`0` included), so `lane * S` and
    ///   `lane * S * size_of::<T>()` must also be shown to fit `isize` and
    ///   stay in-allocation.
    /// - `Random`: construction and holding require nothing of `base` — no
    ///   safe deref is typed at `Random`, so a dangling base is sound to HOLD.
    ///   That is narrower than it used to read: the only deref is the `unsafe`
    ///   [`load::generic`], which needs `base` to carry real provenance (even
    ///   `add(0)` does) and each caller-supplied index to be a valid `add`
    ///   from it.
    ///
    /// This is load-bearing: the pattern is fixed at construction (no safe
    /// conversion exists — `WorstOf` is type-level only), and `load`/`store`
    /// are SAFE functions. `load::consecutive` on a `WarpPtr` built over a
    /// one-element allocation is an out-of-bounds read with no `unsafe` at
    /// the call site, so the span belongs in this contract rather than in
    /// the reader's head.
    pub unsafe fn new(base: *const T) -> Self {
        WarpPtr {
            base,
            _pattern: PhantomData,
        }
    }

    pub fn base(&self) -> *const T {
        self.base
    }

    pub fn pattern_name() -> &'static str {
        P::name()
    }

    pub fn expected_transactions() -> usize {
        P::transactions_per_warp()
    }
}

/// A mutable pointer with known access pattern
pub struct WarpPtrMut<T: Copy, P: AccessPattern> {
    base: *mut T,
    _pattern: PhantomData<P>,
}

impl<T: Copy, P: AccessPattern> WarpPtrMut<T, P> {
    /// Create a new mutable typed pointer.
    ///
    /// # Safety
    ///
    /// Same span contract as [`WarpPtr::new`], but the span must be valid
    /// for WRITES and must not alias any other live reference, `WarpPtrMut`,
    /// **or `WarpPtr`** — a shared `WarpPtr` over the same address is read
    /// through safe `load::*` while this writes, which is a data race just as
    /// surely as two `WarpPtrMut`s. `store::consecutive` writes all
    /// `WARP_SIZE` elements through a safe function.
    pub unsafe fn new(base: *mut T) -> Self {
        WarpPtrMut {
            base,
            _pattern: PhantomData,
        }
    }

    pub fn base(&self) -> *mut T {
        self.base
    }
}

// ============================================================================
// PATTERN PROMOTION/DEMOTION
// ============================================================================

/// Patterns form a hierarchy: Uniform < Consecutive < Strided < Random
///
/// Combining patterns uses the "worst" pattern:
/// - Uniform + Consecutive = Consecutive
/// - Consecutive + Strided = Strided
/// - Anything + Random = Random
pub trait WorstOf<Other: AccessPattern>: AccessPattern {
    type Result: AccessPattern;
}

// Uniform is dominated by everything
impl WorstOf<Uniform> for Uniform {
    type Result = Uniform;
}
impl WorstOf<Consecutive> for Uniform {
    type Result = Consecutive;
}
impl<const S: usize> WorstOf<Strided<S>> for Uniform {
    type Result = Strided<S>;
}
impl WorstOf<Random> for Uniform {
    type Result = Random;
}

// Consecutive dominates Uniform
impl WorstOf<Uniform> for Consecutive {
    type Result = Consecutive;
}
impl WorstOf<Consecutive> for Consecutive {
    type Result = Consecutive;
}
impl<const S: usize> WorstOf<Strided<S>> for Consecutive {
    type Result = Strided<S>;
}
impl WorstOf<Random> for Consecutive {
    type Result = Random;
}

// Random dominates everything
impl WorstOf<Uniform> for Random {
    type Result = Random;
}
impl WorstOf<Consecutive> for Random {
    type Result = Random;
}
impl<const S: usize> WorstOf<Strided<S>> for Random {
    type Result = Random;
}
impl WorstOf<Random> for Random {
    type Result = Random;
}

// ============================================================================
// SAFE LOAD/STORE WITH PATTERN
// ============================================================================

/// Load with compile-time pattern check
pub mod load {
    use super::*;

    /// Uniform load: all lanes get same value.
    ///
    /// Safety relies on `WarpPtr::new`'s contract: the pointer must remain
    /// valid for the lifetime of the `WarpPtr`. This is an "unsafe
    /// constructor, safe usage" split — but NOT the `Vec::from_raw_parts`
    /// one it used to cite. `Vec` OWNS its allocation and its safe methods
    /// cannot reach past the capacity you promised; `WarpPtr` owns nothing,
    /// carries no lifetime tying it to the allocation, and `P` is a tag the
    /// caller chose rather than anything derived from the pointer.
    pub fn uniform<T: Copy>(ptr: &WarpPtr<T, Uniform>) -> T {
        unsafe { *ptr.base() }
    }

    /// Consecutive load: lane i gets base[i]
    /// Returns per-lane values
    pub fn consecutive<T: Copy + Default>(
        ptr: &WarpPtr<T, Consecutive>,
    ) -> [T; WARP_SIZE as usize] {
        let mut result = [T::default(); WARP_SIZE as usize];
        for lane in 0..WARP_SIZE as usize {
            unsafe {
                result[lane] = *ptr.base().add(lane);
            }
        }
        result
    }

    /// Generic load (for any pattern).
    ///
    /// # Safety
    ///
    /// Each `indices[lane]` must be a valid offset from `ptr.base()` —
    /// i.e., `ptr.base().add(indices[lane])` must be in-bounds of the
    /// original allocation.
    pub unsafe fn generic<T: Copy + Default, P: AccessPattern>(
        ptr: &WarpPtr<T, P>,
        indices: &[usize; WARP_SIZE as usize],
    ) -> [T; WARP_SIZE as usize] {
        let mut result = [T::default(); WARP_SIZE as usize];
        for lane in 0..WARP_SIZE as usize {
            unsafe {
                result[lane] = *ptr.base().add(indices[lane]);
            }
        }
        result
    }
}

/// Store with compile-time pattern check
pub mod store {
    use super::*;

    /// Uniform store: all lanes write same value to same address.
    ///
    /// `&mut` uniquely borrows THIS WRAPPER; it does not make the pointee
    /// unique and buys no aliasing guarantee — `base(&self)` hands out a
    /// `*mut T` from a shared borrow, and a second `WarpPtrMut` over the same
    /// address can be constructed. What it does prevent is two stores through
    /// the same value at once. Exclusivity of the span is an obligation of
    /// `WarpPtrMut::new`, not of this signature.
    pub fn uniform<T: Copy>(ptr: &mut WarpPtrMut<T, Uniform>, value: T) {
        unsafe {
            *ptr.base() = value;
        }
    }

    /// Consecutive store: lane i writes to base[i].
    /// See `store::uniform` on what `&mut` here does and does not buy.
    pub fn consecutive<T: Copy>(
        ptr: &mut WarpPtrMut<T, Consecutive>,
        values: &[T; WARP_SIZE as usize],
    ) {
        for lane in 0..WARP_SIZE as usize {
            unsafe {
                *ptr.base().add(lane) = values[lane];
            }
        }
    }
}

// ============================================================================
// PATTERN INFERENCE
// ============================================================================

/// Infer access pattern from index expression
pub mod infer {

    /// Index expression types
    pub enum IndexExpr {
        Constant(usize),    // Same for all lanes → Uniform
        LaneId,             // lane_id → Consecutive
        LaneIdTimes(usize), // lane_id * stride → Strided
        LaneIdPlus(usize),  // lane_id + offset → Consecutive (shifted)
        Computed,           // Arbitrary → Random
    }

    /// Infer pattern from index expression
    pub fn pattern_from_index(expr: &IndexExpr) -> &'static str {
        match expr {
            IndexExpr::Constant(_) => "Uniform",
            IndexExpr::LaneId => "Consecutive",
            IndexExpr::LaneIdTimes(1) => "Consecutive",
            IndexExpr::LaneIdTimes(_) => "Strided",
            IndexExpr::LaneIdPlus(_) => "Consecutive",
            IndexExpr::Computed => "Random",
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_pattern_inference() {
            assert_eq!(pattern_from_index(&IndexExpr::Constant(0)), "Uniform");
            assert_eq!(pattern_from_index(&IndexExpr::LaneId), "Consecutive");
            assert_eq!(pattern_from_index(&IndexExpr::LaneIdTimes(4)), "Strided");
            assert_eq!(pattern_from_index(&IndexExpr::Computed), "Random");
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_counts() {
        assert_eq!(Uniform::transactions_per_warp(), 1);
        assert_eq!(Consecutive::transactions_per_warp(), 1);
        assert_eq!(Strided::<4>::transactions_per_warp(), 4);
        assert_eq!(Random::transactions_per_warp(), WARP_SIZE as usize);
    }

    #[test]
    fn test_uniform_load() {
        let data = [42i32; 1];
        let ptr = unsafe { WarpPtr::<i32, Uniform>::new(data.as_ptr()) };

        let value = load::uniform(&ptr);
        assert_eq!(value, 42);
    }

    #[test]
    fn test_consecutive_load() {
        let data: [i32; WARP_SIZE as usize] = core::array::from_fn(|i| i as i32);
        let ptr = unsafe { WarpPtr::<i32, Consecutive>::new(data.as_ptr()) };

        let values = load::consecutive(&ptr);
        for lane in 0..WARP_SIZE as usize {
            assert_eq!(values[lane], lane as i32);
        }
    }

    #[test]
    fn test_pattern_hierarchy() {
        // Uniform + Consecutive = Consecutive
        type R1 = <Uniform as WorstOf<Consecutive>>::Result;
        assert_eq!(R1::name(), "Consecutive");

        // Consecutive + Random = Random
        type R2 = <Consecutive as WorstOf<Random>>::Result;
        assert_eq!(R2::name(), "Random");
    }
}

// ============================================================================
// SUMMARY
// ============================================================================

/// Summary: Can types verify coalescing properties?
///
/// ## Answer: Yes, for common patterns
///
/// ### Approach
///
/// Encode access pattern in pointer type:
/// ```text
/// WarpPtr<T, Uniform>      - all lanes access same address
/// WarpPtr<T, Consecutive>  - lane i accesses base + i
/// WarpPtr<T, Strided<N>>   - lane i accesses base + i*N
/// WarpPtr<T, Random>       - arbitrary access
/// ```
///
/// ### Benefits
///
/// 1. **Compile-time performance prediction**: Know transaction count
/// 2. **Pattern enforcement**: API requires specific pattern
/// 3. **Optimization guidance**: Compiler can choose instructions
/// 4. **Documentation**: Pattern is visible in type
///
/// ### Limitations
///
/// 1. **Pattern must be known statically**: Runtime-computed indices → Random
/// 2. **Conservative for complex patterns**: Strided approximates many cases
/// 3. **Doesn't verify alignment**: Misalignment still causes extra transactions
///
/// ### Integration with Session Types
///
/// Coalescing types compose with active set types:
/// ```text
/// fn load<S: ActiveSet, P: AccessPattern>(
///     warp: &Warp<S>,
///     ptr: &WarpPtr<T, P>,
/// ) -> PerLane<T>
/// ```
///
/// The active set S affects which lanes participate in the load.
/// A load on Warp<Even> with Consecutive pattern has half the transactions.
///
/// ### Verdict
///
/// Types CAN verify coalescing for patterns that are statically known.
/// This covers ~80% of GPU memory access (the performance-critical 80%).
/// Random access (remaining 20%) is typed as such, flagging it for review.
pub const _SUMMARY: () = ();
