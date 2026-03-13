//! Divergent Values: Handling Variables in Branched Code
//!
//! Research question: "How to handle values computed in divergent code?"
//!
//! # Background
//!
//! When code diverges, variables can be:
//! 1. Computed only in one branch (some lanes have no value)
//! 2. Assigned different values in different branches
//! 3. Unchanged in one branch, modified in another
//!
//! The type system must track this to prevent invalid reads.
//!
//! # Key Insight
//!
//! Values have SCOPE tied to the active set when they were computed:
//! - `UniformWithin<T, All>` = same value in all lanes
//! - `UniformWithin<T, Even>` = same value in even lanes, undefined in odd
//! - `ValueIn<T, S>` = value only exists in lanes where S is active
//!
//! After merge, values from different branches must be COMBINED into PerLane<T>.
//!
//! # Patterns
//!
//! 1. **Phi nodes**: Like SSA, merge point combines values from branches
//! 2. **Optional values**: `Option<T>` for lanes that didn't compute
//! 3. **Default values**: Provide default for inactive lanes
//! 4. **Assertion**: Require value computed in all lanes

use std::marker::PhantomData;

// ============================================================================
// ACTIVE SET MARKERS (reusing from static_verify)
// ============================================================================

pub trait ActiveSet: Copy + 'static {
    const MASK: u32;
}

pub trait ComplementOf<T>: ActiveSet {}

#[derive(Copy, Clone)]
pub struct All;
impl ActiveSet for All { const MASK: u32 = 0xFFFFFFFF; }

#[derive(Copy, Clone)]
pub struct Even;
impl ActiveSet for Even { const MASK: u32 = 0x55555555; }

#[derive(Copy, Clone)]
pub struct Odd;
impl ActiveSet for Odd { const MASK: u32 = 0xAAAAAAAA; }

impl ComplementOf<Odd> for Even {}
impl ComplementOf<Even> for Odd {}

// ============================================================================
// VALUE TYPES WITH SCOPE
// ============================================================================

/// A value that exists only in lanes where S is active
#[derive(Copy, Clone, Debug)]
pub struct ValueIn<T, S: ActiveSet> {
    values: [Option<T>; 32],
    _scope: PhantomData<S>,
}

impl<T: Copy + Default, S: ActiveSet> ValueIn<T, S> {
    /// Create a value computed in the given active set
    pub fn new(compute: impl Fn(usize) -> T) -> Self {
        let mut values = [None; 32];
        for lane in 0..32 {
            if S::MASK & (1 << lane) != 0 {
                values[lane] = Some(compute(lane));
            }
        }
        ValueIn { values, _scope: PhantomData }
    }

    /// Create a uniform value across the active set
    pub fn uniform(value: T) -> Self {
        let mut values = [None; 32];
        for lane in 0..32 {
            if S::MASK & (1 << lane) != 0 {
                values[lane] = Some(value);
            }
        }
        ValueIn { values, _scope: PhantomData }
    }

    /// Get value for a lane (None if lane not in S)
    pub fn get(&self, lane: usize) -> Option<T> {
        self.values[lane]
    }
}

/// A value guaranteed uniform (same) within active set S
#[derive(Copy, Clone, Debug)]
pub struct UniformIn<T, S: ActiveSet> {
    value: T,
    _scope: PhantomData<S>,
}

impl<T: Copy, S: ActiveSet> UniformIn<T, S> {
    pub fn new(value: T) -> Self {
        UniformIn { value, _scope: PhantomData }
    }

    pub fn get(&self) -> T {
        self.value
    }
}

/// Per-lane values (may differ across all lanes)
#[derive(Copy, Clone, Debug)]
pub struct PerLane<T>(pub [T; 32]);

// ============================================================================
// PHI NODES: MERGING VALUES FROM BRANCHES
// ============================================================================

/// Merge values from complementary branches into PerLane
///
/// This is the "phi node" from SSA: at a merge point, combine values
/// from different paths into a single variable.
pub fn phi_merge<T: Copy + Default, S1: ActiveSet + ComplementOf<S2>, S2: ActiveSet>(
    left: ValueIn<T, S1>,
    right: ValueIn<T, S2>,
) -> PerLane<T> {
    let mut result = [T::default(); 32];
    for lane in 0..32 {
        if S1::MASK & (1 << lane) != 0 {
            result[lane] = left.values[lane].unwrap_or_default();
        } else if S2::MASK & (1 << lane) != 0 {
            result[lane] = right.values[lane].unwrap_or_default();
        }
    }
    PerLane(result)
}

/// Merge uniform values from complementary branches
///
/// Two values that were uniform in their respective branches
/// become per-lane after merge (different lanes have different values).
pub fn phi_merge_uniform<T: Copy + Default, S1: ActiveSet + ComplementOf<S2>, S2: ActiveSet>(
    left: UniformIn<T, S1>,
    right: UniformIn<T, S2>,
) -> PerLane<T> {
    let mut result = [T::default(); 32];
    for lane in 0..32 {
        if S1::MASK & (1 << lane) != 0 {
            result[lane] = left.value;
        } else if S2::MASK & (1 << lane) != 0 {
            result[lane] = right.value;
        }
    }
    PerLane(result)
}

// ============================================================================
// OPTIONAL VALUES: FOR ASYMMETRIC COMPUTATION
// ============================================================================

/// Value only computed in some lanes, with explicit Option
///
/// Unlike ValueIn, this tracks the optionality at runtime too.
#[derive(Copy, Clone, Debug)]
pub struct MaybeValue<T>(pub [Option<T>; 32]);

impl<T: Copy> MaybeValue<T> {
    pub fn new() -> Self {
        MaybeValue([None; 32])
    }

    /// Set value for specific lanes
    pub fn set_where<S: ActiveSet>(&mut self, value: T) {
        for lane in 0..32 {
            if S::MASK & (1 << lane) != 0 {
                self.0[lane] = Some(value);
            }
        }
    }

    /// Get value or default
    pub fn get_or_default(&self, lane: usize, default: T) -> T {
        self.0[lane].unwrap_or(default)
    }

    /// Map to PerLane with default for missing
    pub fn to_per_lane(&self, default: T) -> PerLane<T> {
        let mut result = [default; 32];
        for lane in 0..32 {
            if let Some(v) = self.0[lane] {
                result[lane] = v;
            }
        }
        PerLane(result)
    }
}

// ============================================================================
// STRUCTURED DIVERGENCE WITH VALUE TRACKING
// ============================================================================

/// Warp with tracked active set
#[derive(Copy, Clone)]
pub struct Warp<S: ActiveSet> {
    _marker: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self {
        Warp { _marker: PhantomData }
    }
}

/// Diverge into two branches, compute values, merge
///
/// This combinator handles the full pattern:
/// 1. Split warp into complementary sets
/// 2. Execute each branch, producing a value
/// 3. Merge values into PerLane result
pub fn diverge_compute<T, F1, F2, S1, S2>(
    _warp: Warp<All>,
    _pred: impl Fn(usize) -> bool,
    then_branch: F1,
    else_branch: F2,
) -> PerLane<T>
where
    T: Copy + Default,
    S1: ActiveSet + ComplementOf<S2>,
    S2: ActiveSet + ComplementOf<S1>,
    F1: FnOnce(Warp<S1>) -> ValueIn<T, S1>,
    F2: FnOnce(Warp<S2>) -> ValueIn<T, S2>,
{
    let then_val = then_branch(Warp::new());
    let else_val = else_branch(Warp::new());
    phi_merge(then_val, else_val)
}

// ============================================================================
// REASSIGNMENT IN DIVERGENT CODE
// ============================================================================

/// Track variable state through divergence
///
/// Pattern: Variable X has value before diverge, modified in one branch,
/// unchanged in other. After merge:
/// - Modified lanes have new value
/// - Unchanged lanes have original value
#[derive(Copy, Clone, Debug)]
pub struct TrackedVar<T: Copy> {
    values: [T; 32],
}

impl<T: Copy> TrackedVar<T> {
    pub fn uniform(value: T) -> Self {
        TrackedVar { values: [value; 32] }
    }

    pub fn per_lane(values: [T; 32]) -> Self {
        TrackedVar { values }
    }

    /// Update value in specific lanes
    pub fn update_where<S: ActiveSet>(&mut self, new_value: T) {
        for lane in 0..32 {
            if S::MASK & (1 << lane) != 0 {
                self.values[lane] = new_value;
            }
        }
    }

    /// Update with per-lane values
    pub fn update_where_with<S: ActiveSet>(&mut self, compute: impl Fn(usize) -> T) {
        for lane in 0..32 {
            if S::MASK & (1 << lane) != 0 {
                self.values[lane] = compute(lane);
            }
        }
    }

    pub fn get(&self, lane: usize) -> T {
        self.values[lane]
    }

    pub fn to_per_lane(self) -> PerLane<T> {
        PerLane(self.values)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_in_scope() {
        // Value computed only in even lanes
        let even_val: ValueIn<i32, Even> = ValueIn::uniform(100);

        assert_eq!(even_val.get(0), Some(100));  // Lane 0 is even
        assert_eq!(even_val.get(1), None);       // Lane 1 is odd
        assert_eq!(even_val.get(2), Some(100));  // Lane 2 is even
    }

    #[test]
    fn test_phi_merge_values() {
        // Different values in each branch
        let even_val: ValueIn<i32, Even> = ValueIn::uniform(100);
        let odd_val: ValueIn<i32, Odd> = ValueIn::uniform(200);

        let merged = phi_merge(even_val, odd_val);

        assert_eq!(merged.0[0], 100);  // Even lane
        assert_eq!(merged.0[1], 200);  // Odd lane
        assert_eq!(merged.0[2], 100);  // Even lane
        assert_eq!(merged.0[3], 200);  // Odd lane
    }

    #[test]
    fn test_phi_merge_uniform() {
        // Uniform values become per-lane after merge
        let even_uniform: UniformIn<i32, Even> = UniformIn::new(42);
        let odd_uniform: UniformIn<i32, Odd> = UniformIn::new(99);

        let merged = phi_merge_uniform(even_uniform, odd_uniform);

        // Was uniform in branches, now per-lane
        assert_eq!(merged.0[0], 42);
        assert_eq!(merged.0[1], 99);
        assert_eq!(merged.0[2], 42);
        assert_eq!(merged.0[3], 99);
    }

    #[test]
    fn test_maybe_value_asymmetric() {
        let mut v: MaybeValue<i32> = MaybeValue::new();

        // Only even lanes compute a value
        v.set_where::<Even>(100);

        assert_eq!(v.0[0], Some(100));
        assert_eq!(v.0[1], None);

        // Convert to PerLane with default for missing
        let per_lane = v.to_per_lane(0);
        assert_eq!(per_lane.0[0], 100);
        assert_eq!(per_lane.0[1], 0);  // Default
    }

    #[test]
    fn test_tracked_var_reassignment() {
        // Variable starts uniform
        let mut x = TrackedVar::uniform(0i32);

        // Only even lanes update it
        x.update_where::<Even>(100);

        // After "merge", check values
        assert_eq!(x.get(0), 100);  // Updated in even branch
        assert_eq!(x.get(1), 0);    // Unchanged in odd branch
        assert_eq!(x.get(2), 100);
        assert_eq!(x.get(3), 0);
    }

    #[test]
    fn test_tracked_var_varying_update() {
        let mut x = TrackedVar::uniform(0i32);

        // Even lanes get lane_id * 10
        x.update_where_with::<Even>(|lane| lane as i32 * 10);

        assert_eq!(x.get(0), 0);    // 0 * 10
        assert_eq!(x.get(1), 0);    // Unchanged
        assert_eq!(x.get(2), 20);   // 2 * 10
        assert_eq!(x.get(4), 40);   // 4 * 10
    }

    #[test]
    fn test_nested_divergence_values() {
        // Start with uniform value
        let mut x = TrackedVar::uniform(0i32);

        // First diverge: even lanes set to 1
        x.update_where::<Even>(1);

        // Second diverge within even: low even lanes set to 2
        // (This is simplified - in real code we'd need EvenLow type)
        for lane in 0..16 {
            if lane % 2 == 0 {
                x.values[lane] = 2;
            }
        }

        // Check final state
        assert_eq!(x.get(0), 2);   // EvenLow: updated twice
        assert_eq!(x.get(1), 0);   // Odd: never updated
        assert_eq!(x.get(2), 2);   // EvenLow
        assert_eq!(x.get(16), 1);  // EvenHigh: only first update
        assert_eq!(x.get(17), 0);  // Odd
    }
}

// ============================================================================
// KEY FINDINGS
// ============================================================================

/// Summary of findings for "How to handle values computed in divergent code?"
///
/// ## The Problem
///
/// In divergent code, different lanes may:
/// 1. Compute different values (phi node needed)
/// 2. Compute or not compute (optional value)
/// 3. Modify or keep original (tracked reassignment)
///
/// ## Solutions
///
/// | Pattern | Type | Use Case |
/// |---------|------|----------|
/// | `ValueIn<T, S>` | Value exists only in S | Branch-local computation |
/// | `UniformIn<T, S>` | Uniform within S only | Optimizable branch values |
/// | `phi_merge` | Combine complementary values | SSA-style merge |
/// | `MaybeValue<T>` | Runtime-optional per lane | Asymmetric computation |
/// | `TrackedVar<T>` | Mutable through divergence | Variable reassignment |
///
/// ## Key Insight
///
/// Value SCOPE is tied to active set. A value computed in Even lanes
/// simply doesn't exist in Odd lanes - the type system enforces this.
/// At merge points, values from different branches combine into PerLane.
///
/// This is the GPU analog of SSA phi nodes, but with the active set
/// encoded in the type.
pub const _DOC: () = ();
