//! Uniformity Inference
//!
//! **STATUS: Research** — Exploratory prototype, not promoted to main API.
//!
//! Research question: "Can we infer uniformity or must it be annotated?"
//!
//! # Background
//!
//! Values in GPU warp programming are either:
//! - `Uniform<T>`: Same value across all lanes (can optimize)
//! - `PerLane<T>`: May differ across lanes (default)
//!
//! ISPC (Intel SPMD Program Compiler) uses `uniform` and `varying` keywords.
//! Can we infer these instead of requiring annotations?
//!
//! # Findings
//!
//! ## What Can Be Inferred
//!
//! 1. **Constants are uniform**: `let x = 42;` → `Uniform<i32>`
//! 2. **Lane ID is varying**: `lane_id()` → `PerLane<usize>`
//! 3. **Uniform ops on uniform args**: `uniform + uniform` → `Uniform`
//! 4. **Any varying contaminates**: `uniform + varying` → `PerLane`
//! 5. **Reductions produce uniform**: `reduce_sum(varying)` → `Uniform`
//! 6. **Broadcasts produce uniform**: `broadcast(single_lane)` → `Uniform`
//!
//! ## What Needs Annotation
//!
//! 1. **Function boundaries**: Parameters and returns need explicit types
//! 2. **External data**: Loads from memory depend on address pattern
//! 3. **Unsafe assertions**: `assume_uniform()` requires programmer knowledge
//!
//! # Implementation
//!
//! We implement a simple data-flow analysis that propagates uniformity.

use std::marker::PhantomData;

// ============================================================================
// VALUE TYPES
// ============================================================================

/// A value guaranteed identical across all lanes
#[derive(Copy, Clone, Debug)]
pub struct Uniform<T>(pub T);

/// A value that may differ across lanes
#[derive(Copy, Clone, Debug)]
pub struct PerLane<T>(pub [T; 32]);

/// Trait for values with uniformity information
pub trait Valued {
    type Element;
    fn is_uniform() -> bool;
}

impl<T> Valued for Uniform<T> {
    type Element = T;
    fn is_uniform() -> bool { true }
}

impl<T> Valued for PerLane<T> {
    type Element = T;
    fn is_uniform() -> bool { false }
}

// ============================================================================
// UNIFORMITY RULES
// ============================================================================

/// Rule 1: Constants are uniform
pub fn constant<T: Copy>(value: T) -> Uniform<T> {
    Uniform(value)
}

/// Rule 2: Lane ID is varying
pub fn lane_id() -> PerLane<usize> {
    let mut result = [0usize; 32];
    for i in 0..32 {
        result[i] = i;
    }
    PerLane(result)
}

/// Rule 3: Uniform + Uniform = Uniform
pub fn add_uniform<T: std::ops::Add<Output = T> + Copy>(
    a: Uniform<T>,
    b: Uniform<T>,
) -> Uniform<T> {
    Uniform(a.0 + b.0)
}

/// Rule 4: Uniform + PerLane = PerLane (contamination)
pub fn add_mixed<T: std::ops::Add<Output = T> + Copy>(
    a: Uniform<T>,
    b: PerLane<T>,
) -> PerLane<T> {
    let mut result = [a.0; 32];
    for i in 0..32 {
        result[i] = a.0 + b.0[i];
    }
    PerLane(result)
}

/// Rule 4b: PerLane + PerLane = PerLane
pub fn add_varying<T: std::ops::Add<Output = T> + Copy + Default>(
    a: PerLane<T>,
    b: PerLane<T>,
) -> PerLane<T> {
    let mut result = [T::default(); 32];
    for i in 0..32 {
        result[i] = a.0[i] + b.0[i];
    }
    PerLane(result)
}

/// Rule 5: Reductions produce uniform
pub fn reduce_sum<T: std::ops::Add<Output = T> + Copy + Default>(
    values: PerLane<T>,
) -> Uniform<T> {
    let mut sum = T::default();
    for i in 0..32 {
        sum = sum + values.0[i];
    }
    Uniform(sum)
}

/// Rule 6: Broadcast produces uniform
pub fn broadcast<T: Copy>(value: T, _from_lane: usize) -> Uniform<T> {
    Uniform(value)
}

// ============================================================================
// INFERENCE ENGINE (SIMPLE DATA-FLOW)
// ============================================================================

/// Uniformity state for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uniformity {
    Uniform,
    Varying,
    Unknown,
}

impl Uniformity {
    /// Meet operation: Uniform ∧ Uniform = Uniform, else Varying
    pub fn meet(self, other: Uniformity) -> Uniformity {
        match (self, other) {
            (Uniformity::Uniform, Uniformity::Uniform) => Uniformity::Uniform,
            (Uniformity::Unknown, x) | (x, Uniformity::Unknown) => x,
            _ => Uniformity::Varying,
        }
    }

    /// Join operation for control flow merge
    pub fn join(self, other: Uniformity) -> Uniformity {
        match (self, other) {
            (Uniformity::Uniform, Uniformity::Uniform) => Uniformity::Uniform,
            (Uniformity::Unknown, x) | (x, Uniformity::Unknown) => x,
            _ => Uniformity::Varying,
        }
    }
}

/// Expression for inference demo
#[derive(Debug, Clone)]
pub enum Expr {
    Const(i32),
    LaneId,
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    ReduceSum(Box<Expr>),
    Broadcast(Box<Expr>),
    Load(Box<Expr>),  // Load from address
}

/// Infer uniformity of an expression
pub fn infer_uniformity(expr: &Expr, env: &std::collections::HashMap<String, Uniformity>) -> Uniformity {
    match expr {
        // Constants are uniform
        Expr::Const(_) => Uniformity::Uniform,

        // Lane ID is varying
        Expr::LaneId => Uniformity::Varying,

        // Variables: look up in environment
        Expr::Var(name) => env.get(name).copied().unwrap_or(Uniformity::Unknown),

        // Binary ops: meet of operands
        Expr::Add(a, b) | Expr::Mul(a, b) => {
            let ua = infer_uniformity(a, env);
            let ub = infer_uniformity(b, env);
            ua.meet(ub)
        }

        // Reductions produce uniform
        Expr::ReduceSum(_) => Uniformity::Uniform,

        // Broadcast produces uniform
        Expr::Broadcast(_) => Uniformity::Uniform,

        // Load: depends on address uniformity
        Expr::Load(addr) => {
            let addr_uniformity = infer_uniformity(addr, env);
            match addr_uniformity {
                // Uniform address = all lanes load same value = uniform
                Uniformity::Uniform => Uniformity::Uniform,
                // Varying address = each lane loads different value = varying
                _ => Uniformity::Varying,
            }
        }
    }
}

// ============================================================================
// INTERACTION WITH DIVERGENCE
// ============================================================================

/// Key insight: Uniformity interacts with divergence!
///
/// After diverge, a "uniform" value is only uniform within the active lanes.
/// When we merge, values from different branches may differ.
///
/// Example:
/// ```text
/// let x: Uniform<i32> = 42;  // Uniform across all lanes
/// diverge(even) {
///     x = 100;  // Uniform within Even lanes
/// } else {
///     x = 200;  // Uniform within Odd lanes
/// } merge;
/// // x is now PerLane! (100 in even lanes, 200 in odd lanes)
/// ```
///
/// This suggests: Uniform<T> should be Uniform<T, S> where S is the active set.
/// A value uniform across Even lanes is not the same as uniform across All.

/// Uniform within a specific active set
#[derive(Copy, Clone, Debug)]
pub struct UniformWithin<T, S> {
    value: T,
    _set: PhantomData<S>,
}

/// Marker for active sets (reusing from static_verify)
pub trait ActiveSet {}
pub struct All;
pub struct Even;
pub struct Odd;
impl ActiveSet for All {}
impl ActiveSet for Even {}
impl ActiveSet for Odd {}

impl<T: Copy, S: ActiveSet> UniformWithin<T, S> {
    pub fn new(value: T) -> Self {
        UniformWithin { value, _set: PhantomData }
    }

    pub fn get(&self) -> T {
        self.value
    }
}

/// Merging uniform values from different branches produces PerLane
pub fn merge_uniform<T: Copy + Default>(
    _even_val: UniformWithin<T, Even>,
    _odd_val: UniformWithin<T, Odd>,
) -> PerLane<T> {
    // Each lane gets the value from its branch
    // Result is no longer uniform!
    PerLane([T::default(); 32])  // Placeholder
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_constant_is_uniform() {
        let env = HashMap::new();
        let expr = Expr::Const(42);
        assert_eq!(infer_uniformity(&expr, &env), Uniformity::Uniform);
    }

    #[test]
    fn test_lane_id_is_varying() {
        let env = HashMap::new();
        let expr = Expr::LaneId;
        assert_eq!(infer_uniformity(&expr, &env), Uniformity::Varying);
    }

    #[test]
    fn test_uniform_plus_uniform() {
        let env = HashMap::new();
        let expr = Expr::Add(
            Box::new(Expr::Const(1)),
            Box::new(Expr::Const(2)),
        );
        assert_eq!(infer_uniformity(&expr, &env), Uniformity::Uniform);
    }

    #[test]
    fn test_uniform_plus_varying() {
        let env = HashMap::new();
        let expr = Expr::Add(
            Box::new(Expr::Const(1)),
            Box::new(Expr::LaneId),
        );
        assert_eq!(infer_uniformity(&expr, &env), Uniformity::Varying);
    }

    #[test]
    fn test_reduce_is_uniform() {
        let env = HashMap::new();
        let expr = Expr::ReduceSum(Box::new(Expr::LaneId));
        assert_eq!(infer_uniformity(&expr, &env), Uniformity::Uniform);
    }

    #[test]
    fn test_load_uniform_address() {
        let env = HashMap::new();
        // Load from constant address -> uniform
        let expr = Expr::Load(Box::new(Expr::Const(0)));
        assert_eq!(infer_uniformity(&expr, &env), Uniformity::Uniform);
    }

    #[test]
    fn test_load_varying_address() {
        let env = HashMap::new();
        // Load from lane-dependent address -> varying
        let expr = Expr::Load(Box::new(Expr::LaneId));
        assert_eq!(infer_uniformity(&expr, &env), Uniformity::Varying);
    }

    #[test]
    fn test_divergence_breaks_uniformity() {
        // After diverge+merge, values that were uniform in branches become varying
        let even_val: UniformWithin<i32, Even> = UniformWithin::new(100);
        let odd_val: UniformWithin<i32, Odd> = UniformWithin::new(200);

        let merged = merge_uniform(even_val, odd_val);
        // Result is PerLane, not Uniform!
        let _: PerLane<i32> = merged;
    }
}
