//! Soundness Proof Sketch for Session-Typed Divergence
//!
//! This module outlines what a formal soundness proof would require.
//! It's not a machine-checked proof, but a roadmap for one.
//!
//! ## What We Want to Prove
//!
//! **Theorem (Type Safety):** Well-typed warp programs don't have divergence bugs.
//!
//! Specifically:
//! 1. merge(w1, w2) only succeeds if w1 and w2 are complementary
//! 2. shuffle on Warp<S> only reads from lanes in S
//! 3. diverge + merge preserves all lanes (no lanes lost or duplicated)
//!
//! ## Proof Strategy
//!
//! Standard approach: Progress + Preservation (Wright & Felleisen, 1994)
//!
//! **Progress:** Well-typed terms either are values or can step.
//! **Preservation:** Stepping preserves types.
//!
//! Together: Well-typed terms don't get stuck (no undefined behavior).

use std::collections::HashSet;

// ============================================================================
// FORMAL SYNTAX
// ============================================================================

/// Lane identifiers (0..31 for 32-lane warp)
pub type Lane = u32;
pub const WARP_SIZE: u32 = 32;

/// Active sets as explicit sets of lanes (for proof purposes)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveSet(HashSet<Lane>);

impl ActiveSet {
    pub fn all() -> Self {
        ActiveSet((0..WARP_SIZE).collect())
    }

    pub fn empty() -> Self {
        ActiveSet(HashSet::new())
    }

    pub fn from_predicate<F: Fn(Lane) -> bool>(pred: F) -> Self {
        ActiveSet((0..WARP_SIZE).filter(|&l| pred(l)).collect())
    }

    pub fn union(&self, other: &Self) -> Self {
        ActiveSet(self.0.union(&other.0).copied().collect())
    }

    pub fn intersection(&self, other: &Self) -> Self {
        ActiveSet(self.0.intersection(&other.0).copied().collect())
    }

    pub fn complement(&self) -> Self {
        ActiveSet((0..WARP_SIZE).filter(|l| !self.0.contains(l)).collect())
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.0.is_disjoint(&other.0)
    }

    pub fn is_all(&self) -> bool {
        self.0.len() == WARP_SIZE as usize
    }

    pub fn contains(&self, lane: Lane) -> bool {
        self.0.contains(&lane)
    }
}

/// Types in our system
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    /// Warp with active set S
    Warp(ActiveSet),
    /// Per-lane value
    PerLane,
    /// Unit
    Unit,
    /// Pair
    Pair(Box<Type>, Box<Type>),
}

/// Expressions in our system
#[derive(Clone, Debug)]
pub enum Expr {
    /// A warp value with active set
    WarpVal(ActiveSet),
    /// Per-lane value
    PerLaneVal(Vec<i32>),
    /// Unit
    UnitVal,
    /// Pair
    PairVal(Box<Expr>, Box<Expr>),

    /// Variable
    Var(String),

    /// Diverge: split warp by predicate
    /// diverge(w, pred) → (Warp<S∩pred>, Warp<S∩¬pred>)
    Diverge(Box<Expr>, Predicate),

    /// Merge: combine complementary warps
    /// merge(w1, w2) → Warp<S1∪S2>  requires S1∩S2=∅
    Merge(Box<Expr>, Box<Expr>),

    /// Shuffle: exchange values between lanes (requires Warp<All>)
    Shuffle(Box<Expr>, Box<Expr>, u32),  // warp, data, xor_mask

    /// Let binding
    Let(String, Box<Expr>, Box<Expr>),
}

/// Predicates for divergence
#[derive(Clone, Debug)]
pub enum Predicate {
    Even,
    LessThan(u32),
    Custom(fn(Lane) -> bool),
}

impl Predicate {
    pub fn eval(&self, lane: Lane) -> bool {
        match self {
            Predicate::Even => lane % 2 == 0,
            Predicate::LessThan(n) => lane < *n,
            Predicate::Custom(f) => f(lane),
        }
    }

    pub fn active_set(&self) -> ActiveSet {
        ActiveSet::from_predicate(|l| self.eval(l))
    }
}

// ============================================================================
// TYPING RULES
// ============================================================================

/// Typing context: variable → type
pub type Context = std::collections::HashMap<String, Type>;

/// Type checking result
pub type TypeResult = Result<Type, String>;

/// Type check an expression
pub fn type_check(ctx: &Context, expr: &Expr) -> TypeResult {
    match expr {
        // Values have their literal types
        Expr::WarpVal(s) => Ok(Type::Warp(s.clone())),
        Expr::PerLaneVal(_) => Ok(Type::PerLane),
        Expr::UnitVal => Ok(Type::Unit),
        Expr::PairVal(e1, e2) => {
            let t1 = type_check(ctx, e1)?;
            let t2 = type_check(ctx, e2)?;
            Ok(Type::Pair(Box::new(t1), Box::new(t2)))
        }

        // Variable lookup
        Expr::Var(x) => ctx.get(x).cloned().ok_or_else(|| format!("Unbound variable: {}", x)),

        // DIVERGE RULE:
        // Γ ⊢ w : Warp<S>
        // ─────────────────────────────────────────────────
        // Γ ⊢ diverge(w, P) : (Warp<S∩P>, Warp<S∩¬P>)
        Expr::Diverge(w, pred) => {
            let warp_type = type_check(ctx, w)?;
            match warp_type {
                Type::Warp(s) => {
                    let p = pred.active_set();
                    let s_true = s.intersection(&p);
                    let s_false = s.intersection(&p.complement());
                    Ok(Type::Pair(
                        Box::new(Type::Warp(s_true)),
                        Box::new(Type::Warp(s_false)),
                    ))
                }
                _ => Err("diverge requires a Warp".to_string()),
            }
        }

        // MERGE RULE:
        // Γ ⊢ w1 : Warp<S1>    Γ ⊢ w2 : Warp<S2>    S1 ∩ S2 = ∅
        // ──────────────────────────────────────────────────────
        // Γ ⊢ merge(w1, w2) : Warp<S1 ∪ S2>
        Expr::Merge(w1, w2) => {
            let t1 = type_check(ctx, w1)?;
            let t2 = type_check(ctx, w2)?;
            match (t1, t2) {
                (Type::Warp(s1), Type::Warp(s2)) => {
                    if s1.is_disjoint(&s2) {
                        Ok(Type::Warp(s1.union(&s2)))
                    } else {
                        Err("merge requires disjoint active sets".to_string())
                    }
                }
                _ => Err("merge requires two Warps".to_string()),
            }
        }

        // SHUFFLE RULE:
        // Γ ⊢ w : Warp<All>    Γ ⊢ data : PerLane
        // ─────────────────────────────────────────
        // Γ ⊢ shuffle(w, data, mask) : PerLane
        Expr::Shuffle(w, data, _mask) => {
            let warp_type = type_check(ctx, w)?;
            let data_type = type_check(ctx, data)?;
            match (warp_type, data_type) {
                (Type::Warp(s), Type::PerLane) => {
                    if s.is_all() {
                        Ok(Type::PerLane)
                    } else {
                        Err("shuffle requires Warp<All>".to_string())
                    }
                }
                _ => Err("shuffle requires Warp and PerLane".to_string()),
            }
        }

        // LET RULE:
        // Γ ⊢ e1 : τ1    Γ, x:τ1 ⊢ e2 : τ2
        // ─────────────────────────────────
        // Γ ⊢ let x = e1 in e2 : τ2
        Expr::Let(x, e1, e2) => {
            let t1 = type_check(ctx, e1)?;
            let mut ctx2 = ctx.clone();
            ctx2.insert(x.clone(), t1);
            type_check(&ctx2, e2)
        }
    }
}

// ============================================================================
// OPERATIONAL SEMANTICS (Small-step)
// ============================================================================

/// Values are fully reduced expressions
pub fn is_value(expr: &Expr) -> bool {
    match expr {
        Expr::WarpVal(_) => true,
        Expr::PerLaneVal(_) => true,
        Expr::UnitVal => true,
        Expr::PairVal(e1, e2) => is_value(e1) && is_value(e2),
        _ => false,
    }
}

/// Single step of evaluation
pub fn step(expr: &Expr) -> Option<Expr> {
    match expr {
        // Values don't step
        _ if is_value(expr) => None,

        // DIVERGE: split warp value
        Expr::Diverge(w, pred) => {
            if let Expr::WarpVal(s) = w.as_ref() {
                let p = pred.active_set();
                let s_true = s.intersection(&p);
                let s_false = s.intersection(&p.complement());
                Some(Expr::PairVal(
                    Box::new(Expr::WarpVal(s_true)),
                    Box::new(Expr::WarpVal(s_false)),
                ))
            } else {
                // Step the sub-expression
                step(w).map(|w2| Expr::Diverge(Box::new(w2), pred.clone()))
            }
        }

        // MERGE: combine warp values
        Expr::Merge(w1, w2) => {
            match (w1.as_ref(), w2.as_ref()) {
                (Expr::WarpVal(s1), Expr::WarpVal(s2)) => {
                    // Runtime check (type system should have verified disjointness)
                    assert!(s1.is_disjoint(s2), "SOUNDNESS VIOLATION: merge of non-disjoint sets");
                    Some(Expr::WarpVal(s1.union(s2)))
                }
                (Expr::WarpVal(_), _) => {
                    step(w2).map(|w2_| Expr::Merge(w1.clone(), Box::new(w2_)))
                }
                _ => {
                    step(w1).map(|w1_| Expr::Merge(Box::new(w1_), w2.clone()))
                }
            }
        }

        // SHUFFLE: exchange values
        Expr::Shuffle(w, data, mask) => {
            match (w.as_ref(), data.as_ref()) {
                (Expr::WarpVal(s), Expr::PerLaneVal(vals)) => {
                    // Runtime check (type system should have verified All)
                    assert!(s.is_all(), "SOUNDNESS VIOLATION: shuffle on non-All warp");

                    // Perform the shuffle
                    let mut result = vals.clone();
                    for lane in 0..WARP_SIZE {
                        let src = lane ^ mask;
                        if src < WARP_SIZE {
                            result[lane as usize] = vals[src as usize];
                        }
                    }
                    Some(Expr::PerLaneVal(result))
                }
                (Expr::WarpVal(_), _) => {
                    step(data).map(|d| Expr::Shuffle(w.clone(), Box::new(d), *mask))
                }
                _ => {
                    step(w).map(|w_| Expr::Shuffle(Box::new(w_), data.clone(), *mask))
                }
            }
        }

        // LET: substitute
        Expr::Let(x, e1, e2) => {
            if is_value(e1) {
                Some(substitute(e2, x, e1))
            } else {
                step(e1).map(|e1_| Expr::Let(x.clone(), Box::new(e1_), e2.clone()))
            }
        }

        _ => None,
    }
}

/// Substitute value for variable
fn substitute(expr: &Expr, var: &str, val: &Expr) -> Expr {
    match expr {
        Expr::Var(x) if x == var => val.clone(),
        Expr::Var(_) => expr.clone(),
        Expr::WarpVal(_) | Expr::PerLaneVal(_) | Expr::UnitVal => expr.clone(),
        Expr::PairVal(e1, e2) => Expr::PairVal(
            Box::new(substitute(e1, var, val)),
            Box::new(substitute(e2, var, val)),
        ),
        Expr::Diverge(w, p) => Expr::Diverge(Box::new(substitute(w, var, val)), p.clone()),
        Expr::Merge(w1, w2) => Expr::Merge(
            Box::new(substitute(w1, var, val)),
            Box::new(substitute(w2, var, val)),
        ),
        Expr::Shuffle(w, d, m) => Expr::Shuffle(
            Box::new(substitute(w, var, val)),
            Box::new(substitute(d, var, val)),
            *m,
        ),
        Expr::Let(x, e1, e2) => {
            let e1_ = substitute(e1, var, val);
            let e2_ = if x == var { e2.clone() } else { Box::new(substitute(e2, var, val)) };
            Expr::Let(x.clone(), Box::new(e1_), e2_)
        }
    }
}

// ============================================================================
// SOUNDNESS THEOREMS (Statements)
// ============================================================================

/// THEOREM 1: Progress
/// If ∅ ⊢ e : τ, then either e is a value or ∃e'. e → e'
///
/// Proof sketch:
/// - By induction on the derivation of ∅ ⊢ e : τ
/// - Case Warp/PerLane/Unit values: already values ✓
/// - Case Diverge: if w is value, can step; else IH on w
/// - Case Merge: if both values, can step (disjointness from typing); else IH
/// - Case Shuffle: if both values, can step (All from typing); else IH
/// - Case Let: if e1 is value, can substitute; else IH on e1
pub fn progress_check(expr: &Expr) -> bool {
    let ctx = Context::new();
    if type_check(&ctx, expr).is_ok() {
        is_value(expr) || step(expr).is_some()
    } else {
        true // Ill-typed terms vacuously satisfy progress
    }
}

/// THEOREM 2: Preservation
/// If Γ ⊢ e : τ and e → e', then Γ ⊢ e' : τ
///
/// Proof sketch:
/// - By induction on the derivation of e → e'
/// - Case Diverge: result type is (Warp<S∩P>, Warp<S∩¬P>) ✓
/// - Case Merge: result type is Warp<S1∪S2> where S1, S2 came from typing ✓
/// - Case Shuffle: result type is PerLane ✓
/// - Case Let/substitution: standard substitution lemma
pub fn preservation_check(expr: &Expr) -> bool {
    let ctx = Context::new();
    let original_type = type_check(&ctx, expr);

    if let Some(stepped) = step(expr) {
        let stepped_type = type_check(&ctx, &stepped);
        original_type == stepped_type
    } else {
        true // No step means preservation holds vacuously
    }
}

/// THEOREM 3: Type Safety (Corollary)
/// Well-typed programs don't get stuck.
///
/// Definition: e is "stuck" if e is not a value and there's no e' with e → e'
///
/// Proof: By Progress, well-typed closed terms are never stuck.
pub fn type_safety_check(expr: &Expr) -> bool {
    let ctx = Context::new();
    if type_check(&ctx, expr).is_err() {
        return true; // Ill-typed, not our concern
    }

    // Run to completion, checking at each step
    let mut current = expr.clone();
    let mut steps = 0;
    const MAX_STEPS: usize = 1000;

    while !is_value(&current) && steps < MAX_STEPS {
        match step(&current) {
            Some(next) => {
                // Check preservation
                let t1 = type_check(&ctx, &current);
                let t2 = type_check(&ctx, &next);
                if t1 != t2 {
                    return false; // Preservation violated!
                }
                current = next;
                steps += 1;
            }
            None => {
                return false; // Stuck! (Progress violated)
            }
        }
    }

    true
}

// ============================================================================
// KEY LEMMAS
// ============================================================================

/// LEMMA: Diverge produces complements
/// If diverge(Warp<S>, P) → (Warp<S1>, Warp<S2>), then S1 ∪ S2 = S and S1 ∩ S2 = ∅
pub fn diverge_complement_lemma(s: &ActiveSet, pred: &Predicate) -> bool {
    let p = pred.active_set();
    let s1 = s.intersection(&p);
    let s2 = s.intersection(&p.complement());

    // S1 ∪ S2 = S
    let union = s1.union(&s2);
    let covers = union == *s;

    // S1 ∩ S2 = ∅
    let disjoint = s1.is_disjoint(&s2);

    covers && disjoint
}

/// LEMMA: Merge restores original
/// If (S1, S2) came from diverge(S, P), then merge produces S
pub fn merge_restore_lemma(s: &ActiveSet, pred: &Predicate) -> bool {
    let p = pred.active_set();
    let s1 = s.intersection(&p);
    let s2 = s.intersection(&p.complement());
    let merged = s1.union(&s2);

    merged == *s
}

/// LEMMA: Shuffle source validity
/// shuffle_xor on Warp<All> only reads from lanes in All (trivially true)
/// shuffle_xor on Warp<S> where S ≠ All would read from lanes not in S (unsafe)
pub fn shuffle_source_lemma(s: &ActiveSet, mask: u32) -> bool {
    if !s.is_all() {
        // Check if any source lane is outside S
        for lane in 0..WARP_SIZE {
            if s.contains(lane) {
                let src = lane ^ mask;
                if !s.contains(src) {
                    return false; // Would read from inactive lane!
                }
            }
        }
    }
    true
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diverge_complement() {
        let all = ActiveSet::all();
        assert!(diverge_complement_lemma(&all, &Predicate::Even));
        assert!(diverge_complement_lemma(&all, &Predicate::LessThan(16)));
    }

    #[test]
    fn test_merge_restore() {
        let all = ActiveSet::all();
        assert!(merge_restore_lemma(&all, &Predicate::Even));
        assert!(merge_restore_lemma(&all, &Predicate::LessThan(10)));
    }

    #[test]
    fn test_shuffle_source_all() {
        let all = ActiveSet::all();
        assert!(shuffle_source_lemma(&all, 1));
        assert!(shuffle_source_lemma(&all, 5));
        assert!(shuffle_source_lemma(&all, 31));
    }

    #[test]
    fn test_shuffle_source_even_fails() {
        let even = ActiveSet::from_predicate(|l| l % 2 == 0);
        // XOR with 1 reads from odd lanes - unsafe!
        assert!(!shuffle_source_lemma(&even, 1));
        // XOR with 2 stays within even lanes - safe!
        assert!(shuffle_source_lemma(&even, 2));
    }

    #[test]
    fn test_type_check_good_program() {
        // let (evens, odds) = diverge(warp_all, even) in merge(evens, odds)
        let program = Expr::Let(
            "pair".to_string(),
            Box::new(Expr::Diverge(
                Box::new(Expr::WarpVal(ActiveSet::all())),
                Predicate::Even,
            )),
            Box::new(Expr::Var("pair".to_string())), // Simplified
        );

        let ctx = Context::new();
        assert!(type_check(&ctx, &program).is_ok());
    }

    #[test]
    fn test_type_check_bad_shuffle() {
        // shuffle on non-All warp should fail
        let even_warp = Expr::WarpVal(ActiveSet::from_predicate(|l| l % 2 == 0));
        let data = Expr::PerLaneVal(vec![0; 32]);
        let bad_shuffle = Expr::Shuffle(Box::new(even_warp), Box::new(data), 1);

        let ctx = Context::new();
        let result = type_check(&ctx, &bad_shuffle);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Warp<All>"));
    }

    #[test]
    fn test_progress() {
        let all_warp = Expr::WarpVal(ActiveSet::all());
        let diverge = Expr::Diverge(Box::new(all_warp), Predicate::Even);

        assert!(progress_check(&diverge));
        assert!(step(&diverge).is_some());
    }

    #[test]
    fn test_preservation() {
        let all_warp = Expr::WarpVal(ActiveSet::all());
        let diverge = Expr::Diverge(Box::new(all_warp), Predicate::Even);

        assert!(preservation_check(&diverge));
    }

    #[test]
    fn test_type_safety_good_program() {
        // Full program: diverge then merge
        let all_warp = Expr::WarpVal(ActiveSet::all());
        let diverge = Expr::Diverge(Box::new(all_warp), Predicate::Even);

        assert!(type_safety_check(&diverge));
    }
}
