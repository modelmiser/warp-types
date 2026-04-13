//! BMC unrolling: time-index state variables and encode into SAT.
//!
//! Given a transition system with `n` state variables, unrolling to depth `k`
//! creates SAT variables `[0, (k+1)*n)` where variable `v + t*n` represents
//! state variable `v` at time step `t`.
//!
//! The SAT instance at depth k is:
//!   I(s₀) ∧ T(s₀,s₁) ∧ T(s₁,s₂) ∧ ... ∧ T(sₖ₋₁,sₖ) ∧ ¬P(sₖ)

use warp_types_sat::bcp::ClauseDb;
use warp_types_sat::literal::Lit;

use crate::model::TransitionSystem;

/// Time-shift a literal by `offset` state variables.
/// Maps state var `v` at time 0 to SAT var `v + offset`.
fn shift_lit(lit: Lit, offset: u32) -> Lit {
    let var = lit.var() + offset;
    if lit.is_negated() {
        Lit::neg(var)
    } else {
        Lit::pos(var)
    }
}

/// Encode a BMC instance at depth `k` into a SAT clause database.
///
/// Returns `(clause_db, num_sat_vars)`.
///
/// The encoding is:
/// - I(s₀): initial clauses with vars in [0, n)
/// - T(sₜ, sₜ₊₁) for t in 0..k: transition clauses shifted to [t*n, (t+2)*n)
/// - ¬P(sₖ): negated property at the final frame
///
/// Property negation: if P = c₁ ∧ c₂ ∧ ... ∧ cₘ, then ¬P = ¬c₁ ∨ ¬c₂ ∨ ... ∨ ¬cₘ.
/// Each cᵢ is a disjunction of literals, so ¬cᵢ is a conjunction of negated literals.
/// We introduce a Tseitin variable for each clause and assert their disjunction.
pub fn encode_bmc(sys: &TransitionSystem, depth: u32) -> (ClauseDb, u32) {
    let n = sys.num_state_vars;
    // State vars: (depth+1) * n for frames 0..=depth
    // Plus Tseitin vars for property negation
    let num_state_sat_vars = (depth + 1) * n;
    let num_tseitin = sys.property.len() as u32;
    let total_vars = num_state_sat_vars + num_tseitin;

    let mut db = ClauseDb::new();

    // ── Initial state: I(s₀) ──
    for clause in &sys.initial {
        let lits: Vec<Lit> = clause.lits.iter().map(|&l| shift_lit(l, 0)).collect();
        db.add_clause(lits);
    }

    // ── Transition relation: T(sₜ, sₜ₊₁) for t = 0..depth ──
    for t in 0..depth {
        let current_offset = t * n;
        for tc in &sys.transition {
            let lits: Vec<Lit> = tc.lits.iter().map(|&l| {
                let v = l.var();
                if v < n {
                    // Current-state variable → offset by t*n
                    shift_lit(l, current_offset)
                } else {
                    // Next-state variable (v >= n) → offset by t*n
                    // Original: v in [n, 2n), after shift: v - n + (t+1)*n = v + t*n
                    shift_lit(l, current_offset)
                }
            }).collect();
            db.add_clause(lits);
        }
    }

    // ── Negated property: ¬P(sₖ) ──
    // P = c₁ ∧ c₂ ∧ ... ∧ cₘ where each cᵢ is a clause.
    // ¬P = ¬c₁ ∨ ¬c₂ ∨ ... ∨ ¬cₘ
    // Tseitin encoding: for each cᵢ = (l₁ ∨ ... ∨ lⱼ):
    //   tᵢ → ¬l₁ ∧ ... ∧ ¬lⱼ  (if tᵢ true, all literals in cᵢ are false)
    //   i.e., for each lⱼ: (¬tᵢ ∨ ¬lⱼ)
    // Plus: (t₁ ∨ t₂ ∨ ... ∨ tₘ)  (at least one clause is falsified)
    let prop_offset = depth * n;
    let tseitin_base = num_state_sat_vars;

    // Activation clause: at least one property clause must be violated
    let activation: Vec<Lit> = (0..num_tseitin)
        .map(|i| Lit::pos(tseitin_base + i))
        .collect();
    if !activation.is_empty() {
        db.add_clause(activation);
    }

    // Per-clause implications: tᵢ → all literals in cᵢ are false
    for (i, clause) in sys.property.iter().enumerate() {
        let t_var = tseitin_base + i as u32;
        for &lit in &clause.lits {
            let shifted = shift_lit(lit, prop_offset);
            // (¬tᵢ ∨ ¬shifted_lit)
            db.add_clause(vec![Lit::neg(t_var), shifted.complement()]);
        }
    }

    (db, total_vars)
}

/// Extract a counterexample trace from a SAT assignment.
///
/// Returns a vector of frames, where each frame is the state variable
/// assignments at that time step.
pub fn extract_trace(
    assignment: &[bool],
    num_state_vars: u32,
    depth: u32,
) -> Vec<Vec<bool>> {
    let n = num_state_vars as usize;
    (0..=depth as usize)
        .map(|t| {
            let offset = t * n;
            assignment[offset..offset + n].to_vec()
        })
        .collect()
}
