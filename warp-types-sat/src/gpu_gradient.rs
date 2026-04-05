//! GPU kernel for clause-parallel gradient SAT (Axis 1).
//!
//! Maps the inner loop of `gradient_search` onto warp-level parallelism:
//! each lane evaluates one clause's product-form loss, then `reduce_sum`
//! aggregates across the warp.
//!
//! # Data Layout
//!
//! Structure-of-Arrays (SoA) for coalesced GPU memory access. Three
//! variable-index arrays + three negation arrays (3-SAT specific).
//! Consecutive lanes read consecutive addresses — 10-100x faster than
//! scattered Array-of-Structures reads on GPU.
//!
//! # Testing Strategy
//!
//! `SimWarp` provides real multi-lane shuffle semantics on CPU. The
//! single-threaded `GpuShuffle` path (identity) is NOT useful here —
//! `reduce_sum` returns `val × 32` instead of the actual sum.
//!
//! # GPU Target
//!
//! The kernel body compiles to PTX via `cargo build --target nvptx64-nvidia-cuda`.
//! Each CUDA thread = one lane, each cooperative groups tile = one warp.
//! The SimWarp tests validate the exact same math that runs on GPU.
//!
//! # Primitive Mapping (from gradient.rs design doc)
//!
//! ```text
//! Axis 1 — Clause-parallel loss evaluation:
//!   PerLane<f64>  →  per-clause loss (3 loads + 2 multiplies)
//!   reduce_sum    →  batch loss aggregation (5 butterfly steps)
//!   Uniform<f64>  →  batch total (same in every lane)
//!
//! Axis 2 — Ballot verification (future):
//!   ballot        →  discrete satisfaction bitmask
//!
//! Axis 3 — Confidence ranking (future):
//!   bitonic_sort  →  variable confidence for CDCL seeding
//! ```

use crate::bcp::ClauseDb;
use crate::literal::Lit;

const WARP_SIZE: usize = 32;

// ─── SoA Clause Data ─────────────────────────────────────────────────

/// Structure-of-Arrays layout for GPU clause evaluation.
///
/// Stores 3-SAT clauses in three parallel variable-index arrays + three
/// negation arrays. Padded to a multiple of `WARP_SIZE` (32) so every
/// warp batch is full — padding lanes contribute zero loss.
#[derive(Debug, Clone)]
pub struct ClauseDataSoA {
    /// Variable index for literal position 0, 1, 2 of each clause.
    pub vars: [Vec<u32>; 3],
    /// True if literal position 0, 1, 2 is negated.
    pub negs: [Vec<bool>; 3],
    /// Per-clause weights.
    pub weights: Vec<f64>,
    /// Number of real (non-padding) clauses.
    pub num_clauses: usize,
    /// Total padded length (multiple of WARP_SIZE).
    pub padded_len: usize,
}

impl ClauseDataSoA {
    /// Pack a `ClauseDb` into SoA layout for GPU access.
    ///
    /// Only processes clauses with exactly 3 literals (3-SAT).
    /// Clauses with other widths are skipped — the count of skipped
    /// clauses is returned alongside the packed data.
    ///
    /// Padding clauses have `weight = 0.0` — they contribute zero loss
    /// regardless of variable values.
    pub fn from_clause_db(db: &ClauseDb, clause_weights: &[f64]) -> (Self, usize) {
        let mut vars = [Vec::new(), Vec::new(), Vec::new()];
        let mut negs = [Vec::new(), Vec::new(), Vec::new()];
        let mut weights = Vec::new();
        let mut skipped = 0usize;

        for ci in 0..db.len() {
            let lits = &db.clause(ci).literals;
            if lits.len() != 3 {
                skipped += 1;
                continue;
            }
            for pos in 0..3 {
                vars[pos].push(lits[pos].var());
                negs[pos].push(lits[pos].is_negated());
            }
            weights.push(clause_weights[ci]);
        }

        let num_clauses = weights.len();
        // Round up to next multiple of WARP_SIZE (handles num_clauses=0 correctly: 0/32*32=0).
        let padded_len = (num_clauses + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;

        // Pad to warp boundary with zero-weight dummy clauses.
        for _ in num_clauses..padded_len {
            for pos in 0..3 {
                vars[pos].push(0);
                negs[pos].push(false);
            }
            weights.push(0.0);
        }

        (
            ClauseDataSoA {
                vars,
                negs,
                weights,
                num_clauses,
                padded_len,
            },
            skipped,
        )
    }

    /// Number of warp batches needed to cover all clauses.
    pub fn num_batches(&self) -> usize {
        self.padded_len / WARP_SIZE
    }
}

// ─── Per-lane clause loss (kernel body) ──────────────────────────────

/// Compute the product-form loss for clause `ci`.
///
/// This is the per-lane kernel body — identical on CPU and GPU.
/// For 3-SAT: 3 variable loads + 2 multiplies + 1 weight multiply.
///
/// The falseness term for a literal:
/// - Positive `x_j`: `1 - x_j` (zero when `x_j = 1`, i.e. literal true)
/// - Negative `¬x_j`: `x_j` (zero when `x_j = 0`, i.e. literal true)
///
/// Clause loss = `weight * term0 * term1 * term2`.
/// Zero iff any literal is fully satisfied.
#[inline]
fn clause_loss(data: &ClauseDataSoA, x: &[f64], ci: usize) -> f64 {
    let mut product = 1.0f64;
    for pos in 0..3 {
        let v = x[data.vars[pos][ci] as usize];
        let term = if data.negs[pos][ci] { v } else { 1.0 - v };
        product *= term;
    }
    data.weights[ci] * product
}

// ─── SimWarp loss evaluation ─────────────────────────────────────────

/// Evaluate total loss using SimWarp with real lane exchange.
///
/// Each warp batch: 32 lanes compute clause losses in parallel,
/// butterfly reduce sums them. This matches the GPU kernel exactly —
/// the same math, the same reduction tree, the same result.
pub fn total_loss_simwarp(data: &ClauseDataSoA, x: &[f64]) -> f64 {
    use warp_types::simwarp::{butterfly_reduce, SimWarp};

    let mut total = 0.0f64;
    for batch in 0..data.num_batches() {
        let offset = batch * WARP_SIZE;
        let losses = SimWarp::<f64>::new(|lane| clause_loss(data, x, offset + lane as usize));
        let reduced = butterfly_reduce(&losses, |a, b| a + b);
        // After full butterfly reduction, all 32 lanes hold the same sum.
        total += reduced.lane(0);
    }
    total
}

// ─── Per-clause gradient contribution ────────────────────────────────

/// Gradient contribution of clause `ci` for the variable at `target_pos`.
///
/// For clause c containing variable v at position `target_pos`:
/// ```text
///   d(loss_c)/d(x_v) = weight * sign(lit) * PROD_{j≠pos} term(lit_j)
/// ```
/// where sign = -1 for positive literal (`d/dx` of `1-x`),
///       sign = +1 for negative literal (`d/dx` of `x`).
#[inline]
fn clause_grad_contribution(data: &ClauseDataSoA, x: &[f64], ci: usize, target_pos: usize) -> f64 {
    let sign = if data.negs[target_pos][ci] { 1.0 } else { -1.0 };
    let mut prod_others = 1.0f64;
    for pos in 0..3 {
        if pos == target_pos {
            continue;
        }
        let v = x[data.vars[pos][ci] as usize];
        let term = if data.negs[pos][ci] { v } else { 1.0 - v };
        prod_others *= term;
    }
    data.weights[ci] * sign * prod_others
}

// ─── Variable-to-clause reverse index ────────────────────────────────

/// Maps each variable to `(clause_index, literal_position)` pairs in the SoA.
///
/// Same role as `VarIndex` in `gradient.rs`, but over `ClauseDataSoA`.
/// Needed for gradient accumulation: for each variable, sum contributions
/// from all clauses containing it.
pub struct VarIndexSoA(pub Vec<Vec<(usize, usize)>>);

impl VarIndexSoA {
    /// Build the reverse index from SoA clause data.
    pub fn build(data: &ClauseDataSoA, num_vars: u32) -> Self {
        let mut occ = vec![Vec::new(); num_vars as usize];
        for ci in 0..data.num_clauses {
            for pos in 0..3 {
                let var = data.vars[pos][ci] as usize;
                if var < num_vars as usize {
                    occ[var].push((ci, pos));
                }
            }
        }
        VarIndexSoA(occ)
    }
}

/// Compute gradient of loss w.r.t. each variable using SoA data.
///
/// Sequential over variables, but validates correctness against
/// `gradient.rs::gradient()` and finite differences. The variable-parallel
/// GPU version (Axis 1b: each lane = one variable) is future work.
pub fn gradient_soa(
    data: &ClauseDataSoA,
    x: &[f64],
    var_index: &VarIndexSoA,
    grad: &mut [f64],
) {
    grad.iter_mut().for_each(|g| *g = 0.0);
    for (v, occs) in var_index.0.iter().enumerate() {
        for &(ci, pos) in occs {
            grad[v] += clause_grad_contribution(data, x, ci, pos);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::generate_3sat_phase_transition;
    use crate::gradient;

    /// Deterministic variable initialization: golden ratio spacing in [0.05, 0.95].
    fn init_vars(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let raw = (i as f64 * std::f64::consts::FRAC_1_SQRT_2) % 1.0;
                0.05 + raw * 0.9
            })
            .collect()
    }

    #[test]
    fn soa_packing_preserves_clauses() {
        let db = generate_3sat_phase_transition(20, 42);
        let weights = vec![1.0; db.len()];
        let (soa, skipped) = ClauseDataSoA::from_clause_db(&db, &weights);

        assert_eq!(skipped, 0);
        assert_eq!(soa.num_clauses, db.len());
        assert_eq!(soa.padded_len % WARP_SIZE, 0);
        assert!(soa.padded_len >= soa.num_clauses);

        for ci in 0..db.len() {
            let lits = &db.clause(ci).literals;
            for pos in 0..3 {
                assert_eq!(soa.vars[pos][ci], lits[pos].var(), "clause {ci} pos {pos}");
                assert_eq!(soa.negs[pos][ci], lits[pos].is_negated(), "clause {ci} pos {pos}");
            }
        }
    }

    #[test]
    fn simwarp_loss_matches_cpu() {
        // The core correctness test: SimWarp clause-parallel loss must
        // match the sequential CPU loss() for the same clause database.
        for seed in 0..10 {
            let db = generate_3sat_phase_transition(20, seed);
            let weights = vec![1.0; db.len()];
            let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

            let x = init_vars(20);

            let cpu = gradient::loss(&db, &x, &weights);
            let gpu = total_loss_simwarp(&soa, &x);

            assert!(
                (cpu - gpu).abs() < 1e-10,
                "seed {seed}: CPU loss {cpu:.12} != SimWarp loss {gpu:.12}"
            );
        }
    }

    #[test]
    fn simwarp_loss_with_weights() {
        // Non-uniform weights: clause weight adaptation changes the loss.
        let db = generate_3sat_phase_transition(20, 7);
        let weights: Vec<f64> = (0..db.len()).map(|i| 0.5 + (i as f64) * 0.01).collect();
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        let x = init_vars(20);

        let cpu = gradient::loss(&db, &x, &weights);
        let gpu = total_loss_simwarp(&soa, &x);

        assert!(
            (cpu - gpu).abs() < 1e-10,
            "Weighted: CPU {cpu:.12} != SimWarp {gpu:.12}"
        );
    }

    #[test]
    fn gradient_soa_matches_cpu() {
        // SoA gradient must match gradient.rs::gradient() for the same data.
        for seed in 0..5 {
            let db = generate_3sat_phase_transition(20, seed);
            let weights = vec![1.0; db.len()];
            let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

            let x = init_vars(20);

            // CPU reference gradient
            let cpu_idx = gradient::VarIndex::build(&db, 20);
            let mut cpu_grad = vec![0.0; 20];
            gradient::gradient(&db, &x, &cpu_idx, &weights, &mut cpu_grad);

            // SoA gradient
            let soa_idx = VarIndexSoA::build(&soa, 20);
            let mut soa_grad = vec![0.0; 20];
            gradient_soa(&soa, &x, &soa_idx, &mut soa_grad);

            for v in 0..20 {
                assert!(
                    (cpu_grad[v] - soa_grad[v]).abs() < 1e-10,
                    "seed {seed} var {v}: CPU grad {:.12} != SoA grad {:.12}",
                    cpu_grad[v],
                    soa_grad[v]
                );
            }
        }
    }

    #[test]
    fn gradient_soa_matches_finite_differences() {
        // Implementation-independent check: analytical gradient ≈ numerical gradient.
        let db = generate_3sat_phase_transition(20, 99);
        let weights = vec![1.0; db.len()];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        let x = init_vars(20);
        let soa_idx = VarIndexSoA::build(&soa, 20);
        let mut grad = vec![0.0; 20];
        gradient_soa(&soa, &x, &soa_idx, &mut grad);

        let eps = 1e-7;
        for v in 0..20 {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[v] += eps;
            x_minus[v] -= eps;
            let numerical = (total_loss_simwarp(&soa, &x_plus) - total_loss_simwarp(&soa, &x_minus))
                / (2.0 * eps);
            assert!(
                (grad[v] - numerical).abs() < 1e-5,
                "var {v}: analytical {:.8} != numerical {:.8}",
                grad[v],
                numerical
            );
        }
    }

    #[test]
    fn padding_contributes_zero() {
        // 5 clauses → padded to 32. Padding lanes must not affect loss.
        let mut db = ClauseDb::new();
        for _ in 0..5 {
            db.add_clause(vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)]);
        }
        let weights = vec![1.0; 5];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        assert_eq!(soa.padded_len, WARP_SIZE);
        for i in 5..WARP_SIZE {
            assert_eq!(soa.weights[i], 0.0, "padding clause {i} should have zero weight");
        }

        let x = vec![0.3, 0.7, 0.5];
        let loss = total_loss_simwarp(&soa, &x);

        // Each clause: (x0 ∨ x1 ∨ x2), loss = (1-0.3)(1-0.7)(1-0.5) = 0.7×0.3×0.5 = 0.105
        let expected = 5.0 * 0.7 * 0.3 * 0.5;
        assert!(
            (loss - expected).abs() < 1e-10,
            "loss {loss} != expected {expected}"
        );
    }

    #[test]
    fn empty_clause_db() {
        let db = ClauseDb::new();
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &[]);
        assert_eq!(soa.num_clauses, 0);
        assert_eq!(soa.padded_len, 0);
        assert_eq!(soa.num_batches(), 0);
        assert_eq!(total_loss_simwarp(&soa, &[0.5; 10]), 0.0);
    }

    #[test]
    fn all_satisfied_near_zero_loss() {
        // Clause (x0 ∨ x1 ∨ x2) with x0=1.0 → first term is 0 → clause loss = 0.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)]);
        let weights = vec![1.0];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        let x = vec![1.0, 0.5, 0.5];
        let loss = total_loss_simwarp(&soa, &x);
        assert!(loss.abs() < 1e-15, "loss should be ~0 when x0=1.0, got {loss}");
    }

    #[test]
    fn skips_non_3sat_clauses() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)]); // 3-SAT ✓
        db.add_clause(vec![Lit::pos(0), Lit::pos(1)]); // 2-SAT, skipped
        db.add_clause(vec![Lit::pos(0)]); // unit, skipped
        db.add_clause(vec![Lit::neg(0), Lit::neg(1), Lit::neg(2)]); // 3-SAT ✓

        let weights = vec![1.0; 4];
        let (soa, skipped) = ClauseDataSoA::from_clause_db(&db, &weights);
        assert_eq!(skipped, 2);
        assert_eq!(soa.num_clauses, 2);
    }

    #[test]
    fn large_instance_simwarp_matches_cpu() {
        // 100 vars at phase transition: ~427 clauses → 14 warp batches.
        let db = generate_3sat_phase_transition(100, 42);
        let weights = vec![1.0; db.len()];
        let (soa, skipped) = ClauseDataSoA::from_clause_db(&db, &weights);
        assert_eq!(skipped, 0);

        let x = init_vars(100);

        let cpu = gradient::loss(&db, &x, &weights);
        let gpu = total_loss_simwarp(&soa, &x);

        assert!(
            (cpu - gpu).abs() < 1e-8,
            "100-var: CPU {cpu:.12} != SimWarp {gpu:.12}"
        );
    }

    #[test]
    fn negated_literals_correct() {
        // (¬x0 ∨ ¬x1 ∨ ¬x2) with all vars at 0.8.
        // Loss = 0.8 × 0.8 × 0.8 = 0.512 (negated terms use x directly).
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::neg(0), Lit::neg(1), Lit::neg(2)]);
        let weights = vec![1.0];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        let x = vec![0.8, 0.8, 0.8];
        let loss = total_loss_simwarp(&soa, &x);
        let expected = 0.8 * 0.8 * 0.8;
        assert!(
            (loss - expected).abs() < 1e-15,
            "negated loss {loss} != expected {expected}"
        );
    }
}
