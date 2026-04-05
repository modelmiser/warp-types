//! Gradient-based SAT solver: continuous relaxation with projected gradient descent.
//!
//! Relaxes boolean variables from {0,1} to [0,1] and minimizes a product-form
//! loss whose global minimum at zero corresponds to a satisfying assignment.
//!
//! For clause (l_1 OR ... OR l_k), the unsatisfaction loss is:
//!   loss_c = PROD_i term(l_i)
//! where term(x_j) = 1-x_j for positive literal, term(!x_j) = x_j for negative.
//!
//! Gradient is analytical (product rule, no autodiff). Multi-start search maps
//! to per-lane parallelism on GPU — sequential on CPU for this validation build.
//!
//! **Incomplete solver**: can find SAT assignments but cannot prove UNSAT.

use crate::bcp::ClauseDb;
use crate::literal::Lit;

// ─── RNG (deterministic, no dependency) ───────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Random f64 in [0.05, 0.95] — avoids boundary initialization.
    fn unit(&mut self) -> f64 {
        0.05 + (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 0.9
    }
}

// ─── Configuration ────────────────────────────────────────────────────

/// Gradient solver configuration.
pub struct GradientConfig {
    /// Independent random starting points (maps to warp lanes on GPU).
    pub num_starts: usize,
    /// Max gradient descent iterations per start.
    pub max_iters: usize,
    /// Initial learning rate.
    pub learning_rate: f64,
    /// LR multiplicative decay per step.
    pub lr_decay: f64,
    /// Base RNG seed (start i uses seed + i).
    pub seed: u64,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            num_starts: 32,
            max_iters: 1000,
            learning_rate: 0.1,
            lr_decay: 0.999,
            seed: 42,
        }
    }
}

// ─── Result types ─────────────────────────────────────────────────────

/// Per-start outcome.
#[derive(Debug, Clone)]
pub struct StartOutcome {
    pub best_loss: f64,
    pub iterations: usize,
    pub satisfied: bool,
}

/// Gradient search result.
#[derive(Debug)]
pub struct GradientResult {
    /// Satisfying assignment, if found.
    pub assignment: Option<Vec<bool>>,
    /// Per-start diagnostics.
    pub starts: Vec<StartOutcome>,
    /// Total clause evaluations across all starts.
    pub clause_evals: usize,
}

// ─── Variable-to-clause index ─────────────────────────────────────────

/// Maps each variable to the clauses it appears in: (clause_idx, position_in_clause).
struct VarIndex(Vec<Vec<(usize, usize)>>);

impl VarIndex {
    fn build(db: &ClauseDb, num_vars: u32) -> Self {
        let mut occ = vec![Vec::new(); num_vars as usize];
        for ci in 0..db.len() {
            for (pos, lit) in db.clause(ci).literals.iter().enumerate() {
                occ[lit.var() as usize].push((ci, pos));
            }
        }
        VarIndex(occ)
    }
}

// ─── Core math ────────────────────────────────────────────────────────

/// Falseness of a literal under continuous assignment.
/// Positive x_j: term = 1 - x_j (zero when x_j = 1, i.e. literal true).
/// Negative !x_j: term = x_j (zero when x_j = 0, i.e. literal true).
#[inline]
fn lit_term(lit: Lit, x: &[f64]) -> f64 {
    let v = x[lit.var() as usize];
    if lit.is_negated() {
        v
    } else {
        1.0 - v
    }
}

/// Total loss: sum over all clauses of product of literal falseness terms.
/// Zero iff every clause has at least one fully-true literal.
fn loss(db: &ClauseDb, x: &[f64]) -> f64 {
    (0..db.len())
        .map(|ci| {
            db.clause(ci)
                .literals
                .iter()
                .map(|&l| lit_term(l, x))
                .product::<f64>()
        })
        .sum()
}

/// Gradient of total loss w.r.t. each variable.
///
/// For clause c containing variable v at position j:
///   d(loss_c)/d(x_v) = sign(l_j) * PROD_{i!=j} term(l_i)
/// where sign = -1 for positive literal, +1 for negative.
fn gradient(db: &ClauseDb, x: &[f64], idx: &VarIndex, grad: &mut [f64]) {
    grad.iter_mut().for_each(|g| *g = 0.0);
    for (v, occs) in idx.0.iter().enumerate() {
        for &(ci, pos) in occs {
            let lits = &db.clause(ci).literals;
            let prod_others: f64 = lits
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != pos)
                .map(|(_, &l)| lit_term(l, x))
                .product();
            let sign = if lits[pos].is_negated() { 1.0 } else { -1.0 };
            grad[v] += sign * prod_others;
        }
    }
}

/// Check whether a discrete assignment satisfies all clauses.
pub fn verify(db: &ClauseDb, assign: &[bool]) -> bool {
    (0..db.len()).all(|ci| {
        db.clause(ci).literals.iter().any(|&lit| {
            let val = assign[lit.var() as usize];
            if lit.is_negated() {
                !val
            } else {
                val
            }
        })
    })
}

/// Round continuous variables to {false, true} at threshold 0.5.
fn discretize(x: &[f64]) -> Vec<bool> {
    x.iter().map(|&v| v >= 0.5).collect()
}

// ─── Solver ───────────────────────────────────────────────────────────

/// Search for a satisfying assignment via projected gradient descent.
///
/// Multi-start: tries `config.num_starts` random initializations, each
/// running up to `config.max_iters` gradient steps. Returns the first
/// satisfying assignment found, or `None` if all starts are exhausted.
///
/// **Incomplete**: cannot prove UNSAT. A `None` result means "not found",
/// not "unsatisfiable".
pub fn gradient_search(db: &ClauseDb, num_vars: u32, config: &GradientConfig) -> GradientResult {
    let n = num_vars as usize;

    if db.is_empty() {
        return GradientResult {
            assignment: Some(vec![false; n]),
            starts: vec![],
            clause_evals: 0,
        };
    }
    if num_vars == 0 {
        return GradientResult {
            assignment: None,
            starts: vec![],
            clause_evals: 0,
        };
    }

    let idx = VarIndex::build(db, num_vars);
    let mut result = GradientResult {
        assignment: None,
        starts: Vec::with_capacity(config.num_starts),
        clause_evals: 0,
    };

    for si in 0..config.num_starts {
        if result.assignment.is_some() {
            break;
        }

        let mut rng = Rng::new(config.seed.wrapping_add(si as u64));
        let mut x: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
        let mut grad = vec![0.0; n];
        let mut lr = config.learning_rate;
        let mut best_loss = f64::MAX;
        let mut found = false;

        for iter in 0..config.max_iters {
            let l = loss(db, &x);
            result.clause_evals += db.len();

            if l < best_loss {
                best_loss = l;
            }

            // Discretize and verify periodically, or when loss is low.
            if l < 1.0 || iter % 10 == 0 {
                let assign = discretize(&x);
                if verify(db, &assign) {
                    result.assignment = Some(assign);
                    result.starts.push(StartOutcome {
                        best_loss: l,
                        iterations: iter + 1,
                        satisfied: true,
                    });
                    found = true;
                    break;
                }
            }

            // Gradient step
            gradient(db, &x, &idx, &mut grad);

            // Early exit on vanishing gradient (stuck at stationary point).
            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-20 {
                break;
            }

            // Projected gradient descent: step then clamp to [0, 1].
            for i in 0..n {
                x[i] = (x[i] - lr * grad[i]).clamp(0.0, 1.0);
            }
            lr *= config.lr_decay;
        }

        if !found {
            // Final discretization attempt at convergence point.
            let assign = discretize(&x);
            if verify(db, &assign) {
                result.assignment = Some(assign);
                result.starts.push(StartOutcome {
                    best_loss,
                    iterations: config.max_iters,
                    satisfied: true,
                });
            } else {
                result.starts.push(StartOutcome {
                    best_loss,
                    iterations: config.max_iters,
                    satisfied: false,
                });
            }
        }
    }

    result
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{generate_3sat_phase_transition, generate_k_sat};
    use crate::solver;
    use std::time::Instant;

    #[test]
    fn trivial_sat() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);

        let r = gradient_search(&db, 1, &GradientConfig::default());
        let assign = r.assignment.expect("should find SAT");
        assert!(assign[0], "x0 must be true");
    }

    #[test]
    fn two_clause_forced_var() {
        // (x0 v x1) ^ (!x0 v x1) => x1 must be true
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1)]);
        db.add_clause(vec![Lit::neg(0), Lit::pos(1)]);

        let r = gradient_search(&db, 2, &GradientConfig::default());
        let assign = r.assignment.expect("should find SAT");
        assert!(assign[1], "x1 must be true");
    }

    #[test]
    fn unsat_returns_none() {
        // x0 ^ !x0 — trivially UNSAT. Gradient is identically zero.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);
        db.add_clause(vec![Lit::neg(0)]);

        let r = gradient_search(&db, 1, &GradientConfig::default());
        assert!(r.assignment.is_none());
        assert!(r.starts.iter().all(|s| !s.satisfied));
        // Loss is constant at 1.0 for this instance
        assert!(r.starts.iter().all(|s| (s.best_loss - 1.0).abs() < 0.01));
    }

    #[test]
    fn empty_formula() {
        let db = ClauseDb::new();
        let r = gradient_search(&db, 5, &GradientConfig::default());
        assert!(r.assignment.is_some());
    }

    #[test]
    fn loss_decreases() {
        let db = generate_3sat_phase_transition(20, 42);
        let n = 20usize;
        let idx = VarIndex::build(&db, 20);

        let mut rng = Rng::new(42);
        let mut x: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
        let mut grad = vec![0.0; n];

        let initial = loss(&db, &x);
        for _ in 0..200 {
            gradient(&db, &x, &idx, &mut grad);
            for i in 0..n {
                x[i] = (x[i] - 0.01 * grad[i]).clamp(0.0, 1.0);
            }
        }
        let final_loss = loss(&db, &x);
        assert!(
            final_loss < initial,
            "loss should decrease: {initial} -> {final_loss}"
        );
    }

    #[test]
    fn random_3sat_finds_solutions() {
        let mut found = 0;
        let total = 20;
        for seed in 0..total {
            let db = generate_3sat_phase_transition(20, seed);
            let r = gradient_search(&db, 20, &GradientConfig::default());
            if let Some(ref assign) = r.assignment {
                assert!(verify(&db, assign), "seed {seed}: invalid assignment");
                found += 1;
            }
        }
        // At phase transition ~50% are SAT. Gradient should find most.
        assert!(
            found > 0,
            "gradient should find at least some SAT instances (found {found}/{total})"
        );
    }

    #[test]
    fn gradient_never_claims_false_sat() {
        // Soundness: if gradient says SAT, the assignment must actually satisfy.
        // Compare with CDCL for ground truth on UNSAT.
        for seed in 0..20 {
            let db1 = generate_3sat_phase_transition(30, seed);
            let db2 = generate_3sat_phase_transition(30, seed);

            let grad = gradient_search(&db1, 30, &GradientConfig::default());
            let cdcl = solver::solve(db2, 30);

            if let Some(ref assign) = grad.assignment {
                // Verify the gradient's assignment independently.
                assert!(verify(&db1, assign), "seed {seed}: gradient returned invalid assignment");
                // CDCL must also say SAT (if it doesn't, gradient has a bug).
                assert!(
                    matches!(cdcl, solver::SolveResult::Sat(_)),
                    "seed {seed}: gradient found SAT but CDCL says UNSAT"
                );
            }
        }
    }

    #[test]
    fn bench_gradient_vs_cdcl() {
        // Comparison benchmark. Run with --release --nocapture.
        println!("\n=== Gradient vs CDCL ===");
        println!(
            "{:<6} {:>10} {:>10} {:>8} {:>8}",
            "vars", "grad(us)", "cdcl(us)", "grad_ok", "cdcl_ok"
        );
        println!("{}", "-".repeat(48));

        for &n in &[20, 50, 100] {
            let seeds = 5u64;
            let mut gt = 0u128;
            let mut ct = 0u128;
            let mut gs = 0u32;
            let mut cs = 0u32;

            for seed in 0..seeds {
                let db1 = generate_3sat_phase_transition(n, seed);
                let db2 = generate_3sat_phase_transition(n, seed);

                let t = Instant::now();
                let g = gradient_search(&db1, n, &GradientConfig::default());
                gt += t.elapsed().as_micros();
                if g.assignment.is_some() {
                    gs += 1;
                }

                let t = Instant::now();
                let c = solver::solve(db2, n);
                ct += t.elapsed().as_micros();
                if matches!(c, solver::SolveResult::Sat(_)) {
                    cs += 1;
                }
            }

            println!(
                "{:<6} {:>10} {:>10} {:>8} {:>8}",
                n,
                gt / seeds as u128,
                ct / seeds as u128,
                gs,
                cs
            );
        }
    }

    #[test]
    fn larger_clause_widths() {
        // Test gradient on 5-SAT and 7-SAT (wider clauses).
        for &k in &[5, 7] {
            let db = generate_k_sat(30, 100, k, 42);
            let r = gradient_search(&db, 30, &GradientConfig::default());
            if let Some(ref assign) = r.assignment {
                assert!(verify(&db, assign), "k={k}: invalid assignment");
            }
            // Just verify it doesn't crash — wider clauses have more terms per product.
        }
    }
}
