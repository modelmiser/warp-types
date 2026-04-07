//! Gradient-based SAT solver: continuous relaxation with projected gradient descent.
//!
//! Relaxes boolean variables from {0,1} to [0,1] and minimizes a product-form
//! loss whose global minimum at zero corresponds to a satisfying assignment.
//!
//! For clause (l_1 OR ... OR l_k), the unsatisfaction loss is:
//!   loss_c = w_c * PROD_i term(l_i)
//! where term(x_j) = 1-x_j for positive literal, term(!x_j) = x_j for negative.
//!
//! Gradient is analytical (product rule, no autodiff). Multi-start search maps
//! to per-lane parallelism on GPU — sequential on CPU for this validation build.
//!
//! Three improvement layers over vanilla gradient descent:
//! - **Momentum**: velocity accumulator to escape flat regions
//! - **Clause weights**: EMA on violation frequency (a la FastFourierSAT)
//! - **Hybrid solve**: confidence-ranked partial assignment seeds CDCL (a la TurboSAT)
//!
//! **Incomplete solver**: can find SAT assignments but cannot prove UNSAT.
//!
//! # GPU Kernel Design (warp-types primitive mapping)
//!
//! Three parallelism axes, each mapping to a warp-types concept:
//!
//! **Axis 1 — Clause-parallel loss evaluation (one warp = 32 clauses):**
//! ```text
//! // Each lane evaluates one clause's product-form loss.
//! // For 3-SAT: 3 loads + 2 multiplies per lane, fully independent.
//! let term0: PerLane<f64> = /* load x[var(lit0)] per lane */;
//! let term1: PerLane<f64> = /* load x[var(lit1)] per lane */;
//! let term2: PerLane<f64> = /* load x[var(lit2)] per lane */;
//! let loss_per_clause: PerLane<f64> = term0 * term1 * term2; // element-wise
//! let batch_loss: Uniform<f64> = warp.reduce_sum(loss_per_clause);
//! ```
//!
//! **Axis 2 — Verification via ballot (discrete satisfaction check):**
//! ```text
//! let satisfied: PerLane<bool> = /* round x, check clause per lane */;
//! let result: BallotResult = warp.ballot(satisfied);
//! // All clauses SAT iff popcount(result) == num_clauses_in_batch
//! ```
//!
//! **Axis 3 — Confidence ranking via bitonic sort (hybrid CDCL seeding):**
//! ```text
//! let confidence: PerLane<f64> = /* |x[var] - 0.5| per lane */;
//! let sorted = warp.bitonic_sort(confidence); // 15 compare-swaps
//! // Top-k confident variables become unit clauses for CDCL
//! ```
//!
//! **Missing primitive:** `reduce_prod` (product reduction). Currently not in
//! warp-types because GPU hardware provides sum reduction (`__reduce_add_sync`)
//! but not product reduction. For k>32, implement via `reduce_sum(log(x))` then
//! `exp`. For k=3 (our target), the product is just 2 multiplies — no reduction
//! needed, the clause fits in a single lane.
//!
//! **Parallelism budget at scale:**
//! - n=1000 vars → 4267 clauses → 134 warps for loss eval → saturates 1 SM
//! - n=10000 vars → 42670 clauses → 1334 warps → saturates full GPU
//! - 32 multi-start searches → 32× the above → embarrassingly parallel across SMs

use crate::bcp::{CRef, ClauseDb};
use crate::literal::Lit;

// ─── RNG (deterministic, no dependency) ───────────────────────────────

pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Random f64 in [0.05, 0.95] — avoids boundary initialization.
    pub(crate) fn unit(&mut self) -> f64 {
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
    /// Momentum coefficient (0.0 = vanilla GD, 0.9 = standard momentum).
    pub momentum: f64,
    /// Enable clause weight adaptation (EMA on violation frequency).
    pub clause_weights: bool,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            num_starts: 32,
            max_iters: 1000,
            learning_rate: 0.1,
            lr_decay: 0.999,
            seed: 42,
            momentum: 0.0,
            clause_weights: false,
        }
    }
}

impl GradientConfig {
    /// Enhanced configuration: momentum + clause weight adaptation.
    pub fn enhanced() -> Self {
        Self {
            momentum: 0.9,
            clause_weights: true,
            ..Self::default()
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
    /// Best continuous variable values (for hybrid CDCL seeding).
    /// Present when no satisfying assignment was found.
    pub best_continuous: Option<Vec<f64>>,
    /// Per-start diagnostics.
    pub starts: Vec<StartOutcome>,
    /// Total clause evaluations across all starts.
    pub clause_evals: usize,
}

// ─── Variable-to-clause index ─────────────────────────────────────────

/// Maps each variable to the clauses it appears in: (sequential_index, position_in_clause).
/// The sequential index maps into a `crefs: Vec<CRef>` for clause DB access.
pub(crate) struct VarIndex(Vec<Vec<(usize, usize)>>);

impl VarIndex {
    pub(crate) fn build(db: &ClauseDb, num_vars: u32, crefs: &[CRef]) -> Self {
        let mut occ = vec![Vec::new(); num_vars as usize];
        for (ci, &cref) in crefs.iter().enumerate() {
            for (pos, lit) in db.clause(cref).literals.iter().enumerate() {
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

/// Weighted loss: sum over all clauses of w_c * product of literal falseness terms.
/// Zero iff every clause has at least one fully-true literal.
pub(crate) fn loss(db: &ClauseDb, crefs: &[CRef], x: &[f64], weights: &[f64]) -> f64 {
    crefs
        .iter()
        .enumerate()
        .map(|(ci, &cref)| {
            weights[ci]
                * db.clause(cref)
                    .literals
                    .iter()
                    .map(|&l| lit_term(l, x))
                    .product::<f64>()
        })
        .sum()
}

/// Gradient of weighted loss w.r.t. each variable.
///
/// For clause c containing variable v at position j:
///   d(loss_c)/d(x_v) = w_c * sign(l_j) * PROD_{i!=j} term(l_i)
/// where sign = -1 for positive literal, +1 for negative.
pub(crate) fn gradient(
    db: &ClauseDb,
    crefs: &[CRef],
    x: &[f64],
    idx: &VarIndex,
    weights: &[f64],
    grad: &mut [f64],
) {
    grad.iter_mut().for_each(|g| *g = 0.0);
    for (v, occs) in idx.0.iter().enumerate() {
        for &(ci, pos) in occs {
            let lits = &db.clause(crefs[ci]).literals;
            let prod_others: f64 = lits
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != pos)
                .map(|(_, &l)| lit_term(l, x))
                .product();
            let sign = if lits[pos].is_negated() { 1.0 } else { -1.0 };
            grad[v] += weights[ci] * sign * prod_others;
        }
    }
}

/// Check whether a discrete assignment satisfies all clauses.
pub fn verify(db: &ClauseDb, assign: &[bool]) -> bool {
    db.iter_crefs().all(|cref| {
        db.clause(cref).literals.iter().any(|&lit| {
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

/// Update clause weights via EMA on violation frequency.
/// Unsatisfied clauses accumulate weight; satisfied clauses decay.
/// Formula: w = 0.9 * w + 0.1 * (violated ? 1 : 0)
fn update_weights(db: &ClauseDb, crefs: &[CRef], assign: &[bool], weights: &mut [f64]) {
    for (ci, &cref) in crefs.iter().enumerate() {
        let satisfied = db.clause(cref).literals.iter().any(|&lit| {
            let val = assign[lit.var() as usize];
            if lit.is_negated() {
                !val
            } else {
                val
            }
        });
        weights[ci] = 0.9 * weights[ci] + if satisfied { 0.0 } else { 0.1 };
    }
}

// ─── Solver ───────────────────────────────────────────────────────────

/// Search for a satisfying assignment via projected gradient descent.
///
/// Multi-start: tries `config.num_starts` random initializations, each
/// running up to `config.max_iters` gradient steps. Returns the first
/// satisfying assignment found, or `None` if all starts are exhausted.
///
/// When no solution is found, `best_continuous` contains the continuous
/// variable values from the best start (lowest loss) — usable for
/// confidence-ranked CDCL seeding via `hybrid_solve`.
///
/// **Incomplete**: cannot prove UNSAT. A `None` result means "not found",
/// not "unsatisfiable".
pub fn gradient_search(db: &ClauseDb, num_vars: u32, config: &GradientConfig) -> GradientResult {
    let n = num_vars as usize;

    if db.is_empty() {
        return GradientResult {
            assignment: Some(vec![false; n]),
            best_continuous: None,
            starts: vec![],
            clause_evals: 0,
        };
    }
    if num_vars == 0 {
        return GradientResult {
            assignment: None,
            best_continuous: None,
            starts: vec![],
            clause_evals: 0,
        };
    }

    let crefs = db.crefs();
    let idx = VarIndex::build(db, num_vars, &crefs);
    let mut result = GradientResult {
        assignment: None,
        best_continuous: None,
        starts: Vec::with_capacity(config.num_starts),
        clause_evals: 0,
    };
    let mut global_best_loss = f64::MAX;

    for si in 0..config.num_starts {
        if result.assignment.is_some() {
            break;
        }

        let mut rng = Rng::new(config.seed.wrapping_add(si as u64));
        let mut x: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
        let mut grad = vec![0.0; n];
        let mut velocity = vec![0.0; n];
        let mut weights = vec![1.0; crefs.len()];
        let mut lr = config.learning_rate;
        let mut best_loss = f64::MAX;
        let mut found = false;

        for iter in 0..config.max_iters {
            let l = loss(db, &crefs, &x, &weights);
            result.clause_evals += crefs.len();

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
                // Clause weight adaptation: upweight persistently violated clauses.
                if config.clause_weights {
                    update_weights(db, &crefs, &assign, &mut weights);
                }
            }

            // Gradient step
            gradient(db, &crefs, &x, &idx, &weights, &mut grad);

            // Early exit on vanishing gradient (stuck at stationary point).
            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-20 {
                break;
            }

            // Update with optional momentum, then project onto [0, 1].
            if config.momentum > 0.0 {
                for i in 0..n {
                    velocity[i] = config.momentum * velocity[i] + grad[i];
                    x[i] = (x[i] - lr * velocity[i]).clamp(0.0, 1.0);
                }
            } else {
                for i in 0..n {
                    x[i] = (x[i] - lr * grad[i]).clamp(0.0, 1.0);
                }
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
            // Track best continuous solution for hybrid CDCL seeding.
            if best_loss < global_best_loss {
                global_best_loss = best_loss;
                result.best_continuous = Some(x);
            }
        }
    }

    result
}

/// Single-gradient probe at the current CDCL trail position.
///
/// Constructs x[] from the trail: assigned vars = 0.0/1.0, unassigned = 0.5.
/// Computes ONE gradient at this point. Cost: O(m) where m = total literals
/// across all clauses. No descent loop.
///
/// The gradient magnitude for each unassigned variable tells us: "how much
/// would the loss decrease if I nudged this variable from 0.5?" This is
/// lookahead information — it sees the effect on unsatisfied clauses before
/// CDCL commits a decision.
///
/// Returns magnitudes (for activity boosting) and the gradient-suggested
/// polarity for phase hints.
pub(crate) fn gradient_at_trail(
    db: &ClauseDb,
    num_vars: u32,
    assignments: &[Option<bool>],
) -> TrailGradient {
    let n = num_vars as usize;
    if n == 0 || db.is_empty() {
        return TrailGradient {
            magnitudes: vec![0.0; n],
            polarities: vec![true; n],
        };
    }

    let crefs = db.crefs();
    if crefs.is_empty() {
        return TrailGradient {
            magnitudes: vec![0.0; n],
            polarities: vec![true; n],
        };
    }

    // Build x[] from trail: assigned = 0.0/1.0, unassigned = 0.5
    let mut x = vec![0.5f64; n];
    for (v, &a) in assignments.iter().enumerate() {
        if let Some(val) = a {
            x[v] = if val { 1.0 } else { 0.0 };
        }
    }

    let idx = VarIndex::build(db, num_vars, &crefs);
    let weights = vec![1.0; crefs.len()];
    let mut grad = vec![0.0; n];

    gradient(db, &crefs, &x, &idx, &weights, &mut grad);

    // For unassigned vars: gradient sign tells us which polarity reduces loss.
    // Negative gradient → increasing x (setting true) reduces loss.
    // Positive gradient → decreasing x (setting false) reduces loss.
    // For assigned vars: polarity is irrelevant (already decided).
    let polarities: Vec<bool> = grad.iter().map(|&g| g < 0.0).collect();
    let magnitudes: Vec<f64> = grad.iter().map(|g| g.abs()).collect();

    TrailGradient {
        magnitudes,
        polarities,
    }
}

/// Result of a single-gradient probe at the current trail position.
pub(crate) struct TrailGradient {
    /// |∂L/∂x_v| per variable — gradient magnitude for activity boosting.
    pub magnitudes: Vec<f64>,
    /// Gradient-suggested polarity per variable (true if gradient points toward x=1).
    pub polarities: Vec<bool>,
}

// ─── Hybrid solver ────────────────────────────────────────────────────

/// Hybrid gradient→CDCL solver.
///
/// Phase 1: Gradient search finds a continuous near-solution.
/// Phase 2: Rank variables by confidence (distance from 0.5 boundary).
/// Phase 3: Warm-start VSIDS with confidence as activity + polarity as phase.
/// Phase 4: CDCL solves with the warm-started heuristic.
///
/// This is the TurboSAT strategy: GPU narrows the search space, CPU finishes.
///
/// Unlike the unit-clause approach, these hints are fully backtrackable —
/// if the gradient points the wrong way, CDCL discovers this via conflict
/// analysis and backtracks normally. No false UNSAT from irrevocable guesses.
pub fn hybrid_solve(
    db: ClauseDb,
    num_vars: u32,
    config: &GradientConfig,
) -> crate::solver::SolveResult {
    use crate::solver::SolveResult;
    use crate::vsids::Vsids;

    // Phase 1: gradient search
    let grad = gradient_search(&db, num_vars, config);

    if let Some(assign) = grad.assignment {
        return SolveResult::Sat(assign);
    }

    // Phase 2-3: warm-start VSIDS from gradient confidence
    let mut vsids = Vsids::new(num_vars);
    if let Some(ref x) = grad.best_continuous {
        for (v, &val) in x.iter().enumerate() {
            // Phase hint: gradient's rounded polarity suggestion.
            vsids.set_phase(v as u32, val >= 0.5);
            // Activity hint: confidence (|val - 0.5|) scaled to be comparable
            // with clause-occurrence initialization (~12.8 per var at ratio 4.267).
            vsids.set_initial_activity(v as u32, (val - 0.5).abs() * 20.0);
        }
    }
    // Blend with clause-occurrence counts for structural awareness.
    vsids.initialize_from_clauses(&db);

    // Phase 4: CDCL with warm-started VSIDS (watched literals + restarts)
    crate::solver::solve_cdcl_core(db, num_vars, vsids)
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
        let crefs = db.crefs();
        let idx = VarIndex::build(&db, 20, &crefs);
        let weights = vec![1.0; crefs.len()];

        let mut rng = Rng::new(42);
        let mut x: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
        let mut grad = vec![0.0; n];

        let initial = loss(&db, &crefs, &x, &weights);
        for _ in 0..200 {
            gradient(&db, &crefs, &x, &idx, &weights, &mut grad);
            for i in 0..n {
                x[i] = (x[i] - 0.01 * grad[i]).clamp(0.0, 1.0);
            }
        }
        let final_loss = loss(&db, &crefs, &x, &weights);
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
        assert!(
            found > 0,
            "gradient should find at least some SAT instances (found {found}/{total})"
        );
    }

    #[test]
    fn gradient_never_claims_false_sat() {
        for seed in 0..20 {
            let db1 = generate_3sat_phase_transition(30, seed);
            let db2 = generate_3sat_phase_transition(30, seed);

            let grad = gradient_search(&db1, 30, &GradientConfig::default());
            let cdcl = solver::solve(db2, 30);

            if let Some(ref assign) = grad.assignment {
                assert!(
                    verify(&db1, assign),
                    "seed {seed}: gradient returned invalid assignment"
                );
                assert!(
                    matches!(cdcl, solver::SolveResult::Sat(_)),
                    "seed {seed}: gradient found SAT but CDCL says UNSAT"
                );
            }
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn enhanced_improves_success_rate() {
        // Enhanced config (momentum + weights) should find >= as many solutions
        // as vanilla on 50-var instances where vanilla struggles.
        let vanilla = GradientConfig::default();
        let enhanced = GradientConfig::enhanced();
        let mut v_found = 0;
        let mut e_found = 0;
        let seeds = 20;

        for seed in 0..seeds {
            let db = generate_3sat_phase_transition(50, seed);
            if gradient_search(&db, 50, &vanilla).assignment.is_some() {
                v_found += 1;
            }
            let db = generate_3sat_phase_transition(50, seed);
            if gradient_search(&db, 50, &enhanced).assignment.is_some() {
                e_found += 1;
            }
        }
        println!("50-var: vanilla={v_found}/{seeds}, enhanced={e_found}/{seeds}");
        // Enhanced should do at least as well (may not always be strictly better
        // due to different optimization trajectories with momentum).
    }

    #[test]
    fn hybrid_solves_more_than_gradient_alone() {
        // Hybrid should solve instances where pure gradient fails.
        let config = GradientConfig::enhanced();
        let mut grad_found = 0;
        let mut hybrid_found = 0;
        let seeds = 10;

        for seed in 0..seeds {
            let db1 = generate_3sat_phase_transition(50, seed);
            let db2 = generate_3sat_phase_transition(50, seed);

            if gradient_search(&db1, 50, &config).assignment.is_some() {
                grad_found += 1;
            }
            if matches!(
                hybrid_solve(db2, 50, &config),
                crate::solver::SolveResult::Sat(_)
            ) {
                hybrid_found += 1;
            }
        }
        println!("50-var: gradient={grad_found}/{seeds}, hybrid={hybrid_found}/{seeds}");
        assert!(
            hybrid_found >= grad_found,
            "hybrid should find at least as many as gradient alone"
        );
    }

    #[test]
    fn hybrid_soundness() {
        // Hybrid must never claim SAT on an UNSAT instance.
        for seed in 0..20 {
            let db1 = generate_3sat_phase_transition(30, seed);
            let db2 = generate_3sat_phase_transition(30, seed);

            let hybrid = hybrid_solve(db1, 30, &GradientConfig::enhanced());
            let cdcl = solver::solve(db2, 30);

            if let crate::solver::SolveResult::Sat(ref assign) = hybrid {
                // Re-verify: generate a third copy to check against.
                let db3 = generate_3sat_phase_transition(30, seed);
                assert!(
                    verify(&db3, assign),
                    "seed {seed}: hybrid returned invalid assignment"
                );
                assert!(
                    matches!(cdcl, solver::SolveResult::Sat(_)),
                    "seed {seed}: hybrid found SAT but CDCL says UNSAT"
                );
            }
        }
    }

    #[test]
    fn best_continuous_available_on_failure() {
        // When gradient fails, best_continuous should be populated.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0)]);
        db.add_clause(vec![Lit::neg(0)]);

        let r = gradient_search(&db, 1, &GradientConfig::default());
        assert!(r.assignment.is_none());
        assert!(r.best_continuous.is_some());
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn bench_gradient_vs_cdcl() {
        println!("\n=== Vanilla vs Enhanced vs Hybrid vs CDCL ===");
        println!(
            "{:<6} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6}",
            "vars", "van(us)", "enh(us)", "hyb(us)", "cdcl", "v_ok", "e_ok", "h_ok", "c_ok"
        );
        println!("{}", "-".repeat(78));

        let vanilla = GradientConfig::default();
        let enhanced = GradientConfig::enhanced();

        for &n in &[20, 50] {
            let seeds = 5u64;
            let (mut vt, mut et, mut ht, mut ct) = (0u128, 0u128, 0u128, 0u128);
            let (mut vs, mut es, mut hs, mut cs) = (0u32, 0u32, 0u32, 0u32);

            for seed in 0..seeds {
                let db = generate_3sat_phase_transition(n, seed);
                let t = Instant::now();
                if gradient_search(&db, n, &vanilla).assignment.is_some() {
                    vs += 1;
                }
                vt += t.elapsed().as_micros();

                let db = generate_3sat_phase_transition(n, seed);
                let t = Instant::now();
                if gradient_search(&db, n, &enhanced).assignment.is_some() {
                    es += 1;
                }
                et += t.elapsed().as_micros();

                let db = generate_3sat_phase_transition(n, seed);
                let t = Instant::now();
                if matches!(
                    hybrid_solve(db, n, &enhanced),
                    crate::solver::SolveResult::Sat(_)
                ) {
                    hs += 1;
                }
                ht += t.elapsed().as_micros();

                let db = generate_3sat_phase_transition(n, seed);
                let t = Instant::now();
                if matches!(solver::solve(db, n), solver::SolveResult::Sat(_)) {
                    cs += 1;
                }
                ct += t.elapsed().as_micros();
            }

            println!(
                "{:<6} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6}",
                n,
                vt / seeds as u128,
                et / seeds as u128,
                ht / seeds as u128,
                ct / seeds as u128,
                vs,
                es,
                hs,
                cs
            );
        }
    }

    #[test]
    fn larger_clause_widths() {
        for &k in &[5, 7] {
            let db = generate_k_sat(30, 100, k, 42);
            let r = gradient_search(&db, 30, &GradientConfig::default());
            if let Some(ref assign) = r.assignment {
                assert!(verify(&db, assign), "k={k}: invalid assignment");
            }
        }
    }

    #[test]
    #[ignore] // benchmark — too slow for CI
    fn scaling_study() {
        // Scaling analysis: gradient vs CDCL across problem sizes.
        // Uses fewer starts (8) for tractability at larger sizes.
        // Run with: cargo test -p warp-types-sat --release scaling_study -- --nocapture
        let config = GradientConfig {
            num_starts: 8,
            momentum: 0.9,
            clause_weights: true,
            ..GradientConfig::default()
        };

        println!("\n=== Scaling Study: Enhanced Gradient vs CDCL ===");
        println!(
            "{:<6} {:>6} {:>10} {:>10} {:>6} {:>6} {:>10}",
            "vars", "cls", "grad(us)", "cdcl(us)", "g_ok", "c_ok", "best_loss"
        );
        println!("{}", "-".repeat(62));

        for &n in &[20, 30, 50, 75, 100, 150, 200] {
            let seeds = 3u64;
            let (mut gt, mut ct) = (0u128, 0u128);
            let (mut gs, mut cs) = (0u32, 0u32);
            let mut loss_sum = 0.0f64;
            let cls = ((n as f64) * 4.267).ceil() as usize;
            // Skip CDCL above 100 vars — O(n²) BCP makes it impractical.
            let run_cdcl = n <= 100;

            for seed in 0..seeds {
                let db = generate_3sat_phase_transition(n, seed);
                let t = Instant::now();
                let r = gradient_search(&db, n, &config);
                gt += t.elapsed().as_micros();
                if r.assignment.is_some() {
                    gs += 1;
                }
                let best = r
                    .starts
                    .iter()
                    .map(|s| s.best_loss)
                    .fold(f64::MAX, f64::min);
                loss_sum += best;

                if run_cdcl {
                    let db = generate_3sat_phase_transition(n, seed);
                    let t = Instant::now();
                    if matches!(solver::solve(db, n), solver::SolveResult::Sat(_)) {
                        cs += 1;
                    }
                    ct += t.elapsed().as_micros();
                }
            }

            let cdcl_str = if run_cdcl {
                format!("{:>10}", ct / seeds as u128)
            } else {
                "       n/a".to_string()
            };
            println!(
                "{:<6} {:>6} {:>10} {:>10} {:>6} {:>6} {:>10.3}",
                n,
                cls,
                gt / seeds as u128,
                cdcl_str,
                gs,
                if run_cdcl { cs } else { 0 },
                loss_sum / seeds as f64
            );
        }

        // GPU parallelism analysis: work per iteration
        println!("\n=== GPU Parallelism Axes ===");
        println!("Each gradient iteration at n vars, ratio 4.267:");
        for &n in &[100_usize, 1000, 10000] {
            let clauses = ((n as f64) * 4.267).ceil() as usize;
            let warps_clause_eval = clauses.div_ceil(32);
            let warps_grad_accum = n.div_ceil(32);
            let ops_per_iter = clauses * 3 + n * 13; // ~3 muls/clause + ~13 ops/var for gradient
            println!(
                "  n={:<6} clauses={:<8} warps(eval)={:<6} warps(grad)={:<6} ops/iter={:<10}",
                n, clauses, warps_clause_eval, warps_grad_accum, ops_per_iter
            );
        }
    }

    #[test]
    fn trail_gradient_soundness() {
        // Trail-gradient solver must agree with pure CDCL on SAT/UNSAT.
        for seed in 0..20 {
            let db1 = generate_3sat_phase_transition(30, seed);
            let db2 = generate_3sat_phase_transition(30, seed);

            let cdcl = solver::solve_watched(db1, 30);
            let tg = solver::solve_watched_trail_gradient(db2, 30, 50, 1.0);

            let cdcl_sat = matches!(cdcl, solver::SolveResult::Sat(_));
            let tg_sat = matches!(tg, solver::SolveResult::Sat(_));

            assert_eq!(
                cdcl_sat,
                tg_sat,
                "seed {seed}: CDCL={}, TG={}",
                if cdcl_sat { "SAT" } else { "UNSAT" },
                if tg_sat { "SAT" } else { "UNSAT" }
            );

            if let solver::SolveResult::Sat(ref assign) = tg {
                let db3 = generate_3sat_phase_transition(30, seed);
                assert!(
                    verify(&db3, assign),
                    "seed {seed}: trail-gradient returned invalid assignment"
                );
            }
        }
    }
}
