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
        let crefs = db.crefs();

        for (ci, &cref) in crefs.iter().enumerate() {
            let lits = &db.clause(cref).literals;
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

// ─── Axis 2: Ballot-based discrete verification ─────────────────────

/// Check whether clause `ci` is satisfied under a discrete assignment.
///
/// Per-lane kernel body for Axis 2. On GPU, this feeds `warp.ballot()`.
/// On SimWarp, we collect booleans into a bitmask manually.
///
/// For 3-SAT: 3 variable loads + 3 comparisons + 2 ORs.
#[inline]
fn clause_satisfied(data: &ClauseDataSoA, assign: &[bool], ci: usize) -> bool {
    for pos in 0..3 {
        let val = assign[data.vars[pos][ci] as usize];
        let sat = if data.negs[pos][ci] { !val } else { val };
        if sat {
            return true; // clause satisfied by this literal
        }
    }
    false
}

/// Verify all clauses via SimWarp ballot simulation.
///
/// Each warp batch: 32 lanes check one clause each, results collected
/// into a bitmask (simulated ballot). All clauses satisfied iff every
/// real clause's lane is set in the ballot mask.
///
/// On GPU this maps to:
/// ```text
/// let sat: PerLane<bool> = PerLane::new(clause_satisfied(...));
/// let ballot: BallotResult = warp.ballot(sat);
/// // All SAT iff popcount(ballot) >= num_real_clauses_in_batch
/// ```
///
/// Returns `(all_satisfied, num_satisfied, num_total)`.
pub fn verify_simwarp(data: &ClauseDataSoA, assign: &[bool]) -> (bool, usize, usize) {
    use warp_types::simwarp::SimWarp;

    let mut total_sat = 0usize;
    for batch in 0..data.num_batches() {
        let offset = batch * WARP_SIZE;
        // Per-lane: check clause satisfaction
        let sat_lanes = SimWarp::<u32>::new(|lane| {
            let ci = offset + lane as usize;
            if ci < data.num_clauses {
                clause_satisfied(data, assign, ci) as u32
            } else {
                1 // padding lanes count as "satisfied"
            }
        });
        // Simulated ballot: collect per-lane booleans into a bitmask.
        // On real GPU: warp.ballot(PerLane::new(sat)) → BallotResult
        let mut ballot_mask = 0u64;
        for lane in 0..WARP_SIZE {
            if sat_lanes.lane(lane) != 0 {
                ballot_mask |= 1u64 << lane;
            }
        }
        // popcount of real clauses in this batch
        let batch_clauses = (data.num_clauses - offset).min(WARP_SIZE);
        let real_mask = (1u64 << batch_clauses) - 1; // lower batch_clauses bits
        let batch_sat = (ballot_mask & real_mask).count_ones() as usize;
        total_sat += batch_sat;
    }
    (total_sat == data.num_clauses, total_sat, data.num_clauses)
}

/// Discretize continuous variables and verify via simulated ballot.
///
/// Combines `discretize` (threshold at 0.5) with ballot verification.
/// Returns `(all_satisfied, assignment)`.
pub fn discretize_and_verify_simwarp(
    data: &ClauseDataSoA,
    x: &[f64],
) -> (bool, Vec<bool>) {
    let assign: Vec<bool> = x.iter().map(|&v| v >= 0.5).collect();
    let (all_sat, _, _) = verify_simwarp(data, &assign);
    (all_sat, assign)
}

// ─── Axis 3: Confidence ranking for hybrid CDCL seeding ─────────────

/// Confidence of variable `v`: distance from the 0.5 decision boundary.
///
/// Higher confidence = stronger gradient signal about polarity.
/// Used to rank variables for hybrid CDCL seeding (TurboSAT strategy).
#[inline]
fn var_confidence(x: &[f64], v: usize) -> f64 {
    (x[v] - 0.5).abs()
}

/// Rank variables by confidence, returning `(var_index, confidence)` pairs
/// sorted descending by confidence.
///
/// On GPU, this maps to bitonic sort (Axis 3 from the design doc).
/// `bitonic_sort` requires `Ord` which f64 lacks (NaN), so the GPU
/// version would scale to `i32` via `(confidence * 1e6) as i32` and
/// carry variable indices alongside. SimWarp `bitonic_sort` operates
/// on `i32` with this same encoding.
///
/// This CPU version uses a simple sort — the SimWarp version below
/// validates the bitonic sort encoding.
pub fn confidence_ranking(x: &[f64]) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = x.iter().enumerate().map(|(v, _)| (v, var_confidence(x, v))).collect();
    // Sort descending by confidence. f64 partial_cmp is fine here —
    // confidence values are always in [0, 0.5], no NaN possible.
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked
}

/// Confidence ranking via SimWarp bitonic sort (i32 encoding).
///
/// Encodes each variable's confidence as `(confidence_i32, var_index)` packed
/// into a single i32: `confidence_i32 << 16 | var_index`. Bitonic sort on
/// this composite key sorts by confidence (high bits) with var_index as
/// tiebreaker. After sort, unpack to get ranked variable indices.
///
/// Limited to 32 variables per warp (one per lane). For n > 32, multiple
/// warps handle successive variable chunks; cross-warp merge is future work.
///
/// On GPU: `warp.bitonic_sort(PerLane::new(packed_key))`.
pub fn confidence_ranking_simwarp(x: &[f64]) -> Vec<(usize, f64)> {
    use warp_types::simwarp::{bitonic_sort, SimWarp};

    let n = x.len().min(WARP_SIZE);

    // Pack: high 15 bits = inverted scaled confidence, low 16 bits = var index.
    // Bitonic sort is ascending, so invert confidence for descending order.
    // Confidence ∈ [0.0, 0.5] → scaled to [0, 30000] → inverted to [30000, 0].
    // Max packed value: 30000 << 16 | 31 = 1,966,080,031 — fits in i32.
    let packed = SimWarp::<i32>::new(|lane| {
        if (lane as usize) < n {
            let conf = var_confidence(x, lane as usize);
            let conf_scaled = (30_000.0 - conf * 60_000.0) as i32; // invert: low = high confidence
            (conf_scaled << 16) | (lane as i32)
        } else {
            i32::MAX // padding sorts to end (highest = lowest confidence)
        }
    });

    let sorted = bitonic_sort(&packed);

    // Unpack: extract var_index and original confidence.
    (0..n)
        .map(|lane| {
            let key = sorted.lane(lane);
            let var_idx = (key & 0xFFFF) as usize;
            let conf = var_confidence(x, var_idx);
            (var_idx, conf)
        })
        .collect()
}

// ─── End-to-end: gradient search via GPU kernel path ─────────────────

/// Update clause weights via SoA layout (EMA on violation frequency).
///
/// Same formula as `gradient.rs::update_weights` but over `ClauseDataSoA`:
/// `w = 0.9 * w + 0.1 * (violated ? 1 : 0)`.
fn update_weights_soa(data: &mut ClauseDataSoA, assign: &[bool]) {
    for ci in 0..data.num_clauses {
        let satisfied = clause_satisfied(data, assign, ci);
        data.weights[ci] = 0.9 * data.weights[ci] + if satisfied { 0.0 } else { 0.1 };
    }
}

/// Gradient search using the GPU kernel path (SimWarp evaluation).
///
/// Functionally identical to `gradient::gradient_search` — same algorithm,
/// same RNG, same convergence behavior — but uses the warp-parallel
/// primitives for loss, gradient, and verification:
///
/// - **Loss**: `total_loss_simwarp` (Axis 1: clause-parallel butterfly reduce)
/// - **Gradient**: `gradient_soa` (SoA reverse index, same math)
/// - **Verify**: `verify_simwarp` (Axis 2: simulated ballot + popcount)
/// - **Weight update**: `update_weights_soa` (per-clause EMA via SoA)
///
/// On GPU, each of these becomes a kernel launch (or fused kernel).
/// On CPU via SimWarp, they validate the parallel algorithm produces
/// identical optimization trajectories to the sequential CPU path.
pub fn gradient_search_simwarp(
    db: &ClauseDb,
    num_vars: u32,
    config: &crate::gradient::GradientConfig,
) -> crate::gradient::GradientResult {
    use crate::gradient::{GradientResult, Rng, StartOutcome};

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

    // One-time SoA packing (on GPU: host→device transfer).
    let initial_weights = vec![1.0; db.len()];
    let (soa_template, _skipped) = ClauseDataSoA::from_clause_db(db, &initial_weights);
    let var_idx = VarIndexSoA::build(&soa_template, num_vars);

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
        // Clone SoA for this start (weights reset to 1.0).
        let mut soa = soa_template.clone();
        let mut lr = config.learning_rate;
        let mut best_loss = f64::MAX;
        let mut found = false;

        for iter in 0..config.max_iters {
            // Axis 1: clause-parallel loss via SimWarp butterfly reduce.
            let l = total_loss_simwarp(&soa, &x);
            result.clause_evals += soa.num_clauses;

            if l < best_loss {
                best_loss = l;
            }

            // Periodically discretize and verify (Axis 2: simulated ballot).
            if l < 1.0 || iter % 10 == 0 {
                let assign: Vec<bool> = x.iter().map(|&v| v >= 0.5).collect();
                let (all_sat, _, _) = verify_simwarp(&soa, &assign);
                if all_sat {
                    result.assignment = Some(assign);
                    result.starts.push(StartOutcome {
                        best_loss: l,
                        iterations: iter + 1,
                        satisfied: true,
                    });
                    found = true;
                    break;
                }
                if config.clause_weights {
                    update_weights_soa(&mut soa, &assign);
                }
            }

            // Gradient via SoA reverse index.
            gradient_soa(&soa, &x, &var_idx, &mut grad);

            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-20 {
                break;
            }

            // Variable update (same as CPU — no warp primitive needed).
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
            let assign: Vec<bool> = x.iter().map(|&v| v >= 0.5).collect();
            let (all_sat, _, _) = verify_simwarp(&soa, &assign);
            if all_sat {
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
            if best_loss < global_best_loss {
                global_best_loss = best_loss;
                result.best_continuous = Some(x);
            }
        }
    }

    result
}

// ─── Hybrid solver via GPU kernel path ───────────────────────────────

/// Hybrid gradient→CDCL solver using GPU kernel primitives.
///
/// Same strategy as `gradient::hybrid_solve` (TurboSAT), but uses
/// the SimWarp-validated GPU kernel path for all three phases:
///
/// - **Phase 1**: `gradient_search_simwarp` (Axis 1 + Axis 2)
/// - **Phase 2-3**: `confidence_ranking_simwarp` (Axis 3 bitonic sort)
///   → warm-start VSIDS activity + phase hints
/// - **Phase 4**: `solve_cdcl_core` (unchanged CPU CDCL)
///
/// On GPU, Phases 1-3 would be kernel launches; Phase 4 stays on CPU.
/// The confidence ranking sorts variables by `|x - 0.5|` via bitonic sort,
/// then maps rank to VSIDS activity (highest confidence → highest activity
/// → decided first by CDCL).
pub fn hybrid_solve_simwarp(
    db: crate::bcp::ClauseDb,
    num_vars: u32,
    config: &crate::gradient::GradientConfig,
) -> crate::solver::SolveResult {
    use crate::solver::SolveResult;
    use crate::vsids::Vsids;

    // Phase 1: gradient search via SimWarp GPU kernel path.
    let grad = gradient_search_simwarp(&db, num_vars, config);

    if let Some(assign) = grad.assignment {
        return SolveResult::Sat(assign);
    }

    // Phase 2-3: warm-start VSIDS from gradient confidence via bitonic sort (Axis 3).
    let mut vsids = Vsids::new(num_vars);
    if let Some(ref x) = grad.best_continuous {
        // Axis 3: bitonic sort ranks variables by confidence.
        let ranked = confidence_ranking_simwarp(x);

        // Map rank position to VSIDS activity: rank 0 (most confident) gets
        // the highest activity. Scale to be comparable with clause-occurrence
        // initialization (~12.8 per var at ratio 4.267).
        for &(var, conf) in ranked.iter() {
            let var = var as u32;
            vsids.set_phase(var, x[var as usize] >= 0.5);
            // Activity from confidence: higher confidence → higher activity.
            // Scale by 20.0 to match the CPU hybrid's magnitude.
            vsids.set_initial_activity(var, conf * 20.0);
        }
    }
    vsids.initialize_from_clauses(&db);

    // Phase 4: CDCL with warm-started VSIDS.
    crate::solver::solve_cdcl_core(db, num_vars, vsids)
}

// ─── GPU hardware path ──────────────────────────────────────────────

/// Gradient search with GPU-accelerated loss evaluation.
///
/// Identical to `gradient_search_simwarp` except the inner-loop loss
/// call runs on real GPU hardware via `GpuContext::total_loss()`.
/// Gradient, verification, and weight update remain on CPU.
///
/// Upload SoA once; re-upload `x` each iteration (cheap: num_vars × 8 bytes).
/// Re-upload weights only when clause weight adaptation triggers.
#[cfg(feature = "gpu")]
pub fn gradient_search_gpu(
    db: &ClauseDb,
    num_vars: u32,
    config: &crate::gradient::GradientConfig,
    ctx: &crate::gpu_launcher::GpuContext,
) -> crate::gradient::GradientResult {
    use crate::gradient::{GradientResult, Rng, StartOutcome};

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

    // One-time SoA packing + GPU upload.
    let initial_weights = vec![1.0; db.len()];
    let (soa_template, _skipped) = ClauseDataSoA::from_clause_db(db, &initial_weights);

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
        let mut soa = soa_template.clone();
        let mut gpu_data = ctx.upload_clause_data(&soa)
            .expect("GPU upload failed");
        let mut lr = config.learning_rate;
        let mut best_loss = f64::MAX;
        let mut found = false;

        for iter in 0..config.max_iters {
            // Fused: loss + gradient in one kernel launch (Axis 1 + 1b).
            let (l, gpu_grad) = ctx.total_loss_and_grad(&gpu_data, &x, n)
                .expect("GPU fused kernel failed");
            grad.copy_from_slice(&gpu_grad);
            result.clause_evals += soa.num_clauses;

            if l < best_loss {
                best_loss = l;
            }

            // Periodically discretize and verify (CPU — Axis 2 ballot not yet on GPU).
            if l < 1.0 || iter % 10 == 0 {
                let assign: Vec<bool> = x.iter().map(|&v| v >= 0.5).collect();
                let (all_sat, _, _) = verify_simwarp(&soa, &assign);
                if all_sat {
                    result.assignment = Some(assign);
                    result.starts.push(StartOutcome {
                        best_loss: l,
                        iterations: iter + 1,
                        satisfied: true,
                    });
                    found = true;
                    break;
                }
                if config.clause_weights {
                    update_weights_soa(&mut soa, &assign);
                    ctx.update_weights(&mut gpu_data, &soa)
                        .expect("GPU weight re-upload failed");
                }
            }

            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-20 {
                break;
            }

            // Variable update (CPU — trivial, no GPU benefit).
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
            let assign: Vec<bool> = x.iter().map(|&v| v >= 0.5).collect();
            let (all_sat, _, _) = verify_simwarp(&soa, &assign);
            if all_sat {
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
            if best_loss < global_best_loss {
                global_best_loss = best_loss;
                result.best_continuous = Some(x);
            }
        }
    }

    result
}

/// Device-resident gradient search: all 3 kernels launch per iteration,
/// data stays on GPU. Only downloads loss partials + grad norm (tiny).
///
/// Kernel sequence per iteration:
///   1. clause_loss_grad_fused  → loss partials + grad accumulation
///   2. variable_update         → reads grad, updates x + velocity
///   3. grad_norm_reduce        → reads grad, reduces sum(grad²)
///
/// Kernels 2 and 3 both read grad[] — Pattern 5 fusion candidate.
#[cfg(feature = "gpu")]
pub fn gradient_search_gpu_resident(
    db: &ClauseDb,
    num_vars: u32,
    config: &crate::gradient::GradientConfig,
    ctx: &crate::gpu_launcher::GpuContext,
) -> crate::gradient::GradientResult {
    use crate::gradient::{GradientResult, Rng, StartOutcome};

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

    let initial_weights = vec![1.0; db.len()];
    let (soa_template, _skipped) = ClauseDataSoA::from_clause_db(db, &initial_weights);

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
        let x_init: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
        let mut soa = soa_template.clone();
        let mut gpu_data = ctx.upload_clause_data(&soa)
            .expect("GPU upload failed");
        let mut state = ctx.alloc_iter_state(&x_init)
            .expect("GPU iter state alloc failed");
        let mut lr = config.learning_rate;
        let mut best_loss = f64::MAX;
        let mut found = false;

        for iter in 0..config.max_iters {
            // All 3 kernels launch; data stays on GPU.
            let (l, grad_norm_sq) = ctx.gpu_iteration(&gpu_data, &mut state, lr, config.momentum)
                .expect("GPU iteration failed");
            result.clause_evals += soa.num_clauses;

            if l < best_loss {
                best_loss = l;
            }

            // Periodically download x for verification (CPU — Axis 2 ballot not yet on GPU).
            if l < 1.0 || iter % 10 == 0 {
                let x_host = ctx.download_x(&state).expect("x download failed");
                let assign: Vec<bool> = x_host.iter().map(|&v| v >= 0.5).collect();
                let (all_sat, _, _) = verify_simwarp(&soa, &assign);
                if all_sat {
                    result.assignment = Some(assign);
                    result.starts.push(StartOutcome {
                        best_loss: l,
                        iterations: iter + 1,
                        satisfied: true,
                    });
                    found = true;
                    break;
                }
                if config.clause_weights {
                    update_weights_soa(&mut soa, &assign);
                    ctx.update_weights(&mut gpu_data, &soa)
                        .expect("GPU weight re-upload failed");
                }
            }

            if grad_norm_sq < 1e-20 {
                break;
            }

            lr *= config.lr_decay;
        }

        if !found {
            let x_host = ctx.download_x(&state).expect("x download failed");
            let assign: Vec<bool> = x_host.iter().map(|&v| v >= 0.5).collect();
            let (all_sat, _, _) = verify_simwarp(&soa, &assign);
            if all_sat {
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
            if best_loss < global_best_loss {
                global_best_loss = best_loss;
                result.best_continuous = Some(x_host);
            }
        }
    }

    result
}

/// Hybrid gradient→CDCL solver with GPU-accelerated gradient phase.
///
/// Same TurboSAT strategy as `hybrid_solve_simwarp`:
/// - **Phase 1**: `gradient_search_gpu` (GPU loss + CPU gradient/verify)
/// - **Phase 2-3**: `confidence_ranking_simwarp` (CPU bitonic sort → VSIDS warm-start)
/// - **Phase 4**: `solve_cdcl_core` (CPU CDCL)
#[cfg(feature = "gpu")]
pub fn hybrid_solve_gpu(
    db: crate::bcp::ClauseDb,
    num_vars: u32,
    config: &crate::gradient::GradientConfig,
    ctx: &crate::gpu_launcher::GpuContext,
) -> crate::solver::SolveResult {
    use crate::solver::SolveResult;
    use crate::vsids::Vsids;

    let grad = gradient_search_gpu(&db, num_vars, config, ctx);

    if let Some(assign) = grad.assignment {
        return SolveResult::Sat(assign);
    }

    let mut vsids = Vsids::new(num_vars);
    if let Some(ref x) = grad.best_continuous {
        let ranked = confidence_ranking_simwarp(x);
        for &(var, conf) in ranked.iter() {
            let var = var as u32;
            vsids.set_phase(var, x[var as usize] >= 0.5);
            vsids.set_initial_activity(var, conf * 20.0);
        }
    }
    vsids.initialize_from_clauses(&db);

    crate::solver::solve_cdcl_core(db, num_vars, vsids)
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::generate_3sat_phase_transition;
    use crate::gradient;
    use crate::literal::Lit;

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

        for (ci, cref) in db.iter_crefs().enumerate() {
            let lits = &db.clause(cref).literals;
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

            let crefs = db.crefs();
            let cpu = gradient::loss(&db, &crefs, &x, &weights);
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

        let crefs = db.crefs();
        let cpu = gradient::loss(&db, &crefs, &x, &weights);
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
            let crefs = db.crefs();
            let cpu_idx = gradient::VarIndex::build(&db, 20, &crefs);
            let mut cpu_grad = vec![0.0; 20];
            gradient::gradient(&db, &crefs, &x, &cpu_idx, &weights, &mut cpu_grad);

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

        let crefs = db.crefs();
        let cpu = gradient::loss(&db, &crefs, &x, &weights);
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

    // ─── Axis 2: Ballot verification tests ───────────────────────────

    #[test]
    fn ballot_verify_matches_cpu() {
        // Ballot-based verification must agree with CPU verify() on
        // the same discrete assignment.
        for seed in 0..10 {
            let db = generate_3sat_phase_transition(20, seed);
            let weights = vec![1.0; db.len()];
            let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

            // Two assignments: all-true and all-false
            for &val in &[true, false] {
                let assign = vec![val; 20];
                let cpu_sat = gradient::verify(&db, &assign);
                let (gpu_sat, sat_count, total) = verify_simwarp(&soa, &assign);

                assert_eq!(
                    cpu_sat, gpu_sat,
                    "seed {seed}, all-{val}: CPU={cpu_sat}, GPU={gpu_sat} ({sat_count}/{total})"
                );
            }
        }
    }

    #[test]
    fn ballot_verify_with_known_solution() {
        // Use gradient solver to find a SAT assignment, then verify via ballot.
        for seed in 0..5 {
            let db = generate_3sat_phase_transition(20, seed);
            let weights = vec![1.0; db.len()];
            let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

            let r = gradient::gradient_search(&db, 20, &gradient::GradientConfig::enhanced());
            if let Some(ref assign) = r.assignment {
                let (gpu_sat, sat_count, total) = verify_simwarp(&soa, assign);
                assert!(
                    gpu_sat,
                    "seed {seed}: gradient found SAT but ballot says UNSAT ({sat_count}/{total})"
                );
            }
        }
    }

    #[test]
    fn ballot_verify_unsat_instance() {
        // x0 ∧ ¬x0 — trivially UNSAT. No assignment satisfies both.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(0), Lit::pos(0)]); // (x0 ∨ x0 ∨ x0)
        db.add_clause(vec![Lit::neg(0), Lit::neg(0), Lit::neg(0)]); // (¬x0 ∨ ¬x0 ∨ ¬x0)
        let weights = vec![1.0; 2];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        // x0=true satisfies clause 0 but not clause 1
        let (sat_t, count_t, _) = verify_simwarp(&soa, &[true]);
        assert!(!sat_t, "x0=true should not satisfy both clauses");
        assert_eq!(count_t, 1); // only first clause satisfied

        // x0=false satisfies clause 1 but not clause 0
        let (sat_f, count_f, _) = verify_simwarp(&soa, &[false]);
        assert!(!sat_f, "x0=false should not satisfy both clauses");
        assert_eq!(count_f, 1);
    }

    #[test]
    fn discretize_and_verify_round_trip() {
        // Continuous values near 1.0 for all vars → discretize to true → check.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)]);
        let weights = vec![1.0];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        let x = vec![0.9, 0.8, 0.7]; // all > 0.5 → all true → clause SAT
        let (sat, assign) = discretize_and_verify_simwarp(&soa, &x);
        assert!(sat, "high continuous values should discretize to SAT");
        assert!(assign.iter().all(|&b| b), "all vars should round to true");
    }

    #[test]
    fn ballot_padding_does_not_affect_result() {
        // 3 clauses padded to 32 — padding should not affect verification.
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::pos(0), Lit::neg(1), Lit::pos(2)]);
        db.add_clause(vec![Lit::neg(0), Lit::pos(1), Lit::neg(2)]);
        let weights = vec![1.0; 3];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        let assign = vec![true, true, true];
        let cpu_sat = gradient::verify(&db, &assign);
        let (gpu_sat, sat_count, total) = verify_simwarp(&soa, &assign);

        assert_eq!(cpu_sat, gpu_sat);
        assert_eq!(total, 3); // only 3 real clauses counted
        assert_eq!(sat_count, 3); // all-true satisfies all three
    }

    // ─── Axis 3: Confidence ranking tests ────────────────────────────

    #[test]
    fn confidence_ranking_order() {
        // Variables at different distances from 0.5 should rank correctly.
        let x = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        // Confidences: 0.4, 0.0, 0.4, 0.2, 0.2
        // Expected order: var0 or var2 (0.4), then var3 or var4 (0.2), then var1 (0.0)
        let ranked = confidence_ranking(&x);
        assert_eq!(ranked.len(), 5);
        // Top 2 should have confidence 0.4
        assert!((ranked[0].1 - 0.4).abs() < 1e-10);
        assert!((ranked[1].1 - 0.4).abs() < 1e-10);
        // Last should have confidence 0.0
        assert!((ranked[4].1 - 0.0).abs() < 1e-10);
        assert_eq!(ranked[4].0, 1); // var1 at exactly 0.5
    }

    #[test]
    fn confidence_ranking_simwarp_agrees() {
        // SimWarp bitonic sort ranking should produce the same order
        // as the CPU argsort (for n <= 32 variables).
        let x = init_vars(20);
        let cpu_ranked = confidence_ranking(&x);
        let gpu_ranked = confidence_ranking_simwarp(&x);

        assert_eq!(cpu_ranked.len(), gpu_ranked.len());

        // Confidence values should match (order may differ for ties,
        // but confidence values in sorted order must be the same).
        let cpu_confs: Vec<f64> = cpu_ranked.iter().map(|r| r.1).collect();
        let gpu_confs: Vec<f64> = gpu_ranked.iter().map(|r| r.1).collect();
        for i in 0..cpu_confs.len() {
            assert!(
                (cpu_confs[i] - gpu_confs[i]).abs() < 0.01,
                "rank {i}: CPU conf {:.6} != GPU conf {:.6}",
                cpu_confs[i],
                gpu_confs[i]
            );
        }
    }

    #[test]
    fn confidence_top_k_for_cdcl_seeding() {
        // Top-k most confident variables should be the ones farthest from 0.5.
        let x = vec![
            0.99, 0.50, 0.01, 0.50, 0.50, // var0=high, var2=high, rest=undecided
            0.50, 0.50, 0.50, 0.50, 0.50,
        ];
        let ranked = confidence_ranking_simwarp(&x);

        // Top 2 should be var0 (conf 0.49) and var2 (conf 0.49)
        let top2_vars: Vec<usize> = ranked[..2].iter().map(|r| r.0).collect();
        assert!(
            top2_vars.contains(&0) && top2_vars.contains(&2),
            "top-2 should be var0 and var2, got {:?}",
            top2_vars
        );
    }

    // ─── End-to-end: SimWarp gradient search tests ───────────────────

    #[test]
    fn simwarp_search_matches_cpu_vanilla() {
        // Vanilla gradient search: SimWarp path must find the same solutions
        // as CPU path with the same RNG seeds.
        let config = gradient::GradientConfig::default();
        for seed in 0..10 {
            let db1 = generate_3sat_phase_transition(20, seed);
            let db2 = generate_3sat_phase_transition(20, seed);

            let cpu = gradient::gradient_search(&db1, 20, &config);
            let gpu = gradient_search_simwarp(&db2, 20, &config);

            assert_eq!(
                cpu.assignment.is_some(),
                gpu.assignment.is_some(),
                "seed {seed}: CPU found={} GPU found={}",
                cpu.assignment.is_some(),
                gpu.assignment.is_some()
            );

            // If both found solutions, verify both are valid.
            if let (Some(ref ca), Some(ref ga)) = (&cpu.assignment, &gpu.assignment) {
                let db3 = generate_3sat_phase_transition(20, seed);
                assert!(gradient::verify(&db3, ca), "seed {seed}: CPU assignment invalid");
                let db4 = generate_3sat_phase_transition(20, seed);
                assert!(gradient::verify(&db4, ga), "seed {seed}: GPU assignment invalid");
            }

            // Per-start diagnostics should match.
            assert_eq!(
                cpu.starts.len(),
                gpu.starts.len(),
                "seed {seed}: different number of starts completed"
            );
            for (i, (cs, gs)) in cpu.starts.iter().zip(gpu.starts.iter()).enumerate() {
                assert_eq!(
                    cs.satisfied, gs.satisfied,
                    "seed {seed} start {i}: CPU satisfied={} GPU satisfied={}",
                    cs.satisfied, gs.satisfied
                );
                assert!(
                    (cs.best_loss - gs.best_loss).abs() < 1e-8,
                    "seed {seed} start {i}: CPU loss {:.10} != GPU loss {:.10}",
                    cs.best_loss,
                    gs.best_loss
                );
                assert_eq!(
                    cs.iterations, gs.iterations,
                    "seed {seed} start {i}: CPU iters={} GPU iters={}",
                    cs.iterations, gs.iterations
                );
            }
        }
    }

    #[test]
    fn simwarp_search_matches_cpu_enhanced() {
        // Enhanced config (momentum + clause weights): more divergence-prone
        // because weight updates compound differently with floating point.
        let config = gradient::GradientConfig::enhanced();
        for seed in 0..10 {
            let db1 = generate_3sat_phase_transition(20, seed);
            let db2 = generate_3sat_phase_transition(20, seed);

            let cpu = gradient::gradient_search(&db1, 20, &config);
            let gpu = gradient_search_simwarp(&db2, 20, &config);

            assert_eq!(
                cpu.assignment.is_some(),
                gpu.assignment.is_some(),
                "seed {seed}: CPU found={} GPU found={}",
                cpu.assignment.is_some(),
                gpu.assignment.is_some()
            );

            // Loss values should match closely (clause weight EMA may accumulate
            // tiny floating-point differences, but the trajectory should be identical
            // since both use the same RNG and the same math).
            for (i, (cs, gs)) in cpu.starts.iter().zip(gpu.starts.iter()).enumerate() {
                assert_eq!(cs.satisfied, gs.satisfied, "seed {seed} start {i}");
                assert!(
                    (cs.best_loss - gs.best_loss).abs() < 1e-6,
                    "seed {seed} start {i}: CPU loss {:.10} != GPU loss {:.10}",
                    cs.best_loss,
                    gs.best_loss
                );
            }
        }
    }

    #[test]
    fn simwarp_search_never_claims_false_sat() {
        // Soundness: SimWarp path must never return a solution that doesn't verify.
        for seed in 0..20 {
            let db1 = generate_3sat_phase_transition(30, seed);
            let db2 = generate_3sat_phase_transition(30, seed);

            let gpu = gradient_search_simwarp(&db1, 30, &gradient::GradientConfig::enhanced());
            if let Some(ref assign) = gpu.assignment {
                assert!(
                    gradient::verify(&db2, assign),
                    "seed {seed}: SimWarp returned invalid assignment"
                );
            }
        }
    }

    #[test]
    fn simwarp_search_50var() {
        // Larger instance: 50 vars at phase transition (~213 clauses).
        // SimWarp path should find solutions at the same rate as CPU.
        let config = gradient::GradientConfig::enhanced();
        let mut cpu_found = 0;
        let mut gpu_found = 0;
        let seeds = 10;

        for seed in 0..seeds {
            let db1 = generate_3sat_phase_transition(50, seed);
            let db2 = generate_3sat_phase_transition(50, seed);

            if gradient::gradient_search(&db1, 50, &config).assignment.is_some() {
                cpu_found += 1;
            }
            if gradient_search_simwarp(&db2, 50, &config).assignment.is_some() {
                gpu_found += 1;
            }
        }
        assert_eq!(
            cpu_found, gpu_found,
            "50-var: CPU found {cpu_found}/{seeds}, GPU found {gpu_found}/{seeds}"
        );
    }

    // ─── Hybrid solver: GPU kernel pipeline tests ────────────────────

    #[test]
    fn hybrid_simwarp_solves_sat() {
        // Hybrid with GPU kernel path should find SAT instances.
        let config = gradient::GradientConfig::enhanced();
        let mut found = 0;
        let seeds = 10;

        for seed in 0..seeds {
            let db = generate_3sat_phase_transition(50, seed);
            if matches!(
                hybrid_solve_simwarp(db, 50, &config),
                crate::solver::SolveResult::Sat(_)
            ) {
                found += 1;
            }
        }
        // Hybrid should find at least as many as gradient alone.
        assert!(found > 0, "hybrid_simwarp should find some SAT instances");
    }

    #[test]
    fn hybrid_simwarp_matches_cpu_hybrid() {
        // GPU hybrid should agree with CPU hybrid on SAT/UNSAT.
        let config = gradient::GradientConfig::enhanced();
        for seed in 0..10 {
            let db1 = generate_3sat_phase_transition(30, seed);
            let db2 = generate_3sat_phase_transition(30, seed);

            let cpu = gradient::hybrid_solve(db1, 30, &config);
            let gpu = hybrid_solve_simwarp(db2, 30, &config);

            // Both should agree on satisfiability.
            let cpu_sat = matches!(cpu, crate::solver::SolveResult::Sat(_));
            let gpu_sat = matches!(gpu, crate::solver::SolveResult::Sat(_));
            assert_eq!(
                cpu_sat, gpu_sat,
                "seed {seed}: CPU hybrid sat={cpu_sat}, GPU hybrid sat={gpu_sat}"
            );

            // If SAT, both assignments must verify.
            if let crate::solver::SolveResult::Sat(ref assign) = gpu {
                let db3 = generate_3sat_phase_transition(30, seed);
                assert!(
                    gradient::verify(&db3, assign),
                    "seed {seed}: GPU hybrid returned invalid assignment"
                );
            }
        }
    }

    #[test]
    fn hybrid_simwarp_soundness() {
        // Soundness: hybrid must never claim SAT on UNSAT, never return
        // invalid assignments.
        for seed in 0..20 {
            let db1 = generate_3sat_phase_transition(30, seed);
            let db2 = generate_3sat_phase_transition(30, seed);

            let gpu = hybrid_solve_simwarp(db1, 30, &gradient::GradientConfig::enhanced());
            let cdcl = crate::solver::solve_watched(db2, 30);

            if let crate::solver::SolveResult::Sat(ref assign) = gpu {
                let db3 = generate_3sat_phase_transition(30, seed);
                assert!(
                    gradient::verify(&db3, assign),
                    "seed {seed}: GPU hybrid invalid assignment"
                );
                assert!(
                    matches!(cdcl, crate::solver::SolveResult::Sat(_)),
                    "seed {seed}: GPU hybrid SAT but CDCL says UNSAT"
                );
            }
        }
    }

    #[test]
    fn hybrid_simwarp_finds_more_than_gradient_alone() {
        // The whole point of hybrid: CDCL finishes what gradient can't.
        let config = gradient::GradientConfig::enhanced();
        let mut grad_found = 0;
        let mut hybrid_found = 0;
        let seeds = 10;

        for seed in 0..seeds {
            let db1 = generate_3sat_phase_transition(50, seed);
            let db2 = generate_3sat_phase_transition(50, seed);

            if gradient_search_simwarp(&db1, 50, &config).assignment.is_some() {
                grad_found += 1;
            }
            if matches!(
                hybrid_solve_simwarp(db2, 50, &config),
                crate::solver::SolveResult::Sat(_)
            ) {
                hybrid_found += 1;
            }
        }
        assert!(
            hybrid_found >= grad_found,
            "hybrid should find at least as many: grad={grad_found}, hybrid={hybrid_found}"
        );
    }
}
