//! PDR checker: the IC3/Property-Directed Reachability engine.
//!
//! Connects the transition system, frame sequence, and SAT solver
//! to prove unbounded safety or find counterexamples.
//!
//! # Algorithm overview
//!
//! 1. **Initiation**: Check I ∧ ¬P. If SAT, property violated initially.
//! 2. **Strengthen**: Find CTIs at the frontier, recursively block via
//!    predecessor queries. If a predecessor chain reaches F₀, a real
//!    counterexample exists.
//! 3. **Propagate**: Push inductive clauses forward. If Fᵢ = Fᵢ₊₁,
//!    an inductive invariant is found.
//! 4. **Extend**: Add a new frame and repeat.

use crate::cube::{shift_lit, Cube};
use crate::frames::{Frame, FrameSequence};
use crate::session::{self, PdrSession};
use crate::phase::*;

use warp_types_bmc::TransitionSystem;
use warp_types_sat::bcp::ClauseDb;
use warp_types_sat::literal::Lit;
use warp_types_sat::solver::{solve_watched_budget, SolveResult};

/// Result of a PDR run.
#[derive(Debug)]
pub enum PdrResult {
    /// Inductive invariant found — property is safe at all depths.
    Safe {
        /// The frame index where convergence was detected.
        invariant_frame: usize,
    },
    /// Counterexample found — concrete state trace to a bad state.
    CounterexampleFound {
        /// Depth of the counterexample (number of transitions).
        depth: u32,
        /// State trace: `trace[0]` is the initial state, `trace[depth]` violates the property.
        trace: Vec<Vec<bool>>,
    },
    /// Frame budget exhausted without conclusive result.
    Exhausted {
        /// Number of frames explored.
        frames_explored: usize,
    },
}

/// A proof obligation: "block this cube at this frame level."
struct Obligation {
    cube: Cube,
    level: usize,
    /// Index of the parent obligation that spawned this one (for trace reconstruction).
    parent: Option<usize>,
}

/// Run Property-Directed Reachability on a transition system.
///
/// Proves the safety property at all depths (returning `Safe` with the
/// invariant frame), finds a concrete counterexample (`CounterexampleFound`),
/// or exhausts the frame budget (`Exhausted`).
///
/// # Arguments
/// - `sys`: the transition system (initial states, transitions, safety property)
/// - `max_frames`: maximum number of frames before giving up
/// - `conflict_budget`: SAT conflict budget per query (0 = unlimited)
pub fn check(
    sys: &TransitionSystem,
    max_frames: u32,
    conflict_budget: u64,
) -> PdrResult {
    let n = sys.num_state_vars;

    session::with_session(|init: PdrSession<'_, Init>| {
        let modeled = init.build_model();

        // ── Step 1: Initiation check ──
        // Is I(s) ∧ ¬P(s) satisfiable?
        if let Some(assignment) = check_initiation(sys, conflict_budget) {
            let trace = vec![assignment[..n as usize].to_vec()];
            let _cex = modeled.check_counterexample();
            return PdrResult::CounterexampleFound { depth: 0, trace };
        }

        // ── Step 2: Initialize frame sequence ──
        let mut frames = FrameSequence::new();

        // F₀ = initial-state clauses
        let init_clauses: Vec<Vec<Lit>> = sys
            .initial
            .iter()
            .map(|c| c.lits.clone())
            .collect();
        frames.push(Frame::from_clauses(init_clauses));

        // F₁ = empty (will accumulate blocking clauses)
        frames.push(Frame::new());

        // ── Step 3: Main PDR loop ──
        for _iteration in 0..max_frames {
            let k = frames.frontier();

            // STRENGTHEN: block all CTIs at the frontier
            loop {
                let cti = find_cti(sys, &frames, k, conflict_budget);
                match cti {
                    None => break, // No more CTIs — frontier is clean
                    Some(cube) => {
                        match block_cube(sys, &mut frames, cube, k, conflict_budget) {
                            BlockResult::Blocked => continue,
                            BlockResult::Counterexample(trace) => {
                                let _cex = modeled.check_counterexample();
                                return PdrResult::CounterexampleFound {
                                    depth: trace.len() as u32 - 1,
                                    trace,
                                };
                            }
                        }
                    }
                }
            }

            // PROPAGATE: push clauses forward, check convergence
            if let Some(inv_frame) = propagate_clauses(sys, &mut frames, conflict_budget) {
                let _safe = modeled.check_safe();
                return PdrResult::Safe {
                    invariant_frame: inv_frame,
                };
            }

            // EXTEND: add a new frame
            frames.push(Frame::new());
        }

        let _exhausted = modeled.check_exhausted();
        PdrResult::Exhausted {
            frames_explored: frames.len(),
        }
    })
}

// ============================================================================
// Result of blocking attempt
// ============================================================================

enum BlockResult {
    /// Cube successfully blocked (and generalized clause added to frames).
    Blocked,
    /// Real counterexample found — returns state trace.
    Counterexample(Vec<Vec<bool>>),
}

// ============================================================================
// SAT query: initiation
// ============================================================================

/// Check I(s) ∧ ¬P(s). Returns Some(assignment) if SAT (initial violation).
fn check_initiation(sys: &TransitionSystem, conflict_budget: u64) -> Option<Vec<bool>> {
    let n = sys.num_state_vars;
    let num_tseitin = sys.property.len() as u32;
    let total_vars = n + num_tseitin;

    let mut db = ClauseDb::new();

    // Initial-state clauses I(s)
    for clause in &sys.initial {
        db.add_clause(clause.lits.clone());
    }

    // Negated property ¬P(s) via Tseitin
    add_negated_property(&mut db, sys, 0, n);

    let (result, _) = solve_watched_budget(db, total_vars, conflict_budget);
    match result {
        SolveResult::Sat(assign) => Some(assign),
        _ => None,
    }
}

// ============================================================================
// SAT query: find CTI
// ============================================================================

/// Find a counterexample-to-induction at frame `level`.
/// Checks: Fₖ ∧ ¬P — is there a bad state in the frame?
/// Returns the bad-state cube if SAT.
fn find_cti(
    sys: &TransitionSystem,
    frames: &FrameSequence,
    level: usize,
    conflict_budget: u64,
) -> Option<Cube> {
    let n = sys.num_state_vars;
    let num_tseitin = sys.property.len() as u32;
    let total_vars = n + num_tseitin;

    let mut db = ClauseDb::new();

    // Frame clauses (current-state)
    add_frame_clauses(&mut db, frames.frame(level), 0);

    // Negated property: ¬P(s)
    add_negated_property(&mut db, sys, 0, n);

    let (result, _) = solve_watched_budget(db, total_vars, conflict_budget);
    match result {
        SolveResult::Sat(assign) => Some(Cube::from_assignment(&assign, n)),
        _ => None,
    }
}

// ============================================================================
// SAT query: consecution (relative induction)
// ============================================================================

/// Check consecution: is Fₖ ∧ ¬clause ∧ T ∧ cube' satisfiable?
/// If SAT, the cube has a predecessor in Fₖ — returns the predecessor.
/// If UNSAT, the cube is blocked at this level.
fn check_predecessor(
    sys: &TransitionSystem,
    frames: &FrameSequence,
    cube: &Cube,
    level: usize,
    conflict_budget: u64,
) -> Option<Cube> {
    let n = sys.num_state_vars;
    let total_vars = 2 * n;

    let mut db = ClauseDb::new();

    // Frame clauses at level (current-state)
    add_frame_clauses(&mut db, frames.frame(level), 0);

    // Transition relation
    for tc in &sys.transition {
        db.add_clause(tc.lits.clone());
    }

    // Cube as unit clauses over next-state variables
    let shifted = cube.shift(n);
    for &lit in &shifted.lits {
        db.add_clause(vec![lit]);
    }

    let (result, _) = solve_watched_budget(db, total_vars, conflict_budget);
    match result {
        SolveResult::Sat(assign) => Some(Cube::from_assignment(&assign, n)),
        _ => None,
    }
}

// ============================================================================
// Block cube (recursive via obligation queue)
// ============================================================================

/// Attempt to block a cube at the given frame level.
/// Returns `Blocked` if successful, or `Counterexample(trace)` if the cube
/// is reachable from the initial states.
///
/// Uses a min-heap priority queue (lowest level first) to process obligations.
/// When a cube is blocked, its parent may become blockable too.
fn block_cube(
    sys: &TransitionSystem,
    frames: &mut FrameSequence,
    cube: Cube,
    level: usize,
    conflict_budget: u64,
) -> BlockResult {
    let n = sys.num_state_vars;

    // Work queue: (level, cube, parent_index) — process lowest level first
    let mut queue: Vec<Obligation> = vec![Obligation {
        cube,
        level,
        parent: None,
    }];

    // Process lowest-level obligations first
    while let Some(min_idx) = find_min_level(&queue) {
        let obl_level = queue[min_idx].level;

        if obl_level == 0 {
            // Check if cube is actually reachable from initial states
            // (i.e., is I ∧ cube SAT?)
            if is_initial_reachable(sys, &queue[min_idx].cube, conflict_budget) {
                return BlockResult::Counterexample(
                    reconstruct_trace(&queue, min_idx, n),
                );
            }
            // Not actually reachable from I — remove this obligation
            queue.remove(min_idx);
            continue;
        }

        // Check if cube has a predecessor in F_{level-1}
        let predecessor = check_predecessor(
            sys,
            frames,
            &queue[min_idx].cube,
            obl_level - 1,
            conflict_budget,
        );

        match predecessor {
            Some(pred_cube) => {
                // Has predecessor — add new obligation at lower level
                let parent_idx = min_idx;
                queue.push(Obligation {
                    cube: pred_cube,
                    level: obl_level - 1,
                    parent: Some(parent_idx),
                });
            }
            None => {
                // No predecessor — cube is blocked at this level
                let clause = generalize(
                    sys,
                    frames,
                    &queue[min_idx].cube,
                    obl_level,
                    conflict_budget,
                );
                frames.add_blocked_clause(obl_level, clause);
                // Remove the blocked obligation
                queue.remove(min_idx);
            }
        }
    }

    BlockResult::Blocked
}

/// Find the index of the obligation with the minimum level.
fn find_min_level(queue: &[Obligation]) -> Option<usize> {
    if queue.is_empty() {
        return None;
    }
    let mut min_idx = 0;
    for i in 1..queue.len() {
        if queue[i].level < queue[min_idx].level {
            min_idx = i;
        }
    }
    Some(min_idx)
}

/// Check if a cube overlaps with the initial states.
fn is_initial_reachable(
    sys: &TransitionSystem,
    cube: &Cube,
    conflict_budget: u64,
) -> bool {
    let n = sys.num_state_vars;
    let mut db = ClauseDb::new();

    // Initial-state clauses
    for clause in &sys.initial {
        db.add_clause(clause.lits.clone());
    }

    // Cube as unit clauses
    for &lit in &cube.lits {
        db.add_clause(vec![lit]);
    }

    let (result, _) = solve_watched_budget(db, n, conflict_budget);
    matches!(result, SolveResult::Sat(_))
}

/// Reconstruct a counterexample trace from the obligation chain.
/// Follows parent pointers from the initial-state obligation up to the
/// property-violating state.
fn reconstruct_trace(
    obligations: &[Obligation],
    start_idx: usize,
    num_state_vars: u32,
) -> Vec<Vec<bool>> {
    // The start_idx is the level-0 obligation (closest to initial states).
    // Follow parent pointers to build the trace from initial to bad state.
    let mut trace = Vec::new();
    let mut current = Some(start_idx);

    while let Some(idx) = current {
        trace.push(obligations[idx].cube.to_state_vec(num_state_vars));
        current = obligations[idx].parent;
    }

    // Trace is from initial → bad, but parent pointers go child → parent
    // (bad → initial). So reverse.
    // Actually: parent of level-0 obligation is the level-1 obligation that
    // spawned it. The chain goes: level-0 → level-1 → ... → level-k (bad state).
    // So the trace collected IS from initial towards bad. No reverse needed.
    // Wait — parent points FROM child TO parent (from lower level to higher level).
    // So following parent from level-0 gives: level-0, level-1, ..., level-k.
    // That IS the correct order: initial state first, bad state last.
    trace
}

// ============================================================================
// Generalization
// ============================================================================

/// Generalize a blocked cube: try dropping each literal and check if the
/// reduced clause is still inductive relative to the frame at `level`.
///
/// Returns the generalized clause (negation of the reduced cube).
fn generalize(
    sys: &TransitionSystem,
    frames: &FrameSequence,
    cube: &Cube,
    level: usize,
    conflict_budget: u64,
) -> Vec<Lit> {
    let mut reduced_lits = cube.lits.clone();

    // Try dropping each literal
    let mut i = 0;
    while i < reduced_lits.len() {
        // Try without literal i
        let mut candidate = reduced_lits.clone();
        candidate.remove(i);

        if candidate.is_empty() {
            i += 1;
            continue;
        }

        let candidate_cube = Cube::new(candidate.clone());

        // Check: is the reduced cube still blocked?
        // i.e., is F_{level-1} ∧ T ∧ candidate' UNSAT?
        if level > 0 && check_predecessor(sys, frames, &candidate_cube, level - 1, conflict_budget).is_none()
        {
            // Still blocked — keep the reduction
            reduced_lits = candidate;
            // Don't increment i — the next literal is now at position i
        } else {
            i += 1;
        }
    }

    // Return the negation (clause) of the reduced cube
    Cube::new(reduced_lits).negate()
}

// ============================================================================
// Propagation
// ============================================================================

/// Propagate clauses forward through the frame sequence.
/// For each clause in Fᵢ, check if it's inductive relative to Fᵢ.
/// If so, add it to Fᵢ₊₁.
///
/// Returns Some(i) if convergence detected (Fᵢ = Fᵢ₊₁), None otherwise.
fn propagate_clauses(
    sys: &TransitionSystem,
    frames: &mut FrameSequence,
    conflict_budget: u64,
) -> Option<usize> {
    let n = sys.num_state_vars;
    let frontier = frames.frontier();

    for level in 1..frontier {
        // Collect clauses to propagate (can't borrow frames mutably while iterating)
        let clauses_to_check: Vec<Vec<Lit>> = frames.frame(level).clauses().to_vec();

        for clause in &clauses_to_check {
            // Check: is the clause inductive relative to Fₗₑᵥₑₗ?
            // i.e., is Fₗₑᵥₑₗ ∧ ¬clause ∧ T ∧ clause' UNSAT?
            // Equivalently: does clause hold in all successors of Fₗₑᵥₑₗ ∧ clause?
            if is_clause_inductive(sys, frames, clause, level, n, conflict_budget) {
                // Already in Fₗₑᵥₑₗ₊₁? Check to avoid duplicates
                let next = frames.frame(level + 1);
                let already_present = next.clauses().iter().any(|c| {
                    let mut a: Vec<u32> = c.iter().map(|l| l.code()).collect();
                    let mut b: Vec<u32> = clause.iter().map(|l| l.code()).collect();
                    a.sort();
                    b.sort();
                    a == b
                });
                if !already_present {
                    let clause_copy = clause.clone();
                    frames.frame_mut(level + 1).add_clause(clause_copy);
                }
            }
        }
    }

    frames.check_convergence()
}

/// Check if a clause is inductive relative to a frame.
/// Tests: Fₗₑᵥₑₗ ∧ clause ∧ T → clause' (in the next state).
/// Encoded as: Fₗₑᵥₑₗ ∧ clause ∧ T ∧ ¬clause' — if UNSAT, clause is inductive.
fn is_clause_inductive(
    sys: &TransitionSystem,
    frames: &FrameSequence,
    clause: &[Lit],
    level: usize,
    n: u32,
    conflict_budget: u64,
) -> bool {
    let total_vars = 2 * n;
    let mut db = ClauseDb::new();

    // Frame clauses at level (current-state)
    add_frame_clauses(&mut db, frames.frame(level), 0);

    // The clause itself must hold (current-state)
    db.add_clause(clause.to_vec());

    // Transition relation
    for tc in &sys.transition {
        db.add_clause(tc.lits.clone());
    }

    // ¬clause' (negated clause over next-state)
    // clause = (l₁ ∨ l₂ ∨ ... ∨ lₘ)
    // ¬clause = (¬l₁ ∧ ¬l₂ ∧ ... ∧ ¬lₘ) — each as a unit clause, shifted to next-state
    for &lit in clause {
        let shifted = shift_lit(lit.complement(), n);
        db.add_clause(vec![shifted]);
    }

    let (result, _) = solve_watched_budget(db, total_vars, conflict_budget);
    matches!(result, SolveResult::Unsat)
}

// ============================================================================
// SAT encoding helpers
// ============================================================================

/// Add all clauses from a frame to the clause database, shifting variables by `offset`.
fn add_frame_clauses(db: &mut ClauseDb, frame: &Frame, offset: u32) {
    for clause in frame.clauses() {
        let shifted: Vec<Lit> = clause.iter().map(|&l| shift_lit(l, offset)).collect();
        db.add_clause(shifted);
    }
}

/// Add ¬P(s) (negated property) using Tseitin encoding.
/// `prop_offset`: variable offset for property literals (0 for current-state, n for next-state).
/// `tseitin_base`: first Tseitin variable index.
fn add_negated_property(db: &mut ClauseDb, sys: &TransitionSystem, prop_offset: u32, tseitin_base: u32) {
    let num_tseitin = sys.property.len() as u32;

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
            db.add_clause(vec![Lit::neg(t_var), shifted.complement()]);
        }
    }
}
