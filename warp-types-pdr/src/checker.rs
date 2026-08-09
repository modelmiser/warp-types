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
use crate::phase::*;
use crate::session::{self, PdrSession};

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
        /// The inductive invariant itself: the converged frame's blocking
        /// clauses. Each clause is a disjunction of literals over state
        /// variables `[0, num_state_vars)`; their conjunction over-approximates
        /// every reachable state and is closed under the transition relation.
        /// Empty means the safety property was already inductive with no
        /// strengthening (P implies its own closure).
        invariant: Vec<Vec<Lit>>,
    },
    /// Counterexample found — concrete state trace to a bad state.
    CounterexampleFound {
        /// Depth of the counterexample (number of transitions).
        depth: u32,
        /// State trace: `trace[0]` is the initial state, `trace[depth]` violates the property.
        trace: Vec<Vec<bool>>,
    },
    /// Budget exhausted (frame budget, or a SAT query's conflict budget)
    /// without conclusive result.
    Exhausted {
        /// Number of frames explored.
        frames_explored: usize,
    },
}

/// Tri-state outcome of a budgeted SAT query.
///
/// `Unknown` means the conflict budget ran out before a verdict. It must be
/// kept distinct from `Unsat`: every blocking step in PDR uses UNSAT results
/// as proofs, and treating budget exhaustion as a proof would make an
/// under-budgeted run return an unsound `Safe` verdict.
enum Query<T> {
    /// Satisfiable, with witness.
    Sat(T),
    /// Proven unsatisfiable.
    Unsat,
    /// Conflict budget exhausted — neither proven.
    Unknown,
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
pub fn check(sys: &TransitionSystem, max_frames: u32, conflict_budget: u64) -> PdrResult {
    let n = sys.num_state_vars;

    session::with_session(|init: PdrSession<'_, Init>| {
        let modeled = init.build_model();

        // ── Step 0: Vacuously true property ──
        // P is a conjunction of clauses; the empty conjunction is `true`,
        // so ¬P is unsatisfiable and no state is bad. (The Tseitin encoding
        // of ¬P would otherwise emit nothing, wrongly turning I alone into
        // a depth-0 "violation".)
        if sys.property.is_empty() {
            let _safe = modeled.check_safe();
            return PdrResult::Safe {
                invariant_frame: 0,
                invariant: Vec::new(),
            };
        }

        // ── Step 1: Initiation check ──
        // Is I(s) ∧ ¬P(s) satisfiable?
        match check_initiation(sys, conflict_budget) {
            Query::Sat(assignment) => {
                let trace = vec![assignment[..n as usize].to_vec()];
                let _cex = modeled.check_counterexample();
                return PdrResult::CounterexampleFound { depth: 0, trace };
            }
            Query::Unsat => {}
            Query::Unknown => {
                let _exhausted = modeled.check_exhausted();
                return PdrResult::Exhausted { frames_explored: 0 };
            }
        }

        // ── Step 2: Initialize frame sequence ──
        let mut frames = FrameSequence::new();

        // F₀ = initial-state clauses
        let init_clauses: Vec<Vec<Lit>> = sys.initial.iter().map(|c| c.lits.clone()).collect();
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
                    Query::Unsat => break, // No more CTIs — frontier is clean
                    Query::Unknown => {
                        // Conflict budget ran out — a clean frontier was NOT
                        // proven, so neither propagation nor convergence may run.
                        let _exhausted = modeled.check_exhausted();
                        return PdrResult::Exhausted {
                            frames_explored: frames.len(),
                        };
                    }
                    Query::Sat(cube) => {
                        match block_cube(sys, &mut frames, cube, k, conflict_budget) {
                            BlockResult::Blocked => continue,
                            BlockResult::Counterexample(trace) => {
                                let _cex = modeled.check_counterexample();
                                return PdrResult::CounterexampleFound {
                                    depth: trace.len() as u32 - 1,
                                    trace,
                                };
                            }
                            BlockResult::BudgetExhausted => {
                                let _exhausted = modeled.check_exhausted();
                                return PdrResult::Exhausted {
                                    frames_explored: frames.len(),
                                };
                            }
                        }
                    }
                }
            }

            // PROPAGATE: push clauses forward, check convergence
            if let Some(inv_frame) = propagate_clauses(sys, &mut frames, conflict_budget) {
                let _safe = modeled.check_safe();
                let invariant = frames.frame(inv_frame).clauses().to_vec();
                return PdrResult::Safe {
                    invariant_frame: inv_frame,
                    invariant,
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
    /// A SAT query exhausted its conflict budget — no sound verdict exists
    /// for this cube, so the whole run must report `Exhausted`.
    BudgetExhausted,
}

// ============================================================================
// SAT query: initiation
// ============================================================================

/// Check I(s) ∧ ¬P(s). `Sat(assignment)` means an initial violation.
fn check_initiation(sys: &TransitionSystem, conflict_budget: u64) -> Query<Vec<bool>> {
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
        SolveResult::Sat(assign) => Query::Sat(assign),
        SolveResult::Unsat => Query::Unsat,
        SolveResult::Unknown => Query::Unknown,
    }
}

// ============================================================================
// SAT query: find CTI
// ============================================================================

/// Find a counterexample-to-induction at frame `level`.
/// Checks: Fₖ ∧ ¬P — is there a bad state in the frame?
/// `Sat` carries the bad-state cube; `Unsat` proves the frontier clean.
fn find_cti(
    sys: &TransitionSystem,
    frames: &FrameSequence,
    level: usize,
    conflict_budget: u64,
) -> Query<Cube> {
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
        SolveResult::Sat(assign) => Query::Sat(Cube::from_assignment(&assign, n)),
        SolveResult::Unsat => Query::Unsat,
        SolveResult::Unknown => Query::Unknown,
    }
}

// ============================================================================
// SAT query: consecution (relative induction)
// ============================================================================

/// Check consecution: is Fₖ ∧ T ∧ cube' satisfiable?
/// If `Sat`, the cube has a predecessor in Fₖ — carries the predecessor.
/// If `Unsat`, the cube is blocked at this level.
fn check_predecessor(
    sys: &TransitionSystem,
    frames: &FrameSequence,
    cube: &Cube,
    level: usize,
    conflict_budget: u64,
) -> Query<Cube> {
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
        SolveResult::Sat(assign) => Query::Sat(Cube::from_assignment(&assign, n)),
        SolveResult::Unsat => Query::Unsat,
        SolveResult::Unknown => Query::Unknown,
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
            match intersects_initial(sys, &queue[min_idx].cube, conflict_budget) {
                Query::Sat(()) => {
                    return BlockResult::Counterexample(reconstruct_trace(&queue, min_idx, n));
                }
                Query::Unsat => {
                    // Not actually reachable from I — remove this obligation
                    queue.remove(min_idx);
                    continue;
                }
                Query::Unknown => return BlockResult::BudgetExhausted,
            }
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
            Query::Sat(pred_cube) => {
                // Has predecessor — add new obligation at lower level
                let parent_idx = min_idx;
                queue.push(Obligation {
                    cube: pred_cube,
                    level: obl_level - 1,
                    parent: Some(parent_idx),
                });
            }
            Query::Unknown => return BlockResult::BudgetExhausted,
            Query::Unsat => {
                // No predecessor — the consecution side holds. Before blocking,
                // check the initiation side of relative induction: a cube that
                // intersects the initial states must never be blocked (its
                // blocking clause would cut an initial state out of the frame,
                // breaking I ⊆ Fᵢ and cascading to unsound Safe verdicts).
                //
                // Obligation cubes are complete states, so intersection means
                // this state IS initial — and the obligation chain reaches a
                // property violation, so it is a real counterexample.
                match intersects_initial(sys, &queue[min_idx].cube, conflict_budget) {
                    Query::Sat(()) => {
                        return BlockResult::Counterexample(reconstruct_trace(&queue, min_idx, n));
                    }
                    Query::Unknown => return BlockResult::BudgetExhausted,
                    Query::Unsat => {
                        // Initiation holds — safe to block and generalize.
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

/// Check if a cube overlaps with the initial states: is I ∧ cube SAT?
///
/// This is both the level-0 reachability test and the initiation side of
/// relative induction (a cube may only be blocked if `Unsat` here).
fn intersects_initial(sys: &TransitionSystem, cube: &Cube, conflict_budget: u64) -> Query<()> {
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
    match result {
        SolveResult::Sat(_) => Query::Sat(()),
        SolveResult::Unsat => Query::Unsat,
        SolveResult::Unknown => Query::Unknown,
    }
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

        // A literal drop widens the cube, so it is kept only when BOTH sides
        // of relative induction still hold for the widened cube:
        //
        // - Initiation: I ∧ candidate is UNSAT. Without this, a widened cube
        //   can swallow an initial state and its blocking clause cuts that
        //   state out of the frame — breaking I ⊆ Fᵢ (unsound).
        // - Consecution: F_{level-1} ∧ T ∧ candidate' is UNSAT.
        //
        // `Unknown` (conflict budget exhausted) rejects the drop:
        // generalization is an optimization and must stay conservative.
        let initiation_ok = matches!(
            intersects_initial(sys, &candidate_cube, conflict_budget),
            Query::Unsat
        );
        let consecution_ok = level > 0
            && matches!(
                check_predecessor(sys, frames, &candidate_cube, level - 1, conflict_budget),
                Query::Unsat
            );
        if initiation_ok && consecution_ok {
            // Still blocked and still excludes no initial state — keep it.
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
fn add_negated_property(
    db: &mut ClauseDb,
    sys: &TransitionSystem,
    prop_offset: u32,
    tseitin_base: u32,
) {
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluate a clause under a full state assignment.
    fn eval_clause(clause: &[Lit], state: &[bool]) -> bool {
        clause.iter().any(|l| {
            let v = l.var() as usize;
            if l.is_negated() {
                !state[v]
            } else {
                state[v]
            }
        })
    }

    // ------------------------------------------------------------------
    // Finding A: missing initiation check in blocking/generalization
    // ------------------------------------------------------------------

    /// Genuinely-safe system where the unguarded generalization used to
    /// produce an "invariant" that excludes the initial state.
    ///
    /// I = {00}; T: every state → 11 (s₀' = 1, s₁' = 1); P = ¬(s₁ ∧ ¬s₀),
    /// so the only bad state is 01 (s₀=0, s₁=1) — unreachable.
    ///
    /// Without the initiation check, generalizing the CTI [¬s₀, s₁] first
    /// fails to drop ¬s₀ (11 is reachable, so s₁'=1 has a predecessor) and
    /// then succeeds dropping s₁, keeping [¬s₀] — whose blocking clause
    /// (s₀) excludes the initial state 00 from F₁, breaking I ⊆ Fᵢ.
    #[test]
    fn blocking_never_excludes_initial_states() {
        let mut sys = TransitionSystem::new(2);
        sys.add_initial(vec![Lit::neg(0)]);
        sys.add_initial(vec![Lit::neg(1)]);
        // s₀' = 1, s₁' = 1 from every state
        sys.add_transition(vec![Lit::pos(2)]);
        sys.add_transition(vec![Lit::pos(3)]);
        // P: (¬s₁ ∨ s₀)
        sys.add_property(vec![Lit::neg(1), Lit::pos(0)]);

        match check(&sys, 20, 0) {
            PdrResult::Safe { invariant, .. } => {
                let init_state = [false, false];
                for clause in &invariant {
                    assert!(
                        eval_clause(clause, &init_state),
                        "invariant clause {:?} excludes the initial state 00 — \
                         initiation (I ⊆ Fᵢ) violated",
                        clause
                    );
                }
            }
            other => panic!("expected Safe, got {:?}", other),
        }
    }

    /// Reviewer witness for finding A: I = {00}, chain 00 → 01 → 11-free
    /// bad state (s₀=0, s₁=1) at depth 2, property ¬(s₁ ∧ ¬s₀).
    ///
    /// T: s₀' ↔ (¬s₀ ∧ ¬s₁), s₁' ↔ (s₀ ∨ s₁); the bad state self-loops.
    #[test]
    fn initiation_witness_finds_counterexample() {
        let mut sys = TransitionSystem::new(2);
        sys.add_initial(vec![Lit::neg(0)]);
        sys.add_initial(vec![Lit::neg(1)]);
        // s₀' ↔ (¬s₀ ∧ ¬s₁)
        sys.add_transition(vec![Lit::neg(2), Lit::neg(0)]);
        sys.add_transition(vec![Lit::neg(2), Lit::neg(1)]);
        sys.add_transition(vec![Lit::pos(0), Lit::pos(1), Lit::pos(2)]);
        // s₁' ↔ (s₀ ∨ s₁)
        sys.add_transition(vec![Lit::neg(3), Lit::pos(0), Lit::pos(1)]);
        sys.add_transition(vec![Lit::neg(0), Lit::pos(3)]);
        sys.add_transition(vec![Lit::neg(1), Lit::pos(3)]);
        // P: (¬s₁ ∨ s₀)
        sys.add_property(vec![Lit::neg(1), Lit::pos(0)]);

        match check(&sys, 20, 0) {
            PdrResult::CounterexampleFound { depth, trace } => {
                assert_eq!(depth, 2, "bad state is reachable in exactly 2 steps");
                assert_eq!(trace.len(), 3);
                // Starts in the initial state 00
                assert!(!trace[0][0] && !trace[0][1]);
                // Ends in the bad state (s₀=0, s₁=1)
                assert!(!trace[2][0] && trace[2][1]);
            }
            other => panic!("expected CounterexampleFound at depth 2, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Finding B: conflict-budget exhaustion must never become a Safe proof
    // ------------------------------------------------------------------

    /// Build an m-bit binary counter with property ¬(s₀ ∧ ... ∧ s_{m-1}).
    /// The all-ones state is reached at depth 2^m - 1, so the system is
    /// genuinely unsafe.
    fn counter_system(bits: u32) -> TransitionSystem {
        let mut sys = TransitionSystem::new(bits);
        for v in 0..bits {
            sys.add_initial(vec![Lit::neg(v)]);
        }
        // sᵢ' = sᵢ ⊕ (s₀ ∧ ... ∧ sᵢ₋₁)
        for i in 0..bits {
            let cur = i;
            let next = bits + i;
            // Case: full prefix true → sᵢ' = ¬sᵢ
            let mut flip_a: Vec<Lit> = (0..i).map(Lit::neg).collect();
            flip_a.push(Lit::neg(cur));
            flip_a.push(Lit::neg(next));
            sys.add_transition(flip_a);
            let mut flip_b: Vec<Lit> = (0..i).map(Lit::neg).collect();
            flip_b.push(Lit::pos(cur));
            flip_b.push(Lit::pos(next));
            sys.add_transition(flip_b);
            // Case: some prefix bit false → sᵢ' = sᵢ
            for j in 0..i {
                sys.add_transition(vec![Lit::pos(j), Lit::neg(cur), Lit::pos(next)]);
                sys.add_transition(vec![Lit::pos(j), Lit::pos(cur), Lit::neg(next)]);
            }
        }
        // P: ¬(s₀ ∧ ... ∧ s_{m-1})
        sys.add_property((0..bits).map(Lit::neg).collect());
        sys
    }

    /// With a tiny conflict budget, SAT queries return Unknown. Unknown must
    /// surface as Exhausted (or a genuine counterexample if the solver stays
    /// within budget) — never as a Safe verdict on an unsafe system.
    #[test]
    fn tiny_budget_never_reports_safe() {
        let sys = counter_system(4);
        for budget in [1u64, 2, 3, 5, 8] {
            match check(&sys, 40, budget) {
                PdrResult::Safe { .. } => panic!(
                    "conflict budget {} exhausted mid-proof but verdict is Safe \
                     on an unsafe 4-bit counter — Unknown treated as UNSAT",
                    budget
                ),
                PdrResult::CounterexampleFound { trace, .. } => {
                    // If a cex is claimed it must be genuine: ends all-ones.
                    let last = trace.last().unwrap();
                    assert!(
                        last.iter().all(|&b| b),
                        "claimed cex does not end in the bad state"
                    );
                }
                PdrResult::Exhausted { .. } => {}
            }
        }
    }

    /// Safe system whose initiation query I ∧ ¬P is UNSAT but NOT refutable
    /// by unit propagation alone: I is an even-parity constraint over three
    /// variables (x₀ ⊕ x₁ ⊕ x₂ = 0) and P is the same parity constraint,
    /// with T = identity. The CNF has no unit clauses, so the solver must
    /// make decisions and take at least one conflict to prove UNSAT —
    /// guaranteeing that conflict_budget = 1 yields `Unknown`.
    fn parity_system() -> TransitionSystem {
        let mut sys = TransitionSystem::new(3);
        // Even parity: forbid the four odd assignments (100, 010, 001, 111).
        let parity: [Vec<Lit>; 4] = [
            vec![Lit::neg(0), Lit::pos(1), Lit::pos(2)],
            vec![Lit::pos(0), Lit::neg(1), Lit::pos(2)],
            vec![Lit::pos(0), Lit::pos(1), Lit::neg(2)],
            vec![Lit::neg(0), Lit::neg(1), Lit::neg(2)],
        ];
        for c in &parity {
            sys.add_initial(c.clone());
            sys.add_property(c.clone());
        }
        // T: identity
        for v in 0..3 {
            sys.add_transition(vec![Lit::neg(v), Lit::pos(v + 3)]);
            sys.add_transition(vec![Lit::pos(v), Lit::neg(v + 3)]);
        }
        sys
    }

    /// A conflict budget of 1 makes the very first initiation query return
    /// Unknown (its UNSAT proof needs at least one conflict). That must
    /// surface as Exhausted with zero frames explored — never as a Safe
    /// verdict claimed without a proof.
    #[test]
    fn tiny_budget_initiation_unknown_is_exhausted() {
        match check(&parity_system(), 40, 1) {
            PdrResult::Exhausted { frames_explored } => assert_eq!(
                frames_explored, 0,
                "budget died in the initiation query, before any frame"
            ),
            other => panic!(
                "conflict budget 1 cannot prove I ∧ ¬P unsat; expected Exhausted, got {:?}",
                other
            ),
        }
    }

    /// Sanity: with an unlimited budget the same parity system is proven Safe.
    #[test]
    fn parity_system_unlimited_budget_is_safe() {
        match check(&parity_system(), 40, 0) {
            PdrResult::Safe { .. } => {}
            other => panic!("expected Safe, got {:?}", other),
        }
    }

    /// Sanity: with an unlimited budget the same counter is refuted with the
    /// full-depth counterexample.
    #[test]
    fn counter_unlimited_budget_finds_cex() {
        let sys = counter_system(3);
        match check(&sys, 40, 0) {
            PdrResult::CounterexampleFound { depth, trace } => {
                assert_eq!(depth, 7, "3-bit counter reaches 111 at depth 7");
                assert!(trace.last().unwrap().iter().all(|&b| b));
            }
            other => panic!("expected CounterexampleFound at depth 7, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Finding C: empty property is vacuously true ⇒ Safe
    // ------------------------------------------------------------------

    #[test]
    fn empty_property_is_safe() {
        // No property clauses: P is the empty conjunction = true. Regression:
        // check_initiation used to solve I alone (SAT) and report a false
        // depth-0 counterexample.
        let mut sys = TransitionSystem::new(1);
        sys.add_initial(vec![Lit::neg(0)]);
        // s₀' = ¬s₀ (toggle)
        sys.add_transition(vec![Lit::pos(0), Lit::pos(1)]);
        sys.add_transition(vec![Lit::neg(0), Lit::neg(1)]);

        match check(&sys, 20, 0) {
            PdrResult::Safe { .. } => {}
            other => panic!(
                "empty (vacuously true) property must be Safe, got {:?}",
                other
            ),
        }
    }
}
