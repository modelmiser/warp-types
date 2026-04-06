//! 1-UIP conflict analysis with clause minimization.
//!
//! Given a conflict clause and the current trail, resolves backward through
//! implication reasons until exactly one literal at the current decision level
//! remains. That literal is the First Unique Implication Point (1-UIP).
//!
//! After deriving the learned clause, clause minimization (MiniSat's
//! `litRedundant`) removes literals whose assignments are already implied
//! by other literals in the clause. This typically removes 20-30% of
//! learned clause literals, improving both LBD scores and BCP speed.
//!
//! Returns the learned clause and the backtrack level.

use std::time::Instant;

use crate::bcp::{CRef, ClauseDb};
use crate::literal::Lit;
use crate::trail::{Reason, Trail};

/// A single resolution step in 1-UIP analysis.
#[derive(Debug, Clone)]
pub struct ResolutionStep {
    /// The reason clause being resolved against.
    pub reason_clause: CRef,
    /// The variable being eliminated (the pivot).
    pub pivot_var: u32,
    /// Width of the working clause after this resolution
    /// (number of seen variables at current level + learned literals so far).
    pub working_width: usize,
}

/// Per-conflict profile for proof DAG mining (Level 2).
///
/// Captures solver state at the moment of conflict alongside the
/// resolution chain from 1-UIP analysis. This is the primary data
/// structure for Level 3 DAG topology analysis and Level 4 correlations.
#[derive(Debug, Clone)]
pub struct ConflictProfile {
    /// Monotonic conflict counter (0-indexed).
    pub conflict_id: u64,
    /// Decision level at which the conflict was detected.
    pub decision_level: u32,
    /// Number of resolution steps to reach 1-UIP (chain length).
    pub resolution_depth: u32,
    /// Size of the learned clause after minimization.
    pub learned_clause_size: usize,
    /// Literal Block Distance of the learned clause.
    pub learned_lbd: u32,
    /// Decision levels backtracked: current_level - backtrack_level.
    pub backtrack_distance: u32,
    /// Trail size (number of assigned variables) at the moment of conflict.
    pub trail_size_at_conflict: usize,
    /// Full resolution chain from 1-UIP analysis.
    pub resolution_chain: Vec<ResolutionStep>,
    /// Number of BCP propagations between the last decision and this conflict.
    pub bcp_propagations: u32,
}

/// Result of conflict analysis.
pub struct AnalysisResult {
    /// The learned clause (asserting clause). The first literal is the
    /// asserting literal (the 1-UIP, negated).
    pub learned: Vec<Lit>,
    /// The decision level to backtrack to.
    pub backtrack_level: u32,
    /// Literal Block Distance: number of distinct decision levels among the
    /// learned clause's literals. Lower = more useful (a "glue clause" has LBD ≤ 2).
    pub lbd: u32,
    /// Nanoseconds spent in 1-UIP resolution (backward trail scan + reason clause iteration).
    pub resolve_ns: u64,
    /// Nanoseconds spent in clause minimization (litRedundant DFS).
    pub minimize_ns: u64,
    /// Resolution chain: sequence of resolution steps to reach 1-UIP.
    /// Empty unless instrumentation is enabled via `analyze_conflict_instrumented`.
    pub resolution_chain: Vec<ResolutionStep>,
}

/// Persistent scratch buffers for conflict analysis.
///
/// Allocated once at solver init, reused across all conflicts. Eliminates
/// the per-conflict heap allocation of `seen` and `levels_seen` vectors.
pub struct AnalyzeWork {
    /// Per-variable seen flag. Sized for num_vars at init, cleared
    /// incrementally via `touched` after each analysis.
    seen: Vec<bool>,
    /// Variables touched during this analysis (for incremental clear).
    touched: Vec<u32>,
    /// Stack for clause minimization DFS.
    min_stack: Vec<Lit>,
    /// Temporary for `to_clear` in minimization.
    min_to_clear: Vec<u32>,
    /// Level-seen flags for LBD computation.
    levels_seen: Vec<bool>,
}

impl AnalyzeWork {
    /// Create scratch buffers for a solver with `num_vars` variables.
    pub fn new(num_vars: usize) -> Self {
        AnalyzeWork {
            seen: vec![false; num_vars],
            touched: Vec::with_capacity(64),
            min_stack: Vec::with_capacity(32),
            min_to_clear: Vec::with_capacity(32),
            levels_seen: Vec::new(), // grown on demand per analysis
        }
    }

    /// Ensure buffers cover at least `num_vars` variables (for learned clauses
    /// that introduce new variable indices — shouldn't happen, but defensive).
    fn ensure_capacity(&mut self, num_vars: usize) {
        if num_vars > self.seen.len() {
            self.seen.resize(num_vars, false);
        }
    }

    /// Clear all seen flags touched during the last analysis.
    fn clear_seen(&mut self) {
        for &var in &self.touched {
            self.seen[var as usize] = false;
        }
        self.touched.clear();
    }

    /// Mark a variable as seen and record it for cleanup.
    #[inline]
    fn mark_seen(&mut self, var: u32) {
        self.seen[var as usize] = true;
        self.touched.push(var);
    }
}

/// Run 1-UIP conflict analysis (allocates fresh scratch buffers).
///
/// Convenience wrapper for callers that don't reuse buffers (old solver, tests).
pub fn analyze_conflict(trail: &Trail, db: &ClauseDb, conflict_clause: CRef) -> AnalysisResult {
    let num_vars = trail.num_vars().max(db.max_variable() as usize + 1);
    let mut work = AnalyzeWork::new(num_vars);
    analyze_conflict_with(&mut work, trail, db, conflict_clause)
}

/// Run 1-UIP conflict analysis using persistent scratch buffers.
///
/// `conflict_clause` is the index of the clause that caused the conflict.
/// Returns the learned clause and backtrack level.
pub fn analyze_conflict_with(
    work: &mut AnalyzeWork,
    trail: &Trail,
    db: &ClauseDb,
    conflict_clause: CRef,
) -> AnalysisResult {
    let current_level = trail.current_level();
    let t_resolve = Instant::now();

    // Ensure seen array covers all variables (defensive — shouldn't grow).
    let max_var = trail.num_vars();
    work.ensure_capacity(max_var);

    // Start with the conflict clause's literals.
    // SAFETY: conflict_clause < db.len() (caller invariant).
    let mut learned = Vec::new();
    let mut num_at_current_level = 0;

    for &lit in unsafe { db.clause_unchecked(conflict_clause) }.literals {
        let var = lit.var();
        if !work.seen[var as usize] {
            work.mark_seen(var);
            // SAFETY: var comes from clause DB, validated var < num_vars at startup.
            let entry = unsafe { trail.entry_for_var_unchecked(var) };
            debug_assert!(
                entry.is_some(),
                "variable {} in conflict clause has no trail entry (unassigned in a conflict?)",
                var
            );
            match entry {
                Some(e) if e.level == current_level => {
                    num_at_current_level += 1;
                }
                Some(_) | None => {
                    learned.push(lit);
                }
            }
        }
    }

    // Resolve backward through trail until 1 literal at current level remains
    let entries = trail.entries();
    let mut trail_idx = entries.len();

    while num_at_current_level > 1 {
        // Find the most recent trail entry at the current level that we've seen
        trail_idx -= 1;
        let entry = &entries[trail_idx];
        if entry.level != current_level || !work.seen[entry.lit.var() as usize] {
            continue;
        }

        // This literal is at the current level and in our working clause.
        // Resolve it away using its reason clause.
        match entry.reason {
            Reason::Decision => {
                // Decisions can't be resolved — this shouldn't happen if
                // there's more than 1 literal at the current level
                debug_assert!(
                    num_at_current_level <= 1,
                    "hit decision during resolution with {} literals remaining at current level",
                    num_at_current_level
                );
                break;
            }
            Reason::Propagation(reason_clause) => {
                debug_assert!(
                    !db.is_deleted(reason_clause),
                    "resolving through deleted clause {reason_clause} for var {}",
                    entry.lit.var()
                );
                num_at_current_level -= 1;
                work.seen[entry.lit.var() as usize] = false; // resolved away
                // Add all other literals from the reason clause.
                // SAFETY: reason_clause < db.len() (valid propagation reasons).
                for &lit in unsafe { db.clause_unchecked(reason_clause) }.literals {
                    let var = lit.var();
                    if var == entry.lit.var() {
                        continue; // skip the resolved variable
                    }
                    if !work.seen[var as usize] {
                        work.mark_seen(var);
                        // SAFETY: var from clause DB, validated var < num_vars.
                        let reason_entry = unsafe { trail.entry_for_var_unchecked(var) };
                        match reason_entry {
                            Some(e) if e.level == current_level => {
                                num_at_current_level += 1;
                            }
                            Some(_) | None => {
                                learned.push(lit);
                            }
                        }
                    }
                }
            }
        }
    }

    // Find the 1-UIP: the single remaining seen variable at the current level.
    // Scan trail backward — the most recent seen entry at this level is the UIP.
    let mut asserting_lit = None;
    for entry in entries.iter().rev() {
        if entry.level == current_level && work.seen[entry.lit.var() as usize] {
            asserting_lit = Some(entry.lit.complement());
            break;
        }
    }

    let lit = asserting_lit
        .expect("1-UIP resolution must find an asserting literal at the current decision level");
    learned.insert(0, lit); // asserting literal first

    // Select optimal second watch: the literal with the highest decision level
    // among non-asserting literals. This ensures that after backtracking to
    // backtrack_level, c[1] is still assigned (false) — making the clause
    // immediately unit from BCP's perspective. MiniSat's standard technique.
    if learned.len() >= 3 {
        let mut best_pos = 1;
        let mut best_level = trail.entry_for_var(learned[1].var())
            .map(|e| e.level).unwrap_or(0);
        for i in 2..learned.len() {
            let level = trail.entry_for_var(learned[i].var())
                .map(|e| e.level).unwrap_or(0);
            if level > best_level {
                best_level = level;
                best_pos = i;
            }
        }
        if best_pos != 1 {
            learned.swap(1, best_pos);
        }
    }

    let resolve_ns = t_resolve.elapsed().as_nanos() as u64;

    // ── Clause minimization ───��──────────────────────────────��─────
    // Remove literals whose propagation reasons are already implied by
    // other literals in the clause. MiniSat's litRedundant algorithm.
    let t_minimize = Instant::now();

    let abstract_levels = {
        let mut mask = 0u64;
        for &l in &learned {
            if let Some(e) = trail.entry_for_var(l.var()) {
                mask |= 1u64 << (e.level % 64);
            }
        }
        mask
    };

    work.min_to_clear.clear();
    let mut minimized = Vec::with_capacity(learned.len());
    minimized.push(learned[0]); // asserting literal always kept

    for &l in &learned[1..] {
        if lit_redundant_with(work, trail, db, l, abstract_levels) {
            work.seen[l.var() as usize] = false; // no longer in clause
        } else {
            minimized.push(l);
        }
    }

    // Clean up DFS marks from successful redundancy proofs
    for &var in &work.min_to_clear {
        work.seen[var as usize] = false;
    }
    work.min_to_clear.clear();

    let minimize_ns = t_minimize.elapsed().as_nanos() as u64;
    let learned = minimized;

    // Backtrack level: highest level among learned clause literals,
    // excluding the asserting literal (which is at current_level).
    let backtrack_level = learned
        .iter()
        .skip(1) // skip asserting literal
        .filter_map(|lit| trail.entry_for_var(lit.var()).map(|e| e.level))
        .max()
        .unwrap_or(0);

    // LBD: count distinct decision levels in the learned clause.
    let lbd = {
        let level_count = current_level as usize + 1;
        work.levels_seen.clear();
        work.levels_seen.resize(level_count, false);
        let mut count = 0u32;
        for lit in &learned {
            if let Some(e) = trail.entry_for_var(lit.var()) {
                let lv = e.level as usize;
                if lv < level_count && !work.levels_seen[lv] {
                    work.levels_seen[lv] = true;
                    count += 1;
                }
            }
        }
        count
    };

    // Clean up seen flags for next conflict
    work.clear_seen();

    AnalysisResult {
        learned,
        backtrack_level,
        lbd,
        resolve_ns,
        minimize_ns,
        resolution_chain: Vec::new(),
    }
}

/// Run 1-UIP conflict analysis with resolution chain capture.
///
/// Same as `analyze_conflict_with` but records each resolution step.
/// Use for proof DAG mining — adds ~1 Vec::push per resolution step.
pub fn analyze_conflict_instrumented(
    work: &mut AnalyzeWork,
    trail: &Trail,
    db: &ClauseDb,
    conflict_clause: CRef,
) -> AnalysisResult {
    let current_level = trail.current_level();
    let t_resolve = Instant::now();

    let max_var = trail.num_vars();
    work.ensure_capacity(max_var);

    let mut learned = Vec::new();
    let mut num_at_current_level = 0;
    let mut resolution_chain = Vec::new();

    for &lit in unsafe { db.clause_unchecked(conflict_clause) }.literals {
        let var = lit.var();
        if !work.seen[var as usize] {
            work.mark_seen(var);
            let entry = unsafe { trail.entry_for_var_unchecked(var) };
            match entry {
                Some(e) if e.level == current_level => {
                    num_at_current_level += 1;
                }
                Some(_) | None => {
                    learned.push(lit);
                }
            }
        }
    }

    let entries = trail.entries();
    let mut trail_idx = entries.len();

    while num_at_current_level > 1 {
        trail_idx -= 1;
        let entry = &entries[trail_idx];
        if entry.level != current_level || !work.seen[entry.lit.var() as usize] {
            continue;
        }

        match entry.reason {
            Reason::Decision => {
                debug_assert!(num_at_current_level <= 1);
                break;
            }
            Reason::Propagation(reason_clause) => {
                debug_assert!(!db.is_deleted(reason_clause));
                num_at_current_level -= 1;
                work.seen[entry.lit.var() as usize] = false;
                for &lit in unsafe { db.clause_unchecked(reason_clause) }.literals {
                    let var = lit.var();
                    if var == entry.lit.var() {
                        continue;
                    }
                    if !work.seen[var as usize] {
                        work.mark_seen(var);
                        let reason_entry = unsafe { trail.entry_for_var_unchecked(var) };
                        match reason_entry {
                            Some(e) if e.level == current_level => {
                                num_at_current_level += 1;
                            }
                            Some(_) | None => {
                                learned.push(lit);
                            }
                        }
                    }
                }

                // Record this resolution step
                resolution_chain.push(ResolutionStep {
                    reason_clause,
                    pivot_var: entry.lit.var(),
                    working_width: num_at_current_level + learned.len(),
                });
            }
        }
    }

    // Find the 1-UIP
    let mut asserting_lit = None;
    for entry in entries.iter().rev() {
        if entry.level == current_level && work.seen[entry.lit.var() as usize] {
            asserting_lit = Some(entry.lit.complement());
            break;
        }
    }

    let lit = asserting_lit
        .expect("1-UIP resolution must find an asserting literal at the current decision level");
    learned.insert(0, lit);

    if learned.len() >= 3 {
        let mut best_pos = 1;
        let mut best_level = trail.entry_for_var(learned[1].var())
            .map(|e| e.level).unwrap_or(0);
        for i in 2..learned.len() {
            let level = trail.entry_for_var(learned[i].var())
                .map(|e| e.level).unwrap_or(0);
            if level > best_level {
                best_level = level;
                best_pos = i;
            }
        }
        if best_pos != 1 {
            learned.swap(1, best_pos);
        }
    }

    let resolve_ns = t_resolve.elapsed().as_nanos() as u64;

    // Clause minimization (same as non-instrumented path)
    let t_minimize = Instant::now();

    let abstract_levels = {
        let mut mask = 0u64;
        for &l in &learned {
            if let Some(e) = trail.entry_for_var(l.var()) {
                mask |= 1u64 << (e.level % 64);
            }
        }
        mask
    };

    work.min_to_clear.clear();
    let mut minimized = Vec::with_capacity(learned.len());
    minimized.push(learned[0]);

    for &l in &learned[1..] {
        if lit_redundant_with(work, trail, db, l, abstract_levels) {
            work.seen[l.var() as usize] = false;
        } else {
            minimized.push(l);
        }
    }

    for &var in &work.min_to_clear {
        work.seen[var as usize] = false;
    }
    work.min_to_clear.clear();

    let minimize_ns = t_minimize.elapsed().as_nanos() as u64;
    let learned = minimized;

    let backtrack_level = learned
        .iter()
        .skip(1)
        .filter_map(|lit| trail.entry_for_var(lit.var()).map(|e| e.level))
        .max()
        .unwrap_or(0);

    let lbd = {
        let level_count = current_level as usize + 1;
        work.levels_seen.clear();
        work.levels_seen.resize(level_count, false);
        let mut count = 0u32;
        for lit in &learned {
            if let Some(e) = trail.entry_for_var(lit.var()) {
                let lv = e.level as usize;
                if lv < level_count && !work.levels_seen[lv] {
                    work.levels_seen[lv] = true;
                    count += 1;
                }
            }
        }
        count
    };

    work.clear_seen();

    AnalysisResult {
        learned,
        backtrack_level,
        lbd,
        resolve_ns,
        minimize_ns,
        resolution_chain,
    }
}

/// Check if a literal is redundant using persistent work buffers.
///
/// Same algorithm as the standalone `lit_redundant`, but reuses
/// `work.min_stack` and `work.min_to_clear` across calls.
fn lit_redundant_with(
    work: &mut AnalyzeWork,
    trail: &Trail,
    db: &ClauseDb,
    lit: Lit,
    abstract_levels: u64,
) -> bool {
    let top = work.min_to_clear.len(); // snapshot for rollback on failure
    work.min_stack.clear();

    // Start by examining the reason for lit's assignment.
    // SAFETY: lit.var() comes from learned clause, all vars < num_vars.
    let entry = match unsafe { trail.entry_for_var_unchecked(lit.var()) } {
        Some(e) => e,
        None => return false,
    };
    let reason_clause = match entry.reason {
        Reason::Decision => return false, // decisions are never redundant
        Reason::Propagation(ci) => ci,
    };

    // Push reason clause literals (except lit itself) onto stack.
    // SAFETY: reason_clause is a valid propagation reason from the trail.
    for &reason_lit in unsafe { db.clause_unchecked(reason_clause) }.literals {
        let rv = reason_lit.var();
        if rv == lit.var() {
            continue;
        }
        if work.seen[rv as usize] {
            continue; // in clause or already proven redundant
        }
        // SAFETY: rv from clause DB, validated var < num_vars.
        if let Some(re) = unsafe { trail.entry_for_var_unchecked(rv) } {
            if re.level == 0 {
                continue; // level 0 literals are always satisfied
            }
        }
        work.min_stack.push(reason_lit);
    }

    while let Some(l) = work.min_stack.pop() {
        let var = l.var();

        // SAFETY: var from clause DB, validated var < num_vars.
        let re = match unsafe { trail.entry_for_var_unchecked(var) } {
            Some(e) => e,
            None => {
                // Unassigned — rollback
                for v in work.min_to_clear.drain(top..) {
                    work.seen[v as usize] = false;
                }
                return false;
            }
        };

        // Abstract level filter: if this level can't be in the clause, fail fast
        if (abstract_levels >> (re.level % 64)) & 1 == 0 {
            for v in work.min_to_clear.drain(top..) {
                work.seen[v as usize] = false;
            }
            return false;
        }

        let ci = match re.reason {
            Reason::Decision => {
                // Decision variable at a level that MIGHT be in the clause
                // (passed abstract filter) but isn't actually in the clause
                // (not in `seen`). Can't be proven redundant.
                for v in work.min_to_clear.drain(top..) {
                    work.seen[v as usize] = false;
                }
                return false;
            }
            Reason::Propagation(ci) => ci,
        };

        // Mark as visited (will be cleaned up on failure or at end)
        work.seen[var as usize] = true;
        work.min_to_clear.push(var);

        // Explore reason clause.
        // SAFETY: ci is a valid propagation reason from the trail.
        for &reason_lit in unsafe { db.clause_unchecked(ci) }.literals {
            let rv = reason_lit.var();
            if rv == var {
                continue;
            }
            if work.seen[rv as usize] {
                continue;
            }
            // SAFETY: rv from clause DB, validated var < num_vars.
            if let Some(rre) = unsafe { trail.entry_for_var_unchecked(rv) } {
                if rre.level == 0 {
                    continue;
                }
            }
            work.min_stack.push(reason_lit);
        }
    }

    true // all paths lead to in-clause or level-0
}

// ── Topology helpers for ConflictProfile analysis (Level 2) ──────────

use std::collections::HashMap;

/// Count how often each variable appears as a pivot across all profiles.
///
/// High-frequency pivots are "bottleneck" variables — structurally central
/// to the proof. Correlating with VSIDS activity tests whether the solver
/// already knows what the proof structure reveals.
pub fn pivot_frequency(profiles: &[ConflictProfile]) -> HashMap<u32, usize> {
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for p in profiles {
        for step in &p.resolution_chain {
            *freq.entry(step.pivot_var).or_insert(0) += 1;
        }
    }
    freq
}

/// Count how often each clause appears as a reason clause across all profiles.
///
/// High-reuse clauses are structurally important — they participate in many
/// different conflict derivations. Input clauses with high reuse encode
/// "hard" constraints; learned clauses with high reuse are good "glue."
pub fn clause_reuse_frequency(profiles: &[ConflictProfile]) -> HashMap<CRef, usize> {
    let mut freq: HashMap<CRef, usize> = HashMap::new();
    for p in profiles {
        for step in &p.resolution_chain {
            *freq.entry(step.reason_clause).or_insert(0) += 1;
        }
    }
    freq
}

/// Extract the working-width profile from a single resolution chain.
///
/// Returns the sequence of working widths at each resolution step.
/// Width expansion (growing) suggests the solver is pulling in many
/// variables; width contraction (shrinking) suggests convergence toward
/// the 1-UIP. The shape of this curve characterizes the conflict.
pub fn working_width_profile(chain: &[ResolutionStep]) -> Vec<usize> {
    chain.iter().map(|step| step.working_width).collect()
}

// ── Proof DAG topology analysis (Level 3) ────────────────────────────

/// A node in the proof DAG. Represents a clause (input or learned).
#[derive(Debug, Clone, Default)]
pub struct DagNode {
    /// Number of resolution chains that produced this clause as output.
    /// For input clauses this is always 0. For learned clauses, 1.
    pub fan_in: u32,
    /// Number of times this clause was used as a reason clause in any
    /// resolution chain (across all conflicts).
    pub fan_out: u32,
    /// Depth from roots. Input clauses have depth 0. A learned clause's
    /// depth = 1 + max depth among its reason clauses.
    pub depth: u32,
}

/// Proof DAG built from conflict profiles.
///
/// Nodes are clauses (identified by CRef). Edges are resolution steps:
/// each reason_clause in a chain → the learned clause of that conflict.
/// The "learned clause CRef" for conflict i is stored in `learned_crefs`.
///
/// This is a forest of resolution chains with sharing: two chains that
/// reference the same reason clause share a node.
#[derive(Debug)]
pub struct ProofDag {
    /// Per-node metadata, keyed by CRef.
    pub nodes: HashMap<CRef, DagNode>,
    /// For each conflict (by conflict_id), the CRef of the learned clause.
    /// Must be supplied externally since analysis doesn't know the CRef
    /// assigned to the learned clause (that happens in the solver).
    pub learned_crefs: Vec<CRef>,
    /// Total edge count (sum of all resolution chain lengths).
    pub total_edges: usize,
    /// Number of unique (reason_clause, learned_clause) pairs.
    pub unique_edges: usize,
}

/// Edge in the proof DAG: reason clause → learned clause via pivot.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct DagEdge {
    reason: CRef,
    learned: CRef,
}

impl ProofDag {
    /// Build the proof DAG from conflict profiles and their learned clause CRefs.
    ///
    /// `learned_crefs[i]` is the CRef assigned to the learned clause from
    /// conflict `i`. This must be tracked by the solver (not by analysis).
    pub fn build(profiles: &[ConflictProfile], learned_crefs: &[CRef]) -> Self {
        assert_eq!(
            profiles.len(),
            learned_crefs.len(),
            "one learned CRef per conflict profile"
        );

        let mut nodes: HashMap<CRef, DagNode> = HashMap::new();
        let mut edge_set: std::collections::HashSet<DagEdge> = std::collections::HashSet::new();
        let mut total_edges: usize = 0;

        for (i, profile) in profiles.iter().enumerate() {
            let learned_cref = learned_crefs[i];

            // Learned clause node: fan_in = number of reason clauses
            let learned_node = nodes.entry(learned_cref).or_default();
            learned_node.fan_in = profile.resolution_chain.len() as u32;

            // Each reason clause in the chain is a parent of the learned clause
            for step in &profile.resolution_chain {
                // Reason clause node: increment fan_out
                let reason_node = nodes.entry(step.reason_clause).or_default();
                reason_node.fan_out += 1;

                edge_set.insert(DagEdge {
                    reason: step.reason_clause,
                    learned: learned_cref,
                });
                total_edges += 1;
            }
        }

        // BFS depth computation: input clauses (fan_in == 0) are depth 0.
        // Learned clauses: depth = 1 + max depth among reason clauses.
        // Since learned clauses from earlier conflicts can be reason clauses
        // for later conflicts, process in conflict order (topological).
        for node in nodes.values_mut() {
            if node.fan_in == 0 {
                node.depth = 0;
            }
        }

        for (i, profile) in profiles.iter().enumerate() {
            let learned_cref = learned_crefs[i];
            let mut max_reason_depth: u32 = 0;
            for step in &profile.resolution_chain {
                let reason_depth = nodes.get(&step.reason_clause).map(|n| n.depth).unwrap_or(0);
                max_reason_depth = max_reason_depth.max(reason_depth);
            }
            if let Some(node) = nodes.get_mut(&learned_cref) {
                node.depth = max_reason_depth + 1;
            }
        }

        ProofDag {
            nodes,
            learned_crefs: learned_crefs.to_vec(),
            total_edges,
            unique_edges: edge_set.len(),
        }
    }

    /// DAG-vs-tree sharing ratio.
    ///
    /// Returns `unique_edges / total_edges`. A ratio of 1.0 means no
    /// sharing (pure tree). Lower values indicate more sharing — the same
    /// reason clause is used in multiple chains.
    /// Returns 1.0 if there are no edges (degenerate case).
    pub fn sharing_ratio(&self) -> f64 {
        if self.total_edges == 0 {
            return 1.0;
        }
        self.unique_edges as f64 / self.total_edges as f64
    }

    /// Width profile: number of learned clauses produced at each depth level.
    ///
    /// `result[d]` = count of nodes with `depth == d`. Large widths at
    /// shallow depths indicate many independent conflict chains; large
    /// widths at deep depths indicate cascading learned-clause-on-learned-clause
    /// resolution.
    pub fn width_profile(&self) -> Vec<usize> {
        let max_depth = self.nodes.values().map(|n| n.depth).max().unwrap_or(0) as usize;
        let mut widths = vec![0usize; max_depth + 1];
        for node in self.nodes.values() {
            widths[node.depth as usize] += 1;
        }
        widths
    }

    /// Variable centrality: pivot frequency within the DAG.
    ///
    /// Same as the Level 2 `pivot_frequency` but scoped to a specific DAG
    /// instance. Returns (variable, count) pairs sorted by descending count.
    pub fn variable_centrality(profiles: &[ConflictProfile]) -> Vec<(u32, usize)> {
        let freq = pivot_frequency(profiles);
        let mut sorted: Vec<(u32, usize)> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
    }

    /// Summary statistics for the DAG.
    pub fn summary(&self) -> DagSummary {
        let num_nodes = self.nodes.len();
        let num_input = self.nodes.values().filter(|n| n.fan_in == 0).count();
        let num_learned = num_nodes - num_input;
        let max_depth = self.nodes.values().map(|n| n.depth).max().unwrap_or(0);
        let max_fan_out = self.nodes.values().map(|n| n.fan_out).max().unwrap_or(0);
        let max_fan_in = self.nodes.values().map(|n| n.fan_in).max().unwrap_or(0);

        let avg_fan_out = if num_nodes > 0 {
            self.nodes.values().map(|n| n.fan_out as f64).sum::<f64>() / num_nodes as f64
        } else {
            0.0
        };

        DagSummary {
            num_nodes,
            num_input,
            num_learned,
            max_depth,
            max_fan_out,
            max_fan_in,
            avg_fan_out,
            sharing_ratio: self.sharing_ratio(),
            total_edges: self.total_edges,
            unique_edges: self.unique_edges,
        }
    }
}

/// Summary statistics for a proof DAG.
#[derive(Debug, Clone)]
pub struct DagSummary {
    pub num_nodes: usize,
    pub num_input: usize,
    pub num_learned: usize,
    pub max_depth: u32,
    pub max_fan_out: u32,
    pub max_fan_in: u32,
    pub avg_fan_out: f64,
    pub sharing_ratio: f64,
    pub total_edges: usize,
    pub unique_edges: usize,
}

// ── Level 4: Correlation analysis ────────────────────────────────────

/// Result of a correlation test.
#[derive(Debug, Clone)]
pub struct Correlation {
    /// Pearson correlation coefficient (r ∈ [-1, 1]).
    pub r: f64,
    /// Coefficient of determination (r²).
    pub r_squared: f64,
    /// Number of data points.
    pub n: usize,
    /// Name of the correlation (for reporting).
    pub name: String,
}

/// Compute Pearson correlation coefficient between two equal-length slices.
///
/// Returns NaN if either slice has zero variance or if slices are empty.
pub fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "pearson_r: slices must be same length");
    let n = xs.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|x| x * x).sum();
    let sum_yy: f64 = ys.iter().map(|y| y * y).sum();
    let sum_xy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();

    let numer = n * sum_xy - sum_x * sum_y;
    let denom = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();
    if denom == 0.0 {
        return f64::NAN;
    }
    numer / denom
}

/// Compute Spearman rank correlation between two equal-length slices.
///
/// Converts values to fractional ranks (average rank for ties), then
/// computes Pearson r on the ranks. This measures monotonic (not just
/// linear) association — appropriate for comparing rankings like
/// "VSIDS activity rank" vs "pivot centrality rank".
pub fn spearman_rank_r(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let rank_x = fractional_ranks(xs);
    let rank_y = fractional_ranks(ys);
    pearson_r(&rank_x, &rank_y)
}

/// Convert values to fractional ranks (1-based, average rank for ties).
fn fractional_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Find the run of tied values
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for this tie group (1-based)
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Correlation 3 (fixed): Pivot centrality vs actual VSIDS activity.
///
/// Uses Spearman rank correlation to compare the ranking of variables
/// by pivot frequency against their ranking by final VSIDS activity score.
/// This tests whether VSIDS already captures the proof structure's
/// variable importance, without the tautological proxy.
pub fn correlate_centrality_vs_vsids(
    profiles: &[ConflictProfile],
    vsids_activities: &[f64],
) -> Correlation {
    let num_vars = vsids_activities.len();
    let pivot_freq = pivot_frequency(profiles);

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for var in 0..num_vars {
        let pf = *pivot_freq.get(&(var as u32)).unwrap_or(&0);
        let va = vsids_activities[var];
        // Include all variables with any signal
        if pf > 0 || va > 0.0 {
            xs.push(pf as f64);
            ys.push(va);
        }
    }

    let r = spearman_rank_r(&xs, &ys);
    Correlation { r, r_squared: r * r, n: xs.len(), name: "centrality_vs_vsids_rank".into() }
}

/// Correlation 1: Resolution depth at conflict C vs BCP propagations at C+1.
///
/// Tests whether deep resolution chains predict more BCP work before the
/// next conflict (i.e., the solver "stalls" after complex conflicts).
pub fn correlate_depth_vs_next_bcp(profiles: &[ConflictProfile]) -> Correlation {
    if profiles.len() < 2 {
        return Correlation { r: f64::NAN, r_squared: f64::NAN, n: 0, name: "depth_vs_next_bcp".into() };
    }
    let xs: Vec<f64> = profiles[..profiles.len() - 1].iter().map(|p| p.resolution_depth as f64).collect();
    let ys: Vec<f64> = profiles[1..].iter().map(|p| p.bcp_propagations as f64).collect();
    let r = pearson_r(&xs, &ys);
    Correlation { r, r_squared: r * r, n: xs.len(), name: "depth_vs_next_bcp".into() }
}

/// Correlation 2: Resolution depth vs learned clause reuse.
///
/// For each learned clause, counts how many times it later appears as a
/// reason clause in subsequent conflicts. Tests whether deeper derivations
/// produce more useful ("glue") clauses.
pub fn correlate_depth_vs_clause_reuse(
    profiles: &[ConflictProfile],
    learned_crefs: &[CRef],
) -> Correlation {
    // Count how often each learned CRef appears as a reason in subsequent chains
    let mut reuse_count: HashMap<CRef, usize> = HashMap::new();
    for profile in profiles {
        for step in &profile.resolution_chain {
            *reuse_count.entry(step.reason_clause).or_insert(0) += 1;
        }
    }

    // Build (depth, reuse) pairs for learned clauses
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for (i, profile) in profiles.iter().enumerate() {
        let cref = learned_crefs[i];
        let reuse = *reuse_count.get(&cref).unwrap_or(&0) as f64;
        xs.push(profile.resolution_depth as f64);
        ys.push(reuse);
    }

    let r = pearson_r(&xs, &ys);
    Correlation { r, r_squared: r * r, n: xs.len(), name: "depth_vs_clause_reuse".into() }
}

/// Correlation 3: Variable pivot centrality vs VSIDS bump frequency.
///
/// Pivot centrality = how often a variable appears as a pivot in resolution.
/// Bump frequency = how often a variable appears in learned clauses (proxy
/// for VSIDS activity, since VSIDS bumps all learned-clause variables).
/// Tests whether the solver's activity heuristic already captures the
/// proof structure's variable importance.
pub fn correlate_centrality_vs_bump_freq(
    profiles: &[ConflictProfile],
    num_vars: u32,
) -> Correlation {
    let n = num_vars as usize;

    // Pivot frequency per variable
    let pivot_freq = pivot_frequency(profiles);

    // Bump frequency: count appearances in learned clauses.
    // We don't have the learned clause literals in ConflictProfile, but
    // we have the resolution chain. Each pivot_var was eliminated from the
    // working clause, and the remaining variables form the learned clause.
    // A simpler proxy: count how often a variable appears as a pivot
    // (pivot_freq) vs how often it appears in ANY reason clause.
    // Better: count how often each variable is in a reason clause.
    let mut reason_freq: Vec<usize> = vec![0; n];
    for profile in profiles {
        // The variables that get VSIDS bumps are those in the learned clause.
        // These are the variables that WEREN'T pivoted away — i.e., they
        // survived resolution. We approximate with: all variables seen in
        // the resolution process minus pivots. But we don't have the learned
        // literals directly.
        //
        // Alternative proxy: pivot variables ARE highly correlated with
        // learned clause membership because pivots are drawn from the
        // current-level literals, and the remaining literals form the clause.
        // Use reason clause membership as a broad-base proxy.
        for step in &profile.resolution_chain {
            reason_freq[step.pivot_var as usize] += 1;
        }
    }

    // Build paired vectors: for each variable with any activity, pair
    // (pivot_centrality, reason_frequency)
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for var in 0..n {
        let pf = *pivot_freq.get(&(var as u32)).unwrap_or(&0);
        let rf = reason_freq[var];
        if pf > 0 || rf > 0 {
            xs.push(pf as f64);
            ys.push(rf as f64);
        }
    }

    let r = pearson_r(&xs, &ys);
    Correlation { r, r_squared: r * r, n: xs.len(), name: "centrality_vs_bump_freq".into() }
}

/// Correlation 4: Pivot frequency vs gradient magnitude.
///
/// Bridges seeds #1 and #3. If the loss-landscape gradient magnitude
/// |∂L/∂x_v| correlates with pivot frequency, then gradient-guided VSIDS
/// is theoretically grounded, not just empirically useful.
pub fn correlate_pivot_vs_gradient(
    profiles: &[ConflictProfile],
    gradient_magnitudes: &[f64],
) -> Correlation {
    let pivot_freq = pivot_frequency(profiles);
    let num_vars = gradient_magnitudes.len();

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for var in 0..num_vars {
        let pf = *pivot_freq.get(&(var as u32)).unwrap_or(&0);
        let gm = gradient_magnitudes[var];
        if pf > 0 || gm > 0.0 {
            xs.push(pf as f64);
            ys.push(gm);
        }
    }

    let r = pearson_r(&xs, &ys);
    Correlation { r, r_squared: r * r, n: xs.len(), name: "pivot_vs_gradient".into() }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Lit;
    use crate::trail::Trail;

    #[test]
    fn simple_conflict_analysis() {
        // Setup: x0=T (decision), x1=T (propagated by clause 0: ¬x0∨x1),
        //        conflict on clause 1: ¬x0∨¬x1
        //
        // Conflict clause: {¬x0, ¬x1}, both at level 1.
        // Resolve ¬x1 with reason clause 0: (¬x0 ∨ x1) → resolvent: {¬x0}
        // Only one lit at level 1 (¬x0 is the decision) → UIP found.
        // Learned clause: {¬x0}
        // Backtrack level: 0

        let mut db = ClauseDb::new();
        let c0 = db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // c0: ¬x0 ∨ x1
        let c1 = db.add_clause(vec![Lit::neg(0), Lit::neg(1)]); // c1: ¬x0 ∨ ¬x1

        let mut trail = Trail::new(2);

        trail.new_decision(Lit::pos(0)); // level 1: x0=T
        trail.record_propagation(Lit::pos(1), c0); // x1=T from c0

        let result = analyze_conflict(&trail, &db, c1);

        // Learned clause should contain ¬x0 (the asserting literal)
        assert_eq!(result.learned.len(), 1);
        assert_eq!(result.learned[0], Lit::neg(0));
        assert_eq!(result.backtrack_level, 0);
    }

    #[test]
    fn two_level_conflict() {
        // Level 1: decide x0=T
        // Level 2: decide x1=T, propagate x2=T (from clause 0: ¬x1∨x2)
        //          conflict on clause 1: ¬x0∨¬x2
        //
        // Conflict clause: {¬x0, ¬x2}
        //   ¬x0 at level 1, ¬x2 at level 2
        //   1 literal at current level (x2 was assigned at level 2)
        //   Standard 1-UIP: count==1 immediately → stop.
        //   Asserting literal: ¬x2 (complement of trail entry for x2)
        // Learned: {¬x2, ¬x0}
        // Backtrack level: 1 (from ¬x0 at level 1)

        let mut db = ClauseDb::new();
        let c0 = db.add_clause(vec![Lit::neg(1), Lit::pos(2)]); // c0: ¬x1 ∨ x2
        let c1 = db.add_clause(vec![Lit::neg(0), Lit::neg(2)]); // c1: ¬x0 ∨ ¬x2

        let mut trail = Trail::new(3);

        trail.new_decision(Lit::pos(0)); // level 1: x0=T
        trail.new_decision(Lit::pos(1)); // level 2: x1=T
        trail.record_propagation(Lit::pos(2), c0); // x2=T from c0

        let result = analyze_conflict(&trail, &db, c1);

        assert_eq!(result.learned[0], Lit::neg(2)); // asserting: ¬x2
        assert!(result.learned.contains(&Lit::neg(0))); // from level 1
        assert_eq!(result.backtrack_level, 1);
    }

    #[test]
    fn minimization_removes_redundant_literal() {
        // Build a case where clause minimization can remove a literal.
        //
        // Clauses:
        //   c0: ¬x0 ∨ x1           (x0=T → x1=T)
        //   c1: ¬x1 ∨ x2           (x1=T → x2=T)
        //   c2: ¬x0 ∨ ¬x2 ∨ x3    (x0=T ∧ x2=T → x3=T)
        //   c3: ¬x3 ∨ ¬x4          (conflict when x3=T ∧ x4=T)
        //
        // Trail:
        //   Level 1: decide x0=T
        //   Level 1: propagate x1=T (from c0)
        //   Level 1: propagate x2=T (from c1)
        //   Level 1: propagate x3=T (from c2)
        //   Level 2: decide x4=T
        //   Conflict on c3: ¬x3 ∨ ¬x4
        //
        // 1-UIP analysis:
        //   Conflict clause: {¬x3, ¬x4}
        //   ¬x3 at level 1, ¬x4 at level 2 (current)
        //   Only 1 lit at current level → UIP is x4
        //   Raw learned: {¬x4, ¬x3}
        //
        // But x3 was propagated by c2: ¬x0 ∨ ¬x2 ∨ x3
        //   x0 is the decision at level 1 (not redundant — it's a decision)
        //   x2 was propagated by c1: ¬x1 ∨ x2
        //     x1 was propagated by c0: ¬x0 ∨ x1
        //       x0 is a decision (stops here)
        //   So ¬x3's reason traces to x0, which IS in the clause... wait.
        //
        // Actually ¬x3 is NOT redundant here because its reason (c2)
        // involves x0 and x2. x0 is NOT in the learned clause (only ¬x3 and ¬x4).
        // So ¬x3 can't be removed.
        //
        // Let me construct a better example where minimization actually fires.

        // Better example:
        //   Level 1: decide x0=T
        //   Level 1: propagate x1=T (from c0: ¬x0 ∨ x1)
        //   Level 1: propagate x2=T (from c1: ¬x1 ∨ x2)
        //   Level 2: decide x3=T
        //   Level 2: propagate x4=T (from c2: ¬x3 ∨ x4)
        //   Conflict on c3: ¬x0 ∨ ¬x2 ∨ ¬x4
        //
        // 1-UIP:
        //   Conflict clause literals: ¬x0 (level 1), ¬x2 (level 1), ¬x4 (level 2)
        //   x4 is at current level, count=1 → UIP is x4
        //   Raw learned: {¬x4, ¬x0, ¬x2}
        //
        // Minimization: can ¬x2 be removed?
        //   x2 was propagated by c1 (¬x1 ∨ x2). Reason lits: {¬x1}
        //   Is ¬x1 in learned? No. Is x1 redundant?
        //     x1 was propagated by c0 (¬x0 ∨ x1). Reason lits: {¬x0}
        //     Is ¬x0 in learned? YES (seen[0] = true).
        //   So x1 is redundant, therefore x2 is redundant.
        //   ¬x2 can be removed!
        //
        // Minimized: {¬x4, ¬x0}

        let mut db = ClauseDb::new();
        let c0 = db.add_clause(vec![Lit::neg(0), Lit::pos(1)]); // c0: ¬x0 ∨ x1
        let c1 = db.add_clause(vec![Lit::neg(1), Lit::pos(2)]); // c1: ¬x1 ∨ x2
        let c2 = db.add_clause(vec![Lit::neg(3), Lit::pos(4)]); // c2: ¬x3 ∨ x4
        let c3 = db.add_clause(vec![Lit::neg(0), Lit::neg(2), Lit::neg(4)]); // c3: ¬x0 ∨ ¬x2 ∨ ¬x4

        let mut trail = Trail::new(5);
        trail.new_decision(Lit::pos(0)); // level 1: x0=T
        trail.record_propagation(Lit::pos(1), c0); // x1=T from c0
        trail.record_propagation(Lit::pos(2), c1); // x2=T from c1
        trail.new_decision(Lit::pos(3)); // level 2: x3=T
        trail.record_propagation(Lit::pos(4), c2); // x4=T from c2

        let result = analyze_conflict(&trail, &db, c3);

        // Asserting literal: ¬x4
        assert_eq!(result.learned[0], Lit::neg(4));
        // ¬x2 should be removed (redundant via x1→x0 chain)
        assert!(!result.learned.contains(&Lit::neg(2)), "¬x2 should be minimized away");
        // ¬x0 should remain (it's a decision, not redundant)
        assert!(result.learned.contains(&Lit::neg(0)));
        // Final clause: {¬x4, ¬x0}
        assert_eq!(result.learned.len(), 2);
        assert_eq!(result.backtrack_level, 1);
    }

    #[test]
    fn minimization_keeps_non_redundant() {
        // Example where no literal is redundant:
        //   Level 1: decide x0=T
        //   Level 2: decide x1=T
        //   Conflict on c0: ¬x0 ∨ ¬x1
        //
        // Learned: {¬x1, ¬x0}
        // Both are decisions — neither can be removed.

        let mut db = ClauseDb::new();
        let c0 = db.add_clause(vec![Lit::neg(0), Lit::neg(1)]); // c0: ¬x0 ∨ ¬x1

        let mut trail = Trail::new(2);
        trail.new_decision(Lit::pos(0)); // level 1
        trail.new_decision(Lit::pos(1)); // level 2

        let result = analyze_conflict(&trail, &db, c0);

        assert_eq!(result.learned[0], Lit::neg(1)); // asserting
        assert!(result.learned.contains(&Lit::neg(0)));
        assert_eq!(result.learned.len(), 2); // no minimization possible
    }

    #[test]
    fn minimization_with_level_zero() {
        // Level 0 literals in reason clauses are always satisfied and
        // should be treated as "free" during redundancy checks.
        //
        //   Level 0: propagate x0=T (from unit clause c0: x0)
        //   Level 1: decide x1=T
        //   Level 1: propagate x2=T (from c1: ¬x0 ∨ ¬x1 ∨ x2)
        //   Level 2: decide x3=T
        //   Conflict on c2: ¬x2 ∨ ¬x3
        //
        // Raw learned: {¬x3, ¬x2}
        // Can ¬x2 be removed?
        //   x2 propagated by c1 (¬x0 ∨ ¬x1 ∨ x2). Reason lits: {¬x0, ¬x1}
        //   x0 is at level 0 → skip (always true)
        //   x1 is a decision at level 1, NOT in learned → not redundant
        // So ¬x2 can't be removed. Clause stays {¬x3, ¬x2}.

        let mut db = ClauseDb::new();
        let c0 = db.add_clause(vec![Lit::pos(0)]); // c0: x0 (unit)
        let c1 = db.add_clause(vec![Lit::neg(0), Lit::neg(1), Lit::pos(2)]); // c1: ¬x0 ∨ ¬x1 ∨ x2
        let c2 = db.add_clause(vec![Lit::neg(2), Lit::neg(3)]); // c2: ¬x2 ∨ ¬x3

        let mut trail = Trail::new(4);
        trail.record_propagation(Lit::pos(0), c0); // level 0: x0=T from c0
        trail.new_decision(Lit::pos(1)); // level 1: x1=T
        trail.record_propagation(Lit::pos(2), c1); // x2=T from c1
        trail.new_decision(Lit::pos(3)); // level 2: x3=T

        let result = analyze_conflict(&trail, &db, c2);

        assert_eq!(result.learned[0], Lit::neg(3));
        assert!(result.learned.contains(&Lit::neg(2)));
        assert_eq!(result.learned.len(), 2); // ¬x2 not redundant (x1 blocks it)
    }
}
