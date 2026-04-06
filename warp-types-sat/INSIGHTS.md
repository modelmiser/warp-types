# warp-types-sat — Insights

## 1. LBD (Literal Block Distance) — The Quality Metric That Matters

LBD (Audemard & Simon, 2009) measures learned clause quality by counting distinct decision levels among its literals. A clause with LBD=2 ("glue clause") connects exactly two decision levels — these provide critical implication chains and should never be deleted. Higher LBD clauses are more situational.

For 200-var random 3-SAT at the phase transition, protecting LBD ≤ 2 clauses cut the conflict count in HALF compared to only protecting binary clauses (87k vs 162k). The quality signal from LBD is that strong.

## 2. The Ghost Empty Clause Bug

Deleted clauses with cleared literals (`Vec::clear()`) look like empty clauses — and empty clauses mean "trivially false" in SAT semantics. When BCP's `queue_head` resets to 0 after a restart (no level-0 propagations), it re-scans all clauses for unit/empty, finds tombstoned clauses, and reports spurious UNSAT.

Same class of bug as "ghost assignments" (documented in trail.rs) — state that looks semantically meaningful but is leftover from a retracted operation. Tombstone patterns need consistent guards everywhere.

## 3. Cache Locality Trumps Algorithmic Tuning

With tombstone-based clause deletion, live clauses get scattered across a multi-MB Vec. For 200-var UNSAT with 88k total clause entries but only ~4k live, BCP's random clause accesses cause L2 cache misses (~100ns each vs ~1ns L1 hit). This single factor accounted for ~35% of total solver time.

The fix: compact the database after each reduction — move live clauses to contiguous indices, rebuild watches from the compact db. This shrunk the working set from 2.8MB to ~130KB (fits comfortably in L2), dropping per-conflict time from 0.44ms to 0.145ms.

## 4. Deletion Aggressiveness vs Clause Utility — The Core Tradeoff

| Config | Conflicts | Live Clauses | Time | Per-conflict |
|--------|-----------|-------------|------|-------------|
| No deletion (original) | ~82k | unbounded | 106s | 1.3ms |
| 50% delete, 2k interval, compact | 139k | 4.4k | 18s | 0.13ms |
| 67% delete, 2k interval, compact | 162k | 4.4k | 22s | 0.14ms |
| 50% delete, growing interval, compact | 77k | 11.9k | 22.5s | 0.29ms |

More aggressive deletion → fewer live clauses → faster per-conflict (cache). But also more total conflicts (useful knowledge lost). The sweet spot for 200-var: fixed 2000 interval, 50% deletion, LBD ≤ 2 protection.

## 5. Clause Minimization — 7.8x SAT speedup from shorter learned clauses

MiniSat's `litRedundant` removes literals from learned clauses when their assignments are already implied by other literals in the clause. On 200-var random 3-SAT, this produced a 7.8x speedup for SAT (2,423ms → 309ms) and 1.6x for UNSAT (18,197ms → 11,162ms). Three compounding effects: (1) fewer BCP operations per clause evaluation, (2) improved LBD scores promoting more clauses to protected "glue" status, and (3) slower clause database growth reducing cache pressure.

The abstract levels optimization is critical: a u64 bitmask of decision levels present in the learned clause rejects most redundancy candidates in O(1) without entering the implication graph DFS. On failure, all marks from the failed DFS are rolled back; on success, they persist as a transitivity cache for subsequent checks.

## 5. VSIDS Priority Heap — The MiniSat Pattern

Replacing the O(n) linear scan in `pick()` with a binary max-heap (O(log n)) requires careful coordination with the solver:

**Heap position map is mandatory.** The `heap_pos[var] → position` array enables O(1) lookup for `bump()` to sift a variable up after activity increase. Without it, finding a variable in the heap is O(n), negating the benefit.

**Tie-breaking determines initial decision order.** When all activities are zero (cold start), the heap's tie-breaking rule determines the first ~32 decisions before VSIDS kicks in. Changing from "highest index first" to "lowest index first" turned a sub-second SAT solve into an 80-second one on the same seed — same algorithm, same clauses, exponentially different search tree. The fix: `activity_gt` compares `a > b` on ties, matching MiniSat convention and the old `max_by`'s "last equal element" behavior.

**Re-insertion on backtrack.** Variables are popped from the heap by `pick()` and must be re-inserted when unassigned during backtrack. The `entries_above(level)` method avoids scanning the full trail — it slices directly into the retracted entries using the `level_starts` array that `Trail` already maintains.

## 7. Flat Arena Clause Storage — 2.8-3.7x from Eliminating Pointer Chases

Replacing `Vec<Clause>` (each with heap-allocated `Vec<Lit>`) with a flat `Vec<Lit>` arena + offset/length arrays eliminated the second pointer chase on every clause access during BCP. Results on 200-var random 3-SAT: SAT seed 0 dropped from 309ms to 112ms (2.8x), UNSAT seed 1 from 11,162ms to 2,981ms (3.7x).

**UNSAT benefits more because** it runs far more conflicts (thousands of BCP rounds), so the cache win compounds over many more watch replacement searches. SAT instances find solutions within hundreds of conflicts — less time in the inner loop to amortize.

**API-preserving refactor:** `ClauseRef<'a> { pub literals: &'a [Lit] }` matches the old `Clause { pub literals: Vec<Lit> }` access pattern. All 30 call sites across 8 files needed only a trivial mechanical fix (removing a leading `&`). Invasive internal restructuring invisible to consumers.

## 8. Why Binary Clause Implicit Propagation Regresses at 200 Vars

Binary clause optimization is a "standard" MiniSat technique — store 2-literal clauses in the watch entry itself, propagate without clause DB lookup. But at 200 vars it **regressed** by 50%.

**The existing blocker already handles binary clauses well.** For a binary clause (a ∨ b) with blocker=b: when a becomes false, the blocker check tests b. If b is true (50-70% of the time), we skip without any clause DB access — identical to what a dedicated binary fast-path would do. For the remaining cases, the replacement search loop iterates zero times (both literals are watched → both skipped → immediate fall-through to propagate/conflict).

**Two approaches tried, both regressed:**
- *Separate binary watch lists*: Changed propagation order (binary before long), causing search tree divergence. Seed 6: 31K→60K conflicts. Also added cache pollution from the extra `Vec<Vec<BinaryWatch>>`.
- *Inline flag bit (MSB of clause index)*: Preserved propagation order but added a branch per watch entry in the hot loop. 731-line x86 function — the extra branch inflated i-cache pressure more than the empty-loop removal saved.

**The optimization becomes worthwhile** at 500+ vars where the clause DB is large enough that `db.clause(ci)` causes L2 cache misses (~100ns each). At 200 vars the entire clause DB fits in L2, so the "saved" access costs ~1ns — less than the branch overhead.

## 9. Unchecked Indexing — 25% BCP Speedup from Eliminating Bounds Checks

The BCP inner loop calls `eval_lit` ~5 times per watch entry visit (blocker, partner, replacement search, propagation check). Each call does `assignments[lit.var() as usize]` — a bounds-checked array access that generates a conditional branch + call to `panic_bounds_check`.

Replacing with `get_unchecked` in the hot loop eliminated these branches, yielding a **25% per-conflict cost reduction** (pinned-core: 15.5 → 11.6 μs/conflict at 200 vars). The safety argument is straightforward: all literals come from clauses validated at startup (`db.max_variable() < num_vars`), and the assignment array has exactly `num_vars` entries. `debug_assert` guards catch violations in debug builds.

The generated x86 function had 9 `panic_bounds_check` calls before the change. The bounds check cost isn't just the branch itself — it's the i-cache pressure from cold panic paths inflating the function body (746 assembly lines for the BCP function). Each eliminated check removes ~10 bytes of cold code from the hot function.

## 10. Literal-Indexed Assignment Array + Deep Unchecked Indexing — 28% Combined Speedup

Two optimizations that compound:

**Literal-indexed array:** MiniSat's `value(p)` is a single array lookup: `assigns[lit.code()]`. Our old approach stored values per-variable and computed `assignments[lit.var()].map(|v| if lit.is_negated() { !v } else { v })` — a shift, load, and conditional XOR per evaluation. The literal-indexed array stores truth values by `lit.code()`, so `eval_lit` becomes `lit_values[lit.code()]` — one load, zero branches. Memory cost: 2N bytes instead of N (negligible). Both arrays are maintained in sync; the hot-path BcpTrail exposes `lit_values`, cold-path code uses the original `assignments`.

**Deep unchecked indexing:** Extended unchecked access from just `eval_lit` to `db.is_deleted()` (1 check), `watches.watched[ci]` (1 check), and `db.clause_unchecked()` (3 checks: offsets[], lengths[], arena slice). All use clause indices from watch entries, which are constructed only from valid clause indices. Same safety argument — `debug_assert` guards in debug builds.

**Combined result:** BCP function assembly dropped from 731 to 634 lines, panic paths from 49 to 5. At 500 vars: 21.0 → 14.1 μs/conflict (28-42% improvement depending on seed). Per-propagation: ~0.27 → ~0.18 μs. Gap to MiniSat: ~4.5x → ~3.0x. The 5 remaining panic paths are from `watches.lists[]` accesses (indexed by literal codes — bounded but compiler can't prove it statically).

## 11. Phase Timing Breakdown — BCP Confirmed as Bottleneck

Per-phase instrumentation in `solve_cdcl_core_inner` reveals BCP dominates at 63-78% of wall time across problem sizes. VSIDS is negligible (2-4%). Analysis grows from 15% at 200v to 27% at 500v.

BCP per-propagation cost: 115-274ns vs MiniSat's ~60ns. The 2-4.5x BCP gap is the primary optimization target.

Key structural delta vs MiniSat: separate `watched` Vec requires an extra cache line access per clause visit. MiniSat stores watches inline at c[0]/c[1] and searches from c[2] — no w0/w1 comparison needed. This fundamental cache-line layout difference is a major contributor to the latency gap and points to the next optimization frontier: inline watched literal storage.

## 12. NLL Single-Borrow BCP and Analysis Sub-Profiling

**BCP double-borrow eliminated via NLL.** The long-clause BCP path called `clause_unchecked` twice: once to read c[0]/c[1], then again for the replacement search (starting at c[2]). Rust's Non-Lexical Lifetimes allow a single borrow scope: read c[0], c[1], and search for replacement in one `ClauseRef` lifetime, then release before the mutable `swap_literal_unchecked`. This eliminates redundant `offsets[ci]` + `lengths[ci]` lookups and gives the compiler a longer register-alive window. Result: ~5% ns/prop improvement at 500v.

**Option<bool> niche optimization is already optimal.** Assembly inspection confirmed Rust stores `Option<bool>` as u8 {0=Some(false), 1=Some(true), 2=None}. The proposed u8 sentinel encoding would produce identical codegen. The hot-path blocker check is `movzx ecx, BYTE [rdx+rbp]; test ecx, ecx; cmp ecx, 2; jne skip` — no unwrapping overhead.

**Analysis breakdown at 500v: resolution 60%, minimization 40%.** Adding `resolve_ns` and `minimize_ns` sub-timing reveals the litRedundant DFS (clause minimization) is ~10% of total solve time at 500v, growing faster than resolution as problem size increases (longer implication chains → deeper DFS). Resolution includes the backward trail scan + reason clause iteration.

**Second-watch selection matters.** Placing the highest-level non-asserting literal at c[1] (standard MiniSat technique) ensures the clause is immediately unit after backtracking. Without it, c[1] might be at a level above backtrack_level (unassigned after backtrack), creating an unnecessarily slack clause. This changes search behavior substantially — different learned clauses, different conflict counts. Not just an optimization; it affects solver quality.

## 8. Inline-Header Arena — Co-locating Metadata with Data

**Eliminating parallel arrays for clause metadata yields 15-20% BCP speedup.** The previous design stored clause data across 5 parallel vectors (arena, offsets, lengths, lbd, deleted). Each clause access required two loads from separate cache lines (offsets[i] + lengths[i]) before touching the literal data. The inline-header layout packs `[header|lit0|lit1|...]` contiguously — one header read (adjacent to literals in the same cache line) replaces two distant loads. At 500 variables: ns/prop dropped from 130-165 to 111-130.

**CRef (arena offset) vs sequential index is a fundamental design choice.** Replacing sequential clause indices with arena offsets (CRef = u32) propagates through the entire system: WatchEntry, Reason::Propagation, BcpResult::Conflict, conflict analysis. The hot path benefits (WatchEntry stores CRef directly → no offsets[] indirection), but cold-path code (gradient solver, SoA construction) needs an explicit `crefs: Vec<CRef>` for sequential iteration. The tradeoff: hot path is faster, cold path adds one Vec allocation at init time.

**Header packing: 1 bit deleted + 11 bits LBD + 20 bits length in one u32.** The `#[repr(transparent)]` on Lit enables safe `&[u32]` → `&[Lit]` transmute via `slice::from_raw_parts`, making the mixed-type arena (headers as u32, literals as Lit) work without copies. The 20-bit length limit (1M literals per clause) is more than sufficient — real SAT instances rarely exceed ~100 literals per clause.

## 9. Blocker-Before-Deleted — Hot Path Instruction Elimination

**Moving the blocker check before the deleted check in BCP eliminates a random arena access from 50-70% of watch entries.** The original code checked `is_deleted(cref)` first (requiring an arena load to read the clause header) before checking the blocker literal. But MiniSat checks the blocker FIRST — if the blocker is satisfied, the clause is skipped without any clause DB access.

**Assembly impact at the inner loop (`.LBB112_27`):** Hot path shrank from 13 instructions (2 stack loads + 1 random arena access) to 6 instructions (1 stack load + 1 L1-hitting lit_values access). The compiler also improved the blocker check from two comparisons (`testl` + `cmpl $2`) to a single bit test (`testb $1, %dl`), exploiting that `Some(true) = 1` is the only `Option<bool>` value with bit 0 set.

**Correctness of keeping stale entries:** Deleted clauses whose watch entries pass the blocker check are kept (not compacted out). This is harmless — stale entries are cleaned up during the periodic watch rebuild after clause compaction (`Watches::new` rebuild in solver.rs). This is the standard MiniSat tradeoff.

**Measured: 4-6% ns/prop improvement at 500v, 20-35% at 200v.** The larger improvement at 200v is expected — smaller arenas fit in L1, so the eliminated arena access was the dominant latency. At 500v, the arena is larger and the blocker hit rate lower, reducing the relative benefit.

## 10. Level 2 Proof DAG Mining — ConflictProfile as Observable

Per-conflict profiling at 200-var phase transition (10K budget) reveals the solver's conflict structure: avg resolution depth 16.1, avg LBD 8.8, avg backtrack distance 1.3, avg BCP propagations 42.4. Key findings:

**Pivot frequency is heavily skewed.** 200 unique pivot variables, max frequency 1,999 (10x average). Some variables are structurally central to proofs — they appear as pivots in many independent conflict derivations. This is the "bottleneck variable" hypothesis that Level 4 will test against VSIDS activity scores.

**Clause reuse follows a power law.** 10,304 unique reason clauses, max reuse 1,026. A tiny fraction of clauses do most of the resolution work. This aligns with glue clause theory — LBD ≤ 2 clauses should correlate with high reuse frequency.

**Depth and clause size are weakly correlated (r=0.192).** Deeper resolution chains produce slightly larger clauses, but the relationship is loose — they carry partially independent information. Both are worth tracking for Level 4 heuristic correlations.

**Backtrack distance is surprisingly small (avg 1.3).** Most conflicts resolve by backtracking 1-2 levels, even on hard instances. This suggests the solver rarely makes deeply wrong decisions — it gets stuck locally, not globally.

## 11. Level 3 Proof DAG — Near-Tree Structure with Shared Inputs

The proof DAG on 200-var 3-SAT (10K conflicts) has 10,923 nodes and 160K edges, with sharing ratio 0.997 (near-tree). This reveals an important structural distinction: the proof's *derivation structure* is almost a tree (each learned clause's derivation is unique), but the *input clauses* feeding those derivations are heavily shared (max fan-out 1,026). The "sharing" that matters for solver heuristics is fan-out, not edge-level reuse.

Width profile is hourglass-shaped: depth 0 (1,983), depth 1 (2,064 — wider!), then exponential decay. The depth-1 bulge means the solver generates more first-generation learned clauses than there are input clauses participating in resolution. Variable centrality is extreme: x1 has 1,999 pivot appearances vs ~800 average — a 2.5x skew for the top variable.

## 12. Level 4 Correlations — One Actionable, One Tautological, Two Null

Four topology↔solver correlations across 5 seeds (200-var 3-SAT, 10K budget each):

**C2: depth→clause_reuse (r=-0.174, consistent negative) — ACTIONABLE.** Shallow resolution chains produce clauses that get reused more often as reasons in future conflicts. Suggests clause deletion could penalize high resolution-depth clauses as an independent signal alongside LBD. Depth and LBD are only weakly related (r=0.192 from Level 2), so depth provides partially independent information about clause utility.

**C3 (FIXED): centrality→VSIDS (Spearman r_s=+0.524) — THE REAL FINDING.** Original C3 was tautological. Fixed version uses Spearman rank correlation with actual final VSIDS activity scores. VSIDS captures ~26% of variance in proof-structure variable importance. The ~74% gap is unexploited signal — variables that are frequent pivots (structurally central to proofs) but don't have proportionally high VSIDS scores. Pivots get resolved away and don't appear in learned clauses; VSIDS only bumps learned-clause variables. These are genuinely different structural roles.

**C2 EXPERIMENT (FAILED): depth-weighted deletion.** A/B test on 20 seeds: adding α·resolution_depth to LBD deletion score increases conflicts by 6-14% at α≥0.25. Depth is downstream of LBD (shallow chains → short clauses → low LBD), not independent of it. The correlation (r=-0.174) was real but the causal structure prevents it from being actionable in deletion.

**C1: depth→next_bcp (r=+0.157, consistent) — REAL BUT WEAK.** Deep chains → slightly more BCP work next cycle. Only 2.5% variance explained. Not actionable.

**C4: pivot→gradient (r≈0, mixed sign) — NULL.** Cold gradient magnitude has no relationship to pivot centrality. The seed-1↔seed-3 bridge hypothesis fails: gradient-VSIDS works empirically because gradients are computed *at the trail*, but the proof structure doesn't encode a persistent gradient signal.

## 13. Pivot-Augmented VSIDS — 37% Conflict Reduction at 200 Vars

Bumping pivot variables by `scale × increment` during conflict analysis (in addition to standard learned-clause bumps) produces a massive effect at 200-var phase transition:

| Scale | Conflicts | Solved | vs Base |
|------:|----------:|-------:|--------:|
| 0.00  | 537,448   | 16/20  |    —    |
| 0.25  | 393,767   | 19/20  | -26.7%  |
| 0.50  | 339,922   | 20/20  | -36.8%  |
| 1.00  | 336,138   | 20/20  | -37.5%  |
| 2.00  | 591,942   | 14/20  | +10.1%  |

Sweet spot: scale 0.5-1.0. Scale 2.0 overshoots (over-prioritizes pivots, loses decision diversity).

**Does not scale to 300-var** (all scales within ±5% of baseline, 0-2/20 solved with 50K budget). Either the budget is too small to reveal differences, or pivot frequency becomes less discriminating at larger n.

**Mechanism:** C3 showed VSIDS captures 26% of pivot centrality. Pivots are resolved away during conflict analysis and don't appear in learned clauses, so standard VSIDS misses them. The augmented bump feeds this structural signal back into decisions. The 74% gap is genuine unexploited information — at 200 vars.

**Scaling diagnosis (300v × 200K budget + entropy):** The 300-var null is NOT a budget artifact (+0.4%/+2.8% with 4× budget). Pivot frequency entropy is identical at 200v and 300v (H_norm ≈ 0.97), so the signal doesn't degrade. The real issue: the improvement is constant-factor (~37%) but phase-transition difficulty is exponential. At 200v, most seeds are near the solvability boundary; at 300v, 18/20 are deep in the exponential regime where no constant-factor heuristic helps.

**Combination with gradient (seed-1×3):** Pivot-only (-32.1%) beats combined gradient+pivot (-28.1%). Gradient-only is +8.3% worse than baseline. The signals interfere: gradient probes overwrite VSIDS saved phases, disrupting the search diversity that correct variable ordering (from pivots) relies on. Pivot bumps baked into production `solve_watched()` at DEFAULT_PIVOT_BUMP_SCALE=0.5.
