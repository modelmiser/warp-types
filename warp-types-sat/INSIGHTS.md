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
