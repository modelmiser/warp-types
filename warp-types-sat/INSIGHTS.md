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

## 10. Pointer Iteration — Reducing Loop Control Overhead

**Replacing index-based iteration (i/j/ws.len()) with pointer-based (src/dst/ws_end) eliminated redundant loop counter maintenance.** The compiler was maintaining four representations of the loop index simultaneously (i, i+1, a countdown, and a pointer advance) — 8 instructions of loop overhead per watch entry. With pointers, only src/dst/ws_end need tracking — 3 values vs 5+.

**Conflict drain paths use `std::ptr::copy` (memmove) instead of element-by-element loops.** The regions CAN overlap: dst ≤ src but dst + remaining > src when few entries have been removed (j close to i). The initial implementation incorrectly used `copy_nonoverlapping` (UB for overlapping regions) — caught during assembly review.

**Combined with blocker-before-deleted: 8-10% total improvement at 500v, up to 40% at 200v.** P-core equivalent dropped from ~82-94 to ~74-85 ns/prop. Gap from MiniSat narrowed from 1.3-1.5x to 1.2-1.4x.

## 11. Order-Preserving O(1) Insert in Conflict Analysis

**`learned.insert(0, lit)` is O(n) but the naive fix `push + swap` changes literal ordering and diverges the solver's search path.** The second-watch selection breaks ties by position — reordering non-asserting literals changes which literal becomes c[1], altering watches and cascading through the entire search tree. For seed 0 at 200v: conflict count went from 17,144 to 6,105 — a completely different solve from a semantically-correct optimization.

**Fix: reserve slot 0 with a placeholder before resolution begins.** `learned.push(Lit::pos(0))` at init, `learned[0] = asserting_lit` after UIP found. O(1) overwrite, literal ordering preserved, search path identical.

**Additional analysis optimizations:** Continue UIP scan from `trail_idx` instead of rescanning the full trail; accumulate `abstract_levels` during resolution (eliminate second pass); unchecked indexing in cold paths. Combined: resolution phase shrank ~10% (13% → 11-12% of total at 500v), 2-5% overall wall time.

## 12. Hardware Counters vs Wall-Clock — Measurement Methodology Matters

**Wall-clock timing on shared systems has ±5-10% noise that can obscure or fabricate optimization results.** Switching to `perf stat` with hardware counters on an isolated e-core revealed that every optimization this session was real, but wall-clock measurements nearly caused us to dismiss the biggest one.

**The pointer iteration change removed 29.2M instructions (10.3%)** — but wall-clock showed it as a borderline ±4% maybe-win. The first clean wall-clock run looked flat; only the second/third happened to be quieter. With hardware counters, 593→532 insn/prop is unambiguous on the first measurement.

**The blocker reorder ADDED 1.7M instructions but saved cycles.** Wall-clock correctly showed improvement, but for the wrong reason — the win was fewer L2 misses from the deferred arena access, not fewer instructions. Hardware counters revealed the actual mechanism.

**IPC dropped from 1.27 to 1.15 as we optimized.** We removed "cheap" instructions (loop overhead, bounds checks, redundant passes) while keeping "expensive" ones (data-dependent loads, unpredictable branches). Lower IPC after optimization is the expected signature of removing computational slack.

**Branch mispredictions are invariant to code structure** (8.26→8.06 misses/prop across all four states). The ~8 misses/prop are inherent to the BCP algorithm's data-dependent blocker checks. At ~15-20 cycle penalty each, misprediction accounts for **28% of total cycles** — the dominant remaining bottleneck. Reducing this requires algorithmic changes (better watch ordering, branch-free evaluation) not micro-optimization.

**Methodology:** `perf stat` with `cpu_atom/instructions/u` on `taskset -c 16`. Instructions are deterministic (±0.0002%). Cycles are ±0.3%. Use `warp-types-sat/src/bin/perf_bench.rs` (500v, seed 1, 5190 conflicts).

## 13. Branchless false_pos — Profile-Guided Branch Elimination

**`perf record -e cpu_atom/branch-misses/u` + `perf annotate` identified 4 misprediction clusters** in the BCP inner loop:

| Cluster | Source | % of BCP misses | Nature |
|---------|--------|-----------------|--------|
| Replacement search exit | for-loop `break` on first non-false literal | ~19% | Predictor learns "continue", mispredicts on replacement found |
| Blocker hit/miss | `blocker_val == Some(true)` | ~13% | Data-dependent on assignment |
| Partner satisfied | `partner == Some(true)` | ~9% | Data-dependent on assignment |
| false_pos determination | `c0 == false_lit` | ~8% | ~50/50, compiler emitted branch not cmov |

**The false_pos branch was a simple conditional select** (`if c0 == false_lit { (c1,0) } else { (c0,1) }`) that LLVM compiled to a `cmpl + jne` with two separate load paths instead of a conditional move. The fix: bitmask selection via `wrapping_neg()` — `Lit` is `#[repr(transparent)]` over u32, so `(c1.code() & mask) | (c0.code() & !mask)` is safe and compiles to 3 ALU ops with zero branches.

**Result: -1.28 branch misses/prop (8.06→6.78), -10% cycles, +11% IPC.** Instruction count unchanged (+0.2%). The cycle savings (~43 cyc/prop) imply a ~34-cycle effective misprediction penalty — higher than Gracemont's nominal ~15-cycle pipeline depth, suggesting mispredictions also evict prefetched watch list and clause data from L1, compounding the cost with downstream cache misses.

**Secondary benefit: partner-satisfied misses collapsed from 9% to 2% of BCP.** The eliminated branch was polluting the TAGE predictor's pattern history for the nearby partner check (Gracemont shares predictor entries for nearby addresses). Branch predictor aliasing is a known microarchitectural effect but rarely this visible — eliminating one branch improved a completely different branch.

**LLVM optimized the bitmask to cmov.** The source uses `wrapping_neg()` + bitwise OR, but LLVM recognized the pattern and emitted `cmove` + `setne` — the optimal x86 encoding. Writing branchless arithmetic was the right hint even though the compiler found a better encoding.

**Remaining BCP branches are algorithmically fundamental.** Post-fix profiling shows: replacement search exit (~28% of BCP, loop exit misprediction), blocker hit/miss (~21%, assignment-dependent), deleted-check skid (~14%, blocker secondary). These can't be made branchless without changing the algorithm itself — each reflects genuine uncertainty about the current assignment state.

## 14. Software Prefetch — Why It Failed at 500 Vars

**Attempted: prefetch next watch entry's clause header during current iteration.** Added `_mm_prefetch(arena[next_cref], T0)` early in the BCP loop to warm clause data into L1 while the blocker/deleted/binary checks execute. Result: +28 insn/prop (5.5%), -0.8% cycles (noise), +0.2 bmiss/prop from the `if src < ws_end` guard branch. **Reverted.**

**Root cause: the arena fits in L2.** Cache misses at 500v are 0.014/propagation — essentially zero. The ~300KB arena fits comfortably in Gracemont's ~2MB L2 cluster. L2→L1 promotion saves ~4 cycles per miss × 0.014 misses = 0.06 cyc/prop, which is negligible vs the 28-instruction overhead.

**The 34-cycle effective misprediction penalty is NOT cache-related.** It's purely speculative execution waste — Gracemont's out-of-order window discards more µops on flush than the nominal 15-stage pipeline would suggest. This means branch-miss cost at 500v is "hard" — there's no latency-hiding trick available; the only remedies are fewer mispredictions (algorithmic) or fewer iterations (search behavior).

**Prefetch would matter at 1000+ vars** where the arena exceeds L2 and clause accesses hit L3 (~30 cycles) or DRAM (100+ cycles). At that scale, the 28-instruction overhead becomes a bargain for hiding 30-100 cycle latencies. File this as a conditional optimization: `#[cfg]`-gated, enabled for large instances.

## 15. Analysis Unchecked Indexing — Instructions Down, Cycles Flat

**Eliminated ~13 bounds checks across resolution and minimization hot loops.** Same pattern as BCP Insight #9: `work.seen[var]` → `get_unchecked`, `entries[trail_idx]` → `get_unchecked`. Safety argument: all vars from clause DB (validated < num_vars at startup), trail_idx bounded by algorithm termination.

**Result: -2.4% instructions (505→493/prop), cycles flat (-0.5%, noise).** The BCP equivalent gave 25% because BCP's 731-line function had acute i-cache pressure from panic paths. Analysis functions are shorter — the cold panic paths don't compete for hot cache lines. The always-predicted-correct bounds check branches cost ~0 cycles each (unlike data-dependent branches where misprediction matters).

**Retained for instruction budget and future scaling.** 5.5M fewer retired instructions per solve. At larger problem sizes where analysis grows faster than BCP (Insight #11: analysis is 15% at 200v, 29% at 500v), the i-cache benefit will compound.

## 16. VSIDS Warm-Start — 13x Fewer Conflicts From a One-Liner

**`initialize_from_clauses()` was implemented in Vsids but never called from the solver.** Adding the call at solver init gives each variable an initial activity proportional to its clause occurrence count. Variables appearing in more clauses are more constrained — deciding them first avoids wasting the first ~100 decisions on arbitrary variables (the cold-start default decided by variable index, highest first).

**On 500v seed 1: 5190 → 397 conflicts, 473787 → 37664 propagations. Wall: 69ms → 3ms (23x).** This single line dwarfs the cumulative effect of 5 micro-optimization commits (-15% cycles). The lesson: algorithmic decisions (what to compute) dominate implementation decisions (how fast to compute it) by orders of magnitude.

**Per-propagation cost also improved: 493 → 489 insn, 392 → 364 cyc.** Better decisions keep the solver in search regions where BCP encounters more satisfied blockers, shorter replacement searches, and shallower conflicts. Per-prop cost is NOT independent of decision quality — they compound.

**This is the standard MiniSat initialization**, implemented months ago and sitting unused. The function already handled pre-seeded activities correctly (uses `+=` not assignment, rebuilds heap after). Wiring it up required zero API changes.

## 17. GPU SAT Gradient Kernel — SoA + Butterfly Reduce in f64

**The infrastructure gap was smaller than expected.** warp-types already had: `GpuShuffle for f64` (splits into two `shfl.sync.bfly.b32` instructions), `Warp<All>.reduce_sum(PerLane<f64>)` (5 butterfly stages), `#[warp_kernel]` proc macro, and `WarpBuilder` for PTX compilation. The only missing piece was `block_id_x()` (PTX `%ctaid.x`) for multi-block launches — one function, three lines.

**The kernel is 10 loads + 4 multiplies + 10 shuffles per clause.** SoA layout gives coalesced reads for all 7 SoA arrays (vars × 3, negs × 3, weights × 1). The three variable lookups (`x[var_i]`) are scattered reads, but L1 caches them well since variables are shared across many clauses. The butterfly reduce is pure register traffic — no shared memory needed.

**f64 on H200 Hopper**: Native FP64 at 1:2 ratio vs FP32 (not the 1:64 ratio of consumer GPUs). Each shuffle-XOR for f64 emits two 32-bit shuffles. The kernel is compute-bound on the multiplies, not the shuffles.

**Design pattern: CPU SimWarp → GPU kernel.** Write and validate the math on CPU using SimWarp (real multi-lane shuffle semantics), then the kernel is a mechanical translation — same operations, same order, same result. The GPU test just confirms f64 bit-identical results between SimWarp and real hardware.

## 18a. Device-Resident Gradient Loop — 3 Kernels, Pattern 5 Fusion Target

Added `variable_update` and `grad_norm_reduce` kernels to keep x, grad, velocity on device across iterations. Eliminates per-iteration host↔device transfer of grad (download) and x (upload). Both new kernels read `grad[]` — this is a Pattern 5 (multi-pass reduction) fusion candidate: fusing them into `variable_update_with_norm` saves one full pass over the grad array.

**Profiling on RTX 4000 Ada (5000 vars):** Original (1 kernel + host xfer) 48μs vs Resident (3 kernels, device-only) 46μs → 1.04x. The two extra kernel launches nearly cancel the transfer savings. This is where kernel-fuse Pattern 5 fusion earns its keep: fusing variable_update + grad_norm_reduce eliminates one launch, tipping the balance.

**AtomicAdd trajectory sensitivity:** Single-iteration variable_update matches CPU to 1e-4. Over 200 iterations, trajectory divergence grows to ~0.1 in best_loss due to atomicAdd ordering differences compounding through a chaotic landscape. SAT-finding rates match (6/10 vs 6/10) — the divergence is in path, not in outcome quality. This is inherent to IEEE 754 non-associativity under concurrent atomic writes.

**H200 prediction:** At 100K+ vars, the fused kernel is memory-bound (AI ~0.3 vs balance 7.0). The variable_update + grad_norm_reduce fusion saves ~1.6MB of duplicate grad[] reads per iteration. kernel-fuse should find and implement this automatically.

## 18. GPU Loss Crossover — 14x at 5000 vars, ~17μs Floor

**Crossover at ~1500 vars (~6400 clauses).** Below that, the ~17μs GPU launch floor (upload `x` + kernel dispatch + download partial sums) dominates. Above 2000 vars, GPU wins decisively: 4.3x at 2K, 14x at 5K. GPU time barely grows (19.9→21.9μs from 2K→5K) while CPU scales linearly (85→306μs). At industrial SAT sizes (100K+ clauses), expect 100x+. The 17μs floor is split roughly: ~8μs cudarc launch overhead, ~5μs `x` upload (num_vars × 8 bytes), ~4μs partial sum download. For the gradient loop (loss called every iteration), this means GPU is worth it only when the per-iteration compute dominates the per-iteration transfer — i.e., large clause counts.
