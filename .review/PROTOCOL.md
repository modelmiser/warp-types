# Cold Review Protocol

Methodology for parallel cold-agent code review. Each version reflects
lessons from prior rounds — `git log .review/` shows the evolution.

## Current: v2 (2026-03-17)

3 parallel cold agents with exclusive file assignments, code correctness only.

### Setup

1. Each agent reads `lib.rs` first (public API + types)
2. Project-specific briefing (e.g., CPU shuffles are identity)
3. Skip: style, docs wording, naming, "could be more idiomatic"

### File Assignment

Balance by **complexity**, not file count:
- Agent 1 (CORE): unsafe, asm, FFI, critical invariants
- Agent 2 (TYPE SYSTEM): traits, type-level encoding, macros
- Agent 3 (SURFACE): examples, tests, research, build tooling

### Known Patterns

List already-fixed bug categories so agents focus on NEW issues.
Prevents ~30% noise from re-discovery in later rounds.

### SimWarp Coverage

List algorithms that `simwarp.rs` validates with real lane exchange.
Reviewers skip re-verifying these shuffle sequences and focus on
code NOT covered by SimWarp.

### Review Protocol (per file)

1. Read end-to-end
2. Standard check: arithmetic, off-by-one, state machine, invariant
3. **ADVERSARIAL CHECK**: For each public safe function, construct a
   safe-code call sequence that violates its documented invariant
4. **CALLER-INVARIANT CHECK**: Grep all callers, verify each caller's
   assumed invariant is guaranteed by the callee in all code paths
5. **CROSS-FILE INVARIANT**: List invariants your files EXPORT that
   files outside your assignment depend on
6. **TEST COVERAGE**: Classify untested paths as:
   - UNTESTED-HARD: cfg-gated or requires hardware
   - UNTESTED-SOFT: could have a test but doesn't

### Severity

- **CRITICAL**: soundness hole, silent wrong results, UB in safe code
- **MODERATE**: panic on valid input, wrong result in edge case
- **LOW**: doc/code mismatch, unreachable panic, defensive check missing

### Reporting

```
CLEAN: filename.rs
BUG: filename.rs:line [CRITICAL|MODERATE|LOW] — summary
  Evidence (2-3 lines) + affected callers
UNTESTED-SOFT: filename.rs:line — description
UNTESTED-HARD: filename.rs:line — description (why untestable)
EXPORT: invariant description → consuming files
```

---

## Evolution

### v1 (rounds 1-12)

Original protocol. Flat UNTESTED category, no severity, no cross-file
tracking, no known-pattern filtering.

**Findings by round type:**
- Rounds 1-6: code bugs (wrong arithmetic, missing guards, unsound traits)
- Rounds 7-9: doc/CI gaps (stale counts, missing workspace members)
- Round 10: cfg-gated PTX and dynamic-mirror bugs
- Round 11: CPU-identity-shuffle-masked GPU bugs (sort, tile, scan)
- Round 12: bitonic sort stage_mask (correct first substage masked the bug)

### v1 → v2 changes (after round 13)

| v1 Problem | v2 Fix |
|---|---|
| 47 UNTESTED was noisy | UNTESTED-HARD vs UNTESTED-SOFT |
| No severity levels | CRITICAL / MODERATE / LOW |
| Balanced by file count | Balance by complexity |
| No cross-module tracking | EXPORT invariants per agent |
| Re-flagged fixed issues | KNOWN PATTERNS briefing |
| UNCERTAIN was ambiguous | Removed — require concrete trigger or LOW |
| DOC-ONLY separate category | Folded into LOW severity |

### SimWarp addition (after round 13)

SimWarp (`src/simwarp.rs`) was created to close the structural testing
gap identified across rounds 10-12: CPU identity shuffles hide bugs
that only manifest when two distinct lanes exchange real values.

SimWarp holds `[T; WIDTH]` and implements actual GPU shuffle semantics.
Tests validate algorithm correctness with real data exchange. Algorithms
covered by SimWarp tests don't need re-verification by cold reviewers.

**Covered algorithms:**
- Butterfly reduce (5-step XOR) — `butterfly_reduce`
- Bitonic sort (15-step CAS with direction) — `bitonic_sort`
- Tile-confined shuffle (width parameter) — `tile_shuffle_xor_confined`
- Tile reduce (per-tile butterfly) — `tile_reduce_sum_per_tile`
- Hillis-Steele scan (correct + demonstrates bug) — `hillis_steele_*`
- 64-lane AMD reduce (6-step XOR) — `simwarp_64_lane_reduce`

### Key lessons

- Agent wall clock is dominated by the *deepest* file, not the most files
- Cross-cutting bugs need explicit EXPORT tracking — single-module review misses them
- The adversarial check and caller-invariant check are the two highest-yield steps
- Finding type shifts over rounds: code bugs → doc gaps → CI gaps → infrastructure-masked bugs
- Convergence signal: when no new bug *category* emerges, fix known bugs rather than run more rounds
