# GPU Partition of SAT Work — Pre-registration

**Date:** 2026-11-01
**Status:** Pre-registered, NOT run. Append-only: later corrections go in dated `[brackets]`, never by editing the registered text.
**Context:** Grew out of the reading note `~/Claude/books/hatt/CPU+GPU Hybrid Solving for SAT SMT.md` (sections "Can the work be split into finer GPU units?" and "Could type theory change these limits?"). That note's 30-minute plan — gradient confidence warm-starting CDCL — turned out to be already built and benchmarked here as `warp-types-sat/src/gradient.rs::hybrid_solve` (April 2026). This file registers what is genuinely *not* yet built, and what would falsify each piece, before any of it runs.

---

## 0. The limit that frames everything

Unit propagation is P-complete (Horn-SAT; Jones & Laaser 1976 via Dowling & Gallier), so one hard instance does not decompose into many independent GPU tasks at the CDCL level, and resolution proofs of PHP are superpolynomial (Haken 1985). Nothing below tries to beat that. Every experiment here is about **what happens around the sequential core**: choosing cubes, running many small instances, or warm-starting.

Known in-repo datum that already bears on this: `hybrid_solve` on UNSAT PHP(6) is 65× faster than pure CDCL with a tuned config but *loses* at PHP(4–5) with defaults, and fewer gradient iterations give better UNSAT warm-starts (README, benches 2026-04-16). So gradient confidence demonstrably helps an UNSAT proof — by steering CDCL, not by proving anything itself.

## 1. Three experiments, in gating order

| Id | Question | Touches | Gated by |
|---|---|---|---|
| E0 | Does a real verification workload consist of many small independent queries? | nothing in this repo (measurement only) | — |
| E1 | Do cubes on the *least-confident* gradient variables cut CDCL work on UNSAT? | `warp-types-sat` (`hybrid_solve` extension) | — |
| E2 | Can a warp solve ~32 small instances in lockstep with useful throughput? | `warp-types-sat` (`clause_tile`, `gpu` feature) | E0 passing |

E0 and E1 are independent and can run in either order. E2 does not start unless E0 passes.

## 2. E0 — workload shape (measurement only, CPU only)

**Hypothesis H0:** in a Kani run over one real crate, ≥ 50 % of total SAT solve time is spent in queries that each take < 10 ms.
**Null, derived mechanically (not "H0 is false"):** the 5 longest queries account for ≥ 80 % of total solve time — one hard query dominates.
**Neither:** report the shape (histogram of time vs size) and stop; no prototype.

**Workloads (two, because they have opposite expected shapes):**
- Kani on one Rust crate (owner to name it). CBMC emits roughly one query per harness, so the batch dimension is harness count.
- KLEE on a small C program. Thousands of tiny path-constraint queries.

**Recorded per query:** variables, clauses, solve wall time, result. Tool flags are NOT yet confirmed and must be read from the tools' docs before running — nothing here is to be run from memory of flag names.

**Control that can fail:** a single-harness CBMC run with path merging must classify as *null* under the same rule. If the rule cannot produce "null" on a workload known to be one monolithic query, the rule is broken, not the workload.

**Prior-art step first (30 min, before measuring):** check whether this distribution is already published — Palikareva & Cadar, "Multi-solver support in symbolic execution" (CAV 2013) characterised KLEE's query mix; SMT-COMP incremental-track analyses. If a published distribution answers H0, skip the measurement and record the citation here.

**Decision rule:** H0 → E2 unlocked. Null → E2 dead; record and move on. Neither → stop.

## 3. E1 — cubes on least-confident variables

**What exists:** `hybrid_solve` runs gradient search, then warm-starts VSIDS (activity = confidence `|x − 0.5|`, phase = polarity) and runs one CDCL solve. It never splits.
**What E1 adds:** after gradient search, pick the `d` variables with *lowest* confidence, form the `2^d` cubes, solve each cube with the warm-started CDCL, declare UNSAT only when all cubes are UNSAT (GaloisSAT's scheme; its own UNSAT gain was only 1.29×, so the prior here is "small or none").

**Primary metric:** total CDCL conflicts summed over cubes (deterministic; independent of parallelism and machine noise).
**Secondary:** single-thread wall time, same machine class as the April benches (isolcpus E-core; `cargo bench -p warp-types-sat`).
**Run sequentially on CPU first.** Parallelism across cubes is a separate, later claim; mixing it in would hide whether the *split* or the *parallel hardware* did the work.

**Instances (fixed now):** PHP(4), PHP(5), PHP(6), PHP(7); parity k = 10, 20, 40; random 3-SAT at ratio 4.26, n ∈ {100, 150, 200}, 20 seeds each, SAT and UNSAT kept separate.
**Parameters (fixed now):** d ∈ {3, 5, 7}; gradient config = the README's tuned `num_starts=8, max_iters=100` AND the library default — both, because the April result showed config dominates.
**Baselines:** pure CDCL; `hybrid_solve` (no cubes), both configs.

**Hypothesis H1:** on UNSAT instances, total conflicts with least-confident cubes at the best d is ≤ 0.7 × the no-cube hybrid baseline.
**Null:** total conflicts ≥ 0.9 × baseline at every d (splitting adds work; each cube re-derives shared lemmas).
**Kill condition:** on PHP(6), where warm-start already gives 65×, cubes *increase* total conflicts → cubes hurt where confidence is already good; stop E1.

**Controls that can fail (all three run, all three must behave):**
1. **Random cubes** — same d, variables chosen uniformly at random. If random cubes match least-confident cubes, confidence is not doing the work; H1 is not supported even if both beat baseline.
2. **Most-confident cubes** — the sign flip. Expected to be worse than least-confident. If it is *better*, the confidence signal's sign is wrong for cube selection and the write-up must say so (sign bugs hide as "symbol swap" requests — replay before believing).
3. **Lemma sharing off** — cubes solved with fresh clause databases. This is the honest E1; sharing learned clauses across cubes is a second mechanism and gets its own line if ever added.

**Error-sign ledger:** every disagreement between prediction and outcome gets logged with its sign. If every error points toward H1, the instrument is tuned, not the hypothesis confirmed.

**Done means:** a bench under `warp-types-sat/benches/`, numbers in a table in this file (appended, dated), and a verdict per instance class: SUPPORTED / NULL / KILLED / INCONCLUSIVE.

## 4. E2 — instances across lanes (gated by E0)

**What exists:** `clause_tile.rs` packs clauses across a warp's lanes and `bcp.rs:528` evaluates them in batch — batching *within* one instance. The README states the crate is "not a parallel solver".
**What E2 adds:** one small instance per lane (or per lane group), lockstep BCP over the batch, divergence measured directly.

**Hypothesis H2:** for a batch of 32 instances each ≤ 200 variables (size bound to be reset from E0's measured distribution, recorded here before running), batched GPU throughput ≥ 4× sequential CPU `solve` on the same batch.
**Null:** < 1.5× — divergence eats the lockstep gain.
**Measured, not assumed:** fraction of warp cycles with < 50 % lanes active (the divergence cost), reported alongside throughput. A speedup number without the divergence number is not a result.
**Where the parent library does load-bearing work:** the batch's reductions and ballots must be warp-uniform across instances that are at different search depths — exactly the `Warp<S>` discipline. If E2 cannot be written in the typed fragment, that itself is a finding for the paper's limitations section.

**Hardware:** GPU runs go to RunPod, not the live machine (shared GPU with xray). CPU baselines on the isolcpus core as in April.

## 5. Explicitly out of scope here

- Proof output (DRAT/LRAT) from `warp-types-sat` — absent today; needed before any "certified" claim, tracked as its own TODO item, not an experiment.
- GPU lookahead for cube selection (march-style) — the conventional alternative to gradient confidence; Schuster 2014 judged it GPU-suitable, no benchmarked port found. Only relevant if E1 shows the *split* helps but confidence is not the right selector (control 1 fails).
- Parallel cube solving, lemma sharing across cubes, and any CDCL-core speedups.

## 6. Compute and publication policy

- CPU benches: niced, ≤ 6 cores, isolcpus core as before. No batch work on cores 16–18.
- GPU: RunPod only.
- This repo is public. Nothing in this file is a soundness claim, but any E1/E2 write-up that asserts "cannot" / "always" / "safe" gets a cross-vendor review before push. Pushing is the owner's call.

---

## Log (append below; never edit above)

- 2026-11-01 — registered. Nothing run.
