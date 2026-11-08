# EDP2_1161 compositional proof — a checking-time record attempt. Pre-registration

*Written 2026-11-08, before any record measurement. Owner: "i want us to find and
break a speed record" → "go". Venue: the local RTX 4000 SFF Ada (compute-only,
allowed by default); RunPod only when the owner names it. Survey that chose the
target: `speed-records-survey-2026-11-08.md` (subagent-written; the numbers used
below were re-read from the STTT PDF before being written here).*

## 0. The target, and what "record" can honestly mean

Tan, Heule & Myreen, *Verified Propagation Redundancy and Compositional UNSAT
Checking in CakeML*, STTT 25(2), 2023, §6.4 and Table 6: the Erdős discrepancy
instance EDP2_1161 (`p cnf 4207 25142`) as a **5226-line compositional proof**,
each line an LRAT-justified lemma (5225 cube lemmas from march_cu + iglucose +
drat-trim, then the empty clause). cake_lpr checked it in **3871.7 s on 1 thread**
and **60.2 s on 128 threads** (Xeon E5-2695 v3, 128 GB). Underlying proofs: "20
million clause addition lines", "88 bytes to 110 MB".

There is no official table of proof-checking records. The claim this arc can make
is narrow and must be stated with all four caveats every time:

1. **Different proofs or a different pipeline.** Path A reproduces their pipeline
   (march_cu → iglucose → drat-trim -L; needs the external CnC build — owner's
   call, the classifier refused it). Path B derives each step's formula from the
   published `.comp` lemmas and lets our own solver prove it. Path B proves the
   same theorem through the same 5226 lemmas, but the underlying proofs are ours.
2. **The published time includes I/O.** cake_lpr re-parsed the 437 KB CNF 5226
   times and ran as 5226 processes. Part of any win is parsing once. We report
   the split (read / decode / check) so the reader can subtract it.
3. **Older hardware and an older cake_lpr.** The published number is cited beside
   same-machine reruns of cake_lpr and lrat-check on the same proofs, never
   instead of them.
4. **cake_lpr is a verified checker; ours is not.** The comparison is wall-clock,
   not assurance. Our checker is RUP-only; any RAT step is UNSUPPORTED (path A
   must show zero of them).

## 1. Pilot (path B, 7 of 5226 steps, 2026-11-08, before this document)

Step formula = CNF + one unit per literal of the negated lemma (last step: CNF +
all 5225 lemmas). `solve --lrat` (text), our CPU checker, lrat-check:

| step | solve | proof bytes | ours (CPU, 1 thr) | lrat-check |
|---|---|---|---|---|
| 1 | 8 ms | 33,699 | 6 ms | 14 ms |
| 2 | 6 ms | 1,740 | 5 ms | 18 ms |
| 3 | 10 ms | 103,724 | 6 ms | 19 ms |
| 1000 | 78 ms | 1,624,583 | 25 ms | 36 ms |
| 3000 | 21 ms | 372,495 | 9 ms | 22 ms |
| 5225 | 153 ms | 3,309,275 | 45 ms | 55 ms |
| 5226 | 79 ms | 984,103 | 22 ms | 35 ms |

All seven VERIFIED by both. Times include process start. Lemma lengths in the
`.comp`: 7–25 literals, mode 14. Full path-B generation (5226 steps, 6 niced
cores) was started right after the pilot; its totals are appended in §5.

## 2. What will be built

**B1 — the per-step reference.** A script that, for every step j, builds the
formula and runs our sequential CPU checker, lrat-check and cake_lpr on the same
`(cnf_j, proof_j)`; records three verdicts and three times. This is the
correctness oracle and the same-machine baseline.

**B2 — the batched compositional check.** One process: the CNF parsed once; the
5226 proofs read (binary LRAT) and decoded with the existing parallel/device
decoder; each step checked against *its* formula (CNF ∪ units_j, or CNF ∪ all
lemmas for step 5226) with per-step clause-id spaces laid out side by side in one
device pass, so the fixed launch/upload cost is paid once, not 5226 times. The
verdict is VERIFIED only if every step is. The SimWarp model is the contract, as
in the decode arc: `batched(steps) == [sequential(cnf_j, proof_j) for j]` on
every step and on every mutant.

**B3 — falsifiers before any timing is quoted.** 50 random steps × {one hint
replaced by a random clause id, the empty clause removed, one lemma literal
flipped}: B2 must REJECT each (naming the step) and lrat-check must reject the
same mutant. A mutant any checker accepts kills the run until explained.

## 3. Predictions (ranges fixed now; ledger in §5)

- **P1 (correctness).** All 5226 steps VERIFIED by ours = lrat-check = cake_lpr.
  Any disagreement, or any UNSUPPORTED step, is a KILL for the record claim.
- **P2 (same-machine cake_lpr, sequential, 1 process per step, our proofs).**
  400–1600 s total. Their proofs took 0.74 s/step; ours come from a different
  solver, so the range is wide on purpose.
- **P3 (same-machine lrat-check, sequential).** 100–300 s total (pilot 14–55 ms
  per step including process start).
- **P4 (B2, cold, bytes → verdict for all 5226 steps, CNF parsed once).**
  ≤ 5 s. The claim is only interesting if B2 is ≥ 10× under the published 60.2 s
  *and* ≥ 20× under same-machine lrat-check; otherwise the result is "a batched
  CPU checker would do as well" and the GPU is not the story.
- **P5 (where the time goes).** The device check stage < 30 % of B2's total; file
  reads + decode dominate. The write-up leads with the split, not the ratio.
- **P6 (path A, if the owner allows the build).** The regenerated cubes equal the
  published `.comp` lemmas line for line (deterministic tools); zero RAT steps in
  the 5226 drat-trim proofs; B2 on those proofs within 2× of B2 on ours.

## 4. Protocol

- Same machine for every number; CPU tools on 6 niced cores (`taskset 0-5`, a live
  stack runs here) — stated beside every CPU figure, since the published figure
  used up to 128 threads. B2 uses the card plus ≤ 6 host threads.
- Cold first, then best-of-3, both reported (the 11-04 lesson: best-of-3 hides
  first-touch page faults).
- Nothing is quoted until B3's mutants have all been rejected.
- The published 3871.7 s / 60.2 s appear in the same table as our reruns, with
  hardware and thread counts, and the four caveats of §0 in the same paragraph as
  any "fastest" sentence. Cross-vendor panel before anything leaves the repo.

## 5. Ledger (appended as measured; corrections in brackets)

- 2026-11-08 — **Path B generation done**: 5226/5226 steps UNSAT (`solve --lrat`,
  text), 2070.5 s of solver time on 6 niced cores, hardest step 53.7 s (step 3767),
  **25.31 GB of untrimmed text LRAT**, largest 707 MB (step 3696). That is far
  more than the published pipeline's "20 million additions / up to 110 MB" because
  drat-trim's `-L` output is trimmed and ours is the solver's raw log. Decision,
  before any checking is timed: every path-B proof is trimmed with `lrat-trim`
  (binary for the device and cake_lpr, text for lrat-check), which is this crate's
  standard chain and the like-for-like object to a drat-trim LRAT. The untrimmed
  sizes are kept here as the record of what the solver emitted. Path A (owner runs
  the external tools; `pathA/run.sh`) not yet started at this point.
- 2026-11-08 — **Trimmed**: 5226/5226 with `lrat-trim`, 0 errors, 647 s on 6 cores;
  25.31 GB → **6.894 GB binary / 14.014 GB text**. These are the objects every
  checker below is timed on.
- 2026-11-08 — **B2 built** (`lrat_batch.rs`: shared CNF once, per-step private
  originals incl. copies of any deleted shared clause, lemmas in step order, hints
  remapped; verdicts from each step's own flat proof; 4 contract tests incl.
  placeholder steps and a deleting step that must not leak) and **first full run**
  (`examples/edp_batch.rs`, run 1, cache warm from the trim, 5 host threads on
  cores 1–5 + the Ada, 2 chunks ≤ 6 GB est.): **5226/5226 VERIFIED, 0
  unsupported, batch codes = one-at-a-time SimWarp codes on 50 evenly spaced
  steps.** Volume: 25,224,312 lemmas, 1,999,078,832 hints. **Total 27.363 s** =
  cnf+comp 0.005 · read proofs 1.126 · decode 8.625 · batch layout 11.461 ·
  device check 5.332 (incl. upload) · combine 0.056 · (CUDA init ≈ 0.3, not yet
  timed separately). **P1 MET** on our proofs (lrat-check/cake_lpr per step still
  running, B1). **P4 MISSED** on this run: 27.4 s against ≤ 5 s — under the
  published 60.2 s / 128 threads, but the layout is a single-threaded ≈ 12 GB
  copy and the decode runs on 5 threads; both are host work, not checking.
  **P5 MET**: check 19 % of total. Next: parallel layout, then re-measure (cold
  first, then best-of-3) before any number is quoted outside this file.
- 2026-11-08 — **Parallel layout** (`batch_par`, 5 workers; equal to `batch` array
  for array, tested at 1/2/3/7 threads) and the **measured runs** (B1's pass 1
  still running on core 0 — a confound, stated). Cache evicted before run 2 by
  streaming the 26 GB of untrimmed text (10 s). Each run 5226/5226 VERIFIED.
  | run | cache | read | decode | layout | device check | CUDA init | total |
  |---|---|---|---|---|---|---|---|
  | 2 | cold | 1.834 | 5.150 | 2.772 | 5.324 | 0.201 | **15.707 s** |
  | 3 | warm | 0.705 | 8.027 | 4.040 | 5.348 | 0.208 | 18.806 s |
  | 4 | warm | 0.912 | 8.139 | 3.975 | 5.386 | 0.201 | 19.075 s |
  The device stage is steady within 1 %; the host phases (decode, layout) vary by
  60 % and the cold run is the fastest — unexplained; suspected page-cache reclaim
  against ≈ 20 GB of fresh allocations, to be re-run after B1 finishes. **P4
  MISSED** (15.7–19.1 s vs ≤ 5 s). Against the published figure: 3.2–3.8× under
  60.2 s (128 threads), 200× under 3871.7 s (1 thread) — short of the ≥ 10× bar
  set in §3 for "the GPU is the story". Where the time is: host decode 5–8 s (6.9
  GB at ≈ 1 GB/s on 5 threads) + layout 3–4 s + the check's ≈ 10 GB upload; the
  kernel itself scales with the 2.0 G hints as the decode arc measured. The
  remaining route to P4 is the batched device-resident decode (bytes up once,
  decode + check on the card), which the decode arc built for single proofs.
- 2026-11-08 — **B1 done** (`ref/run.sh`: one process per step, one niced core,
  trimmed proofs; text for lrat-check, binary for cake_lpr and ours): **ours
  (sequential + independent, 1 thread) 665.7 s · lrat-check 199.6 s · cake_lpr
  366.9 s; 5226/5226 VERIFIED by all three.** Hardest step for every tool is
  3696 (ours 12.5 s, lrat-check 1.55 s, cake_lpr 3.24 s). **P1 MET** (three-way
  agreement, 0 unsupported). **P2 MISSED low** (cake_lpr 366.9 s against the
  400–1600 s range — our proofs are easier for it than theirs were). **P3 MET**
  (199.6 s in 100–300). Ratios to B2's cold 15.707 s: lrat-check 12.7×, cake_lpr
  23.4×, ours 42.4×, published 128-thread 3.8×. **§3's bar is not met on either
  clause** (≥ 10× under 60.2 s; ≥ 20× under same-machine lrat-check); the
  honest sentence today is "one workstation with a 20 GB GPU checks the
  compositional EDP proof in 15.7 s, where the published run took 60.2 s on 128
  threads and lrat-check takes 199.6 s on this machine one step at a time" —
  with §0's four caveats attached. Options recorded for the owner: push the
  batched device-resident decode for P4, or write up here.

