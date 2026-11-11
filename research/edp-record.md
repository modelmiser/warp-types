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
- 2026-11-08 — **B2 re-run with core 0 idle** (B1 finished). Runs 5–7, 5 threads
  on cores 1–5: **13.999 / 14.420 / 14.457 s** (decode 4.87–5.26, layout 2.45–2.47,
  check 5.32–5.36); run 8, 6 threads on cores 0–5: **13.670 s** (decode 4.80,
  layout 2.16, check 5.39). All 5226/5226. The 18–19 s of runs 3–4 was the
  per-step reference run sharing the socket, not cache state — confound
  resolved. Run 5 was meant to be cold but its 0.709 s read says the eviction did
  not take (the 26 GB stream was itself cached), so this set has **no clean cold
  number**; run 2's 15.707 s stays the only cold one. Best-of-3 at 5 threads
  14.0 s; best overall 13.7 s (6 threads) = 4.4× under 60.2 s, 14.6× under
  same-machine lrat-check, 26.8× under cake_lpr, 48.7× under our per-step CPU
  route. §3's bar still unmet on both clauses; P4 still MISSED.


## 6. C — the batched device-resident route (pre-registered 2026-11-08, before code)

**Why.** B2's 13.7 s is 7 s of host work (decode 4.8 + layout 2.2) and 5.4 s
of device check, of which ≈ 2 s is the upload of the 10 GB of gathered
literals and hints. The decode arc built a device-resident route for ONE proof
(bytes up once; scan, gather and check on the card; the host sees step records
only). C runs that route over many proofs per launch.

**Measured before any code** (the single-proof resident route on the two largest
EDP steps, `lrat_check` with `LRAT_GPU_RESIDENT=1`, best of 3, hashed table
since 4207 variables exceed the direct layout's 2047):

| step | bytes | hints | upload | pass 1 + 2 | plan | gather | check (resident) | check with upload | host decode (6 thr) |
|---|---|---|---|---|---|---|---|---|---|
| 3696 | 85.8 MB | 23.9 M | 0.013 | 0.022 | 0.005 | 0.004 | 0.035 | 0.060 | 0.077 |
| 1646 | 58.9 MB | 17.3 M | 0.005 | 0.011 | 0.006 | 0.003 | 0.023 | 0.043 | 0.050 |

Per-byte / per-hint rates from step 3696, scaled to the full set (6.894 GB,
2.0 G hints): upload ≈ 1.0 s · scan passes ≈ 1.8 s · gather ≈ 0.3 s · check
kernel ≈ 2.9 s (the 5.3 s B2 check stage was ≈ 2.9 s kernel + ≈ 2.1 s upload
+ meta, consistent). **The device kernels alone sum to ≈ 5.0 s at these
rates; with the bytes upload ≈ 6.0 s.** So, written down before building:
**P4 (≤ 5 s) is predicted MISSED by C on this card**, and the §3 10× clause
(≤ 6.0 s) is predicted missed by the serial design. What C can win is the 7 s
of host work minus what replaces it.

**Design C1 (serial; no kernel changes).** Per chunk of steps: (1) one host
buffer holds every step's bytes, each step front-padded with 0xFF…0x00 so it
ENDS on a 4096-byte slice boundary (a 0xFF run is never a step start; the 0
before the real first byte lets the scan find it); (2) the existing two-pass
scan (`decode_lrat_slices`, resident) runs once over the buffer; (3) per step,
on host threads, its slices' records are rebuilt with positions made relative
and fed to the existing `stitch_plan` against placeholders + the step's extras
— a step goes the **host route** (`from_binary`, the proven decoder) if any of
its slices is dirty or carries an error, if any chain end passes the step's
end, or if the plan reports a structural verdict; (4) one batched layout is
sized and filled on host threads: per step a region of clause_start /
deleted_at holding the CNF's clause starts (the literals shared once at the
front), one unreferenced sentinel entry so the last CNF clause's end is right,
then the extras and lemmas; lemma arrays with offsets; per step a dense table
remapped to batch indices; (5) literals gathered by one launch of
`lrat_gather_lits`, hints by one launch of `lrat_gather_hints` PER STEP over
views of chunk-wide arrays (its dense table is per step — the one reason for
per-step launches), host-route steps copied in directly; a hint naming an
unknown id sends that step to the host route and the chunk's layout is redone;
(6) the raw arrays are freed and `check_lrat_codes_mode` runs once over the
chunk's meta; (7) verdicts per step from each step's own plan or flat proof
(`combine`, the hint id of a Deleted/Forward reason fetched from the device as
`combine_resident` does). Chunks are cut by an estimated device footprint.

**Oracles.** O1: the per-lemma codes of EVERY step equal B2's
(`batch_par` + `check_lrat_codes`) on the full EDP run — a `--verify-b2` mode
of the new example runs both and compares all 25,224,312 codes. O2: codes of
50 evenly spaced steps equal the one-at-a-time host decode against the real
formula + SimWarp (the existing contract check). O3, gpu tests: synthetic
multi-step batches (the `lrat_batch` shapes: shared-clause deletion, the
mutants, many extras) plus a truncated step, a bit-flipped step, a step with a
hint naming an unknown id, a step longer than one slice, and a padded first
step — each step's verdict and codes equal to `from_binary` + `check_warp_codes`
+ `combine`; the host route taken exactly by the malformed steps. Watched to
fail before being believed: a mutant dropping the sentinel shift (private
dense `d` mapped without the +1) and a mutant dropping the chain-end rule.

**Predictions** (same protocol as §4: cold first, then best of 3; 6 host
threads on cores 0–5, niced; the card otherwise idle):
- P7: C1 total (warm cache) **≤ 9.5 s**, expected 7.5–9.5 s; cold ≤ 11 s.
- P8: the three device stages (scan, gather, check) are ≥ 55 % of C1's total.
- P9: zero steps take the host route on the 5226 real proofs; zero bad hints.
- P10: O1 and O2 report zero mismatches; O3's mutants are each killed.
- P4 restated: MISSED (floor ≈ 5.0 s of kernels + 1.0 s upload). §3's 10× clause
  (≤ 6.0 s): MISSED by C1.
- C2 (optional, only if C1 lands inside P7): pinned host buffers and a second
  stream so chunk k+1's upload and scan run under chunk k's check; predicted
  total ≤ 6.5 s. Not built until C1 is measured.

**Kill.** Any O1/O2 mismatch that is not a bug found, fixed and re-run with a
clean result; or a C1 total not under B2's 13.7 s (then the route is recorded
as refuted on this card and the write-up uses 13.7 s).

- 2026-11-08 — **C1 built and measured** (`lrat_warp::batch_device`, `examples/edp_resident`;
  commits 942228129, 4c33de751). Built as pre-registered with two deviations found on
  the first runs, both recorded here: (i) the "chain ran past the step's end" rule as
  written would have sent 8 of the first 300 well-formed steps to the host route — a
  mis-synchronised speculative chain in a step's LAST slice legitimately runs into the
  next step's padding (the single-proof route bounds it by the proof's length); such a
  slice is now re-parsed on the host within the step's bytes, exactly as a dirty slice
  is, and the stitch decides; (ii) the host staging buffer (2.5 GB of 0xFF per chunk,
  first-touch) cost 2.6 s, so each proof is copied straight into its place in a device
  buffer and the scan runs over that (`decode_lrat_slices_device`). One field slip
  caught by the oracle run: the device's `st_after` is unused (0) for `d` steps and my
  "record before the step's start" guard read it — every deletion step took the host
  route until fixed. **Oracles: O1** every one of the 25,224,312 codes of all 5226 steps
  equal to B2's (`--verify-b2`, three chunks, 0 mismatches); **O2** 50 steps equal to
  one-at-a-time SimWarp, 0 mismatches; **O3** three gpu tests pass (the `lrat_batch`
  shapes; malformed / empty / RAT / bad-deletion / id-order / bad-hint / flipped-byte /
  every truncation of a proof, each on the right route; long proofs across 13-byte
  slices and the padding, both scan kernels). **Watched to fail** (mutants on a copy,
  restored by `cp`): M1 sentinel shift dropped → 2 of 3 tests fail; M2 overrun rule
  dropped (`overrun = false`) → 2 of 3 fail (the truncation sweep reaches it); M3
  host-route hint copy dropped → 1 of 3 fails. Full gpu suite on the restored tree with
  the four archive dirs: 289 passed, 1 failed = `archive_agreement_with_external_checkers`
  demanding `LRAT_CHECK` (harness configuration); rerun with `LRAT_CHECK` = drat-trim's lrat-check and `CAKE_LPR` set: 632/632 agree, 147 verified, pass — the suite is clean on the committed tree.
  **Measured** (6 host threads on cores 0–5, niced, the card otherwise idle; cold =
  `posix_fadvise(DONTNEED)` on all 5226 files, confirmed by the read time):
  | run | cache | read | upload | scan (p1+p2 / dl) | plan | layout | gather | check | total |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | warm | 0.717 | 0.979 | 1.878 (1.425 / 0.421) | 0.505 | 0.401 | 0.340 | 3.669 | 8.996 s |
  | 2 | **cold** | 1.824 | 0.983 | 1.913 (1.419 / 0.459) | 0.523 | 0.432 | 0.342 | 3.708 | **10.300 s** |
  | 3 | warm | 0.618 | 0.975 | 1.880 (1.432 / 0.417) | 0.498 | 0.393 | 0.339 | 3.706 | **8.904 s** |
  | 4 | warm | 0.627 | 0.984 | 1.928 (1.436 / 0.456) | 0.527 | 0.423 | 0.341 | 3.752 | 9.068 s |
  | 5 | warm | 0.640 | 0.981 | 1.905 (1.420 / 0.450) | 0.518 | 0.419 | 0.341 | 3.760 | 9.031 s |
  (CUDA init 0.11–0.20 and combine 0.07 not shown; ≈ 0.3 s per run is allocation and
  freeing between stages.) Three chunks (2.47 / 2.59 / 1.84 GB of bytes); device peak
  10.15 GB of 19.2 (sampled at 100 ms); 0 host-route steps; 14 dirty slices of 1.69 M
  (the chunk's first slice and 11 false starts). **Best of 3 (runs 3–5): 8.90 s; cold
  10.30 s.**
  **Ledger.** P7 **MET** (8.90 s ≤ 9.5; cold 10.30 ≤ 11). P8 **MET** (scan passes 1.43 +
  gather 0.34 + check 3.71 = 5.48 s = 62 % of 8.90). P9 **MET** (0 host-route steps, 0
  bad hints). P10 **MET** (O1, O2 zero mismatches; three mutants killed). P4 **MISSED as
  predicted** (floor stated before building: the kernels alone are 5.5 s here; the check
  kernel is 3.7 s on the hashed layout, 2.0 G hints at 0.54 G/s). Against §3's bar:
  **≥ 20× under same-machine lrat-check MET** (199.6 / 8.90 = 22.4×); **≥ 10× under the
  published 60.2 s MISSED** (6.8×). Also 41× under same-machine cake_lpr (366.9 s), 75×
  under our per-step CPU route (665.7 s), 435× under the published single thread
  (3871.7 s). The honest sentence today: "one workstation with a 20 GB GPU checks the
  compositional EDP proof in 8.9 s (10.3 s cold), where the published run took 60.2 s on
  128 threads and lrat-check takes 199.6 s on this machine one step at a time" — with
  §0's four caveats attached, and nothing leaves the repo before the cross-vendor panel.
  **Where the time is now:** device 5.5 s (check 3.7, scan 1.4, gather 0.3) + upload 1.0
  + record download 0.45 + host plan/layout 0.9 + read 0.6 + init/alloc 0.5. **C2**
  (pinned buffers, a second stream, chunk k+1's upload and scan under chunk k's check)
  would overlap everything but the device sum: predicted 6.0–6.5 s, AT the 10× line, not
  under it. Under 5 s needs the check kernel faster — the direct table layout extended
  past 2047 variables (EDP has 4207), S5's 1.8× if its footprint still fits the cache —
  plus C2. Both are separate arcs, owner's call; neither is started.

## 7. Path A — the published pipeline rerun (owner-run, 2026-11-08)

- 13:38Z `pathA/run.sh` (march_cu `-e 0.4` exactly as `gen-cubes.sh`; CnC at 705b60c, the
  repo's only commit, 2025-07-14; iglucose from the same tree; drat-trim master 2e3b2dc):
  **14,721 cubes, not 5,225** (19.4 s, 120 refuted leaves). The first 12 lemmas equal the
  published ones as literal sets; the 13th differs (ours ends in −3028 where theirs has
  3035); and the trees are not one a deeper cut of the other: 2,288 of the 5,225 published
  lemmas are a subset of the negation of no cube of ours, 5,907 of our 14,721 cubes are
  covered by no published lemma (scratch `patha_cubes.py`). The 2023 march_cu is
  unrecoverable from the squashed repo. **P6 clause 1 (cubes = published) FAILED** — the
  tools are deterministic, the version is not. All 14,721 cubes UNSAT under iglucose (mean
  54 ms, 259 s on 6 cores); last step CNF + 14,721 lemmas UNSAT in ≈ 9 s, 4.95 MB DRAT.
- The drat-trim stage produced **14,722 empty LRATs, every step `s NOT VERIFIED`
  ("ERROR: no conflict")**: iglucose's per-cube DRAT ends with the lemma and has no empty
  clause (84 additions for cube 1, 0 deletions), and stock drat-trim — master 2e3b2dc
  (2024-11-25) and develop fb3fe74 (2023-07-09, the branch `setup.sh` checks out; built in
  `drat-trim-develop/`) alike — requires a conflict in backward mode; forward mode (`-f`)
  says `s DERIVATION` but writes a single deletion line as the LRAT; the undocumented `-B`
  mode bus-errors. How the 2023 LRATs (a 36-line trimmed derivation of lemma 25226 for
  step 1, `comproof/lrat/proof-1.lrat`) were produced is not reproducible from the
  published scripts as they stand today [unresolved; no claim made about the authors'
  setup].
- **What works and is equivalent:** append the negation of lemma j as unit clauses to step
  j's formula, after the previous lemmas so the ids match drat-trim's numbering. drat-trim
  then trims and verifies a refutation: step 1 → `s VERIFIED`, 24 of 85 lemmas in core
  (drat-trim's count; the published derivation keeps 25), 46 LRAT lines, a final empty
  clause hinted by the 13 units and the core. This is "CNF ∧ L₁..L_{j−1} ⊢ L_j" in
  refutation form — the form our checker already consumes (path B's step formula was CNF +
  the same units, without the previous lemmas).
- **Path A2 prepared** (`pathA2/run.sh`, owner-run; waits for path A's DONE line): the
  published lemmas fed to iglucose as cubes ¬L_j (same solver, same composition, 5,225 + 1
  steps), the emitted lemma recorded per step and classified against the published one
  (identical / strict subset / other), the last step CNF + 5,225 lemmas, drat-trim `-L` per
  step with the unit rule above. **Pre-registered before the run:** **P11** iglucose emits
  the published lemma itself for ≥ 90 % of steps and a strict subset for the rest, none
  "other". **P12** drat-trim verifies 5,226/5,226. **P13** our checker (resident route with
  cumulative extras: lemmas 1..j−1, then the units) verifies 5,226/5,226 with 0 UNSUPPORTED
  (zero RAT, as P6 asked) and 0 malformed. **P14** the drat-trim-trimmed proofs total fewer
  bytes than our lrat-trim'd ones (< 6.894 GB binary), and the resident route runs them in
  ≤ 11.6 s warm (1.3 × our 8.9 s): fewer bytes, but the per-step extras grow to 5,224
  clauses for the last steps (13.6 M extra clauses in all, a new cost for the layout
  stage), so a 1.0–1.3× band around our own time is the prediction, not a speed-up claim.
  P14 is P6's "within 2×" clause restated for the proofs that will actually exist.
- 13:51–14:01Z **Path A2 run by the owner** (`pathA2/run.sh`): 5,225 cubes from the
  published lemmas, all UNSAT under iglucose (mean 231 ms — 4× path A's per-cube time, these
  cubes carry fewer assumptions; 213 s on 6 cores; 5.34 GB of DRAT); last step CNF + 5,225
  lemmas UNSAT in 0.15 s. **Emitted lemma vs published: 1,811 identical, 3,414 strict
  subsets, 0 other.** **P11 MISSED** on its ≥ 90 % identical clause (34.7 %), met on "none
  other"; the composition everything below runs on is the published one with 3,414 lemmas
  strengthened to subsets. drat-trim `-L` with the unit rule: 5,226 LRATs, 13.52 GB text,
  mean 419 ms per step (6.5 min on 6 cores). `run.sh` printed "5226 not verified" — an
  artefact of its own grep: drat-trim prints `\rs VERIFIED` and the anchor `^s` never
  matches. Steps 1, 2600, 5225, 5226 re-run by hand: `s VERIFIED`, 0 RAT lemmas in core
  each (step 1: 23 of 68 lemmas in core; the published derivation keeps 25). Full CR-safe
  verdict re-run (`pathA2/verdicts.sh`) launched 14:44Z for P12 [result appended below].
- **Converted** with `lrat_check --to-binary` (our parser and writer): 5,226/5,226 in
  22.6 s, **6.611 GB binary** (ours: 6.894 GB) — P14's byte clause MET, by 4 %. Volume:
  19,974,053 lemmas, 1,791,914,187 hints (ours 25.2 M / 2.00 G).
- **Our checker on them** (`edp_resident --cumulative`: step j's formula = CNF + lemmas
  1..j−1 + the units of ¬L_j, 13.6 M extra clauses in all; outputs in `a2check/`).
  **run0 oracles: 5,226/5,226 VERIFIED, 0 UNSUPPORTED, 0 malformed, 0 host-route steps;
  `--verify-b2`: 5,226 steps compared, 0 code mismatches** (the B2 host route took
  36.6 min on these — it re-parses the cumulative extras per step); **`--simwarp 50`:
  0 mismatches. P13 MET.** Timed (6 niced host threads on cores 0–5; cold = fadvise
  DONTNEED on all 5,226 files, confirmed by read 1.75 s vs 0.62 warm):
  | run | cache | read | upload | scan (p1+p2 / dl) | plan | layout | gather | check | total |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | **cold** | 1.753 | 0.961 | 2.719 (1.463 / 1.221) | 0.970 | 0.466 | 0.424 | 3.215 | **11.816 s** |
  | 2 | warm | 0.620 | 0.967 | 2.869 (1.464 / 1.365) | 0.998 | 0.487 | 0.424 | 3.229 | **10.914 s** |
  | 3 | warm | 0.623 | 0.966 | 2.866 (1.458 / 1.369) | 0.996 | 0.487 | 0.423 | 3.275 | 10.973 s |
  | 4 | warm | 0.620 | 0.967 | 2.869 (1.466 / 1.365) | 0.994 | 0.485 | 0.423 | 3.278 | 10.960 s |
  (cnf+comp 0.56 s — building the 13.6 M extras — and init 0.18 s not shown.) **Best of 3:
  10.91 s warm; cold 11.82 s. P14 MET** (≤ 11.6 s predicted; 10.91 / 8.90 = 1.23×, inside
  the 1.0–1.3 band). Fewer bytes, lemmas and hints than ours, yet 2.0 s slower: plan 1.00
  vs 0.50, layout 0.49 vs 0.40, cnf+comp +0.5 — the per-step extras, as predicted — and scan
  download 1.37 vs 0.42 (drat-trim writes a deletion line after most lemmas, 22 for 24
  lemmas in step 1, so there are more records to bring down). The check kernel itself is
  faster (3.23 vs 3.71 s; 10 % fewer hints).
  **What this settles (P6 restated):** the published solver's proofs for the published
  composition, trimmed by the published checker, contain zero RAT steps, and our checker
  verifies every one of them, agreeing with the host route on all 19,974,053 lemma codes.
  **What it does not settle:** these are not the 2023 files (march_cu drifted; 3,414 lemmas
  are strict subsets; the derivation became a refutation through the unit rule), and the
  same-machine cake_lpr / lrat-check reruns of §5 were on OUR proofs — not yet re-timed on
  these [open; cheap].
- **Pre-registered before running `ref2/run.sh`** (same-machine lrat-check and cake_lpr on
  the A2 proofs, one niced core, one tool at a time, the per-step formula built outside the
  timed window — the §5 protocol): **P15** lrat-check total within 0.7–1.3× of its 199.6 s
  on our proofs, cake_lpr within 0.7–1.3× of its 366.9 s, both 5,226/5,226 verified. The
  resident route's ratio to each is then quoted beside the §6 ratios, not instead of them.
- 14:41–14:47Z `pathA2/verdicts.sh` (drat-trim without `-L`, CR-safe match, 6 niced cores):
  **5,226/5,226 `s VERIFIED`, 0 steps with RAT lemmas in core — P12 MET**; mean 348 ms per
  step, 1,819 s of core time. The published checker's own verdict on the proofs our checker
  was timed on, step by step.
- 14:47–15:00Z `ref2/run.sh`: **lrat-check 193.5 s (5,226/5,226 VERIFIED; 0.97× its 199.6 s
  on our proofs — P15 MET); cake_lpr 478.6 s (5,226/5,226 VERIFIED; 1.30× its 366.9 s —
  P15 MISSED by a hair, 1.3045 against a 1.3 bound).** Both tools now parse a growing
  formula per step (CNF + up to 5,225 lemmas + units), which our proofs never asked of
  them; the ratio is reported, not excused. **Resident route on the same proofs: 10.91 s →
  17.7× under lrat-check, 43.9× under cake_lpr.** The §3 "≥ 20× under same-machine
  lrat-check" clause therefore holds on our proofs (22.4×) and NOT on these (17.7×); both
  numbers go in the write-up side by side. **§7 ledger:** P11 MISSED (34.7 % identical),
  P12 MET, P13 MET, P14 MET, P15 half (lrat-check met, cake_lpr missed by 0.5 %); P6 clause 1
  FAILED (version drift), clause 2 (zero RAT) MET, clause 3 restated as P14 and MET.

## 8. C2 — two streams: chunk k+1 prepared under chunk k's check (2026-11-08)

Owner 08:50Z: "continue. direction is your call." Chosen: C2, the pre-registered
follow-on of §6, because the arc's missed clause is the 10× over the published 60.2 s
(≤ 6.02 s) and §6 predicted C2 "at the line". The wider direct table and the write-up
stay untouched. Written before any code.

**Where C1's 8.90 s goes** (§6, warm): device kernels 5.48 s (check 3.71 + scan passes
1.43 + gather 0.34) — everything else is host or PCIe work the card idles through: upload
0.98, scan-record download 0.42, plan 0.50, layout 0.39, read 0.62, init 0.11, combine 0.07,
allocation and freeing ≈ 0.3. Three chunks run strictly in sequence on the legacy default
stream.

**Design C2 (no kernel changes).** `check_chunk` is cut at step (5)/(6) into
`prepare_chunk` (upload, scan, plan, layout, gathers; ends with the bad-hint download,
which synchronises its stream, so every array it hands over is complete) and
`check_prepared` (the check launch, combine); `check_chunk` = the two in sequence, so the
three §6 gpu tests exercise the same code. `GpuContext::fork` gives a second context on a
new NON_BLOCKING stream sharing the loaded module (the legacy null stream would
synchronise with every blocking stream; `new_stream` streams do not). A library driver
`check_pipelined` runs three threads: reader (loads one chunk's proofs ahead, `threads`
workers), preparer (on the forked context), checker (on the caller's context), joined by
rendezvous channels, so at most one prepared chunk waits. Device memory handed across is
allocated on the preparer's stream and freed there (`cuMemFreeAsync`, stream-ordered) after
the checker's synchronous code download — so a chunk's arrays linger until the preparer's
stream reaches the free. The example's measured path IS the library driver (the loader is
the file read; the test's loader is an in-memory slice), with a per-chunk timeline
(read / prepare / check intervals from t0) beside the §6 stage columns. `--serial` keeps
the C1 path for comparison.

**Two decisions.** (1) **No pinned staging in this build**: the upload is hidden under the
previous chunk's check, and pinning gigabytes of host RAM on the live machine is a cost
with no predicted return; revisited only if the measured fill (time to the first check
launch) is upload-bound. (2) **Chunk budget 6 GB instead of 12** (≈ 6 chunks): two chunks
are in flight, so the fill halves and the footprint stays bounded — predicted peak ≈ 1.9 GB
of check arrays for the chunk being checked + ≈ 5 GB prepare peak + the lingering arrays of
the chunk before ≈ 9 GB.

**Predictions** (same protocol: oracles first, then cold via fadvise DONTNEED, then best
of 3 warm; 6 niced host threads on cores 0–5, now shared by the reader and the preparer;
the card otherwise idle):
- **P16** C2 warm best-of-3 **≤ 6.5 s**; point estimate **5.9 s** (5.4–6.4). Two regimes
  written down: if the scan/gather kernels merely time-slice with the check kernel the
  total is the device sum 5.5 + fill ≈ 0.4 + init ≈ 0.15 → 6.0–6.5; if they co-run (the
  hashed check launch is 6 blocks × 4 warps per SM = 24 of 48 warp slots, latency-bound)
  → 5.0–5.8. **P16b: the §3 10× clause (≤ 6.02 s) MET** — called narrowly, on the co-run
  regime. **P4 (≤ 5 s) still MISSED.**
- **P17** cold ≤ warm best + 0.7 s (only the first chunk's read is exposed).
- **P18** overlap is visible in the numbers: the per-chunk stage sum (read + upload + scan
  + plan + layout + gather + check + combine) exceeds the wall by ≥ 2.5 s, and the check
  stage sum grows from 3.71 s to 3.9–4.8 s under the co-running scan.
- **P19** oracles clean on the pipelined route: `--verify-b2` 0 code mismatches on all
  5,226 steps, `--simwarp 50` 0 mismatches, 5,226/5,226 VERIFIED, 0 host-route steps.
- **P20** the three §6 gpu tests pass unchanged; a new test runs the pipelined driver over
  the `lrat_batch` shapes and the malformed set in 1- and 2-step chunks against the
  one-at-a-time reference; one mutant watched to fail: the extras slice handed to
  `prepare_chunk` offset by one chunk (M4).
- **P21** the path A2 proofs (`--cumulative`) on C2 within **1.0–1.15×** of C2's own-proof
  time (C1: 1.23× — their extra plan/record-download work is host and PCIe time that now
  overlaps).
- **P22** device peak ≤ 12 GB (sampled at 100 ms).

**Kill.** Any P19 mismatch not traced to a bug that is fixed and re-run clean; or a C2
total not under C1's 8.90 s (then C2 is recorded as refuted on this card and §6 stands).

- 17:02Z **C2 as pre-registered, measured** (`c2/run-all.sh`; the §8 design: one preparer
  on a forked stream, the check on the caller's legacy null stream, 6 GB chunk budget, no
  pinned staging): **oracles clean** (`--verify-b2` 5,226 steps, 0 code mismatches;
  `--simwarp 50` 0; 5,226/5,226 VERIFIED, 0 host-route) — **P19 MET**. Timing: **warm
  7.863 / 7.871 / 7.902 → best 7.86 s; cold 8.00 s** (6 chunks; device peak 5.29 GB).
  **P16 MISSED** (7.86 > 6.5; the point estimate 5.9 was wrong by 2 s), **P16b MISSED**,
  **P17 MET** (cold − warm = 0.14 s: the read IS hidden), **P18 half** (stage sum − wall =
  3.7 s ≥ 2.5 MET; the check sum FELL, 3.71 → 3.45, instead of growing — no co-running
  happened, see below), **P22 MET** (5.3 GB ≤ 12), P20 MET (4/4 gpu tests; M4 killed, and
  a second mutant, the reader returning a chunk's proofs reversed, killed). The serial
  control through the refactored code: 9.11 s (C1 8.90–9.07). **Why 7.86 and not 5.9:**
  the per-chunk timeline (`c2/run3.txt`) shows chunk 1's upload at 0.20 s and every later
  chunk's at 0.77–0.82 s whatever its size, ending when the previous check kernel ends:
  the preparer was being serialised behind the check, so the pipeline degenerated into
  C1 with smaller chunks (which is where the 1 s gain came from).
- 17:10–17:50Z **Diagnostics** (`c2/diag/`, one run each, not protocol runs):
  E1 check moved to a forked non-blocking stream — upload stalls move, total 7.60;
  E2 pool release threshold → max — 7.41, stalls unchanged, peak unchanged;
  E3 `--no-coop` (thread-per-slice scan) — 7.68, unchanged; E4 allocator
  `REUSE_ALLOW_INTERNAL_DEPENDENCIES` off — 7.32, unchanged; E5 fine timers — the stall is
  one `cuMemAllocAsync` on the preparer's stream, 0.47–0.59 s, the first allocation after
  the check kernel went in flight (a 2.5 MB one in chunk 2, the 1.3 GB buffer in chunk 3);
  E6 pool pre-grown by 10 GB (peak 9.7 GB) — unchanged, so not pool growth; E7 split alloc
  / memset / tiny alloc — the `cuMemAllocAsync` itself blocks, the memset issue and a
  following tiny alloc do not. nsys on this machine lacks its importer. **E8: a
  `cuStreamSynchronize` on the check stream before the pageable code download removes
  the stall entirely** (uploads 0.13–0.19 s every chunk): the thread spinning inside the
  pageable device-to-host copy was holding up allocations from the other thread for the
  kernel's duration [driver behaviour inferred from timing, not from a trace]. Total
  still 7.36 s — the kernels now genuinely co-run and slow each other (check sum 3.45 →
  5.25, scan passes 1.45 → 3.7) and the pipeline is PREPARE-bound: each 1.28 GB chunk's
  preparation is ≈ 1.2 s serial (upload 0.15, scan 0.85 co-running, records 0.08, plan +
  layout 0.12, gather 0.06) against a 1.0 s check. E9: a smaller first chunk cuts the fill
  (1.0 → 0.45 → 0.22 s) and the total does not move; the 10 GB reserve is not load-bearing
  (7.354 vs 7.365) → removed. **E11: two preparer threads, each on its own stream, taking
  chunks in turn — 6.41 s (4 GB chunks), 6.34 (3 GB), 6.33 (2 GB); three preparers 6.35;**
  the check thread now runs back to back (check span = wall) with the GPU saturated by the
  co-running kernels. E13 knob matrix on that: check blocks 576 / 1152 → 7.28 / 7.41
  (worse); scan blocks 1024 / 256 → 6.40 / 6.62; gather blocks 1024 → 6.60; first chunk
  1 GB → 6.62; 4 host threads → 6.70; the pre-registered knobs are the best. E14 without
  the pool retention: 6.45 / 6.52 / 6.63 against 6.33–6.41 with → kept.
  **What the diagnostics say about the floor:** the kernel work is ≈ 5.2 s standalone
  (check 3.45 + scan 1.45 + gather 0.33 at these chunk sizes); co-running costs ≈ 0.5 s of
  interference; fill ≈ 0.45 and init ≈ 0.12 are exposed. ≈ 6.3 s is this kernel set's
  practical floor on this card; the §3 10× clause (≤ 6.02 s) needs a faster check or scan
  kernel, not more scheduling.

**C2 as built — pre-registered before the protocol run** (same protocol; `edp_resident`
defaults are now the chosen configuration: two preparers on forked streams, the check on a
third, chunk budget 3 GB after a 2 GB first chunk, pool retention, stream wait before the
code download; `--serial` = C1):
- **P23** warm best-of-3 **≤ 6.5 s**, point 6.35 (6.25–6.45). **P23b: the 10× clause
  (≤ 6.02 s) MISSED.** P4 MISSED.
- **P24** cold ≤ warm best + 0.5 s.
- **P25** oracles clean again on the as-built route (verify-b2 0 mismatches, simwarp 50
  clean, 5,226/5,226, 0 host-route); the 5 gpu tests pass (the pipelined test now also
  with 2 and 3 preparers); mutant M5 — the checker taking chunks in arrival order instead
  of chunk order — killed in at least one of three runs of the test (it is timing-dependent
  with several preparers, and that is recorded as such).
- **P26** device peak ≤ 8 GB.
- **P27** path A2 (`--cumulative`) within **1.0–1.2×** of the as-built own-proof time
  (its 0.55 s of extras construction runs before the pipeline and cannot overlap).
- **P28** the serial control (`--serial`) within 8.6–9.3 s.
**Kill** as before: any P25 mismatch not traced to a bug fixed and re-run clean.

- 17:26Z **C2 as built, measured** (`c2/run-all.sh`, outputs `c2/run*.txt`, `c2/a2-run*.txt`;
  12 chunks, 2 GB first). **Oracles: `--verify-b2` 5,226 steps, 0 code mismatches;
  `--simwarp 50` 0; 5,226/5,226 VERIFIED, 0 UNSUPPORTED, 0 host-route — P25 MET** (with
  the five gpu tests and M5 killed 3/3).
  | run | cache | read (overlapped) | fill | checks span | total |
  |---|---|---|---|---|---|
  | 1 | **cold** | 2.114 | 0.435 | 0.435..6.287 | **6.424 s** |
  | 2 | warm | 1.134 | 0.492 | 0.492..6.368 | 6.518 s |
  | 3 | warm | 1.215 | 0.483 | 0.483..6.298 | 6.428 s |
  | 4 | warm | 1.117 | 0.453 | 0.453..6.280 | **6.423 s** |
  (stage sums per run: upload 1.2, scan passes 1.7 + 2.0 co-running, plan 0.65, layout 0.6,
  gather 0.9–1.3, check 5.4–5.5, all overlapped; init 0.12–0.14; device peak 4.59 GB.)
  **Best of 3: 6.42 s warm; cold 6.42 s.** **P23 MET** (≤ 6.5; the 6.35 point was 0.07
  low), **P23b MISSED** — the §3 10× clause stays missed at **9.4×** under the published
  60.2 s; P4 MISSED. **P24 MET** (cold = warm to 1 ms: the cold read is entirely hidden).
  **P26 MET** (4.59 ≤ 8 GB). **P28 MISSED by 3 ms**: the serial control took 9.303 s
  (8.6–9.3 predicted) with its check stage at 4.22 s against 3.71–3.77 s in every earlier
  serial run on this card — the kernel itself varied by 0.45 s run to run here, cause not
  investigated [noted, no claim]. Path A2 proofs (`--cumulative`, 11 chunks): cold 7.278,
  warm 7.203 / 7.203 / 7.490 → **best 7.20 s = 1.12× — P27 MET** (1.0–1.2); fill 0.80 s
  (the 0.55 s extras construction before the pipeline plus a heavier first chunk).
  **Ratios now.** Our proofs, 6.42 s: **31.1× under same-machine lrat-check (199.6 s)**,
  57.2× under cake_lpr (366.9 s), 9.4× under the published 128-thread 60.2 s, 603× under
  the published single thread. Path A2 proofs, 7.20 s: **26.9× under lrat-check (193.5 s)
  — the §3 ≥ 20× clause now holds on these too** (17.7× in §7), 66.5× under cake_lpr
  (478.6 s). C2 over C1: 8.90 → 6.42 s (1.39×); 10.91 → 7.20 s on the A2 proofs (1.52×).
  **The honest sentence today:** "one workstation with a 20 GB GPU checks the compositional
  EDP proof in 6.4 s warm or cold (7.2 s on the published pipeline's own proofs), where the
  published run took 60.2 s on 128 threads and lrat-check takes 199.6 s on this machine one
  step at a time" — §0's caveats attached, nothing leaves the repo before the cross-vendor
  panel. **§8 ledger:** P16 MISSED, P16b MISSED, P17 MET, P18 half, P19 MET, P20 MET,
  P21 (as-designed A2) not run — superseded by P27, P22 MET; as built: P23 MET, P23b MISSED,
  P24 MET, P25 MET, P26 MET, P27 MET, P28 MISSED by 3 ms.
  **Where the remaining time is:** the kernels, co-running, fill the card from 0.45 s to
  6.3 s; standalone they sum to ≈ 5.2 s. Under 6.02 s and under 5 s both need a faster
  check kernel (the direct table past 4,096 codes, S5's predicted 1.8×, footprint vs the
  48 MB L2 to be measured) or a faster scan — a kernel arc, separately pre-registered,
  NOT started here.

## 9. K — the check kernel: the direct table widened past 4,096 codes (pre-registered 2026-11-08, before code)

Owner 19:3xZ: "go. pre-register the direct-table kernel arc." §8 left the pipeline at
6.42 s with the card busy from 0.45 s to 6.3 s and the kernels summing to ≈ 5.2 s
standalone, of which the check is 3.71–3.77 s (hashed layout, 2.0 G hints at 0.54 G/s,
25.2 M lemmas → ≈ 79 hints per lemma). The scan passes (1.45 s) are untouched here.

**Why the table is hashed on EDP today.** `LRAT_TABLE = 4096` is one constant doing two
jobs: the per-warp slot count of BOTH layouts and (minus one) the occupancy cap that
keeps `TableOverflow` on the same lemmas as SimWarp's `LitTable` (`CAP = 4096`). The
direct layout indexes slot = literal code, so it needs every code below the slot count;
EDP2_1161 has 4,207 variables → codes up to 2·4207+1 = **8,415**, and the launcher falls
back to hashed. S5 measured direct over hashed at 1.8× warm (0.079 vs 0.141 s) and
1.5× cold (0.095–0.104 vs 0.149–0.151 s) on the 415 MB proof, 1.47× cold on the 898 MB one
(0.247–0.251 vs 0.364–0.368) — the gain narrowed as the proof grew.

**Design K (kernel + launcher; the hashed path's behaviour unchanged).**
- K1 `lrat_check_lemmas` takes a new `slots: u32` argument — the per-warp region size of
  the DIRECT layout (`base = gw · slots`; insert refuses `k > slots`; a lookup of a code
  `≥ slots` fails the lemma closed as today). The hashed layout ignores it and keeps
  `LRAT_TABLE` and its mask. The occupancy cap stays `LRAT_TABLE − 1 = 4,095` in both
  layouts, so overflow falls on exactly the lemmas SimWarp reports — the cap and the slot
  count are now two things.
- K2 Launcher: `lrat_max_code` (already computed per launch) gives `m`; `slots = round_up
  (m + 1, 32)` (whole 128 B lines, min 32); the table allocation is `blocks · 4 · slots ·
  8 B`; the u32 clamp on `blocks` is recomputed from `slots` (today's 65,536 assumes 4,096).
  **Layout policy, fixed now:** direct whenever the table at the chosen block count is
  ≤ 1 GiB (at 1,152 blocks: `slots ≤ 29,120`, i.e. ≤ 14,559 variables); hashed above —
  competition-scale proofs (10^5–10^6 variables) stay hashed. If P30 below is MISSED the
  policy reverts to "direct only below 4,096 codes" and the wide table stays reachable by
  request (`Some(true)`); either way the rule is written here before the number exists.
- K3 Block count: `blocks = 0` keeps 24 blocks per SM for direct (S5's sweep); the sweep
  in the protocol decides whether EDP's wider footprint moves it, and if it does the
  default becomes a function of `slots` (two points: S5's and this one), not a new constant.
- K4 Oracle changes: (a) S5's per-lemma code test (`lrat_decode_device.rs`, battery +
  632 archive files, both layouts forced, blocks 1/7/64/4,096) runs unchanged — direct now
  at `slots` sized to each proof; (b) a new battery proof with variables up to 6,000
  (codes ≥ 4,096, previously an error when forced direct) forced direct vs SimWarp, with a
  lemma at the 4,095 cap so the cap is seen to be independent of `slots`; (c) the
  fail-closed test (`check_kernel_direct_layout_fails_closed_past_the_table`) is
  re-expressed through the internal launch with an undersized `slots`; (d) `edp_resident
  --verify-b2`'s reference B2 is pinned to the HASHED layout (today it is "auto", which
  would follow the change under test); `--simwarp 50` stays. (e) Two kernel mutants on a
  copy, each watched to fail the oracle before the real kernel is trusted: the direct
  lookup dropping the sign bit (S5's), and the direct region stride left at `LRAT_TABLE`
  while `slots ≠ 4,096` (overlapping regions).

**Footprint arithmetic (before measuring).** A lemma's live entries ≈ 79 units + its own
literals ≈ 100; at one 64-bit slot per code, 16 codes per 128 B line, a fully scattered
lemma touches ≤ 100 lines ≈ 12.8 KB per warp — the SAME line count the packed hashed
layout touches; the direct layout's gain is one load with no probe chain and line sharing
when a clause's variables are neighbours (EDP's auxiliary counter variables are
consecutive; its base variables x_d, x_2d, … are not). 4,608 warps (1,152 blocks) × 12.8 KB
= 59 MB, over the 48 MB L2 when fully scattered; 576 blocks → 29 MB. Allocation 1,152 ·
4 · 8,416 · 8 B = 310 MB (151 MB today), zeroed per chunk (≈ 1 ms).

**Predictions (ranges fixed now).**
- **P29** pipelined wall, our proofs, best of 3 warm: **4.6–5.6 s** (point 5.1); cold
  within 0.1 s of warm (P24's property, kept).
- **P29b** ≤ 6.02 s → the §3 **10× clause MET**. **P4** (≤ 5 s): predicted **MISSED
  narrowly** (point 5.1) — stated as the coin flip it is.
- **P30** serial-path check stage on EDP (`--serial`, warm best of 3): **2.0–2.85 s**
  from 3.71–3.77 (1.3–1.85×, S5's 1.47–1.8× widened downward for the larger footprint).
  **Kill / no-gain line:** ≥ 3.4 s → the wide direct layout is not faster here; K2's
  policy reverts as written; the arc records "no gain" and stops.
- **P31** block sweep (serial path, check stage, 3 runs each at 288 / 576 / 1,152 / 2,304
  blocks): best at **576 or 1,152**; 2,304 slower than the best by ≥ 10 % (L2 overflow).
- **P32** codes equal everywhere: K4(a)–(d) zero disagreements before and after; both K4(e)
  mutants killed. Any disagreement that is not a mutant = **KILL** (stop, diagnose, no
  timing claimed).
- **P33** device peak ≤ 5.0 GB (4.59 + the table's 0.16 GB, pool retained).
- **P34** path A2 proofs (`--cumulative`): 1.0–1.2× our proofs' wall (P27's pattern).
- **P35** no regression where the direct layout already applied: the 415 MB archive proof's
  check, cold, ≤ 0.11 s (S5 after the panel fixes: 0.095–0.104); the full gpu suite passes.
- **P36** serial total (`--serial`): 7.9–8.7 s from 9.1–9.3.

**Protocol** (as §8, run as `~/staging/2026-11-08-edp-record/k/run-all.sh`): oracles first
(run0: `--verify-b2 --simwarp 50`), `posix_fadvise(DONTNEED)` then cold (run1), warm runs
2–4 with the 100 ms device-memory sampler on run 2, `--serial` control (run5), the P31
sweep (runs 6–17, serial path), then A2 cold + 3 warm. 6 niced host threads on cores 0–5.
The 415 MB archive check (P35) by `lrat_check` cold, three single runs. Corrections to any
of this in brackets; nothing above changes after the first measurement.

**K as measured (2026-11-08, 19:50–19:55Z; `~/staging/2026-11-08-edp-record/k/`).** Built as
designed: `slots` argument, region = `gw · slots` in the direct layout, cap unchanged at
4,095; launcher `lrat_direct_slots` (whole lines), the u32 clamp from the region, the 1 GiB
auto policy; `--verify-b2`'s B2 pinned hashed; battery + "wide small" + "wide duplicate at
the cap"; both mutants killed before the kernel was trusted (A, sign dropped: battery
blocks=1 lemma 50 code 2 vs 0, and 5,226/5,226 EDP steps mismatching; B, stride left at
4,096: battery corruption 13 blocks=64 lemma 212 — code 39, an out-of-region write into
`codes` — and 5,033 of 5,226 EDP steps mismatching). Full gpu suite 291 passed
(`archive_agreement_with_external_checkers` fails only for want of `LRAT_CHECK` in the
environment — not run; the kernel oracle on the archive ran over the three campaign
directories; the `negatives` directory is no longer at the path the S5 runs used).
**Oracles (run0): 0 mismatching steps against the hashed B2 on all 5,226 steps; SimWarp 50/50;
P32 MET.** Then, direct at the launcher's 24 blocks per SM (1,152 blocks, 310 MB table):
| run | cache | read | fill | checks span | total |
|---|---|---|---|---|---|
| 1 | **cold** | 2.036 | 0.481 | 0.481..6.072 | **6.214 s** |
| 2 | warm | 1.082 | 0.461 | 0.461..5.956 | 6.131 s |
| 3 | warm | 1.222 | 0.463 | 0.463..5.960 | **6.091 s** |
| 4 | warm | 1.043 | 0.482 | 0.482..6.009 | 6.177 s |
(co-running stage sums: check 4.4–4.7 against §8's 5.4–5.5, scan passes 3.7–4.5, gather
1.0; peak device memory **5.42 GB**.) **Best of 3: 6.09 s warm; cold 6.21 s** — from 6.42 /
6.42 (§8), 1.05×. **P29 MISSED** (4.6–5.6). **P29b MISSED by 0.07 s** — 6.09 against the 6.02
line; **9.9× under the published 60.2 s**. **P4 MISSED.** P24's property weakened: cold is
0.12 s over warm (the read is no longer entirely hidden).
**P30 MISSED, and the no-gain line crossed:** the serial-path check stage at the launcher's
1,152 blocks is **4.39 s — SLOWER than the hashed 3.71–3.77 s**; the sweep (runs 6–17, check
stage, 3 runs each): **288 blocks 3.447 / 3.454 / 3.469 s**, 576 blocks 4.45 / 4.55 / 4.59 s,
1,152 blocks 4.65 / 4.66 / 4.69 s, 2,304 blocks 4.63 / 4.68 / 4.69 s. The widened direct
layout's best point, 3.45 s at 6 blocks per SM, is 1.07–1.09× over hashed and sits ON the
pre-registered line (≥ 3.4 s = no gain). **P31 MISSED** (best at 288, not 576 / 1,152; the
"2,304 ≥ 10 % slower than the best" half held, by 35 %). **P36 MISSED** at the default
(9.49 s); 8.53–8.58 s at 288 blocks, inside 7.9–8.7 but not the configuration P36 named.
**P33 MISSED** (5.42 GB against ≤ 5.0: the table grew 0.16 GB, the peak 0.83 GB — not
investigated). **P34 and P35 INVALID — host contended:** from ≈ 19:52:30Z a Coccinelle run
(five `spatch` processes and a `yices-smt2`, nice 15, not this session's) shared the host;
runs 14–17's totals went to 16–18 s with their check stage unchanged (4.63–4.69 s), the A2
runs' host stages ran at 2–4× §8's (read 4.5–4.9 s vs 0.84, plan 2.9–4.3 vs 0.9) for totals
of 12.0–14.5 s, and the 415 MB runs' plan stage doubled; their check figures (0.120–0.130
s against ≤ 0.11) are therefore not read. Both to be re-measured on a quiet host, below.
**The consequence the pre-registration fixed:** P30 missed → **K2's auto policy reverted**
to "direct only below 4,096 codes" (`LRAT_DIRECT_AUTO_CODES`; the 1 GiB budget constant is
gone); the wide region stays reachable by request (`Some(true)`, `--check-table direct`).
**Reading.** (1) Footprint, not probing, is what the card feels: the direct region at
8,416 slots touches the same ≈ 100 lines per lemma as the packed hashed table but spread
over 67 KB per warp instead of 32 KB; at 4,608 warps that is 59 MB of live lines against a
48 MB L2 and the kernel is 25 % SLOWER than hashed, at 1,152 warps (15 MB) it is 7 % faster.
The S5 gain (1.5–1.8×) was a 350-variable proof's — 704 slots, 5.6 KB per warp — and did not
survive a twelvefold wider region. (2) The pipeline nevertheless gained 0.33 s with the
SLOWER standalone kernel: co-running with the scan passes, 4,608 check warps take a larger
share of the SMs than 1,152 do (check stage 4.4 s co-running vs 5.4 s for hashed at 288
blocks) — a scheduling-share effect, not a kernel speedup. The standalone optimum (288) and
the co-running optimum may therefore differ, which is what the follow-on below measures.

**K-follow-on, pre-registered 2026-11-08 20:05Z, before measuring (numbers fixed now).**
K3 said the sweep decides the direct block count; the sweep says 288 for 8,416 slots. All
runs below are forced (`--check-table …`), on our proofs, pipelined, 6 niced threads, a quiet
host (no `spatch`); `~/staging/2026-11-08-edp-record/k/run-follow.sh`.
- **F1** direct, 288 check blocks, best of 3 warm: **6.0–6.5 s** (point 6.25) — predicted NOT
  better than direct at 1,152 (the co-running share shrinks with the warps).
- **F2** direct at the default 1,152, best of 3 warm: 6.0–6.2 s (reproducing 6.09) and cold;
  peak device memory recorded again (P33's 5.42 GB unexplained).
- **F3** hashed (= §8's configuration), best of 3 warm: 6.3–6.5 s (reproducing 6.42).
- **F4** path A2 proofs, auto (= hashed after the revert): 7.0–7.4 s (reproducing 7.20) — the
  valid P34 re-measurement; and A2 direct at 1,152: 6.8–7.3.
- **F5** (P35 re-measured) 415 MB archive proof, `lrat_check` resident, cold ×3: check ≤ 0.11
  s (the direct layout at 704 slots is the S5 kernel plus one bound compare).
- **Decision rule, fixed now:** the auto policy stays "direct below 4,096 codes". If F2 beats
  F3 by ≥ 0.2 s in BOTH best-of-3 and cold, the ledger records that a wider auto policy
  would buy that much on this proof and leaves changing it to the owner (a 5 % gain from a
  slower kernel is a scheduling accident, not a property to build a default on). The K3
  block-count function is installed only if F1 beats F2 by ≥ 0.1 s; otherwise the direct
  default stays 24 blocks per SM and the sweep is recorded as "standalone and co-running
  optima differ". Kill: any `s VERIFIED` missing or any oracle disagreement.

**K-follow-on as measured (20:01–20:04Z, quiet host, `run-follow.sh`; every run `s VERIFIED`).**
- **F1 direct at 288 blocks, warm: 5.411 / 5.400 / 5.349 → best 5.35 s** — **F1 REFUTED in the
  good direction** (predicted 6.0–6.5, "not better than 1,152"): the smaller footprint wins
  co-running too (check stage 4.5 s co-running, scan passes 4.0–4.2, gather 1.3). **5.35 s is
  under the 6.02 s line: 60.2 / 5.35 = 11.3× — the §3 10× clause is MET on this configuration,
  which is NOT the launcher's default** (see the rule below). P4 (≤ 5 s) still missed. Not
  measured cold [F1 had no cold run in the pre-registration — an omission; added below as
  the final run].
- **F2** direct at 1,152: cold 6.199, warm 6.091 / 6.125 / 6.241 → 6.09 s, **MET** (reproduced);
  peak 5.42 GB again.
- **F3** hashed (§8's configuration): 6.500 / 6.534 / 6.591 → 6.50 s, **MET** at the edge
  (6.3–6.5; §8 measured 6.42 — a 0.08 s drift between sessions).
- **F4** A2 auto (hashed): cold 7.552, warm 7.412 / 7.564 / 7.383 → **7.38 s MET** (7.0–7.4;
  §8: 7.20); A2 direct at 1,152: 7.662 / 7.399 / 7.641 → 7.40 s, **MISSED** (6.8–7.3) — no
  gain on the A2 proofs at 1,152 blocks (their upload stage ran 3.0–3.2 s co-running against
  1.1 s hashed: the wider check kernel starves the preparers' copies). A2 at 288 not run.
- **F5 MISSED:** 415 MB check cold **0.120 / 0.120 / 0.123 s** (thread-decode line 0.117–0.120)
  against ≤ 0.11 and S5's 0.095–0.104 — **a ≈ 20 % regression on the small proof** that the
  contended run also showed. The only differences for this proof are the region (704 slots,
  5.6 KB per warp, against 4,096) and one bound compare per probe. **Diagnostic D1,
  pre-registered now:** region floored at 4,096 slots (`slots.max(4096)` — S5's layout
  exactly for every proof it already covered), F5 rerun cold ×3; if the check returns to ≤
  0.11 s the floor stays (the small region, not the kernel code, was the cost); if not, the
  bound compare is suspected and the pre-K PTX is diffed against the new one.
**Decision rule applied as written.** F2 − F3 = 0.41 s warm (cold: 6.20 vs §8's 6.42, 0.22 s)
≥ 0.2 → recorded: **a wider auto policy would buy 0.4 s on this proof at 1,152 blocks and
1.15 s at 288; changing the policy is the owner's call** (`LRAT_DIRECT_AUTO_CODES`); the
launcher's default stays direct-below-4,096-codes and the headline configuration needs
`--check-table direct`. F1 − F2 = 0.74 s ≥ 0.1 → **the K3 block-count function is installed.**
Its form, chosen from the three points now in hand (hashed 32 KB regions: best at 6 per SM;
direct 5.6 KB: 24; direct 67 KB: 6): **24 blocks per SM when all resident warps' regions
together fit the device's L2 (`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`, 48 MB here: 4,608 ×
region ≤ L2), else 6** — one rule for both layouts, replacing the per-layout constants.
**Final run, pre-registered now:** after D1 and the rule, `--check-table direct` with the
launcher's block count (→ 288 here): cold then best of 3 warm, device memory sampled.
Predicted **warm 5.3–5.5 s, cold 5.4–5.7 s, peak ≤ 5.0 GB** (the 288-block table is 78 MB);
A2 proofs `--check-table direct` best of 3: 6.9–7.4 s. The gpu suite must pass after the
rule (the block-count tests pin explicit counts and are unaffected).
- **D1 result: the floor did NOT restore it** — 0.121 / 0.118 / 0.122 s cold with 4,096-slot
  regions. The floor stays (it costs nothing and keeps S5's layout) but the cause is
  elsewhere. **D2, pre-registered now:** the pre-K kernel and launcher (`*.pre-k` scratch
  copies: §8's code exactly) rebuilt in this tree, same three cold runs. ≤ 0.105 s → the
  regression is K's (the bound compare / region multiply — then the PTX is diffed); ≈ 0.12
  s → it predates K (S5's 0.095–0.104 was measured 11-04, before §8's stream synchronise
  and today's driver state) and is recorded as session drift, not K's cost.
- **D2 result: the pre-K code measures the same — 0.116 / 0.120 / 0.120 s.** The 415 MB
  regression against S5's 0.095–0.104 **predates K**: session drift (recorded, cause not
  chased; §8's synchronise before the download and the day's driver state are the candidates).
  F5 is therefore MISSED against its number but **K's cost on the small proof is nil**.
  [The D1 floor is DROPPED again: it bought nothing, and under the K3 rule below it would
  have put every small proof's direct table on the 6-per-SM count S5 showed to be slower.]

**Final run as measured (20:09Z, quiet host, `run-final.sh`; the launcher's rule chose 288
blocks for both layouts on this proof — printed in every log — and `--check-table direct`):**
| run | cache | read | fill | checks span | total |
|---|---|---|---|---|---|
| cold | **cold** | 2.116 | 0.448 | 0.448..5.322 | **5.524 s** |
| 1 | warm | 1.216 | 0.442 | 0.442..5.200 | **5.339 s** |
| 2 | warm | 1.211 | 0.484 | 0.484..5.348 | 5.500 s |
| 3 | warm | 1.113 | 0.463 | 0.463..5.315 | 5.451 s |
(co-running: check 4.5–4.7, scan passes 4.0–4.2, gather 1.35–1.6; peak device memory 5.16 GB;
standalone check stage on the serial path 3.440 s.) **Best of 3: 5.34 s warm; cold 5.52 s —
warm 5.3–5.5 MET, cold 5.4–5.7 MET; peak ≤ 5.0 GB MISSED (5.16).** Path A2 proofs, direct:
6.234 / 6.288 / 6.293 → **6.23 s** — predicted 6.9–7.4, **MISSED in the good direction** (from
7.38 hashed, 1.18×: at 288 blocks the preparers' uploads are no longer starved, 1.0–1.2 s).
**§9 ledger:** P29 MISSED, P29b MISSED (as the default), P4 MISSED, P30 MISSED (no-gain line,
policy reverted), P31 MISSED, P32 MET, P33 MISSED, P34 INVALID → F4 MET, P35 INVALID → F5
MISSED (predates K, D2), P36 MISSED; F1 REFUTED upward, F2 MET, F3 MET, F4 MET / MISSED,
F5 MISSED; final warm MET, cold MET, memory MISSED, A2 MISSED upward.
**Ratios now, with the direct table requested** (our proofs 5.34 s): **11.3× under the published
128-thread 60.2 s — the §3 10× clause MET for the first time; 37.4× under same-machine
lrat-check (199.6 s)**, 68.7× under cake_lpr (366.9 s), 725× under the published single
thread (3,871.7 s). Path A2 proofs 6.23 s: 9.7× under 60.2 s (still short of 10× on the
published pipeline's own proofs), 31.1× under lrat-check (193.5 s), 76.8× under cake_lpr
(478.6 s). The launcher's DEFAULT on this proof is the hashed table: 6.50 s today (§8: 6.42),
9.3×. K over C2: 6.42 → 5.34 s (1.20×) requested, 6.42 → 6.50 default (none).
**The honest sentence today:** "one workstation with a 20 GB GPU checks the compositional EDP
proof in 5.3 s warm or 5.5 s cold when the checker is told to use its direct assignment
table (6.2 s on the published pipeline's own proofs), where the published run took 60.2 s on
128 threads and lrat-check takes 199.6 s on this machine one step at a time; with the
checker's default table it is 6.5 s" — §0's caveats attached, nothing leaves the repo before
the cross-vendor panel. **Open, for the owner:** whether `LRAT_DIRECT_AUTO_CODES` moves (the
rule above kept it; the measured case for moving it is 6.50 → 5.34 s here and 7.38 → 6.23 s
on A2, with the K3 block rule in place, and no proof past 4,207 variables measured). **Not
done:** the kernel itself is unchanged in its inner loop — under 5 s now needs either the
scan passes (1.45 s standalone, 4 s co-running) or a check kernel that touches fewer lines
per lemma; the 415 MB drift (D2) is unexplained.

**Policy decision (owner 20:2xZ: "your call"; recorded before the confirmation run).** The
direct layout becomes the launcher's default wherever its table, at the block count the K3
rule picks, is at most 1 GiB (`LRAT_DIRECT_TABLE_BUDGET`, the K2 figure, now paired with the
rule that made it hold): at 6 blocks per SM on this card that is 116,000 slots, ≈ 58,000
variables; above it the hashed table. Why: both layouts are oracle-equal (every test forces
each on every proof); at the same block count direct beat hashed standalone (3.45 vs 3.73 s)
and in the pipeline (5.34 vs 6.50; A2 6.23 vs 7.38); each probe is one load with no hash and
no chain, which the region's width does not change, while the budget bounds the memory. What
is NOT measured: any proof between 8,416 slots and the budget — the default there is an
extrapolation, flagged here and in the constant's comment; the first such proof measured
should be added to the sweep. **Confirmation run, pre-registered:** no `--check-table` flag,
cold then best of 3 warm: **warm 5.3–5.5 s, cold 5.4–5.7 s** (the final run's configuration,
now chosen by the launcher), the log's rule line showing direct at 288; the gpu suite passes.
- Confirmation run 20:46Z **CONTENDED** (another Coccinelle job, six `spatch` at nice 15, started
  during it): the rule line shows direct at 288 as predicted; cold 5.667, warm 5.712 / 5.749 /
  5.738 with the host stages inflated (read 1.8–2.0 s vs 1.1–1.2, plan 1.0–1.1 vs 0.65) — not
  read against the 5.3–5.5 / 5.4–5.7 prediction; rerun on a quiet host below.
- Confirmation rerun 20:54Z, after the Coccinelle job ended, **host under memory pressure**
  (a `lean` process at nice 15 — lean rss 6.4 GB, 460 s elapsed; a kworker at 90 % reclaiming; mem used 29 GB of 62, available 33 GB at the time
  of writing): the rule line shows direct at 288; **cold 5.528 s — MET (5.4–5.7)**; warm 5.516 /
  5.823 / 6.044 — the "warm" read stage was 2.2–2.9 s against 1.1–1.2 s on a quiet host, so the
  page cache was being evicted between runs and the warm set is not a warm set; **best 5.52 s,
  0.02 over the 5.3–5.5 prediction — not read as a verdict either way.** The default
  configuration produced the same device-side profile as the final run (check 4.45–4.59
  co-running, scan 3.85–4.14, fill 0.44–0.48). **A clean confirmation on a quiet host is
  pending** (`run-confirm.sh`; TODO).
  [CORRECTED 20:58Z: "under memory pressure" overstates it — 33 GB of 62 were available while
  the lean process held 6.4 GB; the slow "warm" reads (2.2–2.9 s) are unexplained, with the
  kworker at 90 % and the lean job's own I/O on the same `/home` enclosure as the candidates.
  The reading stands: the warm set is not a warm set, and the verdict waits for a quiet host.]
- Confirmation 2026-11-10 19:57Z, two `lean` jobs alive on core 6 at nice 19 (outside the run's
  cores 0–5; the owner asked for a hold before they were stopped): cold 5.489, warm 5.540 /
  5.375 / 5.394 — inside both windows, but logged as a contended point, not the verdict.
- **Confirmation 2026-11-10 19:59Z, QUIET host (owner: "host is clear"; lean 0, spatch 0, load
  2.1 from the resident cockpit), binary confirmed up to date by cargo against 5649e23c2.**
  Rule line: `hashed 288 | direct at 8416 slots 288` as predicted. **Cold 5.470 s — MET
  (5.4–5.7). Warm 5.347 / 5.419 / 5.354, best 5.35 s — MET (5.3–5.5).** The warm set is a warm
  set this time (read 1.17–1.25 s, plan 0.65–0.67 s, both at the quiet-host figures). Device
  side unchanged from the final run (check 4.44–4.61 co-running, scan 3.97–4.13). All four
  runs 5226/5226 verified. **The launcher's default now reproduces the §9 final-run figure
  without any flag: 5.35 s warm / 5.47 s cold = 11.3× / 11.0× under the published 60.2 s
  (10× clause MET as the DEFAULT configuration), 37.3× under same-machine lrat-check
  (199.6 s), 68.6× under same-machine cake_lpr (366.9 s).** Logs:
  `~/staging/2026-11-08-edp-record/k/run-confirm4.log`, `confirm-{cold,1,2,3}.txt` (the 19:57Z
  contended point is `run-confirm3.log`, whose per-run files it overwrote are those four names'
  previous contents — not kept).

## 10. H — the check kernel: hints evaluated a batch at a time (pre-registered 2026-11-10, before code)

Owner 20:0xZ: "go. pre-register the under-5-s arc." §9's confirmed default is 5.35 s warm /
5.47 s cold. P4 (≤ 5 s) has been MISSED in every section since §3.

**Where the 5.35 s goes (confirm-1, warm).** Card busy from the fill (0.44 s) to 5.20 s; the
check thread runs back to back (check stage sum 4.44 s co-running, combine 0.07, ≈ 0.25 s
waiting) — the pipeline is CHECK-bound. Standalone the check kernel is 3.44 s at 288 blocks
(§9 sweep), the scan passes 1.45 s, the gathers ≈ 0.35 s: 5.2 s of kernel work done in 4.75
s of co-running wall. The two preparers' co-running stage sums are 7.8 s together (upload
1.16 PCIe, scan 4.0, plan 0.67 + layout 0.55 host, gather 1.4), ≈ 3.9 s each inside the 4.75
s — 80 % busy, so a much faster check kernel makes the pipeline PREPARE-bound; that boundary
is measured below and decides whether a scan stage follows.

**What the check kernel does per hint, read from the source (`lrat_check_lemmas`).** The
proof's shape: 25,224,312 lemmas, 1,999,078,832 hints (79 per lemma), 379.6 M literals over
156.7 M clause entries — **2.4 literals per clause**, and the lemma clauses themselves ≈ 2
literals. The hint metadata is already fetched 32 hints per lane per round (K-era code). Then
each hint is resolved SERIALLY: 4 broadcasts, one coalesced literal load (2–3 lanes active of
32), one table probe per literal (2–3 scattered loads), a reduce, a ballot, a broadcast, the
unit insert (a dependent load + store), a `syncwarp` — ≈ 3 dependent memory round trips and
8 warp-wide operations per hint, 79 times per lemma, with 29 of 32 lanes idle. At 1,152
resident warps the measured cost is 157 µs per lemma per warp ≈ 2 µs per hint: a latency
chain, not bandwidth (the §9 sweep showed the kernel memory-bound only above 1,152 warps).

**Design H (direct layout only; the hashed path keeps its code).** Hints are evaluated
in BATCHES of as many hints as fit in 32 literal lanes, with the sequential RUP semantics
reconstructed in registers:
- H1 From the 32-hint metadata round each hint lane knows its clause length (0 for a
  forward hint, whose bounds were never loaded). An inclusive sum over lengths (5 shuffle
  steps) and a ballot select the longest prefix of hints whose literals fit in 32 lanes; a
  segment-start mask (one reduce-or) gives every literal lane its hint index and offset by
  popcount arithmetic; one indexed shuffle fetches its clause start. **One** coalesced
  literal load and **two** scattered probes per lane (the literal and its negation — the
  negation's presence decides whether a later unit insert is new) replace the per-hint
  chain: ≈ 6 batch rounds of 2 memory round trips per lemma instead of 79 × 3.
- H2 The ordered resolution loop (warp-uniform trip count = hints in the batch, ≈ 13):
  for hint j in order — admissibility first (forward → 4, deleted → 3, from two per-batch
  ballots over the hint lanes); then `free_j = popcount(ballot(free) & seg_j)`: 0 → code 0,
  ≥ 2 → code 1, else the unit `u_j` is broadcast from its lane and every lane of a LATER
  hint in the batch with literal `== −u_j` marks itself falsified (that is exactly what the
  serial insert of `−u_j` would have made the next probe see) and every later lane with
  literal `== u_j` marks its own eventual insert a duplicate; `new_j = !present(−u_j) &&
  !duplicate`; the cap test `count ≥ 4,095` BEFORE the insert, in order (→ code 5 at j);
  `count += new_j`; the unit lane stores the entry (direct slot `= code(−u_j)`), no probe
  chain. A `syncwarp` closes the batch so the next batch's loads see the stores.
- H3 A hint longer than the lanes left in a batch starts the next batch; one longer than
  32 literals is evaluated alone by TODAY's strided loop (kept verbatim as the fallback).
- H4 Lemma-literal inserts (≈ 2 per lemma) stay serial as today; the hashed layout's hint
  loop is the existing code, selected by `direct` at the same branch as now.
- H5 Why the semantics are preserved, stated so the oracle can refute it: an insert of
  `−u` changes a later probe's answer for literal `l` iff `l == −u`; "present before" for
  the count is the table at batch start OR an earlier unit of the same batch (`u_i == u_j`);
  the first failing hint in order fixes the code, and no store of a later hint is issued
  after a failure — all three are register facts in H2. Any divergence from
  SimWarp's `LitTable` sequence is a bug, not a tolerance.

**Oracles (K4's battery and archive tests run unchanged — both layouts forced on every
proof, blocks 1 / 7 / 64 / 4,096; `--verify-b2` against the HASHED B2 over all 5,226 steps;
`--simwarp 50`), plus battery proofs written for H's failure modes, each with a SimWarp
reference:** (a) a propagation chain inside one batch — hint j's unit falsifies hint j+1's
only other literal (a unit chain of ≥ 6 binary clauses); (b) two hints unit on the same
literal in one batch with the count at 4,094 before them (the duplicate must NOT trip the
cap) [CORRECTED below, 2026-11-11: the reference trips it] and the same at 4,095 (it must, at the first); (c) a conflict at hint j followed by a
forward hint at j+1 (code 0, not 4) and a deleted hint at j followed by a conflict at j+1
(code 3, not 0); (d) hint clauses of 31, 32, 33 and 70 literals (the fallback), the 33-long
one unit on its last literal; (e) a hint whose only free literal is the lemma's own literal
negated (falsified at the lemma stage, so the hint is a conflict); (f) a hint with a
repeated literal (`x x y`, y falsified → unit on x, free count 2 by lanes but one literal —
SimWarp's answer is the reference, whatever it is, and the serial kernel's today). **Mutants
on copies, each watched to fail before the kernel is trusted:** M-H1 the in-batch
falsification by `−u_j` dropped (killed by a); M-H2 duplicate units counted as new (killed
by b); M-H3 admissibility tested after the free count (killed by c); M-H4 the > 32 fallback
removed, the clause truncated to 32 (killed by d). Any oracle disagreement that is not a
mutant = **KILL** (stop, diagnose; no timing claimed).

**Predictions (ranges fixed now).**
- **P37** pipelined wall, our proofs, launcher default, best of 3 warm: **4.0–4.9 s** (point
  4.4) → **P4 (≤ 5 s) MET**; cold ≤ warm + 0.3 s. If the pipeline turns PREPARE-bound the
  wall sits at the top of the range: P37b, the check thread's idle time (checks span −
  Σ check − Σ combine) is **≥ 0.5 s** (today 0.25) — that is the signal for stage S below.
- **P38** serial-path check stage at 288 blocks, warm best of 3: **1.2–2.2 s** from 3.44
  (1.6–2.9×). **No-gain line: ≥ 3.0 s** → H is recorded as no gain and the K kernel is
  restored from the scratch copy.
- **P39** block sweep (serial path, 288 / 576 / 1,152, 3 runs each): best at 288 or 576;
  1,152 within 1.35× of the best (fewer round trips per lemma → less to hide → occupancy
  matters less; the L2 footprint is unchanged).
- **P40** zero oracle disagreements (battery, 632 archive files, B2, SimWarp) and 4/4
  mutants killed; the gpu suite's count grows by the new battery shapes and passes. [The shapes
  live inside one test function; the count cannot grow by them — panel round 2, 2026-11-11.]
- **P41** device peak ≤ 5.3 GB (no new arrays; 5.16 today).
- **P42** path A2 proofs (`--cumulative`), best of 3: **1.05–1.25×** our proofs' wall.
- **P43** 415 MB archive proof, `lrat_check` cold ×3: check **≤ 0.12 s** (D2's 0.116–0.120
  is the baseline; no regression on the small proof).
- **P44** serial total (`--serial`): **6.0–7.3 s** from 8.53–8.58.
- **P45** `--check-table hashed`: 6.3–6.7 s — the hashed path unchanged (F3: 6.50).
- **P46** kernel signature unchanged (13 PTX params); the per-thread register count is
  reported (`cuobjdump`/`ptxas -v` if reachable) because register pressure from the
  per-lane batch state is the named risk to P38 and P39.

**Stage S (scan), conditional — the decision rule fixed now, its numbers fixed when it is
pre-registered.** If P37b fires (idle ≥ 0.5 s) or the wall is > 5.0 s with P38 met, the
preparers are the bound and the scan is next: the two byte passes (1.45 s standalone, 4 s
co-running; 6.9 GB parsed twice at ≈ 9.5 GB/s per pass by one-warp blocks at 24 per SM —
a 32-byte window per warp iteration, ≈ 3.7 µs each, instruction-latency-bound) — four
warps per block and/or four bytes per lane per window, same records, same `dirty` slices,
the host plan untouched. If P37b does not fire and the wall is ≤ 5.0 s, the arc ends at H.
Either way S is a separate pre-registration with its own ranges before any S measurement.

**Protocol** (`~/staging/2026-11-10-edp-record-h/run-all.sh`, the §9 shape): the host
quiet by `pgrep -c -x lean` = `spatch` = 0 recorded in every log header (the §9
confirmation history); oracles first (run0: `--verify-b2 --simwarp 50`); `posix_fadvise
(DONTNEED)` then cold (run1); warm runs 2–4 with the 100 ms device-memory sampler on run 2;
`--serial` control (run5); the P39 sweep (runs 6–14, serial path); `--check-table hashed`
(run15); A2 cold + 3 warm; the 415 MB check cold ×3. 6 niced host threads on cores 0–5.
Corrections in brackets; nothing above changes after the first measurement. Scratch copies
of the pre-H kernel, launcher and tests go to the scratchpad before the first edit
(`*.pre-h`); mutants are built on copies and restored by `cp`.
[CORRECTED 2026-11-10 20:3xZ, before any H measurement, on reading the reference while
building: shape (b)'s parenthesis "the duplicate must NOT trip the cap" misstates the
reference. `lrat_insert` and SimWarp's `LitTable::insert` test the cap BEFORE looking the
key up, so at 4,094 + a new unit + a duplicate unit the duplicate FAILS (code 5) at count
4,095. The shape stays; its expected answer is whatever SimWarp says (that is the oracle,
not my sentence), and the kernel implements cap-before-lookup as the serial code does.]
[CORRECTED 2026-11-10 20:2xZ, before the protocol run: §9 and §10 say "632 archive files";
the archive the oracle tests read today is **180 proofs** (`2026-11-01-cert-campaign/run1`,
`run2-default2000`, `run3-cake-cnfuzz`; the 632 figure predates the `negatives` directory's
removal and is not re-derivable). And the gpu suite's first run under H had three
`archive_agreement_*` tests fail with "0 proofs found" because I passed the run directories
as RELATIVE names from the wrong cwd — while `check_kernel_codes_equal_simwarp_on_the_archive`
PASSED on the same empty listing (it had no `n > 0` assertion: a check that cannot fail). With
absolute paths all archive tests pass on H (180 proofs; the kernel oracle at 4 block counts × 3
layouts; 180/180 agree with lrat-check, 147 verified). The assertion is added; the suite is
reported below as 289 + 3 = 292 passing across the two invocations.]
[CORRECTED 2026-11-10 22:5xZ, before the H ledger: two protocol runs exist and NEITHER is a
clean H measurement. (1) The first run (`run-all.log`, 20:20–20:23Z) ran the H binary (built
20:17Z; its PTX is the `kernels.ptx.h` snapshot) but was CONTENDED — the owner was using
the machine ("hmm, i was using the machine right then") — so none of its numbers is read
against a prediction. (2) After it, at 20:24:54Z, I issued the no-gain-rule command (copy the
snapshots to staging, `cp` the pre-H kernel back into the tree, rebuild the examples) as a
backgrounded shell line; the owner rejected the tool call 16 s later, but the line had
already run to completion (PTX and both example binaries timestamped 20:24:55–58Z; `git
diff` of the kernel against HEAD empty; the target PTX byte-identical to `kernels.ptx.k`). I
did not check the tree after the rejection ([[feedback_interrupted_bash_may_have_run]]: a
rejected or interrupted Bash may already have written). So the second run (`run-all2.log`,
owner's "clear now." 22:39:50Z, load 1.08, lean = spatch = 0) measured the **K kernel**: warm
5.492 / 5.379 / 5.419 → 5.38 s, cold 5.440 s, serial check 3.409 s, sweep 288: 3.422 / 3.447
/ 3.451, 576: 4.425 / 4.507 / 4.532, 1,152: 4.641 / 4.675 / 4.684, hashed 6.652, A2 cold
6.318 / warm best 6.537, 415 MB check 0.117–0.124 s — the §9 K figures reproduced to within
noise (§9: 5.35 / 5.47; sweep 3.45 / 4.45–4.59 / 4.65–4.69; A2 6.23; D2 0.116–0.120), which is
how the misattribution was noticed: a batched kernel cannot reproduce the serial kernel's
occupancy curve. That run stands as a THIRD quiet-host confirmation of §9, nothing more; its
device peak read 6.03 GB against §9's 5.16 (sampler: 48 samples of 100 ms; the first sample
2 MiB, so no co-tenant) — unexplained, not read. The H source was restored from the
`kernels-lib.rs.h` snapshot at 22:45Z (`cmp` clean), both examples rebuilt (4.9 s, PTX
byte-identical to `kernels.ptx.h`). P46 is answerable from the two PTX snapshots alone
(`ptxas -v -arch=sm_89`): **K 35 registers, H 47 registers, 0 spill bytes both, 440 bytes
cmem[0] both (13 params)** — 12 more registers, no spills; at 47 registers a 256-thread
block uses 12,032 of the SM's 65,536, so occupancy is not register-limited at 288 or 1,152
blocks. [WRONG — the blocks are 128 threads; corrected in the panel round-1 bracket below, 2026-11-11.] The clean H protocol run awaits the owner's next clear; its log is `run-all3.log`.]

**As measured (2026-11-11 03:17–03:20Z, owner: "go. host is clear."; load 1.30, lean =
spatch = 0; the log header records the loaded PTX byte-identical to `kernels.ptx.h` and the
tree kernel identical to `kernels-lib.rs.h`; `run-all3.log`, per-run files under
`h-run-0317Z/`).** Oracles first (run0): `--verify-b2` 0 mismatching steps of 5,226 against
the hashed B2, SimWarp 50/50, 5,226/5,226 verified; earlier today the battery (incl. the six
§10 shapes, 3 layouts × 4 block counts), the 180-proof archive at 4 block counts, and M-H1–4
all killed. Then:
- **P37 MET: warm 4.721 / 4.681 / 4.671 → 4.67 s** (4.0–4.9; point 4.4); **cold 4.852 s**
  (≤ 4.97 MET). **P4 (≤ 5 s) MET for the first time since §3.** 12.9× under the published
  60.2 s warm (12.4× cold); 42.7× under same-machine lrat-check (199.6 s). [Caveats and the
  same-machine cake_lpr figure in the same paragraph: see the panel round-1 bracket below, 2026-11-11.] Check stage
  co-running 3.76–3.84, scan passes 3.70–3.74, gather 1.76–1.82; combine 0.074.
- **P37b did NOT fire:** check-thread idle (span − Σcheck − Σcombine) 0.249 / 0.235 / 0.169 s
  (< 0.5; today's K rerun 0.17–0.25 as well). The pipeline is still check-bound after H:
  3.8 s of check in a 4.1 s span.
- **Stage S rule, as fixed above:** P37b did not fire and the wall is ≤ 5.0 s → **the arc
  ends at H.** No scan pre-registration follows from this result.
- **P38 MISSED: serial check at 288 blocks 3.012 / 3.003 / 3.014 → 3.003 s** (1.2–2.2
  predicted; 1.15× from §9's 3.447, from today's K rerun's 3.422). **The no-gain line
  (≥ 3.0 s) is crossed by 3 ms on the pre-registered best of 3**; the serial control in the
  same log (run5) read 2.994 s, the four 288-block runs span 2.994–3.014 (run-to-run spread
  ≈ 20 ms). Read as written: **no gain by the standalone line.**
- **P39 MET:** best at 288; 576: 3.443 / 3.464 / 3.485; 1,152: 3.675 / 3.698 / 3.744 →
  1,152 is 1.22× the best (≤ 1.35). The curve flattened exactly as predicted: K's is 3.42 /
  4.43 / 4.64 (1.36×); at 1,152 blocks H is 0.97 s faster than K, at 288 only 0.42 s.
- **P40 MET** (zero disagreements; 4/4 mutants; suite 292 with the new shapes).
- **P41 MET:** device peak 5.13 GB (≤ 5.3; 42 samples; K read 5.16 in §9 and 6.03 today).
- **P42 MET:** A2 proofs warm 6.261 / 5.920 / 5.774 → 5.774 s = **1.236×** ours (1.05–1.25);
  cold 5.751. A2 is now 10.4× under 60.2 s on the published composition too. [On proofs
  regenerated HERE with the published pipeline (§7), not the published files — corrected below, 2026-11-11.]
- **P43 MET:** 415 MB check 0.104–0.107 s (≤ 0.12) — and the D2 drift is gone: K measured
  0.117–0.124 on the same files 4½ hours earlier, H 0.104–0.107, so the drift was in the
  kernel's hint loop, not the host. [Overreach — softened in the panel bracket below, 2026-11-11.]
- **P44 MISSED:** `--serial` total 8.027 (run5), 8.057–8.087 (sweep) against 6.0–7.3 — the
  serial path's other stages are 5.0 s and the check gave back 0.44 s, not the 1.2–2.2 the
  range assumed; the miss is P38's miss, not a second finding.
- **P45 MET:** hashed 6.603 s (6.3–6.7; F3 6.50) — the hashed path's code is untouched.
- **P46 reported:** 13 PTX params unchanged; registers K 35 → H 47, 0 spill bytes either.
  Not the P38 limiter: a 256-thread block at 47 registers takes 18 % of the register file,
  and P39's flattened curve says occupancy matters LESS under H, not more. [WRONG arithmetic —
  128-thread blocks; at 1,152 blocks H holds 40 warps/SM to K's 48 — corrected below, 2026-11-11.]

**Tally: 7 of 9 ranges MET (P37 + P4 + cold, P39, P40, P41, P42, P43, P45); P38 and P44
MISSED; P46 reported.** The honest sentence: **the pipelined wall gained 0.68 s (5.35 →
4.67, 13 %) from a check kernel that gained 0.44 s standalone (3.44 → 3.00, 13 %)** — the
pipeline is check-bound, so the kernel's gain passed through almost one for one, but the
kernel gained a seventh where the design predicted a half. The design's model was wrong in
the same place §9's was: "a latency chain, not bandwidth" (§10, profile paragraph) did not
survive measurement. H removed the per-hint literal-probe chain and the 29-idle-lane rounds,
and P39 shows the latency it hid is real (the 1,152-block point improved 0.97 s); but the
per-hint METADATA loads — `hc`, `del`, `cs`, `len`, three scattered lines per hint, ≈ 237 L2
lines per lemma — are untouched by H and are, by this result, most of the standalone cost,
and H adds a negation probe per literal lane (≈ 190 lines per lemma). [Both counts wrong:
≈ 2 scattered lines per hint, and the negation probe shares its line with the literal's slot
— corrected in the panel brackets below, 2026-11-11.] [Reading, not
measured.] The next kernel step, if there is one, is a packed 16-byte (deleted-at,
clause-start, length) record per clause — one line per hint instead of three — and the
negation probe only for lanes still free; **not pre-registered, not started.**

**The rule and the decision.** P38 ≥ 3.0 s says: H is recorded as no gain and the K kernel
is restored. The same run met P4, the arc's purpose, for the first time, with every oracle
clean, by the margin the rule was written to protect against rationalising (3 ms of 3,003).
§9 ended in this exact shape — standalone no gain, pipeline gain — and the owner decided
("your call"; direct became the default). The decision is the owner's again and is put to
them with the recommendation to KEEP H; until it is made the H kernel stays in the working
tree uncommitted, and the record's headline stays §9's 5.35 s. What is not in question: the
numbers above stand as measured either way.

**Decision (owner, 2026-11-11): "keep. your call stands."** H is the check kernel for the
direct layout; the K kernel's hint loop survives as `lrat_hint_strided` (the hashed path and
the > 32-literal fallback) and as the `kernels-lib.rs.pre-h` snapshot in
`~/staging/2026-11-10-edp-record-h/`. The record's figure for EDP2_1161 on this machine is
now **4.67 s warm / 4.85 s cold, launcher default, no flag** (12.9× / 12.4× under the
published 60.2 s; 42.7× under same-machine lrat-check; A2 proofs 5.77 s, 10.4×) [with the §0
caveats and same-machine cake_lpr 366.9 s = 78.5× as restated in the panel bracket below, 2026-11-11]. P38 stays
MISSED in the ledger and the no-gain line stays crossed — the decision overrides the rule's
consequence, not its reading. The arc ends here by the stage-S rule; the §0 caveats travel
with every sentence above, and nothing leaves the repo before the cross-vendor panel.

[PANEL ROUND 1, 2026-11-11 (`~/staging/2026-11-11-panel-edp-record/`; seats Opus 5.5 and
Fable 5.1 on the API, Grok 4.7 and composer-2.5 through the jail; 0 CRITICAL, 13 MODERATE,
22 LOW raw). Corrections, each verified in the code by the orchestrator before writing:
- **P46 arithmetic was wrong.** `lrat_check_lemmas` launches 128-thread blocks (4 warps),
  not 256. On sm_89 (65,536 registers, 48 resident warps per SM, 8-register allocation unit)
  H's 47 registers allocate 48 × 32 = 1,536 per warp → **40 resident warps = 10 blocks per
  SM**; K's 35 → 1,280 → the 48-warp cap, 12 blocks. So at 288 blocks (6 per SM, 24 warps)
  registers bind neither kernel; **at 1,152 blocks H is register-limited to 40 warps per SM
  against K's 48 — and was still 0.97 s faster there.** P39's reading stands with that
  correction: the flattened curve was measured with FEWER resident warps, not the same.
  All four seats refuted the sentence; the orchestrator's arithmetic above is the fix.
- **What the oracles cover, stated plainly.** `--verify-b2` compares every lemma's code of
  all 5,226 steps against the SAME kernel binary in the hashed layout; the two paths share
  the lemma-literal inserts, `lrat_insert`, `lrat_code` and `lrat_hint_strided` (also H's
  > 32-literal fallback), so a defect in that shared code would cancel. The kernel-INDEPENDENT
  oracles are SimWarp's `LitTable` (host): the battery (now 6 + 2 shapes, 3 layouts × 4
  block counts, every lemma), the 180-proof archive (4 block counts, every lemma), and
  `--simwarp` on the EDP steps — 50 of 5,226 in the protocol run; **extended to all 5,226
  steps in the bracket that follows.**
- **The timer.** `total` is the launcher's in-process wall from before the CNF parse
  (`t0`) to after the last chunk's verdicts (`t4`): it includes the CNF/composition parse
  (0.007 s), GPU-context creation (0.11–0.19 s), the proof reads and every pipeline stage
  including combine; it excludes process start-up and teardown and the oracles. The
  co-running check stage alone sums to 3.76–3.84 s. Read "checks all 5,226 proofs in 4.67 s"
  as "launcher wall 4.67 s, CNF parse through the last verdict, one process, proofs in page
  cache".
- **The ratios, with the caveats in the same paragraph as §4 requires.** 4.671 s launcher
  wall against the published 60.2 s (another machine, 128 threads, the authors' proofs, an
  older cake_lpr, a verified checker against an unverified one; 5,226 processes re-parsing
  the CNF) = 12.9× (cold 4.852 s: 12.4×); against this machine's reruns on OUR proofs,
  one process per step, one niced core: **lrat-check 199.6 s (text proofs) = 42.7×,
  cake_lpr 366.9 s = 78.5×.** The A2 figure, 5.774 s, is on proofs regenerated HERE with
  the published pipeline (iglucose + drat-trim -L, §7), not on the published proof files:
  10.4× under 60.2 s with the same caveats.
- **The hint-metadata count was overstated.** The hint ids load coalesced (32 per line);
  `deleted_at[hc]` is scattered; `clause_start[hc]` and `[hc+1]` are adjacent (one line
  unless straddling). ≈ **2 scattered lines per hint, ≈ 160 per lemma** — an upper bound
  with no profile behind it, still marked reading. "A seventh" is an eighth (0.44 / 3.44);
  1.15× from §9's 3.447, 1.14× from the same-day K rerun's 3.422.
- **Units.** The sampler reports MiB: the peak was **5,130 MiB** (5.38 GB, 5.01 GiB). P41's
  "≤ 5.3 GB (5.16 today)" and §9's "5.16 GB" used the same mistaken convention for 5,162
  MiB, so the verdict stands in consistent units (5,130 ≤ 5,300 MiB).
- **The `kn > slots` test.** The batched path tests a literal AND its negation against the
  region bound before the free count; the strided loop tests the literal there and the
  unit's negation at insert time. For an even region the two agree exactly (codes 2|l| and
  2|l|+1 lie on the same side of an even bound); `lrat_direct_slots` returns multiples of
  32, and the launcher now asserts evenness, so the divergence (code 5 where the strided
  loop says 0 or 1) is unreachable. Kernel unchanged; the measured PTX is the committed PTX.
- **Shape (d) was one-directional.** `(33 ∨ 35)` after the 33-wide hint is NotUnit whether
  or not `¬33` was stored, so it killed M-H4 but never tested that the strided fallback's
  insert is seen by the batch that follows. Added `(¬33 ∨ 35)` and `(¬70 ∨ 71)` as "wide 33 /
  70 propagates" (SimWarp: NoConflict; an invisible store would read NotUnit). The battery
  is 6 + 2 shapes; the built (e) is "both lemma literals in the hint clause", not the
  sentence's "own literal negated". Full gpu suite on the current tree: **292 passed, 0
  failed, 13 ignored** (`warp-types-sat` lib target; 13 more in the other targets) — the
  "289 + 3 = 292" above was two invocations of this count; §9's "291" lacked one
  `archive_agreement_*` test that `LRAT_CHECK` now enables.
- **Stamps.** The shape-(b) bracket was written at 20:3xZ, AFTER the contended run (20:20Z)
  and before the clean one; its content (cap before lookup) dates from the build at 20:17Z.
  The "22:5xZ" bracket was written at 22:4xZ (commit 22:47Z). The archive oracle prints
  `SKIPPED` when `LRAT_ARCHIVE_DIRS` is unset — by design, the archive lives outside the
  tree; the vacuous pass that was fixed is the set-but-empty case.
- **The drift sentence** overreached: H measures 0.104–0.107 s where K measured 0.117–0.124 s
  on the same files the same day; the cause of K's drift since 11-04 stays open.
- **Tally wording:** 7 of the 9 pre-registered ranges P37–P45 MET (P4 and the cold bound are
  clauses of P37); P38 and P44 MISSED; P46 reported.]
[SIMWARP ON EVERY STEP, 2026-11-11 03:54–03:59Z (`simwarp-all.txt`, host load 4.2 — not a
timing run): `edp_resident --simwarp 5226` re-decodes every one of the 5,226 steps against
the real formula on the host and compares SimWarp's full per-lemma code vector with the
resident pipeline's: **5,226 steps compared, 0 mismatches**, 5,226/5,226 verified. The
kernel-independent oracle now covers every lemma of every EDP step, not 50 of them; with the
battery (8 shapes) and the 180-proof archive it is the whole of what P40 rests on.]

[PANEL ROUND 2, 2026-11-11 (`round2/`; 0 CRITICAL, 6 MODERATE, 15 LOW raw; Opus and Grok caught the
defect below, Fable and composer-2.5 passed it — the split was the signal again):
- **Round 1's shape-(d) fix was vacuous, and I had marked it FIXED.** Growing the (d) CNF from
  6 to 8 clauses left the lemma at id 7; `flatten` rejects `id ≤ cnf.len()` as IdNotIncreasing
  at step 0, so all six (d) shapes — the four old ones and the two new — flattened to NO
  lemmas, and `codes_equal` compared empty against empty on both sides. The "292 passed" of
  round 1 covered that, and the battery had no per-shape guard against it: the same check-
  that-cannot-fail the archive test had the day before, re-made by hand within hours of fixing
  it. Fixed: lemma id 9; the battery now asserts that every shape produces lemmas, allowing a
  lemma-less result only for a hint corruption planted in the FIRST lemma (the host rejects at
  step 0 by design; exactly 1 such shape, counted in the test's output). Re-verified on the
  fixed tree: battery PASSES (codes seen Ok 66 / NotUnit 4,048 / NoConflict 2,776 / Deleted
  11,714 / Forward 9 / Overflow 7); mutants on copies, each watched to fail: M-H1 (killed by
  "small 0"), M-H2 ("duplicate units near the cap 4093"), M-H3 ("synthetic e=false"), **M-H4
  ("wide 33 then" — the (d) shapes are live again)**, M-H5 (the strided fallback's insert
  dropped in every layout; killed by a hashed shape), and **M-H6 (the fallback's insert dropped
  in the DIRECT layout only; killed by "wide 33 propagates": kernel 1 NotUnit vs SimWarp 2
  NoConflict)** — the defect the old (33 ∨ 35) shape could not see, now seen. Kernel source
  unchanged except comments; PTX identical; examples rebuilt.
- **The negation probe costs no extra line.** Codes 2|l| and 2|l|+1 are adjacent 8-byte slots
  at a 16-byte-aligned offset, so a literal's slot and its negation's always share one
  128-byte line in the direct layout. "≈ 190 lines per lemma" was false and the follow-on
  idea "probe the negation only for lanes still free" rested on it — withdrawn. The reading
  now: ≈ 2 scattered metadata lines per hint plus the literal line; nothing profiled.
- **Ratios, same paragraph.** Dated inline pointers now sit beside every ratio sentence above
  (append-only kept: nothing removed, each addition dated). One more caveat belongs with the
  same-machine ratios (§0.2): lrat-check's 199.6 s and cake_lpr's 366.9 s each pay 5,226
  process start-ups and CNF parses that the resident pipeline pays once — part of the 42.7×
  and 78.5× is that.
- **The K rerun's check-thread idle** (run-all2 runs 2–4) is 0.116 / 0.130 / 0.202 s — "0.12–
  0.20", not the "0.17–0.25" written in the round-1 bracket.
- **The even-region assert** is now gated on the direct layout (the hashed path ignores
  `slots`); `assert!` is not stripped in release builds (one seat said it was — it is
  `debug_assert!` that is), and it is host-side: the measured binary predates it, the PTX is
  the same. SimWarp's `LitTable` is hashed-only on the host: the oracle is independent in
  implementation (host, open addressing) and the agreement with the direct batched path is
  empirical — the battery, the archive, and all 5,226 EDP steps — not a semantic clone.
- **Comments fixed:** the kernel header no longer says every lane performs every insert (the
  batched path's unit lane alone stores; the others read after the `syncwarp`); the
  launcher's "24 blocks (96 warps) per SM" now says 24 requested, residency 12 (K) / 10 (H).
- **P40's sentence** "the suite's count grows by the new battery shapes" was wrong in kind:
  the shapes live in one test; pointer added above. The inline marker at shape (b) is dated.]
