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
