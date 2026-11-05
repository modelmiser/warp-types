# LRAT binary decode on the device — pre-registration

*Written 2026-11-03, after the checker arc closed (`research/gpu-lrat-checker.md`)
and before any code for this one. Venue: the local RTX 4000 SFF Ada (compute-only
card, allowed by default since 2026-11-03; RunPod only when the owner names it).*

## 0. Why this

On the 415 MB binary proof (`r3sat_n350_s3.blrat`: 4.77 M steps, 3.22 M lemmas, <!-- unguarded: 415 — proof size in MB, local chain log -->
105.7 M hints, 53.2 M literals) the local pipeline is read 0.140 s + host decode
0.808 s (6 threads) + device check 0.275 s (upload of ≈ 700 MB of flat arrays +
launch + download) ≈ 1.22 s. The decode is two thirds of it; on the H200 pod it was
0.88 s of 1.6 s. The host decoder is already the parallel speculative one
(`FlatProof::from_binary_par`): slices parsed independently, the true chain
stitched across slice boundaries, payloads copied in parallel. Its three phases
map directly onto a device formulation, and the flat arrays it produces are
exactly what the check kernel reads — so if the decode happens on the device the
≈ 700 MB upload of the check stage disappears too. <!-- unguarded: 415, 4.77, 3.22, 105.7, 53.2, 700 — sizes from the local chain log -->

## 1. What will be built

**Device-resident bytes.** The proof bytes are uploaded once (415 MB ≈ 0.03 s on
PCIe 4 ×16). Everything below runs on them in place.

**K1 `lrat_scan_slices` — a warp per slice** of `S` bytes (`S` a launch parameter,
default 4 KB; 415 MB → ≈ 101 k slices). The warp walks its slice 32 bytes at a
time, one byte per lane:

- *tokens*: a byte with the high bit clear ends a token (a varint, or the one-byte
  tag). `ballot(term)` gives the token ends of the window; a lane-guarded
  inclusive scan numbers them; the lane at a token's last byte reconstructs its
  value by reading back at most 10 bytes (more = the one-thread decoder's
  `shift > 63` error).
- *roles*: a 5-state machine over tokens — TAG →(`a`) ID →LITS →(zero) HINTS
  →(zero) TAG; TAG →(`d`) DELS →(zero) TAG; anything else in TAG is the
  `expected 'a' or 'd'` error. Each lane holds its token's transition table
  (5 states × 3 bits); a warp scan composes them, so every token knows the state
  before it from the state at window entry; the last lane's state carries to the
  next window. This is the parallel-FSM-by-scan construction; it is what the
  `ballot` / `shuffle_up` / `broadcast_lane` vocabulary is for.
- *speculative start*, as on the host: the first `a`/`d` byte that follows a
  0x00 inside the slice (offset 0 for the first), found by one ballot per window;
  the walk runs past the slice end until the step in progress completes, exactly
  as `parse_run` does.
- *two passes*: pass 1 counts steps / literals / hints / deletion-ids per slice
  and records where the chain stopped and whether a format error occurred; the
  host prefix-sums the counts (101 k numbers); pass 2 walks again and writes the
  step records (`pos`, `id`, `pos_after_id`, running `lit_end`/`hint_end`/
  `del_end`, `tag`, `rat`) and the payloads at exact offsets.

**Host phases 2 and 3a unchanged.** The step records (≈ 24 B each; 115 MB for the
415 MB proof) are downloaded and the existing stitch (follow the true chain across <!-- unguarded: 415 — proof size in MB, local chain log -->
boundaries, bridge where a worker was mis-synchronised) and bookkeeping (ids
increasing, dense map, deletions, first empty clause, earliest verdict) run as they
do today. **Any slice the device marks dirty** (a format error, a varint longer
than 10 bytes, a step that outran a fixed window budget) **is re-parsed on the
host by `parse_run` from the same start**, so the device has to agree with the
host only on slices it calls clean — a true-chain error happens at most once, and
garbage in a mis-synchronised slice is bounded by the slice. <!-- unguarded: 24, 115 — record size and total -->

**K3 `lrat_gather_hints`** replaces phase 3b: literal payloads of the true chain's
segments are already in place; hints are mapped id → dense index on the device
with the `dense_of` table phase 3a built (the first hint naming an id never added
is found by an atomic min, as the host does). The check kernel then runs on the
device-resident arrays; the host sees step records and codes, never literals or
hints.

**K0 `lrat_scan_slices_serial` — the control.** One *thread* per slice, running
the host's `parse_step` loop verbatim, no collectives, same two-pass protocol.
It is "a GPU without the warp": if K1 is not clearly faster than K0, the warp
vocabulary bought nothing for decoding and the log will say so.

**SimWarp model first.** `lrat_warp::decode_warp_model` runs K1's window / ballot /
scan algorithm on `SimWarp` on the CPU and is held equal to `from_binary` on the
whole parallel-decoder battery (valid proofs at six slice sizes, every truncation,
every single-bit flip, the `0x00 'd'` mimic) before the PTX is first run. The
kernel is a transcription of the model, as the check kernel was.

**A correct warp scan.** `Warp::inclusive_sum` / `exclusive_sum` in the core crate
are deprecated as *incorrect on every target* (Hillis–Steele without the lane
guard; `src/cub.rs`). The kernel needs a correct one; it is written in the kernel
crate with `lane_id()` and `shuffle_up` and tested on SimWarp and on hardware.
Whether it replaces the deprecated core method is the owner's call (public API).

## 2. Oracles and controls — each watched to fail before being believed

- Equality oracle: `FlatProof::from_binary` (one thread), through `assert_same_result`
  / `assert_flat_eq` as for the host parallel decoder — same `Result`, same
  `early` verdict, byte-identical arrays.
- Watched to fail: (a) the deprecated `inclusive_sum` must FAIL the scan test the
  new one passes; (b) a model with the state carry between windows broken must
  fail the valid-proof battery at a slice size that splits a step; (c) the mimic
  proof must force a bridge on the device path (a worker that starts on the
  `0x00 'd'` must be overruled by the chain).
- The archive (`LRAT_ARCHIVE_DIRS`, 632 files: 114 genuine, 66 drat-trim outputs,
  452 hint-level mutants) through the device path, compared with the reference
  and with `lrat-check` / `cake_lpr` verdicts already on file. <!-- unguarded: 632, 114, 66, 452 — archive composition -->
- Timing: `Instant` around each stage in `examples/lrat_check.rs`, best of 3,
  bytes already in RAM (the read is a separate line, as today).

## 3. Predictions (written before any code)

- P1 **Equality**: the device-decoded `FlatProof` equals `from_binary`'s on all
  632 archive files and on the whole battery. Kill: any difference — the device
  path has no tolerance; a disagreement is a kernel bug, not a rounding.
- P2 **Decode time** on the 415 MB proof, bytes in RAM → flat arrays on the device <!-- unguarded: 415 — proof size in MB, local chain log -->
  + step records on the host: ≤ 0.25 s on the Ada, i.e. ≥ 3× the 6-thread host
  decode (0.808 s). Kill: not below 0.808 s after one tuning round (slice size,
  blocks) → "the decode does not pay on this design", recorded as such.
- P3 **Pipeline** bytes in RAM → verdict on the 415 MB proof: ≤ 0.5 s (today <!-- unguarded: 415 — proof size in MB, local chain log -->
  1.08 s), and the check stage itself ≤ 0.15 s once its ≈ 700 MB upload is gone
  (today 0.275 s).
- P4 **Warp vs thread**: K1 ≥ 2× K0 at equal slice size (coalesced 32-byte
  windows against 32 lanes striding 4 KB apart). Range on purpose: below 1.2× the
  honest reading is "the GPU helps, the warp does not".
- P5 **Hardware finds what the model cannot**: the model passes the battery on
  SimWarp before PTX, and the first PTX run still shows at least one defect the
  model could not (the H200 precedent: undefined externs, a missing barrier). If
  the PTX agrees first time, P5 missed — pleasantly, and it goes in the ledger.
- Error-sign ledger kept per the house rule.

## 4. Not in scope

RAT steps (still `Unsupported`); streaming the file read under the upload (a
later arc — the read is 0.14 s here and 0.45 s on the pod); the text format; any
change to the host `from_binary_par`, which stays the CPU route and the oracle's
parallel twin.

## 5. Log (append-only)
- 2026-11-03 — **S1 done: the SimWarp model** (`warp-types-sat/src/lrat_decode_model.rs`, a child module of `lrat_warp` so it shares the host decoder's phase-1 records; the host decoder was split into `slice_bounds` / `parse_slices_host` / `stitch_chunks`, the stitch now producer-independent — 15/15 existing tests unchanged). The model is lane-wise, not a disguised byte walk: `ballot` for token ends, a lane-guarded Hillis–Steele scan composing 18-bit transition tables for the role machine, `popcount(ballot & lanes_below)` for output slots, warp-uniform carry (state, token start, counts, rat flag) between windows. **Contract held slice by slice**: on every slice of the synthetic proof at eight slice sizes, on every truncation at three, and on bit flips (bits 7 and 0 at every byte in the default suite; all eight bits at two sizes in an `#[ignore]`d exhaustive test, run once in release: passed) the model's record is the host's `parse_run` record, or the model says dirty and the host's record carries an error — never a clean, different record; end-to-end `FlatProof` equality through the shared stitch on all of it. Long varints: 10-byte literal decoded, 11-byte dirty with the host erroring. **Watched to fail**: the deprecated-shaped unguarded scan is seen to be wrong on random tables (the test asserts it); mutant 1 (state carry reset between windows) killed by the straddle test; mutant 3 (token-start carry wrong) killed by the same; **mutant 2 (rat carry dropped) SURVIVED the battery** — a RAT hint needs a step spanning three windows with the negative hint in the middle window, which no test had; `rat_hint_anywhere_in_a_step_spanning_windows` added (80 hints, the negative one at six positions, three slice sizes), kills it. Cost: the model simulates 32 lanes on one thread, ≈ 100× the host decoder per byte — hence the sampled default flip test (≈ 10 s release). <!-- unguarded: 15, 18, 10, 11, 80 — test tallies and byte counts in this entry -->
- 2026-11-03 — **S2 done: the kernels** (`sat-kernels::lrat_scan_slices`, warp per slice; `lrat_scan_slices_serial`, thread per slice, the control; `GpuContext::decode_lrat_slices` two-pass driver; `FlatProof::from_binary_device` through the shared stitch). The PTX loaded and agreed at the first run — 0 externs, both kernels listed — on the whole device battery (valid / truncations / bit flips / straddling steps / RAT across windows, both kernels, slices 1–4096) **and on the archive: 632/632 proofs, both kernels, two slice sizes** (`archive_agreement_device_decode`; `LRAT_ARCHIVE_DIRS` = run1 : run2-default2000 : run3-cake-cnfuzz : negatives). On the 415 MB proof the kernels take 0.08 s (warp: pass 1 0.027 + pass 2 0.052) and 0.10 s (thread: 0.039 + 0.065) — but the stage-2 shape downloads ≈ 1.3 GB of payload and rebuilds it on the host, so the total (0.96 s) equals the host decode (0.90 s today). EQUAL to the host decode at three slice sizes. <!-- unguarded: 632, 415, 1.3 — archive count, proof size in MB, download size in GB -->
- 2026-11-03 — **S3 done: device-resident payloads** (`stitch_plan` / `stitch_payloads_host` split of the stitch — phases 2 + 3a over step records only; `lrat_gather_lits` / `lrat_gather_hints` kernels placing the true chain's segments by binary search over the destination prefix, hints mapped through the dense table with an atomic minimum for the first id never added; bridges and host-parsed slices ride along in a small extra buffer; `GpuContext::check_lrat_codes_resident` runs the check kernel on the resident arrays; `combine_resident` fetches the one hint range a verdict names). `device_resident_equals_sequential`: meta + downloaded arrays + verdicts equal to the sequential decoder on the battery, both kernels. **Measured, local RTX 4000 SFF Ada, bytes in RAM, best of 3** (`examples/lrat_check` with `LRAT_GPU_RESIDENT=1`; arrays verified EQUAL to the host decode once per run on both proofs):

  | proof | host decode (6 thr) | old check (with ≈ 700 MB upload) | old bytes→verdict | resident decode | of which plan (host) | resident check | **bytes→verdict** |
  |---|---|---|---|---|---|---|---|
  | 415 MB (4.77 M steps) | 0.901 s | 0.284 s | 1.185 s | **0.299 s** (upload 0.061, pass 1 0.027, pass 2 0.052, download 0.019, plan 0.107, gather 0.021) | 0.107 s | 0.226 s | **0.525 s** (2.3×) |
  | 898 MB (10.2 M steps) | 2.035 s | 0.664 s | 2.699 s | 0.887 s (0.078 / 0.077 / 0.129 / 0.172 / 0.337 / 0.050) | 0.337 s | 0.538 s | 1.426 s (1.9×) |

  Thread-per-slice control, same shapes: pass 1 + pass 2 0.105 s vs the warp's 0.079 s on 415 MB (1.33×); 0.249 s vs 0.206 s on 898 MB (1.21×). **One tuning round** (slice 1024 / 4096 / 16384 / 65536 × blocks 4096 / 8192 on the 415 MB proof): nothing beat slice 4096 / blocks 4096; at 65536-byte slices the thread kernel falls to 0.29 s while the warp kernel holds 0.085 s (few long slices starve 6,339 threads but not 6,339 warps), at 1024 the host plan grows with the slice count. <!-- unguarded: 415, 898, 4.77, 10.2, 700, 6,339, 1024, 4096, 16384, 65536 — sizes, counts and sweep points from the tune-*.txt files in staging -->
  **Ledger.** P1 **MET** (632 archive + battery + both big proofs' arrays, every comparison EQUAL). P2 **MISSED on the number written** — 0.299 s against ≤ 0.25 s; the ratio clause (≥ 3× the host decode) is 3.0× against today's host 0.901 s and 2.7× against the 0.808 s the prediction quoted; on 898 MB 2.3×. The miss is in the host plan (0.107 / 0.337 s): phases 2 + 3a now walk 4.8 / 10.2 M step records rebuilt as structs from the device's arrays, and that is the next thing to cut, not the kernels (0.08 / 0.21 s). P3 **MISSED on both numbers**: bytes→verdict 0.525 s against ≤ 0.5 s, and the check stage 0.226 s against ≤ 0.15 s — the ≈ 700 MB upload was only 0.06 s of the old 0.284 s; the check kernel itself is 0.22 s on this card (0.117 s on the H200), and it is now the largest device stage. P4 **MISSED**: warp 1.33× / 1.21× the thread kernel, inside the range the prediction called "the GPU helps, the warp does not" — the GPU decode is ≈ 10× the six host threads either way; the warp's edge appears only when slices are long enough to starve threads. P5 **MISSED, pleasantly**: the PTX agreed with the model at the first run on everything, including the 4,096-slot-free gathers; nothing was found on hardware that SimWarp had not. Sign of the misses: three of four flattered the design by 10–50 %; one (P5) flattered the risk. <!-- unguarded: 0.808, 4.8, 10.2, 4,096 — quoted prediction figure, step counts in millions, table size -->
- 2026-11-04 — **S4 pre-registered: the host plan** (owner, of the open follow-ups: "please proceed with all, in your order of preference"). Measured before any code, 415 MB proof, warp kernel, temporary timers: the plan stage's 0.107 s is ≈ 0.03 s rebuilding device step arrays as `RawStep` structs and ≈ 0.08–0.11 s for phases 2 + 3a, of which phase 2 is < 0.01 s; inside 3a the deletion loop is ≈ 0.04 s and the lemma pushes the rest. **Design.** (a) The rebuild runs per slice on host threads. (b) Phase 3a gets an *optimistic parallel walk*: per true-chain segment, counts and prefixes (steps, lemmas, literals, hints, deletions); one parallel pass that both validates (ids strictly increasing across segment edges, gap bound, no RAT, no empty segment surprises) and writes the lemma arrays and `dense_of` into disjoint `split_at_mut` ranges — ids strictly increasing makes the `dense_of` ranges disjoint, so no `unsafe`; then one parallel deletion pass over atomic `deleted_at` (swap, a non-sentinel previous value = a second deletion) checking that the deleted id's lemma was added at an earlier step. **Any violation, a pending format error, or a bridge with an error sends the whole proof to the existing sequential walk**, which stays the only code that produces an early verdict. Valid proofs, the performance case, take the fast path; invalid ones take the code that is already proven equal. **Predictions.** P6: plan stage ≤ 0.05 s on the 415 MB proof (from 0.107) and ≤ 0.15 s on 898 MB (from 0.337), six threads; decode ≤ 0.25 s on 415 MB as a consequence (P2's number, recorded as met-late if met, P2 itself stays MISSED). P7: equality unchanged everywhere — battery and 632 archive through the resident route — with the fallback taken by every rejected archive proof and by none of the verified ones; a mutant that drops the "added before this step" check, and one that drops the second-deletion check, each killed by a test. Kill: any disagreement. <!-- unguarded: 415, 898, 0.107, 0.337, 0.03, 0.08, 0.11, 0.01, 0.04, 0.05, 0.15, 0.25, 632 — measured timings and the predictions written from them -->
- 2026-11-04 — **S4 done: the host plan, and a protocol correction to S3's numbers.** Built as pre-registered: `plan_3a_parallel` in `lrat_warp.rs` (three passes over groups of true-chain segments: validate + size, write into `split_at_mut` pieces, deletions against atomic `deleted_at`; `None` on anything to reject → the sequential walk), `stitch_plan` takes a thread count, `StitchPlan::fast` and `ResidentStats::fast_plan` say which walk ran, `chunks_from_device` rebuilds slices on host threads. No `unsafe`. The host route `from_binary_par` uses the same plan. **Tests**: `plan_parallel_equals_sequential_walk_on_valid_proofs` (two synthetic proofs × five slice sizes × four thread counts: fast path taken, every plan output equal); `plan_parallel_falls_back_on_every_rejection` (nine rejection kinds mid-proof, three slice sizes, eight thread counts: fallback taken, result equal, Err equal); `archive_plan_parallel_equals_sequential_walk` (632 archive files, two slice sizes). **Watched to fail**: mutant 1 (no "added at an earlier step" check) and mutant 2 (no second-deletion check) killed by the rejection test, as pre-registered; mutant 3, not pre-registered (no id check across group edges), survived the rejection test as first written and was killed only by the bit-flip test — the test now runs eight thread counts, so a violation's two halves land in different groups, and kills it. **The first version of the rejection test was vacuous**: its valid tail began with an id gap beyond the bound, which alone forces the fallback; caught on reading it back, before any mutant was run. The archive test found a real edge: `parse_slices_host` divided by zero on an empty proof (production routes guard empty input first); fixed. Full gpu suite with `LRAT_CHECK` / `CAKE_LPR` / the four archive dirs: 275 passed after the fix. Archive tally (walk, reference verdict): 147/147 verified proofs on the parallel walk, 106 rejected proofs fell back, 379 rejected proofs passed the plan. [CORRECTED 2026-11-04, the S4 pre-registration above: P7's clause "the fallback taken by every rejected archive proof" was mis-specified when written. A proof rejected for a bad hint has nothing for phase 3a to reject; the check kernel decides it, so 379 rejected proofs taking the parallel walk is the design working. The clause that could fail, "none of the verified ones falls back", held: 0.] <!-- unguarded: two, 106, 379, 147, 632, 275 — test shapes and archive tallies from this entry -->
  **Measured, same card, interleaved A/B against the pre-change commit (`e0f939a80`, built in a separate worktree and target dir), 415 / 898 MB proofs, warp kernel.** Under the pre-registered protocol (best of 3 repetitions in one process): plan 0.106–0.110 → 0.064–0.066 s on 415 MB and 0.303–0.316 → 0.141–0.190 s on 898 MB, but on 415 MB the decode did not move (0.287–0.302 → 0.302–0.310 s) because the step-record download rose from 0.019 to 0.064–0.073 s. Not the allocator's arenas (`MALLOC_ARENA_MAX=1` changed nothing); the new binary takes ≈ 254 k more minor page faults per run. The old code rebuilt 101 k small step vectors on the main thread and freed them, and the next repetition's download landed in that warm heap; the new code rebuilds on worker threads. **So S3's 0.019 s download, and with it the published 0.299 s decode and 0.525 s bytes→verdict, were repetition artefacts**: a one-shot verification always downloads into cold memory. Single-repetition processes, three each, alternating: <!-- unguarded: 415, 898, 254, 101 — proof sizes in MB, fault count in thousands, slice count in thousands -->

  | proof | code | decode | download | plan | check | bytes→verdict |
  |---|---|---|---|---|---|---|
  | 415 MB | before (S3) | 0.385–0.420 s | 0.068–0.079 | 0.155–0.176 | 0.232–0.237 | 0.617–0.658 s |
  | 415 MB | after (S4) | **0.326–0.337 s** | 0.078–0.079 | 0.083–0.090 | 0.230–0.238 | **0.556–0.574 s** |
  | 898 MB | before | 0.893–0.904 s | 0.172–0.173 | 0.332–0.342 | 0.550–0.553 | 1.443–1.455 s |
  | 898 MB | after | **0.731–0.763 s** | 0.138–0.172 | 0.200–0.205 | 0.548–0.552 | **1.279–1.312 s** |

  The host decode in the same processes: 0.90–0.94 s (415 MB) and 1.97–2.17 s (898 MB), single runs both before and after (its plan is now parallel too; it moved by ≈ 0.02–0.1 s). **Ledger.** P6 **MISSED**: 0.064 s against ≤ 0.05 s on 415 MB; 0.141 and 0.190 s in two rounds against ≤ 0.15 s on 898 MB; the decode consequence (≤ 0.25 s) not met under either protocol (0.30 s warm, 0.33 s cold). P7 **MET** on equality everywhere and on the clause that could fail (no verified proof fell back); its other clause was mis-specified, see the bracket. The plan stage roughly halved; what it bought on the 415 MB proof's one-shot verdict is 0.06–0.08 s. **What is left on the host side** of a cold decode is the step-record download (0.08 s, ≈ 250 MB of records page-faulting into fresh memory) and the walk (0.09 s); moving phase 3a itself to the device would remove both, and is the next cut if this axis continues. Data: `~/staging/2026-11-04-host-plan/` (`ab-*`, `cold-*`, `arena1-*`). <!-- unguarded: 632, 147, 106, 379, 275, 254, 101, 250, 0.019, 0.299, 0.525, 0.064, 0.066, 0.106, 0.110, 0.141, 0.190, 0.303, 0.316, 0.287, 0.302, 0.310, 0.073, 0.90, 0.94, 1.97, 2.17, 0.02, 0.1, 0.05, 0.15, 0.25, 0.30, 0.33, 0.06, 0.08, 0.09 — tallies and timings from the files named in this entry -->
- 2026-11-04 — **S5 pre-registered: the check kernel** (the same "proceed with all"; the check is now the largest device stage, 0.23 s on 415 MB, 0.55 s on 898 MB). Reading `lrat_check_lemmas` before any measurement: a block is ONE warp, so on this card (sm_89: at most 24 resident blocks and 48 warps per SM) residency is capped at half; and each hint starts with a dependent chain — load the hint, then its `deleted_at` and `clause_start` pair — before the lanes touch a literal, with the assignment table in global scratch. Rough latency arithmetic (≈ 33 hints and ≈ 16 literals per lemma, a few L2 round trips per hint, 1,152 resident warps) lands within 2× of the measured 0.23 s, so the kernel reads as latency-bound, not bandwidth-bound. **Design.** (A) `LRAT_WARPS_PER_BLOCK` warps per block (4), each warp its own lemma stream and its own table region — the per-warp code is unchanged; (B) the hint metadata is fetched 32 hints at a time, one per lane (hint id, its `deleted_at`, its clause start and end), and each hint's four values reach the warp by `broadcast_lane` instead of by a dependent global load. Shared memory is NOT used: warp-types has no device-side shared-memory primitive and the hook rejects `.extern` declarations; that is its own arc if ever. **Oracle first:** a new gpu test holds the device's per-lemma codes equal to `check_warp_codes` (SimWarp) — not only the combined verdict, which is all the resident test compares — on the synthetic proofs, the steps-after-⊥ cases, a lemma past the 4,095-literal table cap and a lemma whose units push past it, and the 632 archive files, at block counts 1, 7, 64 and 4,096; watched to fail on a planted kernel mutant before the kernel is changed. **Predictions** (check stage as the example reports it, best of 3, and cold single runs both reported): P8 (A) alone ≤ 0.18 s on 415 MB; P9 (A)+(B) ≤ 0.13 s on 415 MB and ≤ 0.30 s on 898 MB; P10 codes equal everywhere, before and after. Kill: any code disagreement. Not measured: the H200 (local card only). <!-- unguarded: 415, 898, 24, 48, 33, 16, 1,152, 4,095, 632, 7, 64, 4,096, 0.23, 0.55, 0.18, 0.13, 0.30 — kernel-reading arithmetic, device limits, test shapes, predictions -->
- 2026-11-04 — **S5 interim, before steps C and D are written.** Measured on the 415 MB proof, best of 3, `LRAT_GPU_BLOCKS` swept (the block count also sets the decode kernel's, which does not move). **(A) refuted its own premise**: at the example's 4,096 blocks four-warp blocks made the check SLOWER, 0.23 → 0.37 s; at an equal total of 4,096 warps (1,024 blocks) still 0.35 s. Capping residency by launching fewer blocks: 0.301 s at 8 warps/SM, **0.222 s at 16**, 0.238 s at 24, ≈ 0.35 s at 48. More warps per SM is slower past 16 — the kernel is not short of latency hiding; it is contending for the memory system. **(B)** on top: 0.258 / **0.196** / 0.223 / 0.370 s at 8 / 16 / 24 / 48 warps per SM — a 10–15 % gain at the best residency. Codes equal throughout (the S5 oracle at 1/7/64/4,096 blocks after each change). P8 is therefore MISSED (A alone never beat the baseline at any residency by more than 4 %) and P9 is on course to miss. **Reading:** each warp's table holds ≈ 50 live entries scattered over 4,096 slots in TWO arrays (key, stamp), so a lemma touches ≈ 100 distinct 128-byte lines; 16–24 warps per SM want 200–300 KB against an L1 of at most 128 KB. **Pre-registered now, before code:** (C) key and stamp packed into one 64-bit slot — one load and one line per probe, the footprint halved; (D) when every literal code (`2·|lit| + sign`) of the proof is below 4,096, the table is indexed by the code directly — one load, no probing, ≈ 32 codes per line — with the SAME occupancy count and 4,095 cap, so `TableOverflow` falls on exactly the same lemmas; the host computes the largest code (on the device for the resident route) and picks the mode per launch. D is a small-variable-count specialisation: both big proofs have 350 variables, and every CNF in the archive has between 21 and 580 (checked: none above 2,047), so the whole archive would run on D's direct path; proofs with more variables keep the hashed table. **Consequence for the oracle:** the per-lemma code test forces each proof through BOTH table modes, so the hashed path keeps its full coverage. Predictions: P11 (A+B+C) check ≤ 0.15 s on 415 MB at its best block count; P12 (D) ≤ 0.10 s on 415 MB and ≤ 0.25 s on 898 MB, and the best block count moves UP (a smaller footprint tolerates more warps). P10 (codes equal everywhere) stands for every step. <!-- unguarded: 415, 4,096, 1,024, 8, 16, 24, 48, 50, 100, 128, 200, 300, 4,095, 350, 21, 580, 2,047, 0.23, 0.37, 0.35, 0.301, 0.222, 0.238, 0.258, 0.196, 0.223, 0.370, 0.15, 0.10, 0.25 — measurements, cache arithmetic and predictions in this entry -->
- 2026-11-04 — **S5 done: the check kernel.** Built: packed 64-bit slots (C), the direct layout with `lrat_max_code` choosing it per launch (D), on top of (A) and (B); `check_lrat_codes_mode` exposes the layout; `blocks = 0` picks 24 blocks per SM for the direct layout and 6 for the hashed one; the non-resident launcher now calls the shared one. **Oracle**: the per-lemma code test forces every layout the proof admits — the whole archive admits both — at 1/7/64/4,096 blocks, on the battery (all six codes produced, both overflow shapes) and on the 632 archive files: equal to SimWarp everywhere; full gpu suite 277 passed. The direct layout cannot overflow (codes below 4,096 leave at most 4,094 distinct entries against a cap of 4,095), and SimWarp's table agrees by the same count. **Watched to fail**: `total >= 3` for the not-unit test (killed); direct lookups ignoring negative literals (killed); a third mutant on the direct insert's "already present" test survived and is EQUIVALENT — in the direct layout a stamped slot always holds its own key — so it tests nothing; recorded, not counted. **Sweep, 415 MB proof, best of 3** (block count → warps per SM in brackets): hashed + packed (A+B+C) 0.233 / 0.159 / **0.141** / 0.161 / 0.196 s at 96 / 192 / **288** / 384 / 576 blocks (8 / 16 / 24 / 32 / 48); direct (D) 0.193 / 0.127 / 0.099 / 0.088 / **0.079** / 0.083 / 0.087 s at 96 / 192 / 288 / 576 / **1,152** / 2,304 / 4,096. 898 MB: hashed 0.349 s at 288 blocks, direct 0.224 s at 1,152. **Cold single runs, three each, alternating against the pre-S5 kernel (`e3e948b86`, the oracle commit), launcher-chosen layout and block count:** <!-- unguarded: 415, 898, 632, 277, 96, 192, 288, 384, 576, 1,152, 2,304, 4,096, 4,094, 4,095, 8, 16, 24, 32, 48 — proof sizes, test tallies and sweep points from this entry -->

  | proof | kernel | check | bytes→verdict |
  |---|---|---|---|
  | 415 MB | before | 0.227–0.234 s | 0.537–0.564 s |
  | 415 MB | after (direct) | **0.085–0.094 s** | **0.407–0.420 s** |
  | 898 MB | before | 0.543–0.547 s | 1.269–1.296 s |
  | 898 MB | after (direct) | **0.226–0.227 s** | **0.977–1.006 s** |

  **Ledger.** P8 **MISSED** — and its premise refuted: four warps per block never helped by itself; the kernel was contention-bound, so more resident warps made it slower. P9 **MISSED** (A+B best 0.196 s vs ≤ 0.13). P10 **MET**. P11 **MET** (0.141 s vs ≤ 0.15, at 24 warps/SM). P12 **MET** (0.079 s vs ≤ 0.10 on 415 MB, 0.224 s vs ≤ 0.25 on 898 MB; the best block count moved from 288 to 1,152 as predicted). **What the gain is and is not.** Packing (C) is general: any proof gets one line per probe. The direct layout (D) is the larger share and applies only to proofs whose largest variable is at most 2,047; both big proofs and the whole archive qualify, competition-scale proofs mostly do not, and NO proof with more variables has been timed — the hashed layout's 0.141 s is the honest number to expect there, on this card, if its hint and clause shape resembles these. The block-count defaults come from one card's sweep; the H200 numbers in the README predate S5 and were not re-measured. S3's P3 (pipeline ≤ 0.5 s, check ≤ 0.15 s) stays MISSED as recorded; on these proofs both numbers are now met late, cold. Data: `~/staging/2026-11-04-host-plan/` (`s5*`). <!-- unguarded: 632, 277, 4,096, 4,094, 4,095, 96, 192, 288, 384, 576, 1,152, 2,304, 415, 898, 8, 16, 24, 32, 48, 2,047, 0.233, 0.159, 0.141, 0.161, 0.196, 0.193, 0.127, 0.099, 0.088, 0.079, 0.083, 0.087, 0.349, 0.224, 0.13, 0.15, 0.10, 0.25, 0.5 — test tallies, sweep points and timings from the files named here -->
