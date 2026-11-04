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
