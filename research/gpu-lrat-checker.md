# A warp-typed LRAT proof checker — pre-registration and log

**Registered 2026-11-02, before any code.** Direction chosen under the owner's "guide our research however you see
fit for maximum potential impact". Append-only; corrections in dated brackets.

## 0. Why this, of everything on the table

Three threads in this repository meet in one artefact. (1) The crate's thesis is that warp-level GPU code can be
*typed* so that lane-safety and uniformity are legible; its flagship kernel today is the gradient solver, which
yesterday's measurements showed to be a research path with no solver benefit — a weak flagship. (2) The last two
days turned every verdict this stack produces into something an outside tool can check: DRAT → LRAT → `cake_lpr`
for SAT, independent fence checks and trace replay for PDR/BMC. (3) The hatt reading programme's own conclusion
is that the one place GPUs demonstrably help SAT is *not* search but embarrassingly parallel side work —
preprocessing and proof checking (`CPU+GPU Hybrid Solving`, `Speeding Up … Where the Real Leverage Is`:
"faster, more compact proof formats and streaming checkers are useful and underrated").

An LRAT proof is a list of lemmas, each with the exact hint clauses that make it a unit-propagation consequence.
Checking one lemma reads only its hints; nothing about lemma *i* depends on the *result* of checking lemma *j*.
That is a warp-per-lemma (or lane-per-lemma) job with a binary, externally checkable answer. Prior art, as far as
two searches on 2026-11-02 show: no GPU LRAT or DRAT checker exists; Osama & Wijs (FMSD 2023) generate proofs on
the GPU and check them on the CPU; the LRAT paper (Cruz-Filipe et al. 2017) and `cake_lpr` (Tan, Heule, Myreen
2021) both note that LRAT checking can be parallelised; PalRUP (Götz, Dörr, Schreiber, SAT 2026 — local copy) does
it across 3,072 CPU cores with a parallel file system, not on a GPU. **Impact case:** a kernel that exercises
`warp-types`' collectives (ballot, reductions, lane-uniform control) on a task whose correctness is settled by
`cake_lpr` and `lrat-check` byte for byte, with a measurable throughput claim against the CPU state of the art.
If it does not pay, that is the second GPU-for-SAT negative of this programme and worth exactly as much.

## 1. What will be built

- **`lrat.rs` in `warp-types-sat`** — a reference checker in plain Rust: parses text LRAT (`drat-trim -L` output)
  and binary LRAT, checks with `lrat-check`'s semantics (RUP lemmas via hints in order; deletions honoured; a
  hint referencing a deleted or later clause is an error; the proof must derive the empty clause). RAT lemmas
  (negative hints) are **out of scope for the GPU path** and rejected as "unsupported" by the reference checker
  with a distinct verdict, never silently accepted — every proof this stack produces is RUP-only (`drat-trim`
  reports 0 RAT lemmas on all 114).
- **Parallel formulation.** Prepass (sequential, cheap): clause arena by ID; for each clause the index of the
  lemma that deletes it (∞ if never). Per lemma: a local assignment set (lemma literals false), then hints in
  order; each hint clause must be unit or conflicting under the current set; a hint is admissible iff
  `hint.id < lemma.id` and `delete_index(hint) > lemma.index`. Lemmas are independent given the arena, so they
  are checked in parallel; verdict = every lemma OK ∧ an empty lemma exists.
- **Warp version on `SimWarp` first** (CPU, deterministic): one warp per lemma; lanes stride over a hint clause's
  literals; `ballot` of "not falsified" → popcount 0 = conflict, 1 = unit (the single set lane publishes it),
  ≥ 2 = FAIL; the assignment set lives in a per-warp table in shared memory. The warp-uniform control flow is
  what `warp-types`' `Warp<All>`/`Uniform` are for, and the ballot-then-select is the `SingleLane` pattern.
- **GPU version in `sat-kernels`** (nvptx), launched through the existing `cudarc` path, *after* the SimWarp
  version agrees with the reference on everything below.
- **Typed verdict.** The checker's `Verified` value is only constructible from the per-lemma results plus the
  empty-clause witness; `Rejected` carries the first failing lemma id and reason. The SAT CLI's `--certify` gains
  an in-crate checker option, clearly labelled as *not* independent (same repository, same author).

## 2. Oracles and controls — each watched to fail before being believed

- **Agreement:** on every LRAT in the archive (`research/cert-campaign-2026-11-01/`: 114 genuine + 30 corrupted)
  and on the scale corpus (§3), the reference checker, the SimWarp checker and (later) the GPU checker must give
  exactly `lrat-check`'s verdict and `cake_lpr`'s verdict. **Kill: any disagreement on a genuine proof stops the
  arc until explained.**
- **Negative controls** (file-level, through all checkers): flip one literal of a mid-proof lemma; drop one hint;
  reorder two hints so the propagation is no longer unit; reference a clause deleted earlier; reference a later
  id; remove the final empty clause; a RAT lemma (negative hint) must come back "unsupported", not "verified".
- **Fault injection** in the SimWarp kernel (cfg(test), as in `proof::fault`): ballot off by one lane; skip the
  deletion-index test; treat popcount 2 as unit. Each must be caught by the agreement test on the archive.

## 3. Scale corpus

Checking cost on our proofs is not a bottleneck (`lrat-check`: 42 MB / 288k lines in 0.85 s; `cake_lpr` 2.1 s),
so throughput claims need proofs of 100 MB–several GB. Sources: kissat 4.0.1 with `--proof` on UNSAT random
3-SAT at n = 300–450 and on HWMCC-derived CNFs, trimmed to LRAT by `drat-trim -L` (slow, offline); our own
`solve --drat` on the same, where it finishes. Binary LRAT is produced by our reference checker's writer and
cross-checked against the text form.

## 4. Predictions (written before any code)

- P1 Reference checker agrees with `lrat-check` and `cake_lpr` on 144/144 archive files and on all scale files.
- P2 SimWarp checker agrees with the reference on everything (it is the same algorithm reorganised; a
  disagreement is a kernel bug and a kill).
- P3 Throughput, text LRAT, CPU: `lrat-check` ≈ 50 MB/s single core (measured 0.85 s / 42 MB); our reference ≈
  within 2× of that. Parsing dominates; this is why the GPU claim must be made on **binary** LRAT.
- P4 GPU (when run): ≥ 5× `lrat-check`'s per-byte rate on binary LRAT ≥ 100 MB, *including* host→device
  transfer; below 2× the honest verdict is "GPU checking does not pay at this scale on this design". Written
  as a range on purpose; the first measurement sets the real number.
- P5 The typed verdict will not catch any bug the agreement test did not — it is a design property, not a
  finder; recorded so that no such claim is made afterwards.
- Error-sign ledger kept per the house rule.

## 5. Not in scope / owner decisions

RAT/PR lemmas (cake_lpr's LPR); DRAT trimming on the GPU (that is `drat-trim`'s job and a different problem);
any claim about solving speed. **Where the GPU run happens is the owner's call**: the standing rule sends GPU
work to RunPod (billable key), and the local RTX 4000 Ada drives the live cockpit's display; nothing here runs on
either without a word from the owner. Until then every result is CPU/SimWarp.

## 6. Log (append-only)

- 2026-11-02 ~12:40 — **Stage 1 done: reference checker.** `warp-types-sat/src/lrat.rs`: text and binary LRAT (binary layout taken from Lean 4's `bv_decide` parser, the only full description found: tag byte `a`/`d`, varint of `2·|x|+sign`, ids as positive literals), `check_sequential` (lrat-check semantics, deletions applied in order, strict on deleting unknown ids and on hints that end without conflict, RAT → `Unsupported`), and `check_independent` (prepass → deletion index per id → every lemma checked on its own, `threads` at a time, earliest failing step wins). **P1 met: 180/180 agreement with `lrat-check` on the archive** (147 verified, 33 corrupted proofs rejected — the archive has more files than the 144 I counted: the three runs' `flip`/`drop0`/`delorig` variants), and `cake_lpr` agrees on every one (asserted per file). Sequential and independent agree on every file. Ten hand-written tests, each corruption class watched to fail: deleted-before-use, non-unit hint, missing conflict, flipped literal, forward id, non-increasing id, missing empty clause, RAT hint. One of my test expectations was wrong on the first run — I had called a hint order "non-unit" that is in fact unit (clause 2 under ¬b has one free literal); the checker was right and the test was fixed to a genuinely non-unit case. One real bug in the independent checker, found by the archive run: thread slices `[lo..hi]` with `lo > n` for small proofs — clamped.
  **P3 met, and parsing dominates as predicted:** on the 98 MB / 627k-step corpus proof, read 0.04 s, **text parse 0.78 s**, sequential check 0.54 s, independent check on 6 threads 0.36 s; `lrat-check` 1.66 s for the same file. So the reference is ~1.3× faster than `lrat-check` end to end and the check itself is a third of the parse. Any GPU claim made on text LRAT would be a parsing benchmark; the measurement will be on binary.
