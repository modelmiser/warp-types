# Certifying warp-types-sat's UNSAT answers — pre-registration and log

**Registered 2026-11-01, before any run.** Owner's instruction: "run the rewritten prompt" — the plan as revised after
the five-seat panel (`~/staging/2026-11-01-cert-plan-review/panel/`, four families). This file is append-only;
corrections go in dated brackets. Rules: `feedback_preregistration_append_only`, `feedback_check_that_cannot_fail`.

## 0. What is being certified, and what is not

**Certified interface:** `solver::solve_watched_drat` and the CLI `solve --drat FILE`, pure SAT (`NoTheory`),
`Unsat` answers only. A certificate is per instance: it says *this* UNSAT answer is correct, not that the solver is
sound. **Not certified:** `Sat` answers (checked against the original file bytes by an independent script — that is
a model check, not a certificate); `Unknown`; every other entry point (`solve`, `solve_watched`, `hybrid_solve`,
DPLL(T)); theory lemmas, which the logger records but which are not RUP over the CNF.

## 1. Pipeline, pinned

- `drat-trim` — github.com/marijnheule/drat-trim, commit `2e3b2dc0ecf938addbd779d42877b6ed69d9a985`, built
  `gcc -std=c99 -O2` 2026-11-01 (`~/staging/2026-11-01-drat-trim/drat-trim/`). Roles: (a) default backward mode →
  `s VERIFIED` on DRAT; (b) `-L` → LRAT; (c) `-v -W` → exits on the first ignored deletion (unit-deletion
  independence); (d) `-f` forward mode as a second DRAT semantics (still ignores top-level unit deletions,
  `drat-trim.c:820`).
- `lrat-check` — same repo and commit (`lrat-check.c`), built 2026-11-01. Checks the LRAT against the CNF.
- `cake_lpr` — the verified LRAT checker (Tan, Heule, Myreen, TACAS 2021). **Not yet built on this machine**; the
  clone is the owner's to run (classifier). Until it is in, the LRAT leg rests on `lrat-check` alone, and that is
  stated wherever a result is reported.
- `kissat` 4.0.1 (bundled with Kani 0.67.0, `~/.kani/kani-0.67.0/bin/kissat`) — **a debugger, not an oracle**: run
  on every instance to *detect* disagreement; disagreements are adjudicated by certificates and model checks, then
  reduced.
- In-crate `proof::check_drat` — strict active-set semantics (honours every deletion, rejects deletion of an absent
  clause). Author-written; counts as a *different semantics*, not as independent evidence.
- Standard cited: SAT Competition 2023 rules, Main track — every UNSAT answer must come with a DRAT proof (text or
  binary) that `drat-trim` verifies; since 2023 proofs are also trimmed to LRAT and checked with `cake_lpr` <!-- unguarded: 2023 — competition year, not a count -->
  (competition organisers' description; the exact rules page is to be quoted in the log when fetched — until then
  this is "a verified-LRAT pipeline", not "the standard").
- Every artefact row records: CNF SHA-256, proof SHA-256, tool, exit code, verdict line.

## 2. Invariant (the corrected statement, to be written into `proof.rs` and tested)

Every clause `proof.add` logs is RUP w.r.t. the clauses present in the log at that moment, because:
1. The learned clause is the 1-UIP resolvent of the conflict clause and trail reasons, all of which are in the
   database at that moment (originals are never deleted; learned reasons are locked — see 3).
2. Resolution keeps level-0 literals in the clause (`analyze.rs` ~280); minimization then removes a literal only
   if every path through its reason chain ends in a clause literal or a level-0 literal. Removing a level-0 literal
   is RUP only if the checker can re-derive the level-0 unit — i.e. **the level-0 reasons must still be present**.
3. `reduce_learned` runs only after a restart has backtracked to level 0, so the only reasons alive during a reduce
   are level-0 reasons, and `build_locked_set` locks every `Reason::Propagation` on the trail. Learned units have
   LBD 1 and clauses with LBD ≤ 2 are never deletion candidates. Hence no clause a later lemma depends on through
   the trail is ever deleted. (The `abstract_levels >> (level % 64)` filter is an early-abort only; aliasing makes
   it less effective, never more permissive — panel Grok's contrary claim was checked and is wrong.)
4. The initial-BCP conflict and the level-0 conflict emit `0` directly: RUP because unit propagation over the
   logged set reproduces the solver's level-0 trail (originals present; level-0 learned units logged).
5. Compaction remaps CRefs, not literals; deletions are logged from the tombstoned clause *before* compaction.

## 3. Falsifiers — each oracle watched to fail

Test-only fault knobs (thread-local, `cfg(test)`), each producing a corrupted log from an otherwise normal solve:

| fault | what it does | expected: strict `check_drat` | expected: `drat-trim` default | expected: `drat-trim -f` |
|---|---|---|---|---|
| F1 `EmptyLock` | `build_locked_set` returns ∅ | REJECT (or solver panics on dangling reason — record which) | **ACCEPT** (ignores unit/pseudo-unit deletions) — prediction | likely ACCEPT |
| F2 `FlipLiteral` | first literal of the 10th learned clause negated in the log only | REJECT | REJECT | REJECT |
| F3 `OmitFinalEmpty` | `proof.empty()` suppressed | REJECT (no empty clause) | REJECT (no conflict found) | REJECT |
| F4 `PhantomDelete` | at the first reduce, one `d` line for a clause never added | REJECT (absent) | ACCEPT with warning — prediction | ACCEPT with warning |
| F5 `WrongDelete` | at the first reduce, `d` lines for the *kept* candidates instead of the deleted ones | REJECT if a later lemma needs one | likely REJECT | likely REJECT |

File-level negative controls (independent of the crate): a valid proof with (a) one literal of a core lemma flipped,
(b) the final `0` removed, (c) an original clause deleted at line 1 — through the full external pipeline; every
checker must reject (a) and (b); (c) is expected to be *accepted* by `drat-trim` if the clause is a unit and
rejected otherwise — recorded, not scored.

**A fault a checker accepts is a finding about that checker's semantics, recorded as such; it is not a pass.**

## 4. Corpus and coverage

- Knob: `reduce_interval` becomes a parameter (default 2000); the CLI exposes `--reduce-interval N` (documented as
  a diagnostic). Campaign runs use 50 so most UNSAT runs delete.
- Families, fixed seeds, exact commands logged: random 3-SAT n ∈ {150, 200, 250} at 4.26; PHP(6), PHP(7), PHP(8);
  parity chains k ∈ {20, 40, 80}; Tseitin on random 3-regular graphs (odd charge → UNSAT); planted-SAT 3-SAT; and
  `cnfuzz` output if the owner installs cnfuzzdd (otherwise an in-house structured fuzzer, declared as such).
- **Coverage metric:** fraction of UNSAT runs whose proof has ≥ 1 `d` line. **Below 50 % the campaign is invalid.**
- Every SAT model: checked by `tools/cert-campaign.py`'s own DIMACS reader against the original file bytes.
- Every UNSAT proof: drat-trim default, `-v -W`, `-f`, `-L` → `lrat-check` (and `cake_lpr` when present).

## 5. Expected outcomes (written before running)

- E1: 100 % of genuine proofs accepted by every checker in the pipeline. **Kill: any rejection of a genuine proof
  stops the campaign; the instance is reduced (cnfdd if present, else bisection over clauses) and the cause found
  before anything else runs.**
- E2: 0 single-literal `d` lines in any proof; `drat-trim -v -W` clean on all. (Units have LBD 1.)
- E3: d-line coverage ≥ 50 % with `--reduce-interval 50`; with the default 2000, coverage well below 50 % (this is
  the panel's point, measured).
- E4: 0 SAT-model failures; 0 kissat disagreements. **Kill: any model failure stops the campaign.**
- E5: F2, F3 rejected by every checker; F4 accepted-with-warning by drat-trim and rejected by `check_drat`;
  F1 accepted by drat-trim default (prediction), rejected by `check_drat` or solver panic.
- E6: proof logging overhead to `/dev/null` ≤ 25 % of solve time on PHP(7); to a file somewhat more.
- Error-sign ledger kept: every surprise recorded with its direction relative to "the solver is sound".

## 6. Log (append-only)

- 2026-11-01 — **Falsifiers run (in-crate, `cargo test proof::tests::fault_*` with `DRAT_TRIM` set; PHP(6) at interval 50 unless noted). Against the §3 predictions:**
  | fault | strict `check_drat` | `drat-trim` default | `drat-trim -f` | vs prediction |
  |---|---|---|---|---|
  | F2 FlipLiteral | REJECT (step 10 not RUP) | NOT VERIFIED | NOT VERIFIED | as predicted |
  | F3 OmitFinalEmpty | REJECT (no empty clause) | **VERIFIED** | **VERIFIED** | **wrong**: drat-trim derives the conflict itself when the final clause set propagates to ⊥; the trailing `0` is a convention, not evidence. Recorded as checker semantics. |
  | F4 PhantomDelete | REJECT (absent clause) | VERIFIED | VERIFIED | as predicted (ignored; warning only under `-v`) |
  | F5 WrongDelete | REJECT (step 129 not RUP) | NOT VERIFIED | NOT VERIFIED | as predicted |
  | F1 EmptyLock, corpus of 21 (PHP(6), PHP(7), 13 UNSAT r3sat n=60, 6 UNSAT r3sat n=120) | 21/21 completed, 21/21 proofs VALID | — | — | **wrong**: predicted REJECT-or-panic; it was **vacuous** on every instance. The LBD ≤ 2 keep rule retains every level-0 reason that occurred, so no live reason was ever in the deleted half. Consequence recorded: a *forgotten lock would not be caught by small-instance tests*. Soundness held throughout. |
  | F6 LogLockedDelete (new, proof-side isolate) | 13 non-vacuous: **12 REJECT, 1 ACCEPT** | **13/13 VERIFIED** | — | as predicted for drat-trim: it ignores (pseudo-)unit deletions, so deleting a level-0 reason *in the log* is invisible to it; only the strict semantics sees it. The 1 ACCEPT is an instance where no later lemma needed the logged-deleted reason. |
  | F7 DeleteLocked (new, solver-integrity isolate) | — | — | — | 13/13 non-vacuous instances **PANIC** in `Trail::remap_reasons` ("live trail reason CRef … not found"); 8 vacuous. The guard is live. |
  Error-sign ledger: two predictions wrong (F3, F1), both in the direction of *overestimating what the pipeline would catch*. Same sign as before. `drat-trim` accepting F3/F4/F6 means: of the seven corruptions, the external checker sees three (F2, F5, and F7 only via the solver dying). The in-crate strict checker sees six. **A certificate from this pipeline says "the UNSAT answer is correct"; it does not say "the deletions were logged correctly" — that property is covered by the strict checker and nothing else.**
  Also: `reduce_interval` knob measured — PHP(6) at 2000: 0 `d` lines; at 50: 667 `d` lines, proof checks. the whole crate test suite passes; clippy clean. <!-- unguarded: 667 — measured deletion lines in one proof -->

- 2026-11-01 — **Campaign run 1 (`tools/cert-campaign.py`, `--reduce-interval 50`, seeds 8; `research/cert-campaign-2026-11-01/run1-*.csv`).** 50 instances: random 3-SAT n ∈ {150, 200, 250} at 4.26 (24), PHP(6–8), parity k ∈ {20, 40, 80}, Tseitin on random 3-regular graphs v ∈ {20, 30, 40} (12), planted 3-SAT n ∈ {200, 400} (8). No cnfuzz (not installed; owner asked). **30 UNSAT, 20 SAT, 0 timeouts.**
  - **E1 met:** 30/30 genuine proofs `s VERIFIED` by drat-trim default, 30/30 by drat-trim `-f`, 30/30 LRAT (`drat-trim -L`) accepted by `lrat-check`. `cake_lpr` leg: not run (binary not yet built).
  - **E2 met:** 0 single-literal `d` lines in 30 proofs; `drat-trim -v -W` reported 0 ignored deletions on all 30. <!-- unguarded: 30 — campaign run 1 proof count, from results.csv -->
  - **E3 (interval 50 half):** d-line coverage **100 % (30/30)**; proofs carry 821 … 59,774 `d` lines. Default-interval half: run 2, pending. <!-- unguarded: 100 — coverage percentage of run 1, from results.csv -->
  - **E4 met:** 20/20 SAT models satisfy every clause of the original file bytes (runner's own reader); kissat 4.0.1 agrees on 50/50 verdicts.
  - **Negative controls (6 UNSAT proofs × 3 corruptions), recorded, not scored:** *flip one literal of a lemma a third of the way in*: drat-trim default rejected **4/6**, `-f` **6/6**, lrat-check 4/6 (the two it "accepted" are the two where backward trimming dropped the corrupted lemma as non-core, so the LRAT never contained it) — **backward mode checks only core lemmas; a corrupted lemma the refutation does not use passes.** *Drop the final `0`*: accepted by every checker 6/6 — as F3 showed, drat-trim derives the conflict itself. *Delete the first original clause at line 1*: default rejected 5/6, `-f` 6/6; the one default acceptance is consistent with the deleted clause being pseudo-unit under its semantics. <!-- unguarded: 6 — negative-control sample size -->
  - **Kill conditions:** none fired.
  - **Timing (E6): the first measurement was CONFOUNDED** — the no-proof run used the library default interval (2000) while the logged runs used 50, so it compared two different searches (kept as `timing-CONFOUNDED-interval.csv` in staging; r3sat_n250 even ran 5.7× *faster* with logging, i.e. with the smaller interval). Fixed: `solve_watched_with(db, n, interval)` added and the CLI honours `--reduce-interval` without `--drat`; re-measured next entry. Error-sign ledger: this one would have made the overhead look *larger* — the first error today against the thesis.

- 2026-11-01 — **`cake_lpr` built by the owner** (github.com/tanyongkiam/cake_lpr, commit `033aad53d1733bd4b8ca41cf6f1b5fb5d84f8de2`, `cc -O2 basis_ffi.c cake_lpr.S`; the binary is the CakeML-compiled checker shipped as assembly, so what runs is the verified artefact, not a reimplementation). `cnfuzz`/`cnfdd` (fmv.jku.at cnfuzzdd2013) built alongside. **Run 1's 30 genuine LRATs through cake_lpr: 30/30 `s VERIFIED UNSAT`.** The pipeline leg the registration marked pending is now in. Corrupted LRATs: cake_lpr's verdicts mirror drat-trim's exactly (rejects with "empty clause not derived at end of proof" where drat-trim refused and left a partial LRAT; verifies where drat-trim's backward trim had already dropped the corrupted non-core lemma) — the LRAT is drat-trim's output, so cake_lpr certifies *that refutation*, and cannot see a corruption drat-trim trimmed away. This is the §8 "not independent" point, observed.
- 2026-11-01 — **E6 (logging overhead), measured like-for-like** (`solve_watched_with` vs `solve_watched_drat_with`, both at interval 50, release CLI, 5 reps, median wall; `research/cert-campaign-2026-11-01/run1-timing-fair.csv`): PHP(7) 51.2 → 59.7 ms to `/dev/null` (**+17 %**), 60.9 ms to a file; PHP(8) 324 → 374 ms (**+16 %**), 383 ms file; r3sat n=200 357 → 392 ms (**+10 %**); r3sat n=250 671 → 742 ms (**+11 %**). **E6 met (≤ 25 %).** The file costs 1–3 % over the sink; the formatting (`to_string` per literal) is the bulk. Binary DRAT would shrink it; not done, not needed for correctness.

- 2026-11-01 — **Campaign run 2, library default `reduce_interval` 2000 (the configuration users actually get), same 50 instances (`research/cert-campaign-2026-11-01/run2-default2000-*`).** 30 UNSAT / 20 SAT / 0 timeouts; **30/30 `s VERIFIED` (drat-trim default, `-f`), 30/30 lrat-check, 30/30 cake_lpr** (run afterwards over the LRATs; the two largest, r3sat n=250 with 1,298,233 and 698,121 `d` lines in DRAT → 288k / 223k LRAT lines, took cake_lpr 1.95 s / 1.52 s). 0 unit deletions; `-v -W` clean; 20/20 models valid; kissat agrees 50/50; no kill fired.
  - **E3, default half: PREDICTION WRONG.** I registered "coverage well below 50 % at 2000"; measured **63.3 % (19/30)**. The 11 proofs without deletions are exactly the small ones (PHP(6), Tseitin v=20/30 and two v=40, all solved in ≤ 6 ms); every random 3-SAT instance at n ≥ 150 and PHP(7)/PHP(8) reach the threshold. So the panel's warning holds for *small* instances and for the crate's own unit tests, and does not hold for this corpus. Error-sign ledger: this error was in the direction of *underestimating* what default-configuration runs exercise — the second error today against the thesis. <!-- unguarded: 11 — run 2 proofs without deletions, from results.csv -->
  - Negative controls: identical pattern to run 1 (4/6, 0/6, 5/6 for drat-trim default; `-f` 6/6, 6/6 on flip/delorig).

- 2026-11-01 — **Campaign run 3: full pipeline with `cake_lpr` in the loop and 32 `cnfuzz` instances added** (`--reduce-interval 50`, `--negatives 10`; `research/cert-campaign-2026-11-01/run3-*`). 82 instances, **54 UNSAT / 28 SAT / 0 timeouts. 54/54 `s VERIFIED` drat-trim default, 54/54 `-f`, 54/54 lrat-check, 54/54 `cake_lpr` (`s VERIFIED UNSAT`).** 0 unit deletions, `-v -W` clean, 28/28 models valid against file bytes, kissat agrees 82/82, no kill fired.
  - **cnfuzz, as the panel predicted:** 32 instances (21–580 variables), 24 UNSAT, only **2 of 24 with any deletion** even at interval 50 — cnfuzz instances are tiny and finish in a few dozen conflicts. They add format and parser diversity (comment lines, odd clause widths, scrambled variables), not deletion coverage. Overall coverage 59.3 % (32/54) — above the 50 % validity line because the structured families carry it.
  - **Negative controls through cake_lpr (10 proofs × 3):** flip — drat-trim default rejected 6/10, `-f` 10/10, cake_lpr rejected 6/10 and verified 4/10 (the four where backward trimming had removed the corrupted non-core lemma, so the LRAT it was handed is a valid refutation); drop final `0` — accepted by everything 10/10; delete first original clause — default 9/10, `-f` 10/10, cake_lpr 9/10. cake_lpr's verdict is a verdict on drat-trim's LRAT, every time. <!-- unguarded: 10 — negative-control sample size, run 3 -->
- 2026-11-01 — **Tally across runs 1–3, the certification statement, and the error-sign ledger.**
  - **114 genuine UNSAT proofs** (30 + 30 + 54; 60 at interval 50, 30 at the default 2000, 24 of them cnfuzz) from six families, **114/114 verified by drat-trim (backward and forward), 114/114 LRAT accepted by lrat-check, 114/114 by cake_lpr.** 68/68 SAT models satisfy the original file bytes. kissat 4.0.1 agreed with every one of 182 verdicts. 0 single-literal deletions in ~3.9 M `d` lines. Fault injection: 7 corruptions, strict checker catches 6 (EmptyLock vacuous, soundness held), drat-trim catches 2 outright and a third via the solver's own panic. <!-- unguarded: 114 — campaign tally across runs 1–3, from the three results.csv files -->
  - **What a certificate from this pipeline says:** *this* `Unsat` answer of `solve_watched_drat` / `solve --drat` on *this* CNF is correct, as checked by a formally verified checker on the refutation `drat-trim` extracted from our log. **What it does not say:** that the solver is sound in general (per-instance only); that the deletions were logged correctly (drat-trim ignores unit deletions and trims non-core lemmas — only the strict in-crate semantics sees those, and it is author-written); anything about `Sat`, `Unknown`, the gradient/hybrid entry points, or DPLL(T) lemmas.
  - **Predictions vs outcomes (6 expected outcomes, 7 faults):** right — E1, E2, E4, E6, E3's interval-50 half, F2, F4, F5; **wrong — F3** (drat-trim accepts a proof without the final `0`), **F1** (EmptyLock vacuous on every instance; a forgotten lock would not be caught by small tests), **E3's default half** (63 % coverage, not "well below 50 %"). Direction: F3 and F1 overestimated what the pipeline catches (toward the thesis "the pipeline is strong"); E3-default and the confounded timing underestimated what default runs exercise (against it). Two each way. Not the one-sided ledger of the 65× arc.
  - **Not done, stated:** binary DRAT; LRAT emitted directly by the solver (would remove drat-trim from the trusted path); DRAT for the hybrid path; a competition-rules page quoted verbatim (the standard is cited by name, not by text). `cnfdd` was built but never needed — nothing to reduce.
