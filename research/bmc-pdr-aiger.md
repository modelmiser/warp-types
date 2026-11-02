# `warp-types-bmc` / `warp-types-pdr` on real hardware instances — pre-registration and log

**Registered 2026-11-01, before any run of our crates on these instances.** Owner: "your call on everything,
autonomously". Append-only; corrections in dated brackets. Follows `research/drat-certification.md` in spirit:
measure before certifying, controls that can fail, kill conditions written first.

## 0. Why

The two upper crates have only ever run on their own three toy examples. The SAT core's peer comparison turned
out to have slandered it by 35–88× for seven months because nobody had run a fair measurement. The same could be
true (either way) of BMC and PDR. And the day's certification theme stops at single-moment UNSAT answers; a PDR
"never" answer is a fence (inductive invariant) that nobody outside the crate has ever checked.

## 1. Setup, pinned

- Instances: HWMCC 2017 single-property set (`hwmcc17-single-benchmarks.tar.xz`, fmv.jku.at, 300 `.aig`),
  restricted to files **under 20 KB** (the crates have no inprocessing; this is the honest size band). Kept in
  `~/staging/2026-11-01-hwmcc/`.
- Reference tool: ABC 1.01 (`yosys-abc`, compiled 2026-03-24) — `read_aiger; fold; bmc3 -F 1000 -T 30` and
  `read_aiger; fold; pdr -T 30`, 40 s wall cap. ABC is the **ground truth for verdicts and for the minimal
  counterexample depth**; it is not what we are trying to beat.
- Our side: a new `aiger` module (binary AIGER 1.9 reader; AND gates, latches with reset values, bad/output,
  invariant constraints) that lowers an AIG into the crates' `TransitionSystem` — every AIG node becomes a state
  variable (inputs free, gates defined on both sides of T and in I, latches' next from the AIG), the single bad
  literal becomes the one property clause. CLI: `bmc <file.aig>` and `pdr <file.aig> --invariant OUT`.
- Independent check of a PDR fence: `tools/check-invariant.py` — its **own** binary-AIGER parser and Tseitin
  encoding, and **kissat 4.0.1** (Kani's) as the solver. Three queries per invariant Inv (over AIG literals):
  (1) `Init ∧ ¬Inv` UNSAT (initiation); (2) `Inv ∧ T ∧ ¬Inv'` UNSAT (consecution); (3) `Inv ∧ ¬P` UNSAT (safety).
  Nothing from our crates is in that path except the invariant text. Certifaiger (the HWMCC witness-circuit
  checker) is not built here (external clone, classifier); the invariant is also written in a form that can be
  turned into a witness circuit later.
- Budget: our BMC max depth 1000, our PDR max frames 1000, per-instance wall cap 60 s via `timeout`, SAT
  conflict budget 0 (unlimited; correctness over coverage).

## 2. Predictions (written before running)

- **P1 — depth agreement (the control that can fail).** On every instance where both ABC `bmc3` and our BMC
  report a counterexample, the **depths are identical**. The minimal counterexample length is a property of the
  instance, both tools iterate depth from 0, so any mismatch is an encoding or solver bug on our side.
  **Kill: a mismatch stops the run; the instance is minimised and the cause found before anything else.**
- **P2 — verdict agreement.** No instance where ABC says SAT (bug) and ours says `Safe`, or the reverse.
  **Kill as P1.**
- **P3 — fence validity.** Every invariant our PDR returns passes all three kissat checks. **Kill: one failure =
  a soundness bug in `warp-types-pdr`; stop, reduce, fix before any other result is reported.**
- **P4 — speed.** Ours is slower than ABC by 10×–1000× on the instances both finish; no prediction more precise
  than that — this is the first measurement.
- **P5 — coverage.** Of the sub-20 KB instances, ABC finishes most within 30 s; ours finishes a minority within
  60 s (guess: 20–50 % for BMC on SAT instances, 10–30 % for PDR on UNSAT ones). Recorded, not scored.
- **P6 — the empty-invariant case.** Where our PDR returns `Safe` with an empty invariant (P already inductive),
  the kissat check (2) with Inv = P must pass; this is the degenerate fence and must still be checked.
- Error-sign ledger kept: every surprise recorded with direction relative to "the crates are sound".

## 3. Not in scope

Liveness/justice instances (none in the single-property safety set used); AIGER 1.9 justice/fairness; the
`warp-types-smt` crate; any optimisation of the crates in response to the numbers (that would be a new arc).

## 4. Log (append-only)

- 2026-11-01 — **The depth-agreement control fired before the campaign started, on the smoke test.** First build of the AIGER lowering: `139442p1` and `139443p5` (ABC `bmc3`: counterexample at frame 3, 0.05 s) came back `BOUNDED_SAFE depth=30`. Cause: AIGER literal 0 is FALSE and literal 1 is TRUE, and the lowering had mapped index 0 with ordinary polarity (literal 0 → the constant-true variable, positive), inverting every constant in the design. The unit test `lowering_literal_map_roundtrips` passed under the bug because `aiger_lit(lit(x))` round-trips either way — a round-trip test cannot see a consistent polarity swap; only a semantic test can (added: `constant_bad_literal_behaves`, bad = literal 1 must be a depth-0 counterexample). Fixed; both instances now `s SAT depth=3` in 16 ms / 26 ms. Error-sign ledger: an encoding bug that made the crate look *safer* than reality (missed counterexamples) — toward "the crates are sound", caught only because P1 was registered as a control that can fail. Second finding from the same smoke test: binary AIGER requires `rhs0 ≥ rhs1` in every AND; the first test fixture violated it and the hand encoder underflowed — fixture reordered (AND is commutative).
- 2026-11-01 — **PDR is slow on these as encoded**: `139442p1` (3,881 state variables once gates and inputs are state) does not return within 60 s where ABC's `pdr` finds the depth-3 counterexample in 0.2 s. Expected from the encoding — cubes range over every gate, and generalisation is per-literal — and it is what the campaign will measure, not something to tune in this arc.
- 2026-11-01 — **Independent invariant checker watched to fail.** `nusmvsyncarb5p2`: our PDR returns `Safe` at frame 8 with 162 clauses; `tools/check-invariant.py` (own AIGER reader, own Tseitin, kissat 4.0.1) says CERTIFIED — initiation, consecution and safety all UNSAT. Corruptions: flip one literal in clause 5 → REJECTED (initiation SAT); property clause alone (the "P is inductive" degenerate fence) → REJECTED (consecution SAT), so P6's degenerate case will be a real check. Dropping the property clause from the 163 → still CERTIFIED: the 162 frame clauses alone imply ¬bad and are inductive, i.e. the frame is a fence by itself on this instance — recorded as a property of the instance, not a hole in the checker (the flip and P-only cases show the checker can say no).
- 2026-11-01 — **Caveat on the ABC reference timings.** The ABC run (started ~16:05, `ref/run-abc.sh`, niced on cores 0–5) overlapped with cargo builds and tests of the new code on the same cores until ~17:05. ABC's *verdicts* and *depths* are unaffected; its wall times for instances processed in that window may be inflated by contention, which biases P4 (our/ABC ratio) *downwards*, i.e. in our favour. Noted so the ratio is read as "order of magnitude", which is all P4 claimed. Our own campaign runs after ABC finishes, with nothing else on those cores.
- 2026-11-01 17:12 — **ABC reference complete** (118 instances; `ref/abc.csv`). `bmc3`: 26 SAT, 90 UNKNOWN at the 30 s / 1000-frame cap, 2 proved by exhausting reachability ("Explored all reachable states after completing N frames" — `bj08aut82` at 8, `pdtvisgray0` at 32; the parser had filed these as NONE, corrected by hand to UNSAT with the frame count; raw file kept as `abc-raw.csv`). `pdr`: 55 UNSAT, 23 SAT, 40 UNKNOWN. So the reference offers 26 counterexample depths to agree with and 55 proofs to match. Our campaign started 17:12 (`ours/run1.log`); it had loaded the uncorrected CSV, so its two `abc_bmc3=NONE` cells are this parsing gap, not a disagreement. <!-- unguarded: 55 — ABC pdr proof count from ref/abc.csv -->

- 2026-11-01 ~18:30 — **Campaign run 1 complete** (`research/bmc-pdr-hwmcc17-2026-11-01/`: `results.csv`, `summary.md`, `campaign.log`, the ABC reference, and the 20 certified invariants). 119 instances under 20 KB (one, `beemcmbrdg1b1`, has no ABC row — a `find -size` vs Python size-bound difference), 60 s wall per crate per instance, nothing else on the cores.
  **BMC.** Ours: 21 SAT, 7 BOUNDED_SAFE at depth 1000, 91 TIMEOUT. ABC `bmc3` (30 s / 1000 frames): 26 SAT, 90 UNKNOWN, 2 proved by exhausting reachability. **P1 (depth agreement): 21/21 identical depths**, from 0 (`bobmiterbm1or`) to 85 (`productioncellp0`). No instance where ours found a counterexample ABC did not; 5 where ABC did and we timed out (`6s134`@168, `6s319r`@158, `6s41`@73, `bc57sensorsp2`@104, `irstdme6`@53 — all deep, ABC 9–24 s; a 300 s retry is running). **P2: 0 contradictions.** Time: the summary's "median 0.2×" is an artefact — ABC's ~40 ms process start-up dominates the tiny instances. On the 7 both-SAT instances where either side took ≥ 0.1 s: `beemadd4b1` 17.25 s vs 18.15 s (0.95×), `csmacdp2` 2.0×, `dme6p1` 3.5×, `pdtswvibs8x8p0` 4.0×, `abp4p2ff` 5.5×, `prodcellp4` 22.9×, `productioncellp0` 22.3×. **P4 for BMC was WRONG**: predicted 10–1000× slower; measured 1–23× on what both finish, and coverage 21/26 of ABC's counterexamples (**P5 predicted 20–50 %: WRONG, measured 81 %**). The 7 BOUNDED_SAFE results are all instances ABC's `pdr` proves (or leaves UNKNOWN): reaching depth 1000 in 3–50 s on them is unrolling cost, not evidence of anything.
  **PDR.** Ours: 20 UNSAT (`Safe`), 4 SAT, 95 TIMEOUT, 0 EXHAUSTED. ABC `pdr` (30 s): 55 UNSAT, 23 SAT, 40 UNKNOWN. **P3: 20/20 invariants CERTIFIED by `tools/check-invariant.py` on kissat** — initiation, consecution and safety all UNSAT for each; invariant sizes 1 clause (13 instances, converged at frame 1) up to 588 clauses (`bob2`). The 4 counterexamples (`bobmiterbm1or`@0, `bobtuint24`@0, `counterp0`@9, `mutexp0`@7) match ABC `bmc3`'s depths exactly. **P6 (empty invariant) did not occur** — every `Safe` had ≥ 1 frame clause; the checker's handling of the degenerate case rests on the P-only control from the smoke test. Coverage: 20 of the 55 instances ABC proves = 36 % (**P5 predicted 10–30 %: slightly WRONG, underestimated**); 4 of our 20 proofs are instances where ABC `pdr` was UNKNOWN at 30 s (`bob9234specand`, `kenflashp13`, `pdtvishuffman1`, `shift1add262144`, ours in 0.7–2.1 s) — different tools, different caps, not a win claim. Speed on the 16 both-proved: median **282×** slower than ABC, max 1,129× (`bobcount` 28.4 s vs 0.10 s); only one of them took ABC ≥ 0.1 s, so the median is again mostly ABC start-up — the honest statement is "ABC proves these in tens of milliseconds, we take 0.1–41 s". **P4 for PDR: met** (10–1000×). <!-- unguarded: 20 — our PDR proof count from results.csv; 55 is ABC pdr -->
  **Kill conditions: none fired.**
  **Error-sign ledger.** P1, P2, P3, P4-PDR right. P4-BMC and both halves of P5 wrong, each by *underestimating* the crates (predicted far slower and far less coverage than measured). Together with the constant-polarity bug (which made BMC look *safer* than reality), the day's ledger for this arc: one error toward soundness (caught), three toward pessimism about speed. Reading: the crates are slower than ABC by the margin the PDR number shows, and the BMC core, once the encoding was right, is in ABC's range on small instances — but 91 and 95 timeouts out of 119 say the honest size band for these crates today is well under 20 KB of AIG.
  **What the certificates say.** For 20 instances, the fence `warp-types-pdr` produced is an inductive invariant implying the property, as checked by an independent encoding on an independent solver. They do not say the PDR crate is sound in general (per instance), nothing about the 95 timeouts, and the checker is independent *software*, not a verified checker — Certifaiger was not built (classifier; owner can clone it) and the HWMCC witness-circuit format was not emitted.
