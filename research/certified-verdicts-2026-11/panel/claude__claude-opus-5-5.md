# claude:claude-opus-5-5

I traced every claim to the repository files. Nine of the ten claims hold in their numbers but several are worded more strongly than the data. One (Q7) is refuted by the data, and the §1 headline (Q10) is false as written.

Q1: CONFIRMED — 114 genuine proofs (cert-campaign-2026-11-01 run1/run2/run3 results: 30+30+54 VERIFIED), 66 drat-trim outputs from corrupted DRATs (33 NONE + 33 VERIFIED in the run*-negatives files), and 452 mutants (negatives-results.csv: 453 lines, 452 distinct names). The total of 632 matches h200-session3-results.txt:34. But 452 = **113**×4: `cnfuzz_s2`, a 2-line proof, has no mutants. gpu-lrat-checker.md:125 says "for each of the 114" (that would be 456). Also, the 632/632 in the single session-3 run is device vs CPU reference only. Agreement with `lrat-check`/`cake_lpr` comes from two pieces: the 392-file run (gpu-lrat-checker.md:128, no data file) plus the 452-mutant CSV. `cake_lpr` columns are empty for 60 genuine proofs and 36 drat-trim outputs in the cert-campaign CSVs. — Replacement: "632 files (114 genuine, 66 drat-trim outputs, 452 hint-level mutants of 113 of the genuine proofs); external-checker agreement from the 392-file archive run plus the mutant run".

Q2: CONFIRMED (arithmetic) — h200-session3-results.txt:47 (device 0.265 s), :66 (decode 0.885 s on 32 threads), :49 (read 0.447 + parse 17.556 + check 7.350 = 25.35 s). 0.447 + 0.885 + 0.265 = 1.60 s. However, the 1.6 s is assembled from two different invocations. In the same run as the 0.265 s device time, decode took 2.891 s on 96 threads (:45), so that run's actual pipeline was ≈ 3.6 s. — Replacement: "≈ 1.6 s (read + 32-thread decode from the scaling run + device; 3.6 s with the 96-thread decode in the timing run)".

Q3: CONFIRMED — h200-run2.txt:70: 0.217 + 7.256 + 4.137 = 11.61 s. Lines :68–70: 0.217 + 0.755 + 0.130 = 1.10 s. 20.851/1.10 = 18.9. Our CPU route used only **4 threads** on a 128-core host (:70 `independent(4)`, :4). — Replacement: "our CPU route 11.6 s (4-thread check)".

Q4: REFUTED (wording) — results.csv has 182 rows. 114 of them carry `lrat_check` = VERIFIED, `cake_lpr` = VERIFIED and `ours` = "s VERIFIED". Those 114 rows are only **54 distinct instances**: 30 appear in all three runs with byte-identical `lrat_bytes`, and 24 appear once. — Replacement: "114/114 proofs (54 distinct instances; 30 repeated across three corpus runs with byte-identical proofs)". The trim row needs the same fix.

Q5: CONFIRMED — run4b results.csv: 36 UNSAT, 10 SAT, 73 TIMEOUT, 36 CERTIFIED. run4b certifaiger.txt: 36 lines, all "valid witness". The 10 PDR depths equal `abc_bmc3_depth`. run2-summary.md:5 and run3-coi-summary.md:5 both show 20 + 4. Line 8 of every summary (runs 1–4c) says "verdict contradictions: 0", and my own cross-check of the CSVs found none. — none.

Q6: CONFIRMED — summary.md:2, run2-summary.md:2 and run3-coi-summary.md:2 each show 21/21, and the CSVs' `depth_match` column is True×21 in each. It is the same 21 instances re-run, not 63 independent agreements. — Replacement: "21/21 in each of runs 1–3 (same 21 instances)".

Q7: REFUTED — The run-4 → 4b comparison gained **five** instances, not four ("restored it and three more" is wrong; bmc-pdr-aiger.md:99; CSV diff: 5 gained, 2 lost; 43 + 5 − 2 = 46). `139452p0` was a **lost verdict** (1.69 s → TIMEOUT); 129 s is only with a 300 s cap (:99). "After halving it" has no support: overhead went LRAT text +19 % → +14 % and DRAT +7 % → +4 %, so the text ratio worsened from ≈ 2.7× to 3.1× (lrat-from-solver.md:95). The pre-registered corpus median was 4.0× (:89). The other figures check out: 94 s (300 s cap, bmc-pdr-aiger.md:97), 2.8×/1.2× (lrat-from-solver.md:91), +4 vs +8 (bmc-pdr-aiger.md:103). — Replacement: "overhead ≈ 3× DRAT's on one n = 250 instance after optimisation (corpus median 4.0× before)"; "restored it and four more and cost two, one badly (1.7 s → timeout at 60 s; 129 s with a 300 s cap)".

Q8: CONFIRMED — `cake_lpr` read the binary file (h200-run2.txt:67; runpod-lrat-chain.sh:68). 21.51/1.10 = 19.5, so "≈ 20×" is the accurate figure. — Replacement: "…at 21.5 s, ≈ 20×".

Q9: CONFIRMED — The tool versions are recorded outside this document: drat-certification.md:17 (drat-trim 2e3b2dc0), :122 (cake_lpr 033aad53), :26 (kissat 4.0.1); bmc-pdr-aiger.md:19 (ABC 1.01), :93 (Certifaiger 10.4.0). One gap: `lrat-check.c` on the chain pod was compiled from /tmp/chain without a recorded commit (runpod-lrat-chain.sh:8,33). — Replacement: add "(versions: drat-certification.md §tools; bmc-pdr-aiger.md §reference)".

Q10: REFUTED — The §1 sentence is false for several verdict types:
- **SAT models** are checked by our own runner (drat-certification.md:117).
- **UNSAFE counterexamples** are replayed by our own `tools/replay-trace.py` (bmc-pdr-aiger.md:86), not by "a tool we did not write".
- **BMC BOUNDED_SAFE** and **PDR `Exhausted`** (warp-types-pdr/src/checker.rs:50) have no certificate at all.
- **The `warp-types-smt` crate's (EUF/BV) verdicts** have no certificate at all.

§2 also calls `check-invariant.py` an outside oracle, but it is ours and mirrors the crate's AIGER conventions (bmc-pdr-aiger.md:80). — Replacement: "Every UNSAT answer from the SAT solver and every SAFE answer from the PDR checker comes with a certificate accepted by a tool we did not write (`lrat-check`, `cake_lpr`, Certifaiger); UNSAFE answers come with a counterexample that replays under our own independent simulator at ABC's depth. SAT models, BMC bounded results, PDR `Exhausted`, and the SMT solver's verdicts are not certified."

**Not asked:**
1. The trim row "within 2 % of `drat-trim -L`" holds for one instance (n250_s4: 41.5 vs 42.3 MB, trim-results.csv). The corpus max ratio is 1.03 (lrat-from-solver.md:94). Replacement: "on the largest instance; corpus median 0.92, max 1.03".
2. The 19× compares against an external proof (kissat + drat-trim). §1's "the same proof [the solver's own] is checked on a GPU" blurs this: for solver-emitted proofs, h200-run2.txt:50 shows only decode + device (0.049 s) vs `lrat-check` 0.85 s, with no read time.
3. gpu-lrat-checker.md:125's tallies cite `~/staging/...`, which is not in the repo; the in-repo CSV does match them (184/42/113/113).

| Q | verdict | reason | file:line |
|---|---|---|---|
| 1 | CONFIRMED | 114+66+452=632; mutants are of 113 proofs; external agreement split across runs | negatives-results.csv; gpu-lrat-checker.md:125,128 |
| 2 | CONFIRMED | sums hold; 1.6 s mixes two invocations | h200-session3-results.txt:45,47,49,66 |
| 3 | CONFIRMED | 11.61 / 1.10 s; CPU route used 4 threads | h200-run2.txt:68–70 |
| 4 | REFUTED | 114 rows = 54 distinct instances | lrat-from-solver-2026-11-03/results.csv |
| 5 | CONFIRMED | 36/10, 36 certified, 10 replayed, 0 contradictions | run4b-projected/summary.md:5,8 |
| 6 | CONFIRMED | 21/21 per run, same instances | run2-summary.md:2 |
| 7 | REFUTED | five gained not four; 1.7 s → timeout; "halving" unsupported | bmc-pdr-aiger.md:99; lrat-from-solver.md:95 |
| 8 | CONFIRMED | binary input; ratio ≈ 19.5 | h200-run2.txt:67 |
| 9 | CONFIRMED | versions recorded in the logs | drat-certification.md:17,122; bmc-pdr-aiger.md:19,93 |
| 10 | REFUTED | SAT models, UNSAFE, BOUNDED_SAFE, Exhausted, SMT not externally certified | bmc-pdr-aiger.md:86; checker.rs:50 |
