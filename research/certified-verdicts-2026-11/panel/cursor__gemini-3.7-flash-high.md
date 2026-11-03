# cursor:gemini-3.7-flash-high

Q1: CONFIRMED — `research/gpu-lrat-checker.md:130` and `research/gpu-lrat-checker-2026-11-02/h200-session3-results.txt:33` record 632/632 agreed (114 genuine, 66 from corrupted DRATs, 452 hint mutants); `research/gpu-lrat-checker-2026-11-02/negatives-results.csv:2-453` contains exactly 452 distinct mutant rows (113 base instances × 4 mutations) — none

Q2: CONFIRMED — `research/gpu-lrat-checker-2026-11-02/h200-session3-results.txt:47,49,65-66` shows device time 0.265 s (rounds to 0.27 s), parallel decode 0.881 s on 32 threads (0.88 s), read 0.447 s giving pipeline sum 0.447 + 0.881 + 0.265 = 1.593 s (≈ 1.6 s), vs CPU route sum 0.447 + 17.556 + 7.350 = 25.353 s (25 s / 25.4 s) — none

Q3: CONFIRMED — `research/lrat-from-solver-2026-11-03/chain/h200-run2.txt:66-70` reports `lrat-check` 20.851 s (20.9 s), `cake_lpr` 21.511 s (21.5 s), CPU route sum read 0.217 + parse 7.256 + independent(4) 4.137 = 11.610 s (11.6 s), and device path sum read 0.217 + decode(32) 0.755 + device 0.130 = 1.102 s (1.1 s), yielding 20.851 / 1.102 = 18.9× and 21.511 / 1.102 = 19.5× (≈ 19×) — none

Q4: CONFIRMED — `research/lrat-from-solver-2026-11-03/results.csv:5-181` contains 114 rows with an `lrat_check` column (27 run 1, 27 run 2, 60 run 3 across 87 distinct instances), and all 114 report VERIFIED for `lrat_check`, `cake_lpr`, `drat_trim`, and `ours`; the document accurately distinguishes "114/114 corpus proofs" from "54/54 instances" in the next row — none

Q5: CONFIRMED — `research/bmc-pdr-hwmcc17-2026-11-01/run4b-projected/summary.md:5-6` and `certifaiger.txt:1-36` show 36 UNSAT (proved) + 10 SAT (refuted) at 60 s, all 36 certified by `check-invariant.py` and Certifaiger, with 10/10 counterexamples replaying; `run2-summary.md:5` and `run3-coi-summary.md:5` confirm the baseline 20 + 4; all run summaries (runs 1–4c) report 0 verdict contradictions — none

Q6: CONFIRMED — `research/bmc-pdr-hwmcc17-2026-11-01/summary.md:2`, `run2-summary.md:2`, and `run3-coi-summary.md:2` show that in each of runs 1, 2, and 3, BMC solved the same 21 instances to SAT and 21/21 agreed in counterexample depth with ABC `bmc3` — none

Q7: CONFIRMED — `research/lrat-from-solver.md:91,95` and `research/bmc-pdr-aiger.md:97,99,104` explicitly record ≈ 3× overhead after halving (text 14% vs DRAT 4%), 2.8× median and 1.2× largest end-to-end speedups, `nusmvreactorp5` taking 94 s to rediscover inductiveness under latch projection, `139452p0` regressing from 1.7 s to 129 s, and descending order union gaining +4 vs threshold +8; §4 reports all signs and numbers faithfully — none

Q8: CONFIRMED — `research/lrat-from-solver-2026-11-03/chain/h200-run2.txt:66-70` shows `cake_lpr` ran on the 415 MB binary proof (415,372,951 bytes) taking 21.511 s vs device path 1.102 s, giving 21.511 / 1.102 = 19.5×, fully consistent with `lrat-check`'s 20.851 / 1.102 = 18.9× (both ≈ 19×) — none

Q9: CONFIRMED — `research/drat-certification.md:17,122`, `research/bmc-pdr-aiger.md:19,27,93`, and `warp-types-sat/src/lrat.rs:12` explicitly record `drat-trim` commit `2e3b2dc0`, `cake_lpr` commit `033aad53`, Certifaiger 10.4.0, kissat 4.0.1, and ABC 1.01 — none

Q10: REFUTED — `research/certified-verdicts-2026-11.md:10` overstates coverage; `research/lrat-from-solver.md:40` states DPLL(T) theory lemmas have no RUP chain (unjustified), SAT satisfiable models emit no external proof certificate, and BMC `BOUNDED_SAFE` is an uncertified bounded depth result — "The decision verdicts measured here — pure SAT UNSAT refutations and PDR SAFE/UNSAFE results — come with a certificate that an outside oracle accepts; SAT models, DPLL(T) theory lemmas, and BMC BOUNDED_SAFE verdicts are outside this chain."

### Unasked Issues (most severe first)

1. **Unchecked model binding in witness generator:** As recorded in `research/bmc-pdr-aiger.md:93`, `aig-witness` translates `.inv` into a witness circuit based strictly on file naming without embedding the model hash recommended by Certifaiger, trusting that `.inv` matches the model.
2. **Asymmetric time caps in BMC comparison:** In `research/bmc-pdr-aiger.md:80`, ABC used a 30 s cap whereas the in-crate BMC/PDR used a 60 s cap, meaning comparative coverage numbers are not like-for-like.
3. **External generator dependence for multi-GB scale proofs:** As noted in `research/lrat-from-solver.md:91` and `research/gpu-lrat-checker.md:66`, the GB-scale proof files evaluated on the GPU (`r3sat_n350_s2`, `r3sat_n350_s3`) were produced by kissat + `drat-trim`, because the repository's solver does not finish instances at that scale.

### Summary Table

| Q | verdict | one-line reason | file:line |
|---|---|---|---|
| Q1 | CONFIRMED | Composition 114 + 66 + 452 = 632 matches log, results, and 452 distinct negative CSV rows | `research/gpu-lrat-checker.md:130`, `research/gpu-lrat-checker-2026-11-02/negatives-results.csv:2` |
| Q2 | CONFIRMED | Timing arithmetic matches pod results (0.265 s device, 0.881 s decode, 1.593 s pipeline vs 25.353 s CPU) | `research/gpu-lrat-checker-2026-11-02/h200-session3-results.txt:47,49,65` |
| Q3 | CONFIRMED | 11.6 s CPU route sum, 1.1 s device sum, and ≈ 19× match same-machine RunPod run 2 log | `research/lrat-from-solver-2026-11-03/chain/h200-run2.txt:66-70` |
| Q4 | CONFIRMED | 114 rows have lrat_check column, all 114 are VERIFIED across all checkers, accurately distinguished from 54 instances | `research/lrat-from-solver-2026-11-03/results.csv:5` |
| Q5 | CONFIRMED | 36+10 at 60 s matches run 4b, 20+4 matches runs 2/3, 36/36 certified by both checkers, 0 contradictions | `research/bmc-pdr-hwmcc17-2026-11-01/run4b-projected/summary.md:5-6` |
| Q6 | CONFIRMED | BMC solved 21 instances to SAT in each of runs 1–3, with 21/21 matching ABC bmc3 depth | `research/bmc-pdr-hwmcc17-2026-11-01/summary.md:2`, `run2-summary.md:2` |
| Q7 | CONFIRMED | All 5 miss figures and signs match log lines without design-flattering drift | `research/lrat-from-solver.md:91,95`, `research/bmc-pdr-aiger.md:97,99,104` |
| Q8 | CONFIRMED | cake_lpr checked binary proof in 21.5 s, giving 19.5× (consistent with lrat-check 18.9×) | `research/lrat-from-solver-2026-11-03/chain/h200-run2.txt:67` |
| Q9 | CONFIRMED | All 5 tools and commit hashes are explicitly recorded in repository research logs and source | `research/drat-certification.md:17,122`, `research/bmc-pdr-aiger.md:19,27,93` |
| Q10 | REFUTED | Sentence overstates scope; SAT models, DPLL(T) lemmas, and BMC BOUNDED_SAFE lack certificates | `research/certified-verdicts-2026-11.md:10`, `research/lrat-from-solver.md:40` |

