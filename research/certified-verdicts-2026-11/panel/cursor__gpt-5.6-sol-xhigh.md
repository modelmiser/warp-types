# cursor:gpt-5.6-sol-xhigh

Q1: CONFIRMED — Session 3 records 632 = 114 + 66 + 452; the CSV has 452 unique mutant names, although from only 113 source proofs, and the actual mutants are not committed (research/gpu-lrat-checker.md:125-130; research/gpu-lrat-checker-2026-11-02/negatives-results.csv:1-453) — none

Q2: CONFIRMED — Same-pod figures give CPU 0.447 + 17.556 + 7.350 = 25.353 s and device pipeline 0.447 + 0.881 + 0.265 = 1.593 s (research/gpu-lrat-checker-2026-11-02/h200-session3-results.txt:45-49,60-68) — none

Q3: CONFIRMED — CPU route is 0.217 + 7.256 + 4.137 = 11.610 s; device path is 0.217 + 0.755 + 0.130 = 1.102 s (research/lrat-from-solver-2026-11-03/chain/h200-run2.txt:65-72) — none

Q4: REFUTED — The CSV has 182 rows but only 114 nonempty checker rows, all verified; these represent 54 distinct UNSAT instances: 30 repeated in all three campaigns plus 24 appearing only in run 3, not 114 distinct corpus instances (research/lrat-from-solver-2026-11-03/results.csv:1-183; chain/h200-run2.csv:1-55) — “114/114 proof-producing campaign rows verified by all three checkers, representing 54 distinct UNSAT instances.”

Q5: CONFIRMED — Run 2/3 each report 20 UNSAT + 4 SAT; run 4b reports 36 + 10, 36 certifications by each checker, 10 replays, and zero campaign-run contradictions (research/bmc-pdr-aiger.md:87-98; run4b-projected/summary.md:1-8; run4b-projected/certifaiger.txt:1-36) — none

Q6: CONFIRMED — Runs 1, 2, and 3 each report the same 21/21 depth agreements; this is not 63 independent cases (research/bmc-pdr-hwmcc17-2026-11-01/summary.md:1-7; run2-summary.md:1-8; run3-coi-summary.md:1-8) — “BMC depths: the same 21 instances agreed 21/21 in each of runs 1–3.”

Q7: REFUTED — All figures except “after halving it” are logged; measured text overhead changed from +19% to +14%, and the ratio from 4× to ≈3×—neither is a halving (research/lrat-from-solver.md:88-95; research/bmc-pdr-aiger.md:97-103) — “Overhead fell from +19% to +14% but remained ≈3× DRAT’s; end-to-end was 2.8× median and 1.2× on n=250.”

Q8: REFUTED — The binary input and ratios hold (21.511/1.102 = 19.5; 20.851/1.102 = 18.9), but “fair” hides that the device route uses 32 host decode threads plus an H200 while cake_lpr is one process (research/lrat-from-solver-2026-11-03/chain/h200-run2.txt:65-72; reproduce/runpod-lrat-chain.sh:35-38,65-69) — “The same-format, same-machine wall-clock comparison is 21.5 s for cake_lpr versus 1.1 s for a 32-thread decode plus GPU pipeline.”

Q9: REFUTED — All identifiers are recorded outside this document, but “Outside oracles only” is false because replay and invariant checking invoke repository-owned Python tools (research/drat-certification.md:17-27,122; research/bmc-pdr-aiger.md:19-29,93; tools/bmc-pdr-campaign.py:60-84) — “External reference tools were combined with separately written in-repository checks.”

Q10: REFUTED — Certification explicitly excludes SAT, DPLL(T), and other entry points; BMC emits uncertified BOUNDED_SAFE/EXHAUSTED and PDR Exhausted is inconclusive (research/drat-certification.md:9-14; warp-types-sat/src/proof.rs:14-17; warp-types-bmc/src/bin/bmc.rs:3-6; warp-types-pdr/src/checker.rs:45-52) — “For the measured pure-SAT UNSAT and PDR SAFE/UNSAFE paths, each conclusive result has per-instance evidence; this does not cover SAT, DPLL(T), BMC BOUNDED_SAFE/EXHAUSTED, or PDR EXHAUSTED.”

Important omissions:
- The document claims completed cross-vendor review, but its panel record is still a placeholder (research/certified-verdicts-2026-11.md:5-6,99-101).
- “Reproduce” omits required external corpora, proofs, and binaries (reproduce/runpod-lrat.sh:4-8; reproduce/runpod-lrat-chain.sh:5-11).
- “Before any code” is stronger than the PDR log’s “before any run” (research/certified-verdicts-2026-11.md:23-25; research/bmc-pdr-aiger.md:1-4).

| Q | verdict | one-line reason | file:line |
|---|---|---|---|
| 1 | CONFIRMED | 632 composition and 452 unique names | gpu-lrat-checker.md:125-130 |
| 2 | CONFIRMED | Both sums are correct | h200-session3-results.txt:45-68 |
| 3 | CONFIRMED | 11.610 s and 1.102 s | h200-run2.txt:65-72 |
| 4 | REFUTED | 114 rows, only 54 instances | results.csv:1-183 |
| 5 | CONFIRMED | Counts, checks, replays agree | bmc-pdr-aiger.md:87-98 |
| 6 | CONFIRMED | Same 21 repeated per run | run3-coi-summary.md:1-8 |
| 7 | REFUTED | Overhead was not halved | lrat-from-solver.md:88-95 |
| 8 | REFUTED | Format-matched, not resource-matched | runpod-lrat-chain.sh:35-69 |
| 9 | REFUTED | In-repo tools are also oracles | bmc-pdr-campaign.py:60-84 |
| 10 | REFUTED | Claim omits uncertified outcomes | drat-certification.md:9-14 |

