# cursor:grok-4.7-medium

Q1: CONFIRMED — `h200-session3-results.txt:34` is only “632 proofs, 632 agree, 147 verified”; the split is 30+30+54 UNSAT (`run1-summary.md:2`, `run2-default2000-summary.md:2`, `run3-cake-cnfuzz-summary.md:2`), 18+18+30 corrupted-DRAT LRATs (`run1-negatives.csv` 18 data rows, run2 18, run3 30), and 452 mutant rows (`negatives-results.csv:2–453`, 113×4 kinds, all `NOT,NOT,Rejected`). 114+66+452=632 and 114+33 accepted negatives=147. The 452 proof files are not in the repo, only the CSV. — none

Q2: CONFIRMED — 941 MB block: device 0.265 s (`h200-session3-results.txt:47`), read 0.447 + parse 17.556 + 96-thread check 7.350 = 25.35 s (`:49`); 32-thread decode 0.885 s (`:66`). 0.447+0.885+0.265=1.60 s. “25 s” rounds 25.4 s; the 1.6 s sum mixes the §B timing block with the §C 32-thread decode. — “read 0.45 s + 32-thread decode 0.88 s + device 0.27 s ≈ 1.6 s, against read+parse+96-thread check = 25.4 s on the same pod.”

Q3: CONFIRMED — `h200-run2.txt:66–70`: lrat-check 20.851 s, cake_lpr 21.511 s; CPU route 0.217+7.256+4.137=11.61 s; device path 0.217+0.755+0.130=1.102 s; 20.851/1.102≈18.9×. The 3.217 s `gpu(...)` line is a different measurement. — none

Q4: CONFIRMED — `results.csv` has 114 rows with `lrat_check`, `cake_lpr`, and `ours` all VERIFIED (e.g. `:5`), plus 68 SAT rows with those fields empty (`:2–4`). They are run1 30 + run2 30 + run3 54, not 114 CNFs and not 54×3: `php6` is three identical proofs (`:5,:55,:137`). Chain file is 54 distinct names, all three checkers VERIFIED (`h200-run2.csv`, 55 lines). “Proofs” matches rows; it does not say 114 instances. — “114/114 proof rows (30+30+54 across three runs; repeated instances) VERIFIED by lrat-check, cake_lpr, and ours.”

Q5: CONFIRMED — run 4b: UNSAT 36, SAT 10, 36 `CERTIFIED`, 10 `REPLAYED` (`run4b-projected/summary.md:3–5`, `results.csv` 36 CERTIFIED / 10 REPLAYED); `certifaiger.txt:1–36` all `valid witness`. Run 2 and run 3 are each UNSAT 20 and SAT 4 (`run2-summary.md:5`, `run3-coi-summary.md:5`). All six summaries, runs 1–4c, say `verdict contradictions: 0`, and no results CSV in that directory contains `False`. — none

Q6: CONFIRMED — each of run 1, run 2, and run 3 says “depth agreement on the 21 both-SAT: 21/21” (`summary.md:2`, `run2-summary.md:2`, `run3-coi-summary.md:2`), and each results CSV has 21 rows ending in depth-match and verdict both true. It is per run; runs 4–4c skipped BMC (`0/0`). — “21/21 depth agreement in each of runs 1, 2, and 3.”

Q7: REFUTED — the figures sit on log lines (`lrat-from-solver.md:91` median 2.8× and 1.2× at n=250; `:95` 16.76/17.50/19.06/18.46 s; `bmc-pdr-aiger.md:97` 94 s; `:99` 1.7 s→129 s; `:103` union 50 vs 46, threshold 8), but §4 is kinder than those lines. “After halving” does not match a 4.0× median (`:89`) falling to ~3.1× on one seed, n=250 s4. After the trim §3 advertises, the same log says the n=250 end-to-end gap is ≈1.07× (`:94`, 13.9 s vs 14.9 s), not 1.2×. n=250 speedups in `results.csv:36,37,39` are 1.16, 1.18, and 1.36. “Three more” is wrong: five gained (`bmc-pdr-aiger.md:99`). — “Overhead on n=250 s4 is ≈3.1× DRAT’s (text), down from a 4.0× median, still outside 1.3–2×. End-to-end speedup was 2.8× at the median and 1.16–1.36× on n=250; after trimming, ≈1.07× on that family. Run 4 lost two proofs and gained five, including the 94 s rediscovery; 139452p0 went 1.7 s → 129 s. The two-order union beat the better order by 4, short of the pre-registered +8.”

Q8: CONFIRMED — `h200-run2.txt:67` is `cake_lpr (binary, 415372951 bytes): 21.511s`; lrat-check on the same line pair is the 901259413-byte text file. 21.511/1.102≈19.5 and 20.851/1.102≈18.9. — “cake_lpr on that binary file is 21.5 s (≈19.5×), next to text lrat-check at 20.9 s (≈19×).”

Q9: CONFIRMED — not only in this note: drat-trim `2e3b2dc0` in `warp-types-sat/src/lrat.rs:12` and `research/drat-certification.md:17`; cake_lpr `033aad53` in `drat-certification.md:122`; ABC 1.01 and kissat 4.0.1 in `research/bmc-pdr-aiger.md:19,27`; Certifaiger 10.4.0 in `bmc-pdr-aiger.md:93`. None of these strings appear in `h200-session3-results.txt` or `h200-run2.txt`. — none

Q10: REFUTED — the sentence is false for the other verdicts this repo prints. SAT models: `solve.rs:6–8` writes a file that “proves nothing”; `--certify` does not cover SAT (`solve.rs:13–14`). DPLL(T): theory lemmas are counted `unjustified` and “no checker will accept” (`proof.rs:422`). BMC `s BOUNDED_SAFE` and `s EXHAUSTED` are bare prints (`bmc.rs:153–156`). PDR `s EXHAUSTED` is a bare print (`pdr.rs:213`); the invariant is written only with `--invariant` (`pdr.rs:172–197`). — “With `--lrat`, an UNSAT answer is an LRAT proof that lrat-check and cake_lpr accept. A SAT model is not that certificate, and a DPLL(T) proof is emitted unjustified and rejected. With `--invariant` / `--trace`, a PDR Safe answer can be a Certifaiger witness and a PDR counterexample can be replayed. BMC BOUNDED_SAFE, BMC EXHAUSTED, and PDR EXHAUSTED carry neither.”

Other issues, most severe first. §4’s “three more” omits one of the five gains named in `bmc-pdr-aiger.md:99`. The pre-registered P4 bar is per-byte rate (`gpu-lrat-checker.md:78`); 19× is wall-clock of a 901 MB text check against a 415 MB binary check, about 8.7× per byte, which still clears 5×. The italic header says this note was cross-vendor reviewed before push; §7 is still the placeholder. The checker log says four mutants “for each of the 114” (`gpu-lrat-checker.md:125`), which would be 456; the CSV is 113 bases × 4. “Within 2% of drat-trim” is the one n=250 s4 file (41537042 vs 42297997 bytes, `trim-results.csv:17`); the log’s median ratio is 0.92 (`lrat-from-solver.md:94`).

| Q | Verdict | Reason | File:line |
|---|---|---|---|
| 1 | CONFIRMED | 114+66+452=632; 452 CSV rows; session3 states only the total | `h200-session3-results.txt:34` |
| 2 | CONFIRMED | 0.45+0.88+0.27≈1.6; read+parse+check=25.4 s | `h200-session3-results.txt:47–66` |
| 3 | CONFIRMED | CPU 11.61 s; read+decode+device 1.10 s | `h200-run2.txt:66–70` |
| 4 | CONFIRMED | 114 verified rows = 30+30+54, not 114 CNFs | `results.csv:5,55,137` |
| 5 | CONFIRMED | 36/36, 10/10, baseline 20+4, no `False` | `run4b-projected/summary.md:3–8` |
| 6 | CONFIRMED | 21/21 in each of runs 1–3 | `summary.md:2` |
| 7 | REFUTED | “halving” and 1.2× omit 4.0×→3.1× and post-trim 1.07× | `lrat-from-solver.md:91–95` |
| 8 | CONFIRMED | cake_lpr line is the binary file; 19.5× vs 19× | `h200-run2.txt:67` |
| 9 | CONFIRMED | commits and versions exist outside this note | `lrat.rs:12`, `drat-certification.md:17,122` |
| 10 | REFUTED | models, theory, BOUNDED_SAFE, EXHAUSTED are uncertified | `solve.rs:6–14`, `bmc.rs:153–156`, `pdr.rs:213` |

