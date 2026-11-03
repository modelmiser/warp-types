# claude:claude-fable-5-1

Verdicts (file paths relative to repo root; `R/` = `research/`):

**Q1: CONFIRMED** — `R/cert-campaign-2026-11-01/run{1,2,3}-results.csv` have 30+30+54 = 114 VERIFIED rows; `*-negatives.csv` have 18+18+30 = 66; `R/gpu-lrat-checker-2026-11-02/negatives-results.csv` has 452 data rows, 452 distinct names (113 × 4 kinds — note: 113, not "each of the 114" as `gpu-lrat-checker.md:125` says); 114+66+452 = 632 matches `h200-session3-results.txt:34` ("632 proofs, 632 agree … 147 verified"). Caveat: session 3 line 34 only shows *device = CPU reference*; the three-way agreement with `lrat-check`/`cake_lpr` on all 632 is in no repo data file (`gpu-lrat-checker.md:128` records 392/392; run1/run2 CSVs have an empty `cake_lpr` column on 60 genuine + 36 negatives). — Replacement for row 1's pointer: "`negatives-results.csv` (452, both external checkers), cert-campaign CSVs (180; `cake_lpr` only on run 3's 54 + the `lrat.rs` archive assertion)".

**Q2: CONFIRMED** — `h200-session3-results.txt:47` device 0.265 s (best of 0.438/0.265/0.321 across block counts); `:65-66` 32-thread decode 0.881/0.885 s; `:49` read 0.447 + parse 17.556 + independent(96) 7.350 = 25.35 s; 0.447+0.885+0.265 = 1.60 s. — "0.27 s (4096 blocks, best of three block counts)".

**Q3: CONFIRMED** — `chain/h200-run2.txt:66-70`: lrat-check 20.851 s, cake_lpr 21.511 s; CPU route 0.217+7.256+4.137 = 11.61 s; device path 0.217+0.755+0.130 = 1.10 s; 20.85/1.10 = 18.9. Caveat: CPU route ran `independent(4)` — 4 check threads on a 128-core pod (`:4`), not comparable with session 3's 96-thread figure. — "our CPU route 11.6 s (4 check threads)".

**Q4: REFUTED (wording)** — `R/lrat-from-solver-2026-11-03/results.csv`: 182 rows, 114 with `lrat_check`=VERIFIED, `cake_lpr`=VERIFIED, `ours`=s VERIFIED. But names are `run1_*`, `run2-default2000_*`, `run3_*`: only 54 distinct CNFs; 30 CNFs appear three times with identical `cnf_sha` (cert-campaign CSVs) and identical `lrat_bytes` (deterministic solver → identical proofs). "114 corpus proofs" silently counts 60 duplicates. — "114/114 campaign proofs (54 distinct CNFs; 30 of them solved identically in three campaign copies)".

**Q5: CONFIRMED** — `run4b-projected/results.csv`: our_pdr UNSAT 36 / SAT 10, inv_check CERTIFIED 36, pdr_replay 10 at ABC's bmc3 depth; `certifaiger.txt` 36 distinct "valid witness", 0 failures; `run2-summary.md`/`run3-coi-summary.md` 20 + 4; "contradictions: 0" in `summary.md`, run2, run3, run4, run4b, run4c. — none.

**Q6: CONFIRMED** — `summary.md:2`, `run2-summary.md:2`, `run3-coi-summary.md:2` each "21/21"; run 4* are `--skip-bmc`. — none.

**Q7: REFUTED (one number)** — `lrat-from-solver.md:95` ≈3× (text; 2.3× binary), `:91` 2.8× median / 1.2× n=250 / 454 s; `bmc-pdr-aiger.md:97` 94 s; `:99` 1.7 s → 129 s (300 s cap); `:103` union +4 vs +8. But `:99` and run4 vs run4b CSV diff: 2 lost, **5 gained** (nusmvreactorp5 + 4 others), so "restored it and three more" understates. Nothing stated more favourably than the log. — "which restored it and four more and cost two".

**Q8: CONFIRMED** — `reproduce/runpod-lrat-chain.sh:68` feeds `big.blrat` to cake_lpr; `h200-run2.txt:67` "cake_lpr (binary, 415372951 bytes): 21.511s"; 21.5/1.10 = 19.5 ≈ 19. — none.

**Q9: CONFIRMED** — `R/drat-certification.md:17,22` (drat-trim/lrat-check 2e3b2dc0…), `:122` (cake_lpr 033aad53…), `R/bmc-pdr-aiger.md:19` (ABC 1.01), `:27` (kissat 4.0.1), `:93` (Certifaiger 10.4.0). Document cites none of these. The pod runs upload a prebuilt `cake_lpr` and `lrat-check.c` whose commit is not logged in the pod output. — Append to §2: "(commits recorded in `research/drat-certification.md` §tools and `research/bmc-pdr-aiger.md`)".

**Q10: REFUTED** — "Every verdict this repository's solvers now produce … comes with a certificate that a tool we did not write accepts" (line 10). SAT models: checked by the campaign's own reader, kissat agrees on verdict only (`warp-types-sat/README.md:200`, `drat-certification.md:133`). DPLL(T): `warp-types-sat/README.md:195-196` "theory lemmas … are not RUP, so a DPLL(T) proof will not check". BMC BOUNDED_SAFE (`warp-types-bmc/src/checker.rs:21`; 7–9 such in runs 2/3) and Exhausted/TIMEOUT: no certificate. — "Every UNSAT answer from the pure-SAT solver and every SAFE/UNSAFE answer from the PDR model checker comes with a certificate an outside tool accepts; SAT models are checked against the original file by our own runner, and DPLL(T), BMC BOUNDED_SAFE and timeouts carry no certificate."

**Not asked, most severe first**
1. §3 row 1 pointer (session 3) does not evidence the external-checker part of 632/632 (Q1 caveat); ~96 files lack a `cake_lpr` result in any repo CSV.
2. 114 ≠ 114 distinct instances (Q4); the chain row's 54 is the real corpus size.
3. §3 "LRAT logging overhead (n = 250, one core)" is one instance (s4), best of 3 (`lrat-from-solver.md:95`) — say so.
4. CPU-route thread counts differ between the two same-machine rows (96 vs 4); readers will compare 25 s vs 11.6 s.
5. §3 row "0.27 s" is the best of three block configurations, not a single pre-chosen setting.

| Q | verdict | reason | file:line |
|---|---|---|---|
| 1 | CONFIRMED | counts sum; 452 distinct; external-agree pointer weak | h200-session3-results.txt:34; negatives-results.csv |
| 2 | CONFIRMED | 0.265/0.885/25.35/1.60 | h200-session3-results.txt:47,49,66 |
| 3 | CONFIRMED | 11.61 / 1.10 / 18.9×; 4 threads | chain/h200-run2.txt:66-70 |
| 4 | REFUTED | 114 rows = 54 CNFs, 30 tripled | results.csv names; run*-results.csv cnf_sha |
| 5 | CONFIRMED | 36/10/36/36/10; 20+4; 0 contradictions | run4b-projected/results.csv, certifaiger.txt |
| 6 | CONFIRMED | 21/21 in each of runs 1–3 | summary.md:2, run2/run3 summaries:2 |
| 7 | REFUTED | "three more" is four (5 gained, 2 lost) | bmc-pdr-aiger.md:99 |
| 8 | CONFIRMED | binary input; 19.5× | runpod-lrat-chain.sh:68; h200-run2.txt:67 |
| 9 | CONFIRMED | commits/versions logged, uncited | drat-certification.md:17,122; bmc-pdr-aiger.md:19,27,93 |
| 10 | REFUTED | SAT models, DPLL(T), BOUNDED_SAFE uncertified | warp-types-sat/README.md:195-200; checker.rs:21 |
