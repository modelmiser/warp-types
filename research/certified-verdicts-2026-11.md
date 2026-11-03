# Certified verdicts from warp-types: a week of pre-registered measurements (2026-11-01 → 11-03)

*A single account of what was measured, what was predicted, what missed, and what each public
claim rests on. Every number below is in a research log with its data directory; this document
adds nothing to them except the order. Written 2026-11-03, cross-vendor reviewed before being
pushed (panel record at the end).*

## 1. The claim, in one paragraph

Two kinds of verdict from this repository now come with a certificate that a tool we did not
write accepts: an UNSAT answer from the pure-SAT solver run with `--lrat`, and a SAFE answer from
the PDR model checker. For SAT the certificate is an LRAT proof written by the solver itself,
trimmed by a linear pass, and accepted by `lrat-check` and by the formally verified `cake_lpr`;
the same proof is checked on a GPU by a warp-typed kernel in a fraction of the time either takes,
on the same machine. For model checking the certificate is an HWMCC witness circuit accepted by
Certifaiger. An UNSAFE answer from PDR comes with a counterexample that replays through the
circuit under *our own* independent simulator at ABC's depth — a second opinion, not an outside
one. Not certified at all: SAT models (checked by our own runner), DPLL(T) answers (theory lemmas
are written without a chain and no checker accepts them), BMC `BOUNDED_SAFE` and `EXHAUSTED`,
PDR `Exhausted`, and the SMT crate's EUF verdicts (cross-checked against Yices since the same
night, `research/smt-crosscheck.md`, not certified). *Added later the same night:* a QF_BV UNSAT
answer through the new bit-blaster carries an LRAT proof of the bit-blasted CNF that
`lrat-check` and `cake_lpr` accept (`research/smt-bitblast.md`: 254/254); the translation from
terms to CNF is ours and uncertified. All of this was pre-registered, run against
outside references, and read by reviewers from four model families before the public text was
written; where the text was wrong the first time, the changelog says so.

## 2. Method, because it is the result

- **Pre-registration.** The two SAT arcs opened with a file naming predictions and kill
  conditions before any code (`research/gpu-lrat-checker.md` §4, `research/lrat-from-solver.md`
  §4); each PDR campaign run was pre-registered in `research/bmc-pdr-aiger.md` before it ran
  (the engine code for runs 4–4c existed first; the predictions did not change after). Misses
  are recorded with their sign.
- **Outside references, plus separately written in-repository checks.** Outside: `lrat-check`
  (drat-trim repository, commit 2e3b2dc0) and `cake_lpr` (commit 033aad53) for proofs; ABC 1.01
  `bmc3`/`pdr` for verdicts and counterexample depths; Certifaiger 10.4.0 (kissat backend) for
  witness circuits. Ours, written separately from the engines and watched to fail:
  `tools/check-invariant.py` (own AIGER reader and Tseitin, kissat 4.0.1 as the solver) and
  `tools/replay-trace.py` (own simulator, no solver). Versions are recorded in
  `research/drat-certification.md` (tools) and `research/bmc-pdr-aiger.md` (references); the
  `lrat-check` compiled on the chain pod was built from the same source file without its commit
  being re-recorded there.
- **Negative controls that exercise the mechanism.** The checker's first archive had 33 "corrupted"
  proofs that were all merely missing their empty clause; a cross-vendor panel caught that no
  rejection had ever exercised the hint logic. The replacement is 452 hint-level mutants of the
  genuine proofs (a hint dropped, a literal flipped, a forward hint, a use after delete), all
  rejected by every checker, with the mechanism's own reasons. Certifaiger was watched to reject
  witnesses built from the property clause alone.
- **Cross-vendor reading of soundness-shaped text.** Five seats, four families, reading the repo,
  disagreements first (`research/gpu-lrat-checker-2026-11-02/panel/`). The panel's refutations
  changed the public text (§5); one of its refutations was itself wrong and `lrat-check.c`
  settled it.

## 3. Results, with the pointer that carries each

| Claim | Number | Where it is |
|---|---|---|
| Warp-typed LRAT checker agrees with `lrat-check` and `cake_lpr` | 632/632 files (114 genuine, 66 drat-trim outputs from corrupted DRATs, 452 hint-level mutants) | `research/gpu-lrat-checker-2026-11-02/`, archive test in `lrat.rs` |
| Device verdict = CPU reference | 632/632 on an H200 | `h200-session3-results.txt` |
| Device check, 941 MB binary proof, incl. transfer | 0.27 s; parallel decode 0.88 s (32 thr); pipeline ≈ 1.6 s vs 25 s CPU route, same pod | same file; `gpu-lrat-checker.md` session 3 | <!-- unguarded: 941 — proof size in MB -->
| Same machine, same proof (415 MB binary / 901 MB text) | `lrat-check` 20.9 s (text, one process), `cake_lpr` 21.5 s (binary, one process), our CPU route 11.6 s (4 threads), device path 1.1 s (32 decode threads + the H200) — ≈ 19× wall-clock; per byte against the text checker ≈ 8.7×, which is the pre-registered bar's unit and still clears its 5× | `research/lrat-from-solver-2026-11-03/chain/h200-run2.txt` |
| LRAT emitted by the solver, verified externally | 114/114 proof-producing campaign rows — **54 distinct UNSAT instances** (30 appear in all three corpus runs with byte-identical proofs, 24 once): `lrat-check`, `cake_lpr`, ours | `research/lrat-from-solver-2026-11-03/results.csv` |
| Whole chain on one GPU machine (solve → trim → device) | 54/54 instances; `lrat-check` and `cake_lpr` agree on every trimmed proof | `.../chain/h200-run2.csv` |
| Linear backward trim | on the largest instance (n = 250 s4) 250 MB → 41.5 MB, within 2 % of `drat-trim -L`'s size; median across the corpus 0.92 of drat-trim's; all 114 campaign rows' trimmed proofs verified | `.../trim-results.csv` |
| LRAT logging overhead (n = 250 s4, one core, best of 3) | DRAT +4 %, LRAT text +14 %, binary +10 % (corpus median of the overhead ratio before the fix: 4.0×) | `lrat-from-solver.md` |
| PDR coverage, 119 HWMCC17 circuits < 20 KB, 60 s | 36 proved + 10 refuted (run 4b), from 20 + 4 (run 2/3) | `research/bmc-pdr-hwmcc17-2026-11-01/run4b-projected/` |
| PDR invariants certified | 36/36 `check-invariant.py` (kissat), 36/36 Certifaiger `valid witness` | same dir, `certifaiger.txt` |
| PDR counterexamples | 10/10 replay at ABC's `bmc3` depth | same dir |
| BMC counterexample depths vs ABC | the same 21 instances agree 21/21 in each of runs 1–3; 0 verdict contradictions in any run 1–4c | `results.csv`, run 1–3 |
| Generalisation order, measured whole | asc 46 verdicts, desc 44, union 50, overlap 40 | `run4c-desc/` |

## 4. What missed, and which way

Every miss is in the logs with a sign; the pattern is the finding.

- **P4 of the checker arc** ("≥ 5× `lrat-check`") was first met on a cross-machine comparison the
  panel rightly refused; it is met on the same-machine one (≈ 19×). The first README wording was
  wrong and the changelog records it.
- **P2 and P4 of the solver-LRAT arc** (overhead 1.3–2× DRAT's; end-to-end ≥ 5× over drat-trim)
  missed: the overhead ratio was 4.0× at the corpus median and, after a `CRef`-indexed id table
  and allocation-free formatting brought LRAT text from +19 % to +14 % of solve time (DRAT from
  +7 % to +4 %), still ≈ 3× on the largest instance — not halved, as one log entry's title says;
  end-to-end 2.8× at the median and 1.2× on the largest instances this solver finishes (≈ 1.07×
  once the trim is included on n = 250) — the 454 s trim that motivated the arc was on a stronger
  solver's proof of an instance ours cannot solve. Both misses flattered the design.
- **P1 of PDR run 4** ("no run-3 verdict lost") missed by one: cubes over latches cannot say "the
  bad gate is on", so an already-inductive property took 94 s (with a 300 s cap) to rediscover;
  the property's variable joined the cube variables (run 4b), which restored it and four more
  (five gained) and cost two — both lost at the 60 s cap, one of them a 1.7 s proof that needs
  129 s with a 300 s cap. **P2 of run 4c** (portfolio threshold +8) missed at +4; the order
  stays a flag by the rule written first.
- Four code defects found by the panel and fixed before any of the above was claimed: a kernel
  hash table that could fill and then search forever; a device result buffer that read "pass"
  for lemmas never written; no warp barrier between a lemma's writes and reads; unbounded
  allocations on one huge id or literal.

## 5. What a passed check means, and does not

A verified LRAT says *this* UNSAT answer is correct; it says nothing about the solver in general,
and RAT steps are reported unsupported, never verified. A valid witness says *this* invariant,
for *this* circuit, is inductive and implies the property under Certifaiger's own encoding; it
says nothing about the 73 circuits that timed out. The GPU numbers are for one H200 and one
proof family (random 3-SAT and the certification corpus); the per-lemma assignment table holds
4,095 literals and lemmas beyond it are reported `TableOverflow`, not checked, on the warp paths.
The 19× is wall-clock, a 32-thread decode plus a GPU against single-process checkers; against
text-format `lrat-check` it is also a different encoding (per byte ≈ 8.7×); the same-encoding
comparison is `cake_lpr` on the binary file at 21.5 s, ≈ 20×.

## 6. Reproduce

`reproduce/runpod-lrat.sh`, `reproduce/runpod-lrat-chain.sh` (one RunPod session each; the
drivers upload the corpus); `tools/lrat-campaign.py`, `tools/lrat-mutate.py`,
`tools/bmc-pdr-campaign.py --skip-bmc`, `tools/check-invariant.py`, `aig-witness` +
`certifaiger`. Not in the repository and needed: the HWMCC17 AIGs (fmv.jku.at), the
certification-corpus CNFs and the kissat/drat-trim-produced large proofs (generated by
`tools/cert-campaign.py` and the corpus scripts named in `research/gpu-lrat-checker.md` §3),
and the external binaries. Toolchain pin `nightly-2026-10-28`; tool commits as in §2.

## 7. Panel record

Five seats (claude-fable-5-1, claude-opus-5-5, gpt-5.6-sol-xhigh, grok-4.7-medium,
gemini-3.7-flash-high), reading the repository against the ten questions in the brief; reports in
`research/certified-verdicts-2026-11/panel/`. Confirmed unanimously: the 632-file composition, the
H200 timings and their arithmetic, the PDR counts and certifications, the BMC depth agreement.
Refuted, and fixed above: "every verdict … comes with a certificate" (5/5 — SAT models, DPLL(T),
BMC bounded results, PDR `Exhausted` and the SMT crate have none; UNSAFE is replayed by our own
tool); "114/114 corpus proofs" hid that they are 54 distinct instances (3/5); "after halving it"
was false — +19 % → +14 %, ratio still ≈ 3×, corpus median 4.0× (4/5); "three more" was five
gained, and the lost proof was lost at the cap (4/5); "outside oracles only" was false for the
replayer and the invariant checker (1/5, GPT, and right); the 19× is wall-clock with 32 host
threads and a GPU against one process, and ≈ 8.7× per byte on the pre-registered unit (Grok,
GPT); "before any code" overstated the PDR runs' pre-registration; "within 2 %" was one file.
Gemini confirmed every question, including the one all four others refuted. Consensus was, as
the house rule says, the low-signal output: the opening sentence failed 5/5 and the author had
already suspected it; the finding that mattered — the 54-instance count behind "114" — came from
three seats and was not suspected.
