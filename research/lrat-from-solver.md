# LRAT straight from the solver — pre-registration

*Written 2026-11-03 before any code, after the LRAT-checker arc
(`research/gpu-lrat-checker.md`). Owner: "approved with your suggestions"
(2026-11-02 late), option 1 of the two proposed.*

## 1. Why

The certified chain today is `solve --drat` → `drat-trim -L` → a checker. On the
941 MB-binary proof the checker takes 1.6 s on a GPU pod; `drat-trim` took <!-- unguarded: 941 — proof size in MB -->
**454 s** to produce the LRAT it checks, and `drat-trim` is in the trusted path
(its output is what `cake_lpr` certifies, not the solver's). Emitting LRAT from
the solver removes both: the trimmer's time and the trimmer's trust.

## 2. Design (what will be built)

- `ProofLog` grows three defaulted methods: `wants_hints()`, `add_with_hints(lits,
  hints) -> id`, `delete_ids(ids)`, `empty_with_hints(hints)`. DRAT writers ignore
  them; the new `LratWriter` (text, and binary via `lrat::write_binary`'s layout)
  assigns ids: originals `1..=n` in file order (the DIMACS loader keeps every
  clause, including empty ones, in order — verified in `dimacs.rs`), learned
  clauses `n+1, n+2, …` in emission order.
- The solver core keeps a `CRef → id` map when the log wants hints, remapped on
  compaction like the trail's reasons.
- **The hint chain is derived from the final learned clause, not from the
  resolution order.** Under the assumption "every literal of the learned clause
  is false", every literal of every used clause must be either assumed or
  derived: walk the implication graph from the conflict clause, and for each
  literal that is false on the trail and not in the learned clause, take its
  reason clause and recurse. Emit the collected reasons **sorted by the trail
  position of the literal they imply**, then the conflict clause. Trail order is
  implication order, so each reason is unit when its turn comes and the conflict
  clause is falsified last. This is independent of how minimization was done
  and covers level-0 literals (their unit reasons are collected like any other);
  a decision literal that is not in the learned clause is an analysis bug and is
  reported as such (debug assertion; in release the chain is emitted and the
  checker will say so).
- The empty clause: same walk from the level-0 conflict clause with an empty
  learned clause. An empty ORIGINAL clause: `a id 0 <its id> 0`.
- Theory lemmas (DPLL(T)) have no RUP chain; they are written with an empty hint
  list and counted as `unjustified` — the same honesty as the DRAT path's
  "theory lemmas are not RUP". The CLI is pure SAT.
- CLI: `solve --lrat FILE` (text) and `--lrat-binary`; `--certify` with `--lrat`
  runs `lrat-check` (`$LRAT_CHECK` or PATH) instead of `drat-trim`.
- Fault injection (`proof::fault`, cfg(test)): `DropHint` (drop one hint of each
  chain), `ShuffleHints` (reverse the chain), `WrongId` (off-by-one id) — each
  must flip the in-crate checker's verdict on a hand proof, so that the oracle
  is seen to fail.

## 3. Oracles

1. **In-crate, on every test solve:** `lrat::check_sequential(cnf, parse_text(emitted))`
   must be `Verified` for every UNSAT answer in the solver's existing test
   suite that can be routed through the LRAT log (hand CNFs, PHP, random).
2. **External, on the certification corpus** (`tools/cert-campaign.py`'s 114
   instances + the HWMCC-derived CNFs): `lrat-check` and `cake_lpr` on every
   emitted LRAT; our checker too.
3. **Shape of the proof:** size and step count against `drat-trim -L`'s LRAT for
   the same instance (expected larger: nothing is trimmed).

## 4. Predictions (before any code)

- P1 Every emitted LRAT on the corpus is `VERIFIED` by `lrat-check` and
  `cake_lpr`. **Kill: a rejection that is not a logging bug fixed inside the
  arc — i.e. the chain construction is wrong in principle.**
- P2 Solve-time overhead of LRAT logging vs DRAT logging: between 1.3× and 2×
  the DRAT overhead (the implication walk per conflict is the cost; DRAT's
  overhead was +10–17 % over no logging).
- P3 Emitted LRAT bytes vs `drat-trim -L`'s LRAT for the same instance: within
  1×–3× (ours keeps every learned clause; drat-trim's backward pass drops the
  unused ones).
- P4 End to end, UNSAT answer → external `VERIFIED` (`lrat-check`), the solver
  route beats the `--drat` + `drat-trim -L` route by ≥ 5× on the r3sat n ≥ 200
  instances, where `drat-trim` dominates.
- P5 The checker arc's GPU numbers do not change: the same checker consumes the
  solver's LRAT; what changes is what sits in front of it.
- Error-sign ledger kept per the house rule.

## 5. Not in scope

Trimming (emit only what the final proof uses) — that is `drat-trim`'s job and a
different algorithm; RAT/PR; any theory-lemma justification; solving speed.

## 6. Log (append-only)

- 2026-11-03 — **Built and measured in one sitting** (commit `ad6740e48` + the campaign). Design as in §2, with one correction found by the in-crate oracle before any external run: the conflict during the very first level-0 propagation had its own `proof.empty()` site, which emitted an unjustified empty clause (`unjustified = 1` on a 2-variable hand CNF); it now gets a chain like the others. In-crate oracle (every solve checked by `check_sequential`, `check_independent` and the warp path; text and binary emissions equal): PHP(3–7), parity(3–12), 40 random 3-SAT seeds at n = 60 with reductions every 20 conflicts (deletion + compaction + id remap exercised), three edge CNFs; four faults each flip the verdict (`DropHint`, `ReverseHints`, `WrongId`, `FlipLiteral`). <!-- unguarded: 40, 60, 20, four — test parameters -->
  **Campaign** (`tools/lrat-campaign.py` over the certification corpus's 182 CNFs, 114 UNSAT; data in `research/lrat-from-solver-2026-11-03/`): <!-- unguarded: 182, 114 — campaign tallies, results.csv -->
  - **P1 MET: 114/114 solver-emitted LRATs `VERIFIED` by `lrat-check`, 114/114 by `cake_lpr`, 114/114 by our checker.** No `drat-trim` anywhere in that path. <!-- unguarded: 114 — results.csv -->
  - **P2 MISSED (worse):** logging overhead ratio (LRAT − none)/(DRAT − none), median **4.0×** against the predicted 1.3–2×. In absolute terms on the only family with measurable solve time (r3sat n = 250, median solve 8.8 s): LRAT +20 %, DRAT +6 %. The implication walk per conflict plus the hint bytes is the cost; the prediction under-weighted the walk. Error sign: toward the design's favour.
  - **P3 partly MET:** emitted bytes / `drat-trim -L` bytes, median **0.96** (PHP 0.95, r3sat n = 150 0.96, Tseitin 0.93) — but **4.4× on r3sat n = 250** (250 MB vs 57 MB): on long runs most learned clauses never reach the final proof, and the trimmer removes them while we keep everything. Within the predicted 1–3× for five families, outside it for the one that matters.
  - **P4 MISSED:** end to end (solve + `lrat-check`) vs (solve + `drat-trim -L` + `lrat-check`), median **2.8×** (min 1.16 on n = 250, max 4.7 on cnfuzz); predicted ≥ 5× on n ≥ 200, measured 1.9× (n = 200) and 1.2× (n = 250). Why: on proofs *our* solver produces at this scale, `drat-trim` is quick (5 s on the n = 250 proofs) and the solve itself dominates; the 454 s trim that motivated the arc was on kissat's proof of an instance our solver does not finish. Corpus total: 122.7 s against 151.9 s, 1.24×. Error sign: toward the design's favour again — two of two misses in the same direction, which is the tell the house rule names. <!-- unguarded: 250 — instance size n -->
  - **P5 MET by construction:** the same checker consumed the proofs (`ours` column).
  **Verdict on the arc.** The trust claim holds and is the real result: an UNSAT answer now comes with a proof that a verified checker accepts, produced by the solver alone. The speed claim does not hold at the scale this solver reaches; it would at kissat's scale, where trimming takes minutes, but that is a claim about a solver we do not have. What the misses point at: untrimmed proofs grow 4× on long runs, so the next step on *this* axis is trimming on the fly (mark reachability from the empty clause backwards, which is `drat-trim`'s backward pass) or emitting binary LRAT by default (half the bytes).
- 2026-11-03 — **Backward trimming on our own LRAT** (`lrat::trim`, `lrat-trim` binary): reachability from the first empty clause through the hints, deletions restricted to kept clauses, ids unchanged; linear, no propagation (LRAT already names every antecedent — this is `drat-trim`'s backward pass without its search). On all 114 campaign proofs the trimmed proof verifies with `lrat-check` (114/114). Bytes: trimmed / untrimmed median 0.96 overall but **0.17 on the long runs** — r3sat n = 250: 250 MB → 41.5 MB, against `drat-trim -L`'s 42.3 MB for the same instance (trimmed / drat-trim median 0.92, max 1.03); `lrat-check` on the trimmed proof 0.63 s instead of 3.9 s; the trim itself 2.7 s where `drat-trim` took 5.0 s. So P3's miss closes: the solver's proof plus a linear trim is the size of the trimmer's output. P4 barely moves (solve + trim + check 13.9 s vs 14.9 s on n = 250, ≈ 1.07×): the remaining gap is the logging overhead itself (P2), which trimming cannot touch. Data: `research/lrat-from-solver-2026-11-03/trim-results.csv`. <!-- unguarded: 114, 250 — campaign tallies / instance size -->
- 2026-11-03 — **Overhead, measured and halved.** [CORRECTED 2026-11-03, account panel Q7: not halved — LRAT text went from +19 % to +14 % of solve time and the ratio to DRAT's overhead stayed ≈ 3× on this instance; the corpus median before the change was 4.0×.] One core, best of 3, r3sat n = 250 s4: no proof 16.76 s; DRAT 17.50 (+4 %); LRAT text 19.06 (+14 %); LRAT binary 18.46 (+10 %). Before: DRAT +7 %, LRAT text +19 %, binary +13 %. Two changes: the `CRef → id` map is a vector indexed by arena offset (the hash lookup per reason per conflict was the walk's largest cost) and numbers are formatted into reused buffers instead of `to_string()` per number (that was ≈ 1 s per 250 MB; the DRAT writer gained the same). The remaining LRAT-over-DRAT cost, ≈ 1 s here, is the implication walk and the extra bytes; sharing `analyze`'s seen-set would shave part of it and is not worth its coupling today. P2 restated: ≈ 3× the DRAT logger's overhead (text), 2.3× (binary) — still outside the predicted 1.3–2×. <!-- unguarded: 250 — instance size n --> <!-- unguarded: 16.76, 17.50, 19.06, 18.46 — seconds -->
- 2026-11-03 — **The whole chain on one GPU machine, run 2** (H200 `77hccgqd7uj0z2`, upload 27 s this time; data `research/lrat-from-solver-2026-11-03/chain/h200-run2.{csv,txt}`). **54/54 device VERIFIED, 54/54 `lrat-check`, 54/54 `cake_lpr`.** Largest instance (r3sat n = 250 s4, this pod's CPU is slower than the workstation's): solve + binary LRAT 21.3 s, trim 5.6 s (115 MB → 19 MB), decode 0.03 s, device 0.015 s; on the trimmed proof `lrat-check` 0.85 s, `cake_lpr` 1.24 s. The producing path has nothing external in it, and the checking step is now a rounding error next to the trim, which is next to the solve. <!-- unguarded: 54, 250, 115, 19 — tallies / MB from h200-run2 -->
  **The same-machine comparison the panel asked for (Q7), on the 415 MB external proof (`r3sat_n350_s3`, kissat + drat-trim; 901 MB as text):** `lrat-check` 20.85 s (text), `cake_lpr` 21.51 s (binary), our CPU route 11.6 s (parse 7.3 + 4-thread check 4.1), **device path 1.10 s** (read 0.22 + decode 0.76 on 32 threads + device 0.13 including transfer) — **≈ 19× both external checkers, ≈ 10× our own CPU route, a single machine and a single proof.** This replaces the cross-machine ratios the first write-up carried. <!-- unguarded: 415, 901 — proof sizes in MB -->
- 2026-11-03 — **Pre-registration: the chain on the local RTX 4000 SFF Ada** (owner 11-03: local GPU runs allowed; the card is compute-only, `nvidia-compute-only` in memory — the earlier "drives the cockpit display" premise was wrong and is annotated in `gpu-lrat-checker.md`). Same script as the pod's, local paths (`reproduce/local-lrat-chain.sh`), six niced cores, decode threads 6. Written before the run: (P1) 54/54 instances: device VERIFIED, `lrat-check` and `cake_lpr` agree — the card cannot change a verdict; (P2) device time on the 415 MB external proof **0.5–2.0 s** against the H200's 0.13 s (sm_89, ~360 GB/s of memory bandwidth against ~4.8 TB/s, 20 GB of VRAM — the decoded arrays are ~0.7 GB, so no spill); (P3) `lrat-check` and `cake_lpr` on this box **12–20 s** on that proof (this CPU ran the trim in 2.7 s where the pod took 5.6 s, so faster single-thread than the pod's 20.9 / 21.5 s); (P4) the local device path still beats both external checkers by ≥ 5× end to end on this one machine; (P5) PTX for sm_89 loads with 0 externs (the pre-push hook now checks this). Kill: P1. <!-- unguarded: 54, 415 — count / MB -->
- 2026-11-03 — **The chain on the local RTX 4000 SFF Ada, run** (`reproduce/local-lrat-chain.sh`; six niced cores; data `research/lrat-from-solver-2026-11-03/chain/local-rtx4000ada.{csv,txt}`; the "59 bytes / 60 bytes" in the text file are `stat` on the symlinks, not the proofs — 415 MB binary / 901 MB text as before). **P1 MET: 54/54 device VERIFIED, 54/54 `lrat-check`, 54/54 `cake_lpr`.** **P2 MISSED, in the pessimistic direction for once:** device time on the 415 MB proof **0.275 s**, below the 0.5–2.0 s predicted from the bandwidth ratio — 2.1× the H200's 0.13 s, not 4–10×; at this size the kernel is not bandwidth-bound, or transfer and launch overhead carry the H200's number too. **P3 MET: `lrat-check` 16.2 s, `cake_lpr` 17.4 s** on this box (predicted 12–20 s). **P4 MET: device path 1.2 s (read 0.14 + decode 0.81 on 6 threads + device 0.28) — ≈ 13× `lrat-check` and ≈ 14× `cake_lpr`, a single machine and a single proof; our CPU route 9.2 s (parse 6.2 + 4-thread check 3.0).** **P5 MET: sm_89 PTX, 0 externs.** Largest chain instance: solve 18.3 s, trim 4.6 s, decode 0.03 s, device 0.02 s, `lrat-check` 0.65 s. So the same-machine claim now holds on two machines: ≈ 19× on an H200 with 32 decode threads, ≈ 13× on a workstation card with 6. Error-sign ledger: first pessimistic miss of the week; recorded as such. <!-- unguarded: 54, 415, 901 — count / MB -->
