# claude:claude-fable-5-1

**Verdicts**

Q1: CONFIRMED — non-unit hint → `NotUnit` lrat.rs:414-416; deleted hint removed from arena at :467 then `UnknownHint`→`DeletedHint` :407,:505; forward id :494; no empty lemma → `NoEmptyClause` :523; deletion of unknown/already-deleted id → `BadDeletion` :467-473. Caveat: a hint containing an already-true literal is counted "free" (:411-418), so satisfied+one-free → NotUnit, satisfied+none → no-op "unit"; sound either way. — none

Q2: CONFIRMED (with caveat) — negative hint parsed (:126, :210), refused before checking at :485, :568, lrat_warp.rs:130, from_binary :309 (ParseError), kernel has only dense u32 hints. No RAT lemma is ever *checked and accepted*. Counterexample to "proof containing RAT → never Verified": a RAT step *after* the first empty lemma is never reached (:516-517, :579, lrat_warp.rs:320) → reference Verified, while `flatten` (first pass over all steps, :120-150) says Unsupported — reference and SimWarp disagree on that input. — "RAT lemmas are never checked: a proof that reaches a RAT lemma before deriving the empty clause is reported `Unsupported`."

Q3: CONFIRMED for parse_binary (lrat.rs:156-233: tag, varint, `id = m>>1`, two 0-terminated lists; truncation/trailing continuation bit → `Err` :160; id 0 → IdNotIncreasing :478). Equality from_binary ≡ flatten is asserted: lrat_warp.rs:785-815 (all fields, 5 hand cases) and :864-873 (hints, deleted_at, verdict on archive). REFUTED on "no panic": `from_binary` does `dense_of.resize(id+1)` (:285-287) — a single varint id ≈2⁴⁰ aborts on allocation; and from_binary stops at the first empty lemma (:320) while parse_binary/flatten scan the whole file, so trailing garbage or a bad id after the empty lemma gives Ok vs Err (comment :214 "identical" overstated). — none in public text

Q4: CONFIRMED — (i) all four use strict `<` (lrat.rs:612, lrat_warp.rs:499, lib.rs:424); equality impossible since lemma and deletion are distinct steps. (ii) all count non-falsified literals (lrat.rs:411, lrat_warp.rs:511, lib.rs:436); mutually consistent; parity with `lrat-check` on the satisfied-hint corner is UNSUPPORTED (lrat-check source not in repo). (iii) `combine` requires `first_empty.is_some()` and every code Ok (lrat_warp.rs:606-641); TableOverflow → Rejected :624. — none

Q5: (i) REFUTED — host caps at 4095 (lrat_warp.rs:390) but the kernel's insert succeeds at the 4096th entry unless the empty slot is the last probe (lib.rs:353-369: 4095 probes examine 4095 of 4096 slots); table then 100 % full and `lrat_falsified` on an absent literal never terminates (:336-345, no stale/empty slot) — a hang, not code 5. (ii) CONFIRMED — stamp i+1 unique per lemma, host zeroes once (gpu_launcher.rs:427-428), test at :338/:356 correct. (iii) CONFIRMED in practice — reads are satisfied by the same lane's program-order writes; breaks depend on warp-uniform values (:442, :458). Formally the identical concurrent writes are a data race (UB in the Rust model). — add to README: "lemmas whose assignment exceeds 4,095 literals are not checked on this path".

Q6: REFUTED on the characterisation — lrat.rs:860-927 does run `lrat-check` and `cake_lpr` live, but only when `LRAT_ARCHIVE_DIRS`/`LRAT_CHECK`/`CAKE_LPR` are set; the archive is not in the repo and no per-file output is recorded. The 180 = 114 genuine + 66 LRATs that `drat-trim -L` emitted from corrupted *DRATs* (cert-campaign.py:139-151, :224-229); campaign CSVs show 33 of those 66 are valid refutations (lrat-check VERIFIED) and 33 are partial proofs (research/drat-certification.md:122 "empty clause not derived"). So "147 genuine, 33 corrupted" is wrong, and the 33 rejections exercise `NoEmptyClause`, not hint checking. H200 "agreement" (runpod-lrat.sh:55-76) is device vs `check_independent` (`LRAT_SKIP_SLOW=1`), `is_verified()` only (examples/lrat_check.rs:111,157). — "180 archived LRATs (114 from genuine proofs, 66 drat-trim outputs on corrupted DRATs, of which 33 are valid refutations and 33 lack the empty clause); verdicts matched lrat-check and cake_lpr in one local run (test `archive_agreement_with_external_checkers`, archive not in repo)."

Q7: CONFIRMED on 0.28 s (h200-session2-results.txt:46-47; span ta..tb = gpu_launcher.rs:420-450, htod + zero-alloc + launch + dtoh). REFUTED on the ratios: 41.7 s is from the log only (gpu-lrat-checker.md:99), text form, *local* machine; the 10× end uses home decode 3.4 s + pod device 0.28 s — a composite never measured on one machine; "≈2.7×" = 10.7/4.1 is the same composite; pod-measured CPU pipeline is 0.45+5.9+6.9 = 13.2 s vs 6.6 s → 2.0×. — "On the H200 pod the binary pipeline took 6.6 s (read 0.45, decode 5.9, device 0.28); the 128-thread CPU check on the same decoded arrays took 6.9 s (pipeline 13.2 s, 2.0×); lrat-check on the 2.06 GB text form took 41.7 s on a different, local machine (≈6×)."

Q8: UNSUPPORTED (from knowledge, not repo) — I know of no published GPU LRAT/DRAT checker; nearest: Osama/Wijs/Biere, *Certified SAT solving with GPU accelerated inprocessing* (TACAS 2021, proofs generated on GPU, checked on CPU), Lammich's multithreaded GRATgen (CADE 2017), Heule et al. parallel checking (ITP 2017). "No ... exists" should read "none found in two searches".

Q9: REFUTED — README:212 "runs unchanged on SimWarp and on the device": lrat_warp.rs:478-551 and lib.rs:378-473 are two hand-written implementations. — "the same organisation written twice (SimWarp and nvptx), held to equal verdicts on the archive." Also "text and binary LRAT" with lrat-check semantics: lrat-check reads text only; binary parity rests on one cake_lpr acceptance (log :96).

**Not asked, by severity**
1. Kernel hang at exactly 4096 assigned literals (Q5i); the public text never states the 4,095-literal cap.
2. `codes` buffer is zero-initialised = Ok (gpu_launcher.rs:429): any lemma the kernel fails to write reads as passed — fail-open; initialise to 2.
3. H200 agreement is boolean-only against `check_independent`, not the sequential reference.
4. Reference vs flatten disagree on steps after the empty lemma (Q2/Q3).

| Q | verdict | reason | file:line |
|---|---|---|---|
| 1 | CONFIRMED | all four rejections enforced | lrat.rs:416,467,494,523 |
| 2 | CONFIRMED* | RAT refused before check; post-empty RAT unreached | lrat.rs:485,517; lrat_warp.rs:130 |
| 3 | REFUTED (panic/identity) | huge id allocation; stops at first empty | lrat_warp.rs:285,320 |
| 4 | CONFIRMED | strict `<`, same free-count, combine needs empty | lib.rs:424; lrat_warp.rs:634 |
| 5 | REFUTED (i) | 4096th insert succeeds → full table → lookup loops | lib.rs:353-369,336-345 |
| 6 | REFUTED | 147/33 mischaracterised; external run not archived | cert-campaign.py:224; lrat.rs:860 |
| 7 | REFUTED (ratios) | cross-machine composite; pod-only ratio 2.0× | results.txt:46-49; md:115 |
| 8 | UNSUPPORTED | knowledge only; no GPU checker known to me | gpu-lrat-checker.md:20 |
| 9 | REFUTED | "runs unchanged" — two implementations | README.md:212 |
