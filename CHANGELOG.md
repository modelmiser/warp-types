# Changelog

## [Unreleased]

- **`warp-types-smt`: QF_BV through the certified SAT path** (`SmtSession::bitblast`, `BitBlasted::{solve,
  solve_lrat, dimacs}`): a complete bit-blaster (ripple-carry add/sub, bitwise ops, extract, concat, equality
  by bit equivalence, Tseitin) whose UNSAT answers carry an LRAT proof of the written CNF. **Measured
  2026-11-03** (pre-registered, `research/smt-bitblast.md`): on 1,000 random QF_BV instances the verdict equals
  Yices 2.7.0's on all — including the 62 the ground-only module leaves as `Sat` — and all 254 UNSAT proofs are <!-- unguarded: 1,000, 62, 254 — tallies, research/smt-bitblast-2026-11-03/ -->
  accepted by `lrat-check` and `cake_lpr`. The translation itself is not certified; `smt-crosscheck --bitblast
  --lrat-check --cake-lpr` runs the comparison.
- **`warp-types-smt` verdicts cross-checked against Yices** (`SmtSession::to_smtlib2`, `smt-crosscheck`): on 1,000
  random QF_UF instances every verdict equals Yices 2.7.0's; on 1,000 random QF_BV instances no `Unsat` is
  contradicted, and the ground-only BV module's incompleteness is measured at 23 % of Yices-unsat instances
  (pre-registered, `research/smt-crosscheck.md`). A verdict cross-check, not a certificate.
- **PDR: latch-projected cubes** (`PdrOptions::project_cubes`, default on; `pdr --no-project` for the old engine).
  `TransitionSystem` gains combinational clauses (`add_comb`) and a cube-variable mask (`set_cube_vars`); the
  AIGER lowering declares the latches and the property's variable as cube variables and the gate definitions as
  combinational clauses; the CTI query carries them; obligations keep full states for traces. **Measured
  2026-11-03** (runs 4/4b, pre-registered in `research/bmc-pdr-aiger.md`): on the 119 small HWMCC17 circuits at
  60 s, **36 proved + 10 refuted, from 20 + 4**; all 36 invariants certified by `tools/check-invariant.py` and by <!-- unguarded: 119, 36, 10, 20, 4 — campaign counts, research/bmc-pdr-hwmcc17-2026-11-01/run4b-projected/ -->
  Certifaiger; all 10 counterexamples replay at ABC's depth; fences 2–70× smaller. Two earlier proofs now exceed
  the cap. The literal order in generalisation (`PdrOptions::descending_cubes`, `pdr --cube-order asc|desc`) was
  measured whole: 44 verdicts descending vs 46 ascending, union 50 — below the pre-registered portfolio threshold, <!-- unguarded: 44, 46, 50 — campaign counts, research/bmc-pdr-hwmcc17-2026-11-01/run4c-desc/ -->
  so it stays a flag.
- **Certifaiger witnesses for PDR proofs** (`warp-types-bmc`): `aiger::write_witness` and the `aig-witness` binary turn a
  `pdr --invariant` file into an HWMCC witness circuit (the model verbatim with bad := ¬(Inv ∧ P), identity mapping);
  `aiger::parse_invariant_file`. **Checked 2026-11-03:** all 20 HWMCC17 instances PDR proves are `valid witness` under <!-- unguarded: 20 — tally, research/bmc-pdr-hwmcc17-2026-11-01/certifaiger/ -->
  Certifaiger 10.4.0 (kissat backend); a witness built from the property alone is rejected (`Inductive check failed`).
  Record: `research/bmc-pdr-aiger.md`.
- **LRAT straight from the solver** (`warp-types-sat`): `solve --lrat FILE [--lrat-binary]` and
  `solver::solve_watched_lrat_with` write clause ids and a RUP hint chain per learned clause as the solver runs
  (`lrat_emit::rup_chain`: implication-graph walk from the final learned clause, reasons in trail order, conflict
  clause last); `--certify` with `--lrat` runs `lrat-check`. `ProofLog` gains defaulted `wants_hints` /
  `add_with_hints` / `delete_ids` / `empty_with_hints`; `proof::LratWriter` (text or binary); `Trail::position_of`.
  **Measured 2026-11-03** (pre-registered, `research/lrat-from-solver.md`; `tools/lrat-campaign.py`): all 114 UNSAT (54 distinct instances across the corpus's three runs) <!-- unguarded: 114 — campaign tally, research/lrat-from-solver-2026-11-03/ -->
  proofs of the certification corpus verified by `lrat-check`, `cake_lpr` and the in-crate checker, with no
  `drat-trim` in the path. Two predictions missed, both in the design's favour and recorded: logging overhead ≈ 4×
  the DRAT logger's (+14 % text / +10 % binary vs +4 %, after a `CRef`-indexed id table and allocation-free number
  formatting — the DRAT writer gained from the latter too), and the end-to-end gain over DRAT + `drat-trim` is 2.8× at the median, 1.2× on the largest
  instances this solver finishes (untrimmed proofs grow 4× on long runs). **`lrat-trim`** (`lrat::trim`,
  `lrat::write_text`) then trims by backward reachability through the hints, linear and without propagation:
  250 MB → 41.5 MB on the largest proofs, within 2 % of `drat-trim -L`'s output size, every trimmed proof verified.
- **LRAT binary decode on the device** (`warp-types-sat`, `feature = "gpu"`): `sat-kernels::lrat_scan_slices`
  decodes binary LRAT with one warp per 4 KB slice — token ends by `ballot`, the role machine by a lane-guarded
  composition scan over `shuffle_up`, output slots by popcount — in two passes around a host prefix sum; the host
  stitch (`stitch_plan`) then runs over step records only, `lrat_gather_lits` / `lrat_gather_hints` place the
  payloads on the device, and `check_lrat_codes_resident` checks them there. Slices the one-thread decoder would
  reject fall back to it on the host; `lrat_scan_slices_serial` (a thread per slice) is kept as the control. The
  `SimWarp` model (`lrat_warp::decode_model`) is held slice-for-slice equal to the host parser on every truncation
  and bit flip of a synthetic proof, long varints, straddling steps and RAT hints across windows, and the device
  path to the sequential decoder on the same battery and on all 632 archive files, both kernels. **Measured <!-- unguarded: 632 — archive count, research/gpu-lrat-decode.md -->
  2026-11-03** on the workstation RTX 4000 SFF Ada (pre-registered, `research/gpu-lrat-decode.md`; arrays verified
  equal to the host decode on both proofs): the 415 MB proof decodes in 0.33 s against 0.90 s on six host threads, <!-- unguarded: 415 — proof size in MB -->
  one cold run each (kernels 0.08 s; the rest is the upload, the step-record download and the host's walk over 4.8 M <!-- unguarded: 4.8 — step count in millions -->
  step records), and bytes-in-RAM to verdict falls from 1.19 s to 0.57 s; the 898 MB proof from 2.70 s to 1.3 s. <!-- unguarded: 898 — proof size in MB -->
  Phase 3a of the stitch runs on host threads (`plan_3a_parallel`) for any proof it has nothing to reject, the
  sequential walk deciding the rest; held equal to the sequential walk on the battery, nine rejection kinds and the
  archive. Corrected 2026-11-04: the first published figures (0.30 s decode, 0.53 s to verdict) were the best of
  three repetitions in one process, where the download reused memory the previous repetition freed. Of the five
  predictions one was met: on those first figures the decode missed its 0.25 s by 0.05 s and the pipeline its
  0.5 s by 0.03 s (cold, by more), the warp kernel is 1.3× the thread kernel
  where 2× was predicted (the GPU is ≈ 10× the host threads either way), and the PTX agreed with the model at the
  first run where a hardware-only defect was predicted. The check kernel, 0.23 s here, was then the largest device
  stage.
- **LRAT check kernel: packed table, direct layout** (`warp-types-sat`, `feature = "gpu"`): `lrat_check_lemmas`
  stores each assignment-table slot as one 64-bit word (one load per probe), indexes the table by literal code
  when every code of the proof fits it (`lrat_max_code` decides per launch; variables up to 2,047), fetches hint <!-- unguarded: 2,047 — largest variable for the direct layout -->
  metadata 32 hints at a time across the lanes, and runs four warps per block; `check_lrat_codes_mode` picks
  the layout explicitly, and a block count of 0 picks it per layout from the SM count. In the direct layout a
  code past the table fails its lemma closed. Per-lemma codes held equal to the SimWarp model in both layouts on
  the battery and all 632 archive files. **Measured 2026-11-04**, same card, one cold run each: check 0.23 → <!-- unguarded: 632 — archive count -->
  0.10 s on the 415 MB proof and 0.54 → 0.25 s on the 898 MB one, bytes-in-RAM to verdict 0.55 → 0.42 s and <!-- unguarded: 415, 898 — proof sizes in MB -->
  1.3 → 1.0 s. The direct layout is most of the gain; the hashed one, which proofs with more variables keep,
  measured 0.15 s on the 415 MB proof when forced. Pre-registered (`research/gpu-lrat-decode.md`, S5): two <!-- unguarded: 415 — proof size in MB -->
  predictions missed — more warps per block made the kernel slower, it was contending for memory, not short of
  latency hiding — and three met.
- **Fixed: the LRAT check kernel's table cap and block-count arithmetic** (found by the S5 cross-vendor panel,
  both present since the kernel was added): the kernel looked a key up before testing the 4,095-entry cap, so a <!-- unguarded: 4,095 — table cap -->
  lemma whose 4,096th insert repeated a key read `Ok` on the device and `TableOverflow` in the SimWarp model;
  the cap now comes first, as in the model. A caller-chosen block count of 262,144 or more wrapped the kernel's <!-- unguarded: 262,144 — first wrapping block count -->
  32-bit table-size arithmetic into out-of-bounds device writes; the launcher now clamps the block count and
  refuses proofs past the kernel's 32-bit indices.
- **LRAT binary decode on the device** (`warp-types-sat`, `feature = "gpu"`): `sat-kernels::lrat_scan_slices`
  decodes binary LRAT with one warp per 4 KB slice — token ends by `ballot`, the role machine by a lane-guarded
  composition scan over `shuffle_up`, output slots by popcount — in two passes around a host prefix sum; the host
  stitch (`stitch_plan`) then runs over step records only, `lrat_gather_lits` / `lrat_gather_hints` place the
  payloads on the device, and `check_lrat_codes_resident` checks them there. Slices the one-thread decoder would
  reject fall back to it on the host; `lrat_scan_slices_serial` (a thread per slice) is kept as the control. The
  `SimWarp` model (`lrat_warp::decode_model`) is held slice-for-slice equal to the host parser on every truncation
  and bit flip of a synthetic proof, long varints, straddling steps and RAT hints across windows, and the device
  path to the sequential decoder on the same battery and on all 632 archive files, both kernels. **Measured <!-- unguarded: 632 — archive count, research/gpu-lrat-decode.md -->
  2026-11-03** on the workstation RTX 4000 SFF Ada (pre-registered, `research/gpu-lrat-decode.md`; arrays verified
  equal to the host decode on both proofs): the 415 MB proof decodes in 0.33 s against 0.90 s on six host threads, <!-- unguarded: 415 — proof size in MB -->
  one cold run each (kernels 0.08 s; the rest is the upload, the step-record download and the host's walk over 4.8 M <!-- unguarded: 4.8 — step count in millions -->
  step records), and bytes-in-RAM to verdict falls from 1.19 s to 0.57 s; the 898 MB proof from 2.70 s to 1.3 s. <!-- unguarded: 898 — proof size in MB -->
  Phase 3a of the stitch runs on host threads (`plan_3a_parallel`) for any proof it has nothing to reject, the
  sequential walk deciding the rest; held equal to the sequential walk on the battery, nine rejection kinds and the
  archive. Corrected 2026-11-04: the first published figures (0.30 s decode, 0.53 s to verdict) were the best of
  three repetitions in one process, where the download reused memory the previous repetition freed. Of the five
  predictions one was met: on those first figures the decode missed its 0.25 s by 0.05 s and the pipeline its
  0.5 s by 0.03 s (cold, by more), the warp kernel is 1.3× the thread kernel
  where 2× was predicted (the GPU is ≈ 10× the host threads either way), and the PTX agreed with the model at the
  first run where a hardware-only defect was predicted. The check kernel, 0.23 s here, was then the largest device
  stage.
- **LRAT check kernel: packed table, direct layout** (`warp-types-sat`, `feature = "gpu"`): `lrat_check_lemmas`
  stores each assignment-table slot as one 64-bit word (one load per probe), indexes the table by literal code
  when every code of the proof fits it (`lrat_max_code` decides per launch; variables up to 2,047), fetches hint <!-- unguarded: 2,047 — largest variable for the direct layout -->
  metadata 32 hints at a time across the lanes, and runs four warps per block; `check_lrat_codes_mode` picks
  the layout explicitly, and a block count of 0 picks it per layout from the SM count. Both layouts keep the
  4,095-entry cap, so `TableOverflow` falls on the same lemmas. Per-lemma codes held equal to the SimWarp model
  in both layouts on the battery and all 632 archive files. **Measured 2026-11-04**, same card, one cold run <!-- unguarded: 632 — archive count -->
  each: check 0.23 → 0.09 s on the 415 MB proof and 0.54 → 0.23 s on the 898 MB one, bytes-in-RAM to verdict <!-- unguarded: 415, 898 — proof sizes in MB -->
  0.55 → 0.41 s and 1.3 → 1.0 s. The direct layout is most of the gain; the hashed one, which proofs with more
  variables keep, measured 0.14 s on the 415 MB proof when forced. Pre-registered (`research/gpu-lrat-decode.md`, <!-- unguarded: 415 — proof size in MB -->
  S5): two predictions missed — more warps per block made the kernel slower, it was contending for memory, not
  short of latency hiding — and three met.
- **Warp-typed LRAT proof checker** (`warp-types-sat`): `lrat` (text + binary LRAT; the reference checker follows
  `lrat-check`'s RUP rules — hints in order, deletions honoured, every step checked including those after the empty
  clause, RAT = `Unsupported` — and reports the earliest failing step), `lrat_warp` (one warp per lemma on the
  crate's `ballot` / `reduce_sum` / `broadcast_lane`, on `SimWarp`; `FlatProof::from_binary` and the parallel
  `from_binary_par` decode binary LRAT straight into the device layout, held byte-for-byte equal on every truncation
  and bit flip of a synthetic proof), the nvptx kernel `lrat_check_lemmas` in `sat-kernels` (a second implementation
  of the same algorithm), `GpuContext::check_lrat_codes` behind `feature = "gpu"`, `examples/lrat_check`, and
  `tools/lrat-mutate.py` for hint-level negative controls. **Measured 2026-11-02**, pre-registered in
  `research/gpu-lrat-checker.md`, cross-vendor reviewed (five seats, four families; the first public wording of this
  entry mislabelled the archive and spliced two machines in its ratios — corrected here, record in the research log):
  on 632 files (114 genuine proofs, 66 `drat-trim` outputs from corrupted DRATs, 452 hint-level mutants) the <!-- unguarded: 632, 114, 66, 452 — archive tallies, research/gpu-lrat-checker-2026-11-02/ -->
  reference returns `lrat-check`'s and `cake_lpr`'s verdict every time, and SimWarp, both decoders and the H200
  device path return the reference's; on a 941 MB binary proof, same pod, the device check is 0.27 s including
  transfer, the parallel decode 0.9 s (32 threads), end to end ≈ 1.6 s against 25 s for the CPU route on that pod.
  Same machine, same 415 MB proof: `lrat-check` 20.9 s, `cake_lpr` 21.5 s, device path 1.1 s (≈ 19×); on a <!-- unguarded: 415 — proof size in MB -->
  workstation RTX 4000 SFF Ada: 16.2 s, 17.4 s, 1.2 s (≈ 13×; `reproduce/local-lrat-chain.sh`). The whole
  producing chain — `solve --lrat-binary` → `lrat-trim` → device check — verified 54/54 corpus instances on one pod, <!-- unguarded: 54 — chain tally, research/lrat-from-solver-2026-11-03/chain/ -->
  with `lrat-check` and `cake_lpr` agreeing on every trimmed proof (`reproduce/runpod-lrat-chain.sh`).
  Limits: lemmas whose assignment exceeds 4,095 literals are `TableOverflow` on the warp paths; ids and variables
  above 2^26 are refused at parse time.
- **Fixes from the review of the above** (none shipped in a release): the kernel could store a 4,096th literal and
  then loop forever on a lookup (occupancy is now capped like the simulator's); the device code buffer was
  zero-initialised with 0 = Ok, so an unwritten lemma read as passed (now pre-filled `Unwritten`, and `combine`
  refuses a code vector of the wrong length); no warp barrier between a lemma's inserts and its reads (added);
  one huge id or literal in a malformed file was one huge allocation (bounded); `TableOverflow` was reported as
  `NoConflict` (own reason now). `Reason` gains `TableOverflow` and `Unchecked`; `flatten` returns `FlatProof`
  (its error type is gone — the step sequence's verdict rides along as `FlatProof::early`).
- **Fix: `WarpBuilder` passed `-C target-cpu` only to the kernel crate**, which current rustc rejects as an ABI
  mismatch against the build-std `core`; it now goes through `RUSTFLAGS` so the whole device graph agrees. The
  `gpu` feature had been building from a cached PTX, so this surfaced only when `sat-kernels` changed.
- **Doc: `broadcast_lane` and device panics.** A `src_lane` the compiler cannot bound keeps the bounds `assert!`
  alive and its `panic_fmt`/`Display` externs make the PTX module fail to load; the new kernel hit it (`& 31`).
- **Toolchain pin: `nightly-2026-09-15` → `nightly-2026-10-28`** (rustc 1.101.0-nightly), after the full CI matrix passed
  with the pin unedited; no code change.
- **`solve --certify`** (with `--drat`): runs `drat-trim` on the written proof before answering and prints
  `s UNSATISFIABLE` only on `s VERIFIED`; otherwise `s UNKNOWN`, reason on stderr, exit 2. Fail-closed — a missing
  checker, a checker that cannot run, or one that says `s NOT VERIFIED` all yield UNKNOWN (each case exercised).
  SAT answers are unaffected.
- **`warp-types-bmc` reads AIGER; `bmc`/`pdr` take files.** New `aiger` module (binary and ASCII, AIGER 1.9 bad /
  constraints / reset values) lowering an AIG into `TransitionSystem`; `bmc <file.aig>` and `pdr <file.aig>
  --invariant OUT` print `s` verdict lines, and `pdr` writes its inductive invariant over AIGER literals.
  `tools/check-invariant.py` certifies such an invariant with its own encoding and kissat (initiation,
  consecution, safety); `tools/bmc-pdr-campaign.py` runs both crates over HWMCC17 against an ABC reference.
  Pre-registration and results: `research/bmc-pdr-aiger.md`. **Measured 2026-11-01** on the 119 HWMCC17
  instances under 20 KB (60 s each; ABC 1.01 at its own 30 s caps as reference): BMC counterexample depths agree
  with `bmc3` on 21/21 both-found instances (21 of ABC's 26 found; 0.95–25× ABC's wall time where either side took ≥ 0.1 s, ABC re-timed on idle cores); PDR proves 20 — 13 of them
  by the property clause alone, 7 non-trivial fences — and **all 20 invariants pass the independent check, also
  with the property clause removed**; 4 counterexamples at ABC's depths; **every counterexample (21 BMC, 4 PDR) replays through the AIG under an
  independent simulator** (`--trace` on both binaries, `tools/replay-trace.py`); no verdict disagreement; ABC proves
  these in tens of milliseconds where we take up to 41 s; 91/95 of 119 time out at 60 s; a second run with the
  constant-true unit on both sides of T reproduced everything, and a third with **cone-of-influence reduction**
  (`aiger::lower_coi`, now the CLI default; `--no-coi` keeps the full AIG) reproduced every verdict and depth and
  moved two constant-false instances from timeout to bounded-safe — the corpus is already cone-reduced. A pre-registered comparison
  with ABC caught an AIGER constant-polarity bug before the campaign. Cross-vendor reviewed (five seats, four
  families); the panel's corrections are applied here and in the research log. Both crate READMEs carry the
  numbers and what the check does and does not establish.
- **UNSAT proofs certified through a verified-LRAT pipeline; the proof logger has falsifiers.** `solve_watched_drat_with(db,
  n, out, reduce_interval)` and `solve --reduce-interval N` (diagnostic; default 2000) so short solves exercise clause
  deletion; `solve_watched_with` is the no-proof twin for like-for-like timing. `proof.rs` states the RUP invariant
  (reductions only at level 0, every trail reason locked, LBD ≤ 2 never deleted, minimization depends on level-0
  reasons staying present) and carries a test-only `fault` module with seven corruptions. Campaign
  (`tools/cert-campaign.py`, record in `research/drat-certification.md`): 114 UNSAT proofs (54 distinct instances across three runs), six families, all
  verified by `drat-trim` (backward and forward), `lrat-check` and `cake_lpr`; 68/68 SAT models valid against file
  bytes; kissat agreed on 182/182 verdicts; logging overhead +10–17 % to `/dev/null`. Found and recorded: `drat-trim`
  accepts a proof without the final `0`, ignores phantom and unit deletions and checks only core lemmas in backward
  mode, so three of the seven corruptions pass it and are caught only by the strict in-crate checker; a forgotten
  reason lock would not be caught by small-instance tests (vacuous on a 21-instance corpus); `cake_lpr`'s verdict is
  a verdict on `drat-trim`'s LRAT. `solve --drat` now refuses to overwrite its own input.
- **`solve --drat <file>`.** The CLI can write a DRAT proof: `solve --drat proof.drat problem.cnf` runs
  `solve_watched_drat` and writes the proof to the file (flag position free; `--drat` without a path, a second
  `--drat`, an unknown `--flag`, or a missing CNF print usage and exit 1; an unwritable proof path exits 1 with the
  I/O error). Verified 2026-11-01 with `drat-trim` on PHP(6) and PHP(7) written by the binary: `s VERIFIED`. Without
  the flag the binary is unchanged.
- **Fix: the DIMACS parser dropped empty clauses.** `dimacs::parse_dimacs` added a clause on `0` only when it had
  literals, so a CNF containing a bare `0` line — the empty clause, UNSAT by definition — parsed as if the line were
  absent, and the solver answered `Sat` (`p cnf 2 2 / 1 2 0 / 0` → `Sat`). Empty clauses now reach the database and
  every entry point (`solve`, `solve_watched`, `hybrid_solve`) returns `Unsat` on them. With that change a `%` line
  now ends the input instead of being skipped as a comment: SATLIB `uf*.cnf` files close with `%` then a lone `0`,
  which would otherwise have become an empty clause and made every such file UNSAT. Found while writing the DRAT
  tests on 2026-11-01; two parser tests added.
- **DRAT proof output for `warp-types-sat`** (`src/proof.rs`; `solver::solve_watched_drat(db, num_vars, &mut impl Write)
  -> io::Result<SolveResult>`). The CDCL core now takes a `&mut dyn ProofLog` and reports every learned clause
  before it is added, every clause `reduce_learned` drops before compaction frees it, every theory lemma it adds,
  and the empty clause at each `Unsat` return; the ordinary entry points pass a no-op logger. `DratWriter` emits
  DRAT text against the input formula's 1-based variables. `proof::check_drat` is an in-crate forward RUP checker
  (unit propagation over the clause set, independent of the solver's watched BCP) used by the tests: PHP(3)–PHP(7)
  (PHP(7) crosses the deletion threshold, so its proof carries `d` lines), and the UNSAT instances among 40 random
  3-SAT seeds at n = 60, all verified; a further test runs `drat-trim` when it is on `PATH` or named by `DRAT_TRIM`
  and is skipped otherwise — run 2026-11-01 with drat-trim (marijnheule/drat-trim, HEAD): PHP(6) and PHP(7) both `s VERIFIED`
  (PHP(7): 3,778-line proof, 2,616 of 2,755 lemmas in core, 0 RAT lemmas, 0.065 s). Pure SAT only: a DPLL(T) proof contains theory lemmas that are not RUP and will not
  check. `SolveResult` now derives `Clone, PartialEq, Eq`. No change to the solve path or its answers.
- **Peer comparison re-measured** (`warp-types-sat/benches/README.md`): the April 2026 table (our solver 35–88× behind
  `batsat`, explained as "we do not implement watched literals") was measured with `solver::solve`, the basic path.
  With `solve_watched`, random 3-SAT at the phase transition (n = 50–100) runs ~1.7× slower than `batsat` 0.6 and
  ~2× faster than `splr` 0.17, flat over the range; log in `research/bench-random3sat-peers-2026-11-01.txt`. The
  crate README's "peer numbers pending" sentence now states the measured ratios. Also from the 2026-11-01
  cross-vendor panel: the pigeonhole table names its gradient configs (`enhanced()` + 8×100 vs library default);
  the correction paragraph gives the default/tuned conflict range (0.62×–1.43×) instead of "not CDCL conflicts";
  parity chains: k = 10 and 20 are gradient-phase solves, k = 40 still runs CDCL (the E1 example's `gradient_sat`
  flag was `conflicts == 0 && SAT`, now also requires zero decisions and propagations); `solve_watched`'s doc
  comment said Luby base 100, the core uses 32.
- **`solve` CLI (`warp-types-sat/src/bin/solve.rs`) now runs `solver::solve_watched`**, the watched-literal
  VSIDS core, instead of `solver::solve`, the basic reference solver it had called since it was written.
  Same input, same output format; answers unchanged where the old binary returned one. Measured 2026-11-01
  (release, one core): PHP(6) 173 ms → 6 ms, PHP(7) 2.79 s → 38 ms; planted random 3-SAT with 300 and 1,000
  variables, which the old binary did not finish in 120 s, now returns a model in under 25 ms (models checked
  clause-by-clause against the input). The core is the one `benches/` and `hybrid_solve` already used.
- **Correction — `warp-types-sat/README.md` "Performance expectations".** The pigeonhole table's "Pure CDCL"
  column was produced by `solver::solve`, the basic-BCP reference path, while the hybrid columns ran
  `hybrid_solve`, whose CDCL phase is the watched-literal core (`solve_cdcl_core`). The headline
  "65× on PHP(6)" was therefore measuring two BCP implementations against each other, not the gradient
  warm-start. Re-paired like-for-like against `solver::solve_watched`, the hybrid is *slower* than pure
  CDCL at PHP(4), PHP(5) and PHP(6), and its CDCL phase does about the same number of conflicts
  (E1 sweep, 2026-11-01: PHP(6) 579 vs 558; UNSAT random 3-SAT median 1.05×). The claim that fewer gradient
  iterations give better warm-starts on UNSAT was a separate artefact: both hybrids run the watched core and
  differ mainly in gradient wall time (conflicts: median ratio 0.98×, range 0.62–1.43). The claim entered in
  `471804155` (2026-04-18) and is carried by tags v0.3.2–v0.4.3; it reaches archived copies only through the
  next release. The
  table is replaced with the like-for-like numbers; `benches/cardinality.rs`, `benches/random_3sat.rs` and
  `benches/satlib_sweep.rs` now call `solve_watched` for their CDCL column and say so. Found while running the pre-registered E1
  experiment (`research/gpu-sat-partition.md`); the typestate claims of the parent crate are not affected.
  Cross-vendor reviewed 2026-11-01 by three Cursor seats (GPT-5.6, Grok 4.7, Gemini 3.7); the two refuting seats'
  verified findings are applied in the follow-up commit (see the research log, 2026-11-01 panel entry).

## [0.4.3] — 2026-10-18

- Release corresponding to tag `v0.4.3`, Zenodo DOI 10.5281/zenodo.23457876 (concept DOI 10.5281/zenodo.19040615); crates.io `warp-types 0.4.3` (`warp-types-builder` stays 0.3.3).
- **`DynWarp` scalar collectives do the work (nvptx64, 32-lane).** `shuffle_xor_scalar`, `shuffle_down_scalar`,
  `reduce_sum_scalar` and `broadcast_scalar` performed the all-active check and then returned their input
  (`reduce_sum_scalar`: `value × width`) — a prototype that was a silent wrong result on the device, behind an `Ok`.
  They now issue the full-warp `shfl.sync` after the check (CPU: identity, as for `Warp<All>`; the butterfly uses
  `reduce_add`'s stage order; `broadcast_scalar` now specifies lane 0). The check also requires the handle's width
  to be this build's warp width, as `ascribe` does — a `DynWarp::all_64()` on a 32-lane build is refused instead of
  running a sixth butterfly stage. `ballot(&[bool; 32])` remains a CPU model (packs the supplied array; no device
  vote) and says so. `examples/gpu-project` Tests 7 and 8 run the success and error paths on hardware. (0.4.1 panel,
  Opus "not asked" #2; 0.4.3 panel Q1–Q3.)
- **`examples/*.rs` say what they are; the real types get the same four bugs as doctests.** The eight examples are
  self-contained re-implementations with their own toy `Warp` (the 0.4.1 research review's R3); their headers now say
  so, and `src/lib.rs` ("Real bugs against the real types") checks cuda-samples#398, PIConGPU#2514, CUB/CCCL#854 and
  LLVM#155682 against the shipped `Warp<S>` as `compile_fail,E0599` doctests, with two passing shapes marked as
  illustrative where they are not the upstream fix. Doc tests 40 → 46 (22 compile-fail).

## [0.4.2] — 2026-10-18

- Release corresponding to tag `v0.4.2`, Zenodo DOI 10.5281/zenodo.23457435 (concept DOI 10.5281/zenodo.19040615); crates.io `warp-types 0.4.2`, `warp-types-builder 0.3.3`.
- warp-types-builder 0.3.3: `build.rs` support emits `cargo:rerun-if-changed` for the kernel crate's path dependencies
  (a change under `warp-types/src` now rebuilds the PTX; a stale `kernels.ptx` was run twice on 2026-10-18 before this).
- **Warp collectives are `convergent`.** The nvptx primitives (`shfl.sync.{bfly,down,up,idx}`, `vote.sync.ballot`,
  `bar.warp.sync`) were `asm!` blocks, which LLVM treats as opaque side-effecting calls that are *not* `convergent`:
  a transformation that adds a control dependency (jump threading, tail duplication, loop unswitching) may legally
  duplicate such a call into divergent blocks — the hazard this crate exists to rule out. They now call the
  `llvm.nvvm.*` intrinsics by `#[link_name]`, which LLVM declares `convergent nocallback nounwind
  memory(inaccessiblemem: readwrite)` (`bar.warp.sync`: `convergent nocallback nounwind`); the NVPTX backend emits
  the same instructions with constant operands folded. The attribute is on the intrinsic call; wrappers are
  `#[inline(always)]`/`#[inline]` and the guarantee is checked on the end state: `scripts/check-ptx-operands.sh`
  reads the LLVM IR and fails if any collective is inline asm, if any of the six intrinsics is not declared
  `convergent`, or if an `.extern .func` survives (release profile; a dev-profile device build is not link-complete
  by construction — shared generic instantiations and live `core::panicking` — and the builder's debug option now says so). On sm_70+ the ISA's rendezvous
  semantics make many such duplications benign; this is defence in depth against a legal transformation, not a
  repaired miscompile. The `.reg .pred`/`setp` ballot workaround is gone (`bool` lowers to `i1` on the pinned
  nightly — a toolchain property, which the check notices if it changes). Device build gains
  `#![feature(link_llvm_intrinsics)]` (compiler-internal; nightly was already required). Raised by the 2026-10-18
  0.4.1 panel (Opus seat, "not asked"); the 0.4.2 panel then found the rest of this entry.
- **Every non-generic public function is `#[inline]`.** A non-generic device-reachable function that rustc does not
  export as cross-crate-inlinable becomes an `.extern .func` in the kernel PTX and fails PTX JIT with
  `CUDA_ERROR_INVALID_PTX` — `BallotResult::first_lane()` in any kernel did this in 0.4.1. 164 functions across the
  crate now carry `#[inline]`; the check script's probe exercises the API surface and fails on any `.extern .func`.
- **`#[warp_kernel]` rejects fat pointers hidden behind type aliases.** The thin-pointee check is syntactic and saw
  `type S = [i32]; p: *mut S` as a path type (panel, GPT seat). The macro now also emits
  `const _: () = assert!(size_of::<*mut S>() == size_of::<usize>())` per pointer parameter, so the alias case fails to
  compile with a named message; a device-side negative probe guards it.

## [0.4.1] — 2026-10-18

- Release corresponding to tag `v0.4.1`, Zenodo DOI 10.5281/zenodo.23456256 (concept DOI 10.5281/zenodo.19040615); crates.io `warp-types 0.4.1`, `warp-types-builder 0.3.2`.
- **Soundness: `shuffle_xor_within` now guards its own execution.** `diverge_*` emits no branch, so after
  `diverge_even_odd()` every lane still runs the code that follows. The method issued `shfl.sync` with membermask
  `S::MASK` on all 32 lanes, so lanes outside `S` executed an instruction whose membermask did not name them, which
  the PTX ISA leaves undefined; the method's own doc example did this. Lanes outside `S` now skip the instruction and
  return their input. `Warp<Empty>::shuffle_xor_within` fails at codegen (post-monomorphization `E0080`; not under
  `cargo check`), on the host (`compile_fail,E0080` doctest) and on nvptx64 (`scripts/check-ptx-operands.sh`
  `empty-probe`). Found by the 2026-10-18 unsafe and boundary audits; the fixed semantics run on an H200 SXM and an RTX 4000 Ada
  (`examples/gpu-project` Test 6; compute-sanitizer memcheck + synccheck 0 errors on the H200 — `results/h200-2026-10-18-main.txt`, `results/rtx4000ada-2026-10-18.txt`;
  a value test cannot witness the old UB).
  Reviewed by a five-seat cross-vendor panel before release (`.review/panel-2026-10-18-041`, not committed).
- **warp-types-builder 0.3.2: `find_ptx_file` searches the new asm output layout.** nightly-2026-09-15 writes
  `--emit=asm` output under `build/<crate>/<hash>/out/`, not `deps/`; the builder kept picking a stale `deps/` file,
  so the worked example silently ran kernels that were not in its source. Exact-name candidates now compete on
  mtime too (they were returned first), and a sibling crate sharing the name prefix is no longer matched.
  Regression tests added. mtime remains a heuristic.
- `scripts/check-ptx-operands.sh` + `tools/ptxprobe`: the register in each `shfl.sync`'s membermask operand is
  loaded with the promised constant, the lane guard is present, the width operands are right, and the `Empty`
  instantiation fails the device build (reads PTX rustc writes to a named path; nvptx CI job + pre-push hook; no
  GPU needed). Pattern and one-hop dataflow, not a dominance proof.
- **Lean: the reduction rules now carry the safety property.** `Step.shuffleVal` (and `fenceVal`/`finalizeVal` in the
  core metatheory) accepted any active set, so an ill-typed shuffle on a diverged warp *stepped* and progress/
  preservation said nothing about shuffles. They now fire only on the full set; `shuffle_diverged_stuck`,
  `fence_partial_stuck` and `finalize_partial_stuck` prove the partial cases have no successor (33 named theorems).
- Paper/README/lean README: the calculus is linear for let-bound warps only (`fst`/`snd` and `warpVal` literals are
  unrestricted); "zero axioms" → Lean's three standard axioms; the WebGPU claim corrected (WGSL defines
  `subgroupShuffle` with an indeterminate value for inactive sources, and mandates a uniformity analysis); the
  lean/README priority claim removed; CCCL#854 stated as suspected throughout; §5.1/§5.2/§5.6 listings marked as
  calculus syntax and design sketches, §5.7 table downgraded accordingly; §6.7 shows the real rustc diagnostic;
  `examples/*.rs` described as self-contained re-implementations; new related-work paragraph (divergence analysis,
  Sierra, typestate, dynamic MPST, Faial) with references [48]–[59].
- Docs: "Trusted entry points" registry on `Warp::kernel_entry` (named from every SAFETY comment; `DynWarp::all()`
  and `from_mask()` added to the list); tile docs no longer say safety "comes from the partition structure"; the
  false claim that the Lean model enforces one-shot entry removed. README Limitations and paper §6.9 updated.
- Tests: `shuffle_down`, the `*_raw` variants, `Tile::shuffle_down` and its bounds panics, `broadcast_lane`'s bounds
  panic, `reduce_sum_wrapping_{i64,u64}`, `exclusive_sum`, `Warp<Odd>::shuffle_xor_within` on the CPU path.

## [0.4.0] — 2026-10-18

- Release corresponding to tag `v0.4.0`, Zenodo DOI 10.5281/zenodo.23451884.

- **Breaking:** `GpuShuffle`'s methods (`gpu_shfl_xor`, `gpu_shfl_xor_masked`, `gpu_shfl_down`, `gpu_shfl_up`,
  `gpu_shfl_idx`, `*_width`) are now `unsafe fn` with a `# Safety` contract. They were safe methods on plain values,
  re-exported in the prelude, so safe code could issue a full-warp `shfl.sync` under divergence (UB) without any
  `Warp` handle; the impl-level SAFETY note claiming they were "only reachable through the typed wrappers" was false.
  The typed `Warp<All>` / `Tile` methods are unchanged. A compile-fail doctest guards the change.
- Paper, README, crate docs, CITATION.cff, .zenodo.json: priority claims removed (Descend, SPMD barrier inference,
  Titanium, Lucid Synchrone precede them); "cannot shuffle" qualified to full-warp shuffle; new paper §8.6; trusted
  entry points listed in §6.9 and README Limitations (`kernel_entry`, `DynWarp` ascription, merge without
  provenance). Reviewed cross-vendor over four rounds.
- Workspace crates `warp-types-sat`, `-bmc`, `-smt`, `-pdr` now require `warp-types` 0.4.0 in this repository; their
  crates.io releases (built against 0.3.2) are unchanged and not republished — none of them calls `GpuShuffle` directly.

## [0.3.2] — 2026-04-18

*(Entry added retroactively 2026-08-13 — the tag/Cargo/CITATION/Zenodo release
shipped without a CHANGELOG entry.)*

- Release corresponding to tag `v0.3.2` and Zenodo DOI 10.5281/zenodo.19040615.
- **Erratum (deposit metadata):** the v0.3.2 Zenodo deposit's notes say "31
  theorems"; the tagged tree contains 32 (`grep -c '^theorem'` over
  Basic.lean + Metatheory.lean), as the tagged README correctly stated. The
  deposit's test counts (317/50/28) describe the tagged tree and are correct
  for the deposit; `.zenodo.json` at HEAD now carries current counts for the
  next deposit.

## [0.3.1] — 2026-03-29

### Fixed
- **`WarpBuilder` target-cpu** — now sets `-C target-cpu=sm_70` by default. Previously defaulted to `sm_30` (PTX 3.2), which lacks `shfl.sync` (requires PTX 6.0). This caused `CUDA_ERROR_INVALID_PTX` at runtime for anyone using the gpu-project example. Users can override with `.sm_arch("sm_90")` for Hopper-specific codegen.
- **Ballot `setp`/`selp` workaround** — `ballot_sync` now compiles for nvptx64 by declaring `.reg .pred` inside the asm block and converting to/from `u32`, sidestepping Rust's missing `pred` register class. PTX-verified, not yet exercised in a GPU kernel.
- **`warp_kernel` macro hardening** — rejects qualified type paths, non-void return types, and generic parameters with clear error messages instead of generating broken PTX
- **`DynWarp::reduce_sum_scalar`** — changed `wrapping_mul` to non-wrapping `*` (matches `reduce_sum` overflow semantics)
- **Research `coalescing.rs`** — fixed hardcoded 32-lane constant to use `WARP_SIZE`
- `reproduce/runpod-h200.sh` — installs `rust-src` for both pinned and generic nightly toolchains

### Changed
- GPU kernels verified on H200 SXM (compute 9.0, Hopper) in addition to RTX 4000 Ada (compute 8.9): 4 typed kernels PASS, shuffle semantics confirmed, zero-overhead PTX on sm_90
- Paper and docs updated with H200 verification results
- Lean: closed `DivTree.leaves_cover_root` — zero sorries in SolExperiment

## [0.3.0] — 2026-03-21

### Added
- `research` feature flag — gates experimental `research/` module (24 modules, 12K lines of design-space prototypes). Always compiled during `cargo test` but excluded from default `cargo doc` and downstream builds
- Prelude: added `LaneId`, `WarpId`, `warp_kernel` to `prelude` module
- Lean 4: all four §5.1 loop typing rules mechanized (LOOP-UNIFORM, LOOP-CONVERGENT, LOOP-VARYING, LOOP-PHASED) with full progress, preservation, and substitution coverage
- Lean 4: nested merge mechanized — `IsComplement s1 s2 parent` (generalized from `IsComplementAll`)
- Lean 4: `letPair` linear pair destructor with full metatheory
- C++20 interop header (`include/warp_types.h`) — concepts, requires clauses, CUDA/HIP/host-only modes
- CMake example project for C++ host + Rust PTX workflow

### Fixed
- **warp64 completion**: `cub::reduce`, `bitonic_sort` (3 variants), and `shuffle::Permutation` algebra all now handle 64-lane wavefronts correctly — previously hardcoded 32-lane constants produced silently wrong results under `--features warp64`
- Permutation masks use `WARP_SIZE - 1` instead of `0x1F`; rotate ops use `WARP_SIZE` instead of `32`
- `FullButterfly` type alias includes `ButterflyStage5 = Xor<32>` under warp64
- `shuffle_by` accepts `[T; 64]` under warp64
- Aliasing UB in research `coalescing.rs` (`WarpPtrMut` Clone, `store` taking `&` instead of `&mut`)
- FFI 64-lane: `warp_types.h` `ComplementOf` concept generalized via `ComplementWithin`
- `GpuWarp32::shuffle` wraps mod 32 (hardware behavior), not clamp
- Stale doc counts, version references, and terminology across README, paper, blog, tutorial
- `DynWarp` now `#[must_use]` — matches `Warp<S>` and `DynDiverge` (warns on accidental drop)
- `SharedRegion`/`WorkQueue` use `WARP_SIZE` instead of hardcoded `[T; 32]` (warp64 compatible)
- `proof.rs` `type_safety_check` returns false on step limit (was incorrectly returning true)
- `shuffle_xor_within` panic message shows full u64 mask on warp64 (was truncating via `as u32`)
- `platform.rs` stale comment corrected (CpuSimd clamps, not wraps)
- Blog AMD claim: "Real GPU execution" → "mask-correctness verified via HIP"

### CI
- Clippy job now runs `--all-features` — lints 12K lines of feature-gated code previously unchecked

### Changed
- API encapsulation: `Role`, `BlockId`, `ThreadId` fields now private; `DynWarp` no longer derives `Clone`
- `GpuValue` sealed separately from `ActiveSet` (distinct sealing concerns)
- `proof` module gated behind `cfg(any(test, feature = "formal-proof"))` (was always-compiled)
- Paper terminology: remaining "session types" → "linear typestate" in §5.3, §10

## [0.2.0] — 2026-03-18

### Added
- **SimWarp**: Multi-lane warp simulator with real shuffle semantics — butterfly reduce, bitonic sort, tile-confined shuffle, Hillis-Steele scan, 64-lane AMD reduce all verified with actual lane exchange

### Fixed
- `SimWarp::shuffle_idx` OOB behavior: now wraps modularly via `% WIDTH` (matching GPU `shfl.sync.idx` semantics); previously returned lane 0's value
- Bitonic sort `stage_mask` must be constant per stage (`2^k`), not `2 * xor_mask`
- 64-lane AMD path: 3 bugs in wide-warp shuffle/reduce
- Hard-seal `Sealed` trait: `_sealed()` now requires `SealToken` (pub(crate)), making external impls impossible (previously could be bypassed via default method body)
- `ValidTileSize` sealed via `Sealed` supertrait — external impls of invalid tile sizes blocked
- `DynWarp::from_mask` width inference: correctly auto-detects 32/64-lane from mask value
- `PortableVector::extract/insert`: `debug_assert` + direct index (no `% WIDTH` wrapping that silently masked OOB)
- `CpuSimd<0>`: const assert prevents empty-iterator panics in `reduce_sum/max/min`
- `butterfly_reduce_sum`: WIDTH-generic with const power-of-2 assertion (no longer hardcodes `[16,8,4,2,1]`)
- `DynWarp::reduce_sum_scalar`: uses `count_ones()` not hardcoded `* 32`
- `DynWarp::ballot`: rejects 64-lane warps (u32 return incompatible with >32 lanes)
- Paper: corrected LLVM IR example (butterfly is `shl`, not identity)
- 100+ documentation, terminology, and test count corrections across 23 cold review rounds

### Changed
- Terminology: "session types" → "linear typestate" throughout — the mechanism is typestate over a Boolean lattice, not session types proper
- Paper condensed from ~25 to ~19 pages — peripheral sections trimmed, core type system (§3) preserved at full size
- Removed Sol references — standalone warp-types project
- GPU kernels re-verified on RTX 4000 SFF Ada (compute 8.9, 2026-03-17)

### Documentation
- Tutorial updated for crates.io dependency (not path dep)

## [0.1.0] — 2026-03-18

Initial release on crates.io.

- Linear typestate for GPU warp divergence safety
- Zero-overhead implementation via Rust phantom types
- `Warp<S>`, `ComplementOf`, `merge`, `diverge` — core type system
- `DynWarp` gradual typing bridge (runtime → compile-time promotion)
- Platform abstraction (`CpuSimd`, `GpuWarp32`)
- Fence-divergence type-state machine
- Proc macros: `warp_sets!`, `#[warp_kernel]`
- Build-time GPU compilation: `WarpBuilder`
- 369 tests (291 unit + 50 example + 28 doc)
- Lean 4 mechanization: progress, preservation, substitution lemma (zero sorry, zero axioms)
- 5 bug untypability proofs (cuda-samples#398, CCCL#854, PIConGPU#2514, LLVM#155682, shuffle-after-diverge)
