# warp-types-sat

A phase-typed CDCL SAT solver in Rust. Built as the SAT core for the
`warp-types-*` verification stack (SMT, BMC, PDR) and as a home for
gradient-based incomplete solving on cardinality-heavy instances.

## What to use this for

**Embedded SMT/BMC/PDR in Rust applications.** The `warp-types-smt`,
`warp-types-bmc`, and `warp-types-pdr` crates stack cleanly on top:
DPLL(T) theory integration via `TheorySolver` (`theory.rs`), incremental
unroll-to-depth BMC, and IC3 with frames and cubes. Everything runs
in-process — no FFI, no subprocess, no serialization at layer boundaries.
If you're building a verification tool in Rust, the end-to-end pipeline
avoids the Z3/CVC5 bindings tax even if the underlying SAT is slower.

**The gradient solver (research path — see *Performance expectations* before relying on it).** `src/gradient.rs`
follows the FastFourierSAT line (Kyrillidis et al.) — gradient-descent
reformulation with clause-weight EMA on violation frequency. `src/gpu_gradient.rs`
puts per-lane loss + butterfly-reduce + ballot on GPU via `warp-types`.
This is where the parent `warp-types` library does load-bearing work: the
reductions must be warp-uniform and the ballot semantics depend on
lane-safety typing. CPU-path `SimWarp` testing validates the algorithm
before real-hardware execution.

**Reading and learning CDCL with legible types.** Watched literals, 1-UIP
with minimization, VSIDS with phase saving, Luby restarts, LBD-based clause
deletion, DRAT proof output — a clean first-pass CDCL with the phases surfaced
in the type system for auditability. Used as the reference SAT core in Sol (typed
verification front-end).

## What this is NOT trying to be

**Not a general-purpose CDCL competing with Kissat or CaDiCaL.** Those
represent a decade-plus of Armin Biere's surgical tuning — cache-sized
clause headers, multi-tier LBD clause DB, vivification, BVE, failed-literal
probing, chronological backtracking, interleaved inprocessing. We implement
none of those, deliberately. Expect performance roughly in the
`batsat`/`splr` range on random SAT benchmarks — meaningfully slower than
Kissat, and we don't plan to close that gap by grinding on BCP inner loops.
If you need competition-class raw throughput, link against Kissat via FFI;
that's the right tool.

**Not a parallel solver.** `ClauseToken` in `src/clause.rs` is an affine
ownership token designed for a future parallel BCP path. At v0.3.x it is
unused in the main `solve_cdcl_core_inner` loop and appears only in `bcp.rs`
tests and one non-default BCP path. Treat it as reserved infrastructure —
when a parallel BCP path lands, the token will make concurrent
re-acquisition of the same clause a type error. Today it is a runtime
bitset check inside `ClausePool::acquire`.

## The typestate, precisely

The `SolverSession<'s, P>` pattern encodes CDCL phases (`Decide → Propagate
→ Analyze → Backtrack`) as zero-sized types with invariant branded lifetimes
(`with_session(f: for<'s> FnOnce(...))`, `fn(&'s ()) -> &'s ()`). For code
that routes state changes through the session, the phase ordering is
compile-time enforced: you cannot consume a `Propagate` session twice, and
you cannot fabricate a different phase without going back through the
declared transition methods. `solve_cdcl_core_inner` routes everything
through the session, and the control flow is easier to audit as a result.

This is *legible control flow*, not a type-system fortress. A few honest
qualifications:

- `analyze::analyze_conflict` and `trail.backtrack_to` are free functions
  that take no phase proof — callers can invoke them outside the session
  if they choose. Only `bcp::run_bcp` requires a phase witness, and it's
  passed as `_phase_proof: &SolverSession<'_, Propagate>` — a compile-time
  gate, not a runtime check.
- The `'s` brand lifetime prevents nested `with_session` calls from
  unifying their phase proofs. It does not bind the session to the solver
  state; a stronger design would carry `&'s mut SolverState` so BCP
  couldn't accept another solver's phase proof. That's v0.4+ work.

The right mental model is `MutexGuard`: it makes the common safe path
easier to write and audit, and misuse requires deliberately bypassing the
session API. We think that is a useful ergonomic property — especially for
embedding in Rust verification tooling where the call graph is under your
control — but it is narrower than "invalid phase transitions are compile
errors."

## Performance expectations

**CDCL baseline is batsat/splr-class, not Kissat-class.** Measured
2026-11-01 on random 3-SAT at the phase transition (n = 50–100):
`solve_watched` is ~1.7× slower than `batsat` 0.6 and ~2× faster than
`splr` 0.17, flat over that range (`benches/README.md`, criterion log in
`research/bench-random3sat-peers-2026-11-01.txt`).

**Hybrid gradient + CDCL on pigeonhole (UNSAT), like-for-like.** `cargo bench
-p warp-types-sat --bench cardinality -- pigeonhole`, criterion 20 samples ×
2 s, release, 2026-11-01. Every column parses the same DIMACS; the CDCL
column is `solver::solve_watched`, the same watched-literal core that
`hybrid_solve` runs after its gradient phase.

| n | Pure CDCL (`solve_watched`) | gradient only (`enhanced()`, `num_starts=8, max_iters=100`; bounded budget, gives up) | `hybrid_solve` (tuned: `enhanced()`, `num_starts=8, max_iters=100`) | `hybrid_solve` (`GradientConfig::default()`: 32 × 1000, no momentum, no clause weights) |
|---|------------------------------|-------------------------------------------|------------------------------------------------------|----------------------------------|
| 4 | 28.2 µs | 534 µs | 564 µs (0.05×) | 17.8 ms (0.002×) |
| 5 | 206 µs  | 947 µs | 1.16 ms (0.18×) | 31.6 ms (0.007×) |
| 6 | 1.77 ms | 1.54 ms | 3.38 ms (0.52×) | 52.5 ms (0.03×) |

The hybrid is slower than pure CDCL at every size, by roughly the cost of
its gradient phase. The warm-start has no consistent effect on the CDCL work either. Conflicts
in the CDCL phase, hybrid vs pure: 25 vs 35 on PHP(4), 111 vs 117 on PHP(5),
579 vs 558 on PHP(6), 4,341 vs 2,754 on PHP(7); on the 29 UNSAT random
3-SAT instances at the phase transition (n = 100–200) the median ratio is
1.05× and the hybrid does fewer conflicts on 31 % of them. Default- and
tuned-config hybrids differ in gradient cost far more than in conflicts
(median ratio 0.98×, range 0.62–1.43; PHP(7) 2,710 vs 4,341), so the gap
between those two columns is mostly gradient wall time. All CDCL and hybrid
runs return `Unsat`; the gradient-only column returns no answer on its
bounded budget (it cannot prove UNSAT). Machine: 16-core desktop, run pinned
to 6 cores under `nice`; criterion output in
`research/bench-pigeonhole-2026-11-01.txt`, conflict data in
`research/e1-cubes-2026-11-01.csv`, both at this repository's root.

**Correction (2026-11-01).** From April to October 2026 this table read
"Pure CDCL 336 ms → `hybrid_solve` 5.18 ms (**65×**) at PHP(6)", with a
paragraph inferring that fewer gradient iterations give better UNSAT
warm-starts. The "Pure CDCL" column was produced by `solver::solve`, the basic
reference solver (no watched literals, no VSIDS, no restarts), while the
hybrid columns ran the watched-literal VSIDS core — the 65× was the gap
between two solvers, not the gradient. The warm-start inference was a
separate artefact: both hybrids use the watched core, and their timing gap
was mostly gradient wall time; across the 33 UNSAT instances their CDCL
conflicts had median ratio 0.98× but ranged from 0.62× to 1.43×. The claim entered
in commit `471804155` (2026-04-18) and is carried by tags v0.3.2 through
v0.4.3 and their archived copies, which this edit cannot change; it reaches
them only through the next release. Found while running a pre-registered
experiment (`research/gpu-sat-partition.md`, E1); `benches/cardinality.rs`,
`benches/random_3sat.rs` and `benches/satlib_sweep.rs` now call
`solve_watched` for their CDCL column, and the `solve` CLI
(`src/bin/solve.rs`) now runs `solve_watched` too (it had called
`solver::solve` since it was written).

In the three families measured so far (pigeonhole n = 4–7, random 3-SAT
n = 100–200, parity chains) the gradient warm-start shows no consistent
reduction in CDCL work. What the gradient path is for, in this crate: it is the GPU-executed
kernel that exercises `warp-types` (`src/gpu_gradient.rs`), and an
incomplete solver that finds models on easy SAT instances (parity chains
k = 10 and k = 20 are solved in the gradient phase with no CDCL decision;
at k = 40 CDCL still runs: 1 decision, 40 propagations, 0 conflicts). Treat it as a
research path, and measure on your own workload before relying on it.

The peer comparison above (`batsat`/`splr`) is a three-point curve on one
instance family; read the CDCL column in the pigeonhole table as a
within-solver reference, not a cross-solver claim.

## Usage

```bash
cargo run --bin solve -- problem.cnf
cargo run --bin solve -- --drat proof.drat problem.cnf   # also write a DRAT proof
cargo run --bin solve -- --drat proof.drat --certify problem.cnf   # check it before answering
```

Reads DIMACS CNF, prints `s SATISFIABLE` or `s UNSATISFIABLE` with the
variable assignment in standard SAT competition format. Runs
`solver::solve_watched`, the watched-literal VSIDS core (since 2026-11-01;
earlier builds ran the basic reference solver `solver::solve`). With
`--drat FILE` the run goes through `solve_watched_drat` and writes a DRAT
proof of an UNSATISFIABLE answer to FILE; check it with
`drat-trim problem.cnf proof.drat`. On SATISFIABLE the file is written but
proves nothing. `--certify` runs `drat-trim` (from `$DRAT_TRIM` or `PATH`) on
the proof itself and prints `s UNSATISFIABLE` only if it says `s VERIFIED`;
otherwise `s UNKNOWN`, the reason on stderr, exit status 2. It is
fail-closed: no checker is not a pass.

## What's inside

- **Phase types** (`phase.rs`) — zero-sized marker types for CDCL phases,
  with sealed `CanTransition` trait
- **Phase-typed session** (`session.rs`) — `SolverSession<'s, P>` with
  invariant branded lifetimes
- **Affine clause tokens** (`clause.rs`) — non-`Copy`, non-`Clone`
  ownership tokens. Reserved for a future parallel BCP path; not yet used
  in the main solver
- **Tile-local BCP** (`bcp.rs`, `clause_tile.rs`) — watched-literal
  propagation, plus a ballot-based variant that maps to GPU warp operations
- **1-UIP conflict analysis** (`analyze.rs`) — implication graph traversal,
  learned clause derivation with LBD scoring and minimization
- **LBD clause deletion** (`solver.rs`) — learned clauses scored by LBD and
  aged out. Single-pool deletion, not multi-tier
- **Trail** (`trail.rs`) — assignment stack with decision levels and reasons
- **Theory solver** (`theory.rs`) — `TheorySolver` trait for DPLL(T):
  `check`, `backtrack`, `explain`
- **DIMACS parser** (`dimacs.rs`) — full CNF format support
- **Solver** (`solver.rs`) — top-level CDCL loop with `solve_with_theory()`
  for SMT integration
- **Gradient solver** (`gradient.rs`, `gpu_gradient.rs`) — gradient-descent
  reformulation; GPU path via `warp-types` behind `feature = "gpu"`
- **DRAT proof output** (`proof.rs`) — `solver::solve_watched_drat(db, n, &mut out)`
  and `solve --drat FILE` write every learned clause, every deletion and the
  final empty clause in DRAT text. Pure SAT only: theory lemmas are logged
  but are not RUP, so a DPLL(T) proof will not check. **Checked, 2026-11-01:**
  114 UNSAT proofs (54 distinct instances across three runs) from six families (random 3-SAT n = 150–250, PHP(6–8), <!-- unguarded: 114 — campaign tally, research/cert-campaign-2026-11-01/ -->
  parity, Tseitin, planted, cnfuzz), at the default reduction interval and at
  50, all verified by `drat-trim` (backward and forward), and their LRAT by
  `lrat-check` and by the formally verified `cake_lpr`; 68/68 SAT models
  checked against the original file bytes; no single-literal deletion in
  ~3.9 M deletion lines. A certificate says *this* UNSAT answer is correct;
  it does not say the solver is sound in general, and `drat-trim` does not
  check that deletions were logged faithfully (it ignores unit deletions and
  trims non-core lemmas) — the in-crate strict checker does, and the fault
  tests in `proof.rs` show which checker sees which corruption. Full record:
  `research/drat-certification.md`; runner: `tools/cert-campaign.py`
- **LRAT straight from the solver** (`lrat_emit.rs`, `proof::LratWriter`) —
  `solver::solve_watched_lrat_with(db, n, &mut out, interval, binary)` and
  `solve --lrat FILE [--lrat-binary]` write clause ids and the RUP chain of
  every learned clause as the solver runs, so the proof checks with
  `lrat-check`, `cake_lpr` or the checker below with no `drat-trim` pass;
  `--certify` with `--lrat` runs `lrat-check`. Each chain is derived from the
  final learned clause by walking the implication graph and emitting the
  reasons in trail order (`research/lrat-from-solver.md`). **Measured
  2026-11-03** on the certification corpus: all 114 UNSAT answers' proofs (54 distinct instances; the corpus's three runs share instances) <!-- unguarded: 114 — campaign tally, research/lrat-from-solver-2026-11-03/ -->
  verified by `lrat-check`, `cake_lpr` and this crate's checker. Costs, stated
  because the pre-registration predicted better: logging overhead about 3× the
  DRAT logger's (+14 % text, +10 % binary, vs +4 % for DRAT, of solve time on
  random 3-SAT n = 250, one core; was +20 % before the id table became a
  vector and the number formatting stopped allocating); proof <!-- unguarded: 250 — instance size n -->
  bytes about equal to `drat-trim -L`'s output on short runs and 4× on long
  ones (nothing is trimmed); end to end against the DRAT + `drat-trim` route
  2.8× faster at the median and 1.2× on the largest instances this solver
  finishes. Pure SAT: a theory lemma is written without a chain and counted
  (`LratStats::unjustified`); such a proof is rejected, by design.
  **`lrat-trim IN OUT [--binary]`** (`lrat::trim`) keeps only what the empty
  clause depends on — linear, no propagation, ids unchanged: on the n = 250 <!-- unguarded: 250 — instance size n -->
  proofs 250 MB → 41.5 MB, the size of `drat-trim -L`'s output (42.3 MB), in
  2.7 s; all 114 trimmed proofs verify. <!-- unguarded: 114 — campaign tally -->
- **LRAT proof checker, warp-typed** (`lrat.rs`, `lrat_warp.rs`, `sat-kernels::lrat_check_lemmas`,
  `GpuContext::check_lrat_codes` under `feature = "gpu"`) — reads text and
  binary LRAT. The reference checker follows `lrat-check`'s rules for RUP
  proofs (hints in order, deletions honoured, every step checked, also after
  the empty clause; a satisfied hint is a no-op; RAT steps are `Unsupported`,
  never verified) and reports the earliest failing step. The warp
  formulation checks one lemma per warp with `ballot` / `reduce_sum` /
  `broadcast_lane`; it is written twice, once on `SimWarp` and once as the
  nvptx kernel, and the two are held to identical verdicts. **Agreement,
  2026-11-02** (record: `research/gpu-lrat-checker.md`, data in
  `research/gpu-lrat-checker-2026-11-02/`): on 632 files — 114 genuine <!-- unguarded: 632, 114, 66, 452 — archive tallies, research/gpu-lrat-checker-2026-11-02/ -->
  proofs, 66 `drat-trim` outputs from corrupted DRATs (half of them valid
  refutations), and 452 hint-level mutants of the genuine proofs (a hint
  dropped, a literal flipped, a forward hint, a use after delete) — the
  reference returns `lrat-check`'s and `cake_lpr`'s verdict every time, and
  the simulator, the flat decoders and the H200 device path return the
  reference's. **Speed, same H200 pod, 941 MB binary proof:** device check <!-- unguarded: 941 — proof size in MB -->
  0.27 s including transfer; parallel binary decode 0.9 s on 32 threads
  (4.2 s on one); read 0.45 s — about 1.6 s end to end, against 25 s for the
  CPU route on the same pod (parse 17.6 s, 96-thread check 7.4 s). **Same
  machine, same proof (415 MB binary / 901 MB text, H200 pod, 2026-11-03):** <!-- unguarded: 415, 901 — proof sizes in MB -->
  `lrat-check` 20.9 s, `cake_lpr` 21.5 s, our CPU route 11.6 s, device path
  1.1 s end to end — about 19× the external checkers. On a workstation RTX
  4000 SFF Ada with six CPU threads, same proof: 16.2 s, 17.4 s, 9.2 s and
  1.2 s — about 13×. Limits: per-lemma
  assignments above 4,095 literals are reported `TableOverflow` (not
  checked on the warp paths); ids and variables above 2^26 are refused at
  parse time. `cargo run --release --example lrat_check -- f.cnf f.lrat
  [threads]`; `--to-binary out.blrat` writes binary LRAT;
  `tools/lrat-mutate.py` builds the negatives.

## License

MIT
