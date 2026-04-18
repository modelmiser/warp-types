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

**Cardinality-heavy instances via the gradient solver.** `src/gradient.rs`
follows the FastFourierSAT line (Kyrillidis et al.) — gradient-descent
reformulation with clause-weight EMA on violation frequency. `src/gpu_gradient.rs`
puts per-lane loss + butterfly-reduce + ballot on GPU via `warp-types`.
This is where the parent `warp-types` library does load-bearing work: the
reductions must be warp-uniform and the ballot semantics depend on
lane-safety typing. CPU-path `SimWarp` testing validates the algorithm
before real-hardware execution.

**Reading and learning CDCL with legible types.** Watched literals, 1-UIP
with minimization, VSIDS with phase saving, Luby restarts, LBD-based clause
deletion — a clean first-pass CDCL with the phases surfaced in the type
system for auditability. Used as the reference SAT core in Sol (typed
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

**CDCL baseline is batsat/splr-class, not Kissat-class.** Peer-solver
comparison numbers pending (see `benches/README.md` for scaffolding).

**Hybrid gradient + CDCL on the advertised niche** (UNSAT pigeonhole,
where resolution-based CDCL faces exponential lower bounds):

| n | Pure CDCL | `hybrid_solve` (default config) | `hybrid_solve` (tuned `num_starts=8, max_iters=100`) |
|---|-----------|--------------------------------|------------------------------------------------------|
| 4 | 1.44 ms   | 30.0 ms (0.05×, **loses**)     | 1.00 ms (1.4×) |
| 5 | 17.2 ms   | 58.1 ms (0.3×, **loses**)      | 2.03 ms (8.5×) |
| 6 | 336 ms    | 93.2 ms (**3.6×**)             | 5.18 ms (**65×**) |

All 12 runs verified returning `Unsat` — CDCL's completeness preserved;
gradient contributes VSIDS phase hints and initial activities, CDCL
closes the UNSAT proof from a warm start. Median of 3 runs on an
isolcpus E-core at 4100 MHz; criterion 20-sample CV < 1%.

**Read this table carefully.** At smaller sizes the default-config
gradient overhead exceeds the CDCL savings (default hybrid loses to
pure CDCL at PHP(4) and PHP(5)). The advantage only materializes
once CDCL's resolution search gets expensive enough to amortize the
gradient work. Counter-intuitively, **fewer gradient iterations give
better warm-starts on UNSAT instances** — over-converged gradient
commits confidently to wrong polarities (no satisfying assignment
exists by construction), which CDCL then has to backtrack through.
Tuning matters; treat these numbers as guidance to measure on your
own workload, not a drop-in competition setup.

Peer-solver comparison (`batsat`/`splr` on random 3-SAT) is the
remaining Tier-2 follow-up — until published, read the CDCL column
here as a within-solver reference, not a cross-solver claim.

## Usage

```bash
cargo run --bin solve -- problem.cnf
```

Reads DIMACS CNF, prints `s SATISFIABLE` or `s UNSATISFIABLE` with the
variable assignment in standard SAT competition format.

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

## License

MIT
