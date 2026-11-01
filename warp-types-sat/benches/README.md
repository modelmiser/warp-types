# warp-types-sat benches

Honest benchmark scaffolding. The goal: publish numbers against Rust
peer solvers (`batsat`, `splr`) — not against Kissat or CaDiCaL. See
the top-level README for the rationale.

## What's wired today

- `random_3sat.rs` — random 3-SAT at the phase-transition ratio
  (~4.267), sizes 50/75/100 variables. Measures our CDCL
  unconditionally; under `--features compare`, also measures
  `batsat 0.6` and `splr 0.17` on the same inputs, and runs a
  one-shot agreement check that panics if the three solvers
  disagree on SAT/UNSAT. Primary signal: does our CDCL performance
  regress between releases?
- `cardinality.rs` — pigeonhole PHP(n) and parity XOR chains.
  Four columns per family: `cdcl` (`solver::solve_watched`), `gradient`
  (`gradient::gradient_search`), `hybrid` (tuned config) and, for
  pigeonhole, `hybrid-default`. Primary signal: does the gradient
  warm-start reduce CDCL time like-for-like? (As of 2026-11-01 it does
  not; see the crate README's correction note. Until then the `cdcl`
  column ran `solver::solve`, the basic path, which produced the
  retracted 65×.)
- `satlib_sweep.rs` — auto-discovering sweep over user-provided
  SATLIB `*.cnf` instances. Drop files into `benches/satlib/`
  (gitignored); see `benches/satlib/README.md` for the fetch
  recipe. Under `--features compare`, benches each instance
  against batsat and splr. Runs cleanly on an empty directory
  (prints skip, exits 0). Default cap 10 files; override with
  `SATLIB_MAX=<n>`. Primary signal: do our ratios hold on
  standard benchmark inputs, not just our in-tree RNG?

## Running

```bash
cargo bench -p warp-types-sat
cargo bench -p warp-types-sat --bench random_3sat

# With peer-solver comparison (batsat + splr):
cargo bench -p warp-types-sat --bench random_3sat --features compare

# SATLIB sweep (requires user-provided CNFs in benches/satlib/):
cargo bench -p warp-types-sat --bench satlib_sweep
cargo bench -p warp-types-sat --bench satlib_sweep --features compare
SATLIB_MAX=20 cargo bench -p warp-types-sat --bench satlib_sweep
```

Results land in `target/criterion/` with HTML reports.

## Measured peer-comparison numbers (2026-11-01)

Random 3-SAT at the phase-transition ratio, seed `0xDEADBEEF`,
100 Criterion samples per cell. Medians shown; the 95% CI spans at
most ±1% of the median (widest: ours and splr at n = 100). All three solvers verified as SAT at
each size by the in-harness agreement check. Our column is
`solver::solve_watched` (the watched-literal VSIDS core); log in
`research/bench-random3sat-peers-2026-11-01.txt` at the repo root.

| n   | warp-types-sat (`solve_watched`) | batsat 0.6 | splr 0.17 | ours / batsat | ours / splr |
|-----|----------------------------------|------------|-----------|---------------|-------------|
| 50  | 28.43 µs                         | 17.22 µs   | 57.77 µs  | 1.65×         | 0.49×       |
| 75  | 45.64 µs                         | 27.65 µs   | 99.24 µs  | 1.65×         | 0.46×       |
| 100 | 59.11 µs                         | 34.22 µs   | 120.57 µs | 1.73×         | 0.49×       |

What each timed closure includes: ours and batsat parse the DIMACS
text per iteration; splr receives pre-parsed clauses and clones the
`Vec` per iteration. The parse is a small fraction at these sizes
but it is not zero and it is not symmetric — read the ratios with
that in mind.

**Correction (2026-11-01).** The table this replaces (dated
2026-04-18) read 1.25 / 2.79 / 6.16 ms for our solver — 35×, 49×,
88× slower than batsat — and explained the widening gap by saying
"we do not [implement watched literals] (by design at v0.3)". That
column was measured with `solver::solve`, the basic reference solver
(no watched literals, no VSIDS, no restarts); the crate had the
watched core the whole time, the bench did not call it. Measured
like-for-like, the solver sits between the two peers: ~1.7× batsat
and ~0.5× splr across n = 50–100, with no visible super-linear
widening over this range. See the crate README's "Correction
(2026-11-01)" for the same root cause in the pigeonhole table.

### Configuration

- **CPU:** Intel i9-13900H. **Cores:** `taskset -c 0-5` under
  `nice -n 10` (P-cores; the machine was running other work, load
  average ~2.5). The April table ran on the isolated E-cores
  (`taskset -c 16-19`), so absolute numbers are not comparable
  across the two tables; ratios within a table are.
- **Governor:** `powersave`, as in April — absolute numbers are
  lower than a best-case run; all three solvers run under identical
  conditions, so the ratios are governor-invariant.
- **Kernel:** Linux 7.1.5-76070105-generic.
- **Outlier fraction** (Criterion classification, n = 50/75/100):
  ours 4/11/11%, batsat 5/11/5%, splr 7/6/15%.

### Reading the ratios

A flat ~1.7× against batsat over n = 50–100 is the profile of a
solver with watched literals and VSIDS but no inprocessing and a
single-tier learnt-clause store; the April table's super-linear
widening was the basic solver's scan-based BCP, not this core. If
a change widens the ratio with n, that is the regression signal.

This table exists to detect *regressions* and to give readers
honest context — not to claim competitiveness. The bench file
header is explicit: peer range, not peer parity.

## Follow-up work (not yet wired)

**Gradient-path benchmarks.** Once `gradient::solve` has a stable
public signature, add matching `bench_pigeonhole_gradient` and
`bench_parity_gradient` functions to `cardinality.rs`. This is the
load-bearing measurement for the "promote the gradient path in the
README" decision.

## What we deliberately don't benchmark against

**Kissat, CaDiCaL, Glucose.** These are C/C++ solvers representing a
decade-plus of surgical tuning (multi-tier LBD clause DB,
vivification, BVE, failed-literal probing, chronological
backtracking, inprocessing). We implement none of those and don't
plan to. The relevant peer set for a v0.3 Rust CDCL is `batsat` /
`splr` / `varisat`. If you need competition-class raw throughput,
link Kissat via FFI — that's the right tool.

## Adding a new benchmark

1. Create `benches/my_bench.rs` with a `criterion_group!` and
   `criterion_main!`.
2. Add a `[[bench]]` entry to `Cargo.toml` with `harness = false`.
3. Generate inputs deterministically from a seeded RNG so results
   are reproducible. Shipping CNF files in-tree is a last resort.
4. Document expected performance class in a comment header. If a
   bench exists only to watch for regressions (no competitive
   claim), say so.
