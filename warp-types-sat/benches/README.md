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
  Cardinality-heavy instances where the gradient path should shine.
  Currently measures the CDCL path only; gradient-path benches land
  once that public API stabilizes. Primary signal: does
  `gradient::solve` beat CDCL on pigeonhole?

## Running

```bash
cargo bench -p warp-types-sat
cargo bench -p warp-types-sat --bench random_3sat

# With peer-solver comparison (batsat + splr):
cargo bench -p warp-types-sat --bench random_3sat --features compare
```

Results land in `target/criterion/` with HTML reports.

## Follow-up work (not yet wired)

**Gradient-path benchmarks.** Once `gradient::solve` has a stable
public signature, add matching `bench_pigeonhole_gradient` and
`bench_parity_gradient` functions to `cardinality.rs`. This is the
load-bearing measurement for the "promote the gradient path in the
README" decision.

**SATLIB corpus.** For reproducible comparison against standard
benchmarks, drop SATLIB CNF files into `benches/satlib/` (add to
`.gitignore`). Fetch from:

- <https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html>
- `uf50-218.tar.gz`, `uf75-325.tar.gz`, `uuf50-218.tar.gz`

A harness that auto-discovers `benches/satlib/**.cnf` is easy to
add on top of the current scaffolding. Never commit the corpus.

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
