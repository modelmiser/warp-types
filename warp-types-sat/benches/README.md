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

## Measured peer-comparison numbers (2026-04-18)

Random 3-SAT at the phase-transition ratio, seed `0xDEADBEEF`,
100 Criterion samples per cell. Medians shown; 95% CI widths are
all below 0.3% of the median. All three solvers verified as SAT at
each size by the in-harness agreement check.

| n   | warp-types-sat | batsat 0.6 | splr 0.17 | ours / batsat | ours / splr |
|-----|----------------|------------|-----------|---------------|-------------|
| 50  | 1.251 ms       | 35.63 µs   | 103.73 µs | ~35×          | ~12×        |
| 75  | 2.787 ms       | 56.75 µs   | 173.68 µs | ~49×          | ~16×        |
| 100 | 6.164 ms       | 69.70 µs   | 210.74 µs | ~88×          | ~29×        |

### Configuration (honest disclosure)

- **CPU:** Intel i9-13900H (Raptor Lake-H, laptop).
- **Cores:** `taskset -c 16-19` — the kernel's isolated E-cores
  (`isolcpus=16-19`), which stay quiet under normal P-core desktop
  load.
- **Governor:** `powersave`. Not `performance` — absolute numbers
  are lower than a best-case run, but all three solvers run under
  identical conditions, so the *ratios* are governor-invariant.
- **Kernel:** Linux 6.18.7-76061807-generic.
- **Load average during run:** ~3.0 (P-cores; E-cores stay clean).
- **Outlier fraction** (Criterion classification): ours 6–10%,
  batsat 1%, splr 0–5%. E-cores at `powersave` explain the higher
  variance in our longer-running iterations.

### Reading the ratios

The informative signal is not the absolute gap — it is the
super-linear *widening* (35× at n=50 to 88× at n=100 against
batsat). That is exactly what you would expect from a solver
that lacks watched-literal BCP and multi-tier learnt-clause
management. Both peers implement watched literals; we do not
(by design at v0.3). If a future refactor narrows that
n-dependence, it will show up here as a flatter ratio curve,
not just smaller absolute numbers.

This table exists to detect *regressions* and to give readers
honest context — not to claim competitiveness. The bench file
header is explicit: peer range, not peer parity.

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
