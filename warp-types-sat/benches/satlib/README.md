# SATLIB corpus drop directory

The `satlib_sweep.rs` bench auto-discovers `*.cnf` files in this
directory (non-recursively) and runs them through warp-types-sat —
and, under `--features compare`, also through `batsat 0.6` and
`splr 0.17`.

The corpus itself is **not committed** (see the workspace-level
`.gitignore`). Drop files here yourself:

```bash
cd warp-types-sat/benches/satlib
curl -LO https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf50-218.tar.gz
tar xzf uf50-218.tar.gz --strip-components=1
# Or for the 75-variable or 100-variable sets:
#   uf75-325.tar.gz, uuf50-218.tar.gz
```

Then:

```bash
cargo bench -p warp-types-sat --bench satlib_sweep
cargo bench -p warp-types-sat --bench satlib_sweep --features compare
```

If the directory is empty, the bench prints a skip message and
exits cleanly — `cargo bench` never fails on a missing corpus.

Fetch catalog: <https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html>
