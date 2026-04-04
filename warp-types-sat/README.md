# warp-types-sat

Phase-typed CDCL SAT solver built on [warp-types](https://crates.io/crates/warp-types).

The CDCL state machine is encoded in Rust's type system: you can't propagate
before deciding, can't analyze without a conflict, can't backtrack without
analysis. Invalid phase transitions are compile errors, not runtime assertions.

## Usage

```bash
cargo run --bin solve -- problem.cnf
```

Reads a DIMACS CNF file, prints `s SATISFIABLE` or `s UNSATISFIABLE` with the
variable assignment in standard SAT competition format.

## What's inside

- **Phase types** (`phase.rs`) — zero-sized marker types for CDCL phases, with sealed `CanTransition` trait
- **Affine clause tokens** (`clause.rs`) — non-Copy, non-Clone ownership tokens prevent the #1 parallel SAT bug (double-assignment)
- **Tile-local BCP** (`bcp.rs`, `clause_tile.rs`) — ballot-based clause checking that maps to GPU warp operations
- **1-UIP conflict analysis** (`analyze.rs`) — implication graph traversal, learned clause derivation
- **Trail** (`trail.rs`) — assignment stack with decision levels and reasons
- **DIMACS parser** (`dimacs.rs`) — full CNF format support
- **Solver** (`solver.rs`) — top-level CDCL loop using the phase-typed session

## License

MIT
