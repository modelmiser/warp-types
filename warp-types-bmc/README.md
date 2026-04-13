# warp-types-bmc

Phase-typed bounded model checker built on [warp-types-sat](https://crates.io/crates/warp-types-sat).

Encodes the BMC workflow in Rust's type system: you can't check a property
before encoding it, can't encode before unrolling, can't deepen after finding
a counterexample. Invalid phase transitions are compile errors.

## Usage

```bash
cargo run --bin bmc
```

Runs built-in examples: a 2-bit counter that reaches a bad state (UNSAFE at
depth 3) and a constant system that is always safe.

## What's inside

- **Phase types** (`phase.rs`) — zero-sized markers: Init, Modeled, Unrolled, Encoded, Safe, Counterexample, Exhausted
- **Session** (`session.rs`) — lifetime-branded `BmcSession<'s, P>` with affine phase transitions
- **Transition system** (`model.rs`) — `TransitionSystem` with initial states, transitions, and safety property in CNF
- **Unrolling** (`unroll.rs`) — time-indexed variable encoding, Tseitin property negation
- **Checker** (`checker.rs`) — BMC loop: unroll → encode → SAT oracle → deepen or report

## License

MIT
