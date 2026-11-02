# warp-types-bmc

Phase-typed bounded model checker built on [warp-types-sat](https://crates.io/crates/warp-types-sat).

Encodes the BMC workflow in Rust's type system: you can't check a property
before encoding it, can't encode before unrolling, can't deepen after finding
a counterexample. Invalid phase transitions are compile errors.

## Usage

```bash
cargo run --release --bin bmc -- design.aig [--max-depth 1000]
cargo run --bin bmc            # built-in examples
```

With a file, reads binary or ASCII AIGER (1.9: bad literal, invariant
constraints, latch reset values; exactly one bad literal or one output) and
prints `s SAT depth=K`, `s BOUNDED_SAFE depth=N`, or `s EXHAUSTED depth=K`.
Without one, runs built-in examples: a 2-bit counter that reaches a bad state
(UNSAFE at depth 3) and a constant system that is always safe.

## Measured (2026-11-01)

First run on real instances: the HWMCC 2017 single-property set, files under
20 KB (119 instances), 60 s per instance, against ABC 1.01 `bmc3` (30 s,
1000 frames) as the reference. Record, data and pre-registration:
`research/bmc-pdr-aiger.md` and `research/bmc-pdr-hwmcc17-2026-11-01/` at the
repository root.

- **Counterexample depth agrees with ABC on every instance both found: 21 of
  21**, depths 0 to 85. This was the pre-registered control, and it caught a
  real bug on the smoke test (AIGER constant literals mapped with inverted
  polarity made two refutable instances look safe).
- Coverage: ABC found 26 counterexamples in its budget; we found 21 of those
  and none it did not. The five we missed are deep (depth 53–168).
- Time on the seven instances where either side took at least 0.1 s: from
  0.95× to 23× ABC's wall time. On smaller ones ABC's process start-up
  dominates and ratios are meaningless.
- 91 of 119 instances time out at 60 s. The honest size band for this crate
  today is small AIGs; there is no inprocessing and every gate is a state
  variable in the lowering (`src/aiger.rs` documents the encoding).

## What's inside

- **Phase types** (`phase.rs`) — zero-sized markers: Init, Modeled, Unrolled, Encoded, Safe, Counterexample, Exhausted
- **Session** (`session.rs`) — lifetime-branded `BmcSession<'s, P>` with affine phase transitions
- **Transition system** (`model.rs`) — `TransitionSystem` with initial states, transitions, and safety property in CNF
- **Unrolling** (`unroll.rs`) — time-indexed variable encoding, Tseitin property negation
- **Checker** (`checker.rs`) — BMC loop: unroll → encode → SAT oracle → deepen or report
- **AIGER** (`aiger.rs`) — binary/ASCII reader and lowering into `TransitionSystem` (every AIG variable a state variable, gates defined at every step, constraints at every step)

## License

MIT
