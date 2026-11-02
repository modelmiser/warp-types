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
20 KB (119 instances), 60 s per instance for us; ABC 1.01 `bmc3` (30 s, 1000
frames) as the reference, so coverage is not like-for-like. Record, data and
pre-registration: `research/bmc-pdr-aiger.md` and
`research/bmc-pdr-hwmcc17-2026-11-01/` at the repository root. Reviewed by a
five-seat, four-family cold panel before publication.

- **Counterexample depth agrees with ABC on every instance both found: 21 of
  21**, depths 0 to 85, plus `6s319r` at depth 158 and `irstdme6` at 53 in a
  300 s retry. **Every counterexample replays**: `bmc --trace` writes the
  inputs per step and `tools/replay-trace.py` re-simulates the AIG from reset
  with its own parser and simulator, 21 of 21 (run 2). A
  pre-registered comparison with ABC caught a real bug on the smoke test:
  AIGER's constant literals were mapped with inverted polarity, and two
  instances ABC refutes at depth 3 came back BOUNDED_SAFE.
- Coverage: ABC found 26 counterexamples in its budget; we found 21 of those
  at 60 s and none it did not. The five we missed are deep (depth 53–168).
- Time, on the seven both-found instances where either side took at least
  0.1 s: 0.95× to 25× ABC's wall time (`beemadd4b1` 17.3 s vs 18.1 s;
  `prodcellp4` and `productioncellp0` 2.0 s and 1.7 s vs 0.08 s). ABC was
  re-timed on idle cores, three runs, minimum taken, after the panel noted
  the first reference ran under build contention; the numbers did not move.
  On smaller instances ABC's process start-up dominates and ratios are
  meaningless. The two sides had different caps (ABC 30 s, ours 60 s).
- 89 of 119 instances time out at 60 s (91 before cone-of-influence
  reduction, which is now the default lowering; `--no-coi` keeps the full
  AIG). A third full run with it reproduced every verdict and depth; the
  competition instances are already cone-reduced, so it changed two
  constant-false instances from timeout to bounded-safe and little else.
  The honest size band for this crate today is small AIGs: there is no
  inprocessing, the whole CNF is rebuilt at every depth, and every kept gate
  is a state variable in the lowering (`src/aiger.rs` documents it).

## What's inside

- **Phase types** (`phase.rs`) — zero-sized markers: Init, Modeled, Unrolled, Encoded, Safe, Counterexample, Exhausted
- **Session** (`session.rs`) — lifetime-branded `BmcSession<'s, P>` with affine phase transitions
- **Transition system** (`model.rs`) — `TransitionSystem` with initial states, transitions, and safety property in CNF
- **Unrolling** (`unroll.rs`) — time-indexed variable encoding, Tseitin property negation
- **Checker** (`checker.rs`) — BMC loop: unroll → encode → SAT oracle → deepen or report
- **AIGER** (`aiger.rs`) — binary/ASCII reader and lowering into `TransitionSystem` (cone of influence of the bad literal and constraints by default; every kept AIG variable a state variable, gates defined at every step, constraints at every step; trace writer for `tools/replay-trace.py`)

## License

MIT
