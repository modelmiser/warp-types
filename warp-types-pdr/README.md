# warp-types-pdr

Phase-typed Property-Directed Reachability (IC3) engine, built on
[warp-types-sat](https://crates.io/crates/warp-types-sat) and
[warp-types-bmc](https://crates.io/crates/warp-types-bmc).

Proves **unbounded** safety properties by discovering inductive invariants.
Unlike bounded model checking (which only proves safety up to a depth bound),
PDR can prove that a property holds at all depths — or find a concrete
counterexample trace if the property is violated.

## Usage

```bash
cargo run --bin pdr
```

Runs built-in examples: a trivially safe system (invariant found at frame 1),
an unsafe counter (counterexample at depth 3 with full state trace), and an
invariant-discovery example requiring frame strengthening.

## What's inside

- **Phase types** (`phase.rs`) — zero-sized markers: Init, Modeled, Safe, CounterexampleFound, Exhausted
- **Session** (`session.rs`) — lifetime-branded `PdrSession<'s, P>` with affine transitions
- **Cubes** (`cube.rs`) — conjunction-of-literals representation with negate/shift/extract
- **Frames** (`frames.rs`) — frame sequence with deduplicating add and convergence detection
- **Checker** (`checker.rs`) — IC3 engine: strengthen (block CTIs), propagate (push clauses forward), converge (detect inductive invariant)

## Algorithm

PDR maintains a sequence of clause sets F_0, F_1, ..., F_k:

1. **Strengthen**: find bad states in F_k, recursively block via predecessor queries
2. **Propagate**: push inductive clauses forward through frames
3. **Converge**: if F_i = F_{i+1}, the property is proved (F_i is an inductive invariant)

Uses the SAT solver as an oracle for consecution, predecessor, and generalization queries.
Shares the `TransitionSystem` model type with warp-types-bmc.

## License

MIT
