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
cargo run --release --bin pdr -- design.aig [--max-frames 1000] [--invariant fence.txt]
cargo run --bin pdr            # built-in examples
```

With a file, reads AIGER (via `warp-types-bmc::aiger`) and prints
`s UNSAT frame=K clauses=N` (safe: inductive invariant found), `s SAT depth=K`,
or `s EXHAUSTED frames=N`. `--invariant` writes the invariant — the converged
frame's clauses plus the property — as clauses over AIGER literals, so it can
be checked by something that is not this crate: `tools/check-invariant.py`
at the repository root re-reads the AIG with its own parser, builds the
initiation, consecution and safety queries with its own encoding, and asks
kissat. Without a file, runs built-in examples: a trivially safe system
(invariant found at frame 1), an unsafe counter (counterexample at depth 3
with full state trace), and an invariant-discovery example requiring frame
strengthening.

## Measured and checked (2026-11-01)

HWMCC 2017 single-property set, files under 20 KB (119 instances), 60 s per
instance for us; ABC 1.01 `pdr` with a 30 s cap as the reference (so coverage
is not like-for-like). Record, data, and the 20 invariants:
`research/bmc-pdr-aiger.md` and `research/bmc-pdr-hwmcc17-2026-11-01/` at the
repository root. Reviewed by a five-seat, four-family cold panel before
publication; its corrections are in the log.

- **20 instances proved; all 20 invariants pass the independent check, and
  pass it again with the property clause removed** (so each frame alone is
  inductive and implies the property). <!-- unguarded: 20 — campaign count, research/bmc-pdr-hwmcc17-2026-11-01/ -->
  **13 of the 20 are the property clause itself** — the property was already
  inductive, the frame holds one clause — and two of those have a
  constant-false bad output. **7 are non-trivial fences**: `bj08aut82` (224
  frame clauses), `bob2` (588), `bobcount` (395), `intel001` (257),
  `nusmvsyncarb5p2` (162), `pdtpmsarbiter` (132), `shift1add262144` (126).
- The checker (`tools/check-invariant.py`) is separately written software on
  an independent solver (kissat), not an independent reading of the format:
  it mirrors the crate's AIGER conventions. It was watched to reject a
  flipped literal (initiation), the property alone where the property is not
  inductive (consecution), and the empty invariant (safety) before it was
  believed. The crate's own reading of the files is checked semantically by
  the BMC crate's 21-of-21 counterexample-depth agreement with ABC.
- 4 counterexamples, each at exactly the depth ABC's `bmc3` reports, and
  each replays through the AIG under an independent simulator
  (`pdr --trace`, `tools/replay-trace.py`). No verdict disagrees with ABC on
  any instance. A second full run with the constant-true unit on both sides
  of the transition relation reproduced every verdict, depth and
  certificate (`run2-*` in the data directory).
- Coverage: ABC proves 55 of these instances in 30 s; 16 of our 20 are
  among them (29 %). Of the other four, three are outputs ABC declines as
  trivial in 10 ms; one (`shift1add262144`) ABC left unknown at 30 s and we
  prove in 2.1 s.
- Speed: on the 16 instances both prove, ABC takes tens of milliseconds and
  we take 0.00–41.23 s; the ratio runs from under 4× to about 1,100×. Cubes
  range over every gate and input in this lowering, there is no
  cone-of-influence reduction, and generalisation is per literal.
- 95 of 119 instances time out at 60 s, with or without cone-of-influence
  reduction at the lowering (the instances arrive cone-reduced); the
  remedy is inside the engine — cubes over latches rather than over every
  gate — and is not done.

What a passed check means: *this* invariant, for *this* instance and this
lowering, is inductive and implies the property, as judged by separately
written software on an independent solver. It is not a proof that the engine
is sound in general, it says nothing about the instances that timed out,
and the checker and the replayer are ordinary Python, not verified checkers
(the HWMCC witness-circuit format and Certifaiger were not used).

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
