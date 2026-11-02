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
instance, ABC 1.01 `pdr` (30 s) as the reference. Record, data, and the 20
invariants: `research/bmc-pdr-aiger.md` and
`research/bmc-pdr-hwmcc17-2026-11-01/` at the repository root.

- **20 proofs, and every one of the 20 invariants passed the independent <!-- unguarded: 20 — campaign count, research/bmc-pdr-hwmcc17-2026-11-01/ -->
  check** (initiation, consecution, safety all UNSAT under kissat on an
  independent encoding). Invariant sizes: one clause on 13 instances, up to
  588 clauses (`bob2`). The checker was watched to reject a flipped literal
  and the property-only fence before it was believed.
- 4 counterexamples, each at exactly the depth ABC's `bmc3` reports. No
  verdict disagrees with ABC on any instance.
- Coverage: ABC proves 55 of these instances in 30 s; we prove 20 in 60 s,
  four of them instances ABC left unknown at its budget.
- Speed: on the 16 instances both prove, ABC takes tens of milliseconds and we
  take 0.1–41 s (median ratio 282×, worst 1,129×). Cubes range over every
  gate in this lowering, and generalisation is per literal.
- 95 of 119 instances time out at 60 s.

What a passed check means: *this* invariant, for *this* instance and this
lowering, is inductive and implies the property, as judged by independent
software on an independent solver. It is not a proof that the engine is
sound in general, it says nothing about the instances that timed out, and
the checker is ordinary Python, not a verified checker (the HWMCC
witness-circuit format and Certifaiger were not used).

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
