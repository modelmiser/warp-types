# warp-types-smt

Phase-typed SMT solver for QF_EUF (Quantifier-Free Equality with Uninterpreted
Functions), built on [warp-types-sat](https://crates.io/crates/warp-types-sat).

Encodes the SMT workflow in Rust's type system: declare sorts and functions,
assert formulas, check satisfiability. Phase transitions are compile-time
enforced — you can't assert before declaring, can't check before asserting.

## Usage

```bash
cargo run --bin smt
```

Runs built-in examples: satisfiable equalities, congruence-closure UNSAT
(`a = b, f(a) != f(b)`), transitivity chains, and Boolean + theory reasoning.

## What's inside

- **Phase types** (`phase.rs`) — zero-sized markers: Init, Declared, Asserted, Sat, Unsat, Unknown
- **Session** (`session.rs`) — lifetime-branded `SmtSession<'s, P>` carrying declaration/assertion state
- **Terms** (`term.rs`) — arena-based with hash-consing for structural identity
- **Formulas** (`formula.rs`) — `SmtFormula` enum with Tseitin CNF transformation and bidirectional `AtomMap`
- **EUF solver** (`euf.rs`) — congruence closure implementing `TheorySolver` from warp-types-sat (DPLL(T))
- **Solver** (`solver.rs`) — pipeline: formula abstraction → EUF theory → SAT oracle

## Theory: QF_EUF

The EUF theory reasons about equality over uninterpreted functions:
- **Congruence**: if `a = b` then `f(a) = f(b)`
- **Transitivity**: if `a = b` and `b = c` then `a = c`

Implemented via backtrackable union-find with a signature-based congruence table.
Integrates with the SAT solver through the DPLL(T) protocol: check after BCP
fixpoint, lazy explanation during conflict analysis, undo on backtrack.

## License

MIT
