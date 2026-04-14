# warp-types-smt

Phase-typed SMT solver with Nelson-Oppen theory combination, built on
[warp-types-sat](https://crates.io/crates/warp-types-sat).

Supports **QF_EUF** (equality with uninterpreted functions) and **QF_BV**
(fixed-width bitvectors). The two theories cooperate through a combining
solver that shares equalities via the SAT solver's DPLL(T) protocol.

## Quick start

```bash
cargo run --bin smt
```

Output:

```
── 5: x = 3, y = 4, bvadd(x,1) ≠ y — BV constant eval ──
  (SAT with EUF only, UNSAT with BV — bvadd is interpreted)
  EUF only: SAT
  EUF + BV: UNSAT
── 6: x = 3, y = 4, f(bvadd(x,1)) ≠ f(y) — BV + congruence ──
  (BV evaluates bvadd(3,1) = 4 = y → EUF congruence: f(bvadd(x,1)) = f(y))
  EUF only: SAT
  EUF + BV: UNSAT
```

## Usage

```rust
use warp_types_smt::{with_session, BvOpKind, SmtFormula, SmtResult};

let result = with_session(|session| {
    let (session, s) = session.declare_sort("BV5");
    let (session, f) = session.declare_fun("f", &[s], s);
    let (session, x) = session.var("x", s);
    let (session, y) = session.var("y", s);
    let (session, three) = session.bv_const(5, 3, s);
    let (session, four) = session.bv_const(5, 4, s);
    let (session, one) = session.bv_const(5, 1, s);
    let (session, add_x_1) = session.bv_op(BvOpKind::Add, 5, &[x, one], s);
    let (session, f_add) = session.apply(f, &[add_x_1]);
    let (session, f_y) = session.apply(f, &[y]);

    let declared = session.finish_declarations();
    let asserted = declared
        .assert_formula(SmtFormula::And(vec![
            SmtFormula::Eq(x, three),     // x = 3
            SmtFormula::Eq(y, four),      // y = 4
            SmtFormula::Neq(f_add, f_y),  // f(bvadd(x,1)) ≠ f(y)
        ]))
        .finish_assertions();
    asserted.check_sat_bv()  // EUF + BV
});
assert_eq!(result, SmtResult::Unsat);
// BV: bvadd(3,1) = 4 = y  →  EUF congruence: f(bvadd(x,1)) = f(y)  →  conflict
```

Use `check_sat()` for pure EUF, `check_sat_bv()` to enable bitvector reasoning.

## Architecture

```
SAT solver (CDCL, warp-types-sat)
    │  check / backtrack / explain
    ▼
CombiningSolver ──── implements TheorySolver
    ├── EufSolver  ── congruence closure, trail scanning
    ├── BvSolver   ── constant propagation, ground BV evaluation
    └── equality sharing via DPLL(T) BCP loop
```

**Nelson-Oppen combination**: when the BV module discovers an equality (e.g.,
`bvadd(x,1) = y`), the combining solver propagates it to the SAT solver as a
theory propagation. On the next theory check, EUF picks it up from the trail
and fires congruence closure. No internal fixpoint loop — the SAT solver's
existing BCP cycle drives the multi-theory convergence.

**Argument pair purification**: the formula abstraction layer automatically
creates equality atoms for argument pairs of matching function applications.
If the formula has `f(a) ≠ f(b)`, the atom `(a, b)` is created so cross-theory
equalities can be communicated. No manual interface atoms needed.

## Modules

| File | Role |
|------|------|
| `session.rs` | Phase-typed API: `Init → Declared → Asserted → check_sat()` |
| `term.rs` | Arena with hash-consing: `Variable`, `Apply`, `BvConst`, `BvOp` |
| `formula.rs` | Tseitin CNF encoding + argument pair purification |
| `euf.rs` | Congruence closure (backtrackable union-find + signature table) |
| `bv.rs` | BV constant propagation + ground evaluation (Add, And, Or, Xor) |
| `combine.rs` | `CombiningSolver<M: TheoryModule>` + `TheoryModule` trait |
| `solver.rs` | Pipeline: abstraction → theory solvers → DPLL(T) |
| `phase.rs` | Zero-sized phase markers (sealed trait) |

## Adding a theory

Implement `TheoryModule`:

```rust
pub trait TheoryModule {
    fn notify_equality(&mut self, t1: TermId, t2: TermId);
    fn notify_disequality(&mut self, t1: TermId, t2: TermId);
    fn propagate(&mut self) -> ModuleResult;
    fn push_level(&mut self);
    fn backtrack(&mut self, level: u32);
}
```

The combining solver handles trail dispatch, equality sharing, conflict clause
construction, and explanation. Your module just tracks its theory's invariants
and reports equalities (with premises) or conflicts.

## License

MIT
