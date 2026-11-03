# warp-types-smt against an outside solver — pre-registration

*Written 2026-11-03 before the first run. Prompted by the week's account
(`research/certified-verdicts-2026-11.md` §1), which had to say that the SMT
crate's verdicts carry no certificate and no outside check at all.*

## 1. What this is, and is not

A **verdict cross-check**: random QF_UF and QF_BV instances, built through the
crate's own session API, exported as SMT-LIB 2 (`SmtSession::to_smtlib2`,
new), answered by Yices 2.7.0 (`yices-smt2`, the solver installed on this
machine) and by us. It is the same move the model-checking arc made with ABC.
It is **not a certificate**: nothing here can be handed to a checker. A
certificate story for SMT (Alethe/LFSC proofs, or bit-blasting BV through the
certified SAT path) is a separate arc and is not started here.

## 2. The two theories are not symmetric

- `check_sat` (EUF only) is complete for pure EUF: `Sat` and `Unsat` are both
  verdicts. **Rule: every EUF verdict must equal Yices's.**
- `check_sat_bv` is ground-only — it evaluates operations once their arguments
  are concrete and never bit-blasts — so its `Sat` means "no ground conflict
  found" (documented on the method; canonical witness `x ≠ 0 ∧ x ≠ 1` at width
  1). **Rule: our `Unsat` must be Yices `unsat` (soundness); our `Sat` against
  Yices `unsat` is the documented incompleteness and is counted, not killed.**

## 3. Predictions

- P1 EUF: 0 disagreements on 1,000 random instances (3–6 variables, `f` unary,
  `g` binary, 3–7 literals with one disjunction and one implication). Kill: any.
- P2 BV: 0 unsound `Unsat` on 1,000 random instances (width 4 or 8, 2–4
  variables, constants, add/and/or/xor/sub/not, 2–5 literals). Kill: any.
- P3 BV incompleteness rate (our `Sat`, Yices `unsat`): **between 5 % and
  40 %** of the instances Yices calls `unsat`. Written as a range on purpose;
  the first run sets the number. Outside the range in either direction is a
  miss to record, not a kill.
- P4 `Unknown` from our side: 0 (no budget is set).
- Error-sign ledger kept.

## 4. Oracle watched to fail

Before trusting the harness: Yices on the crate's demo instances must return
the demo's documented verdicts (instance 2 `unsat`, instance 5 `unsat`), and
the exporter's text must be accepted by Yices for every generated instance (an
`error` from Yices is counted as unknown and reported; more than 0 is a
harness defect to fix before the numbers count).

## 5. Log (append-only)
- 2026-11-03 — **First run** (`smt-crosscheck --euf 1000 --bv 1000 --seed 2026`, Yices 2.7.0; data `research/smt-crosscheck-2026-11-03/results.csv`). Harness watched to fail first: Yices accepted every exported script (0 errors), and the BV comparison is visibly live (it reports disagreements). **P1 MET: EUF 1,000/1,000 verdicts equal Yices's (827 sat, 173 unsat).** **P2 MET: BV 0 unsound `Unsat` in 1,000 (238 `unsat` agreed).** **P3 MET, inside the range: of the 309 instances Yices calls `unsat`, ours says `Sat` on 71 — 23.0 %** — the ground-only module's documented incompleteness, now with a number. **P4 MET: 0 `Unknown`.** What this adds to the crate's claim: its verdicts agree with an outside solver on every instance where it claims completeness, and it is never unsound where it claims soundness, on this generator. What it does not add: a certificate; the generator is small (≤ 6 variables, depth ≤ 2 terms, widths 4 and 8, no extract/concat); Yices is one reference. <!-- unguarded: 1000, 827, 173, 238, 309, 71 — tallies from results.csv -->
- 2026-11-03 — §4's demo check, for the record: the two demo instances were written by hand as SMT-LIB (`a = b ∧ f(a) ≠ f(b)` in QF_UF; `x = 3, y = 4, bvadd(x, 1) ≠ y` at width 5 in QF_BV) and Yices returned `unsat` for both before the harness ran; the width-1 incompleteness witness `x ≠ 0 ∧ x ≠ 1` was also confirmed `unsat` under Yices, which is the case the BV rule treats as "incomplete", not "kill".
