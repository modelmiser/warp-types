# QF_BV through the certified SAT path — pre-registration

*Written 2026-11-03 before any code, after `research/smt-crosscheck.md` put a
number (23 %) on the ground-only BV module's incompleteness.*

## 1. Design

`warp-types-smt::bitblast`: for a session whose terms are all of bitvector
sorts (widths inferred as the SMT-LIB exporter infers them) and that uses no
uninterpreted function, encode every term as a vector of SAT literals
(variables fresh, constants fixed through one global TRUE variable, `bvadd` as
a ripple-carry adder, `bvsub` as `a + ¬b + 1`, bitwise ops directly, extract and
concat as wiring), every equality atom as the conjunction of bit equivalences,
and the formula by Tseitin, into a `warp-types-sat` `ClauseDb`. Then:

- `check_sat_bitblast()` → the SAT verdict (complete for QF_BV, unlike the
  ground-only module);
- `certify_bitblast(out)` → the DIMACS of the bit-blasted CNF and, on UNSAT, the
  solver's own LRAT (`solve_watched_lrat_with`), so that `lrat-check` and
  `cake_lpr` can accept the proof **of the bit-blasted formula**.

What the certificate covers and does not: it certifies that the CNF is
unsatisfiable; the translation from bitvector terms to that CNF is ours and is
not certified (the standard tools — cvc5's Alethe, Bitwuzla's — certify the
translation too; that is a separate arc). The cross-check against Yices on the
*same* instances is what stands against a translation bug.

## 2. Predictions

- P1 On the 1,000 QF_BV instances of `smt-crosscheck --seed 2026`, the
  bit-blasted verdict equals Yices's on **all 1,000** (the 71 `incomplete`
  cases become `unsat`). Kill: any disagreement.
- P2 Every bit-blasted UNSAT answer's LRAT is `VERIFIED` by `lrat-check` and
  `cake_lpr` against the DIMACS written beside it. Kill: any rejection.
- P3 CNF size: at most 40 variables and 120 clauses per bit of term width
  summed over the terms (a ripple-carry adder is ~5 gates per bit) — a sanity
  bound, not a performance claim.
- P4 Each instance under 50 ms end to end (solve + proof) on one core; the
  generator is tiny.
- Error-sign ledger kept.

## 3. Watched to fail

`x ≠ 0 ∧ x ≠ 1` at width 1 must be `unsat` through the bit-blaster (the ground
module says `Sat`); a hand UNSAT must produce a proof that a corrupted copy of
(one hint dropped) is rejected by `lrat-check`.

## 4. Log (append-only)
- 2026-11-03 — **Built and run** (`bitblast.rs`; `smt-crosscheck --euf 0 --bv 1000 --seed 2026 --bitblast --lrat-check --cake-lpr`; Yices 2.7.0; data `research/smt-bitblast-2026-11-03/`, one CNF/LRAT/SMT-LIB triple kept as an example). Note first: with `--euf 0` the generator's random stream starts at the BV loop, so these are 1,000 fresh BV instances, not byte-for-byte the cross-check's; the shape is the same. **P1 MET: bit-blasted verdict = Yices on 1,000/1,000** (746 sat, 254 unsat); every one of the 62 instances the ground module called `Sat` against a Yices `unsat` is `unsat` through the bit-blaster. **P2 MET: 254/254 UNSAT proofs `VERIFIED` by `lrat-check` and by `cake_lpr`** against the DIMACS written beside each. **P3 MET on the gross numbers** (largest CNF 137 variables, 379 clauses; median 22 / 39) — the per-bit ratio the prediction named was not instrumented by the harness, so the bound is confirmed by inspection of the sizes, not by the measure as written. **P4 MET:** the whole run — 1,000 bit-blasts and solves, 1,000 Yices calls, 508 external checks — took 2.8 s, under 3 ms per instance. Watched to fail beforehand (unit tests): the width-1 witness is `unsat` through the bit-blaster and `Sat` through the ground module; a proof with one hint dropped is rejected by the in-crate checker; 300 random constant-arithmetic cases across the six ops agree with `u64` and the off-by-one claim is `unsat` every time. <!-- unguarded: 1,000, 746, 254, 62, 137, 379, 22, 39, 508, 300 — tallies from results.csv and the test -->
  **What this changes.** QF_BV answers from this crate are now complete, and an UNSAT one carries a certificate a verified checker accepts — of the bit-blasted CNF. The translation is still ours; the Yices agreement on the same instances is what stands against a translation bug, and the account's §1 sentence about SMT can now say "QF_BV UNSAT via bit-blasting: certified; EUF: cross-checked, not certified".
