# LRAT straight from the solver — pre-registration

*Written 2026-11-03 before any code, after the LRAT-checker arc
(`research/gpu-lrat-checker.md`). Owner: "approved with your suggestions"
(2026-11-02 late), option 1 of the two proposed.*

## 1. Why

The certified chain today is `solve --drat` → `drat-trim -L` → a checker. On the
941 MB-binary proof the checker takes 1.6 s on a GPU pod; `drat-trim` took
**454 s** to produce the LRAT it checks, and `drat-trim` is in the trusted path
(its output is what `cake_lpr` certifies, not the solver's). Emitting LRAT from
the solver removes both: the trimmer's time and the trimmer's trust.

## 2. Design (what will be built)

- `ProofLog` grows three defaulted methods: `wants_hints()`, `add_with_hints(lits,
  hints) -> id`, `delete_ids(ids)`, `empty_with_hints(hints)`. DRAT writers ignore
  them; the new `LratWriter` (text, and binary via `lrat::write_binary`'s layout)
  assigns ids: originals `1..=n` in file order (the DIMACS loader keeps every
  clause, including empty ones, in order — verified in `dimacs.rs`), learned
  clauses `n+1, n+2, …` in emission order.
- The solver core keeps a `CRef → id` map when the log wants hints, remapped on
  compaction like the trail's reasons.
- **The hint chain is derived from the final learned clause, not from the
  resolution order.** Under the assumption "every literal of the learned clause
  is false", every literal of every used clause must be either assumed or
  derived: walk the implication graph from the conflict clause, and for each
  literal that is false on the trail and not in the learned clause, take its
  reason clause and recurse. Emit the collected reasons **sorted by the trail
  position of the literal they imply**, then the conflict clause. Trail order is
  implication order, so each reason is unit when its turn comes and the conflict
  clause is falsified last. This is independent of how minimization was done
  and covers level-0 literals (their unit reasons are collected like any other);
  a decision literal that is not in the learned clause is an analysis bug and is
  reported as such (debug assertion; in release the chain is emitted and the
  checker will say so).
- The empty clause: same walk from the level-0 conflict clause with an empty
  learned clause. An empty ORIGINAL clause: `a id 0 <its id> 0`.
- Theory lemmas (DPLL(T)) have no RUP chain; they are written with an empty hint
  list and counted as `unjustified` — the same honesty as the DRAT path's
  "theory lemmas are not RUP". The CLI is pure SAT.
- CLI: `solve --lrat FILE` (text) and `--lrat-binary`; `--certify` with `--lrat`
  runs `lrat-check` (`$LRAT_CHECK` or PATH) instead of `drat-trim`.
- Fault injection (`proof::fault`, cfg(test)): `DropHint` (drop one hint of each
  chain), `ShuffleHints` (reverse the chain), `WrongId` (off-by-one id) — each
  must flip the in-crate checker's verdict on a hand proof, so that the oracle
  is seen to fail.

## 3. Oracles

1. **In-crate, on every test solve:** `lrat::check_sequential(cnf, parse_text(emitted))`
   must be `Verified` for every UNSAT answer in the solver's existing test
   suite that can be routed through the LRAT log (hand CNFs, PHP, random).
2. **External, on the certification corpus** (`tools/cert-campaign.py`'s 114
   instances + the HWMCC-derived CNFs): `lrat-check` and `cake_lpr` on every
   emitted LRAT; our checker too.
3. **Shape of the proof:** size and step count against `drat-trim -L`'s LRAT for
   the same instance (expected larger: nothing is trimmed).

## 4. Predictions (before any code)

- P1 Every emitted LRAT on the corpus is `VERIFIED` by `lrat-check` and
  `cake_lpr`. **Kill: a rejection that is not a logging bug fixed inside the
  arc — i.e. the chain construction is wrong in principle.**
- P2 Solve-time overhead of LRAT logging vs DRAT logging: between 1.3× and 2×
  the DRAT overhead (the implication walk per conflict is the cost; DRAT's
  overhead was +10–17 % over no logging).
- P3 Emitted LRAT bytes vs `drat-trim -L`'s LRAT for the same instance: within
  1×–3× (ours keeps every learned clause; drat-trim's backward pass drops the
  unused ones).
- P4 End to end, UNSAT answer → external `VERIFIED` (`lrat-check`), the solver
  route beats the `--drat` + `drat-trim -L` route by ≥ 5× on the r3sat n ≥ 200
  instances, where `drat-trim` dominates.
- P5 The checker arc's GPU numbers do not change: the same checker consumes the
  solver's LRAT; what changes is what sits in front of it.
- Error-sign ledger kept per the house rule.

## 5. Not in scope

Trimming (emit only what the final proof uses) — that is `drat-trim`'s job and a
different algorithm; RAT/PR; any theory-lemma justification; solving speed.

## 6. Log (append-only)
