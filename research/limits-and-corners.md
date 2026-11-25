# Limits and corners — how the impossibility results frame this repository

**Written 2026-11-25; rewritten three times the same day after three rounds of a five-seat cross-vendor review
(§4).** A framing note. Repository facts below are cited to the file that states them. Statements that are this
note's own reading are marked as such. Where this note is shorter than `README.md` (Limitations), `lean/README.md`,
`research/drat-certification.md` or `research/lrat-from-solver.md`, those files govern.

## 1. The limits, with their hypotheses

- Under Curry–Howard, the fragment of a language that is used as a logic must rule out unrestricted general
  recursion, or a non-terminating term proves every proposition. The usual way is totality. (The language around
  the logic need not be total: Lean has `partial` and `unsafe` definitions, whose bodies the kernel does not
  reason about.)
- A total language whose programs can be enumerated and run mechanically cannot define every total computable
  function; diagonalising against its own evaluator shows this.
- A consistent, effectively axiomatised theory that interprets enough arithmetic has true arithmetic sentences it
  cannot prove (Gödel I) and, under the standard derivability conditions, cannot prove its own consistency
  (Gödel II).
- By Rice's theorem, for programs in a Turing-complete language no non-trivial semantic property is decidable.

Systems that work do not escape these limits. They pick a corner on purpose and add escape hatches. The question
for any one system is whether the hatches are named.

## 2. Which corner each part of this repository picks, and what it leaves trusted

**The typestate API (`src/`, the published crates).** The corner is decidable checking over a trusted root.
Divergence expressed through `diverge`/`merge` is checked by the compiler. `DynWarp` adds a run-time check, but
that check compares stored masks only: "nothing here observes the hardware or which lanes execute"
(`src/gradual.rs`, on `require_all`). The rest is caller discipline, enforced by neither. README Limitations and
the "Trusted entry points" list on `Warp::kernel_entry` (`src/warp.rs`) name it; among the items:

- `kernel_entry()` is callable anywhere: "The `Warp<All>` witness is not unforgeable … nothing enforces that yet."
- Gradual ascription is a trusted entry point that can produce any `Warp<S>`.
- Merge checks set shape, not provenance.
- A `Warp<S>` can be dropped without merging (Rust is affine, not linear); `#[must_use]` warns, nothing rejects.
- A raw data-dependent branch around a warp intrinsic is ordinary Rust and is exactly the hardware hazard: "The
  guarantee covers the typed fragment."
- `diverge` emits no branch: after it, every lane still runs the code that follows.
- The host-only `research` feature compiles explorations whose `diverge`/`merge` mint handles freely.

So a property the types cannot decide does **not** always become a run-time check on the hardware. Some become
documented caller obligations.

**The Lean mechanization (`lean/WarpTypes/`).** The corner is the kernel-checked logic of Lean. `lean/README.md`
reports zero `sorry` and "a `CollectAxioms` scan of all 530 theorem constants in the `WarpTypes.*` modules <!-- unguarded: 530 — quoted from lean/README.md and README.md; CollectAxioms scan dated 2026-10-18 -->
(2026-10-18)" that "finds only Lean's standard `propext`, `Classical.choice`, `Quot.sound`"; the README's
Soundness row says all 530 theorem constants "in the library" depend only on those three. Three things here are <!-- unguarded: 530 — quoted from lean/README.md and README.md; CollectAxioms scan dated 2026-10-18 -->
this note's reading, not the READMEs':

1. That is an axiom footprint, not the whole trust base. The Lean kernel and toolchain are trusted, and so is the
   fidelity of the definitions to the intended calculus.
2. The scan's script and output are not in the repository, so its exact module scope cannot be checked from here.
   `CoreExperiment.lean` and `CspCoreExperiment.lean` are not imported by `WarpTypes.lean`.
3. The theorems are about a Lean calculus. That the Rust crates follow it is not mechanised: the paper lists
   "Verified Rust implementation via Aeneas translation" as remaining work (`paper/paper.md` §9.2) and records a
   difference in §4.8 — the mechanized `shuffle` consumes its warp operand where the Rust method borrows `&self`;
   "the Rust borrow is not modelled".

None of this bears on the consistency of Lean's logic, which, if consistent, Lean cannot prove of itself
(Gödel II). (The review of this note found `paper/paper.md` §9.2 saying "no axioms" where §4.8 and both READMEs
name the three standard ones; §9.2 was corrected the same day to "no axioms beyond Lean's standard three".)

**Certified UNSAT answers (`warp-types-sat`).** The corner is the small-trusted-kernel pattern: an untrusted
solver, a checked certificate. A certificate is per instance and UNSAT-only, for the `solve_watched_drat` /
`solve --drat` interface with no theory attached; it does not say the solver is sound
(`research/drat-certification.md` §0). The trust base is a pipeline, not one checker. It includes:

- the CNF bytes the checker is given;
- on the DRAT route, `drat-trim`: what `cake_lpr` certifies is the LRAT that `drat-trim` extracted, not the
  solver's own log — "`drat-trim` is in the trusted path (its output is what `cake_lpr` certifies, not the
  solver's)" (`research/lrat-from-solver.md`). So the route supports "this CNF is unsatisfiable"; it does not
  certify that the solver's log was itself a valid proof;
- `cake_lpr`, "the verified LRAT checker (Tan, Heule, Myreen, TACAS 2021)", run as the binary built from its
  shipped CakeML assembly with a C compiler and `basis_ffi.c` (`research/drat-certification.md`); this note's
  reading is that the proof behind "verified", and its specification, are trusted with it;
- whatever script reads the verdict (this note's reading: `research/lrat-from-solver.md` records an ad hoc script
  in the later arc tallying 0 of 114 because it searched for `s VERIFIED` where `lrat-check` prints `c VERIFIED`
  — a miss in the safe direction, but a parser all the same).

`research/drat-certification.md` records by fault injection what a `VERIFIED` does and does not establish: in
`drat-trim`'s default backward mode a corrupted lemma outside the extracted core passes (forward mode `-f` caught
them all); and a final `0` dropped from the log is accepted. `research/lrat-from-solver.md` describes the later
route that emits LRAT from the solver and takes `drat-trim` out of the path.

**`unsafe`.** These are escape hatches, and they are not uniformly discharged. The typed shuffles rely on
the `Warp` witness described above, which safe code can forge ("Every `unsafe` block in the typed shuffles relies
on it", `src/warp.rs`). Some `unsafe fn`s carry no `# Safety` section — for example the private LRAT helper
functions in `warp-types-sat/sat-kernels/src/lib.rs` — and the `#[warp_kernel]` macro suppresses
`missing_safety_doc` on the entry points it generates, giving its reason in a comment: the GPU driver calls them,
not Rust code (`warp-types-kernel/src/lib.rs`). This note did not audit every `unsafe` site.

## 3. Rules this implies for anything written here

1. A statement that something is **sound**, **impossible**, **certified** or **axiom-free** names its corner in the
   same sentence: relative to which logic, with which trusted base, through which escape hatches. (The first
   version of this note broke this rule in three of its four rows; the review caught it.)
2. A property that splits is reported as a split, in **three** parts where they exist: typed, checked at run time,
   and left to caller discipline.
3. Reaching for "undecidable" is checked as carefully as reaching for "proved". A decidable partial check often
   exists in the middle.
4. A result that appears to get expressiveness, guarantees and decidable checking together is treated as a sign that
   something is being trusted silently, and the trusted thing is looked for before the result is written down.

## 4. Review record

Reviewed 2026-11-25 in three rounds by five seats from four model families, each reading the repository. Round 1:
all five refuted the first version's typestate, certificate and `unsafe` rows, its statement of the metatheorems
without their hypotheses, and its claim to change nothing. Round 2, on the rewrite: all five found a citation that
did not support its sentence (a claim-audit row about bug witnesses, cited for the Lean-to-Rust link), and several
found the `DynWarp` wording, the scan-scope attribution and the trust list too strong or incomplete. Round 3: no
seat reported a critical finding; three found another citation that did not support its sentence (a Lean README
section cited for model-versus-Rust differences) and two attributions the cited file does not make. This version
fixes those and was not sent for a fourth round. This record is the author's; the seat reports are kept outside
the repository and cannot be checked from it.
