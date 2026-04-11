# Complemented Typestate: A Unified Framework for SIMT Divergence Safety and CSP Protocol Compliance

**Date:** 2026-04-10
**Status:** Research note — design thread from Moore shrine / Forth / warp-core conversation
**Context:** Emerged from discussion of Chuck Moore's GA144 CSP model, warp-core J1 mesh topology, and whether the warp-types mechanism generalizes from GPU to FPGA

---

## 1. Research Question

> Can the type-safety mechanisms developed for GPU warp divergence (linear typestate, ComplementOf, Boolean lattice) be applied to heterogeneous multi-core CSP communication to provide compile-time protocol guarantees with zero runtime overhead — and if so, is the mechanism a single framework that covers both SIMT and MIMD, or are they genuinely different type theories that share surface syntax?

Secondary question:

> What is the minimal compiler that can enforce these guarantees on real hardware — and does the compilation-unit constraint of a microcontroller (520KB SRAM) force a design that is architecturally interesting in its own right?

## 2. Existing Assets

| Asset | Status | Role |
|-------|--------|------|
| **warp-types** | v0.3.1, crates.io, Zenodo DOI 10.5281/zenodo.19040615 | GPU side proven: linear typestate, ComplementOf, Lean 4 metatheory |
| **crossbar_protocol.rs** | Research module in warp-types/src/research/ | Sketch: TileGroup<S>, TileComplement, butterfly/ring_pass, stale-data bug demo |
| **Lean 4 metatheory** | warp-types/lean/, zero sorry, zero axioms | Progress, preservation, substitution, 5 bug untypability proofs, DivTree |
| **warp-core J1 mesh** | Star topology current (SYSTEM.md rev 3.0), grid proposed (GRID_TOPOLOGY.md rev 0.3) | Physical target for CSP experiments |
| **Sol** | Lean 4 verification framework, all phases complete | Could mechanize the generic framework |

## 3. The Generic Framework (Lean 4 Sketch)

### Level 0: Complemented Participant Sets (domain-independent)

The core observation: the current Lean 4 metatheory hard-codes `BitVec 32` but the proofs don't use the `32`. All of these theorems work at any bitvector width `n`:

- `diverge_partition` — `(s &&& pred)` and `(s &&& ~~~pred)` are disjoint and cover `s`
- `complement_symmetric` — `IsComplementAll a b → IsComplementAll b a`
- `nested_complement_partition` — nested complements produce pairwise-disjoint triple covering parent
- `DivTree.leaves_cover_root` — arbitrary divergence trees partition the root set
- `diverge_merge_reduces_to_identity` — round-trip diverge+merge = identity

Generalizing to `PSet n := BitVec n` is a mechanical refactor.

### Level 1: Generic Complemented Typestate (domain-independent)

Types: `group (s : PSet n)`, `data`, `unit`, `pair`.

Expressions (core only): `groupVal`, `diverge`, `merge`, `letBind`, `pairVal`, `fst`, `snd`, `letPair`.

Typing judgment: `CoreHasType n : Ctx → CoreExpr n → Ty n → Ctx → Prop`

Key rules:
- **diverge** produces pair of complementary sub-groups
- **merge** requires `IsComplement s1 s2 parent` (THE safety gate)
- **letBind** enforces linearity (freshness + consumption)

Theorems: progress, preservation, substitution — proved ONCE, hold for all domains.

### Level 2a: GPU Domain (current warp-types)

Extends core with `shuffle`. One new typing rule: shuffle requires `group PSet.all`.

Bug untypability proofs: shuffle on diverged sub-warp is untypable (5 real CVEs).

### Level 2b: CSP Domain (new — for J1 mesh)

Extends core with `send`, `recv`, `collective`.

New typing rules:
- **send**: destination must be active AND adjacent (topology constraint)
- **recv**: source must be active AND adjacent
- **collective**: requires all participants (same gate as shuffle)

Parameterized by `Topology n` (adjacency predicate for the mesh).

**Status (2026-04-06):** Shipped as `lean/WarpTypes/Csp.lean` — 376 lines, 13 theorems, 0 sorry, 0 axioms.

### Level 2c: Fence / Partial-Write Domain (added 2026-04-11)

Extends core with `write`, `fence`. No topology, no payload direction.

- `PSet n` reinterpreted as a write mask over an n-byte buffer (bit `i` = byte `i` has been written).
- `group s` is now "linear permission to write the bytes in `s`".
- **write** threads the group handle unchanged, consuming a data payload.
- **fence** requires `group (PSet.all n)` — same gate as `collective` and `shuffle`.

**Status:** Shipped as `lean/WarpTypes/Fence.lean` — 271 lines, 0 sorry, 0 axioms. Commit `c405c3f08`. Concrete instance uses `ByteBuf := PSet 8`. Experiment B documented in DEVLOG.

### Level 2d: Tree All-Reduce Domain (added 2026-04-11)

Extends core with `leafReduce`, `combineRed`, `finalize`, **and a new type constructor** `reduced (s : PSet n)` added to the Level 1 type family.

- `leafReduce : group s → reduced s` — turn a group handle into an accumulator at the same participant set.
- `combineRed : reduced s1 → reduced s2 → reduced parent` (requires `IsComplement s1 s2 parent`) — **structural twin of `merge`**, acting on the new `.reduced` family.
- `finalize : reduced (PSet.all n) → data` — gate on the full-group accumulator; extract a data value.

**Status:** Shipped as `lean/WarpTypes/Reduce.lean` — 302 lines, 0 sorry, 0 axioms. Commit `67eeda474`. Concrete instance uses `Col := PSet 4` with halves split via `halfway_complement`. Experiment C documented in DEVLOG.

**Significance:** Level 2d is the first domain that required extending the Level 1 type family (adding `reduced`). This extension is **conservative over the core**: all 9 core typing rules copy-renamed from `Fence.lean` unchanged, no existing theorems perturbed, `Generic.lean` md5 verified unchanged before and after. See §9-retrospective for the implication: "type-family extension is safe; rule extension is not."

### Level 3: Protocol State (THE RESEARCH QUESTION)

**Option A (conservative extension):** Protocol compliance is a SEPARATE judgment that composes with Level 2b typing. Both must pass. Core unchanged.

**Option B (core modification):** Protocol state threads through the core typing judgment. More expressive but breaks the generic core.

**The critical test:** Does protocol branching (choice) couple with participant-set diverge? In the J1 grid, all cores are always running — they take different protocol branches but the participant set stays `All`. Protocol branching and set divergence are orthogonal. This suggests Option A works.

For GPU warps, diverge IS protocol branching (lanes are masked). This suggests GPU needs Option B — but GPU doesn't have protocols, so the question doesn't arise.

**If Option A holds:** The paper writes itself. Factor warp-types metatheory into generic core + GPU extension. Instantiate for CSP. Add protocol checking as conservative extension. Prove composition.

## 4. Topology-Aware CSP Typing

The warp-core J1 mesh (WJJW grid from GRID_TOPOLOGY.md rev 0.3) is a 2×3 grid:

```
J10 ─── J20       (row 0)
│       │
J11 ─── J21       (row 1)
│       │
J12 ─── J22       (row 2)
```

7 bidirectional CSP links. Max 3 hops (corner to opposite corner). No hub.

Each J1 also bridges exactly one SIMT warp via Wishbone tap (separate plane, not session-typed).

The topology enters the typing rule:

```
structure Topology (n : Nat) where
  adj : Fin n → Fin n → Bool
  sym : ∀ i j, adj i j = adj j i

| send ... (self dst : Fin n) :
    s.getLsb dst = true →         -- dst is active
    topo.adj self dst = true →    -- dst is direct neighbor
    ...
```

Multi-hop protocols compose local (neighbor-to-neighbor) steps. Routing correctness is a theorem: every protocol step uses a real CSP link.

## 5. Two Communication Planes

| Plane | Mechanism | Participants | Typing |
|-------|-----------|--------------|--------|
| **J1 mesh** | CSP channels (blocking rendezvous) | J1 ↔ J1 | Session-typed protocol (Level 2b + 3) |
| **Warp bridges** | Wishbone taps (register read/write) | J1 → Warp | Typed MMIO (simple, not session-typed) |

Clean scope boundary. Session types cover the hard part (peer-to-peer, multi-hop, blocking). Warp bridges are master/slave register access.

## 6. ESP32 as Compiler Host

### Architecture

A Rust-like language with ownership and session types, compiled on the ESP32 (240 MHz, 520KB SRAM, 4MB flash). Desktop IDE over WiFi provides editing and heavy analysis.

### Split

**Desktop (global, rare):**
- Full cross-module type checking
- Overlay map computation (call-graph coloring for J1 IMEM)
- Global protocol composition + deadlock freedom check
- Protocol projection onto each core's role
- Emits: per-core projection metadata + per-function source

**ESP32 (local, frequent):**
- Parses one function at a time (~10-50ms)
- Type-checks against cached projection from flash
- Emits J1 stack code or warp SIMT code
- Loads immediately via SPI (~1ms)
- Edit-compile-run cycle: < 100ms

### Key insight

520KB SRAM forces per-function compilation — the same constraint that made Moore compile one BLOCK at a time on 8K minicomputers. The compiler naturally produces Forth-shaped output: sequences of stack ops, separately compiled, linked via dictionary.

### Unvalidated (from cold review)

- 36KB peak SRAM estimate is unmeasured
- Desktop cross-compiler over WiFi gives comparable latency (~25ms) without microcontroller constraints
- Need: 500-line toy compiler on real ESP32 with measured SRAM + time

## 7. Cold Review Findings (2026-04-10)

A fresh-context agent evaluated each claim:

1. **ESP32 compiler:** Why not just desktop cross-compile? WiFi RTT + compile + SPI ≈ 25ms anyway. Board-resident only wins when desktop disconnected, but then who edits?

2. **Session types:** "Show me the bug." How many deadlocks in current signage code? If zero, this is speculative. (Counter: this is research, not engineering. The question is "does the mechanism generalize?" not "does signage need it.")

3. **Homomorphism:** SIMT (spatial, same code) vs MIMD (temporal, different code) are fundamentally different phenomena. The mechanisms (ComplementOf, linear typestate) transfer, but the phenomena don't. (Counter: the homomorphism is at Level 0-1, not Level 2. The generic core doesn't care about the phenomenon.)

4. **Overlays:** Does 4KB (or 8KB with Super-J1) actually constrain signage? Write the loop first, measure.

5. **Complexity budget:** Six projects in one (language, compiler, session types, overlay manager, IDE protocol, FPGA integration). Any one is substantial.

**Meta-verdict:** This is research, not engineering. The signage app is a test harness. The research contribution is the type theory and empirical evidence on real hardware.

## 8. Research Experiments (in order)

### Experiment 1: The homomorphism, formally (cheapest)

Take `WarpTypes/Basic.lean`, replace `ActiveSet` with `PSet n`. See which theorems survive unchanged. Then write `CspExpr` extension and `FollowsProtocol` judgment. Try to prove composition theorem (`CspHasType ∧ FollowsProtocol → safe`).

**Cost:** Days of Lean 4 work. No hardware.
**Positive result:** Generic framework paper.
**Negative result:** Precise characterization of where the analogy breaks.

### Experiment 2: Minimal compiler on ESP32

500-line compiler: 10-word language, J1 + warp backends, session type checking against hardcoded protocol. Measure peak SRAM and compile time on real ESP32 hardware.

**Cost:** A few weeks of ESP32 Rust.
**Positive/negative:** Both are paper material.

### Experiment 3: Real protocol on real hardware

3-message protocol between two J1 cores. Compile with session types on ESP32. Run on ULX3S. Show: (a) compile error when protocol is broken, (b) session types DON'T catch non-protocol deadlocks (honest limitation).

**Cost:** Depends on J1 RTL timeline.
**Positive result:** First session-typed CSP on FPGA hardware with zero runtime overhead.

## 9. Connection to Moore

Moore's GA144 is an accidentally well-typed CSP system. 144 cores, synchronous blocking channels, no shared memory. Moore avoids deadlocks by holding all 144 programs in his head simultaneously. Session types formalize what Moore does informally. The GA144 proves the communication model is right. The compiler proves it doesn't require Chuck Moore.

### 9.1 The "one trait, three domains" claim (2026-04-09 draft — preserved as prediction)

The `ComplementOf` trait is the deepest reusable piece:
- GPU: `Even ⊥ Odd` (lanes reconverge)
- CSP: `Has3D ⊥ Flat` (protocol branches cover all cases)
- Fence: `PartialWrite<S1> + PartialWrite<S2> = FullWrite` (all wrote before fence)

Same trait, same proof obligation, three domains.

### 9.2 Retrospective (2026-04-11 — four mechanized witnesses)

The §9.1 claim was forward-looking when written. As of 2026-04-11 it has four mechanized witnesses in Lean 4, two of which were written in a single session on 2026-04-11 (Experiments B and C below). The claim as stated was partially right and partially wrong in instructive ways.

**What the witnesses say:**

| Level | File | Lines | Type family used | Gate location | Result type |
|-------|------|-------|------------------|---------------|-------------|
| 2a GPU | `Basic.lean` + `Metatheory.lean` | — | `.group` (alias `ActiveSet := PSet 32`) | `shuffle` requires `.group (PSet.all 32)` | `data` (broadcast value) |
| 2b CSP | `Csp.lean` | 376 | `.group` (alias `TileSet := PSet 6`) | `collective` requires `.group (PSet.all 6)` | `data` (broadcast value) |
| 2c Fence | `Fence.lean` | 271 | `.group` (alias `ByteBuf := PSet 8`) | `fence` requires `.group (PSet.all 8)` | `unit` (barrier) |
| 2d Reduce | `Reduce.lean` | 302 | **`.reduced`** (new constructor, `Col := PSet 4`) | `finalize` requires `.reduced (PSet.all 4)` | `data` (accumulator value) |

All four use `PSet.IsComplement s1 s2 parent` as the merge-time gate and `PSet.all n` as the extract-time gate. The §9.1 claim is **correct** at the level of "same gate, multiple domains." But it understated the finding in two directions:

**(a) The gate is orthogonal to *two* axes, not one.** §9.1 implicitly assumed the gate transferred across domains that shared the `.group` type family. Level 2d broke that assumption: `.reduced` is a new type family, and the `PSet.all n` gate transferred to it without modification. The gate is orthogonal to (i) *barrier-vs-extract* (unit return vs data return) AND (ii) *which `PSet n`-indexed type family carries the participant-set index*. Either axis alone is interesting; both together is a stronger statement.

**(b) The gate transfer has a quantitative refactor cost.** Each new domain currently pays ~85 lines of core-rule duplication to get the transfer. Csp.lean vs Fence.lean share ~87 lines of structural duplication in the core typing rules; Fence.lean vs Reduce.lean adds another ~85 lines. The three-domain total is ~170 lines of duplication that a higher-order refactor could eliminate.

**The refactor target that became visible at four witnesses:**

The `combineRed` rule in `Reduce.lean` is byte-identical to the `merge` rule in `Fence.lean` modulo `.group → .reduced`:

```
| merge      : ... ctx e1 (.group   s1) ... → ... ctx' e2 (.group   s2) ... → IsComplement s1 s2 p → ... (.group   p) ...
| combineRed : ... ctx e1 (.reduced s1) ... → ... ctx' e2 (.reduced s2) ... → IsComplement s1 s2 p → ... (.reduced p) ...
```

Similarly, `finalize_requires_all` is line-for-line identical to `fence_requires_all` modulo the same substitution. The refactor target is **not** "factor N lines into a `Core.lean`" — it is:

> Factor a *family of rules* parameterized by a `PSet n`-indexed type constructor `T : PSet n → CoreTy n` and a gate predicate. The same parametric rule covers `merge`/`combineRed` and a future `mergeStream`/whatever. The same parametric inversion-theorem covers `fence_requires_all`/`finalize_requires_all`.

This is a higher-order refactor, not a line-count refactor. Its shape is:

```
-- sketch
| mergeFamily (T : PSet n → CoreTy n) (mkExpr : CoreExpr n → CoreExpr n → CoreExpr n) ... :
    CoreHasType ctx e1 (T s1) ctx' →
    CoreHasType ctx' e2 (T s2) ctx'' →
    PSet.IsComplement s1 s2 parent →
    CoreHasType ctx (mkExpr e1 e2) (T parent) ctx''
```

Whether Lean 4's inductive type system admits this directly (vs needing a typeclass, enum tag, or reflective representation) is an open question. It was not visible until four witnesses with two distinct type families were on the table. Two rows / one type family is ambiguous; three rows / one type family is still ambiguous (you can't tell "accidentally parametric in operand" from "parametric in type family"); four rows / two type families is enough signal.

**Conservativity of Level 1 type-family extension.** A second-order finding from Experiment C: adding `reduced (s : PSet n)` as a new constructor to the Level 1 `ReduceTy` inductive broke zero existing theorems. `reduce_diverge_partition` still delegates to `diverge_partition_generic` unchanged; all 9 core typing rules copy-renamed from `Fence.lean` without modification; the context / lookup / remove infrastructure needed no changes because it is polymorphic in the type family by construction (`List (String × ReduceTy n)`).

This splits "conservative extension" into two sub-categories:
- **Type-family extension** (adding a new `PSet n`-indexed constructor) — **safe**. No existing theorem is perturbed.
- **Rule extension** (adding a new typing rule whose conclusion uses an existing constructor) — **not safe** in general. Must re-check existing theorems' case analyses.

§9.1 did not state this distinction. It is now named.

**What §9.1 got wrong — the GPU entry.** The original table said "GPU: `Even ⊥ Odd` (lanes reconverge)" as if the GPU domain's `ComplementOf` witness were in the same shape as the others. The witnesses in `Basic.lean` / `Metatheory.lean` were originally concrete at `BitVec 32` and hard-coded the starting set at `PSet.all 32`. Commit `518fdeddc` (2026-04-10) parameterized `Basic.lean` and most of `Metatheory.lean` to `{n : Nat}` and made `diverge_partition` / `complement_symmetric` / `shuffle_requires_all` delegate to `Generic.lean`. The final residual — two private inversion helpers (`fst_diverge_warpval_type`, `shuffle_diverged_untypable`) still concrete at `Ty 32` / `Ctx 32` — was closed by commit `5f7d91276` (2026-04-11): both helpers are now parametric in `{n : Nat}` and in the starting set `{s : PSet n}`, mirroring `Csp.lean`'s `csp_fst_diverge_groupval_type` / `collective_after_diverge_untypable`. `bug1..bug5` kept byte-identical statements and now pass `(PSet.all 32)` as one extra argument to the parametric helper — exactly paralleling how `j1_collective_after_column_split` applies `collective_after_diverge_untypable` at concrete n=6. **The GPU row is now structurally symmetric with rows 2b/2c/2d.** See INSIGHTS §N+42 for the "free-in-the-starting-set costs nothing" lesson this port crystallized.

**Paper-title implications.** The §10 title options both assumed two domains (GPU + CSP). With four mechanized witnesses and a higher-order refactor target, a more accurate title is:

> "Complemented Typestate: A Refactor-Ready Framework for Participant-Set-Gated Operations Across GPU, CSP, Memory-Barrier, and All-Reduce Domains"

Verbose but accurate. A punchier version:

> "One Gate, Four Domains: Complemented Typestate from GPU Warps to Tree All-Reduce"

The mechanized claim is no longer "the framework generalizes" — it is "the framework generalizes AND here is the concrete higher-order refactor the generalization exposes."

### 9.3 Retrospective on Experiment D (2026-04-11 — refactor executed)

The higher-order refactor target from §9.2 was executed the same day it was identified. Outcome: **the `Core.lean` refactor is mechanized, all four pre-existing domain theorems in Fence.lean and Reduce.lean are preserved, and the framework now has a single place to prove generic lemmas about the family-parametric rules.** Commits `bfbb4d272` (Core.lean), `563ba05a0` (Fence.lean port), `489486af2` (Reduce.lean port), `1802039df` (root module).

**Rule shape — what worked.** The `mergeFamily` and `finalizeFamily` rules use a **double equality witness** pattern that generalizes probe 3b. Probe 3b validated a single-witness shape for the expression side (one `heq : expr = tagToFinalExpr tag e`), but its toy `finalizeTagged` rule returned a concrete `.finalTy` — not a dispatched type. The real port had to abstract *both* the expression AND the output type:

    | mergeFamily (tag : TyTag) (ctx ctx' ctx'' : CoreCtx n)
        (e1 e2 expr : CoreExpr n) (s1 s2 parent : PSet n) (ty : CoreTy n)
        (hExpr : expr = tagToMergeExpr tag e1 e2)
        (hTy : ty = tagToTy tag parent) :
        CoreHasType ctx e1 (tagToTy tag s1) ctx' →
        CoreHasType ctx' e2 (tagToTy tag s2) ctx'' →
        PSet.IsComplement s1 s2 parent →
        CoreHasType ctx expr ty ctx''

Without the second witness `hTy`, any `cases` on a hypothesis of concrete shape `CoreHasType ctx e .unit ctx'` fails dependent elimination on the `mergeFamily` branch: the unifier has to solve `.unit =?= tagToTy tag parent`, which is stuck on the tag exactly the same way probe 3a's pattern-match version was stuck on `mkFinal`. The double-witness pattern moves both stuck dispatches into equality hypotheses, preserving full parametricity in the rule's other arguments.

This shape is a new finding. Probe 3b only exercised the single-witness case. The double-witness variant should be the default for any future family-parametric rule whose output type is also dispatched.

**Structural cost — what wasn't in the plan.** The refactor port revealed a second-order cost that neither probe 3b nor the §9.2 sketch anticipated: **every nested `cases` in the existing domain theorems now pays a boilerplate dead-branch discharge for the parametric rules, at every nesting level, even at levels whose hypothesis shape trivially clashes against the parametric rule's dispatched expression constructor.**

Example: Fence's `fence_fst_diverge_groupval_type` helper walks `.fst (.diverge (.groupVal s) pred)` through three levels of nested `cases`. Pre-port it was 4 lines. Post-port it is 22 lines, because at each `cases` level, Lean cannot constructor-clash-eliminate the `mergeFamily` or `finalizeFamily` branches — those rules have `expr` as a free pattern variable, and the cases-site unifier does not unfold `@[reducible]` dispatchers to detect that `tagToMergeExpr tag _ _` would reduce to a clashing constructor. The two layers of Lean 4 elaboration (cases-site vs post-cases rewriting; see INSIGHTS §N+40) manifest again here, now on the *caller* side of the parametric rule rather than its own inversion.

The required discharge is mechanical and identical at every site:

    | mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
      cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
    | finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
      cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr

The port of Fence.lean and Reduce.lean carries 14 such discharges in total (Reduce's `reduce_leaf_fst_diverge_type` walks four nested levels, accounting for 8 of the 14). See INSIGHTS §N+41 for the full elaboration of why the cases-site unifier fails to eliminate these branches and the implication for scaling the refactor to more domains.

This cost is new to §9.2. The §9.2 plan treated the refactor as a line-count win (~170 duplicated lines → ~170 in Core.lean). Actual outcome is a *structural* win, not a line-count win:

| File | Pre-port | Post-port | Δ |
|------|----------|-----------|---|
| Core.lean (new) | — | 276 | +276 |
| Fence.lean | 271 | 209 | −62 |
| Reduce.lean | 302 | 196 | −106 |
| **Total** | **573** | **681** | **+108** |

The line count went *up* because the cost of the dead-branch discharges in helpers+inversion theorems more than offsets the savings from removing the duplicated core rules. But the ~170 lines of structural duplication are now in one place: adding a fifth domain that shares the gate costs one `TyTag` constructor + four dispatcher clauses + ~100 lines of domain-specific examples — not re-copying ~300 lines of core rules. The refactor pays for itself at N ≥ 5 domains by a rough factor of (lines-saved-per-domain / dead-branch-cost-per-helper); at N = 2 domains the refactor is roughly break-even.

**The §9.2 open question, resolved:** "Whether Lean 4's inductive type system admits direct parameterization over a `PSet n`-indexed type family" — **answered positively**, with the caveat that the admitting rule shape is the double-equality-witness pattern, not the naïve "pass the constructor as a function parameter" shape (probe 2 rejected) or the "dispatch via `@[reducible]` function" shape alone (probe 3a rejected). The full trajectory is now: probe 2 identified the obstruction; probe 3a ruled out the cheapest workaround; probe 3b validated the working rule shape for a single stuck dispatch; the §9.3 port validated the double-witness extension and quantified the caller-side structural cost.

**What Experiment D did not settle.** Three things remain open:
1. A reflective or typeclass-based encoding that eliminates the caller-side boilerplate discharges might exist. The Experiment D trajectory intentionally took the cheapest working path (explicit equality witnesses); a more elaborate encoding could, in principle, let Lean's cases-site unifier see through the dispatcher. Worth revisiting only if a fifth or sixth domain makes the boilerplate cost intolerable.
2. The Level 2a GPU port onto `Generic.lean`'s width-parametric theorems (the known gap from §9.2) — **closed 2026-04-11 in commit `5f7d91276`** (Phase 1). Turned out to be a ~25-line change to two private inversion helpers in `Metatheory.lean`, not the structural refactor this retrospective anticipated: `Basic.lean` had already been parameterized to `{n : Nat}` in commit `518fdeddc` (2026-04-10), leaving only the `fst_diverge_warpval_type` / `shuffle_diverged_untypable` helpers concrete at `Ty 32` / `Ctx 32`. Parameterizing both helpers over `{n : Nat}` and the starting set `{s : PSet n}` (mirroring Csp.lean) carried zero extra proof work — the `match` bodies and `rfl` tails transferred unchanged. `bug1..bug5` statements stayed byte-identical; each proof gained one `(PSet.all 32)` argument. The GPU row of the four-domain table now has signature-level symmetry with rows 2b/2c/2d, not just informal-claim-level parallelism. The port did NOT plug `Basic.lean` into `Core.lean`'s `CoreHasType` — Core unification for Level 2a remains out-of-scope and is tracked as a possible future experiment alongside Csp unification (Experiment E). See INSIGHTS §N+42.
3. `Csp.lean` was NOT ported onto Core.lean in this commit sequence. Csp has topology (the `TopoRel` parameter) and a different expression-language shape (send/recv channels vs write/fence bulk operations). Unifying Csp into Core.lean would require either widening Core to carry a `TopoRel` parameter everywhere or splitting Core into topology-aware and topology-free variants. Scoped out of Experiment D. **Feasibility closed 2026-04-11** in commit `613640663` (Experiment E probe, `lean/WarpTypes/CspCoreExperiment.lean`, 468 lines): probe returned E1 with design (a) "widen Core" working on first build, all three pre-registered falsification tests (F1 construction, F2 new-axis inversion, F3 existing-family inversion) pass in the same inductive, Fence/Reduce shadow proofs byte-identical. Cost estimate: ~120 lines added to `Core.lean`, ~80 lines removed from `Csp.lean`, net +40. The real port is now a **prioritization call, not a feasibility call** — tracked in `TODO.md`. See INSIGHTS §N+43 for the "orthogonal axes compose" finding that makes the widen-Core path work without regressing Fence/Reduce inversion proofs.

**Paper-writing implication.** The paper's §9-analog should distinguish **the theoretical claim** ("complemented typestate is one gate, four domains, unified under a higher-order factoring") from **the engineering cost** ("the unification has a caller-side boilerplate tax that scales linearly with existing-theorem-cases-depth × family-rule-count"). Both are findings. The first is the framework claim; the second is the honest report on the mechanization.

## 10. Potential Paper Titles

> **Note (2026-04-11):** Both titles below are the 2026-04-09 draft. Option A has since been confirmed (Protocol.lean, commit lineage 2026-04-10) and the framework has been extended to four domains (see §9.2). Current preferred title is in §9.2 near the bottom.

**If Option A (composition) holds:**
"Complemented Typestate: A Unified Framework for SIMT Divergence Safety and CSP Protocol Compliance"

**If Option B (separate frameworks):**
"From Warp Divergence to Channel Protocols: Transferring Typestate Mechanisms Across Computational Models"

---

## 11. Experimental Record

Experiments are numbered; each is a discrete, falsifiable test with a pre-registered hypothesis and a concrete success criterion. Entries below were added 2026-04-11 after the experiments landed.

| # | Name | Date | Commit | File | Lines | Outcome |
|---|------|------|--------|------|-------|---------|
| 1 | The homomorphism, formally | 2026-04-06 | (Csp.lean + Protocol.lean sequence) | `Csp.lean`, `Protocol.lean` | 376 + 834 | Level 2b CSP + Level 3 Protocol shipped; Option A confirmed; branching-diverge orthogonality proved; `protocolTrace` structural characterization; `follows_protocol_rfl` tactic |
| B | Third domain (fence / partial-write) | 2026-04-11 | `c405c3f08` | `Fence.lean` | 271 | F3-mild: 9/9 core rules + 11/11 core expression constructors transferred verbatim modulo copy-rename. ~87 lines structural duplication with Csp.lean. `Generic.lean` untouched. Strong form of "gate is orthogonal to op shape" observed |
| C | Fourth domain (tree all-reduce) + new type family | 2026-04-11 | `67eeda474` | `Reduce.lean` | 302 | Stronger than predicted F3: gate transferred across two orthogonal axes (barrier-vs-extract AND `.group`-vs-`.reduced` family). Higher-order refactor target exposed (see §9.2). Type-family extension shown conservative over core. ~170 lines total duplication across three domains |
| D probe | Higher-order factoring feasibility | 2026-04-11 | `9240e6c19` | `CoreExperiment.lean` | 330 | Three-probe trajectory. Probe 1 (construction with function-valued parameters): works. Probe 2 (inversion with function-valued parameters): fails — higher-order pattern unification at cases-site. Probe 3a (inversion with `@[reducible]` dispatcher): fails — cases-site unifier does not unfold `@[reducible]`. Probe 3b (inversion with explicit equality witness): works. Shape de-risked for port |
| D port | Higher-order factoring executed | 2026-04-11 | `bfbb4d272`–`1802039df` | `Core.lean` + Fence/Reduce ports | 276 + 209 + 196 | Core.lean introduced, both domains ported. `mergeFamily` and `finalizeFamily` use **double** equality witnesses (expr AND output type) — new finding beyond probe 3b's single-witness variant. All 8 pre-existing theorems preserved, `Generic.lean` md5 unchanged, `lake build WarpTypes` green. Structural cost: nested-cases helpers require explicit dead-branch discharges for parametric rules at every level; net line count went from 573 → 681 (+108), not down. Refactor win is structural not numerical (see §9.3). INSIGHTS §N+41 documents the caller-side cost |
| Phase 1 | Level 2a GPU row port (structural match) | 2026-04-11 | `5f7d91276` | `Metatheory.lean` | +23/-19 | The residual GPU `BitVec 32` hard-coding — two private inversion helpers (`fst_diverge_warpval_type`, `shuffle_diverged_untypable`) — parameterized over `{n : Nat}` AND the starting set `{s : PSet n}`, mirroring Csp.lean's `csp_fst_diverge_groupval_type` / `collective_after_diverge_untypable`. `bug1..bug5` kept byte-identical statements. Zero new proofs; `match`/`rfl` bodies transferred unchanged. `Generic.lean` md5 untouched. GPU row now has signature-level symmetry with 2b/2c/2d. INSIGHTS §N+42 (free-in-the-starting-set). |
| E probe | Csp.lean → Core.lean feasibility | 2026-04-11 | `613640663` | `CspCoreExperiment.lean` | 468 | Returned **E1** (widen Core works). Design (a) adds `topo : Option (Topology n)` as inductive parameter + `send` / `recv` / `collective` rules to the widened judgment. Three pre-registered falsification tests all pass in the same inductive: F1 construction, F2 new-axis inversion (send/recv adjacency gate), F3 existing-family inversion (fence_requires_all mirror). F3b added on the finalize side for symmetry with Reduce. Standalone build clean in 489 ms on Lean 4.28.0; `Generic.lean` md5 unchanged; `lake build WarpTypes` still green. Design (b) "split Core" skipped (not needed), sketch preserved at bottom of probe file. **Real port cost estimate:** +120 lines Core, −80 lines Csp, net +40. Csp's existing nested-cases helpers gain +3 auto-eliminated dead branches per level (zero new discharge code — send/recv/collective have concrete expression patterns, unlike the tagged-dispatch rules that need the N+41 pattern). INSIGHTS §N+43 (orthogonal axes compose). |
| 2 | ESP32 compiler on-target | 2026-04-11 (unblocked) | — | `research/esp32-compiler/` | — | Paused (research track only) |
| 3 | Real protocol on real hardware | — | — | — | — | Depends on J1 RTL timeline |

The Level 2a GPU port of `Basic.lean` / `Metatheory.lean` to `Generic.lean`'s width-parametric theorems was closed 2026-04-11 in Phase 1 above (commit `5f7d91276`). All four domain rows now have signature-level symmetry. The `Csp.lean` → `Core.lean` unification feasibility question was answered positively the same day by the Experiment E probe (row above, commit `613640663`): widen-Core works, Fence/Reduce stay byte-identical proof-wise, real-port cost is ~+40 net lines. Csp still does not plug into Core at `HEAD` — the real port is **tracked in `TODO.md` as a prioritization call**, not a feasibility open question.

---

## References

- warp-types crate: `github.com/modelmiser/warp-types` (MIT, v0.3.1)
- Lean 4 metatheory: `warp-types/lean/WarpTypes/` — `Generic.lean` (Level 0-1, frozen md5 `7f125b5f5f26122cc9e97c39522a4d03`), `Basic.lean` (Level 2a GPU), `Csp.lean` (Level 2b), `Core.lean` (higher-order factoring for Level 2c+2d, commit `bfbb4d272`), `Fence.lean` (Level 2c, port commit `563ba05a0`), `Reduce.lean` (Level 2d, port commit `489486af2`), `Protocol.lean` (Level 3), `CoreExperiment.lean` (Experiment D feasibility probe, commit `9240e6c19`)
- crossbar_protocol.rs: `warp-types/src/research/crossbar_protocol.rs`
- warp-core SYSTEM.md: `github/warp-core/docs/SYSTEM.md` (rev 3.0)
- warp-core GRID_TOPOLOGY.md: `github/warp-core/docs/GRID_TOPOLOGY.md` (rev 0.3)
- Moore shrine: `obsidian_shrine/output/moore/` (6 wiki pages, 7 STT transcripts)
