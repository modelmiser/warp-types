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

The `ComplementOf` trait is the deepest reusable piece:
- GPU: `Even ⊥ Odd` (lanes reconverge)
- CSP: `Has3D ⊥ Flat` (protocol branches cover all cases)
- Fence: `PartialWrite<S1> + PartialWrite<S2> = FullWrite` (all wrote before fence)

Same trait, same proof obligation, three domains.

## 10. Potential Paper Titles

**If Option A (composition) holds:**
"Complemented Typestate: A Unified Framework for SIMT Divergence Safety and CSP Protocol Compliance"

**If Option B (separate frameworks):**
"From Warp Divergence to Channel Protocols: Transferring Typestate Mechanisms Across Computational Models"

---

## References

- warp-types crate: `github.com/modelmiser/warp-types` (MIT, v0.3.1)
- Lean 4 metatheory: `warp-types/lean/WarpTypes/`
- crossbar_protocol.rs: `warp-types/src/research/crossbar_protocol.rs`
- warp-core SYSTEM.md: `github/warp-core/docs/SYSTEM.md` (rev 3.0)
- warp-core GRID_TOPOLOGY.md: `github/warp-core/docs/GRID_TOPOLOGY.md` (rev 0.3)
- Moore shrine: `obsidian_shrine/output/moore/` (6 wiki pages, 7 STT transcripts)
