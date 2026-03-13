# warp-types Insights

Architectural insights captured during development. Each insight explains
a non-obvious design decision, tradeoff, or pattern specific to this codebase.

---

## Hardware Crossbar Mapping (2026-03-13)

### Participatory Communication is the Unifying Pattern

Both GPU warp divergence and FPGA crossbar communication are instances of **participatory communication** where participants must be in compatible states. GPU: shuffle requires all lanes active. Crossbar: recv requires the sender to actually be sending. Both problems benefit from session-type tracking.

Three structural parallels make this more than analogy:

1. **Same type erasure** -- `TileGroup<S>` is `PhantomData<S>`, zero-sized, same as `Warp<S>`. The safety guarantee costs nothing at runtime in either domain.

2. **Same complement proof** -- `TileComplement` proves disjoint coverage before merge, identical to `ComplementOf`. The compiler enforces that all participants are accounted for.

3. **Same bug class** -- GPU shuffle reads stale register values from inactive lanes. FPGA crossbar reads stale pipeline register values from non-sending tiles. Both produce silent data corruption, not crashes.

This strengthens the paper's "Beyond SIMT" argument from speculation to demonstration.

### The Stale Pipeline Register Bug

In the vcpu-d crossbar (`crossbar_ntile.v`), each (src,dst) pair has a pipeline register with `pipe_valid`/`pipe_data`. When a tile diverges and stops sending, the pipeline register **retains its old value** and `pipe_valid` stays set until the next clock with `pipe_accept`. Another tile reading from that channel gets stale data -- no hardware error, no indication of staleness.

This is structurally identical to shuffle-from-inactive-lane: the hardware provides no runtime check, the data looks valid, and the bug is silent.

---

## Gradual Typing Bridge (2026-03-13)

### DynWarp Answers Three Paper Criticisms at Once

1. **"Marker types can't handle data-dependent predicates"** -- `DynWarp` handles them with runtime checks. `DynWarp::diverge(runtime_mask)` works for arbitrary predicates without needing `Even`/`Odd`/`LowHalf` marker types.

2. **"How do I migrate existing CUDA code?"** -- Start with `DynWarp` (drop-in with runtime checks), progressively `ascribe()` to `Warp<S>` at function boundaries. The compiler guides the migration.

3. **The ascribe boundary is the key design point.** It's where runtime evidence ("this mask is 0xFFFFFFFF") becomes compile-time proof (`Warp<All>`). This is the gradual typing "cast" from Siek & Taha (2006), specialized for warp active sets.

### DynWarp Has a Real Cost

`DynWarp` carries a `u32` mask (4 bytes) plus a branch per operation. `Warp<S>` is zero-sized. Migrating from `DynWarp` to `Warp<S>` is both a safety upgrade (compile-time vs runtime) and a performance upgrade. This makes the migration self-incentivizing -- you get rewarded for adding types.

---

## Empirical Evidence Integration (2026-03-13)

### The "Third Path" Argument

The paper's contribution is strongest when framed as a third path:

- **Path 1 (CUDA status quo):** Allow divergence, check nothing -- silent bugs
- **Path 2 (Hazy megakernel):** Prohibit divergence entirely -- safe but restrictive
- **Path 3 (warp-types):** Allow divergence, enforce safety at compile time

The empirical evidence (4 bugs + vendor deprecation + architectural avoidance) supports all three legs of this argument. Bug evidence shows Path 1 fails. Hazy evidence shows Path 2 exists but is restrictive. Our type system is the first to make Path 3 possible.

### PIConGPU Claim Precision

The PIConGPU bug (#2514) is subtle: the simulation ran for months with UB **undetected on pre-Volta hardware** (K80 GPUs). We initially wrote "wrong output" but the issue says no wrong output was observed -- the UB was masked by lockstep execution. The precise claim is "undefined behavior undetected," not "wrong results observed." This matters for reviewer trust.

---

## mm-super Robustness Audit of Central Thesis (2026-03-13)

### The Thesis is Robust but Its Framing is Fragile

11 cold evaluations (7 Wave 1 + 4 Wave 2) mapped the robustness topology of: "The type system is strictly more permissive than current best practice while being strictly safer than the CUDA status quo."

**Mechanism core is Plateau (9/9 ADMITTED, 7 deep).** No fuzz variant cracked the phantom-type active-set parameterization. This is bedrock.

**"Strictly safer" is Plateau (7/8 ADMITTED, controls agree).** Only V1's "provably" triggered a RESIDUAL -- the safety claim survives all reasonable phrasings.

**"Strictly more permissive" is Cliff (5/6 ADMITTED, controls DISAGREE).** The cliff lives at "current best practice" being unanchored. Fix: name the comparator explicitly (divergence-prohibition approaches, not vague "best practice").

**Evidentiary claims are Plateau-RESIDUAL (0/4 ADMITTED).** "4 documented bugs caught" and "vendor deprecation" consistently fail CL-3/CL-4. Each bug needs a worked example showing the exact type error.

### The Adversarial's Sharpening is a Gift

The adversarial's strongest component -- practitioners avoid divergence for SIMD throughput, not safety -- was ADMITTED deep. Acknowledge this. But its conclusion ("practically vacuous") was RESIDUAL because the activemask() antipattern is a documented bug class that neither `__shfl_sync` nor performance discipline catches. This residual gap IS the paper's non-trivial contribution.

### One-Word Boundaries

Three boundaries collapsed to single-word substitutions:
- **"provably"** -- adding it demands formal metatheory; removing it passes
- **"FPGA"** -- adding it extends to non-SIMT; removing it passes
- **"current best practice"** -- replace with named comparator and it passes

### Bridge Probes: Two Defenses and One Question

**CAP theorem (Defense):** The analogy breaks at synchronous vs asynchronous execution. GPU warps have shared clock + zero-latency merge observation. CAP requires asynchrony. The impossibility doesn't apply -- synchronous execution is what enables the dual improvement.

**Static analysis precision/recall (Defense + Question):** "Simultaneously safer AND more permissive" is a **Pareto improvement** -- the normal outcome when a richer type language distinguishes previously-conflated program classes. Both bridges converge on this independently. Open Question: is shuffle-from-inactive-lane safety decidable for structured control flow? If yes, the type system could be proven sound AND complete.

**DynWarp as defense:** Cold agent independently identified gradual/dynamic hybrid as an escape from precision/recall tradeoff -- which is exactly what `gradual.rs` already provides.

---

## mm-super as Paper Revision Tool (2026-03-13)

### One-Word Boundaries Produce Actionable Edits

The topology map didn't just confirm the thesis -- it produced specific rewrites. Three changes totaling ~80 words across 5 files, each mapping directly to a topology finding:

1. **Named comparators** fix the Cliff-edge permissiveness sub-claim: "current best practice" → "the divergence-prohibition approach exemplified by Hazy" converts vague to falsifiable. This is the one-word boundary in action -- the load-bearing weakness was the unanchored comparator name.

2. **activemask() argument** is the distinguisher that defeated the adversarial's "vacuous" conclusion: `__activemask()` returns hardware-correct masks that encode wrong intent -- a residual bug class neither `__shfl_sync` nor performance discipline catches.

3. **Performance motivation acknowledgment** preempts the adversarial's strongest surviving component (ADMITTED deep): practitioners avoid divergence for SIMD throughput, not just safety. Ignoring this looks naive to GPU reviewers.
