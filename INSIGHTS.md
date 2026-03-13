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

---

## Per-Bug Worked Examples as Evidentiary Upgrade (2026-03-13)

### Self-Contained Examples Address Plateau-RESIDUAL

The mm-super audit found evidentiary claims were Plateau-RESIDUAL (0/4 ADMITTED at CL-3/CL-4). Each bug needed a worked example showing the exact type error. The 3 new examples (picongpu_2514.rs, cub_cccl_854.rs, llvm_155682.rs) each:

1. **Redefine the minimal type system needed** -- self-contained, no crate dependency. Reviewers read one file, understand one bug.
2. **Model the exact CUDA bug pattern** -- not abstract, but the real code translated to the type system.
3. **Include "Why __shfl_sync doesn't help"** -- the key addition from the audit. Each explains the specific mechanism by which the runtime mask fails to catch this particular bug class.
4. **Demonstrate the compile_fail** -- both as a doctest and as a comment-in-code pattern showing the exact error message.

The design principle: each example is a standalone proof artifact. A reviewer can `cargo test --example picongpu_2514` and see the type system work without understanding the full crate.

---

## mm-xtal-stem Audit: Copy Derive Soundness Gap (2026-03-13)

### The Audit Found a Real Bug

Running mm-xtal-stem:audit on the paper surfaced 11 implicit assumptions. The most critical: `Warp<S>` derives `Copy+Clone`, which breaks the soundness proof's linearity requirement (Lemmas 4.8-4.9). This isn't a paper weakness — it's a real implementation bug. A user can `let w = Warp::<All>::new(); let (e,o) = w.diverge_even_odd(); w.shuffle_xor(data, 1);` — using the warp after diverge because Copy gives them a copy.

The ghost assumption: "Rust's ownership prevents warp reuse." It doesn't, because Copy overrides move semantics. Three of four independent audit agents found this from different analytical angles (boundary analysis, assumption inventory, conservation mapping).

### Cross-Agent Convergence as Signal

When 3-4 independent agents flag the same issue from different methodological angles, it's a strong signal that the finding is real, not an artifact of any single analysis frame. The linearity violation was found by: (1) Agent 2 via the affine-vs-linear regime boundary, (2) Agent 3 via the linearity assumption category check, (3) Agent 4 via conservation law analysis (linearity conservation broken). Three independent paths to the same bug.

### STEM Audit on a PL Paper

The mm-xtal-stem methodology (confidence hierarchy, 10 assumption categories, ghost assumptions) transfers to PL/type-theory papers with minimal adaptation. The 10 categories mapped as: linearity→linearity, independence→type parameter correlations, equilibrium→execution model dynamics, continuity→abstraction gap, isotropy→directionality, stationarity→architecture evolution, reversibility→control flow irreversibility, determinism→non-deterministic scheduling, scale invariance→nesting depth scaling, completeness→GPU feature coverage. Every category produced at least a "not-applicable" assessment, and 7 of 10 produced substantive findings.

### Zero-Breakage Linearity Enforcement

Removing Copy+Clone from Warp<S> broke zero tests out of 274. All correct usage patterns were already linear — nobody was exploiting Copy. The bug was latent: the possibility of unsoundness existed but no code exercised it. This is the safety-factor pattern: patching the hole changes nothing for correct code while closing it for adversarial or accidental misuse.

### Ferrite as Near-Mandatory Citation

Agent 1 identified Ferrite (Chen et al., ECOOP 2022) — session types embedded in Rust using the exact same PhantomData+traits encoding. At a PL venue, not citing Ferrite would be a reviewer red flag. The distinction is clear (inter-process channels vs intra-warp lanes) but the encoding technique is shared. The shared technique actually strengthens our paper: it validates that Rust's type system is expressive enough for session-type embeddings.
