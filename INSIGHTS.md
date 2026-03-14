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

### False Comparability Check on Cross-Domain Claims

The mm-xtal-stem:compare methodology (mechanism/scale/coupling) distinguishes genuine transfers from false parallels. Applied to §9.5 "Beyond SIMT":
- FPGA: All 3 checks pass. Isomorphic bug class (stale pipeline register = shuffle from inactive lane). Working prototype.
- Distributed: Mechanism partially matches (quiescence), scale diverges (bounded deterministic vs unbounded non-deterministic), coupling asymmetric (no GPU analog of network partition).
- Database/proof: Mechanism diverges fundamentally. "Active subset selection" is structural resemblance, not shared failure mode. No communication between filtered rows analogous to shuffle.

The key principle: **transfer fidelity correlates with mechanism match, not structural similarity**. Two systems can satisfy the same abstract pattern (active subset changes during execution) while having completely different failure modes. The paper now grades each claim explicitly rather than listing all four at the same confidence level.

### LLVM IR as Zero-Overhead Proof Artifact

The optimized LLVM IR provides concrete evidence that MIR inspection alone cannot. `zero_overhead_butterfly` (5 shuffle_xor + reduce_sum) compiles to `ret i32 %data`. `zero_overhead_diverge_merge` (diverge + merge round trip) is *aliased* to butterfly by LLVM (recognized as identical machine behavior). The only Warp-containing symbols in the entire optimized IR are error message strings and DynWarp functions. This confirms erasure survives the full Rust→LLVM compilation pipeline.

### BitVec 32 for Active Sets in Lean 4

Using `Fin (2^32)` was the wrong Lean encoding — it represents a number in range, not a bitvector. `BitVec 32` gives native `&&&`, `|||`, `~~~`, and bit-level extensionality. The key proof technique: `ext i; simp_all` reduces bitvector identities to propositional logic, then case-split on `s[i]` for covering proofs. The `decide` tactic handles concrete instances (Even/Odd) because BitVec 32 has decidable equality. The `diverge_partition` proof — arguably the most important theorem — required 4 lines.

### kernel_entry() Prevents Warp Forgery

Making `Warp::new()` `pub(crate)` and adding `Warp::kernel_entry() -> Warp<All>` as the sole public constructor prevents external code from forging `Warp<Even>` handles. Sub-warps are obtainable only via diverge, which consumes the parent — completing the linearity story. Internal code (diverge returning `Warp::new()`, ascribe creating `Warp::new()`) still works because `pub(crate)` grants access within the crate.

### Citation Consistency as Late-Stage Pass

After multiple rounds of paper edits across sessions, internal consistency drifts silently. The Descend author attribution was wrong in paper.md (Steffen) but correct in the standalone files (Kopcke) — different sessions edited different files. The CURD/GMRace conflation (two papers attributed to one) survived multiple editing passes because it was plausible. A final grep-based consistency pass (`grep "et al"`, `grep "10x"`, etc.) catches cross-file divergences that no single edit would notice. This pass also found 8 references that were cited in the body but missing from the references section.

---

## Three Paper-Differentiating Additions (2026-03-13)

### PTX Zero-Overhead: Closing the GPU Backend Gap

The LLVM IR evidence proved type erasure at the Rust→LLVM boundary. But PLDI reviewers ask about the GPU backend. Compiling typed vs untyped butterfly reductions to PTX via `nvcc -ptx -arch=sm_89 -O2` shows byte-identical output after name normalization: same 5× `shfl.sync.bfly.b32` + 5× `add.s32`, same registers (5 predicates, 19 b32). The `__noinline__` attribute is critical — without it nvcc inlines both functions, leaving no distinct bodies to compare. The PTX extraction script needs to match `.func` (not `.visible`) since device-only functions aren't `.entry` kernels.

### value_preserves_ctx: The Key to Linear Progress

The standard challenge in proving progress for linear type systems: sub-expressions of binary forms (merge, shuffle, pair) have non-empty input contexts, but progress is stated for closed terms (empty context). The breakthrough is `value_preserves_ctx`: if `HasType ctx v t ctx'` and `isValue v = true`, then `ctx = ctx'`. This lets progress recurse into the second sub-expression — when the first is a value, its output context equals its input, so the second starts from `[]`. This theorem is fully proved by case analysis on typing derivations (values are warpVal/perLaneVal/unitVal/pairVal, all of which preserve context). The one remaining axiom is `subst_preserves_typing` — the substitution lemma for linear contexts, which requires structural induction with careful context splitting.

### warp_sets! Macro: Compile-Time Hierarchy Validation

The proc macro replaces ~180 lines of repetitive struct+impl code with a ~15-line hierarchy declaration. Key design decisions: (1) validate disjoint/covering/subset properties at macro expansion time via `compile_error!`, not at runtime; (2) deduplicate shared types via `HashSet<String>` — EvenLow appears under both Even and LowHalf but is emitted once; (3) separate `ComplementOf` (top-level, where `parent == 0xFFFFFFFF`) from `ComplementWithin` (always emitted); (4) None/All complement pair is manual (None isn't produced by diverge). The macro also generates `CanDiverge` impls with the trivial `Warp::new()` body, since diverge is a pure type-level operation.

### Rust-to-PTX: Definitive Zero-Overhead Evidence

The `nvptx64-nvidia-cuda` target in Rust nightly compiles actual type system code (PhantomData, trait bounds, ComplementOf, diverge/merge) directly to NVIDIA PTX. Two function pairs produce byte-identical PTX: (1) `butterfly_typed` (through `Warp<All>::shuffle_xor`) vs `butterfly_untyped` (raw arithmetic) — both compile to `shl.b32 %r2, %r1, 5`; (2) `diverge_merge_typed` (`kernel_entry → diverge → merge`) vs `diverge_merge_untyped` (identity) — both compile to `ld.param → st.param → ret`. This is the evidence the cold reviewer identified as the key gap: actual Rust type machinery erased through the full LLVM NVPTX backend, not just "comments don't affect codegen."

### Mechanized Untypability: Bugs Have No Derivation

Five theorems in Lean prove that each documented bug pattern has NO typing derivation. The proof structure: (1) `shuffle_requires_all` gives us that shuffle's warp argument must have type `Warp<All>`; (2) `fst_diverge_warpval_type` gives us that `fst(diverge(warpVal(all), pred))` has type `Warp<all ∧ pred>`; (3) injecting `Ty.warp.injEq` gives `all = all ∧ pred`, which `decide` refutes for any non-trivial predicate. This is unusual for PL papers — most prove soundness (well-typed programs don't go wrong) but not the converse (specific bug patterns are rejected).

### GpuShuffle Trait: Target-Conditional Dispatch (2026-03-13)

The `GpuShuffle` trait bridges the type system and hardware. On `nvptx64`, `gpu_shfl_xor(self, mask)` emits `shfl.sync.bfly.b32` via `core::arch::asm!` with `#![feature(asm_experimental_arch)]`. On CPU, it returns `self` (identity — single-thread emulation). The trait is implemented for `i32`, `f32`, `u32`; f32 reinterprets bits through i32 (`to_bits`/`from_bits`). `Warp<All>::shuffle_xor` dispatches through `data.get().gpu_shfl_xor(mask)`, so the same Rust source compiles to real GPU shuffles OR CPU stubs depending on the target triple. Zero overhead on both paths — PhantomData erased, trait dispatch monomorphized away.

### The Killer Demo: 1 vs 32 (2026-03-13)

`bash reproduce/demo.sh` runs the complete pitch in one terminal. CUDA reduce7 buggy: sum=1 (WRONG, reads from inactive lane 16). Typed Rust fixed: sum=32 (CORRECT, Warp<All> enforced). The buggy pattern — `shfl_down_sync` with `mask=1` — is a compile error in the type system because `Warp<Lane0>` has no `shfl_down` method. It literally cannot be written. The fixed version is the ONLY code that type-checks. Same GPU (RTX 4000 Ada), same algorithm, deterministically wrong vs deterministically correct.

### cudarc 0.19 API: Context + Stream Architecture (2026-03-13)

cudarc 0.19 uses `CudaContext::new(0)` → `ctx.default_stream()` → `stream.memcpy_stod()` / `stream.memcpy_dtov()` for transfers, and `stream.launch_builder(&func).arg(&slice).launch(config)` for kernel dispatch. Module loading: `ctx.load_module(Ptx::from_src(ptx_string))` → `module.load_function("kernel_name")`. Key: `LaunchConfig { grid_dim: (1,1,1), block_dim: (32,1,1), shared_mem_bytes: 0 }` for single-warp kernels. Deprecated methods: `memcpy_stod` → `clone_htod`, `memcpy_dtov` → `clone_dtoh`.

## Month 2 — Cargo-Integrated GPU Pipeline (2026-03-13)

### Conditional no_std for Dual-Target Crates (2026-03-13)

`#![cfg_attr(target_arch = "nvptx64", no_std)]` lets the same crate compile for both x86_64 (with std, tests, research modules) and nvptx64 (core-only, GPU kernels). The research/proof modules are host-only and get gated out with `#[cfg(not(target_arch = "nvptx64"))]`. This is the same pattern rust-gpu uses — the shader crate is dual-target, with host-side utilities excluded on the GPU target. Key: all core modules already use `core::marker::PhantomData`, `core::fmt`, etc. — the no_std migration from Month 1 made this possible.

### cargo rustc --emit=asm for PTX Extraction (2026-03-13)

`cargo build` for a lib crate on nvptx64 produces `.rlib` (archive containing compiled objects) but no standalone PTX file. `cargo rustc -- --emit=asm` tells the Rust compiler to additionally emit the assembly file (`.s`), which for the nvptx64 target IS valid PTX source text. The `.s` file can be loaded directly by cudarc's `Ptx::from_src()`. The output lands in `target/nvptx64-nvidia-cuda/release/deps/` with a hash suffix (e.g., `my_kernels-5c006b3372e23a50.s`).

### Build Script Toolchain Isolation (2026-03-13)

When cargo runs a build script, it exports `RUSTC=/absolute/path/to/stable/rustc`. Child processes (including inner cargo invocations) inherit this, so even setting `RUSTUP_TOOLCHAIN=nightly` doesn't help — the inner cargo uses the parent's `RUSTC` path directly, bypassing rustup's proxy. The fix: `cmd.env_remove("RUSTC")` before invoking the inner cargo. This is the same solution that spirv-builder (rust-gpu) uses. The `+nightly` cargo syntax also doesn't work inside build scripts because it requires rustup's proxy binary, not cargo directly.

### End-to-End Pipeline Architecture (2026-03-13)

The cargo-integrated GPU pipeline has three layers: (1) `warp-types-kernel` proc macro — `#[warp_kernel]` transforms functions to `extern "ptx-kernel"` with `#[no_mangle]`; (2) `warp-types-builder` — build-time library that invokes `cargo rustc --target nvptx64-nvidia-cuda -Z build-std=core -- --emit=asm`, finds the generated `.s` file, copies it to `OUT_DIR`, and generates a Rust module with `include_str!` for the PTX; (3) host code uses `include!` to embed the PTX constant and cudarc to load+launch. User experience: `cargo run` goes from Rust source to GPU execution. Three kernel entry points (butterfly_reduce, diverge_merge_reduce, reduce_n) all produce correct results on RTX 4000 Ada.

### Generated Kernels Struct — PTX Entry Point Parsing (2026-03-13)

The builder parses `.visible .entry <name>(` lines from the generated PTX to discover kernel entry points. It then generates a `Kernels` struct with named `CudaFunction` fields and a `load(&Arc<CudaContext>)` method that calls `ctx.load_module(ptx)` + `module.load_function("name")` for each. Key API detail: cudarc 0.19's `CudaContext::load_module` takes `self: &Arc<Self>`, not `&self`, so the generated code must accept `&Arc<CudaContext>`. The `warp_kernel` macro is re-exported from warp-types itself, so kernel crates need only `warp-types = "0.1"` as a dependency.

## AMD Groundwork (2026-03-13)

### u64 Mask Width — Forward-Compatible Type System (2026-03-13)

`ActiveSet::MASK` widened from `u32` to `u64`. This is the deepest 32-vs-64 assumption in the type system — it was hardcoded into the core trait, the proc macro, DynWarp, and all error types. The change is backwards-compatible (32-bit masks fit in u64) and forward-compatible (64-bit AMD wavefront masks work directly). PTX intrinsics still take u32 — the narrowing happens at the instruction boundary in `gpu.rs`, not in the type system. The `warp_sets!` macro's ComplementOf check was changed from `parent_mask == 0xFFFFFFFF` (width-dependent) to `parent_str == "All"` (name-based, width-agnostic). AMD DPP intrinsic stubs are in place, gated behind `cfg(target_arch = "amdgcn")`, ready for implementation when amdgcn inline asm is stable in Rust nightly.
