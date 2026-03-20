# 8. Related Work

Warp typestate draws on and differs from work in GPU verification, session types, and type systems for parallelism.

## 8.1 GPU Verification

### Descend (PLDI 2024)

Descend [Kopcke et al. 2024] brings Rust-style ownership and borrowing to GPU programming, preventing data races and use-after-free in GPU code.

**Relationship to our work**: Descend and warp typestate are *orthogonal* and *composable*:
- Descend: memory safety (ownership, borrowing, lifetimes)
- Our work: divergence safety (active lane tracking)

Descend does not track which lanes are active. A shuffle in Descend may read from inactive lanes if the programmer gets the mask wrong. Conversely, our system does not track memory ownership.

The ideal system combines both: lanes must be active (our contribution) *and* have valid data (Descend's contribution).

### GPUVerify, CUDA Sanitizers, and Race Detection

GPUVerify [Betts et al. 2012, 2015] uses predicated execution semantics and SMT solving to prove race-freedom. NVIDIA's compute-sanitizer detects memory errors, races, and synchronization bugs at runtime. GMRace [Zheng et al. 2014] and CURD [Peng et al. 2018] detect warp-level data races via static analysis and dynamic instrumentation, respectively. All of these are *external* to the type system: separate tools or runtime passes that provide post-hoc verification. Our approach integrates verification into the type system itself—immediate feedback during editing, lightweight (no SMT), and specifically targeting the shuffle-from-inactive-lane bug class that these tools do not specifically address.

### LLVM Uniformity and Divergence Analysis

LLVM implements uniformity analysis that determines whether SSA values are uniform (same across all threads in a warp) or divergent. This analysis propagates divergence along def-use chains and control dependencies, supporting irreducible control flow.

**Relationship to our work**: LLVM's divergence analysis and our type system track related information but differ fundamentally. LLVM's analysis is a compiler pass—intraprocedural, best-effort, focused on optimization (avoiding unnecessary predication). Our type system is source-level, modular across function boundaries, and focused on safety. LLVM's analysis identifies *which* values are divergent but does not track *which lanes are active*. Our active-set types capture exactly this distinction. Bug 4 (LLVM#155682) demonstrates that LLVM's own optimizations can cause the bug class we prevent.

## 8.2 Session Types

### Binary Session Types

Session types were introduced by Honda [1993] for the π-calculus. A session type describes a communication protocol between two parties.

**Relationship to our work**: Binary session types assume two active parties. GPU divergence involves up to 32 parties where any subset may be inactive. We extend the model with *quiescence*.

### Multiparty Session Types (MPST)

Honda, Yoshida, and Carbone [2008] extended session types to multiple parties. Each party follows a local type projected from a global protocol.

**Relationship to our work**: Our system shares MPST's concern with multi-party coordination but differs in mechanism. MPST types *channels* carrying *directed messages* between parties following a *protocol sequence*. Our system types a *linear resource* (the warp handle) carrying a *set-valued state* (which lanes are active), with no channels, no directed messages, and no protocol sequencing. The structural analogy—branching, compatibility, reconvergence—is genuine and motivating but not a technical extension of MPST. The key novelty (quiescence: parties go temporarily inactive rather than failing) is a concept that *could* extend MPST, but our formalization does not build on MPST foundations.

| MPST | Our System | Match |
|------|-----------|-------|
| N parties, all active | 32 parties, subset active | Motivating analogy |
| Channels with send/receive | Shared register file, symmetric shuffle | No |
| Protocol sequence | Active-set snapshot | No |
| Party fails = session stuck | Party quiesces = temporarily inactive | Novel concept |

### Gradual Session Types

Gradual session types [Igarashi et al. 2017] allow mixing static and dynamic typing for sessions. Unknown types are checked at runtime.

**Relationship to our work**: Our Layer 4 (existential types, §5.2) and `DynWarp` gradual typing bridge (§9.3) are directly inspired by this work. Our `ascribe()` operation corresponds to the cast at the gradual typing boundary.

### Fault-Tolerant Multiparty Session Types

Recent work extends MPST to handle participant failures: crash-stop failures [Adameit et al. 2022] and fault-tolerant event-driven programming [Viering et al. 2021].

**Relationship to our work**: Fault-tolerant MPST models *permanent* failure (crash-stop). GPU divergence involves *temporary* quiescence—lanes go inactive and resume at merge. Crash-stop requires protocol recovery; quiescence requires complement proof. The two extensions are complementary.

### Session Types Embedded in Rust (Ferrite)

Ferrite [Chen et al. 2022] embeds session types in Rust using PhantomData, zero-sized types, and type-level programming—the same encoding techniques we use.

**Relationship to our work**: Ferrite models inter-process communication channels; we model intra-warp lane communication with quiescence. Key differences: Ferrite's channels carry data (ours share a register file), Ferrite's session types describe message sequences (ours describe active-set evolution). The shared encoding validates that Rust's type system is expressive enough for session-type embeddings.

Dardha et al. [2017] apply session types to concurrent objects; the synchronization model (asynchronous objects vs. lock-step warps) differs fundamentally from ours.

## 8.3 Type Systems for Parallelism

### Futhark

Futhark [Henriksen et al. 2017] is a functional GPU language with a type system that guarantees regular parallelism.

**Relationship to our work**: Futhark *avoids* divergence by design. Its parallelism constructs (map, reduce, scan) don't support divergent branches.

Our approach is complementary: we *embrace* divergence and make it safe. This allows expressing algorithms like adaptive sorting where divergence is fundamental.

### ISPC (Intel SPMD Program Compiler)

ISPC [Pharr and Mark 2012] implements SPMD programming on CPU SIMD hardware with `uniform` and `varying` type qualifiers—a `uniform` variable holds a single value shared across all instances in a gang (analogous to our `Uniform<T>`), while `varying` (the default) holds per-instance values (analogous to our `PerLane<T>`). ISPC also provides `foreach_active` and cross-lane operations whose mask correctness is guaranteed by compiler-emitted mask instructions.

**Relationship to our work**: ISPC is the closest existing system in its language-level awareness of divergence. ISPC's compiler manages the execution mask and provides `foreach_active` to iterate over active instances — so ISPC *knows* the active set at runtime and exposes it through language constructs. The key difference is *where* the active set lives: ISPC tracks it at runtime (compiler-emitted mask instructions), while our `Warp<S>` types encode it at compile time (the type parameter `S` is the active set). In ISPC, cross-lane operations are safe because the compiler generates correct masks. In GPU programming, the programmer passes masks manually via `__shfl_sync(mask, ...)` — exactly where the bug class arises. Our type system makes `shuffle_xor` *absent from the type* when lanes are inactive, preventing the mask error at compile time rather than relying on runtime mask management.

### DPJ, Æminium, and Data-Race-Free Type Systems

DPJ [Bocchino et al. 2009] uses region types for determinism in parallel Java. Æminium [Stork et al. 2014] extracts parallelism from sequential code via access permissions. Data-race-free type systems [Boyapati et al. 2002, Flanagan and Freund 2000] ensure race-freedom through types. All three focus on preventing data races or ensuring determinism—orthogonal to our concern. A data race (two threads access same location, at least one writes) and a divergence bug (one thread reads from an inactive lane's register) are distinct bug classes; our active-set types address only the latter.

## 8.4 Linear and Affine Types

### Ownership Types (Rust)

Rust's ownership system [Matsakis and Klock 2014] ensures memory safety through affine types. Our `Warp<S>` uses Rust's move semantics—consumed by diverge, produced by merge—preventing use-after-diverge. Rust's type system is *affine* (values used at most once or dropped), not linear (must be used exactly once), so a `Warp<S>` can be silently dropped without merging; we mitigate this with `#[must_use]` warnings, and our Lean formalization models stricter linear semantics.

### Linear Logic

Linear logic [Girard 1987] provides the foundation for both session types [Caires and Pfenning 2010, Wadler 2012] and linear resource typing. Our warp linearity uses multiplicative conjunction: diverge produces `Warp<S1> ⊗ Warp<S2>`; merge consumes such a pair. This is the *resource* reading of linear logic, not the *session* reading (where ⊗ types a channel that sends a channel)—a distinction that matters for understanding our guarantees.

## 8.5 GPU Programming Models

CUDA [NVIDIA 2007], OpenCL [Khronos 2009], SYCL [Khronos 2020], oneAPI, and HIP [AMD 2016] all expose warp/wavefront/sub-group primitives but provide no type-level divergence safety; our typed layer can wrap any of these. Cooperative Groups [NVIDIA 2017] make group membership explicit but still allow shuffling on groups where threads have diverged—we provide the missing types.

### AMD DPP and Intel Subgroup Operations

AMD's Data Parallel Primitives (DPP) instructions (`v_mov_b32_dpp`, `ds_swizzle_b32`) enable cross-lane communication within 64-lane wavefronts on RDNA/CDNA architectures. Unlike NVIDIA's explicit mask parameter, AMD's DPP operations are implicitly masked by the hardware execution mask — the programmer does not pass a mask, but the operation only executes for active lanes. The divergence risk is different: not a wrong mask, but an unexpected execution mask due to unstructured control flow. Our `u64` active-set masks and `GpuWarp64` platform abstraction are designed to support AMD wavefronts, though the GPU backend is not yet implemented.

Intel's subgroup operations in SYCL/oneAPI (`sub_group::shuffle`, `sub_group::reduce`) and OpenCL (`sub_group_shuffle`) require that "all work-items in the sub-group execute the same call" — the same convergence requirement as NVIDIA's shuffle. Intel's subgroup size varies (8, 16, 32, or 64 depending on hardware), adding a portability dimension to the bug class. Our `Platform` trait and parameterized warp width address this through type-level abstraction.

### Vulkan/SPIR-V Subgroup Operations

Vulkan 1.1 (2018) introduced subgroup operations (`subgroupShuffle`, `subgroupBallot`, `subgroupBroadcast`) in SPIR-V/GLSL — the most widely deployed cross-vendor subgroup API. The GLSL specification explicitly states that subgroup operations have "undefined results if invocations are not converged." The WebGPU specification excluded indexed subgroup shuffles entirely due to this safety concern (Bug 21 in our survey). Our type system would prevent the undefined-results cases that the Vulkan specification warns about — the shuffle method is absent on diverged warps, not merely documented as "undefined."

### NVIDIA's `__shfl_sync` Migration (CUDA 9.0)

NVIDIA deprecated the original `__shfl` family in CUDA 9.0, replacing it with `__shfl_sync` which requires an explicit mask parameter [CUDA Programming Guide §10.22]. This was a vendor acknowledgment that the bug class is severe enough to warrant a breaking API change across the ecosystem. However, the mask remains a runtime value—programmers can still pass the wrong mask, as documented bugs in NVIDIA's own cuda-samples and CUB demonstrate.

**Relationship to our work**: `__shfl_sync` addresses the problem at the API level (require a mask). We address it at the type level (prove the mask correct). The approaches are complementary: `__shfl_sync` prevents *forgetting* the mask; our types prevent *getting it wrong*.

### NVIDIA Cooperative Groups

NVIDIA Cooperative Groups [2017] introduce `thread_block_tile<N>` for statically-sized sub-warp groups and `coalesced_threads()` for dynamically obtaining the set of converged threads — a runtime active-set mechanism. The `sync()` method enforces convergence before cross-lane operations. This is the closest vendor-provided abstraction to our type-level active sets. The key difference: Cooperative Groups track convergence at runtime (the group object is a runtime value), while our `Warp<S>` tracks it at compile time (the active set is a type parameter). A `thread_block_tile<16>` can call `shfl` and the hardware confines it to the tile — safe by construction. But `coalesced_threads()` returns a runtime group whose membership can change at each reconvergence point, and the programmer must ensure the group is still valid when cross-lane operations execute. Our type system would prevent using a stale group handle.

### NVIDIA compute-sanitizer `synccheck`

NVIDIA's `compute-sanitizer --tool synccheck` [CUDA Toolkit] detects warp-level synchronization errors at runtime, including mismatched masks in `__shfl_sync` calls. This is the closest dynamic analysis tool to our static approach. `synccheck` catches bugs post-compilation via instrumented execution; our types prevent them pre-compilation. The approaches are complementary: `synccheck` catches bugs in code that cannot be rewritten in our type system (legacy CUDA, inline PTX), while our types prevent the entire bug class at compile time for new code.

### Scheduling Languages: Halide and TVM

Halide [Ragan-Kelley et al., PLDI 2013] and TVM [Chen et al., OSDI 2018] separate algorithm from execution schedule, generating GPU code from high-level specifications. Both have encountered the shuffle-divergence bug in generated code (TVM#17307 is Bug 15 in our survey). Their scheduling model could in principle avoid generating divergent shuffles — a design-level alternative to type-checking them. We cite these as evidence that the bug class affects compiler-generated code, not just hand-written kernels.

### Recent SIMT Verification (2023)

Gu et al. [OOPSLA 2023] present lockstep execution semantics for SIMT programs and verify safety properties including convergence requirements. Their approach models the hardware execution semantics directly, while ours abstracts it into a type system. The two approaches differ in expressiveness (they model the full SIMT execution model; we model only the active-set fragment) and usability (they require a separate verification pass; ours integrates with the compiler via trait resolution).

### Hazy Megakernel (2025)

The Hazy megakernel [Stanford 2025] fuses ~100 operations into a single persistent-thread kernel with an on-GPU interpreter, maintaining warp-uniform execution—all 32 lanes execute the same operation, every shuffle uses `MASK_ALL`. This is safe but restrictive. Our type system is strictly more permissive: uniform programs type-check trivially (as Hazy's would), while lane-heterogeneous programs become expressible with explicit type annotations. We make divergence *safe* rather than *forbidden*.

## 8.6 Summary

| Related Work | Focus | Our Difference |
|--------------|-------|----------------|
| Descend | Memory safety | We do divergence safety |
| GPUVerify | External verification | We use types |
| MPST | All parties active | We model quiescence |
| ISPC | uniform/varying + runtime active set | We track active sets at compile time, not runtime |
| Futhark | Avoids divergence | We embrace + type it |
| `__shfl_sync` | Require mask (runtime) | We prove mask correct (compile-time) |
| Cooperative Groups | Runtime group objects | We use compile-time type parameters |
| Vulkan/SPIR-V subgroups | "Undefined if not converged" | We make diverged shuffle absent from the type |
| AMD DPP / Intel subgroups | Hardware-masked operations | We add type-level tracking portable across vendors |
| synccheck | Runtime detection | We prevent at compile time |
| Halide / TVM | Scheduling avoids bad code | We type-check divergent code directly |
| Gu et al. (OOPSLA 2023) | Lockstep verification | We integrate with the compiler via traits |
| Hazy megakernel | Prohibit divergence | We make divergence safe |
| DPJ | Determinism | We do lane safety |
| Rust ownership | Memory | We do active sets |

**Our unique contribution**: Linear typestate for active lane masks, with a complement lattice ensuring safe divergence and reconvergence. No prior work types the active lane mask. The structural analogy to session type branching motivates the design; the technical mechanism is typestate over a Boolean lattice, not session types.
