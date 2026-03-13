# 8. Related Work

Session-typed divergence draws on and differs from work in GPU verification, session types, and type systems for parallelism.

## 8.1 GPU Verification

### Descend (PLDI 2024)

Descend [Kopcke et al. 2024] brings Rust-style ownership and borrowing to GPU programming, preventing data races and use-after-free in GPU code.

**Relationship to our work**: Descend and session-typed divergence are *orthogonal* and *composable*:
- Descend: memory safety (ownership, borrowing, lifetimes)
- Our work: divergence safety (active lane tracking)

Descend does not track which lanes are active. A shuffle in Descend may read from inactive lanes if the programmer gets the mask wrong. Conversely, our system does not track memory ownership.

The ideal system combines both: lanes must be active (our contribution) *and* have valid data (Descend's contribution).

### GPUVerify

GPUVerify [Betts et al. 2012, 2015] is a static verification tool for GPU kernels. It uses predicated execution semantics and barrier invariants to prove race-freedom.

**Relationship to our work**: GPUVerify is an *external verifier*, not a type system:
- Separate tool run after compilation
- Provides yes/no answer (not incremental feedback)
- Heavyweight (SMT solving, can timeout)

Our approach integrates verification into the type system:
- Immediate feedback during editing
- Incremental (errors as you type)
- Lightweight (type checking is fast)

GPUVerify does not specifically target shuffle-from-inactive-lane bugs, though its predicated semantics could potentially catch them.

### CUDA Sanitizers

NVIDIA provides sanitizers (compute-sanitizer) for detecting GPU errors at runtime:
- Memory errors (memcheck)
- Race conditions (racecheck)
- Synchronization issues (synccheck)

**Relationship to our work**: Sanitizers are *dynamic* analysis:
- Catch bugs during testing, not compilation
- Only find bugs in executed paths
- Significant runtime overhead (significant [NVIDIA Compute Sanitizer documentation])

Our approach catches bugs at compile time, before any execution.

### GPU Race Detection

GMRace [Zheng et al. 2014] and CURD [Peng et al. 2018] detect warp-level data races using static analysis and dynamic instrumentation, respectively.

**Relationship to our work**: Race detection focuses on data races (concurrent conflicting accesses to shared memory), not divergence bugs (reading from inactive lanes via shuffle). These are related but distinct bug classes.

### LLVM Uniformity and Divergence Analysis

LLVM implements uniformity analysis that determines whether SSA values are uniform (same across all threads in a warp) or divergent. This analysis propagates divergence along def-use chains and control dependencies, supporting irreducible control flow.

**Relationship to our work**: LLVM's divergence analysis and our type system track related information but differ fundamentally. LLVM's analysis is a compiler pass—intraprocedural, best-effort, focused on optimization (avoiding unnecessary predication). Our type system is source-level, modular across function boundaries, and focused on safety. LLVM's analysis identifies *which* values are divergent but does not track *which lanes are active*. Our active-set types capture exactly this distinction. Bug 4 (LLVM#155682) demonstrates that LLVM's own optimizations can cause the bug class we prevent.

## 8.2 Session Types

### Binary Session Types

Session types were introduced by Honda [1993] for the π-calculus. A session type describes a communication protocol between two parties.

**Relationship to our work**: Binary session types assume two active parties. GPU divergence involves up to 32 parties where any subset may be inactive. We extend the model with *quiescence*.

### Multiparty Session Types (MPST)

Honda, Yoshida, and Carbone [2008] extended session types to multiple parties. Each party follows a local type projected from a global protocol.

**Relationship to our work**: MPST assumes all parties remain active (or fail). GPU divergence has parties *go quiescent and resume*. This is the key novelty:

| MPST | Our Extension |
|------|---------------|
| N parties, all active | 32 parties, subset active |
| Party fails = session stuck | Party quiesces = temporarily inactive |
| No reconvergence | Merge = reconvergence point |

### Gradual Session Types

Gradual session types [Igarashi et al. 2017] allow mixing static and dynamic typing for sessions. Unknown types are checked at runtime.

**Relationship to our work**: Our Layer 4 (existential types, §5.2) and `DynWarp` gradual typing bridge (§9.4) are directly inspired by this work. Our `ascribe()` operation corresponds to the cast at the gradual typing boundary.

### Fault-Tolerant Multiparty Session Types

Recent work extends MPST to handle participant failures: crash-stop failures [Adameit et al. 2022] and fault-tolerant event-driven programming [Viering et al. 2021].

**Relationship to our work**: Fault-tolerant MPST models *permanent* failure (crash-stop). GPU divergence involves *temporary* quiescence—lanes go inactive and resume at merge. Crash-stop requires protocol recovery; quiescence requires complement proof. The two extensions are complementary.

### Session Types Embedded in Rust (Ferrite)

Ferrite [Chen et al. 2022] embeds session types in Rust using PhantomData, zero-sized types, and type-level programming—the same encoding techniques we use.

**Relationship to our work**: Ferrite models inter-process communication channels; we model intra-warp lane communication with quiescence. Key differences: Ferrite's channels carry data (ours share a register file), Ferrite's session types describe message sequences (ours describe active-set evolution). The shared encoding validates that Rust's type system is expressive enough for session-type embeddings.

### Session Types for Concurrent Objects

Dardha et al. [2017] apply session types to concurrent objects in object-oriented languages.

**Relationship to our work**: They model object-to-object communication. We model lane-to-lane communication within a warp. The synchronization models differ: objects are asynchronous; warps are lock-step.

## 8.3 Type Systems for Parallelism

### Futhark

Futhark [Henriksen et al. 2017] is a functional GPU language with a type system that guarantees regular parallelism.

**Relationship to our work**: Futhark *avoids* divergence by design. Its parallelism constructs (map, reduce, scan) don't support divergent branches.

Our approach is complementary: we *embrace* divergence and make it safe. This allows expressing algorithms like adaptive sorting where divergence is fundamental.

### DPJ (Deterministic Parallel Java)

DPJ [Bocchino et al. 2009] uses region types to ensure determinism in parallel Java programs.

**Relationship to our work**: DPJ focuses on determinism through effect typing. Our focus is different: we ensure safety of warp-level communication, not determinism.

### Æminium

Æminium [Stork et al. 2014] is an implicitly parallel language with a permission system based on access permissions.

**Relationship to our work**: Æminium extracts parallelism from sequential code. We type explicitly parallel GPU code. The goals are opposite: they hide parallelism; we expose it.

### Data-Race-Free Type Systems

Several systems ensure data-race freedom through types [Boyapati et al. 2002, Flanagan and Freund 2000].

**Relationship to our work**: Race-freedom and divergence safety are distinct:
- Race: two threads access same location, at least one writes
- Divergence bug: one thread reads from inactive lane's register

Our active set types are not about preventing races—they're about preventing reads from inactive lanes.

## 8.4 Linear and Affine Types

### Ownership Types (Rust)

Rust's ownership system [Matsakis and Klock 2014] ensures memory safety through affine types (values used at most once).

**Relationship to our work**: We leverage Rust's type system for our implementation. The `Warp<S>` type is linear—consumed by diverge, produced by merge. This prevents use-after-diverge.

### Linear Logic and Session Types

The connection between linear logic and session types [Caires and Pfenning 2010, Wadler 2012] provides a logical foundation for session types.

**Relationship to our work**: Our warp linearity follows this tradition. Diverge corresponds to ⊗ (tensor) producing two resources; merge corresponds to ⅋ (par) consuming two resources.

## 8.5 GPU Programming Models

### CUDA and OpenCL

CUDA [NVIDIA 2007] and OpenCL [Khronos 2009] are the dominant GPU programming models. Both expose warp/wavefront primitives but provide no type-level safety.

**Relationship to our work**: We build on top of these models, adding a typed layer. Our types can wrap CUDA intrinsics.

### SYCL and oneAPI

SYCL [Khronos 2020] and Intel's oneAPI provide modern C++ abstractions for heterogeneous programming.

**Relationship to our work**: These aim for portability and productivity but do not address divergence safety. Our approach could be integrated into SYCL's sub-group operations.

### HIP

AMD's HIP [AMD 2016] is largely CUDA-compatible. Our approach applies equally to HIP's wavefront primitives.

### Cooperative Groups

CUDA's Cooperative Groups [NVIDIA 2017] provide a unified interface for thread groups at all levels.

**Relationship to our work**: Cooperative Groups make group membership explicit but don't provide type safety. A thread can still shuffle on a group where some threads have diverged. We provide the missing types.

### NVIDIA's `__shfl_sync` Migration (CUDA 9.0)

NVIDIA deprecated the original `__shfl` family in CUDA 9.0, replacing it with `__shfl_sync` which requires an explicit mask parameter [CUDA Programming Guide §10.22]. This was a vendor acknowledgment that the bug class is severe enough to warrant a breaking API change across the ecosystem. However, the mask remains a runtime value—programmers can still pass the wrong mask, as documented bugs in NVIDIA's own cuda-samples and CUB demonstrate.

**Relationship to our work**: `__shfl_sync` addresses the problem at the API level (require a mask). We address it at the type level (prove the mask correct). The approaches are complementary: `__shfl_sync` prevents *forgetting* the mask; our types prevent *getting it wrong*.

### Hazy Megakernel (2025)

The Hazy megakernel [Stanford 2025] is the most sophisticated persistent thread program as of this writing, fusing ~100 operations into a single kernel with an on-GPU interpreter.

**Relationship to our work**: Hazy avoids the divergence problem by maintaining warp-uniform execution—all 32 lanes execute the same operation, every shuffle uses `MASK_ALL`. This is safe but restrictive. Our type system is strictly more permissive: uniform programs type-check trivially (as Hazy's would), while lane-heterogeneous programs become expressible with explicit type annotations. We make divergence *safe* rather than *forbidden*.

## 8.6 Summary

| Related Work | Focus | Our Difference |
|--------------|-------|----------------|
| Descend | Memory safety | We do divergence safety |
| GPUVerify | External verification | We use types |
| MPST | All parties active | We model quiescence |
| Futhark | Avoids divergence | We embrace + type it |
| `__shfl_sync` | Require mask (runtime) | We prove mask correct (compile-time) |
| Hazy megakernel | Prohibit divergence | We make divergence safe |
| DPJ | Determinism | We do lane safety |
| Rust ownership | Memory | We do active sets |

**Our unique contribution**: Session types extended with quiescence for SIMT divergence. No prior work types the active lane mask.

