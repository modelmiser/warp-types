# Empirical Evidence: Shuffle-from-Inactive-Lane Bugs

*Supporting material for §1 (Introduction) and §7 (Evaluation) of the paper.*

## Purpose

The paper's contribution statement — the lead paragraph of §1.2, whole (the
sync check fails if this block stops being that paragraph of `paper.md`):

<!-- quote:paper §1.2 -->
> We present a linear typestate system for intra-warp divergence that statically eliminates diverged shuffle and ballot operations. Well-typed programs that express divergence exclusively through the system's `diverge`/`merge` combinators cannot perform unsafe warp operations on inactive lanes; a raw data-dependent branch around a warp intrinsic remains outside the guarantee (§4.7, §6.9). For the static core the guarantee is zero-overhead—enforcement is purely compile-time (the gradual/dynamic layers of §5 check at runtime, by design). We further extend the approach to a model of an intra-warp fence discipline (§5.6), where the same complement proof ensures that the write steps recorded in the type cover the whole warp before a fence is typed; the region in that model is phantom, and the types observe no store.
<!-- /quote -->

This document provides the empirical grounding that reviewers will demand:
**the bug class is real, silent, and hard to prevent manually.**

We do NOT need to prove shuffle bugs are the *most common* GPU bug class.
We need to prove they are *real and non-trivial to avoid*.

---

## Evidence Tier 1: Existence Proofs (Specific Bugs)

### 1. NVIDIA cuda-samples #398 — Wrong Mask in Reference Reduction

**Source:** https://github.com/NVIDIA/cuda-samples/issues/398

**What:** NVIDIA's official `reduce7` sample passes `__ballot_sync()` result as
the mask to `__shfl_down_sync()`. When block size is 32 and only thread 0 enters
the reduction path, `ballot_result = 1`. Thread 0 calls
`__shfl_down_sync(1, mySum, 16)` — reading from lane 16, which is inactive.

**Failure mode:** Silent wrong sum. Only manifests at specific grid configurations
(grid(1,1,1), block(32,1,1)). No crash, no error. At typical launch configs the
mask happens to be correct.

**Significance:** NVIDIA's own reference code. The canonical sample developers
copy from.

### 2. CUB/CCCL #854 — Compiler Predicates Off Mask Initialization

**Source:** https://github.com/NVIDIA/cccl/issues/854

**What (as reported; unconfirmed):** CUB's `WarpScanShfl` sets `member_mask`
for `shfl_sync`. The reporter suspected that when threads exit a loop early the
compiler predicates off the mask initialization, leaving `member_mask`
uninitialized, so the `shfl.sync.idx.b32` PTX instruction executes with the
wrong mask. The reporter later noted the `synccheck` hit "may have been a false
positive" and could not reproduce; the issue remains open.

**Failure mode (if real):** Silent wrong scan results. Only with sub-warp
logical warp sizes (`LOGICAL_WARP_THREADS < 32`) inside early-exit loops. Found
via `cuda-memcheck --tool synccheck`, not by observing wrong output.

**Significance:** In CUB itself — the library virtually all CUDA reduction/scan
code depends on. Whether or not this instance is real, the source-level pattern
— a membermask computed by code the compiler may predicate — has no
counterpart in the typed API's shuffles, where the mask is a type and has no
register to predicate. (A raw intrinsic call, or a raw branch around one, remains
possible outside that API: the paper's §6.9.)

### 3. PIConGPU #2514 — Hardcoded Full Mask in Divergent Branch

**Source:** https://github.com/ComputationalRadiationPhysics/picongpu/issues/2514

**What:** `atomicAllInc()` in PMacc library calls
`__ballot_sync(0xFFFFFFFF, 1)` inside a conditional branch. Hardcoded full mask
doesn't match actual active threads when the warp is diverged.

**Failure mode:** Undefined behavior by the CUDA specification. No wrong output
was observed — pre-Volta hardware converged the warp and masked it — and the
fix (PR #2600) was preventive, targeting Volta's independent thread scheduling.
Found during CUDA 9 migration, not by observing incorrect results. Took 3 months
from report to fix (PR #2600).

**Significance:** Real scientific simulation. The undefined behaviour went
unnoticed for months; no wrong output was observed, but had it corrupted data,
published results were at stake.

### 4. LLVM #155682 — shfl_sync Causes Branch Elimination

**Source:** https://github.com/llvm/llvm-project/issues/155682

**What:** Clang compiling this pattern:
```cuda
if (laneId == 0) { row = atomicAdd(i, 16); }
row = __shfl_sync(0xffffffff, row, 0) + laneId;
```
eliminates the `if` entirely — `atomicAdd` runs on all 32 lanes. NVCC handles
it correctly. LLVM treats uninitialized `row` on non-lane-0 threads as UB and
"optimizes" by assuming the if-path is always taken.

**Failure mode:** Atomic counter advances 32x too fast. Silent, no warnings.
Only visible in generated PTX.

**Significance:** The shuffle-divergence problem extends into compiler
optimization. Source-level reasoning is insufficient.

---

## Evidence Tier 2: Vendor Acknowledgment

### 5. NVIDIA Deprecated Entire `__shfl` API Family (CUDA 9.0)

**Sources:**
- CUDA Programming Guide §10.22: *"__shfl, __shfl_up, __shfl_down, and
  __shfl_xor have been deprecated in CUDA 9.0 for all devices."*
- Blog "Using CUDA Warp-Level Primitives":
  *"implicit warp-synchronous programming is unsafe and may not work correctly"*
  and *"Because using them can lead to unsafe programs, the legacy warp-level
  primitives are deprecated starting in CUDA 9.0."*
- CUDA Programming Guide §10.22: *"If the target thread is inactive, the
  retrieved value is undefined."*

**What this means:** The GPU vendor concluded the bug class was severe enough to
require a breaking API change across the entire CUDA ecosystem. The old
`__shfl` gave no way to specify which threads participate. The new `__shfl_sync`
requires an explicit mask — but the mask is still a runtime value that can be
wrong (Bugs 1 and 3 are wrong masks; Bug 2 is a suspected one, unconfirmed;
Bug 4 is correct-looking source, undefined for the lanes that skip a branch,
which the compiler may legally transform).

### 6. Volta Independent Thread Scheduling Made It Worse

**Source:** Tesla V100 Architecture Whitepaper, §Independent Thread Scheduling

Pre-Volta: warps executed in lockstep, so `__shfl` without a mask happened to
work in most cases (implicit convergence).

Volta: threads within a warp can genuinely interleave. *"can lead to a rather
different set of threads participating in the executed code than intended."*

Code that was latent UB on Pascal could become observable bugs on Volta.

---

## Evidence Tier 3: Architectural Avoidance

### 7. Hazy Megakernel (2025) — Prohibits Lane-Level Divergence by Design

**Source:** https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
**Repo:** https://github.com/HazyResearch/Megakernels

**What:** The most sophisticated persistent thread program as of 2025 (fuses
~100 ops, 1.5-2.5x speedup, 78% memory bandwidth utilization). Uses warp
specialization with an on-GPU interpreter.

**Key finding:** They avoid intra-warp divergence entirely. The interpreter
dispatches at warp granularity — all 32 lanes execute the same operation. Every
`__shfl_sync` uses `MASK_ALL = 0xFFFFFFFF`. They never allow different lanes to
run different ops.

**What this means:** State-of-the-art practitioners treat lane-level divergence
as too dangerous to manage, even with `__shfl_sync`. They prohibit it
architecturally rather than handling it. Our type system could relax this
restriction as a discipline: with `diverge`/`merge`, full-warp collectives are
unavailable on a diverged handle at compile time. Running different operations
on different lanes still needs a branch that the typed fragment does not cover
(the paper's §6.9), so the execution model is not relaxed by the types alone.

---

## Argument Structure for the Paper

The evidence is used for the paper's closing claim in §7.5, quoted whole
(Evidence 5 is the deprecation it mentions, Evidence 7 — Hazy — the kind of
program that prohibits divergence; the CUB report, Bug 2, is unconfirmed and
the paragraph does not rely on it):

<!-- quote:paper §7.5 -->
> We do not claim shuffle-divergence bugs are the most *frequent* GPU bug class. We claim they are the most *insidious*: they produce silent data corruption rather than crashes, survive testing at common configurations, and resist source-level reasoning (Bug 4 demonstrates that correct-looking source — undefined for the lanes that skip a branch — can be compiled into code its author did not intend). NVIDIA deprecated an entire API family to address the problem; their replacement still relies on runtime masks that programmers get wrong. State-of-the-art persistent thread programs avoid the problem by prohibiting lane-level divergence entirely. Our type system is a *compile-time type discipline* that makes lane-level divergence checkable rather than forbidden, within the typed fragment (runtime and hardware mechanisms — CG static tiles, ISPC's compiler-managed masks — make particular forms safe by other means, and Descend statically rejects synchronization on conditional thread groups; §8).
<!-- /quote -->

---

## Scope Boundary (Explicit)

**In scope:** Full-warp collectives (shuffle, ballot, reductions) reached on a
diverged handle, in code that expresses its divergence through
`diverge`/`merge`. A `Warp<S>` records the lanes a handle claims, not the lanes
that execute, and a raw branch around an intrinsic is outside the typed
fragment (the paper's §6.9).

**Partially addressed:** Intra-warp fence-divergence interactions are modelled in §5.6 by a typestate over a phantom region (recorded write steps; the types observe no store). **Remaining out of scope:** Cross-warp fence interactions — where warp A diverges and warp B's fence depends on A's contribution via global memory. This is a cross-warp, cross-memory-domain problem that requires extending the type system beyond intra-warp active sets.

---

## Gupta & Stuart (2012) — Note on Scope

Gupta & Stuart's four persistent thread use cases (CPU-GPU sync, load balancing,
producer-consumer, global sync) predate warp shuffle instructions (introduced
in Kepler, 2012). Their communication primitives are atomics + shared memory +
global memory. The paper is citable for:
- "Persistent threads are hard" (general difficulty)
- "Debugging can prove to be an extremely challenging task" (§4.2)
- "Guaranteeing both data consistency and deadlock avoidance is non-trivial" (§4)

It is NOT citable for shuffle-safety evidence specifically. Use Bugs 1, 3 and 4
(Bug 2 only as a suspected report) and Evidence 5 above for that.
