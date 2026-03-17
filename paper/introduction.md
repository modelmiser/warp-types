# 1. Introduction

GPU programming has become essential for high-performance computing, machine learning, and graphics. Modern GPUs achieve their performance through *SIMT execution*: groups of 32 or 64 threads called *warps* execute in lockstep, sharing a single instruction stream while operating on different data. When threads in a warp take different control-flow paths—a phenomenon called *divergence*—some threads become inactive while others continue executing.

Divergence creates a pernicious class of bugs. Consider the following CUDA code:

```cuda
__device__ int conditional_exchange(int data, bool participate) {
    if (participate) {
        // Only some threads reach here
        int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);
        return data + partner;
    }
    return 0;
}
```

This code compiles without warnings and may appear to work correctly in testing. But it contains undefined behavior: the shuffle operation reads values from *all* lanes, including those where `participate` is false. Those threads are inactive—their registers may contain stale data, garbage, or trap values. The result is non-deterministic: sometimes correct, sometimes wrong, sometimes a crash.

This bug pattern is not hypothetical. NVIDIA's own `cuda-samples` repository contains a shuffle-mask bug in its reference parallel reduction (`reduce7`): when launched with one block of 32 threads, only lane 0 enters the final reduction, but `__shfl_down_sync` reads from lane 16, which is inactive [cuda-samples#398]. The result is a silently wrong sum—no crash, no error—at a configuration most test suites skip. NVIDIA's core primitives library CUB has an open issue suggesting compiler optimizations may predicate off mask initialization in sub-warp configurations [CCCL#854]—the reporter later noted it may have been a false positive, but the source-level pattern is ill-typed in our system regardless. The PIConGPU plasma physics simulation ran for months on K80 GPUs with `__ballot_sync(0xFFFFFFFF, ...)` inside a divergent branch—real undefined behavior that went undetected because pre-Volta hardware enforced warp-level convergence, masking the violation [PIConGPU#2514]. An LLVM issue illustrates the problem extending into compiler optimization: `__shfl_sync` after a conditional causes the compiler to eliminate the branch entirely, running a lane-0-only atomic on all 32 lanes [LLVM#155682].

The bug class is severe enough that NVIDIA deprecated the entire `__shfl` API family in CUDA 9.0, replacing it with `__shfl_sync` which requires an explicit mask parameter [CUDA Programming Guide §10.22]. But the mask is still a runtime value—and as the bugs above demonstrate, programmers get it wrong. Volta's independent thread scheduling made the situation worse: code that was latent undefined behavior on Pascal became observable bugs when threads within a warp could genuinely interleave.

State-of-the-art practitioners have responded by avoiding the problem entirely. The Hazy megakernel [2025], the most sophisticated persistent thread program as of this writing, prohibits lane-level divergence by design—all 32 lanes execute the same operation, every `__shfl_sync` uses the full mask. Divergence avoidance is reinforced by performance: divergent warps serialize execution across branches, so warp-uniform execution also maximizes SIMD throughput. This works, but at the cost of expressiveness: algorithms that naturally benefit from lane-level heterogeneity (adaptive sorting, predicated filters, work-stealing within a warp) cannot be expressed in this style.

Our type system offers a third path: lane-level divergence that is safe rather than forbidden. It is strictly more permissive than the divergence-prohibition approach exemplified by Hazy (which maintains warp-uniform execution) while being strictly safer than CUDA's `__shfl_sync` API (which defers mask correctness to runtime). The gap between `__shfl_sync` and compile-time safety is concrete: `__activemask()` returns the current execution mask, which hardware always accepts—but a shuffle using `__activemask()` inside divergent code silently communicates among the wrong subset of lanes [CUDA Programming Guide §K.6]. Our type system closes this gap because the active set is a type-level property, not a runtime value.

## 1.1 Our Approach: Linear Typestate for Divergence

We observe that divergence has the structure of a *linear resource protocol*. When a warp diverges, it splits into two sub-warps whose active sets partition the original—analogous to branching in a multiparty session type, but with a crucial difference: there are no channels, no directed messages, and no protocol sequencing. Instead, some participants go *quiescent* (temporarily inactive), and the type system tracks this as a set-level property rather than a communication protocol.

The analogy to session types is motivating: diverge resembles protocol branching, merge resembles session joining, and the complement requirement resembles compatibility. But the technical mechanism is different—we use *linear typestate* over a Boolean lattice of active sets, not session types proper.

We introduce *warp typestate*, a linear type system that tracks which lanes are active at each program point. The key ideas are:

1. **Warps carry active set types.** A warp is typed as `Warp<S>` where `S` describes which lanes are active. `Warp<All>` means all 32 lanes; `Warp<Even>` means only even-numbered lanes.

2. **Divergence produces complementary sub-warps.** When a warp diverges on a predicate, it produces two sub-warps with disjoint active sets that together cover the original:
   ```
   diverge : Warp<S> → (Warp<S ∩ P>, Warp<S ∩ ¬P>)
   ```

3. **Merge requires complement proof.** To reconverge, the type system must verify that the two sub-warps are complements:
   ```
   merge : Warp<S₁> → Warp<S₂> → Warp<S₁ ∪ S₂>  where S₁ ∩ S₂ = ∅
   ```

4. **Operations restrict by active set.** Shuffle operations are only available on `Warp<All>`. You cannot shuffle on a diverged warp—the method simply doesn't exist.

With this type system, the buggy code above becomes a compile error:

```rust
fn conditional_exchange(warp: Warp<All>, data: PerLane<i32>, participate: PerLane<bool>) -> i32 {
    let (active, inactive) = warp.diverge(|lane| participate[lane]);

    // ERROR: no method `shuffle_xor` found for `Warp<Active>`
    // note: shuffle_xor requires Warp<All>
    // help: merge with complement before shuffling
    let partner = active.shuffle_xor(data, 1);

    // ...
}
```

The fix is explicit: merge back to `Warp<All>` before shuffling, ensuring all lanes have valid data:

```rust
fn conditional_exchange(warp: Warp<All>, data: PerLane<i32>, participate: PerLane<bool>) -> i32 {
    let (active, inactive) = warp.diverge(|lane| participate[lane]);

    // Inactive lanes contribute zero
    let active_data = data;
    let inactive_data = PerLane::splat(0);

    // Merge back - type system verifies complement
    let warp: Warp<All> = merge(active, inactive);
    let combined = merge_data(active_data, inactive_data);

    // Now shuffle is safe
    let partner = warp.shuffle_xor(combined, 1);  // OK: Warp<All>
    combined + partner
}
```

## 1.2 Contributions

We present a linear typestate system for intra-warp divergence that statically eliminates diverged shuffle and ballot operations. Well-typed programs cannot perform unsafe warp operations on inactive lanes. The guarantee is zero-overhead—enforcement is purely compile-time. We further extend the approach to intra-warp memory fence ordering (§5.6), where the same complement proof ensures all lanes have written before a fence executes.

This paper makes the following contributions:

1. **A novel type system for GPU divergence** (§3). We present the first type system that tracks active lane masks, preventing undefined behavior from reading inactive lanes. The type system uses linear typestate with a Boolean lattice of active sets, motivated by the structural analogy to multiparty session type branching.

2. **A soundness proof** (§4). We prove that well-typed programs satisfy progress and preservation, ensuring they never read from inactive lanes.

3. **A zero-overhead implementation** (§6). We implement our type system as a Rust library using traits and generics. Types are erased at compile time—the generated code is identical to hand-written unsafe CUDA.

4. **Practical patterns for GPU programming** (§5). We show how to type common GPU idioms including reductions, scans, and filters, and extend the core system to handle loops and arbitrary predicates.

5. **Empirical grounding** (§7). We document real shuffle-mask bugs in NVIDIA's reference code (cuda-samples, CUB) and scientific simulation (PIConGPU) that our type system would have caught at compile time, and demonstrate this concretely by modeling cuda-samples#398 as a runnable example where the bug is a type error.

## 1.3 The Bigger Picture

Warp typestate is one instance of a broader pattern: *participatory computation* where the set of active participants changes during execution. The transfer fidelity varies by domain: we have demonstrated a working prototype for FPGA crossbar protocols (§9.6), where the bug class is isomorphic; identified a partial transfer to distributed systems, where quiescence complements fault-tolerant session types; and noted structural similarity to database predicate filtering and proof case splits, though without actionable type-system transfer. We return to this in §9.

The remainder of this paper is organized as follows. §2 provides background on GPU execution and session types. §3 presents our core type system. §4 proves soundness. §5 extends the system to handle loops and arbitrary predicates. §6 describes our implementation. §7 evaluates bug detection and performance. §8 discusses related work. §9 sketches future directions and §10 concludes.
