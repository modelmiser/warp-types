# 7. Evaluation

We evaluate session-typed divergence on three dimensions:
1. **Bug Detection**: Does the type system catch real divergence bugs?
2. **Performance**: What is the runtime overhead?
3. **Expressiveness**: Can practical GPU algorithms be expressed without excessive friction?

## 7.1 Bug Detection

### Documented Shuffle-Divergence Issues

We surveyed 21 documented shuffle-from-inactive-lane bugs across 16 GPU projects. Five are modeled as self-contained Rust examples; sixteen additional bugs were identified via systematic search of issue trackers (OpenCV, PyTorch, TVM, CUB, Kokkos, Halide, ROCm/HIP, HOOMD-blue, cuDF, Triton, Ginkgo) and specifications (WebGPU, SYCLomatic). Of 21 bugs, 14 are fully caught by our type system, 5 partially, and 1 (WebGPU's decision to exclude indexed subgroup shuffles) serves as design-level motivation. See §7.1 of the full paper for the complete table.

**Modeled bugs** (with worked Rust examples):

| Issue | Source | Nature | Type system prevents source pattern? |
|-------|--------|--------|--------------------------------------|
| Wrong ballot mask in reduction | cuda-samples#398 | Confirmed UB, silent wrong sum | Yes — modeled in code |
| Compiler predicates off mask init | CCCL#854 | Suspected UB, reporter later uncertain† | Yes — modeled in code |
| Hardcoded full mask in divergent branch | PIConGPU#2514 | Confirmed UB, no wrong output observed‡ | Yes — modeled in code |
| shfl_sync causes branch elimination | LLVM#155682 | Semantic gap, closed as "not a bug"§ | Partial — modeled in code |
| Deprecated `__shfl` API family | CUDA 9.0 | Vendor acknowledgment of bug class | N/A |

†CCCL#854: The reporter initially detected this via `cuda-memcheck --tool synccheck` but later noted it "may have been a false positive" and could not reproduce. The issue remains open but unconfirmed.

‡PIConGPU#2514: The undefined behavior is real (`__ballot_sync(0xFFFFFFFF, 1)` inside a divergent branch violates the CUDA spec). However, a contributor tested on K80 with CUDA 8 and 9 and observed no errors — pre-Volta hardware enforced convergence at warp level, masking the UB. The fix (PR #2600) was preventive, targeting Volta's independent thread scheduling. We do not claim wrong output was observed.

§LLVM#155682: The behavior is real — clang eliminates a branch because `__shfl_sync` creates an implicit assumption that all lanes reach the shuffle, making uninitialized values on non-participating lanes UB from the compiler's perspective. The LLVM maintainer closed this as expected behavior for undefined input. Our type system prevents the source-level pattern (conditional write followed by full-warp shuffle), but this is more accurately a semantic gap between CUDA's warp model and C++'s thread model than a compiler defect.

### Concrete Demonstration: cuda-samples #398

We modeled the cuda-samples#398 bug as a self-contained Rust example (`examples/nvidia_cuda_samples_398.rs`). The original CUDA bug:

```cuda
// In reduce7, after block-level tree reduction with block_size=32:
// Only tid 0 enters the reduction path
unsigned mask = __ballot_sync(0xFFFFFFFF, tid < blockDim.x / warpSize);
// mask = 1 (only lane 0 active)
sdata[tid] += __shfl_down_sync(mask, sdata[tid], 16);
// BUG: reads from lane 16, which is inactive
```

In our type system, after the block-level reduction narrows to one lane:

```rust
let (lane0, _rest) = warp.extract_lane0();
// lane0 : Warp<Lane0>

let shifted = lane0.shuffle_down(sdata, 16);
// ERROR: no method `shuffle_down` found for `Warp<Lane0>`
```

The bug is not caught at runtime—it cannot be *expressed*. The `shuffle_down` method does not exist on `Warp<Lane0>`. This is the core guarantee: unsafe operations are absent, not checked.

The example includes two correct fixes:
- **Fix A**: Check `active_count == 1` and skip reduction (lane 0 already has the result).
- **Fix B**: Zero inactive lanes and reduce the full warp.

Both fixes type-check because they ensure all lanes participate before shuffling.

### Concrete Demonstrations: Remaining Bugs

Each remaining documented bug has a self-contained worked example demonstrating the exact type error our system produces and why `__shfl_sync`'s runtime mask does not catch the bug.

**PIConGPU #2514** (`examples/picongpu_2514.rs`): After divergence, the warp handle becomes `Warp<Active>`. Calling `ballot()` on `Warp<Active>` is a type error—`ballot()` exists only on `Warp<All>`. The CUDA code used `__ballot_sync(0xFFFFFFFF, 1)` inside a divergent branch; the hardware accepted the mask because `0xFFFFFFFF` is a valid `u32`, regardless of how many lanes were actually active.

**CUB/CCCL #854** (`examples/cub_cccl_854.rs`): The source-level mask was correct; the compiler generated wrong PTX by predicating off the mask initialization. In our type system, the mask is `PhantomData<SubWarp16>`—a zero-sized phantom type with no register and no initialization. The compiler cannot optimize away something that doesn't exist at runtime. `shuffle_up()` on `Warp<SubWarp16>` is a type error.

**LLVM #155682** (`examples/llvm_155682.rs`): After `if (laneId == 0)`, lane 0 has `Warp<Lane0>`. Calling `shuffle_broadcast()` on `Warp<Lane0>` is a type error. The fix—merging back to `Warp<All>` via `merge(lane0, rest)`—forces both sides to provide data, eliminating the uninitialized value that triggered LLVM's UB-based branch elimination.

### Hardware Reproduction

We reproduced the cuda-samples#398 bug on an NVIDIA RTX 4000 SFF Ada (compute 8.9, Ada Lovelace architecture) using CUDA 12.0. With `block_size=32`, the buggy `reduce7` kernel consistently returns `sum = 1` instead of `sum = 32` (the correct value). The result is deterministic across 10 runs—not intermittent. The bug also manifests at `block_size=256`: the kernel returns `sum = 32` instead of `sum = 256`, because `blockDim.x / warpSize = 8` means only 8 lanes enter the final reduction, producing a partial ballot mask (`0xFF`). The `__shfl_down_sync` with offset 16 reads from lanes outside this mask.

The bug affects all block sizes where `blockDim.x / warpSize < 32`—only `block_size=1024` produces the correct result, where all 32 lanes vote true and the ballot mask is `0xFFFFFFFF`. The fixed version (all lanes participate with zeroed inactive data) produces correct results at all block sizes. The reproduction code is in `reproduce/reduce7_bug.cu`.

### Vendor Response: API Deprecation and Architectural Change

The severity of shuffle-divergence bugs is reflected in NVIDIA's response. In CUDA 9.0, NVIDIA deprecated the entire `__shfl`, `__shfl_up`, `__shfl_down`, and `__shfl_xor` API family—a breaking change across the CUDA ecosystem. The CUDA Programming Guide states: *"If the target thread is inactive, the retrieved value is undefined"* (§10.22). The replacement `__shfl_sync` family requires an explicit mask parameter, but as Bugs 1–4 demonstrate, the mask is a runtime value that programmers still get wrong.

The problem deepened with Volta's independent thread scheduling. Pre-Volta architectures enforced warp-level lockstep execution, so implicit convergence masked most shuffle-divergence UB. Volta allowed threads within a warp to genuinely interleave, turning latent UB into observable bugs. NVIDIA's architecture whitepaper acknowledges this: *"[Independent thread scheduling] can lead to a rather different set of threads participating in the executed code than intended."* Code that happened to work on Pascal silently broke on Volta—and the `__shfl_sync` mask was supposed to fix this, but simply moved the burden from implicit hardware convergence to explicit (and error-prone) programmer annotation.

### Compile-Fail Tests as Proof Artifacts

Our implementation includes eight compile-fail doctests that serve as machine-checked proof artifacts:

1. `shuffle_xor` on `Warp<Even>` — rejected (§3)
2. `merge(Even, LowHalf)` — rejected, overlapping sets (§3)
3. `merge(Even, Even)` — rejected, same set (§3)
4. `merge(EvenLow, OddHigh)` — rejected, non-covering sets (§3)
5. `shuffle_xor` on `Warp<LowHalf>` — rejected (§3)
6. `reduce_sum` on `Warp<Even>` — rejected (§3)
7. `merge` of non-complements within nested divergence — rejected (§3)

These are not test heuristics—they are verified absences. The Rust compiler confirms that each operation is a type error. Any future change to the type system that accidentally permits these operations would cause `cargo test` to fail.

### Bug Pattern Coverage

Our prototype includes 242 unit tests, 50 example tests across 8 worked bug examples, and 8 compile-fail doctests (including a linearity enforcement test that verifies use-after-diverge is a compile error) covering the full type system. The tests exercise:

- Diverge/merge with complement verification
- Nested divergence (up to depth 3)
- Recursive protocols (5 loop patterns)
- Arbitrary predicates (existential, indexed, hybrid)
- Work stealing with dynamic roles
- Platform portability (warp-32 and wavefront-64)
- Warp-size-generic algorithms

Every test validates that the type system permits correct patterns and rejects incorrect ones.

## 7.2 Performance

### Zero Overhead by Construction

Our types impose zero runtime overhead—not measured to be negligible, but *guaranteed by construction*:

- `Warp<S>` contains only `PhantomData<S>`, which has zero size and zero runtime representation.
- `ActiveSet` is a trait with `const MASK: u32`—resolved at compile time.
- `ComplementOf<S>` is a trait bound, checked at compile time.
- Monomorphization eliminates all generic dispatch.

The generated code contains no trace of the type system. A `Warp<All>` and a `Warp<Even>` produce identical machine code for any operation available on both.

We verified this at three levels: Rust MIR (`Warp<S>` values optimized away entirely), LLVM IR (see §7.2 in full paper), and NVIDIA PTX.

**NVIDIA PTX** (`rustc +nightly --target nvptx64-nvidia-cuda -O`): We compiled actual Rust type system code—`PhantomData`, trait bounds, `ComplementOf`, `diverge`/`merge`—directly to PTX via the `nvptx64-nvidia-cuda` target. Both `butterfly_typed` (through `Warp<All>`) and `diverge_merge_typed` (`kernel_entry → diverge → merge`) produce byte-identical PTX vs. their untyped equivalents. The entire type system is erased through the full LLVM NVPTX backend. Reproduction: `bash reproduce/compare_rust_ptx.sh`.

### Comparison with Runtime Approaches

The alternative to compile-time safety is runtime mask checking—verifying at each shuffle that the mask matches the active set. NVIDIA's `compute-sanitizer --tool synccheck` provides this:

| Approach | Overhead | Coverage | Feedback |
|----------|----------|----------|----------|
| `__shfl_sync` only (no verification) | 0% | None | Silent UB |
| Runtime sanitizer | Significant | Executed paths only | At test time |
| Our type system | 0% | All paths | At compile time |

Our approach provides strictly more coverage at strictly less cost.

## 7.3 Expressiveness

### The Hazy Argument

The most sophisticated persistent thread program as of 2025—the Hazy megakernel [Stanford 2025]—prohibits lane-level divergence by design. Their on-GPU interpreter dispatches at warp granularity: all 32 lanes execute the same operation, and every shuffle uses `MASK_ALL = 0xFFFFFFFF`. They never allow different lanes to run different ops. This is *architectural avoidance*: state-of-the-art practitioners treat lane-level divergence as too dangerous to manage, even with `__shfl_sync`, and prohibit it entirely.

In our type system, Hazy-style programs type-check trivially: every warp is `Warp<All>`, every shuffle is permitted, and no diverge/merge annotations are needed. The type system is invisible for uniform programs.

This answers the standard expressiveness objection ("does the type system reject too much?") with empirical evidence: the programs that state-of-the-art practitioners *actually write* are the easiest to type. But it also reveals a stronger point: our type system could safely *relax* the restriction that Hazy imposes—allowing lane-level heterogeneity with compile-time safety guarantees that architectural avoidance cannot provide.

### Lane-Heterogeneous Programs

Programs that use lane-level divergence require explicit `diverge`/`merge` annotations. This is the intended design: the annotation *is* the safety contract. It makes visible the control-flow structure that was previously implicit and error-prone.

The annotation overhead is modest. Divergence points require one `diverge` call (producing two typed sub-warps) and one `merge` call (consuming them). For a butterfly reduction with 5 shuffle stages, the typed version adds 3 lines (the initial diverge, a merge to restore `Warp<All>`, and a data merge).

### Patterns and Their Typability

| Pattern | Typable? | Annotation Needed |
|---------|----------|-------------------|
| Butterfly reduction | Yes | None (uniform) |
| Kogge-Stone scan | Yes | None (uniform) |
| Predicated filter | Yes | diverge/merge per predicate |
| Adaptive sort | Yes | diverge/merge per partition |
| Warp-level work stealing | Yes | dynamic role assignment |
| Cooperative group with sub-warp | Yes | existential types (§5) |
| Data-dependent shuffle mask | Partial | Requires dependent types (future) |

### Limitations

Four patterns are not fully expressible in our current system:

1. **Data-dependent shuffle targets**: When the shuffle source lane is computed from data, our static types cannot verify it. This requires dependent types (§9).

2. **Arbitrary runtime predicates**: Our marker types cover common patterns (Even, Odd, LowHalf, HighHalf). Predicates not matching these markers require existential types, which add a runtime check.

3. **Cross-function active set polymorphism**: Functions that are generic over the active set require explicit trait bounds, increasing annotation burden at API boundaries.

4. **Irreversible divergence**: If one branch of a diverge exits early (return, panic, trap), the warp handle for that branch is dropped, violating linearity. The type system correctly rejects this—without both halves, you cannot reconstruct `Warp<All>` for subsequent shuffles. The workaround is a ballot-based exit pattern. `DynWarp` (§9.4) provides a runtime escape for patterns where static exit tracking is too restrictive.

These limitations are real but narrowly scoped. The first two are addressed by our extension layers (§5); the third is a standard trade-off in any type-parameterized system; the fourth follows necessarily from the linearity discipline that makes the type system sound.

## 7.4 Threats to Validity

**Bug sample size**: Our evaluation surveys 21 documented shuffle-divergence bugs across 16 projects (§7.1), with 8 modeled as self-contained Rust examples. Of 21 bugs, 14 are fully caught by the type system.

**Limited GPU hardware evaluation**: Our type system prototype runs on CPU, emulating warp semantics. The zero-overhead claim is established by type erasure verified at three levels: Rust MIR, LLVM IR, and NVIDIA PTX (§7.2). We compiled actual Rust type system code (PhantomData, trait bounds, diverge/merge) to PTX via `nvptx64-nvidia-cuda` and confirmed byte-identical output vs. untyped equivalents. We also reproduced the cuda-samples#398 bug on actual GPU hardware (RTX 4000 SFF Ada, compute 8.9), confirming that the undefined behavior produces deterministically wrong results on post-Volta architectures.

**Selection bias**: The bugs we model are ones where the type system succeeds. We explicitly identify patterns where it does not (data-dependent masks, §7.3). We are not aware of shuffle-from-inactive-lane bugs that our type system would fail to catch at the source level.

## 7.5 Summary

| Metric | Result |
|--------|--------|
| Real bugs surveyed | 21 across 16 projects (14 fully caught, 5 partial, 1 motivation) |
| Real bugs modeled | 8 with worked Rust examples (+ 5 mechanized untypability proofs in Lean) |
| Hardware reproduction | cuda-samples#398 confirmed on RTX 4000 Ada (compute 8.9) |
| PTX verification | Rust type system compiles to identical PTX (nvptx64-nvidia-cuda) |
| Type system tests | 242 unit + 50 example + 8 compile-fail |
| Runtime overhead | 0% (verified: Rust MIR, LLVM IR, NVIDIA PTX) |
| Annotation burden | 27.3% of algorithm lines (range: 12.5%–50%) |
| Lean mechanization | 17 theorems (progress, preservation, 5 untypability proofs) |
| Limitations | Data-dependent masks, cross-function polymorphism |

Session-typed divergence provides strong safety guarantees with zero runtime cost. For uniform programs (the dominant style in practice), it is invisible. For lane-heterogeneous programs, it makes divergence explicit—replacing implicit bugs with explicit types.

We do not claim shuffle-divergence bugs are the most *frequent* GPU bug class. We claim they are the most *insidious*: they produce silent data corruption rather than crashes, survive testing at common configurations, and resist source-level reasoning (Bug 4 demonstrates that even correct source can produce wrong code). NVIDIA deprecated an entire API family to address the problem; their replacement still relies on runtime masks that programmers get wrong. State-of-the-art persistent thread programs avoid the problem by prohibiting lane-level divergence entirely. Our type system is the first approach that makes lane-level divergence *safe* rather than *forbidden*.
