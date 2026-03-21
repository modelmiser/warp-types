# 9. Future Work

Warp typestate opens several research directions.

## 9.1 Tooling and Data-Dependent Divergence

Our tooling stack includes two implemented proc macros and a build-time library:

1. **`warp_sets!`** (§6.1) generates the static active set hierarchy with compile-time validation of disjoint/covering invariants.
2. **`#[warp_kernel]`** transforms kernel functions into `extern "ptx-kernel"` entry points with `#[no_mangle]`, validating that parameters are GPU-compatible types (raw pointers or scalars).
3. **`WarpBuilder`** cross-compiles kernel crates to PTX via `cargo rustc --target nvptx64-nvidia-cuda -Z build-std=core`, finds the generated `.s` file, and produces a Rust module with a `Kernels` struct providing named `CudaFunction` handles.

Data-dependent predicates (e.g., `data[lane] > threshold`) are now supported via `diverge_dynamic(mask)`, which returns a `DynDiverge` — a paired divergence where the mask is runtime but the complement is structural. Both branches must merge to recover `Warp<All>`. No dependent types are required.

```rust
let warp: Warp<All> = Warp::kernel_entry();
let mask = ballot_result;  // runtime predicate
let diverged = warp.diverge_dynamic(mask);
// Can't shuffle on either branch
let warp: Warp<All> = diverged.merge();  // complement guaranteed by construction
warp.shuffle_xor(data, 1);  // OK — all lanes active
```

A future `#[warp_typed]` proc macro could further optimize this pattern by automatically inserting `diverge_dynamic` calls and tracking merge pairing at compile time, reducing boilerplate for complex data-dependent algorithms.

## 9.2 Formal Mechanization

Our core metatheory is fully mechanized in Lean 4 (§4.8): progress, preservation, and the substitution lemma are all machine-checked with zero `sorry` and zero axioms. Five bug untypability proofs are also mechanized. Nested divergence (`IsComplement s1 s2 parent`) and all four loop typing rules (§5.1: LOOP-UNIFORM, LOOP-CONVERGENT, LOOP-VARYING, LOOP-PHASED) are mechanized with full progress, preservation, and substitution coverage. Remaining future work:
- Verified Rust implementation via Aeneas translation
- Leverage prior Lean-based GPU verification work (MCL framework)

## 9.3 Protocol Inference and Gradual Typing

Our current system requires explicit type annotations. We have explored inference strategies in research prototypes — local inference (within functions), bidirectional checking (mix inference and annotation), and gradual typing — with 14 tests across five approaches (`src/research/protocol_inference.rs`).

The gradual typing approach is promoted to the public API (`src/gradual.rs`, 29 tests): `DynWarp` provides the same operations as `Warp<S>` but checks safety invariants at runtime instead of compile time. The migration path:

1. **Start dynamic**: `DynWarp::all()` — all operations runtime-checked
2. **Ascribe at boundaries**: `dyn_warp.ascribe::<All>()?` — runtime evidence becomes compile-time proof
3. **End static**: `Warp<S>` everywhere — zero-overhead, compile-time safety

`DynWarp` also handles the data-dependent predicate case (§9.1): when the active set depends on runtime data and cannot be expressed as a marker type, `DynWarp` provides runtime safety that `Warp<S>` cannot.

Remaining future work:
- Local inference integration into the public API (infer active sets within functions, require annotations only at boundaries)
- Protocol-first development (design protocol in DSL, generate/check code against it)

## 9.4 Beyond SIMT

The core idea—session types with quiescent participants—may apply beyond GPUs. We grade each potential transfer by mechanism fidelity: does the target domain share the same failure mode (reading from an inactive participant produces silent corruption), or merely a structural resemblance?

**FPGA crossbar protocols** (strong transfer): We have demonstrated this direction with a working prototype (§9.5). The mapping is direct: `TileGroup<S>` ↔ `Warp<S>`, tile sets ↔ active sets, `TileComplement` ↔ `ComplementOf`. The bug class is isomorphic: when a tile doesn't SEND, its pipeline register retains stale data—silent corruption identical to shuffle-from-inactive-lane. Mechanism, scale, and coupling all match.

**Distributed systems** (partial transfer): Node quiescence maps to lane inactivity, but the domains differ: distributed systems have genuine failure modes, non-deterministic failures, and no guarantee of reconvergence. Our quiescence model complements fault-tolerant MPST (§8.2) but is not a direct replacement.

**Database queries and proof search** (structural similarity only): Database predicate filtering and proof case splits share the abstract shape of active subset selection but lack the inter-participant communication that makes the type discipline actionable.

## 9.5 Hardware Crossbar Protocols

We have prototyped typestate crossbar communication (`src/research/crossbar_protocol.rs`, 12 tests) modeling a 16-tile pipelined crossbar. The mapping is direct: `TileGroup<S>` mirrors `Warp<S>`, tile sets mirror active sets, and `TileComplement` mirrors `ComplementOf`. Crossbar collectives (ring pass, butterfly exchange, scatter, gather) exist only on `TileGroup<AllTiles>` — after `diverge_halves()`, the methods vanish from the type.

The hardware bug class is real: when a tile diverges and doesn't SEND, its pipeline register retains data from the previous cycle. Other tiles reading from that channel get stale data with no hardware error — silent corruption identical to shuffle-from-inactive-lane. Our prototype's `stale_data_bug_demonstration` test reproduces this failure mode and shows how warp typestate prevents it.

## 9.6 Remaining Limitations

Several limitations remain:
- Higher-order protocols (protocols parameterized by protocols)
- Compilation overhead at scale (untested on large codebases)
- Cross-warp fence interactions (warp A diverges, warp B's fence depends on A's contribution via global memory — the intra-warp case is handled in §5.6, but cross-warp ordering remains open)

# 10. Conclusion

GPU warp programming is notoriously error-prone. Shuffles that read from inactive lanes produce undefined behavior—bugs that compile silently, work sometimes, and fail unpredictably. NVIDIA's own reference code contains these bugs. A plasma physics simulation ran for months with undefined behavior undetected on pre-Volta hardware. The vendor deprecated an entire API family to address the problem. State-of-the-art persistent thread programs maintain warp-uniform execution rather than manage divergence.

We presented **warp typestate**, a linear type system that makes lane-level divergence safe rather than forbidden:

1. **Warps carry active set types** (`Warp<Even>`, `Warp<All>`), tracking which lanes are active.

2. **Divergence produces complements**. When a warp splits, the type system knows the sub-warps together cover the original.

3. **Merge verifies complements**. The type system statically checks that merged warps are complementary.

4. **Shuffles require all lanes active**. The `shuffle_xor` method exists only on `Warp<All>`. Calling it on a diverged warp is not a runtime error—it is *unrepresentable*.

The concept of tracking which lanes are active is not new. ISPC manages it via compiler-emitted masks, Cooperative Groups expose it as runtime objects, LLVM's uniformity analysis infers it during compilation, and NVIDIA's synccheck detects violations post-execution. What we add is a *type-level* encoding: the active lane mask is a type parameter (`Warp<S>`) on a Boolean lattice, reconvergence is verified by sealed complement traits, and operations requiring all lanes are structurally absent (not checked — absent) on sub-warps. This is the first system that makes shuffle-from-inactive-lane a *missing-method error* rather than a runtime fault, an API documentation warning, or a post-hoc sanitizer finding.

Our implementation in Rust has **zero runtime overhead** — types are erased at compile time. For uniform programs (the dominant style, including state-of-the-art megakernels), the type system is invisible. For lane-heterogeneous programs, it replaces implicit bugs with explicit types. The annotation burden is modest: 16.7% of source lines on average across our 8 examples (range 11.3%–25.3%).

**The takeaway**: The gap between "the compiler knows the active set" (ISPC, LLVM) and "the type system enforces active-set safety" (this work) is the difference between a tool that *could* catch the bug and a type that *cannot express* the bug. We close that gap.

---

## Acknowledgments

The author used Claude (Anthropic, claude-opus-4-6, 2026) extensively in the drafting and editing of this manuscript.

## References

[1] Adameit, M., Viering, M., Peters, K., and Eugster, P. "Fault-Tolerant Multiparty Session Types." OOPSLA, 2022. https://doi.org/10.1145/3527316

[2] AMD. "HIP: C++ Heterogeneous-Compute Interface for Portability." 2016. https://github.com/ROCm/HIP

[3] Anthropic. Claude Opus 4.6 [Large language model]. 2026. https://www.anthropic.com

[4] Betts, A., Chong, N., Donaldson, A., Qadeer, S., and Thomson, P. "GPUVerify: A Verifier for GPU Kernels." OOPSLA, 2012. https://doi.org/10.1145/2384616.2384625

[5] Betts, A., Chong, N., Donaldson, A., Kettlewell, J., Qadeer, S., Thomson, P., and Sherrer, J. "The Design and Implementation of a Verification Technique for GPU Kernels." TOPLAS 37(3), 2015. https://doi.org/10.1145/2743013

[6] Bocchino, R. L., Adve, V. S., Dig, D., Adve, S. V., Heumann, S., Komuravelli, R., Overbey, J., Simmons, P., Sung, H., and Vakilian, M. "A Type and Effect System for Deterministic Parallel Java." OOPSLA, 2009. https://doi.org/10.1145/1640089.1640097

[7] Boyapati, C., Lee, R., and Rinard, M. "Ownership Types for Safe Programming: Preventing Data Races and Deadlocks." OOPSLA, 2002. https://doi.org/10.1145/582419.582440

[8] Caires, L. and Pfenning, F. "Session Types as Intuitionistic Linear Propositions." CONCUR, 2010. https://doi.org/10.1007/978-3-642-15375-4_16

[9] Chen, R., Balzer, S., and Bhatt Toninho, B. "Ferrite: A Judgmental Embedding of Session Types in Rust." ICFP, 2022. https://doi.org/10.1145/3547635

[10] Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Sber, M., Cowan, M., Wang, L., Hu, Y., Ceze, L., Guestrin, C., and Krishnamurthy, A. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI, 2018. https://www.usenix.org/conference/osdi18/presentation/chen

[11] Dardha, O., Giachino, E., and Sangiorgi, D. "Session Types Revisited." Information and Computation 256, 2017. https://doi.org/10.1016/j.ic.2017.06.002

[12] Flanagan, C. and Freund, S. N. "Type-Based Race Detection for Java." PLDI, 2000. https://doi.org/10.1145/349299.349328

[13] Girard, J.-Y. "Linear Logic." Theoretical Computer Science 50(1), 1987. https://doi.org/10.1016/0304-3975(87)90045-4

[14] Gu, Y., Lezama, J. P., Qi, S., Giannakou, A., and Donaldson, A. F. "Lockstep Execution Semantics for Modelling GPU Programs." OOPSLA, 2023. https://doi.org/10.1145/3622811

[15] Hazy Research (Stanford). "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B." Stanford AI Lab Blog, 2025. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

[16] Henriksen, T., Serup, N. G. W., Elsman, M., Henglein, F., and Oancea, C. E. "Futhark: Purely Functional GPU-Programming with Nested Parallelism and In-Place Array Updates." PLDI, 2017. https://doi.org/10.1145/3062341.3062354

[17] Honda, K. "Types for Dyadic Interaction." CONCUR, 1993. https://doi.org/10.1007/3-540-57208-2_35

[18] Honda, K., Yoshida, N., and Carbone, M. "Multiparty Asynchronous Session Types." POPL, 2008. https://doi.org/10.1145/1328438.1328472

[19] Igarashi, A., Thiemann, P., Vasconcelos, V. T., and Wadler, P. "Gradual Session Types." ICFP, 2017. https://doi.org/10.1145/3110282

[20] Khronos Group. "The OpenCL Specification, Version 1.0." 2009. https://www.khronos.org/opencl/

[21] Khronos Group. "SYCL 2020 Specification." 2020. https://www.khronos.org/sycl/

[22] Kopcke, B., Bischof, S., and Steffen, S. "Descend: A Safe GPU Systems Programming Language." PLDI, 2024. https://doi.org/10.1145/3656401

[23] Lange, J. and Yoshida, N. "On the Undecidability of Asynchronous Session Subtyping." FoSSaCS, 2016. https://doi.org/10.1007/978-3-662-49630-5_25

[24] Matsakis, N. D. and Klock, F. S. "The Rust Language." ACM SIGAda Ada Letters 34(3), 2014. https://doi.org/10.1145/2692956.2663188

[25] NVIDIA. "CUDA C Programming Guide." 2007 (updated annually). https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[26] NVIDIA. "Cooperative Groups: Flexible CUDA Thread Programming." GTC, 2017. https://developer.nvidia.com/blog/cooperative-groups/

[27] NVIDIA. "CUDA C Programming Guide, §10.22: Warp Shuffle Functions." 2017. Deprecation notice for `__shfl`, `__shfl_up`, `__shfl_down`, `__shfl_xor` in CUDA 9.0.

[28] NVIDIA. "NVIDIA Tesla V100 GPU Architecture: The World's Most Advanced Data Center GPU." Architecture Whitepaper, 2017. §Independent Thread Scheduling.

[29] NVIDIA. "CUDA Toolkit: compute-sanitizer." https://docs.nvidia.com/compute-sanitizer/

[30] Peng, Y., Grover, V., and Leis, J. "CURD: A Dynamic CUDA Race Detector." PLDI, 2018. https://doi.org/10.1145/3296979.3192368

[31] Pharr, M. and Mark, W. R. "ispc: A SPMD Compiler for High-Performance CPU Programming." InPar, 2012. https://doi.org/10.1109/InPar.2012.6339601

[32] Ragan-Kelley, J., Barnes, C., Adams, A., Paris, S., Durand, F., and Amarasinghe, S. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines." PLDI, 2013. https://doi.org/10.1145/2491956.2462176

[33] Stork, S., Marques, P., and Aldrich, J. "Concurrency by Default: Using Permissions to Express Dataflow in Stateful Programs." OOPSLA, 2014. https://doi.org/10.1145/2660193.2660205

[34] Viering, M., Hu, R., Eugster, P., and Ziarek, L. "A Multiparty Session Typing Discipline for Fault-Tolerant Event-Driven Distributed Programming." OOPSLA, 2021. https://doi.org/10.1145/3485484

[35] Wadler, P. "Propositions as Sessions." ICFP, 2012. https://doi.org/10.1145/2364527.2364568

[36] Wright, A. K. and Felleisen, M. "A Syntactic Approach to Type Soundness." Information and Computation 115(1), 1994. https://doi.org/10.1006/inco.1994.1093

[37] Zheng, M., Ravi, V. T., Qin, F., and Agrawal, G. "GMRace: Detecting Data Races in GPU Programs via a Low-Overhead Scheme." IEEE TPDS 25(1), 2014. https://doi.org/10.1109/TPDS.2013.44

### Bug Reports Cited

[B1] NVIDIA cuda-samples#398: Wrong ballot mask in reference reduction. https://github.com/NVIDIA/cuda-samples/issues/398

[B2] NVIDIA CCCL#854: Compiler predicates off mask initialization in CUB WarpScanShfl. https://github.com/NVIDIA/cccl/issues/854

[B3] PIConGPU#2514: Hardcoded full mask in divergent branch. https://github.com/ComputationalRadiationPhysics/picongpu/issues/2514

[B4] LLVM#155682: shfl_sync causes branch elimination. https://github.com/llvm/llvm-project/issues/155682
