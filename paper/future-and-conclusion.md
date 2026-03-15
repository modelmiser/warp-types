# 9. Future Work

Session-typed divergence opens several research directions.

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

Our core metatheory is fully mechanized in Lean 4 (§4.8): progress, preservation, and the substitution lemma are all machine-checked with zero `sorry` and zero axioms. Five bug untypability proofs are also mechanized. Remaining future work:
- Extend mechanization to nested divergence (generalize `IsComplementAll` to `IsComplement s1 s2 parent`)
- Mechanize the loop typing rules (§5.1) and set-preserving shuffle (§4.6)
- Verified Rust implementation via Aeneas translation
- Leverage prior Lean-based GPU verification work (MCL framework)

## 9.3 IDE Integration

Rich IDE support would enhance usability:
- Visualize active sets at each program point
- Show which lanes are active in hover tooltips
- Suggest merge points when shuffles fail type checking
- Refactoring: extract divergent code into typed helper functions

## 9.4 Protocol Inference and Gradual Typing

Our current system requires explicit type annotations. We have explored inference strategies in research prototypes — local inference (within functions), bidirectional checking (mix inference and annotation), and gradual typing — with 14 tests across five approaches (`src/research/protocol_inference.rs`).

The gradual typing approach is promoted to the public API (`src/gradual.rs`, 18 tests): `DynWarp` provides the same operations as `Warp<S>` but checks safety invariants at runtime instead of compile time. The migration path:

1. **Start dynamic**: `DynWarp::all()` — all operations runtime-checked
2. **Ascribe at boundaries**: `dyn_warp.ascribe::<All>()?` — runtime evidence becomes compile-time proof
3. **End static**: `Warp<S>` everywhere — zero-overhead, compile-time safety

`DynWarp` also handles the data-dependent predicate case (§9.1): when the active set depends on runtime data and cannot be expressed as a marker type, `DynWarp` provides runtime safety that `Warp<S>` cannot.

Remaining future work:
- Local inference integration into the public API (infer active sets within functions, require annotations only at boundaries)
- Protocol-first development (design protocol in DSL, generate/check code against it)

## 9.5 Beyond SIMT

The core idea—session types with quiescent participants—may apply beyond GPUs. We grade each potential transfer by mechanism fidelity: does the target domain share the same failure mode (reading from an inactive participant produces silent corruption), or merely a structural resemblance?

**FPGA crossbar protocols** (strong transfer): We have demonstrated this direction with a working prototype (§9.6). The mapping is direct: `TileGroup<S>` ↔ `Warp<S>`, tile sets ↔ active sets, `TileComplement` ↔ `ComplementOf`. The bug class is isomorphic: when a tile doesn't SEND, its pipeline register retains stale data—silent corruption identical to shuffle-from-inactive-lane. Mechanism, scale, and coupling all match.

**Distributed systems** (partial transfer): Node quiescence maps to lane inactivity, and multiparty session types already model distributed communication. However, the domains diverge on three axes: (1) distributed systems have genuine failure modes (Byzantine, crash-stop, network partition) that GPU warps lack; (2) SIMT divergence is deterministic (predicate-based) while distributed failure is non-deterministic; (3) SIMT guarantees eventual reconvergence at a merge point while distributed systems may not reconverge. The quiescence model is complementary to fault-tolerant MPST (§8.2) but not a direct replacement.

**Database queries and proof search** (structural similarity only): Predicate filtering in databases and case splits in proof search share the abstract shape of "active subset selection," but the mechanism diverges fundamentally. Database rows are independent data items, not lock-step execution units sharing an instruction stream—there is no "communication between filtered rows" analogous to shuffle. Proof sub-goals interact via shared logical context, not register exchange. We note the structural parallel but do not claim actionable type-system transfer to these domains.

## 9.6 Hardware Crossbar Protocols

We have prototyped session-typed crossbar communication (`src/research/crossbar_protocol.rs`, 12 tests) modeling a 16-tile pipelined crossbar. The mapping is direct: `TileGroup<S>` mirrors `Warp<S>`, tile sets mirror active sets, and `TileComplement` mirrors `ComplementOf`. Crossbar collectives (ring pass, butterfly exchange, scatter, gather) exist only on `TileGroup<AllTiles>` — after `diverge_halves()`, the methods vanish from the type.

The hardware bug class is real: when a tile diverges and doesn't SEND, its pipeline register retains data from the previous cycle. Other tiles reading from that channel get stale data with no hardware error — silent corruption identical to shuffle-from-inactive-lane. Our prototype's `stale_data_bug_demonstration` test reproduces this failure mode and shows how session types prevent it.

Future work extends toward hardware synthesis proper:
- Generating crossbar routing configurations from session-typed protocols
- Synthesizing predication logic matching diverge/merge structure
- Area/power optimization guided by protocol structure (unused crossbar paths can be power-gated)

## 9.7 Remaining Limitations

Several limitations remain:
- Higher-order protocols (protocols parameterized by protocols)
- Compilation overhead at scale (untested on large codebases)
- Cross-warp fence interactions (warp A diverges, warp B's fence depends on A's contribution via global memory — the intra-warp case is handled in §5.6, but cross-warp ordering remains open)

# 10. Conclusion

GPU warp programming is notoriously error-prone. Shuffles that read from inactive lanes produce undefined behavior—bugs that compile silently, work sometimes, and fail unpredictably. NVIDIA's own reference code contains these bugs. A plasma physics simulation ran for months with undefined behavior undetected on pre-Volta hardware. The vendor deprecated an entire API family to address the problem. State-of-the-art persistent thread programs maintain warp-uniform execution rather than manage divergence.

We presented **session-typed divergence**, a type system that makes lane-level divergence safe rather than forbidden:

1. **Warps carry active set types** (`Warp<Even>`, `Warp<All>`), tracking which lanes are active.

2. **Divergence produces complements**. When a warp splits, the type system knows the sub-warps together cover the original.

3. **Merge verifies complements**. The type system statically checks that merged warps are complementary.

4. **Shuffles require all lanes active**. The `shuffle_xor` method exists only on `Warp<All>`. Calling it on a diverged warp is not a runtime error—it is *unrepresentable*.

The key insight is that GPU divergence fits the session type model: diverging is branching where some parties go *quiescent* (not failed, just paused), and reconverging is joining where quiescent parties resume. This correspondence gives us a principled type discipline for an ad-hoc problem.

Our implementation in Rust has **zero runtime overhead**—guaranteed by construction, not measured. Types are erased at compile time. For uniform programs (the style used by state-of-the-art megakernels), the type system is invisible. For lane-heterogeneous programs, it replaces implicit bugs with explicit types. The result is strictly more permissive than the divergence-prohibition approach while being strictly safer than CUDA's `__shfl_sync`.

Beyond safety, the type system **unlocks performance that is currently too dangerous to attempt.** State-of-the-art GPU kernels avoid lane-level divergence entirely—not because uniform execution is always optimal, but because divergent shuffles are too risky to get right. Algorithms that could be faster with lane-level heterogeneity (warp-level sort in rasterizers, divergent reductions in ray traversal, per-lane work stealing in ML kernels) are written as uniform instead, leaving performance on the table. Session-typed divergence makes the fast-but-dangerous path safe.

Session-typed divergence is not just a solution for GPU programming. It is an instance of a broader pattern: *participatory computation* where the set of active participants changes over time. We have demonstrated a direct transfer to FPGA crossbar protocols (§9.6), identified partial transfers to distributed systems, and noted structural parallels in databases and proof search (§9.5). The transfer fidelity correlates with mechanism match: domains where inactive participants produce silent data corruption (FPGAs, GPUs) benefit most; domains with merely analogous "active subset selection" benefit least.

**The takeaway**: Divergence bugs are type errors. Types exist to make certain classes of bugs impossible. Now shuffle-from-inactive-lane is one of them.

---

## Acknowledgments

The author used Claude (Anthropic, claude-opus-4-6, 2026) extensively in the drafting and editing of this manuscript.

## References

[References would be formatted according to venue style. Key citations include:]

- Betts et al. 2012. "GPUVerify: A Verifier for GPU Kernels" (OOPSLA)
- Bocchino et al. 2009. "A Type and Effect System for Deterministic Parallel Java" (OOPSLA)
- Caires and Pfenning 2010. "Session Types as Intuitionistic Linear Propositions" (CONCUR)
- Hazy Research 2025. "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B" (Stanford Blog)
- Honda 1993. "Types for Dyadic Interaction" (CONCUR)
- Honda, Yoshida, Carbone 2008. "Multiparty Asynchronous Session Types" (POPL)
- Henriksen et al. 2017. "Futhark: Purely Functional GPU-Programming" (PLDI)
- NVIDIA 2017. "Cooperative Groups: Flexible Thread Synchronization" (GTC)
- NVIDIA 2017. CUDA Programming Guide §10.22: Warp Shuffle Functions (deprecation notice)
- NVIDIA 2017. Tesla V100 Architecture Whitepaper: Independent Thread Scheduling
- NVIDIA cuda-samples#398: Wrong ballot mask in reference reduction
- NVIDIA CCCL#854: Compiler predicates off mask initialization in CUB WarpScanShfl
- PIConGPU#2514: Hardcoded full mask in divergent branch
- LLVM#155682: shfl_sync causes branch elimination
- Kopcke et al. 2024. "Descend: A Safe GPU Systems Programming Language" (PLDI)
- Wadler 2012. "Propositions as Sessions" (ICFP)
- Wright and Felleisen 1994. "A Syntactic Approach to Type Soundness" (IC)
- Anthropic. (2026). Claude Opus 4.6 [Large language model]. https://www.anthropic.com

