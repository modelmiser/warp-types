# warp-types: Session-Typed GPU Divergence

A type system that prevents shuffle-from-inactive-lane bugs in GPU warp programming by tracking active lane masks at compile time.

**Status:** Research prototype. Zero dependencies. 242 unit tests + 11 doc tests. Zero runtime overhead.

## The Problem

GPU warp primitives like shuffle enable fast intra-warp communication, but reading from an inactive lane is undefined behavior:

```cuda
if (participate) {
    int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);  // BUG
    // Reads from inactive lanes — undefined result
}
```

This compiles without warnings, may appear to work, and fails silently. NVIDIA's own cuda-samples contains this bug ([#398](https://github.com/NVIDIA/cuda-samples/issues/398)). Their core library CUB contains a variant ([CCCL#854](https://github.com/NVIDIA/cccl/issues/854)). The PIConGPU plasma physics simulation ran for months with undefined behavior in a divergent branch, undetected because pre-Volta hardware masked the violation ([#2514](https://github.com/ComputationalRadiationPhysics/picongpu/issues/2514)). NVIDIA deprecated the entire `__shfl` API family in CUDA 9.0 to address the bug class.

State-of-the-art persistent thread programs (e.g., the [Hazy megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)) avoid the problem by maintaining warp-uniform execution.

## The Solution

Track which lanes are active in the type system. Shuffle only exists on fully-active warps:

```rust
let (active, inactive) = warp.diverge(|lane| participate[lane]);

// This is a compile error — not a runtime check, but a type-level absence:
// active.shuffle_xor(data, 1);
// ERROR: no method `shuffle_xor` found for `Warp<Active>`

// Fix: merge back to Warp<All> first
let warp: Warp<All> = merge(active, inactive);
let partner = warp.shuffle_xor(data, 1);  // OK
```

The type system is **strictly more permissive** than current best practice (which prohibits divergence) while being **strictly safer** than the CUDA status quo (which allows it unchecked).

## Key Ideas

1. **`Warp<S>`** — Warps carry active set types. `Warp<All>` = all 32 lanes active.
2. **Diverge produces complements** — `diverge` splits a warp into two sub-warps with disjoint active sets.
3. **Merge requires complements** — Reconvergence is verified at compile time via `ComplementOf<T>` trait bounds.
4. **Method availability = safety** — `shuffle_xor` only exists on `Warp<All>`. Not checked — *absent*.
5. **Zero overhead** — `Warp<S>` contains only `PhantomData<S>`. Types are erased completely.

## Quick Start

```bash
cargo test                                    # All 242 unit tests + 12 doc tests
cargo test --examples                         # 21 tests across 5 real-bug examples
cargo test --example nvidia_cuda_samples_398  # Real NVIDIA bug, caught by types
cargo run --example demo_bug_that_types_catch # Core safety demonstration
```

## Claims and Evidence

| Claim | Evidence | Command |
|-------|----------|---------|
| Shuffle safety (diverged warp can't shuffle) | 8 compile-fail doctests | `cargo test --doc` |
| Real bug caught at compile time | NVIDIA cuda-samples #398 modeled | `cargo test --example nvidia_cuda_samples_398` |
| Hardware reproduction | Deterministic wrong result on RTX 4000 Ada | `cd reproduce && ./run.sh` |
| Soundness (progress + preservation) | 9 tests encoding proof steps | `cargo test proof` |
| Fence-divergence safety | Type-state write tracking (3 tests) | `cargo test fence` |
| Nested divergence (depth 3+) | Active set lattice tests | `cargo test nested_diverge` |
| Zero overhead | `PhantomData<S>` erasure — inspect MIR | `cargo test static_verify` |
| Platform portability (32/64 lanes) | Warp-size-generic algorithms | `cargo test warp_size` |
| Hardware crossbar mapping | Session-typed 16-tile crossbar (12 tests) | `cargo test crossbar` |
| Gradual typing (DynWarp ↔ Warp<S>) | Runtime/compile-time bridge (14 tests) | `cargo test gradual` |
| All claims | Full test suite (242 + 11) | `cargo test` |

## Project Structure

```
warp-types/
├── src/
│   ├── lib.rs              # Core exports + GpuValue trait
│   ├── active_set.rs       # Marker types: All, Even, Odd, LowHalf, ...
│   ├── warp.rs             # Warp<S> — the core parameterized type
│   ├── data.rs             # PerLane<T>, Uniform<T>, SingleLane<T, N>
│   ├── diverge.rs          # Split warps by predicate
│   ├── merge.rs            # Rejoin complementary sub-warps
│   ├── shuffle.rs          # Shuffle/ballot/reduce (Warp<All> only)
│   ├── fence.rs            # Fence-divergence type-state machine (§5.6)
│   ├── block.rs            # Block-level shared memory + reductions
│   ├── proof.rs            # Executable soundness sketch (9 lemmas)
│   ├── platform.rs         # CpuSimd<N> / GpuWarp32 dual-mode (§6.6)
│   ├── warp_size.rs        # Const-generic warp size portability
│   ├── gradual.rs          # DynWarp ↔ Warp<S> gradual typing bridge
│   └── research/           # 24 research exploration modules
│       ├── mod.rs
│       ├── protocol_inference.rs   # 5 inference approaches (14 tests)
│       ├── nested_diverge.rs       # Active set lattice (9 tests)
│       ├── shuffle_duality.rs      # Permutation algebra (9 tests)
│       ├── work_stealing.rs        # Ballot-based load balancing
│       ├── crossbar_protocol.rs    # FPGA crossbar session types (§9.6)
│       └── ...                     # 24 research modules
├── examples/
│   ├── nvidia_cuda_samples_398.rs  # Real NVIDIA bug, caught by types
│   └── demo_bug_that_types_catch.rs
├── reproduce/
│   ├── reduce7_bug.cu      # Hardware reproduction (RTX 4000 Ada)
│   └── run.sh              # Build and run instructions
└── paper/                   # Preprint (markdown)
    ├── paper.md             # Full assembled paper
    ├── empirical-evidence.md
    └── *.md                 # Section files
```

## Paper

**"Session Types for SIMT Divergence: Type-Safe GPU Warp Programming"**

The `paper/` directory contains the full preprint. Key sections:

- **Empirical evidence** — 4 documented bugs in NVIDIA cuda-samples, CUB, PIConGPU, and LLVM
- **Core type system** — formal typing rules for diverge, merge, shuffle
- **Metatheory** — progress and preservation proofs
- **Evaluation** — bug detection, zero-overhead argument, expressiveness analysis

## Limitations

- **No GPU code generation.** This is a type system prototype running on CPU. PTX/SPIR-V generation is future work.
- **No hardware benchmarks.** Zero overhead is by construction (PhantomData erasure), not by measurement.
- **Marker types only.** Common predicates (Even, Odd, LowHalf) are covered; fully data-dependent predicates require dependent types.
- **No cross-function inference.** Active set types must be annotated at function boundaries.

## License

MIT
