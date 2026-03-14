# warp-types: Session-Typed GPU Divergence

A type system that prevents shuffle-from-inactive-lane bugs in GPU warp programming by tracking active lane masks at compile time.

**Status:** Research prototype with real GPU execution. 272 unit + 50 example + 23 doc tests (345 total). Zero runtime overhead verified at Rust MIR, LLVM IR, and NVIDIA PTX levels. Cargo-integrated GPU compilation pipeline.

## The Problem

GPU warp primitives like shuffle enable fast intra-warp communication, but reading from an inactive lane is undefined behavior:

```cuda
if (participate) {
    int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);  // BUG
    // Reads from inactive lanes — undefined result
}
```

This compiles without warnings, may appear to work, and fails silently. NVIDIA's own cuda-samples contains this bug ([#398](https://github.com/NVIDIA/cuda-samples/issues/398)). Their core library CUB contains a variant ([CCCL#854](https://github.com/NVIDIA/cccl/issues/854)). The PIConGPU plasma physics simulation ran for months with undefined behavior in a divergent branch, undetected because pre-Volta hardware masked the violation ([#2514](https://github.com/ComputationalRadiationPhysics/picongpu/issues/2514)). NVIDIA deprecated the entire `__shfl` API family in CUDA 9.0 to address the bug class.

We survey 21 documented bugs across 16 real-world projects. State-of-the-art persistent thread programs (e.g., the [Hazy megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)) avoid the problem by maintaining warp-uniform execution.

## The Solution

Track which lanes are active in the type system. Shuffle only exists on fully-active warps:

```rust
use warp_types::*;

let warp: Warp<All> = Warp::kernel_entry();

// After diverge, shuffle is gone from the type:
let (evens, odds) = warp.diverge_even_odd();
// evens.shuffle_xor(data, 1);  // COMPILE ERROR — method not found
let merged: Warp<All> = merge(evens, odds);
let partner = merged.shuffle_xor(data, 1);  // OK
```

The type system is **strictly more permissive** than current best practice (which prohibits divergence) while being **strictly safer** than the CUDA status quo (which allows it unchecked).

## Key Ideas

1. **`Warp<S>`** — Warps carry active set types. `Warp<All>` = all lanes active.
2. **Diverge produces complements** — `diverge` splits a warp into two sub-warps with disjoint active sets.
3. **Merge requires complements** — Reconvergence is verified at compile time via `ComplementOf<T>` trait bounds.
4. **Method availability = safety** — `shuffle_xor` only exists on `Warp<All>`. Not checked — *absent*.
5. **Zero overhead** — `Warp<S>` contains only `PhantomData<S>`. Types are erased completely.
6. **Data-dependent divergence** — `diverge_dynamic(mask)` handles runtime predicates. The mask is dynamic, but the complement is structural — both branches must merge before shuffle.
7. **Cross-function inference** — Generic functions take `Warp<S>` with `S: ActiveSet`. Rust infers `S` at call sites.

## The Killer Demo

cuda-samples #398 on the same GPU (RTX 4000 Ada):

```
Untyped (buggy): sum = 1   ← silent wrong answer
Typed (fixed):   sum = 32  ← correct, AND the bug is a compile error
```

The buggy pattern literally cannot be expressed. `Warp<Lane0>` has no `shfl_down` method.

```bash
bash reproduce/demo.sh  # The entire pitch in one terminal
```

## Quick Start

```bash
cargo test                                    # 272 unit + 23 doc tests
cargo test --examples                         # 50 tests across 8 real-bug examples
cargo test --example nvidia_cuda_samples_398  # Real NVIDIA bug, caught by types
```

### Write GPU Kernels

```rust
// my-kernels/src/lib.rs
use warp_types::*;

#[warp_kernel]
pub fn butterfly_reduce(data: *mut i32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let mut val = unsafe { *data.add(tid as usize) };

    // Type system guarantees: warp is Warp<All>, so shuffle is available
    let d = data::PerLane::new(val);
    val += warp.shuffle_xor(d, 16).get();
    // ... butterfly stages ...
    unsafe { *data.add(tid as usize) = val; }
}
```

```rust
// build.rs
fn main() {
    warp_types_builder::WarpBuilder::new("my-kernels")
        .build()
        .expect("Failed to compile GPU kernels");
}
```

```rust
// src/main.rs
mod kernels { include!(concat!(env!("OUT_DIR"), "/kernels.rs")); }

fn main() {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let k = kernels::Kernels::load(&ctx).unwrap();
    // k.butterfly_reduce — ready to launch
}
```

## Claims and Evidence

| Claim | Evidence | Command |
|-------|----------|---------|
| Shuffle safety (diverged warp can't shuffle) | 8 compile-fail doctests | `cargo test --doc` |
| Real bug caught at compile time | 8 worked bug examples (21 bugs surveyed) | `cargo test --examples` |
| Hardware reproduction | Deterministic wrong result on RTX 4000 Ada | `bash reproduce/demo.sh` |
| Real GPU execution | 3 kernels PASS on RTX 4000 Ada via cudarc | `cd examples/gpu-project && cargo run` |
| Cargo integration | `#[warp_kernel]` + `WarpBuilder` + `Kernels` struct | `cd examples/gpu-project && cargo run` |
| Zero overhead | Verified at MIR, LLVM IR, and PTX levels | `cargo rustc --release --lib -- --emit=llvm-ir` |
| Soundness (progress + preservation) | 17 Lean theorems (zero sorry on progress) | `cd lean && lake build` |
| CUB-equivalent primitives | Typed reduce, scan, broadcast (8 tests) | `cargo test cub` |
| Fence-divergence safety | Type-state write tracking (3 tests) | `cargo test fence` |
| Platform portability (32/64 lanes) | u64 masks, AMD stubs, GpuTarget enum | `cargo test warp_size` |
| Gradual typing (DynWarp ↔ Warp<S>) | Runtime/compile-time bridge (14 tests) | `cargo test gradual` |
| All claims | Full test suite (345 tests) | `cargo test && cargo test --examples` |

## Project Structure

```
warp-types/
├── src/
│   ├── lib.rs              # Core exports + GpuValue trait + warp_kernel re-export
│   ├── active_set.rs       # Marker types: All, Even, Odd, LowHalf, ... (u64 masks)
│   ├── warp.rs             # Warp<S> — the core parameterized type
│   ├── data.rs             # PerLane<T>, Uniform<T>, SingleLane<T, N>
│   ├── diverge.rs          # Split warps by predicate
│   ├── merge.rs            # Rejoin complementary sub-warps
│   ├── shuffle.rs          # Shuffle/ballot/reduce (Warp<All> only) + permutation algebra
│   ├── cub.rs              # CUB-equivalent typed warp primitives
│   ├── dynamic.rs          # Data-dependent divergence (DynDiverge)
│   ├── gpu.rs              # PTX/AMDGPU intrinsics + GpuShuffle trait
│   ├── fence.rs            # Fence-divergence type-state machine
│   ├── block.rs            # Block-level shared memory + reductions
│   ├── proof.rs            # Executable soundness sketch (9 lemmas)
│   ├── platform.rs         # CpuSimd<N> / GpuWarp32 / GpuWarp64 dual-mode
│   ├── warp_size.rs        # Const-generic warp size portability
│   ├── gradual.rs          # DynWarp ↔ Warp<S> gradual typing bridge
│   └── research/           # 24 research exploration modules
├── warp-types-macros/      # warp_sets! proc macro (active set hierarchy generation)
├── warp-types-kernel/      # #[warp_kernel] proc macro (GPU kernel entry points)
├── warp-types-builder/     # WarpBuilder (build.rs cross-compilation to PTX)
├── examples/
│   ├── nvidia_cuda_samples_398.rs  # Real NVIDIA bug, caught by types
│   ├── opencv_12320.rs            # OpenCV warpScanInclusive deadlock
│   ├── pytorch_98157.rs           # PyTorch __activemask() misuse
│   ├── tvm_17307.rs               # TVM LowerThreadAllreduce H100 crash
│   └── gpu-project/               # End-to-end cargo→GPU example
├── reproduce/
│   ├── demo.sh             # Full demonstration script
│   ├── host/               # cudarc host runner for real GPU execution
│   └── *.rs, *.cu          # PTX comparison + hardware reproduction
├── lean/                   # Lean 4 formalization (17 theorems)
└── paper/                  # Preprint (markdown)
```

## Limitations

- **AMD untested.** u64 masks and `amdgpu` target stubs are in place but require AMD hardware for validation.
- **Nightly required for GPU kernels.** `#[warp_kernel]` requires `abi_ptx` and `asm_experimental_arch` features. The type system itself works on stable Rust.

## License

MIT
