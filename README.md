# warp-types: Type-Safe GPU Warp Programming via Linear Typestate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19040615.svg)](https://doi.org/10.5281/zenodo.19040615)

A type system that prevents shuffle-from-inactive-lane bugs in GPU warp programming by tracking active lane masks at compile time.

**Status:** Research prototype with real GPU execution. Main crate 334 unit + 50 example tests, plus 46 doc tests workspace-wide (430 total). Zero runtime overhead verified at Rust MIR, LLVM IR, and NVIDIA PTX levels. Cargo-integrated GPU compilation pipeline. C++20 header (`include/warp_types.h`) for CUDA/HIP interop.

## The Problem

GPU warp primitives like shuffle enable fast intra-warp communication, but reading from an inactive lane is undefined behavior:

```cuda
if (participate) {
    int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);  // BUG
    // Reads from inactive lanes — undefined result
}
```

This compiles without warnings, may appear to work, and fails silently. NVIDIA's own cuda-samples contains this bug ([#398](https://github.com/NVIDIA/cuda-samples/issues/398)). Their core library CUB contains a variant ([CCCL#854](https://github.com/NVIDIA/cccl/issues/854); the reporter later noted it may have been a false positive, but the source-level pattern is ill-typed in our system regardless). The PIConGPU plasma physics simulation ran for months with undefined behavior in a divergent branch, undetected because pre-Volta hardware masked the violation ([#2514](https://github.com/ComputationalRadiationPhysics/picongpu/issues/2514)). NVIDIA deprecated the entire `__shfl` API family in CUDA 9.0 to address the bug class.

We survey 21 documented instances across 16 real-world projects (19 bugs plus a spec-level exclusion and a vendor deprecation). State-of-the-art persistent thread programs (e.g., the [Hazy megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)) avoid the problem by maintaining warp-uniform execution.

## The Solution

Track which lanes are active in the type system. Full-warp shuffle only exists on fully-active warps:

```rust
use warp_types::*;

let warp: Warp<All> = Warp::kernel_entry();

// After diverge, full-warp shuffle_xor is gone from the type:
let (evens, odds) = warp.diverge_even_odd();
// evens.shuffle_xor(data, 1);  // COMPILE ERROR — method not found
let merged: Warp<All> = merge(evens, odds);
let partner = merged.shuffle_xor(data, 1);  // OK
```

The type system is **strictly more permissive** than current best practice (which prohibits divergence) while being **safer, within the typed fragment,** than the CUDA status quo (which allows it unchecked; Limitations below lists what the fragment excludes).

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

Written this way — divergence expressed through `diverge`/`merge` — the buggy pattern cannot be expressed: `Warp<Lane0>` has no `shfl_down` method. (The guarantee is conditional on using the combinators; see Limitations.)

```bash
bash reproduce/demo.sh  # The entire pitch in one terminal
```

## Quick Start

```bash
cargo test                                    # 334 unit tests (main crate)
cargo test --workspace --doc                  # 46 doc tests (22 compile-fail)
cargo test --examples                         # 50 example tests across 8 examples (7 real-bug + 1 synthetic)
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
| Type safety (diverged warp can't call full-warp shuffle, merge requires complements) | 22 compile-fail doctests | `cargo test --workspace --doc` |
| Real bug caught at compile time | 7 real-bug + 1 synthetic examples (21 bugs surveyed) | `cargo test --examples` |
| Hardware reproduction | Deterministic wrong result on H200 SXM, RTX 4000 Ada | `bash reproduce/demo.sh` |
| Real GPU execution | 7 kernels via cudarc PASS on RTX 4000 Ada and 6 on H200 SXM (Test 7, the gradual API's runtime-checked shuffle, was added in 0.4.3 and awaits its H200 run), incl. ballot and sub-warp `shuffle_xor_within`; compute-sanitizer memcheck + synccheck 0 errors on the H200 (`results/h200-2026-10-18-main.txt`, `results/rtx4000ada-2026-10-18.txt`) | `cd examples/gpu-project && cargo run` |
| Shuffle semantics | Verified on H200 (sm_90), RTX 4000 Ada (sm_89), MI300X (gfx942) | `bash reproduce/runpod-h200.sh` |
| Cargo integration | `#[warp_kernel]` + `WarpBuilder` + `Kernels` struct | `cd examples/gpu-project && cargo run` |
| Zero overhead | Verified at MIR, LLVM IR, and PTX levels (search for `warp_types_zero_overhead_butterfly` in IR) | `bash reproduce/compare_ptx.sh` or `cargo rustc --release --lib -- --emit=llvm-ir` |
| Soundness (progress + preservation) | Core calculus mechanized in Lean 4 (33 named theorems in Basic+Metatheory), zero sorry; all 530 theorem constants in the library depend only on `propext`, `Classical.choice`, `Quot.sound` | `cd lean && lake build` | <!-- unguarded: 530 — CollectAxioms scan count (all theorem constants in WarpTypes.*), dated 2026-10-18; the guard totals only Basic + Metatheory -->
| CUB-equivalent primitives | Typed reduce, scan, broadcast (8 tests) | `cargo test cub` | <!-- unguarded: 8 — module-scoped test count; the guard computes no per-module actual -->
| Fence-divergence safety | Type-state write tracking (6 tests) | `cargo test fence` | <!-- unguarded: 6 — module-scoped test count; the guard computes no per-module actual -->
| Platform portability (32-lane warp via CpuSimd, 64-lane support) | u64 masks, AMD wavefronts, Platform trait | `cargo test warp_size` |
| Gradual typing (DynWarp ↔ Warp<S>) | Runtime/compile-time bridge (33 tests) | `cargo test gradual` | <!-- unguarded: 33 — module-scoped test count; the guard computes no per-module actual -->
| Main-crate claims | Main-crate test suite (424 tests) | `cargo test && cargo test --examples` | <!-- unguarded: 424 — measured 2026-10-18: cargo test -p warp-types (374) + --examples (50); no main-crate-scoped DOC actual exists; see TODO -->

## Project Structure

```
warp-types/
├── src/
│   ├── lib.rs              # Core exports + GpuValue trait + warp_kernel re-export
│   ├── active_set.rs       # Marker types: All, Even, Odd, LowHalf, ... (u64 masks, sealed)
│   ├── warp.rs             # Warp<S> — the core parameterized type
│   ├── data.rs             # PerLane<T>, Uniform<T>, SingleLane<T, N>
│   ├── diverge.rs          # Split warps by predicate
│   ├── merge.rs            # Rejoin complementary sub-warps
│   ├── shuffle.rs          # Shuffle/ballot/reduce (Warp<All> only) + permutation algebra
│   ├── cub.rs              # CUB-equivalent typed warp primitives
│   ├── sort.rs             # Bitonic sort (typed warp primitives)
│   ├── tile.rs             # Tile-level warp partitioning
│   ├── dynamic.rs          # Data-dependent divergence (DynDiverge)
│   ├── gpu.rs              # PTX/AMDGPU intrinsics + GpuShuffle trait
│   ├── fence.rs            # Fence-divergence type-state machine
│   ├── block.rs            # Block-level shared memory + reductions
│   ├── proof.rs            # Executable soundness sketch (progress, preservation, type safety)
│   ├── platform.rs         # CpuSimd<N> / GpuWarp32 / GpuWarp64 dual-mode
│   ├── gradual.rs          # DynWarp ↔ Warp<S> gradual typing bridge
│   ├── simwarp.rs           # Multi-lane warp simulator (real shuffle semantics for testing)
│   └── research/           # 24 research exploration modules (incl. warp_size portability)
├── warp-types-macros/      # warp_sets! proc macro (active set hierarchy generation)
├── warp-types-kernel/      # #[warp_kernel] proc macro (GPU kernel entry points)
├── warp-types-builder/     # WarpBuilder (build.rs cross-compilation to PTX)
├── examples/
│   ├── nvidia_cuda_samples_398.rs  # Real NVIDIA bug, caught by types
│   ├── cub_cccl_854.rs            # CUB/CCCL sub-warp shuffle bug
│   ├── picongpu_2514.rs           # PIConGPU months of silent UB
│   ├── llvm_155682.rs             # LLVM lane-0 conditional shuffle
│   ├── opencv_12320.rs            # OpenCV warpScanInclusive deadlock
│   ├── pytorch_98157.rs           # PyTorch __activemask() misuse
│   ├── tvm_17307.rs               # TVM LowerThreadAllreduce H100 crash
│   ├── demo_bug_that_types_catch.rs  # Synthetic demonstration
│   ├── gpu-project/               # End-to-end cargo→GPU example (Rust host)
│   └── cuda/                      # C++ host loading Rust PTX (CUDA Driver API)
├── include/
│   └── warp_types.h               # C++20 header: type-safe warp programming for CUDA/HIP
├── reproduce/
│   ├── demo.sh             # Full demonstration script
│   ├── host/               # cudarc host runner for real GPU execution
│   └── *.rs, *.cu          # PTX comparison + hardware reproduction
├── lean/                   # Lean 4 formalization (33 named theorems, zero sorry)
├── paper/                  # Preprint (markdown)
├── tutorial/               # Step-by-step tutorial
├── blog/                   # Blog post draft
└── INTEGRATION.md          # Integration guide
```

## Limitations

- **Affine, not linear.** Rust's type system is affine (values can be dropped without use). The Lean calculus is affine for warps too: a bound warp is consumed by at most one evaluated use, but `fst`/`snd` can drop it and `warpVal` literals are unrestricted (paper §4.4, §4.8). In Rust, a `Warp<S>` can be silently dropped without merging. `#[must_use]` warnings catch this in practice, but it is not a hard error. See the paper's Section 6 for discussion.
- **AMD partially verified.** u64 masks and `warp64` feature support 64-lane wavefronts. MI300X (gfx942) verified for mask correctness via HIP. Full GPU execution path untested (no amdgcn Rust target yet). NVIDIA verified on H200 SXM (compute 9.0, Hopper) and RTX 4000 Ada (compute 8.9, Ada Lovelace): shuffle semantics, zero-overhead PTX, and typed kernel execution (7 kernels PASS on the Ada, 6 on the H200 — `results/h200-2026-10-18-main.txt`, `results/rtx4000ada-2026-10-18.txt`).
- **The guarantee covers the typed fragment.** A raw data-dependent branch around a warp intrinsic (`if tid < 16 { warp.shuffle_xor(...) }`) is ordinary Rust: it type-checks and is exactly the unsafe pattern on hardware. The safety claim holds for programs that express divergence exclusively via `diverge`/`merge`.
- **`diverge` emits no branch; `shuffle_xor_within` guards its own execution.** After `diverge_even_odd()` every lane still runs the code that follows. Since 0.4.1 `shuffle_xor_within` skips the `shfl.sync` on lanes outside `S` (they return their input), so its membermask names exactly the executing lanes. In 0.4.0 and earlier the lanes outside `S` executed an instruction whose membermask did not name them — undefined behaviour under the crate's own documented usage. `Warp<Empty>::shuffle_xor_within` fails at codegen (post-monomorphization; not under `cargo check`), on the host and on nvptx64.
- **Gradual ascription is a trusted entry point.** `DynWarp::{all, all_64, from_mask_32, from_mask_64, from_mask}(…).ascribe::<S>()` checks `S` only against the caller's own mask, so it can produce any `Warp<S>` anywhere, like a second `kernel_entry()`. The full list of such routes is under "Trusted entry points" on `Warp::kernel_entry`.
- **Merge checks set shape, not provenance.** `Even` from one `kernel_entry()` handle and `Odd` from another merge without error.
- **Raw intrinsics are `unsafe`.** `GpuShuffle::gpu_shfl_*` (re-exported for use as a bound) are `unsafe fn` since 0.4.0 (a breaking change): they are raw `shfl.sync` instructions, UB under divergence. Shuffle through `Warp<All>` / `Tile`.
- **`kernel_entry()` is callable anywhere.** The `Warp<All>` witness is not unforgeable: a fresh handle can be constructed after a diverge, bypassing the typestate discipline. The discipline requires calling it exactly once at kernel entry; nothing enforces that yet.
- **Nightly required for GPU kernels.** `#[warp_kernel]` requires `abi_ptx` and `asm_experimental_arch` features. The type system itself works on stable Rust.

## License

MIT
