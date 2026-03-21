# The Worst Kind of GPU Bug (And How Types Fix It)

There's a class of GPU bug that compiles cleanly, passes most tests, and produces undefined behavior. Not crashes. Not obvious failures. Silent, spec-violating operations that hardware may or may not mask. The kind of bug that a plasma physics simulation runs with for months — undetected because pre-Volta GPUs enforced convergence that the spec never guaranteed.

This post is about that bug, and a type system that makes it impossible.

## The Setup

Modern GPUs execute threads in lockstep groups — a **warp** (32 threads on NVIDIA, 64 on AMD). These threads share a register file and can exchange data instantly via **shuffle** instructions, avoiding slow memory round-trips:

```
Lane 0:  val=7   ──shuffle_xor(1)──▸  reads Lane 1's value (3)
Lane 1:  val=3   ──shuffle_xor(1)──▸  reads Lane 0's value (7)
```

Shuffles power the fastest GPU algorithms: reductions, scans, sorts, histograms. They're a key building block in GPU matrix multiply, where intra-warp communication via shuffles and shared memory helps sustain 90%+ of peak throughput.

## The Bug

What happens when not all 32 lanes participate?

```cuda
if (threadIdx.x < 16) {
    int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);
    // Lane 0 reads from Lane 1: OK, both are in the branch
    // But the mask says all 32 lanes — lanes 16-31 didn't enter
    // Their values are stale register contents: undefined behavior
}
```

The result: your reduction produces 17 instead of 32. Your sort puts elements in almost-the-right-order. Your physics simulation converges to the wrong steady state.

## This Bug Is Everywhere

We surveyed 21 documented instances across 16 real-world projects:

- **NVIDIA cuda-samples #398** — their own reference reduction. Wrong results for certain input sizes.
- **NVIDIA CUB (CCCL#854)** — the standard GPU primitives library. Warp-level scan (WarpScanShfl) suspected of wrong mask in sub-warp configurations.
- **PIConGPU #2514** — plasma physics simulation. Ran for months with divergent branch containing ballot. Pre-Volta hardware masked the UB; the fix was preventive, targeting Volta's independent thread scheduling.
- **PyTorch #98157** — `__activemask()` misuse in radix sort.
- **OpenCV #12320** — warpScanInclusive deadlock on Volta from hardcoded mask.
- **TVM #17307** — sub-mask triggers illegal instruction on H100.

NVIDIA deprecated the entire `__shfl` API family in CUDA 9.0, replacing it with `__shfl_sync` which requires an explicit mask. But programmers still pass the wrong mask. The problem isn't the API — it's that the compiler can't tell when a mask is wrong.

## The Fix: Types

What if the compiler *could* tell?

```rust
let warp: Warp<All> = Warp::kernel_entry();

// Diverge: the warp splits, consuming the original handle
let (active, inactive) = warp.diverge_even_odd();

// This line does not compile:
// active.shuffle_xor(data, 1);
// ERROR: no method `shuffle_xor` found for `Warp<Even>`

// To shuffle, you must merge back to Warp<All>:
let full_warp: Warp<All> = merge(active, inactive);
let partner = full_warp.shuffle_xor(data, 1);  // OK
```

`Warp<S>` is parameterized by an **active set** — a type that records which lanes are active. `shuffle_xor` is only implemented on `Warp<All>`. It doesn't exist on `Warp<Even>`, `Warp<Odd>`, or any sub-warp type. Not checked at runtime. Not validated. The method is *absent from the type*.

## Zero Overhead

`Warp<S>` is a zero-sized type containing only `PhantomData<S>`. The type parameter is erased completely by the compiler. We verified this at three levels:

1. **Rust MIR** — no trace of `Warp` or `PhantomData`
2. **NVIDIA PTX** — byte-identical assembly from `rustc --target nvptx64-nvidia-cuda`
3. **CPU LLVM IR** — typed functions compile to the same shuffle intrinsics as untyped (see `compare_ptx.sh`)

The type system exists only at compile time. The generated GPU code is the same as hand-written CUDA.

## The Killer Demo

We reproduced cuda-samples #398 on an NVIDIA RTX 4000 Ada. Same algorithm, same GPU:

```
CUDA (buggy):   sum of 32 ones = 1   ← wrong
Rust (typed):   sum of 32 ones = 32  ← correct
```

The buggy pattern (`shfl_down` after divergence to only lane 0) is a **compile error** in our type system. `Warp<Lane0>` has no `shfl_down` method. The fixed version — where all lanes participate — is the only code that type-checks.

```bash
bash reproduce/demo.sh  # See it for yourself
```

## Real GPU Kernels

This isn't a paper prototype. You can write real GPU kernels today:

```rust
#[warp_kernel]
pub fn butterfly_reduce(data: *mut i32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let mut val = unsafe { *data.add(tid as usize) };

    let d = data::PerLane::new(val);
    val += warp.shuffle_xor(d, 16).get();
    // ... 4 more stages ...

    unsafe { *data.add(tid as usize) = val; }
}
```

`cargo run` compiles this to PTX and launches it on your GPU. The type system catches bugs at build time. The generated code contains real `shfl.sync.bfly.b32` instructions.

We also implemented:
- **Bitonic sort** — direction-aware type-safe shuffle-XOR (15 steps for 32-lane, 21 for 64-lane)
- **CUB-equivalent primitives** — typed reduce, scan, broadcast
- **Cooperative groups** — thread block tiles with typed safety
- **64-bit shuffles** — automatic two-pass for i64/f64/u64
- **Data-dependent divergence** — `diverge_dynamic(mask)` handles runtime predicates with structural complement guarantees, no dependent types needed

## Why This Matters

The state of the art for GPU warp safety is: don't diverge. Persistent thread programs like the Hazy megakernel maintain warp-uniform execution — all lanes always active, no divergence, no shuffle bugs. It works, but it prohibits an entire class of algorithms.

Our type system is **strictly more permissive**: you CAN diverge, you CAN have sub-warps with different active sets, you just can't shuffle until you merge back. Lane-level heterogeneity becomes safe rather than forbidden.

The core insight is transferable to any language with phantom types or zero-cost abstractions. The `ComplementOf` pattern (compile-time proof of set complement) works in Rust, Haskell, Scala, Swift, and potentially C++ with concepts.

## AMD: Same Bugs, Wider Warps

AMD GPUs use 64-lane wavefronts — twice NVIDIA's width. We verified on an MI300X (gfx942):

```
6-stage butterfly (64-lane):  sum = 2016  ✓
5-stage butterfly (NVIDIA):   sum = 496   ✗ (only reduces within 32-lane halves)
```

The NVIDIA-style 5-stage reduce silently produces half-sums on AMD. No crash, no error — just wrong answers. Our type system catches this: `Warp<All>` on AMD requires 6 stages, and the `warp64` feature flag switches the active set masks from 32-bit to 64-bit.

Diverged shuffles on AMD return 0 for masked-out lanes (NVIDIA returns undefined). Wrong either way. The type system catches both.

## The Artifact

- 396 tests (317 unit + 50 example + 29 doc), all passing on both NVIDIA and AMD
- 31 Lean 4 theorems (progress + preservation + all four §5.1 loop rules + letPair + nested merge: zero `sorry`)
- 21 documented bugs across 16 real-world projects
- Real GPU execution on NVIDIA RTX 4000 Ada and AMD MI300X
- Cargo-integrated pipeline: `cargo run` from source to GPU
- `--features warp64`: full 64-lane AMD wavefront support

The code is at [github.com/modelmiser/warp-types](https://github.com/modelmiser/warp-types).

---

*The type system is simple. Warps carry types. Types track lanes. Shuffle requires all lanes. That's it. The rest — the macros, the builder, the sort, the tiles — follows from that one decision.*
