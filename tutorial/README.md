# warp-types Tutorial

A hands-on guide to type-safe GPU warp programming. Each step builds on the last. By the end, you'll write a GPU kernel where shuffle bugs are compile errors.

**Prerequisites:** Basic Rust. Some GPU/CUDA awareness helps but isn't required.

**Time:** ~15 minutes to read, ~30 minutes to run everything.

## Contents

1. [The Bug That Types Catch](#1-the-bug-that-types-catch)
2. [Your First Typed Warp](#2-your-first-typed-warp)
3. [Diverge and Merge](#3-diverge-and-merge)
4. [Shuffle Safety](#4-shuffle-safety)
5. [Writing a GPU Kernel](#5-writing-a-gpu-kernel)
6. [Running on Real Hardware](#6-running-on-real-hardware)
7. [Beyond Warps: Tiles and CUB Primitives](#7-beyond-warps-tiles-and-cub-primitives)
8. [What's Next](#8-whats-next)

---

## 1. The Bug That Types Catch

GPUs execute 32 threads in lockstep — a **warp**. These threads can exchange data via **shuffle** instructions, which is fast (no memory access) but dangerous:

```cuda
// CUDA — compiles fine, produces wrong results
if (threadIdx.x < 16) {
    int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);
    // BUG: reads from lanes 16-31 which didn't enter this branch
    // Their register values are stale — undefined behavior
}
```

This bug:
- Compiles without warnings
- May appear to work on some GPUs
- Produces silently wrong data on others
- Exists in NVIDIA's own cuda-samples (#398), their CUB library (CCCL#854), and at least 19 other documented cases across PyTorch, OpenCV, TVM, and more

NVIDIA deprecated the entire `__shfl` API family to address this bug class. Their replacement (`__shfl_sync` with explicit masks) still lets you pass the wrong mask.

**Our approach:** Make the bug a compile error.

## 2. Your First Typed Warp

```rust
use warp_types::*;

// At kernel entry, all 32 lanes are active
let warp: Warp<All> = Warp::kernel_entry();

// Warp<All> has shuffle — because all lanes participate
let data = data::PerLane::new(42i32);
let partner = warp.shuffle_xor(data, 1);  // OK
```

`Warp<S>` is a zero-sized type parameterized by an **active set** `S`. The active set tracks which of the 32 lanes are currently participating. `All` means all 32.

Key property: **`Warp<S>` is zero-cost.** It contains only `PhantomData<S>`. No runtime overhead. No memory. The type exists purely at compile time and is fully erased in the generated binary.

## 3. Diverge and Merge

When threads take different branches, the warp **diverges**. In our type system, diverge consumes the warp and produces two sub-warps:

```rust
let warp: Warp<All> = Warp::kernel_entry();

// Diverge: warp is consumed, two sub-warps are created
let (evens, odds) = warp.diverge_even_odd();

// evens: Warp<Even>  — lanes 0, 2, 4, ..., 30
// odds:  Warp<Odd>   — lanes 1, 3, 5, ..., 31

// The original `warp` is gone — using it is a compile error:
// warp.shuffle_xor(data, 1);  // ERROR: use of moved value
```

To get `Warp<All>` back, you must **merge** complementary sub-warps:

```rust
let merged: Warp<All> = merge(evens, odds);
// Now shuffle is available again
let partner = merged.shuffle_xor(data, 1);  // OK
```

The `merge` function requires `ComplementOf` — a compile-time proof that the two sets are disjoint and cover all lanes. You can't merge `Warp<Even>` with `Warp<Even>`:

```rust
// merge(evens, evens);  // ERROR: Even does not implement ComplementOf<Even>
```

## 4. Shuffle Safety

Here's the core safety property: **`shuffle_xor` only exists on `Warp<All>`.**

It's not checked at runtime. It's not validated. The method *does not exist* on any other type:

```rust
let (evens, odds) = warp.diverge_even_odd();

// This is not a runtime error. It's not a warning.
// The method literally does not exist on Warp<Even>:
// evens.shuffle_xor(data, 1);
// ERROR: no method named `shuffle_xor` found for `Warp<Even>`
```

Here's the actual compiler output when you try:

```
error[E0599]: no method named `shuffle_xor` found for struct `Warp<Even>` in the current scope
  --> src/main.rs:8:19
   |
8  |     evens.shuffle_xor(data, 1);
   |           ^^^^^^^^^^^ method not found in `Warp<Even>`
   |
   = note: the method was found for
           - `Warp<All>`
```

The compiler tells you exactly what happened: `shuffle_xor` exists on `Warp<All>` but not `Warp<Even>`. The fix is clear — `merge(evens, odds)` to get `Warp<All>` back.

This is why the cuda-samples #398 bug is impossible in our type system. The buggy pattern requires shuffling with only some lanes active. That code cannot be written — it's a type error.

## 5. Writing a GPU Kernel

With `#[warp_kernel]`, you write kernels in a normal Rust crate:

```rust
// my-kernels/src/lib.rs
#![no_std]
#![no_main]
#![feature(abi_ptx, asm_experimental_arch)]

use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }

use warp_types::*;

#[warp_kernel]
pub fn my_reduce(data: *mut i32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let mut val = unsafe { *data.add(tid as usize) };

    // Butterfly reduction — 5 shuffle-XOR stages
    // Type system guarantees all lanes participate at every step
    let d = data::PerLane::new(val);
    val += warp.shuffle_xor(d, 16).get();
    let d = data::PerLane::new(val);
    val += warp.shuffle_xor(d, 8).get();
    let d = data::PerLane::new(val);
    val += warp.shuffle_xor(d, 4).get();
    let d = data::PerLane::new(val);
    val += warp.shuffle_xor(d, 2).get();
    let d = data::PerLane::new(val);
    val += warp.shuffle_xor(d, 1).get();

    unsafe { *data.add(tid as usize) = val; }
}
```

The `#[warp_kernel]` macro transforms this into a proper PTX kernel entry point (`extern "ptx-kernel"` with `#[no_mangle]`).

## 6. Running on Real Hardware

Three files make the complete pipeline:

**`my-kernels/Cargo.toml`** — the kernel crate:
```toml
[dependencies]
warp-types = { path = "path/to/warp-types" }
```

**`build.rs`** — auto-compiles kernels to PTX:
```rust
fn main() {
    warp_types_builder::WarpBuilder::new("my-kernels")
        .build()
        .expect("Failed to compile GPU kernels");
}
```

**`src/main.rs`** — loads and launches:
```rust
mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

fn main() {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let k = kernels::Kernels::load(&ctx).unwrap();
    // k.my_reduce — CudaFunction handle, ready to launch
}
```

Run `cargo run` — your typed kernel compiles to PTX, loads onto the GPU, and executes. The type system catches shuffle bugs at build time. The generated PTX contains real `shfl.sync.bfly.b32` instructions with zero overhead from the types.

**Try the built-in demo:**
```bash
cd examples/gpu-project && cargo run --release
```

## 7. Beyond Warps: Tiles and CUB Primitives

### Thread Block Tiles

NVIDIA's Cooperative Groups partition warps into tiles of 4, 8, 16, or 32 threads. Unlike diverged sub-warps, tiles are *partitions* — all threads participate:

```rust
let warp: Warp<All> = Warp::kernel_entry();
let tile: tile::Tile<16> = warp.tile();

// Shuffle within tile is SAFE — all 16 threads participate
let data = data::PerLane::new(42i32);
let partner = tile.shuffle_xor(data, 1);  // OK for any tile size

// Sub-partition
let small_tile = tile.partition_8();
let sum = small_tile.reduce_sum(data);
```

Key distinction: `Warp<Even>` (diverged, 16 lanes) has NO shuffle. `Tile<16>` (partitioned, 16 lanes) HAS shuffle. Same hardware state, opposite safety — the type system captures the difference.

### CUB-Equivalent Primitives

Typed versions of NVIDIA CUB's warp-level operations:

```rust
let warp: Warp<All> = Warp::kernel_entry();
let data = data::PerLane::new(val);

// Reduce with custom operator
let max = warp.reduce(data, |a, b| if a > b { a } else { b });

// Inclusive prefix sum
let prefix = warp.inclusive_sum(data);

// Bitonic sort (15 shuffle-XOR steps)
let sorted = warp.bitonic_sort(data);
```

All require `Warp<All>` — the compiler prevents calling them on diverged sub-warps.

### 64-bit Types

GPU shuffle instructions are 32-bit. For `i64`, `f64`, `u64`, the type system handles the two-pass shuffle automatically:

```rust
let data = data::PerLane::new(3.14159_f64);
let partner = warp.shuffle_xor(data, 1);  // Two shfl.sync calls, zero overhead
```

Both halves are always shuffled together — you can't accidentally shuffle only the low 32 bits.

### Data-Dependent Divergence

When the active set depends on runtime data (not a fixed pattern like Even/Odd), use `diverge_dynamic`:

```rust
let warp: Warp<All> = Warp::kernel_entry();

// Mask determined at runtime (e.g., from ballot or data comparison)
let runtime_mask: u64 = 0x0000FFFF;
let diverged = warp.diverge_dynamic(runtime_mask);

// Can't shuffle on either branch — they're partial warps
// Must merge to recover Warp<All>
let warp: Warp<All> = diverged.merge();
let _result = warp.shuffle_xor(data, 1);  // OK
```

The mask is dynamic but the complement is structural — `DynDiverge` guarantees that `true_mask | false_mask == all_mask` by construction. No dependent types needed.

## 8. What's Next

### For Users
- Clone the repo and run `cargo test` (263 unit + 24 doc = 287, plus 50 example = 337 total)
- Try `bash reproduce/demo.sh` to see the cuda-samples #398 bug vs. type-safe fix
- Write your own kernel in `examples/gpu-project/`
- Read the paper in `paper/paper.md`

### For Researchers
- The Lean 4 formalization is in `lean/` (28 theorems, all zero `sorry`, zero axioms — including progress, preservation, and 5 bug untypability proofs)
- Active set masks are `u64` — ready for AMD 64-lane wavefronts
- The `Platform` trait in the builder supports future backends
- The tile system opens cooperative groups to formal typing

### For Language Designers
- The core insight transfers to any language with phantom types or zero-cost abstractions
- Session types for hardware communication protocols (not just software)
- The `ComplementOf` pattern (compile-time proof of set complement) is reusable

### Known Limitations
- AMD GPU intrinsics are stubbed but untested (no hardware available)
- Nightly Rust required for GPU kernel compilation (`abi_ptx`, `asm_experimental_arch`); the type system itself works on stable

---

**The core idea is simple:** Warps carry types. Types track which lanes are active. Shuffle only exists when all lanes are active. Everything else — the proc macros, the builder, the tiles, the sort — follows from that one decision.

If this idea proves useful, we'll be glad to have been a footnote in the story of safe GPU programming.
