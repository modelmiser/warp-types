# Integrating warp-types Into Your Project

Five paths, from lowest friction to highest benefit.

## Path 1: Model Your GPU Logic (No Nightly, No GPU Compilation)

**Use case:** You have existing CUDA kernels and want compile-time safety checks without rewriting them.

**How it works:** Use warp-types on the host side (stable Rust) to model your kernel's warp logic. If the model type-checks, your CUDA code is safe — assuming the model matches the implementation.

```toml
# Cargo.toml
[dependencies]
warp-types = "0.2"
```

```rust
use warp_types::prelude::*;

/// Model of your CUDA kernel's warp logic.
/// If this compiles, the shuffle/diverge pattern is safe.
fn verify_my_kernel_logic() {
    let warp: Warp<All> = Warp::kernel_entry();
    let data = PerLane::new(0i32);

    // Model: your kernel diverges on even/odd lanes
    let (evens, odds) = warp.diverge_even_odd();

    // Model: both branches do independent work
    // (no shuffle here — would be a compile error)

    // Model: merge before reduction
    let warp: Warp<All> = merge(evens, odds);
    let _sum = warp.reduce_sum(data);  // OK — all lanes active
}

// If verify_my_kernel_logic compiles, your CUDA kernel's
// diverge → work → merge → reduce pattern is type-safe.
```

**What this catches:**
- Shuffle after diverge (without merge)
- Missing merge before collective operations
- Wrong complement pairs in merge
- Use-after-diverge (moved value)

**What this doesn't catch:**
- Data-dependent divergence (your model uses Even/Odd, real code uses runtime predicates)
- Off-by-one in mask constants
- Bugs in the CUDA code that the model doesn't represent

**Effort:** 30 minutes. Add the crate, write a model function per kernel, run `cargo check`.

## Path 2: New Kernels in Typed Rust (Nightly Required)

**Use case:** You're writing new GPU kernels and want full type safety plus real GPU execution.

**Setup:**

```
my-project/
├── Cargo.toml          # host code
├── build.rs            # WarpBuilder compiles kernels
├── src/main.rs         # host: load PTX, launch kernels
└── my-kernels/         # kernel crate
    ├── Cargo.toml      # depends on warp-types
    └── src/lib.rs      # #[warp_kernel] functions
```

**Kernel crate (`my-kernels/Cargo.toml`):**
```toml
[package]
name = "my-kernels"
version = "0.1.0"
edition = "2021"

[dependencies]
warp-types = "0.2"
```

**Kernel code (`my-kernels/src/lib.rs`):**
```rust
#![no_std]
#![no_main]
#![feature(abi_ptx, asm_experimental_arch)]

use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }

use warp_types::prelude::*;
use warp_types::warp_kernel;

#[warp_kernel]
pub fn my_reduce(input: *const i32, output: *mut i32, n: u32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();

    let mut val = if tid < n { *input.add(tid as usize) } else { 0 };

    let d = PerLane::new(val);
    val += warp.shuffle_xor(d, 16).get();
    let d = PerLane::new(val);
    val += warp.shuffle_xor(d, 8).get();
    let d = PerLane::new(val);
    val += warp.shuffle_xor(d, 4).get();
    let d = PerLane::new(val);
    val += warp.shuffle_xor(d, 2).get();
    let d = PerLane::new(val);
    val += warp.shuffle_xor(d, 1).get();

    if tid == 0 {
        *output = val;
    }
}
```

**Host build (`build.rs`):**
```rust
fn main() {
    warp_types_builder::WarpBuilder::new("my-kernels")
        .build()
        .expect("Failed to compile GPU kernels");
}
```

**Host code (`src/main.rs`):**
```rust
mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let k = kernels::Kernels::load(&ctx)?;

    let stream = ctx.default_stream();
    let input = vec![1i32; 32];
    let dev_in = stream.memcpy_stod(&input)?;
    let dev_out = stream.memcpy_stod(&[0i32])?;
    let n = 32u32;

    unsafe {
        stream.launch_builder(&k.my_reduce)
            .arg(&dev_in)
            .arg(&dev_out)
            .arg(&n)
            .launch(cudarc::driver::LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let result = stream.memcpy_dtov(&dev_out)?;
    println!("Sum: {}", result[0]);  // 32
    Ok(())
}
```

**Host Cargo.toml:**
```toml
[dependencies]
cudarc = { version = "0.19", features = ["driver", "cuda-12000"] }

[build-dependencies]
warp-types-builder = "0.2"
```

**Requirements:**
- Rust nightly (`rustup install nightly`)
- `rust-src` component (`rustup component add rust-src --toolchain nightly`)
- NVIDIA GPU with CUDA driver
- cudarc 0.19

**Effort:** 1-2 hours for first kernel. Then identical to normal Rust development.

## Path 3: C++ Host with Rust Kernels

**Use case:** Your project is C++ but you want type-safe GPU kernels. Write kernels in Rust, load the PTX from C++.

**How it works:** Rust compiles kernels to PTX (same output as nvcc). Your C++ host loads the PTX at runtime via CUDA Driver API or HIP — identical to loading any PTX module.

**Setup:**

```
my-project/
├── CMakeLists.txt       # or Makefile
├── src/main.cu          # C++ host: load PTX, launch kernels
└── kernels/             # Rust kernel crate
    ├── Cargo.toml
    └── src/lib.rs       # #[warp_kernel] functions
```

**C++ host (`src/main.cu`):**
```cpp
#include <cuda.h>

int main() {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    // Load Rust-compiled PTX
    CUmodule mod;
    cuModuleLoad(&mod, "kernels/target/nvptx64-nvidia-cuda/release/my_kernels.ptx");

    // Get kernel function (name matches #[warp_kernel] fn name)
    CUfunction kernel;
    cuModuleGetFunction(&kernel, mod, "reduce_n");

    // Allocate, copy, launch — standard CUDA Driver API
    CUdeviceptr d_in, d_out;
    cuMemAlloc(&d_in, 32 * sizeof(int));
    cuMemAlloc(&d_out, sizeof(int));

    int input[32]; for (int i = 0; i < 32; i++) input[i] = 1;
    cuMemcpyHtoD(d_in, input, sizeof(input));

    unsigned int n = 32;
    void* args[] = { &d_in, &d_out, &n };
    cuLaunchKernel(kernel, 1,1,1, 32,1,1, 0, 0, args, nullptr);
    cuCtxSynchronize();

    int result;
    cuMemcpyDtoH(&result, d_out, sizeof(int));
    printf("Sum: %d\n", result);  // 32
}
```

**Build:**
```bash
# Build Rust kernels to PTX
cd kernels && cargo +nightly build --release

# Build C++ host
nvcc --std=c++20 -o demo src/main.cu -lcuda

# Or with CMake (see examples/cuda/CMakeLists.txt)
mkdir build && cd build && cmake .. && make
```

**HIP equivalent:** Replace `cuInit` → `hipInit`, `cuModuleLoad` → `hipModuleLoad`, etc. The API shape is 1:1. HIP on NVIDIA can load PTX directly.

**Effort:** 30 minutes if you know the CUDA Driver API.

## Path 4: C++ Kernels with `warp_types.h`

**Use case:** You write CUDA or HIP kernels in C++ and want the same compile-time safety as the Rust version. No Rust toolchain required.

**How it works:** `include/warp_types.h` is a standalone C++20 header that mirrors the Rust type system using concepts and `requires` clauses. `Warp<All>` has shuffle methods; `Warp<Even>` doesn't. Same guarantee, zero overhead.

```cpp
#include "warp_types.h"
using namespace warp_types;

__global__ void safe_reduce(int* data) {
    auto warp = Warp<All>::kernel_entry();
    auto val = PerLane<int>::from(data[threadIdx.x]);

    // OK: shuffle on Warp<All>
    auto sum = warp.reduce_sum(val);
    if (threadIdx.x == 0) data[0] = sum.get();
}

__global__ void buggy_reduce(int* data) {
    auto warp = Warp<All>::kernel_entry();
    auto [evens, odds] = warp.diverge_even_odd();

    auto val = PerLane<int>::from(data[threadIdx.x]);
    // COMPILE ERROR: shuffle_xor requires same_as<S, All>
    // auto result = evens.shuffle_xor(val, 1);

    // Fix: merge first
    auto merged = merge(evens, odds);
    auto result = merged.shuffle_xor(val, 1);  // OK
    (void)result;
}
```

**What you get:**
- Shuffle/reduce/ballot only on `Warp<All>` — diverged warps can't shuffle
- `merge()` requires `ComplementOf` — can't merge `Even + LowHalf`
- Works with CUDA (`__shfl_xor_sync`), HIP (`__shfl_xor`), and host-only (modeling)
- Zero runtime overhead — `Warp<S>` is empty

**Build:**
```bash
nvcc --std=c++20 -I path/to/warp-types/include -o demo my_kernel.cu
# or
hipcc --std=c++20 -I path/to/warp-types/include -o demo my_kernel.cu
```

**Limitation vs Rust:** C++ can't enforce linear types (move-only empty structs). After `diverge_even_odd()`, you *can* still use the original warp handle — C++ trusts you not to. Rust makes this a compile error.

**Effort:** 10 minutes. Copy the header, add `-I` flag, use the API.

## Path 5: Mixed (Gradual Migration)

**Use case:** Large existing CUDA codebase. Migrate incrementally.

1. Start with Path 1 — model existing kernels, find bugs
2. Write new kernels with Path 2
3. Gradually rewrite critical kernels in typed Rust
4. Keep CUDA kernels that don't need warp safety

The type system is additive — it doesn't require all-or-nothing adoption.

## CUDA-to-warp-types Phrasebook

Common CUDA patterns and their warp-types equivalents:

| CUDA | warp-types |
|------|------------|
| `__shfl_xor_sync(0xFFFFFFFF, val, mask)` | `warp.shuffle_xor(PerLane::new(val), mask).get()` or `warp.shuffle_xor_raw(val, mask)` |
| `__shfl_down_sync(0xFFFFFFFF, val, delta)` | `warp.shuffle_down(PerLane::new(val), delta).get()` or `warp.shuffle_down_raw(val, delta)` |
| `__ballot_sync(0xFFFFFFFF, pred)` | `warp.ballot(PerLane::new(pred))` (returns `BallotResult`) |
| `cub::WarpReduce<T>::Sum(val)` | `warp.reduce_sum(PerLane::new(val))` |
| `cub::WarpScan<T>::InclusiveSum(val)` | `warp.inclusive_sum(PerLane::new(val))` |
| `cg::tiled_partition<16>(block)` | `warp.tile::<16>()` |
| `if (threadIdx.x < N) { ... }` | `let diverged = warp.diverge_dynamic(mask); diverged.merge()` |
| `__activemask()` | No equivalent needed — active set is tracked in the type |
| Full-warp kernel entry | `let warp: Warp<All> = Warp::kernel_entry();` |

**Key difference:** In CUDA, `0xFFFFFFFF` is a runtime mask you pass to every shuffle. In warp-types, the mask is a type (`All`) — the compiler verifies it's correct. You never write a mask constant.

## FAQ

**Q: Does this work on stable Rust?**
A: The type system (Path 1) works on stable. GPU kernel compilation (Path 2) requires nightly for `abi_ptx` and `asm_experimental_arch`.

**Q: What GPU hardware is supported?**
A: Any NVIDIA GPU with a CUDA driver. Tested on RTX 4000 Ada (compute 8.9). AMD MI300X (gfx942) verified for mask correctness. Full AMD GPU execution requires an amdgcn Rust target (not yet available).

**Q: What's the runtime overhead?**
A: Zero. `Warp<S>` is `PhantomData`. Verified at MIR, LLVM IR, and PTX levels.

**Q: Can I use this with wgpu/vulkano/ash?**
A: The type system (Path 1) works with any GPU framework. The kernel compilation pipeline (Path 2) currently targets NVIDIA PTX via cudarc. Vulkan/SPIR-V backend is future work.

**Q: What about AMD GPUs?**
A: The mask type is `u64` (supports 64-lane wavefronts). `GpuTarget::Amd` is in the builder. MI300X mask correctness verified. For AMD GPU execution today, use Path 4 (`warp_types.h` with HIP) — Rust doesn't target amdgcn yet.

**Q: Can I use warp-types from C++?**
A: Two options. Path 3: write kernels in Rust, load PTX from C++ via `cuModuleLoad`. Path 4: write kernels in C++ using `include/warp_types.h` for compile-time safety. Both are zero overhead.

**Q: Does the C++ header require Rust?**
A: No. `warp_types.h` is a standalone C++20 header. It requires `--std=c++20` (nvcc or hipcc) but no Rust toolchain. It provides the same type-level guarantees as the Rust version.

**Q: What C++ standard is required?**
A: C++20. The header uses `concept`, `requires` clauses, and `std::same_as` — these don't exist in C++17. CUDA 12+ and ROCm 5+ support C++20.
