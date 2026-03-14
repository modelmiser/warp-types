# Integrating warp-types Into Your Project

Three paths, from lowest friction to highest benefit.

## Path 1: Model Your GPU Logic (No Nightly, No GPU Compilation)

**Use case:** You have existing CUDA kernels and want compile-time safety checks without rewriting them.

**How it works:** Use warp-types on the host side (stable Rust) to model your kernel's warp logic. If the model type-checks, your CUDA code is safe — assuming the model matches the implementation.

```toml
# Cargo.toml
[dependencies]
warp-types = { path = "path/to/warp-types" }  # or version once published
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
warp-types = { path = "path/to/warp-types" }
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
warp-types-builder = { path = "path/to/warp-types/warp-types-builder" }
```

**Requirements:**
- Rust nightly (`rustup install nightly`)
- `rust-src` component (`rustup component add rust-src --toolchain nightly`)
- NVIDIA GPU with CUDA driver
- cudarc 0.19

**Effort:** 1-2 hours for first kernel. Then identical to normal Rust development.

## Path 3: Mixed (Gradual Migration)

**Use case:** Large existing CUDA codebase. Migrate incrementally.

1. Start with Path 1 — model existing kernels, find bugs
2. Write new kernels with Path 2
3. Gradually rewrite critical kernels in typed Rust
4. Keep CUDA kernels that don't need warp safety

The type system is additive — it doesn't require all-or-nothing adoption.

## FAQ

**Q: Does this work on stable Rust?**
A: The type system (Path 1) works on stable. GPU kernel compilation (Path 2) requires nightly for `abi_ptx` and `asm_experimental_arch`.

**Q: What GPU hardware is supported?**
A: Any NVIDIA GPU with a CUDA driver. Tested on RTX 4000 Ada (compute 8.9). AMD support is in progress (u64 masks and target stubs are ready).

**Q: What's the runtime overhead?**
A: Zero. `Warp<S>` is `PhantomData`. Verified at MIR, LLVM IR, and PTX levels.

**Q: Can I use this with wgpu/vulkano/ash?**
A: The type system (Path 1) works with any GPU framework. The kernel compilation pipeline (Path 2) currently targets NVIDIA PTX via cudarc. Vulkan/SPIR-V backend is future work.

**Q: What about AMD GPUs?**
A: The mask type is `u64` (supports 64-lane wavefronts). `GpuTarget::Amd` is in the builder. AMD inline assembly stubs exist but are untested without hardware.
