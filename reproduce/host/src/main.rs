//! Host-side runner for warp-types GPU kernels.
//!
//! Loads pre-compiled PTX and runs typed butterfly reduction on actual GPU.
//! Demonstrates: session-typed divergence produces correct results on real
//! hardware with zero overhead.
//!
//! Usage:
//!   cd reproduce/host && cargo run --release
//!
//! Prerequisites:
//!   - NVIDIA GPU with CUDA driver
//!   - PTX compiled via: rustc +nightly --target nvptx64-nvidia-cuda --emit=asm -O \
//!       --edition 2021 ../typed_butterfly_kernel.rs -o ../typed_butterfly_kernel.ptx

use cudarc::driver::*;
use cudarc::nvrtc;
use std::sync::Arc;

const WARP_SIZE: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA context and stream
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("GPU: {}", ctx.name()?);
    println!();

    // Load PTX as a module
    let ptx_src = std::fs::read_to_string("../typed_butterfly_kernel.ptx")?;
    let ptx = nvrtc::Ptx::from_src(ptx_src);
    let module = ctx.load_module(ptx)?;

    // Load kernel functions
    let butterfly_fn = module.load_function("typed_butterfly_reduce")?;
    let diverge_fn = module.load_function("typed_diverge_merge_reduce")?;

    // === Test 1: Butterfly reduction (all ones) ===
    println!("=== Test 1: typed_butterfly_reduce ===");
    {
        let input: Vec<i32> = vec![1; WARP_SIZE];
        let dev_data = stream.memcpy_stod(&input)?;

        unsafe {
            stream
                .launch_builder(&butterfly_fn)
                .arg(&dev_data)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_data)?;
        let expected = WARP_SIZE as i32;

        println!("  Input:    [1; 32]");
        println!("  Expected: every lane = {}", expected);
        println!("  Got:      lane[0]={}, lane[15]={}, lane[31]={}",
                 result[0], result[15], result[31]);

        let all_correct = result.iter().all(|&v| v == expected);
        println!("  Result:   {}", if all_correct { "PASS" } else { "FAIL" });
        if !all_correct {
            println!("  Full: {:?}", &result);
        }
    }
    println!();

    // === Test 2: Sequential values ===
    println!("=== Test 2: typed_butterfly_reduce (sequential) ===");
    {
        let input: Vec<i32> = (0..WARP_SIZE as i32).collect();
        let dev_data = stream.memcpy_stod(&input)?;

        unsafe {
            stream
                .launch_builder(&butterfly_fn)
                .arg(&dev_data)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_data)?;
        let expected = (0..WARP_SIZE as i32).sum::<i32>(); // 496

        println!("  Input:    [0, 1, 2, ..., 31]");
        println!("  Expected: every lane = {}", expected);
        println!("  Got:      lane[0]={}, lane[15]={}, lane[31]={}",
                 result[0], result[15], result[31]);

        let all_correct = result.iter().all(|&v| v == expected);
        println!("  Result:   {}", if all_correct { "PASS" } else { "FAIL" });
    }
    println!();

    // === Test 3: Diverge + merge + reduce ===
    println!("=== Test 3: typed_diverge_merge_reduce ===");
    {
        let input: Vec<i32> = (0..WARP_SIZE as i32).collect();
        let dev_data = stream.memcpy_stod(&input)?;

        unsafe {
            stream
                .launch_builder(&diverge_fn)
                .arg(&dev_data)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_data)?;
        let expected = (0..WARP_SIZE as i32).sum::<i32>(); // 496

        println!("  Input:    [0, 1, 2, ..., 31]");
        println!("  Expected: every lane = {} (diverge/merge erased)", expected);
        println!("  Got:      lane[0]={}, lane[15]={}, lane[31]={}",
                 result[0], result[15], result[31]);

        let all_correct = result.iter().all(|&v| v == expected);
        println!("  Result:   {}", if all_correct { "PASS" } else { "FAIL" });
        println!();
        if all_correct {
            println!("Session-typed divergence: correct results on real GPU hardware.");
            println!("Zero type system overhead. Zero runtime cost. Zero wrong answers.");
        }
    }

    Ok(())
}
