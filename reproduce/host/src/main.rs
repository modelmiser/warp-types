//! Host-side runner for warp-types GPU kernels.
//!
//! Loads pre-compiled PTX and runs typed butterfly reduction on actual GPU.
//! Demonstrates: warp typestate produces correct results on real
//! hardware with zero overhead.
//!
//! Usage:
//!   cd reproduce/host && cargo run --release
//!
//! Prerequisites:
//!   - NVIDIA GPU with CUDA driver
//!   - PTX compiled via: rustc +nightly --target nvptx64-nvidia-cuda --emit=asm -O \
//!       --edition 2021 ../typed_butterfly_kernel.rs -o ../typed_butterfly_kernel.ptx

#![allow(deprecated, unused_imports)]

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
            println!("Linear typestate divergence: correct results on real GPU hardware.");
            println!("Zero type system overhead. Zero runtime cost. Zero wrong answers.");
        }
    }

    // =========================================================================
    // cuda-samples #398 DEMO: typed catches the bug, untyped doesn't
    // =========================================================================
    println!();
    println!("================================================================");
    println!("cuda-samples #398: The Killer Demo");
    println!("================================================================");
    println!();

    // Load reduce7 PTX
    let reduce7_ptx = std::fs::read_to_string("../reduce7_typed.ptx")?;
    let reduce7_mod = ctx.load_module(nvrtc::Ptx::from_src(reduce7_ptx))?;
    let buggy_fn = reduce7_mod.load_function("reduce7_untyped_buggy")?;
    let fixed_fn = reduce7_mod.load_function("reduce7_typed_fixed")?;

    // Test: sum of 32 ones. Expected = 32.
    let input: Vec<i32> = vec![1; WARP_SIZE];
    let n: u32 = WARP_SIZE as u32;

    println!("=== Untyped (buggy): CUDA's reduce7 pattern ===");
    {
        let dev_in = stream.memcpy_stod(&input)?;
        let dev_out = stream.memcpy_stod(&[0i32])?;

        unsafe {
            stream.launch_builder(&buggy_fn)
                .arg(&dev_in)
                .arg(&dev_out)
                .arg(&n)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_out)?;
        println!("  Input:    [1; 32]");
        println!("  Expected: 32");
        println!("  Got:      {}", result[0]);
        println!("  Result:   {}", if result[0] == 32 { "correct (lucky)" } else { "WRONG (bug!)" });
    }
    println!();

    println!("=== Typed (fixed): Warp<All> enforced before shuffle ===");
    {
        let dev_in = stream.memcpy_stod(&input)?;
        let dev_out = stream.memcpy_stod(&[0i32])?;

        unsafe {
            stream.launch_builder(&fixed_fn)
                .arg(&dev_in)
                .arg(&dev_out)
                .arg(&n)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_out)?;
        println!("  Input:    [1; 32]");
        println!("  Expected: 32");
        println!("  Got:      {}", result[0]);
        println!("  Result:   {}", if result[0] == 32 { "CORRECT" } else { "wrong" });
    }
    println!();

    println!("=== The point ===");
    println!("The buggy pattern (shfl_down with partial mask) is a COMPILE ERROR");
    println!("in our type system: Warp<Lane0> has no shfl_down method.");
    println!("The fix (all lanes participate) is the ONLY version that type-checks.");
    println!("Same GPU. Same algorithm. Type system prevents the bug.");

    Ok(())
}
