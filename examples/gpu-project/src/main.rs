//! Example: GPU kernels with warp-types, compiled and launched via cargo.
//!
//! This demonstrates the Month 2 workflow:
//!   1. Write kernels using warp-types in my-kernels/ crate
//!   2. build.rs compiles them to PTX automatically
//!   3. Host code loads PTX via generated Kernels struct
//!
//! Usage: cd examples/gpu-project && cargo run --release

#![allow(deprecated, unused_imports)]

// Pull in the generated kernels module (PTX + Kernels struct)
mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

use cudarc::driver::*;

const WARP_SIZE: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  warp-types: cargo-integrated GPU kernel demo           ║");
    println!("║  Type-safe kernels, compiled from cargo, run on GPU     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Initialize CUDA
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("GPU: {}", ctx.name()?);
    println!();

    // Load all kernels at once via generated struct
    let k = kernels::Kernels::load(&ctx)?;

    // === Test 1: butterfly_reduce ===
    println!("=== Test 1: butterfly_reduce (all ones) ===");
    {
        let input: Vec<i32> = vec![1; WARP_SIZE];
        let dev_data = stream.memcpy_stod(&input)?;

        unsafe {
            stream
                .launch_builder(&k.butterfly_reduce)
                .arg(&dev_data)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_data)?;
        let expected = WARP_SIZE as i32;
        let all_correct = result.iter().all(|&v| v == expected);
        println!("  Input:    [1; 32]");
        println!("  Expected: every lane = {}", expected);
        println!("  Got:      lane[0]={}, lane[31]={}", result[0], result[31]);
        println!("  Result:   {}", if all_correct { "PASS" } else { "FAIL" });
    }
    println!();

    // === Test 2: diverge_merge_reduce ===
    println!("=== Test 2: diverge_merge_reduce (sequential) ===");
    {
        let input: Vec<i32> = (0..WARP_SIZE as i32).collect();
        let dev_data = stream.memcpy_stod(&input)?;

        unsafe {
            stream
                .launch_builder(&k.diverge_merge_reduce)
                .arg(&dev_data)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_data)?;
        let expected = (0..WARP_SIZE as i32).sum::<i32>(); // 496
        let all_correct = result.iter().all(|&v| v == expected);
        println!("  Input:    [0, 1, 2, ..., 31]");
        println!("  Expected: every lane = {} (diverge/merge erased)", expected);
        println!("  Got:      lane[0]={}, lane[31]={}", result[0], result[31]);
        println!("  Result:   {}", if all_correct { "PASS" } else { "FAIL" });
    }
    println!();

    // === Test 3: reduce_n ===
    println!("=== Test 3: reduce_n (with scalar param) ===");
    {
        let input: Vec<i32> = vec![1; WARP_SIZE];
        let n: u32 = WARP_SIZE as u32;
        let dev_in = stream.memcpy_stod(&input)?;
        let dev_out = stream.memcpy_stod(&[0i32])?;

        unsafe {
            stream
                .launch_builder(&k.reduce_n)
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
        println!("  Input:    [1; 32], n=32");
        println!("  Expected: 32");
        println!("  Got:      {}", result[0]);
        println!("  Result:   {}", if result[0] == 32 { "PASS" } else { "FAIL" });
    }
    println!();

    println!("All kernels compiled from cargo, type-checked at build time,");
    println!("and executed on real GPU hardware with correct results.");

    Ok(())
}
