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
        println!(
            "  Expected: every lane = {} (diverge/merge erased)",
            expected
        );
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
        println!(
            "  Result:   {}",
            if result[0] == 32 { "PASS" } else { "FAIL" }
        );
    }
    println!();

    // === Test 4: bitonic_sort_i32 ===
    println!("=== Test 4: bitonic_sort_i32 (reverse → sorted) ===");
    {
        // Input: 32 values in reverse order [31, 30, 29, ..., 1, 0]
        let input: Vec<i32> = (0..WARP_SIZE as i32).rev().collect();
        let dev_data = stream.memcpy_stod(&input)?;

        unsafe {
            stream
                .launch_builder(&k.bitonic_sort_i32)
                .arg(&dev_data)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_data)?;
        let expected: Vec<i32> = (0..WARP_SIZE as i32).collect();
        let correct = result == expected;
        println!("  Input:    [31, 30, 29, ..., 1, 0]");
        println!("  Expected: [0, 1, 2, ..., 30, 31]");
        println!(
            "  Got:      [{}, {}, {}, ..., {}, {}]",
            result[0], result[1], result[2], result[30], result[31]
        );
        println!("  Result:   {}", if correct { "PASS" } else { "FAIL" });
        if !correct {
            println!("  Full:     {:?}", &result);
        }
    }
    println!();

    // === Test 5: ballot_lower_half — ballot via the convergent vote intrinsic ===
    // Since 0.4.2 `Warp<All>::ballot` lowers through `llvm.nvvm.vote.ballot.sync`
    // (the `bool` predicate becomes the `i1` operand). Before that (b6b737622,
    // 2026-03-29) it was inline asm with a `.reg .pred` + `setp` workaround.
    println!("=== Test 5: ballot_lower_half (vote.sync.ballot) ===");
    {
        let dev_out = stream.memcpy_stod(&[0u32; WARP_SIZE])?;

        unsafe {
            stream
                .launch_builder(&k.ballot_lower_half)
                .arg(&dev_out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_out)?;
        let expected: u32 = 0x0000_FFFF; // lanes 0..16 voted true
        let lane0_correct = result[0] == expected;
        let warp_uniform = result.iter().all(|&v| v == result[0]);
        println!("  Predicate: tid < 16");
        println!(
            "  Expected:  0x{:08X} (lanes 0..16 = 1, lanes 16..32 = 0)",
            expected
        );
        println!("  lane[0]:   0x{:08X}", result[0]);
        println!(
            "  uniform:   {} (all 32 lanes wrote the same value)",
            warp_uniform
        );
        println!(
            "  Result:    {}",
            if lane0_correct && warp_uniform {
                "PASS"
            } else {
                "FAIL"
            }
        );
        if !lane0_correct || !warp_uniform {
            println!(
                "  Full:      {:08X?}",
                result.iter().map(|v| *v).collect::<Vec<_>>()
            );
        }
    }
    println!();

    // === Test 6: shuffle_xor_within after diverge_even_odd, no branch in the caller ===
    // The documented usage. All 32 lanes reach the call; the method's lane
    // guard must keep the odd lanes out of the shfl.sync (membermask Even).
    println!("=== Test 6: shuffle_within_even (sub-warp membermask, no caller branch) ===");
    {
        let dev_out = stream.memcpy_stod(&[-1i32; WARP_SIZE])?;

        unsafe {
            stream
                .launch_builder(&k.shuffle_within_even)
                .arg(&dev_out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let result = stream.memcpy_dtov(&dev_out)?;
        let expected: Vec<i32> = (0..WARP_SIZE as i32)
            .map(|t| if t % 2 == 0 { t ^ 2 } else { t })
            .collect();
        let pass = result == expected;
        println!("  Even lanes: tid ^ 2 (partner within Even); odd lanes: own tid");
        println!("  Result:    {}", if pass { "PASS" } else { "FAIL" });
        if !pass {
            println!("  Got:       {:?}", result);
            println!("  Expected:  {:?}", expected);
        }
    }
    println!();

    // === Test 7: DynWarp::shuffle_xor_scalar on the device ===
    println!("=== Test 7: dyn_shuffle_xor (gradual API, runtime-checked, real shuffle) ===");
    {
        let dev_out = stream.memcpy_stod(&[-1i32; WARP_SIZE])?;
        unsafe {
            stream
                .launch_builder(&k.dyn_shuffle_xor)
                .arg(&dev_out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let result = stream.memcpy_dtov(&dev_out)?;
        let expected: Vec<i32> = (0..WARP_SIZE as i32).map(|t| t ^ 1).collect();
        let pass = result == expected;
        println!("  Every lane: tid ^ 1");
        println!("  Result:    {}", if pass { "PASS" } else { "FAIL" });
        if !pass {
            println!("  Got:       {:?}", result);
            println!("  Expected:  {:?}", expected);
        }
    }
    println!();

    // === Test 8: the gradual API refuses a partial mask on the device ===
    println!("=== Test 8: dyn_shuffle_xor_partial (gradual API, error path) ===");
    {
        let dev_out = stream.memcpy_stod(&[7i32; WARP_SIZE])?;
        unsafe {
            stream
                .launch_builder(&k.dyn_shuffle_xor_partial)
                .arg(&dev_out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let result = stream.memcpy_dtov(&dev_out)?;
        let pass = result.iter().all(|&v| v == -1);
        println!("  Mask 0x0000FFFF: every lane gets Err → -1 sentinel");
        println!("  Result:    {}", if pass { "PASS" } else { "FAIL" });
        if !pass {
            println!("  Got:       {:?}", result);
        }
    }
    println!();

    // === Test 9: warp prefix sums (inclusive and exclusive) ===
    println!("=== Test 9: scan_sums (Warp::inclusive_sum / exclusive_sum) ===");
    let mut scan_failures = 0;
    let input: Vec<i32> = (0..WARP_SIZE as i32).map(|t| t * t + 1).collect();
    {
        let dev_in = stream.memcpy_stod(&input)?;
        let dev_incl = stream.memcpy_stod(&[0i32; WARP_SIZE])?;
        let dev_excl = stream.memcpy_stod(&[0i32; WARP_SIZE])?;
        unsafe {
            stream
                .launch_builder(&k.scan_sums)
                .arg(&dev_in)
                .arg(&dev_incl)
                .arg(&dev_excl)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let incl = stream.memcpy_dtov(&dev_incl)?;
        let excl = stream.memcpy_dtov(&dev_excl)?;
        let want_incl: Vec<i32> = input
            .iter()
            .scan(0, |a, &v| {
                *a += v;
                Some(*a)
            })
            .collect();
        let want_excl: Vec<i32> = (0..WARP_SIZE).map(|t| want_incl[t] - input[t]).collect();
        for (name, got, want) in [
            ("inclusive", &incl, &want_incl),
            ("exclusive", &excl, &want_excl),
        ] {
            let pass = got == want;
            scan_failures += usize::from(!pass);
            println!("  {name}: {}", if pass { "PASS" } else { "FAIL" });
            if !pass {
                println!("    Got:      {:?}", got);
                println!("    Expected: {:?}", want);
            }
        }
    }
    println!();

    // === Test 10: Tile<N> prefix sums ===
    println!("=== Test 10: tile_scans (Tile<4|8|16|32>::inclusive_sum) ===");
    {
        let dev_in = stream.memcpy_stod(&input)?;
        let dev_out = stream.memcpy_stod(&[0i32; 4 * WARP_SIZE])?;
        unsafe {
            stream
                .launch_builder(&k.tile_scans)
                .arg(&dev_in)
                .arg(&dev_out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let got = stream.memcpy_dtov(&dev_out)?;
        for (row, n) in [4usize, 8, 16, 32].into_iter().enumerate() {
            let want: Vec<i32> = (0..WARP_SIZE)
                .map(|t| input[t & !(n - 1)..=t].iter().sum())
                .collect();
            let g = &got[row * WARP_SIZE..(row + 1) * WARP_SIZE];
            let pass = g == want.as_slice();
            scan_failures += usize::from(!pass);
            println!("  Tile<{n}>: {}", if pass { "PASS" } else { "FAIL" });
            if !pass {
                println!("    Got:      {:?}", g);
                println!("    Expected: {:?}", want);
            }
        }
    }
    println!();

    // === Test 11: i64 warp prefix sums ===
    println!("=== Test 11: scan_sums_i64 (64-bit, two-pass shuffle; identity -7; Tile<8>) ===");
    {
        let input: Vec<i64> = (0..WARP_SIZE as i64)
            .map(|t| {
                let m = (t << 33) + 0xC000_0000 + t + 1;
                if t % 2 == 1 {
                    -m
                } else {
                    m
                }
            })
            .collect();
        let dev_in = stream.memcpy_stod(&input)?;
        let dev_incl = stream.memcpy_stod(&[0i64; WARP_SIZE])?;
        let dev_excl = stream.memcpy_stod(&[0i64; WARP_SIZE])?;
        let dev_tile8 = stream.memcpy_stod(&[0i64; WARP_SIZE])?;
        unsafe {
            stream
                .launch_builder(&k.scan_sums_i64)
                .arg(&dev_in)
                .arg(&dev_incl)
                .arg(&dev_excl)
                .arg(&dev_tile8)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let incl = stream.memcpy_dtov(&dev_incl)?;
        let excl = stream.memcpy_dtov(&dev_excl)?;
        let tile8 = stream.memcpy_dtov(&dev_tile8)?;
        let want_tile8: Vec<i64> = (0..WARP_SIZE)
            .map(|t| input[t & !7..=t].iter().sum())
            .collect();
        let want_incl: Vec<i64> = input
            .iter()
            .scan(0, |a, &v| {
                *a += v;
                Some(*a)
            })
            .collect();
        let want_excl: Vec<i64> = (0..WARP_SIZE)
            .map(|t| if t == 0 { -7 } else { want_incl[t] - input[t] })
            .collect();
        for (name, got, want) in [
            ("Tile<8>", &tile8, &want_tile8),
            ("inclusive", &incl, &want_incl),
            ("exclusive", &excl, &want_excl),
        ] {
            let pass = got == want;
            scan_failures += usize::from(!pass);
            println!("  {name}: {}", if pass { "PASS" } else { "FAIL" });
            if !pass {
                println!("    Got:      {:?}", got);
                println!("    Expected: {:?}", want);
            }
        }
    }
    println!();

    if scan_failures > 0 {
        return Err(format!("{scan_failures} scan check(s) failed").into());
    }
    println!("All kernels compiled from cargo, type-checked at build time,");
    println!("and executed on real GPU hardware.");

    Ok(())
}
