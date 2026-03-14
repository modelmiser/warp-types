//! Example GPU kernels using warp-types.
//!
//! This crate is compiled for nvptx64-nvidia-cuda by the host's build.rs.
//! The type system prevents shuffle-from-inactive-lane bugs at compile time.

#![no_std]
#![no_main]
#![feature(abi_ptx, asm_experimental_arch)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }

use warp_types::*;
use warp_types_kernel::warp_kernel;

// ============================================================================
// Kernel 1: Butterfly reduction (all ones → 32)
// ============================================================================

#[warp_kernel]
pub fn butterfly_reduce(data: *mut i32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let mut val = *data.add(tid as usize);

    // Type system guarantees: warp is Warp<All>, so shuffle_xor is available.
    // If we diverge first, shuffle_xor would not exist on the sub-warp type.
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

    *data.add(tid as usize) = val;
}

// ============================================================================
// Kernel 2: Diverge + merge + reduce (type system erased at PTX level)
// ============================================================================

#[warp_kernel]
pub fn diverge_merge_reduce(data: *mut i32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let mut val = *data.add(tid as usize);

    // Diverge consumes the warp handle — shuffle is impossible here
    let (evens, odds) = warp.diverge_even_odd();
    // evens.shuffle_xor(...)  ← COMPILE ERROR: method not found on Warp<Even>

    // Merge restores Warp<All> — shuffle is available again
    let warp: Warp<All> = merge(evens, odds);

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

    *data.add(tid as usize) = val;
}

// ============================================================================
// Kernel 3: Reduction with scalar parameter
// ============================================================================

#[warp_kernel]
pub fn reduce_n(input: *const i32, output: *mut i32, n: u32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();

    // All lanes participate — inactive lanes contribute 0
    let mut val = if tid < n { *input.add(tid as usize) } else { 0 };

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

    if tid == 0 {
        *output = val;
    }
}
