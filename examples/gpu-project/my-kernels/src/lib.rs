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

use warp_types::*; // includes warp_kernel via re-export

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

// ============================================================================
// Kernel 4: Bitonic sort (32 elements, one per lane)
//
// The type system guarantees every shuffle_xor operates on Warp<All>.
// 15 compare-and-swap steps, each using shuffle_xor for lane exchange.
// ============================================================================

#[warp_kernel]
pub fn bitonic_sort_i32(data: *mut i32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let mut val = *data.add(tid as usize);

    // Standard bitonic sort: for stage k and substep j:
    //   partner = shuffle_xor(val, j)
    //   ascending = (tid & k) == 0     ← block-level sort direction
    //   lower     = (tid & j) == 0     ← am I the lower lane in this pair?
    //   want_min  = ascending == lower  ← XOR: ascending+lower→min, etc.
    //   swap if (want_min && val > partner) || (!want_min && val < partner)
    //
    // The type system ensures warp is Warp<All> — every shuffle reads active lanes.

    // Helper: compare-and-swap for one step
    // Using a macro-like inline approach since closures can't capture mutable borrows
    // in no_std + ptx-kernel context.

    // k=2, j=1
    let p = warp.shuffle_xor(data::PerLane::new(val), 1).get();
    let w = ((tid & 2) == 0) == ((tid & 1) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }

    // k=4, j=2
    let p = warp.shuffle_xor(data::PerLane::new(val), 2).get();
    let w = ((tid & 4) == 0) == ((tid & 2) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=4, j=1
    let p = warp.shuffle_xor(data::PerLane::new(val), 1).get();
    let w = ((tid & 4) == 0) == ((tid & 1) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }

    // k=8, j=4
    let p = warp.shuffle_xor(data::PerLane::new(val), 4).get();
    let w = ((tid & 8) == 0) == ((tid & 4) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=8, j=2
    let p = warp.shuffle_xor(data::PerLane::new(val), 2).get();
    let w = ((tid & 8) == 0) == ((tid & 2) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=8, j=1
    let p = warp.shuffle_xor(data::PerLane::new(val), 1).get();
    let w = ((tid & 8) == 0) == ((tid & 1) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }

    // k=16, j=8
    let p = warp.shuffle_xor(data::PerLane::new(val), 8).get();
    let w = ((tid & 16) == 0) == ((tid & 8) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=16, j=4
    let p = warp.shuffle_xor(data::PerLane::new(val), 4).get();
    let w = ((tid & 16) == 0) == ((tid & 4) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=16, j=2
    let p = warp.shuffle_xor(data::PerLane::new(val), 2).get();
    let w = ((tid & 16) == 0) == ((tid & 2) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=16, j=1
    let p = warp.shuffle_xor(data::PerLane::new(val), 1).get();
    let w = ((tid & 16) == 0) == ((tid & 1) == 0);
    if (w && val > p) || (!w && val < p) { val = p; }

    // k=32, j=16 (tid & 32 == 0 always true for 32 lanes → all ascending)
    let p = warp.shuffle_xor(data::PerLane::new(val), 16).get();
    let w = (tid & 16) == 0; // simplified: ascending always true
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=32, j=8
    let p = warp.shuffle_xor(data::PerLane::new(val), 8).get();
    let w = (tid & 8) == 0;
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=32, j=4
    let p = warp.shuffle_xor(data::PerLane::new(val), 4).get();
    let w = (tid & 4) == 0;
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=32, j=2
    let p = warp.shuffle_xor(data::PerLane::new(val), 2).get();
    let w = (tid & 2) == 0;
    if (w && val > p) || (!w && val < p) { val = p; }
    // k=32, j=1
    let p = warp.shuffle_xor(data::PerLane::new(val), 1).get();
    let w = (tid & 1) == 0;
    if (w && val > p) || (!w && val < p) { val = p; }

    *data.add(tid as usize) = val;
}
