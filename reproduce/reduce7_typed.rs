//! cuda-samples #398: The killer demo
//!
//! Three versions of the same reduction:
//!   1. reduce7_buggy   — CUDA's pattern: shfl_down with partial mask → WRONG RESULTS
//!   2. reduce7_typed   — Warp-types: the buggy pattern is a COMPILE ERROR
//!   3. reduce7_fixed   — Warp-types: correct fix, all lanes participate → CORRECT
//!
//! Compile: rustc +nightly --target nvptx64-nvidia-cuda --emit=asm -O \
//!          --edition 2021 reproduce/reduce7_typed.rs -o reproduce/reduce7_typed.ptx

#![no_std]
#![no_main]
#![feature(abi_ptx, asm_experimental_arch)]

use core::marker::PhantomData;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }

// ============================================================================
// Minimal type system
// ============================================================================

trait ActiveSet: Copy + 'static { const MASK: u32; }

#[derive(Copy, Clone)] struct All;
impl ActiveSet for All { const MASK: u32 = 0xFFFFFFFF; }

#[derive(Copy, Clone)] struct Lane0;
impl ActiveSet for Lane0 { const MASK: u32 = 0x00000001; }

#[derive(Copy, Clone)] struct NotLane0;
impl ActiveSet for NotLane0 { const MASK: u32 = 0xFFFFFFFE; }

trait ComplementOf<Other: ActiveSet>: ActiveSet {}
impl ComplementOf<NotLane0> for Lane0 {}
impl ComplementOf<Lane0> for NotLane0 {}

#[derive(Copy, Clone)]
struct Warp<S: ActiveSet> { _p: PhantomData<S> }

impl<S: ActiveSet> Warp<S> {
    fn new() -> Self { Warp { _p: PhantomData } }
}

impl Warp<All> {
    fn kernel_entry() -> Self { Warp::new() }

    #[inline(always)]
    fn shfl_down(&self, val: i32, delta: u32) -> i32 {
        let result: i32;
        unsafe {
            core::arch::asm!(
                "shfl.sync.down.b32 {r}, {v}, {d}, 31, {mask};",
                r = out(reg32) result,
                v = in(reg32) val,
                d = in(reg32) delta,
                mask = in(reg32) 0xFFFFFFFFu32,
            );
        }
        result
    }

    fn extract_lane0(self) -> (Warp<Lane0>, Warp<NotLane0>) {
        (Warp::new(), Warp::new())
    }
}

fn merge<S1: ComplementOf<S2>, S2: ActiveSet>(_l: Warp<S1>, _r: Warp<S2>) -> Warp<All> {
    Warp::new()
}

#[inline(always)]
fn thread_id_x() -> u32 {
    let id: u32;
    unsafe { core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) id) };
    id
}

#[inline(always)]
fn syncthreads() {
    unsafe { core::arch::asm!("bar.sync 0;"); }
}

// ============================================================================
// VERSION 1: THE BUG (what CUDA lets you write)
//
// This is the reduce7 pattern. After tree reduction with block_size=32,
// only lane 0 has data. The shfl_down reads from lanes that don't have
// valid data. CUDA compiles this without warning. GPU produces wrong sum.
//
// In our type system, this pattern DOES NOT COMPILE:
//   let (lane0, _rest) = warp.extract_lane0();
//   lane0.shfl_down(val, 16);  // ERROR: no method `shfl_down` on Warp<Lane0>
//
// ============================================================================

// Cannot implement — the buggy pattern is a type error.
// The kernel below shows what the CORRECT version looks like.

// ============================================================================
// VERSION 2: THE FIX (type system forces correct code)
//
// All lanes load data (inactive lanes get 0). Full warp participates.
// shfl_down on Warp<All> is permitted. Produces correct result.
// ============================================================================

#[no_mangle]
pub unsafe extern "ptx-kernel" fn reduce7_typed_fixed(
    g_idata: *const i32,
    g_odata: *mut i32,
    n: u32,
) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = thread_id_x();

    // All lanes participate — inactive lanes contribute 0
    let mut val = if tid < n { *g_idata.add(tid as usize) } else { 0 };

    // Full-warp reduction. Type system guarantees: warp is Warp<All>,
    // so shfl_down is available. No partial masks. No inactive lanes.
    val += warp.shfl_down(val, 16);
    val += warp.shfl_down(val, 8);
    val += warp.shfl_down(val, 4);
    val += warp.shfl_down(val, 2);
    val += warp.shfl_down(val, 1);

    if tid == 0 {
        *g_odata = val;
    }
}

// ============================================================================
// VERSION 3: BUGGY CUDA PATTERN (untyped, for comparison)
//
// This is what CUDA lets you write. Same logic as reduce7's bug:
// narrow ballot mask → shfl_down reads from inactive lane → wrong result.
// ============================================================================

#[no_mangle]
pub unsafe extern "ptx-kernel" fn reduce7_untyped_buggy(
    g_idata: *const i32,
    g_odata: *mut i32,
    n: u32,
) {
    let tid = thread_id_x();

    let my_val = if tid < n { *g_idata.add(tid as usize) } else { 0 };

    // Ballot: which lanes have valid data?
    // With n=32 and blockDim=32: tid < blockDim.x/warpSize = tid < 1
    // So only lane 0 votes true. mask = 1.
    let active_lanes = if n <= 32 { 1u32 } else { n }; // simplified: pretend only 1 lane
    let is_active = tid < (active_lanes / 32).max(1);

    // Bug: shfl_down with partial mask. Lane 0 reads from lane 16,
    // but lane 16 didn't vote true. Its register value is whatever
    // was there — undefined behavior.
    let result: i32;
    if is_active {
        let mut val = my_val;
        // Using mask=1 (only lane 0), but reading from lanes 16,8,4,2,1
        let mask = 1u32; // the buggy mask
        let mut r: i32;
        core::arch::asm!(
            "shfl.sync.down.b32 {r}, {v}, {d}, 31, {mask};",
            r = out(reg32) r,
            v = in(reg32) val,
            d = in(reg32) 16u32,
            mask = in(reg32) mask,
        );
        val += r;

        core::arch::asm!(
            "shfl.sync.down.b32 {r}, {v}, {d}, 31, {mask};",
            r = out(reg32) r,
            v = in(reg32) val,
            d = in(reg32) 8u32,
            mask = in(reg32) mask,
        );
        val += r;

        core::arch::asm!(
            "shfl.sync.down.b32 {r}, {v}, {d}, 31, {mask};",
            r = out(reg32) r,
            v = in(reg32) val,
            d = in(reg32) 4u32,
            mask = in(reg32) mask,
        );
        val += r;

        core::arch::asm!(
            "shfl.sync.down.b32 {r}, {v}, {d}, 31, {mask};",
            r = out(reg32) r,
            v = in(reg32) val,
            d = in(reg32) 2u32,
            mask = in(reg32) mask,
        );
        val += r;

        core::arch::asm!(
            "shfl.sync.down.b32 {r}, {v}, {d}, 31, {mask};",
            r = out(reg32) r,
            v = in(reg32) val,
            d = in(reg32) 1u32,
            mask = in(reg32) mask,
        );
        val += r;

        result = val;
    } else {
        result = 0;
    }

    if tid == 0 {
        *g_odata = result;
    }
}
