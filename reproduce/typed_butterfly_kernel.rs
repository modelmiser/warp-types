//! Real GPU kernel using the warp-types type system.
//!
//! Compiles to PTX with actual shfl.sync.bfly.b32 instructions.
//! The type system (Warp<All>, ActiveSet, ComplementOf) is fully erased;
//! only the shuffle intrinsics remain in the generated PTX.
//!
//! Compile: rustc +nightly --target nvptx64-nvidia-cuda --emit=asm -O \
//!          --edition 2021 reproduce/typed_butterfly_kernel.rs \
//!          -o reproduce/typed_butterfly_kernel.ptx
//!
//! Requires: #![feature(abi_ptx, asm_experimental_arch)]

#![no_std]
#![no_main]
#![feature(abi_ptx, asm_experimental_arch)]

use core::marker::PhantomData;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }

// ============================================================================
// Minimal warp-types: just enough for the kernel
// (In production, this would be `use warp_types::*`)
// ============================================================================

trait ActiveSet: Copy + 'static { const MASK: u32; }

#[derive(Copy, Clone)] struct All;
impl ActiveSet for All { const MASK: u32 = 0xFFFFFFFF; }

#[derive(Copy, Clone)] struct Even;
impl ActiveSet for Even { const MASK: u32 = 0x55555555; }

#[derive(Copy, Clone)] struct Odd;
impl ActiveSet for Odd { const MASK: u32 = 0xAAAAAAAA; }

trait ComplementOf<Other: ActiveSet>: ActiveSet {}
impl ComplementOf<Odd> for Even {}
impl ComplementOf<Even> for Odd {}

#[derive(Copy, Clone)]
struct Warp<S: ActiveSet> { _p: PhantomData<S> }

impl Warp<All> {
    fn kernel_entry() -> Self { Warp { _p: PhantomData } }

    #[inline(always)]
    fn shuffle_xor(&self, val: i32, mask: u32) -> i32 {
        let result: i32;
        unsafe {
            core::arch::asm!(
                "shfl.sync.bfly.b32 {r}, {v}, {m}, 31, {mask};",
                r = out(reg32) result,
                v = in(reg32) val,
                m = in(reg32) mask,
                mask = in(reg32) All::MASK,
            );
        }
        result
    }

    #[inline(always)]
    fn shuffle_down(&self, val: i32, delta: u32) -> i32 {
        let result: i32;
        unsafe {
            core::arch::asm!(
                "shfl.sync.down.b32 {r}, {v}, {d}, 31, {mask};",
                r = out(reg32) result,
                v = in(reg32) val,
                d = in(reg32) delta,
                mask = in(reg32) All::MASK,
            );
        }
        result
    }

    fn diverge_even_odd(self) -> (Warp<Even>, Warp<Odd>) {
        (Warp { _p: PhantomData }, Warp { _p: PhantomData })
    }
}

fn merge<S1: ComplementOf<S2>, S2: ActiveSet>(_l: Warp<S1>, _r: Warp<S2>) -> Warp<All> {
    Warp { _p: PhantomData }
}

#[inline(always)]
fn thread_id_x() -> u32 {
    let id: u32;
    unsafe { core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) id) };
    id
}

// ============================================================================
// Kernel 1: Typed butterfly reduction
// The type system guarantees all lanes are active before any shuffle.
// ============================================================================

#[no_mangle]
pub unsafe extern "ptx-kernel" fn typed_butterfly_reduce(data: *mut i32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = thread_id_x();
    let mut val = *data.add(tid as usize);

    // Type system guarantees: warp is Warp<All>, so shuffle_xor is available
    val += warp.shuffle_xor(val, 16);
    val += warp.shuffle_xor(val, 8);
    val += warp.shuffle_xor(val, 4);
    val += warp.shuffle_xor(val, 2);
    val += warp.shuffle_xor(val, 1);

    *data.add(tid as usize) = val;
}

// ============================================================================
// Kernel 2: Typed diverge + merge + reduce
// Diverge consumes the warp handle. Shuffle is impossible until merge.
// ============================================================================

#[no_mangle]
pub unsafe extern "ptx-kernel" fn typed_diverge_merge_reduce(data: *mut i32) {
    let warp: Warp<All> = Warp::kernel_entry();
    let tid = thread_id_x();
    let mut val = *data.add(tid as usize);

    // Diverge: warp consumed, evens and odds are sub-warps
    let (evens, odds) = warp.diverge_even_odd();
    // evens.shuffle_xor(val, 1);  // COMPILE ERROR if uncommented!

    // Merge: both sub-warps consumed, full warp restored
    let warp: Warp<All> = merge(evens, odds);

    // Now shuffle is available again
    val += warp.shuffle_xor(val, 16);
    val += warp.shuffle_xor(val, 8);
    val += warp.shuffle_xor(val, 4);
    val += warp.shuffle_xor(val, 2);
    val += warp.shuffle_xor(val, 1);

    *data.add(tid as usize) = val;
}
