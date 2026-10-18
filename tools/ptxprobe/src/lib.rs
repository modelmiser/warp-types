//! PTX operand probe. Built only for `nvptx64-nvidia-cuda` by
//! `scripts/check-ptx-operands.sh`, which greps the emitted assembly for the
//! membermask and width operands the typed API promises. No GPU is needed: the
//! point is that "the membermask reaches the instruction" is a property of the
//! emitted PTX, not of a run.
#![no_std]
#![feature(abi_ptx)]
#![allow(clippy::missing_safety_doc)]

use warp_types::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// `shuffle_xor_within` on `Warp<Even>`: membermask must be `Even::MASK`
/// (0x55555555 = 1431655765) and the instruction must sit behind the lane guard.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn probe_within_even(out: *mut i32) {
    let (w, _odd) = Warp::kernel_entry().diverge_even_odd();
    let tid = warp_types::gpu::thread_id_x();
    let v = w.shuffle_xor_within(data::PerLane::new(tid as i32), 2).get();
    *out.add(tid as usize) = v;
}

/// Same on `Warp<Odd>`: membermask `Odd::MASK` (0xAAAAAAAA = -1431655766 as i32).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn probe_within_odd(out: *mut i32) {
    let (_even, w) = Warp::kernel_entry().diverge_even_odd();
    let tid = warp_types::gpu::thread_id_x();
    let v = w.shuffle_xor_within(data::PerLane::new(tid as i32), 2).get();
    *out.add(tid as usize) = v;
}

/// `Tile<8>::shuffle_down`: full-warp membermask (-1) and width operand
/// `((32-8)<<8)|7` = 0x1807 = 6151.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn probe_tile8_down(out: *mut i32) {
    let w: Warp<All> = Warp::kernel_entry();
    let t: Tile<8> = w.tile();
    let tid = warp_types::gpu::thread_id_x();
    let v = t.shuffle_down(data::PerLane::new(tid as i32), 1).get();
    *out.add(tid as usize) = v;
}

/// `Warp<All>::shuffle_up` on `i64`: exactly two `shfl.sync.up.b32`, `c` = 0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn probe_up_i64(out: *mut i64) {
    let w: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let v = w.shuffle_up(data::PerLane::new(tid as i64), 1).get();
    *out.add(tid as usize) = v;
}

/// `Warp<All>::ballot` and the non-generic wrapping reduce: both must be inlined
/// into the entry — the script fails on any `.extern .func` in the PTX, which is
/// what a non-inlined cross-crate device function looks like.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn probe_ballot_and_wrapping(out: *mut u32) {
    let w: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let b = w.ballot(data::PerLane::new(tid < 16)).mask_u32().get();
    let r = w.reduce_sum_wrapping_u32(data::PerLane::new(tid)).get();
    *out.add(tid as usize) = b ^ r;
}

/// Negative probe (feature `empty-probe`): `Warp<Empty>` has no lanes, and the
/// inline const assert in `shuffle_xor_within` must reject this instantiation at
/// codegen. The script expects this build to fail with E0080.
#[cfg(feature = "empty-probe")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn probe_within_empty(out: *mut i32) {
    let w: Warp<Empty> = DynWarp::from_mask_32(0).ascribe::<Empty>().unwrap();
    let tid = warp_types::gpu::thread_id_x();
    let v = w.shuffle_xor_within(data::PerLane::new(tid as i32), 0).get();
    *out.add(tid as usize) = v;
}

/// Negative probe (feature `alias-probe`): a fat pointer hidden behind a type
/// alias. `#[warp_kernel]`'s syntactic check sees a path type; its `const` size
/// assertion must reject it at compile time.
#[cfg(feature = "alias-probe")]
mod alias_probe {
    #[allow(dead_code)]
    type Hidden = [i32];
    #[warp_types_kernel::warp_kernel]
    pub fn probe_alias(p: *mut Hidden) {
        let _ = p;
    }
}
