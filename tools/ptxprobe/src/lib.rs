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

/// API-surface probe: every non-generic public method a kernel might call. The
/// script fails on any `.extern .func`, so a method that stops being inlined
/// across the crate boundary (the 2026-10-18 `ballot` failure, and
/// `BallotResult::first_lane` before 0.4.2) is caught here without a GPU.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn probe_api_surface(out: *mut u32) {
    let w: Warp<All> = Warp::kernel_entry();
    let tid = warp_types::gpu::thread_id_x();
    let mut acc: u32 = w.active_mask() as u32 ^ w.population();
    acc ^= w.active_set_name().len() as u32;
    w.sync();
    // BallotResult
    let b = w.ballot(data::PerLane::new(tid.is_multiple_of(3)));
    acc ^= b.mask_u32().get() ^ b.popcount().get();
    acc ^= b.lane_voted(data::LaneId::new(5)).get() as u32;
    acc ^= b.first_lane().map(|l| l.get() as u32).unwrap_or(99);
    acc ^= b.mask().get() as u32;
    // idx shuffle (broadcast) and the width-confined tile shuffles
    acc ^= w.broadcast_lane(data::PerLane::new(tid as i32), 3).get() as u32;
    let t16: Tile<16> = w.tile();
    acc ^= t16.shuffle_xor(data::PerLane::new(tid as i32), 1).get() as u32;
    acc ^= t16.partition_8().shuffle_down(data::PerLane::new(tid as i32), 1).get() as u32;
    acc ^= t16.partition_4().size() as u32;
    // dynamic divergence and merge
    let w2: Warp<All> = w.diverge_dynamic(0x0000_FFFF).merge();
    acc ^= w2.reduce_add(data::PerLane::new(tid as i32)) as u32;
    // gradual: DynWarp round trip
    let d = DynWarp::from_mask_32(0xFFFF_FFFF);
    acc ^= d.population() ^ d.active_mask() as u32;
    acc ^= d.shuffle_xor_scalar(tid as i32, 1).unwrap_or(0) as u32;
    acc ^= d.shuffle_down_scalar(tid as i32, 1).unwrap_or(0) as u32;
    acc ^= d.reduce_sum_scalar(1).unwrap_or(0) as u32;
    acc ^= d.broadcast_scalar(tid as i32).unwrap_or(0) as u32;
    let w3: Warp<All> = d.ascribe::<All>().unwrap();
    acc ^= w3.shuffle_down(data::PerLane::new(tid as i32), 1).get() as u32;
    // direct primitives
    warp_types::gpu::syncwarp(0xFFFF_FFFF);
    acc ^= warp_types::gpu::shfl_sync_idx_i32(0xFFFF_FFFF, tid as i32, 7) as u32;
    *out.add(tid as usize) = acc;
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
