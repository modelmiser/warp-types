//! Rust-to-PTX zero-overhead proof.
//!
//! Compiles to PTX via: rustc +nightly --target nvptx64-nvidia-cuda --emit=asm -O
//!
//! Contains two functions:
//!   butterfly_untyped: raw data passing (no type system)
//!   butterfly_typed: same operations through Warp<All> phantom types
//!
//! If the type system has zero overhead, both emit identical PTX.

#![no_std]
#![no_main]
#![feature(abi_ptx)]

use core::marker::PhantomData;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }

// ============================================================================
// Type system (mirrors warp-types crate)
// ============================================================================

trait ActiveSet: Copy + 'static {
    const MASK: u32;
}

#[derive(Copy, Clone)]
struct All;
impl ActiveSet for All { const MASK: u32 = 0xFFFFFFFF; }

#[derive(Copy, Clone)]
struct Even;
impl ActiveSet for Even { const MASK: u32 = 0x55555555; }

#[derive(Copy, Clone)]
struct Odd;
impl ActiveSet for Odd { const MASK: u32 = 0xAAAAAAAA; }

trait ComplementOf<Other: ActiveSet>: ActiveSet {}
impl ComplementOf<Odd> for Even {}
impl ComplementOf<Even> for Odd {}

#[derive(Copy, Clone)]
struct Warp<S: ActiveSet> { _marker: PhantomData<S> }

impl<S: ActiveSet> Warp<S> {
    fn new() -> Self { Warp { _marker: PhantomData } }
}

impl Warp<All> {
    fn kernel_entry() -> Self { Warp::new() }

    // Shuffle: only available on Warp<All>
    #[inline(always)]
    fn shuffle_xor(self, data: i32, _mask: u32) -> i32 { data }

    fn diverge_even_odd(self) -> (Warp<Even>, Warp<Odd>) {
        (Warp::new(), Warp::new())
    }
}

fn merge<S1: ComplementOf<S2>, S2: ActiveSet>(_l: Warp<S1>, _r: Warp<S2>) -> Warp<All> {
    Warp::new()
}

// ============================================================================
// UNTYPED: raw data passing, no phantom types
// ============================================================================

#[no_mangle]
#[inline(never)]
pub fn butterfly_untyped(data: i32) -> i32 {
    let d1 = data.wrapping_add(data);  // simulate shuffle_xor + add
    let d2 = d1.wrapping_add(d1);
    let d3 = d2.wrapping_add(d2);
    let d4 = d3.wrapping_add(d3);
    let d5 = d4.wrapping_add(d4);
    d5
}

// ============================================================================
// TYPED: same operations, through Warp<All> phantom type
// ============================================================================

#[no_mangle]
#[inline(never)]
pub fn butterfly_typed(data: i32) -> i32 {
    let warp: Warp<All> = Warp::kernel_entry();

    // These go through Warp<All>::shuffle_xor which requires Warp<All>
    let d1 = warp.shuffle_xor(data, 16).wrapping_add(data);
    let d2 = warp.shuffle_xor(d1, 8).wrapping_add(d1);
    let d3 = warp.shuffle_xor(d2, 4).wrapping_add(d2);
    let d4 = warp.shuffle_xor(d3, 2).wrapping_add(d3);
    let d5 = warp.shuffle_xor(d4, 1).wrapping_add(d4);
    d5
}

// ============================================================================
// TYPED + DIVERGE/MERGE: type system exercises diverge and merge
// ============================================================================

#[no_mangle]
#[inline(never)]
pub fn diverge_merge_typed(data: i32) -> i32 {
    let warp: Warp<All> = Warp::kernel_entry();
    let (evens, odds) = warp.diverge_even_odd();
    let _merged: Warp<All> = merge(evens, odds);
    data  // just pass through
}

#[no_mangle]
#[inline(never)]
pub fn diverge_merge_untyped(data: i32) -> i32 {
    data  // no type system, same result
}

// Kernel entry to prevent dead-code elimination
#[no_mangle]
pub unsafe extern "ptx-kernel" fn entry(out: *mut i32) {
    let data = 42i32;
    *out.add(0) = butterfly_untyped(data);
    *out.add(1) = butterfly_typed(data);
    *out.add(2) = diverge_merge_untyped(data);
    *out.add(3) = diverge_merge_typed(data);
}
