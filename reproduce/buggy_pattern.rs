//! This file INTENTIONALLY DOES NOT COMPILE.
//!
//! It demonstrates the cuda-samples #398 bug pattern in our type system.
//! The compiler rejects it — that's the point.
//!
//! Expected error:
//!   no method named `shuffle_xor` found for `Warp<Even>`

use warp_types::prelude::*;

/// The bug: shuffle after diverge, without merging first.
///
/// In CUDA, this compiles silently and produces wrong results.
/// In warp-types, the compiler catches it.
fn shuffle_after_diverge() {
    let warp: Warp<All> = Warp::kernel_entry();
    let data = PerLane::new(1i32);

    // Diverge: warp splits into two sub-warps
    let (evens, _odds) = warp.diverge_even_odd();

    // THE BUG: shuffle on a diverged sub-warp.
    // Only even lanes are active. Odd lanes have stale data.
    // CUDA compiles this. Our type system does not.
    let _wrong = evens.shuffle_xor(data, 1);
    //                  ^^^^^^^^^^^
    // error[E0599]: no method named `shuffle_xor` found for `Warp<Even>`
}

fn main() {
    shuffle_after_diverge();
}
