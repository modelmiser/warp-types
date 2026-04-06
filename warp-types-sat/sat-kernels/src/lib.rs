//! GPU kernels for clause-parallel gradient SAT (Axis 1).
//!
//! Compiled to PTX by warp-types-builder. Each kernel uses warp-types
//! type-safe shuffle operations — the type system prevents inactive-lane
//! reads at compile time.
//!
//! # Kernel: clause_loss_reduce
//!
//! One thread per clause. Each warp (32 threads) computes 32 clause losses
//! in parallel, then butterfly-reduces to get the batch sum. Lane 0 of
//! each warp writes the partial sum to output[warp_index].
//!
//! Grid: (num_batches, 1, 1), Block: (32, 1, 1)
//!
//! The math is identical to `gpu_gradient.rs::total_loss_simwarp()` —
//! SimWarp tests validate the exact same computation path.

#![no_std]
#![no_main]
#![feature(abi_ptx, asm_experimental_arch)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! { loop {} }

use warp_types::*;

// ============================================================================
// Kernel 1: Clause-parallel loss evaluation + butterfly reduce
//
// Axis 1 from the gradient design doc:
//   PerLane<f64>  →  per-clause loss (3 loads + 2 multiplies + 1 weight mul)
//   reduce_sum    →  batch loss aggregation (5 butterfly shuffle-XOR steps)
//   Uniform<f64>  →  batch total (same in every lane)
// ============================================================================

/// Compute total clause loss using warp-parallel evaluation.
///
/// Each thread evaluates one clause's product-form loss:
///   loss = weight * term0 * term1 * term2
/// where term_i = x[var_i] if negated, (1 - x[var_i]) if positive.
///
/// Butterfly reduce sums across the warp. Lane 0 writes per-batch partial sum.
///
/// # Parameters
/// - `vars0/1/2`: SoA variable indices for literal positions 0, 1, 2
/// - `negs0/1/2`: SoA negation flags (0 = positive, nonzero = negated)
/// - `weights`: Per-clause weights (f64)
/// - `x`: Variable values in [0, 1] (f64)
/// - `output`: Per-warp-batch partial sums (one f64 per block)
/// - `num_clauses`: Number of real clauses (padding clauses have weight 0)
#[warp_kernel]
pub fn clause_loss_reduce(
    vars0: *const u32,
    vars1: *const u32,
    vars2: *const u32,
    negs0: *const u32,
    negs1: *const u32,
    negs2: *const u32,
    weights: *const f64,
    x: *const f64,
    output: *mut f64,
    num_clauses: u32,
) {
    let warp: Warp<All> = Warp::kernel_entry();

    // Global clause index: blockIdx.x * 32 + threadIdx.x
    let bid = warp_types::gpu::block_id_x();
    let tid = warp_types::gpu::thread_id_x();
    let ci = bid * 32 + tid;

    // Load clause data (coalesced: consecutive threads read consecutive addresses)
    let v0 = *vars0.add(ci as usize);
    let v1 = *vars1.add(ci as usize);
    let v2 = *vars2.add(ci as usize);
    let n0 = *negs0.add(ci as usize);
    let n1 = *negs1.add(ci as usize);
    let n2 = *negs2.add(ci as usize);
    let w = *weights.add(ci as usize);

    // Load variable values (scattered reads, cached in L1)
    let x0 = *x.add(v0 as usize);
    let x1 = *x.add(v1 as usize);
    let x2 = *x.add(v2 as usize);

    // Product-form loss: weight * term0 * term1 * term2
    // Falseness term: positive lit → (1 - x), negated lit → x
    let t0 = if n0 != 0 { x0 } else { 1.0 - x0 };
    let t1 = if n1 != 0 { x1 } else { 1.0 - x1 };
    let t2 = if n2 != 0 { x2 } else { 1.0 - x2 };
    let loss = w * t0 * t1 * t2;

    // Butterfly reduce: 5 shuffle-XOR stages sum 32 lanes → same value in all lanes
    let batch_sum = warp.reduce_sum(data::PerLane::new(loss));

    // Lane 0 of each warp writes the partial sum
    if tid == 0 {
        *output.add(bid as usize) = batch_sum.get();
    }
}

// ============================================================================
// Kernel 2: Fused loss + gradient (loss reduce + 3 atomicAdd per clause)
//
// Same loss computation as kernel 1, plus gradient scatter:
//   d(loss_c)/d(x_v) = weight * sign(lit_v) * PROD_{j≠v} term(lit_j)
//   sign = +1 if negated, -1 if positive
//
// Each thread atomicAdds its 3 gradient contributions to grad[var].
// Zero the grad array before launch (host-side memset).
// ============================================================================

/// Fused loss evaluation + gradient accumulation.
///
/// Combines `clause_loss_reduce` with per-clause gradient scatter.
/// Each thread computes loss AND 3 gradient contributions, atomicAdding
/// them to a per-variable gradient array. Eliminates the CPU gradient
/// pass entirely for large instances.
///
/// # Additional parameters
/// - `grad`: Per-variable gradient accumulator (num_vars f64s, zeroed before launch)
#[warp_kernel]
pub fn clause_loss_grad_fused(
    vars0: *const u32,
    vars1: *const u32,
    vars2: *const u32,
    negs0: *const u32,
    negs1: *const u32,
    negs2: *const u32,
    weights: *const f64,
    x: *const f64,
    output: *mut f64,
    grad: *mut f64,
    num_clauses: u32,
) {
    let warp: Warp<All> = Warp::kernel_entry();

    let bid = warp_types::gpu::block_id_x();
    let tid = warp_types::gpu::thread_id_x();
    let ci = bid * 32 + tid;

    // Load clause data (coalesced)
    let v0 = *vars0.add(ci as usize);
    let v1 = *vars1.add(ci as usize);
    let v2 = *vars2.add(ci as usize);
    let n0 = *negs0.add(ci as usize);
    let n1 = *negs1.add(ci as usize);
    let n2 = *negs2.add(ci as usize);
    let w = *weights.add(ci as usize);

    // Load variable values (scattered, L1 cached)
    let x0 = *x.add(v0 as usize);
    let x1 = *x.add(v1 as usize);
    let x2 = *x.add(v2 as usize);

    // Falseness terms
    let t0 = if n0 != 0 { x0 } else { 1.0 - x0 };
    let t1 = if n1 != 0 { x1 } else { 1.0 - x1 };
    let t2 = if n2 != 0 { x2 } else { 1.0 - x2 };
    let loss = w * t0 * t1 * t2;

    // ── Loss: butterfly reduce (same as kernel 1) ──
    let batch_sum = warp.reduce_sum(data::PerLane::new(loss));
    if tid == 0 {
        *output.add(bid as usize) = batch_sum.get();
    }

    // ── Gradient: 3 atomicAdds per clause ──
    // d(loss_c)/d(x_v) = sign_v * weight * PROD_{j≠v} term_j
    // Only real clauses contribute (padding has w=0, so grad contribution = 0).
    // We skip the atomics for padding to avoid unnecessary contention.
    if (ci as u32) < num_clauses {
        let s0 = if n0 != 0 { 1.0f64 } else { -1.0 };
        let s1 = if n1 != 0 { 1.0f64 } else { -1.0 };
        let s2 = if n2 != 0 { 1.0f64 } else { -1.0 };

        warp_types::gpu::atomic_add_f64(grad.add(v0 as usize), w * s0 * t1 * t2);
        warp_types::gpu::atomic_add_f64(grad.add(v1 as usize), w * s1 * t0 * t2);
        warp_types::gpu::atomic_add_f64(grad.add(v2 as usize), w * s2 * t0 * t1);
    }
}

// ============================================================================
// Kernel 3: Variable update with momentum (device-resident)
//
// Each thread updates one variable:
//   velocity[i] = momentum * velocity[i] + grad[i]
//   x[i] = clamp(x[i] - lr * velocity[i], 0.0, 1.0)
//
// Keeps x, grad, velocity on device — eliminates per-iteration host↔device
// transfer of grad (download) and x (upload).
// ============================================================================

/// Per-variable gradient step with momentum, entirely on GPU.
///
/// Grid: (ceil(num_vars/32), 1, 1), Block: (32, 1, 1)
/// Out-of-bounds threads (vi >= num_vars) are no-ops.
#[warp_kernel]
pub fn variable_update(
    x: *mut f64,
    grad: *const f64,
    velocity: *mut f64,
    lr_ptr: *const f64,
    momentum_ptr: *const f64,
    num_vars: u32,
) {
    let _warp: Warp<All> = Warp::kernel_entry();

    let bid = warp_types::gpu::block_id_x();
    let tid = warp_types::gpu::thread_id_x();
    let vi = bid * 32 + tid;

    if vi < num_vars {
        let lr = *lr_ptr;
        let momentum = *momentum_ptr;
        let g = *grad.add(vi as usize);
        let v_old = *velocity.add(vi as usize);
        let v_new = momentum * v_old + g;
        *velocity.add(vi as usize) = v_new;

        let x_old = *x.add(vi as usize);
        let x_new = x_old - lr * v_new;
        // Clamp to [0, 1]
        let x_clamped = if x_new < 0.0 {
            0.0
        } else if x_new > 1.0 {
            1.0
        } else {
            x_new
        };
        *x.add(vi as usize) = x_clamped;
    }
}

// ============================================================================
// Kernel 4: Gradient norm reduction (butterfly reduce of grad²)
//
// Each thread loads grad[vi], squares it, butterfly reduces per warp.
// Lane 0 of each warp writes the partial sum to output[warp_index].
// Host sums the partial sums to get grad_norm_sq.
//
// This kernel reads grad[] — same data that variable_update reads.
// kernel-fuse Pattern 5 (multi-pass reduction): fusing these two into
// variable_update_with_norm eliminates one full pass over grad[].
// ============================================================================

/// Reduce sum of squared gradients for convergence check.
///
/// Grid: (ceil(num_vars/32), 1, 1), Block: (32, 1, 1)
/// Out-of-bounds threads contribute 0.0 to the reduction.
#[warp_kernel]
pub fn grad_norm_reduce(
    grad: *const f64,
    output: *mut f64,
    num_vars: u32,
) {
    let warp: Warp<All> = Warp::kernel_entry();

    let bid = warp_types::gpu::block_id_x();
    let tid = warp_types::gpu::thread_id_x();
    let vi = bid * 32 + tid;

    let g_sq = if vi < num_vars {
        let g = *grad.add(vi as usize);
        g * g
    } else {
        0.0
    };

    let batch_sum = warp.reduce_sum(data::PerLane::new(g_sq));

    if tid == 0 {
        *output.add(bid as usize) = batch_sum.get();
    }
}
