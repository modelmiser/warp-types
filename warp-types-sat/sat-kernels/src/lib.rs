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
// Only `abi_ptx` is needed: the inline PTX lives in `warp-types`
// (`warp_types::gpu`), not in this crate.
#![feature(abi_ptx)]

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
/// - `_num_clauses`: unused in this kernel — see the note on the parameter
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
    // Unused here, and deliberately so — kept for ABI symmetry with
    // `clause_loss_grad_fused`, which the host launches with the same arg pack.
    // Bounds come from PADDING, not from a guard: `ClauseDataSoA` pads to a
    // multiple of 32 with `var = 0, weight = 0.0`, and the launcher sets
    // `grid_dim = padded_len / 32`, so every `ci` is in range and padding lanes
    // load `x[0]`, which exists whenever any clause does. Padding contributes
    // zero to the sum for FINITE `x`; nothing asserts that, and `0.0 * NaN` is
    // NaN, so a non-finite `x[0]` poisons the whole batch through the padding
    // lanes. Note the bound is the grid, not this parameter — an oversized grid
    // reads past `padded_len` and padding does not save it.
    //
    // What is forbidden here is retiring lanes, NOT bounds checking.
    // `if ci >= num_clauses { return; }` before `reduce_sum` would leave a
    // subset of lanes at a `shfl.sync` whose membermask still names them — the
    // shuffle-from-inactive-lane bug this crate's type system exists to reject.
    // A masked guard is legal and available: predicate the VALUE and reconverge,
    // exactly as `grad_norm_reduce` does below (`if vi < num_vars { .. } else
    // { 0.0 }` and then reduce). Kernel 2's guard sits after the reduce, which
    // is one sufficient pattern rather than the only legal slot.
    _num_clauses: u32,
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
        // Clamp to [0, 1]. `f64::clamp` is in `core`, so it is available on
        // this `no_std` device target, and it matches the hand-rolled form on
        // NaN (both propagate: every comparison against NaN is false).
        let x_clamped = x_new.clamp(0.0, 1.0);
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

// ============================================================================
// Kernel 5: LRAT lemma checking — one warp per lemma (grid-stride)
//
// Device layout = `warp_types_sat::lrat_warp::FlatProof`. Each block is one
// warp; block `b` checks lemmas `b, b + num_blocks, …`. Per hint clause the
// lanes stride over the literals, count the ones not falsified under the
// lemma's running assignment, `reduce_sum` the counts, and if the total is
// one, `ballot` finds the lane that owns the unit and `broadcast_lane`
// hands its literal to the warp. Total zero is the conflict that makes the
// lemma RUP; two or more is a failure.
//
// The running assignment is an open-addressing table in a per-block global
// scratch region (`TABLE` slots of key + stamp). Every lane performs every
// insert — identical writes to identical slots — so no lane ever has to
// read another lane's write and no warp barrier is needed; the stamp is the
// lemma index + 1, so the table needs no clearing between lemmas.
//
// Control flow at every collective is warp-uniform: admissibility, the
// total and the verdict are computed from the same global data by all lanes.
// Codes: 0 Ok, 1 NotUnit, 2 NoConflict, 3 DeletedHint, 4 ForwardHint,
// 5 TableOverflow (same numbering as `lrat_warp::LemmaCode`).
// ============================================================================

/// Open-addressing table capacity per block (power of two).
pub const LRAT_TABLE: u32 = 4096;

#[inline(always)]
fn lrat_code(l: i32) -> u32 {
    ((l.unsigned_abs()) << 1) | (l < 0) as u32
}

#[inline(always)]
fn lrat_hash(k: u32) -> u32 {
    k.wrapping_mul(0x9E37_79B1) & (LRAT_TABLE - 1)
}

/// Is literal `l` falsified in this block's table (stamp-qualified)? The
/// probe count is bounded so a full table cannot spin (it cannot BE full —
/// `lrat_insert` caps occupancy at `LRAT_TABLE - 1` — but a bounded loop is
/// the property a reader can check).
#[inline(always)]
unsafe fn lrat_falsified(keys: *const u32, stamps: *const u32, base: u32, stamp: u32, l: i32) -> bool {
    let k = lrat_code(l) + 1;
    let mut i = lrat_hash(k);
    let mut probes = 0u32;
    while probes < LRAT_TABLE {
        let idx = (base + i) as usize;
        if *stamps.add(idx) != stamp || *keys.add(idx) == 0 {
            return false;
        }
        if *keys.add(idx) == k {
            return true;
        }
        i = (i + 1) & (LRAT_TABLE - 1);
        probes += 1;
    }
    false
}

/// Mark literal `l` false. `count` is this lemma's occupancy; returns false
/// (table overflow, code 5) rather than filling the last slot, so a lookup
/// always finds an empty slot and terminates. Same cap as `SimWarp`'s
/// `LitTable` (`CAP - 1` = 4,095 literals). The previous version capped
/// PROBES at 4,095, which still admitted a 4,096th key whenever the hole was
/// within reach — then the table was full and a miss looped forever
/// (cross-vendor panel, 2026-11-02).
#[inline(always)]
unsafe fn lrat_insert(
    keys: *mut u32,
    stamps: *mut u32,
    base: u32,
    stamp: u32,
    l: i32,
    count: &mut u32,
) -> bool {
    let k = lrat_code(l) + 1;
    let mut i = lrat_hash(k);
    loop {
        let idx = (base + i) as usize;
        if *stamps.add(idx) != stamp || *keys.add(idx) == 0 {
            if *count >= LRAT_TABLE - 1 {
                return false;
            }
            *keys.add(idx) = k;
            *stamps.add(idx) = stamp;
            *count += 1;
            return true;
        }
        if *keys.add(idx) == k {
            return true;
        }
        i = (i + 1) & (LRAT_TABLE - 1);
    }
}

/// Check lemmas `block_id, block_id + num_blocks, …` of a flat LRAT proof.
///
/// Grid: (num_blocks, 1, 1), Block: (32, 1, 1). `table_keys`/`table_stamps`
/// hold `num_blocks * LRAT_TABLE` entries each, zeroed by the host once.
/// `codes[i]` receives lemma `i`'s code.
#[warp_kernel]
pub fn lrat_check_lemmas(
    lits: *const i32,
    clause_start: *const u32,
    deleted_at: *const u32,
    lemma_clause: *const u32,
    lemma_step: *const u32,
    hint_start: *const u32,
    hints: *const u32,
    table_keys: *mut u32,
    table_stamps: *mut u32,
    codes: *mut u8,
    num_lemmas: u32,
    num_blocks: u32,
) {
    let warp: Warp<All> = Warp::kernel_entry();
    let bid = warp_types::gpu::block_id_x();
    let lane = warp_types::gpu::thread_id_x();
    let base = bid * LRAT_TABLE;
    let mut i = bid;
    while i < num_lemmas {
        let stamp = i + 1;
        let lemma_dense = *lemma_clause.add(i as usize);
        let lstep = *lemma_step.add(i as usize);
        let ls = *clause_start.add(lemma_dense as usize);
        let le = *clause_start.add(lemma_dense as usize + 1);
        let mut code: u8 = 2; // NoConflict unless a hint conflicts
        let mut count = 0u32; // this lemma's table occupancy (identical in every lane)
        // Lemma literals are false. Every lane inserts (identical writes).
        let mut k = ls;
        while k < le {
            if !lrat_insert(table_keys, table_stamps, base, stamp, *lits.add(k as usize), &mut count) {
                code = 5;
                break;
            }
            k += 1;
        }
        // Writes before reads. The lanes write identical values to identical
        // slots, so no lane depends on another's write — but without a
        // barrier a lane could in principle observe another lane's
        // key/stamp pair half-updated (two stores, weakly ordered). The
        // barrier is warp-uniform (`code` is the same in every lane).
        warp_types::gpu::syncwarp(0xffff_ffff);
        if code != 5 {
            let hs = *hint_start.add(i as usize);
            let he = *hint_start.add(i as usize + 1);
            let mut h = hs;
            while h < he {
                let hc = *hints.add(h as usize);
                // Admissibility — the same test in every lane.
                if hc >= lemma_dense {
                    code = 4;
                    break;
                }
                if *deleted_at.add(hc as usize) < lstep {
                    code = 3;
                    break;
                }
                let cs = *clause_start.add(hc as usize);
                let ce = *clause_start.add(hc as usize + 1);
                // Each lane strides over the clause.
                let mut free = 0u32;
                let mut last = 0i32;
                let mut j = cs + lane;
                while j < ce {
                    let l = *lits.add(j as usize);
                    if !lrat_falsified(table_keys, table_stamps, base, stamp, l) {
                        free += 1;
                        last = l;
                    }
                    j += 32;
                }
                let total = warp.reduce_sum(data::PerLane::new(free)).get();
                if total == 0 {
                    code = 0; // conflict: RUP
                    break;
                }
                if total >= 2 {
                    code = 1;
                    break;
                }
                // Exactly one free literal in the warp: find its lane, broadcast it.
                // `& 31` keeps the lane provably in range: without it the
                // compiler keeps `broadcast_lane`'s bounds assert for the
                // (unreachable) `voted == 0` case, and that assert drags
                // `core::panicking::panic_fmt` and `Display::fmt` into the
                // PTX as undefined externs — the module then fails to LOAD,
                // before any lemma is checked (RunPod H200, 2026-11-02).
                let voted = warp.ballot(data::PerLane::new(free > 0)).mask_u32().get();
                let src = voted.trailing_zeros() & 31;
                let unit = warp.broadcast_lane(data::PerLane::new(last), src).get();
                let ok = lrat_insert(table_keys, table_stamps, base, stamp, -unit, &mut count);
                warp_types::gpu::syncwarp(0xffff_ffff);
                if !ok {
                    code = 5;
                    break;
                }
                h += 1;
            }
        }
        if lane == 0 {
            *codes.add(i as usize) = code;
        }
        i += num_blocks;
    }
}
