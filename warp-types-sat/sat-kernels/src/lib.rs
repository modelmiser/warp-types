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
// Device layout = `warp_types_sat::lrat_warp::FlatProof`. Each warp checks
// its own lemma stream (see the kernel's doc). Per hint clause the
// lanes stride over the literals, count the ones not falsified under the
// lemma's running assignment, `reduce_sum` the counts, and if the total is
// one, `ballot` finds the lane that owns the unit and `broadcast_lane`
// hands its literal to the warp. Total zero is the conflict that makes the
// lemma RUP; two or more is a failure.
//
// The running assignment is a table in a per-warp global scratch region of
// `TABLE` 64-bit slots, each `stamp << 32 | key` (key = literal code + 1),
// so one load answers a probe; the stamp is the lemma index + 1, so the
// table needs no clearing between lemmas. Two layouts, chosen per launch by
// the host: HASHED (open addressing on the key) for any proof, and DIRECT
// (slot = literal code, no probing) when every literal code of the proof is
// below `TABLE` — denser and cheaper. Both count occupancy and stop at the
// same cap, so `TableOverflow` falls on the same lemmas either way. Every
// lane performs every insert — identical writes to identical slots — so no
// lane has to read another lane's write except through the barrier below.
//
// Control flow at every collective is warp-uniform: admissibility, the
// total and the verdict are computed from the same global data by all lanes.
// Codes: 0 Ok, 1 NotUnit, 2 NoConflict, 3 DeletedHint, 4 ForwardHint,
// 5 TableOverflow (same numbering as `lrat_warp::LemmaCode`).
// ============================================================================

/// Open-addressing table capacity per warp (power of two).
pub const LRAT_TABLE: u32 = 4096;

/// Warps per block of `lrat_check_lemmas`. Each warp checks its own lemma
/// stream with its own table region; nothing is shared between the warps of
/// a block. More than one per block because a one-warp block caps residency
/// at the per-SM BLOCK limit (24 on sm_89), half the per-SM warp limit.
pub const LRAT_WARPS_PER_BLOCK: u32 = 4;

#[inline(always)]
fn lrat_code(l: i32) -> u32 {
    ((l.unsigned_abs()) << 1) | (l < 0) as u32
}

#[inline(always)]
fn lrat_hash(k: u32) -> u32 {
    k.wrapping_mul(0x9E37_79B1) & (LRAT_TABLE - 1)
}

/// Is literal `l` falsified in this warp's table (stamp-qualified)? In the
/// hashed layout the probe count is bounded so a full table cannot spin (it
/// cannot BE full — `lrat_insert` caps occupancy at `LRAT_TABLE - 1` — but a
/// bounded loop is the property a reader can check).
#[inline(always)]
unsafe fn lrat_falsified(table: *const u64, base: u32, stamp: u32, direct: bool, l: i32) -> bool {
    let k = lrat_code(l) + 1;
    if direct {
        // The caller has already refused codes past the table (it turns them
        // into TableOverflow); the mask keeps even a broken caller inside
        // this warp's region, and the key must match too.
        let e = *table.add((base + ((k - 1) & (LRAT_TABLE - 1))) as usize);
        return (e >> 32) as u32 == stamp && e as u32 == k;
    }
    let mut i = lrat_hash(k);
    let mut probes = 0u32;
    while probes < LRAT_TABLE {
        let e = *table.add((base + i) as usize);
        if (e >> 32) as u32 != stamp {
            return false;
        }
        if e as u32 == k {
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
/// `LitTable` (`CAP - 1` = 4,095 literals), in both layouts. The previous
/// version capped PROBES at 4,095, which still admitted a 4,096th key
/// whenever the hole was within reach — then the table was full and a miss
/// looped forever (cross-vendor panel, 2026-11-02).
#[inline(always)]
unsafe fn lrat_insert(table: *mut u64, base: u32, stamp: u32, direct: bool, l: i32, count: &mut u32) -> bool {
    // The cap comes FIRST, before the key is looked up — exactly as SimWarp's
    // `LitTable::insert`. Checking it only on reaching an empty slot (as this
    // kernel did until 2026-11-04) let a repeated key through at occupancy
    // 4,095 where the model reports TableOverflow (cross-vendor panel).
    if *count >= LRAT_TABLE - 1 {
        return false;
    }
    let k = lrat_code(l) + 1;
    // Direct layout: a code past the table means the host's guarantee is
    // broken — fail closed (TableOverflow: no verdict) rather than alias.
    if direct && k > LRAT_TABLE {
        return false;
    }
    let entry = ((stamp as u64) << 32) | k as u64;
    let mut i = if direct { k - 1 } else { lrat_hash(k) };
    loop {
        let idx = (base + i) as usize;
        let e = *table.add(idx);
        if (e >> 32) as u32 != stamp {
            *table.add(idx) = entry;
            *count += 1;
            return true;
        }
        if e as u32 == k {
            return true;
        }
        if direct {
            // A stamped slot holding another key cannot occur when the slot
            // is the code; fail closed if it ever does.
            return false;
        }
        i = (i + 1) & (LRAT_TABLE - 1);
    }
}

/// Check lemmas `w, w + W, …` of a flat LRAT proof, where `w` is the global
/// warp index (`block_id * LRAT_WARPS_PER_BLOCK + thread_id / 32`) and `W =
/// num_blocks * LRAT_WARPS_PER_BLOCK`.
///
/// Grid: (num_blocks, 1, 1), Block: (32 * LRAT_WARPS_PER_BLOCK, 1, 1).
/// `table` holds `W * LRAT_TABLE` 64-bit slots, zeroed by the host once.
/// `direct != 0` selects the direct layout; the host passes it only when
/// every literal code (`2·|l| + (l < 0)`) is below `LRAT_TABLE`. `codes[i]`
/// receives lemma `i`'s code.
#[warp_kernel]
pub fn lrat_check_lemmas(
    lits: *const i32,
    clause_start: *const u32,
    deleted_at: *const u32,
    lemma_clause: *const u32,
    lemma_step: *const u32,
    hint_start: *const u32,
    hints: *const u32,
    table: *mut u64,
    codes: *mut u8,
    num_lemmas: u32,
    num_blocks: u32,
    direct: u32,
) {
    let direct = direct != 0;
    let warp: Warp<All> = Warp::kernel_entry();
    let bid = warp_types::gpu::block_id_x();
    let tid = warp_types::gpu::thread_id_x();
    let lane = tid & 31;
    let gw = bid * LRAT_WARPS_PER_BLOCK + (tid >> 5);
    let n_warps = num_blocks * LRAT_WARPS_PER_BLOCK;
    let base = gw * LRAT_TABLE;
    let mut i = gw;
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
            if !lrat_insert(table, base, stamp, direct, *lits.add(k as usize), &mut count) {
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
            // Hint metadata, 32 hints at a time, one per lane: the hint's
            // dense id, its deletion step and its clause bounds. Each hint's
            // four values then reach the warp by `broadcast_lane` instead of
            // by a chain of dependent loads. A lane whose hint is forward
            // (`>= lemma_dense`) loads nothing more: the warp rejects that
            // hint before using the other three values.
            let mut batch = hs;
            let (mut m_hc, mut m_del, mut m_cs, mut m_ce) = (0u32, 0u32, 0u32, 0u32);
            while h < he {
                if h == batch {
                    // `lane < he - h`, not `h + lane < he`: no u32 wrap near 2^32.
                    if lane < he - h {
                        let j = h + lane;
                        m_hc = *hints.add(j as usize);
                        if m_hc < lemma_dense {
                            m_del = *deleted_at.add(m_hc as usize);
                            m_cs = *clause_start.add(m_hc as usize);
                            m_ce = *clause_start.add(m_hc as usize + 1);
                        }
                    }
                    batch = h + 32;
                }
                // `& 31`: see the note at the unit broadcast below.
                let src = (h + 32 - batch) & 31;
                let hc = warp.broadcast_lane(data::PerLane::new(m_hc), src).get();
                // Admissibility — the same test in every lane.
                if hc >= lemma_dense {
                    code = 4;
                    break;
                }
                if warp.broadcast_lane(data::PerLane::new(m_del), src).get() < lstep {
                    code = 3;
                    break;
                }
                let cs = warp.broadcast_lane(data::PerLane::new(m_cs), src).get();
                let ce = warp.broadcast_lane(data::PerLane::new(m_ce), src).get();
                // Each lane strides over the clause.
                let mut free = 0u32;
                let mut last = 0i32;
                let mut past_table = false;
                let mut j = cs + lane;
                while j < ce {
                    let l = *lits.add(j as usize);
                    if direct && lrat_code(l) >= LRAT_TABLE {
                        past_table = true; // the host's guarantee is broken
                    } else if !lrat_falsified(table, base, stamp, direct, l) {
                        free += 1;
                        last = l;
                    }
                    j += 32;
                }
                if direct {
                    // Warp-uniform (`direct` is a kernel argument): any lane that
                    // met a code past the table fails the lemma closed.
                    let voted = warp.ballot(data::PerLane::new(past_table)).mask_u32().get();
                    if voted != 0 {
                        code = 5;
                        break;
                    }
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
                let ok = lrat_insert(table, base, stamp, direct, -unit, &mut count);
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
        i += n_warps;
    }
}

/// The largest literal code (`2·|l| + (l < 0)`) in `lits[..n]`, folded into
/// `*out` by atomic maximum (the host zeroes it). Picks the check kernel's
/// table layout for a device-resident proof.
///
/// Grid: (num_blocks, 1, 1), Block: (256, 1, 1); grid-stride.
#[warp_kernel]
pub fn lrat_max_code(lits: *const i32, n: u32, out: *mut u32, num_blocks: u32) {
    let tid = warp_types::gpu::thread_id_x();
    let bid = warp_types::gpu::block_id_x();
    let mut i = bid * 256 + tid;
    let mut m = 0u32;
    while i < n {
        let c = lrat_code(*lits.add(i as usize));
        if c > m {
            m = c;
        }
        i += num_blocks * 256;
    }
    let a = core::sync::atomic::AtomicU32::from_ptr(out);
    a.fetch_max(m, core::sync::atomic::Ordering::Relaxed);
}

// ============================================================================
// LRAT binary decode on the device (research/gpu-lrat-decode.md).
//
// Two kernels with the same inputs and outputs, so the host can time one
// against the other:
//   lrat_scan_slices        — a WARP per slice: 32 bytes per window, one byte
//                             per lane; token ends by ballot, the role machine
//                             by a lane-guarded composition scan, output slots
//                             by popcount. The transcription of
//                             `lrat_warp::decode_model::decode_slice_model`.
//   lrat_scan_slices_serial — a THREAD per slice running the one-thread parse
//                             (the control: "a GPU without the warp").
//
// Both run in two passes selected by `pass`: 1 counts per slice (steps,
// literals, hints, deletion ids) and records the start, the end and a flag;
// the host prefix-sums the counts into offsets; 2 walks again and writes the
// step records and payloads at those offsets. A slice on which the one-thread
// decoder would raise a ParseError (bad tag, varint over 10 bytes, literal out
// of range, bytes running out mid-step) is flagged DIRTY with zero counts and
// left to the host; a slice without a candidate start is flagged NONE.
// ============================================================================

/// Per-slice flag values.
pub const DEC_CLEAN: u32 = 0;
pub const DEC_DIRTY: u32 = 1;
pub const DEC_NONE: u32 = 2;

const DEC_S_TAG: u32 = 0;
const DEC_S_ID: u32 = 1;
const DEC_S_LITS: u32 = 2;
const DEC_S_HINTS: u32 = 3;
const DEC_S_DELS: u32 = 4;
const DEC_S_ERR: u32 = 5;
const DEC_N_STATES: u32 = 6;
const DEC_C_ZERO: u32 = 0;
const DEC_C_A: u32 = 1;
const DEC_C_D: u32 = 2;
const DEC_C_OTHER: u32 = 3;
const DEC_MAX_VARINT: u64 = 10;
const DEC_MAX_VAR: u32 = 1 << 26; // = warp_types_sat::lrat::MAX_VAR

#[inline(always)]
const fn dec_transition(state: u32, class: u32) -> u32 {
    match state {
        DEC_S_TAG => match class {
            DEC_C_A => DEC_S_ID,
            DEC_C_D => DEC_S_DELS,
            _ => DEC_S_ERR,
        },
        DEC_S_ID => DEC_S_LITS,
        DEC_S_LITS => {
            if class == DEC_C_ZERO {
                DEC_S_HINTS
            } else {
                DEC_S_LITS
            }
        }
        DEC_S_HINTS => {
            if class == DEC_C_ZERO {
                DEC_S_TAG
            } else {
                DEC_S_HINTS
            }
        }
        DEC_S_DELS => {
            if class == DEC_C_ZERO {
                DEC_S_TAG
            } else {
                DEC_S_DELS
            }
        }
        _ => DEC_S_ERR,
    }
}

#[inline(always)]
const fn dec_table_for(class: u32) -> u32 {
    let mut t = 0u32;
    let mut s = 0u32;
    while s < DEC_N_STATES {
        t |= dec_transition(s, class) << (3 * s);
        s += 1;
    }
    t
}

#[inline(always)]
const fn dec_table_identity() -> u32 {
    let mut t = 0u32;
    let mut s = 0u32;
    while s < DEC_N_STATES {
        t |= s << (3 * s);
        s += 1;
    }
    t
}

#[inline(always)]
const fn dec_apply(table: u32, state: u32) -> u32 {
    (table >> (3 * state)) & 7
}

/// `then ∘ first`.
#[inline(always)]
const fn dec_compose(first: u32, then: u32) -> u32 {
    let mut t = 0u32;
    let mut s = 0u32;
    while s < DEC_N_STATES {
        t |= dec_apply(then, dec_apply(first, s)) << (3 * s);
        s += 1;
    }
    t
}

#[inline(always)]
const fn dec_lanes_below(lane: u32) -> u32 {
    (1u32 << lane) - 1
}

/// Inclusive composition scan, Hillis–Steele on `shuffle_up` WITH the lane
/// guard — the same shape as the core `Warp::inclusive_sum` (guarded since
/// 2026-11-04), but composing transition tables, which is not a sum.
/// Every lane calls it (the shuffles are collective); the guard only decides
/// whether a lane composes.
#[inline(always)]
fn dec_scan_compose(warp: &Warp<All>, lane: u32, table: u32) -> u32 {
    let mut t = table;
    let mut d = 1u32;
    while d < 32 {
        let up = warp.shuffle_up(data::PerLane::new(t), d).get();
        if lane >= d {
            t = dec_compose(up, t);
        }
        d *= 2;
    }
    t
}

/// Decode slice `si` with the warp formulation. Returns the flag; counts and
/// end through the out-params (identical in every lane).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn dec_slice_warp(
    warp: &Warp<All>,
    lane: u32,
    bytes: *const u8,
    len: u64,
    lo: u64,
    hi: u64,
    pass: u32,
    // pass 2 bases (slice-local records go at base + local index)
    step_base: u32,
    lit_base: u32,
    hint_base: u32,
    del_base: u32,
    st_pos: *mut u64,
    st_id: *mut u64,
    st_after: *mut u64,
    st_lit_end: *mut u32,
    st_hint_end: *mut u32,
    st_del_end: *mut u32,
    st_tag: *mut u8,
    st_rat: *mut u8,
    o_lits: *mut i32,
    o_hints: *mut u64,
    o_del_id: *mut u64,
    o_del_pos: *mut u64,
    out_start: &mut u64,
    out_end: &mut u64,
    out_counts: &mut [u32; 4],
) -> u32 {
    let hi = if hi < len { hi } else { len };
    // ---- the speculative start ----
    let mut start = 0u64;
    if lo != 0 {
        let mut found = false;
        let mut base = lo;
        while base < hi && !found {
            let p = base + lane as u64;
            let cand = p < hi
                && *bytes.add(p as usize - 1) == 0
                && (*bytes.add(p as usize) == b'a' || *bytes.add(p as usize) == b'd');
            let b = warp.ballot(data::PerLane::new(cand)).mask_u32().get();
            if b != 0 {
                start = base + (b.trailing_zeros() & 31) as u64;
                found = true;
            }
            base += 32;
        }
        if !found {
            *out_start = lo;
            *out_end = lo;
            return DEC_NONE;
        }
    }
    *out_start = start;
    let identity = dec_table_identity();
    // warp-uniform carry
    let mut state = DEC_S_TAG;
    let mut tok_start = start;
    let mut c_steps = 0u32;
    let mut c_lits = 0u32;
    let mut c_hints = 0u32;
    let mut c_dels = 0u32;
    let mut c_rat = false;
    let mut base = start;
    loop {
        if base >= len {
            return DEC_DIRTY;
        }
        let p = base + lane as u64;
        let has = p < len;
        let byte = if has { *bytes.add(p as usize) } else { 0u8 };
        let term = has && byte & 0x80 == 0;
        let term_b = warp.ballot(data::PerLane::new(term)).mask_u32().get();
        if term_b == 0 {
            return DEC_DIRTY;
        }
        // ---- this lane's token (if it ends one) ----
        let mut value = 0u64;
        let mut class = DEC_C_OTHER;
        let mut table = identity;
        let mut too_long = false;
        if term {
            let below = term_b & dec_lanes_below(lane);
            let tstart = if below != 0 {
                base + (31 - below.leading_zeros()) as u64 + 1
            } else {
                tok_start
            };
            let n = p - tstart + 1;
            if n > DEC_MAX_VARINT {
                too_long = true;
            } else {
                let mut k = 0u64;
                while k < n {
                    value |= ((*bytes.add((tstart + k) as usize) & 0x7f) as u64) << (7 * k);
                    k += 1;
                }
                class = if n == 1 && byte == b'a' {
                    DEC_C_A
                } else if n == 1 && byte == b'd' {
                    DEC_C_D
                } else if value == 0 {
                    DEC_C_ZERO
                } else {
                    DEC_C_OTHER
                };
                table = dec_table_for(class);
            }
        }
        // A too-long token anywhere in the window is a host error at the
        // latest at its 11th byte; the whole slice is dirty.
        if warp.ballot(data::PerLane::new(too_long)).mask_u32().get() != 0 {
            return DEC_DIRTY;
        }
        // ---- the role scan ----
        let incl = dec_scan_compose(warp, lane, table);
        let excl = warp.shuffle_up(data::PerLane::new(incl), 1).get();
        let before = dec_apply(if lane == 0 { identity } else { excl }, state);
        // ---- this lane's role ----
        let mut is_start = false;
        let mut is_id = false;
        let mut is_lit = false;
        let mut is_hint = false;
        let mut is_rat = false;
        let mut is_del = false;
        let mut is_end = false;
        let mut is_err = false;
        let mut lit_val = 0i32;
        if term {
            match before {
                DEC_S_TAG => {
                    if class == DEC_C_A || class == DEC_C_D {
                        is_start = true;
                    } else {
                        is_err = true;
                    }
                }
                DEC_S_ID => is_id = true,
                DEC_S_LITS => {
                    if value != 0 {
                        is_lit = true;
                        let mag = (value >> 1) as i64;
                        let signed = if value & 1 == 1 { -mag } else { mag };
                        if signed > i32::MAX as i64 || signed < i32::MIN as i64 {
                            is_err = true;
                        } else {
                            lit_val = signed as i32;
                            // Literal 0 (encoded 1) names no variable.
                            if lit_val == 0 || lit_val.unsigned_abs() > DEC_MAX_VAR {
                                is_err = true;
                            }
                        }
                    }
                }
                DEC_S_HINTS => {
                    if value != 0 {
                        is_hint = true;
                        is_rat = value & 1 == 1;
                    } else {
                        is_end = true;
                    }
                }
                DEC_S_DELS => {
                    if value != 0 {
                        is_del = true;
                    } else {
                        is_end = true;
                    }
                }
                _ => is_err = true,
            }
        }
        // ---- the cut ----
        let cut_b = warp
            .ballot(data::PerLane::new(is_end && p + 1 >= hi))
            .mask_u32()
            .get();
        let keep = if cut_b != 0 {
            let c = cut_b.trailing_zeros() & 31;
            dec_lanes_below(c) | (1u32 << c)
        } else {
            u32::MAX
        };
        let err_b = warp.ballot(data::PerLane::new(is_err)).mask_u32().get();
        if err_b & keep != 0 {
            return DEC_DIRTY;
        }
        let start_b = warp.ballot(data::PerLane::new(is_start)).mask_u32().get() & keep;
        let id_b = warp.ballot(data::PerLane::new(is_id)).mask_u32().get() & keep;
        let lit_b = warp.ballot(data::PerLane::new(is_lit)).mask_u32().get() & keep;
        let hint_b = warp.ballot(data::PerLane::new(is_hint)).mask_u32().get() & keep;
        let rat_b = warp.ballot(data::PerLane::new(is_rat)).mask_u32().get() & keep;
        let del_b = warp.ballot(data::PerLane::new(is_del)).mask_u32().get() & keep;
        let end_b = warp.ballot(data::PerLane::new(is_end)).mask_u32().get() & keep;
        // ---- writes (pass 2), slot = running + popcount(ballot & below) ----
        if pass == 2 && keep & (1u32 << lane) != 0 {
            let lb = dec_lanes_below(lane);
            let mine = 1u32 << lane;
            if start_b & mine != 0 {
                let k = step_base + c_steps + (start_b & lb).count_ones();
                *st_pos.add(k as usize) = p;
                *st_tag.add(k as usize) = byte;
            }
            let my_step = c_steps + (start_b & (lb | mine)).count_ones();
            if id_b & mine != 0 {
                let k = step_base + my_step - 1;
                *st_id.add(k as usize) = value >> 1;
                *st_after.add(k as usize) = p + 1;
            }
            if lit_b & mine != 0 {
                *o_lits.add((lit_base + c_lits + (lit_b & lb).count_ones()) as usize) = lit_val;
            }
            if hint_b & mine != 0 {
                *o_hints.add((hint_base + c_hints + (hint_b & lb).count_ones()) as usize) =
                    value >> 1;
            }
            if del_b & mine != 0 {
                let k = (del_base + c_dels + (del_b & lb).count_ones()) as usize;
                *o_del_id.add(k) = value >> 1;
                *o_del_pos.add(k) = p + 1;
            }
            if end_b & mine != 0 {
                let k = (step_base + my_step - 1) as usize;
                let starts_at_or_below = start_b & (lb | mine);
                let rat = if starts_at_or_below == 0 {
                    c_rat || (rat_b & lb) != 0
                } else {
                    let s = 31 - starts_at_or_below.leading_zeros();
                    let since = !dec_lanes_below(s) & !(1u32 << s);
                    (rat_b & since & lb) != 0
                };
                *st_rat.add(k) = rat as u8;
                *st_lit_end.add(k) = c_lits + (lit_b & lb).count_ones();
                *st_hint_end.add(k) = c_hints + (hint_b & lb).count_ones();
                *st_del_end.add(k) = c_dels + (del_b & lb).count_ones();
            }
        }
        // ---- carry ----
        c_steps += start_b.count_ones();
        c_lits += lit_b.count_ones();
        c_hints += hint_b.count_ones();
        c_dels += del_b.count_ones();
        if cut_b != 0 {
            let c = cut_b.trailing_zeros() & 31;
            *out_end = base + c as u64 + 1;
            out_counts[0] = c_steps;
            out_counts[1] = c_lits;
            out_counts[2] = c_hints;
            out_counts[3] = c_dels;
            return DEC_CLEAN;
        }
        // state after the window: lane 31's inclusive table applied to the
        // entry state — broadcast from lane 31.
        let incl31 = warp.broadcast_lane(data::PerLane::new(incl), 31).get();
        state = dec_apply(incl31, state);
        tok_start = base + (31 - term_b.leading_zeros()) as u64 + 1;
        c_rat = if start_b != 0 {
            let s = 31 - start_b.leading_zeros();
            (rat_b & !dec_lanes_below(s) & !(1u32 << s)) != 0
        } else {
            c_rat || rat_b != 0
        };
        base += 32;
    }
}

/// Warp-per-slice decode. Grid: (num_blocks, 1, 1), Block: (32, 1, 1);
/// warps stride over the slices. Per-slice arrays have `n_slices` entries
/// (`s_cnt` has 4 per slice: steps, lits, hints, dels). In pass 1 the kernel
/// writes `s_start`, `s_end`, `s_flags`, `s_cnt`; in pass 2 it reads `s_cnt`
/// as the host-computed exclusive offsets and writes the record arrays.
#[allow(clippy::too_many_arguments)]
#[warp_kernel]
pub fn lrat_scan_slices(
    bytes: *const u8,
    len: u64,
    slice: u32,
    n_slices: u32,
    num_blocks: u32,
    pass: u32,
    s_start: *mut u64,
    s_end: *mut u64,
    s_flags: *mut u32,
    s_cnt: *mut u32,
    st_pos: *mut u64,
    st_id: *mut u64,
    st_after: *mut u64,
    st_lit_end: *mut u32,
    st_hint_end: *mut u32,
    st_del_end: *mut u32,
    st_tag: *mut u8,
    st_rat: *mut u8,
    o_lits: *mut i32,
    o_hints: *mut u64,
    o_del_id: *mut u64,
    o_del_pos: *mut u64,
) {
    let warp: Warp<All> = Warp::kernel_entry();
    let bid = warp_types::gpu::block_id_x();
    let lane = warp_types::gpu::thread_id_x();
    let mut si = bid;
    while si < n_slices {
        let lo = si as u64 * slice as u64;
        let hi = lo + slice as u64;
        if pass == 2 && *s_flags.add(si as usize) != DEC_CLEAN {
            si += num_blocks;
            continue;
        }
        let (step_base, lit_base, hint_base, del_base) = if pass == 2 {
            (
                *s_cnt.add(4 * si as usize),
                *s_cnt.add(4 * si as usize + 1),
                *s_cnt.add(4 * si as usize + 2),
                *s_cnt.add(4 * si as usize + 3),
            )
        } else {
            (0, 0, 0, 0)
        };
        let mut start = 0u64;
        let mut end = 0u64;
        let mut counts = [0u32; 4];
        let flag = dec_slice_warp(
            &warp, lane, bytes, len, lo, hi, pass, step_base, lit_base, hint_base, del_base,
            st_pos, st_id, st_after, st_lit_end, st_hint_end, st_del_end, st_tag, st_rat,
            o_lits, o_hints, o_del_id, o_del_pos, &mut start, &mut end, &mut counts,
        );
        if pass == 1 && lane == 0 {
            *s_start.add(si as usize) = start;
            *s_end.add(si as usize) = end;
            *s_flags.add(si as usize) = flag;
            let c = if flag == DEC_CLEAN { counts } else { [0u32; 4] };
            *s_cnt.add(4 * si as usize) = c[0];
            *s_cnt.add(4 * si as usize + 1) = c[1];
            *s_cnt.add(4 * si as usize + 2) = c[2];
            *s_cnt.add(4 * si as usize + 3) = c[3];
        }
        si += num_blocks;
    }
}

/// One varint for the serial kernel: value and the offset after it, or
/// `None`-equivalent (ok = false) on truncation / more than 10 bytes.
#[inline(always)]
unsafe fn dec_varint(bytes: *const u8, len: u64, pos: &mut u64) -> (u64, bool) {
    let mut x = 0u64;
    let mut shift = 0u32;
    loop {
        if *pos >= len {
            return (0, false);
        }
        let b = *bytes.add(*pos as usize);
        *pos += 1;
        x |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return (x, true);
        }
        shift += 7;
        if shift > 63 {
            return (0, false);
        }
    }
}

/// Thread-per-slice decode (the control). Grid: (ceil(n_slices / 32), 1, 1),
/// Block: (32, 1, 1); thread `t` takes slice `t`. Same arrays and passes as
/// `lrat_scan_slices`.
#[allow(clippy::too_many_arguments)]
#[warp_kernel]
pub fn lrat_scan_slices_serial(
    bytes: *const u8,
    len: u64,
    slice: u32,
    n_slices: u32,
    _num_blocks: u32,
    pass: u32,
    s_start: *mut u64,
    s_end: *mut u64,
    s_flags: *mut u32,
    s_cnt: *mut u32,
    st_pos: *mut u64,
    st_id: *mut u64,
    st_after: *mut u64,
    st_lit_end: *mut u32,
    st_hint_end: *mut u32,
    st_del_end: *mut u32,
    st_tag: *mut u8,
    st_rat: *mut u8,
    o_lits: *mut i32,
    o_hints: *mut u64,
    o_del_id: *mut u64,
    o_del_pos: *mut u64,
) {
    let si = warp_types::gpu::block_id_x() * 32 + warp_types::gpu::thread_id_x();
    if si >= n_slices {
        return;
    }
    if pass == 2 && *s_flags.add(si as usize) != DEC_CLEAN {
        return;
    }
    let lo = si as u64 * slice as u64;
    let hi = {
        let h = lo + slice as u64;
        if h < len {
            h
        } else {
            len
        }
    };
    let (step_base, lit_base, hint_base, del_base) = if pass == 2 {
        (
            *s_cnt.add(4 * si as usize),
            *s_cnt.add(4 * si as usize + 1),
            *s_cnt.add(4 * si as usize + 2),
            *s_cnt.add(4 * si as usize + 3),
        )
    } else {
        (0, 0, 0, 0)
    };
    // first candidate
    let mut start = lo;
    if lo != 0 {
        let mut p = lo;
        let mut found = false;
        while p < hi {
            if *bytes.add(p as usize - 1) == 0
                && (*bytes.add(p as usize) == b'a' || *bytes.add(p as usize) == b'd')
            {
                found = true;
                break;
            }
            p += 1;
        }
        if !found {
            if pass == 1 {
                *s_start.add(si as usize) = lo;
                *s_end.add(si as usize) = lo;
                *s_flags.add(si as usize) = DEC_NONE;
                let mut k = 0;
                while k < 4 {
                    *s_cnt.add(4 * si as usize + k) = 0;
                    k += 1;
                }
            }
            return;
        }
        start = p;
    }
    let mut pos = start;
    let mut c_steps = 0u32;
    let mut c_lits = 0u32;
    let mut c_hints = 0u32;
    let mut c_dels = 0u32;
    let mut dirty = false;
    while pos < hi && pos < len {
        let tag = *bytes.add(pos as usize);
        let step_pos = pos;
        pos += 1;
        let k = (step_base + c_steps) as usize;
        if tag == b'a' {
            let (idv, ok) = dec_varint(bytes, len, &mut pos);
            if !ok {
                dirty = true;
                break;
            }
            if pass == 2 {
                *st_pos.add(k) = step_pos;
                *st_tag.add(k) = tag;
                *st_id.add(k) = idv >> 1;
                *st_after.add(k) = pos;
            }
            loop {
                let (m, ok) = dec_varint(bytes, len, &mut pos);
                if !ok {
                    dirty = true;
                    break;
                }
                if m == 0 {
                    break;
                }
                let mag = (m >> 1) as i64;
                let signed = if m & 1 == 1 { -mag } else { mag };
                if signed > i32::MAX as i64 || signed < i32::MIN as i64 {
                    dirty = true;
                    break;
                }
                let l = signed as i32;
                if l == 0 || l.unsigned_abs() > DEC_MAX_VAR {
                    dirty = true;
                    break;
                }
                if pass == 2 {
                    *o_lits.add((lit_base + c_lits) as usize) = l;
                }
                c_lits += 1;
            }
            if dirty {
                break;
            }
            let mut rat = false;
            loop {
                let (m, ok) = dec_varint(bytes, len, &mut pos);
                if !ok {
                    dirty = true;
                    break;
                }
                if m == 0 {
                    break;
                }
                rat |= m & 1 == 1;
                if pass == 2 {
                    *o_hints.add((hint_base + c_hints) as usize) = m >> 1;
                }
                c_hints += 1;
            }
            if dirty {
                break;
            }
            if pass == 2 {
                *st_rat.add(k) = rat as u8;
                *st_lit_end.add(k) = c_lits;
                *st_hint_end.add(k) = c_hints;
                *st_del_end.add(k) = c_dels;
            }
            c_steps += 1;
        } else if tag == b'd' {
            if pass == 2 {
                *st_pos.add(k) = step_pos;
                *st_tag.add(k) = tag;
                *st_id.add(k) = 0;
                *st_after.add(k) = 0;
            }
            loop {
                let (m, ok) = dec_varint(bytes, len, &mut pos);
                if !ok {
                    dirty = true;
                    break;
                }
                if m == 0 {
                    break;
                }
                if pass == 2 {
                    *o_del_id.add((del_base + c_dels) as usize) = m >> 1;
                    *o_del_pos.add((del_base + c_dels) as usize) = pos;
                }
                c_dels += 1;
            }
            if dirty {
                break;
            }
            if pass == 2 {
                *st_rat.add(k) = 0;
                *st_lit_end.add(k) = c_lits;
                *st_hint_end.add(k) = c_hints;
                *st_del_end.add(k) = c_dels;
            }
            c_steps += 1;
        } else {
            dirty = true;
            break;
        }
    }
    if pass == 1 {
        *s_start.add(si as usize) = start;
        *s_end.add(si as usize) = pos;
        *s_flags.add(si as usize) = if dirty { DEC_DIRTY } else { DEC_CLEAN };
        let c = if dirty { [0u32; 4] } else { [c_steps, c_lits, c_hints, c_dels] };
        let mut k = 0;
        while k < 4 {
            *s_cnt.add(4 * si as usize + k) = c[k];
            k += 1;
        }
    }
}

// ============================================================================
// Phase 3b on the device: place the true chain's payloads (stage 3 of
// research/gpu-lrat-decode.md). The host's stitch produces the segment list
// (`copies`); each output element finds its segment by binary search over the
// destination prefix and reads its source — the device's raw decode arrays
// (`sel = 0`) or the small "extra" buffer the host uploaded for bridges and
// host-parsed slices (`sel = 1`). Hints are mapped id → dense index through
// the dense table at the same time; the first hint naming an id never added
// is recorded by an atomic minimum, as the host's phase 3b does.
// ============================================================================

/// Segment of output index `i`: the last `k` with `seg_dst[k] <= i`
/// (`seg_dst[0] == 0`, `n_seg >= 1`).
#[inline(always)]
unsafe fn dec_segment_of(seg_dst: *const u32, n_seg: u32, i: u32) -> u32 {
    let mut lo = 0u32;
    let mut hi = n_seg;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if *seg_dst.add(mid as usize) <= i {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `out[i] = source(segment(i))[seg_src + (i - seg_dst)]` for literals.
/// Grid: (num_blocks, 1, 1), Block: (32, 1, 1), grid-stride over `n_out`.
#[allow(clippy::too_many_arguments)]
#[warp_kernel]
pub fn lrat_gather_lits(
    seg_dst: *const u32,
    seg_src: *const u32,
    seg_sel: *const u32,
    n_seg: u32,
    src0: *const i32,
    src1: *const i32,
    out: *mut i32,
    n_out: u32,
    num_blocks: u32,
) {
    let mut i = warp_types::gpu::block_id_x() * 32 + warp_types::gpu::thread_id_x();
    let stride = num_blocks * 32;
    while i < n_out {
        let k = dec_segment_of(seg_dst, n_seg, i) as usize;
        let j = (*seg_src.add(k) + (i - *seg_dst.add(k))) as usize;
        *out.add(i as usize) = if *seg_sel.add(k) == 0 {
            *src0.add(j)
        } else {
            *src1.add(j)
        };
        i += stride;
    }
}

/// Same gather for hints, mapping each raw id through `dense_of` (0 = never
/// added): `out[i] = dense - 1`, or `u32::MAX` with `bad = min(bad, i)`.
#[allow(clippy::too_many_arguments)]
#[warp_kernel]
pub fn lrat_gather_hints(
    seg_dst: *const u32,
    seg_src: *const u32,
    seg_sel: *const u32,
    n_seg: u32,
    src0: *const u64,
    src1: *const u64,
    dense_of: *const u32,
    dense_len: u32,
    out: *mut u32,
    n_out: u32,
    bad: *mut u32,
    num_blocks: u32,
) {
    let mut i = warp_types::gpu::block_id_x() * 32 + warp_types::gpu::thread_id_x();
    let stride = num_blocks * 32;
    while i < n_out {
        let k = dec_segment_of(seg_dst, n_seg, i) as usize;
        let j = (*seg_src.add(k) + (i - *seg_dst.add(k))) as usize;
        let h = if *seg_sel.add(k) == 0 {
            *src0.add(j)
        } else {
            *src1.add(j)
        };
        let d = if h < dense_len as u64 {
            *dense_of.add(h as usize)
        } else {
            0
        };
        if d == 0 {
            let a = core::sync::atomic::AtomicU32::from_ptr(bad);
            a.fetch_min(i, core::sync::atomic::Ordering::Relaxed);
            *out.add(i as usize) = u32::MAX;
        } else {
            *out.add(i as usize) = d - 1;
        }
        i += stride;
    }
}
