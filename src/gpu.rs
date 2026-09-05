//! GPU intrinsics for nvptx64 and amdgpu targets.
//!
//! Provides actual PTX/GCN instructions for shuffle, ballot, and sync operations.
//! Gated behind `#[cfg(target_arch = "nvptx64")]` or `#[cfg(target_arch = "amdgpu")]`.
//!
//! # Platform Dispatch (Crystal Facet: PlatformDispatch)
//!
//! Three compilation targets with different shuffle semantics:
//!
//! | Target | Shuffle behavior | Mask width | Status |
//! |--------|-----------------|------------|--------|
//! | nvptx64 | Real `shfl.sync.*` instructions | 32-bit | Implemented |
//! | amdgpu | DPP row_xmask / ds_bpermute | 64-bit | Stubbed |
//! | CPU | Identity (returns own value) | N/A | Emulation |
//!
//! **CPU emulation caveat:** Shuffle-XOR returns `self` on CPU, which makes
//! `reduce_sum` accidentally correct (1+1+1...=32 via butterfly doubling)
//! but makes `inclusive_sum` incorrect (produces reduce result, not prefix).
//! Tests that rely on scan semantics must be gated behind `#[cfg(target_arch)]`
//! or use a multi-lane CPU emulator.
//!
//! Requires nightly Rust with `#![feature(asm_experimental_arch)]`.

/// Get the current thread's lane ID within the warp (0..31).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn lane_id() -> u32 {
    let id: u32;
    unsafe { core::arch::asm!("mov.u32 {}, %laneid;", out(reg32) id) };
    id
}

/// CPU fallback: returns 0 (single-thread emulation).
///
/// This is correct for CPU testing where `shuffle_xor` is identity:
/// since `my == partner` always, direction-aware compare-and-swap
/// produces the same result regardless of `lane_id`.
#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
#[inline(always)]
pub fn lane_id() -> u32 {
    0
}

/// Get the current thread's X index within the block.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn thread_id_x() -> u32 {
    let id: u32;
    unsafe { core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) id) };
    id
}

/// Get the current block's X index within the grid.
/// PTX: `%ctaid.x` (equivalent to CUDA's `blockIdx.x`).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn block_id_x() -> u32 {
    let id: u32;
    unsafe { core::arch::asm!("mov.u32 {}, %ctaid.x;", out(reg32) id) };
    id
}

/// Get the block dimension (number of threads) in X.
/// PTX: `%ntid.x` (equivalent to CUDA's `blockDim.x`).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn block_dim_x() -> u32 {
    let dim: u32;
    unsafe { core::arch::asm!("mov.u32 {}, %ntid.x;", out(reg32) dim) };
    dim
}

/// Atomic add for f64 in global memory.
/// PTX: `atom.global.add.f64` (native since sm_60).
///
/// # Safety
///
/// `addr` must name a location in **global** memory, be naturally 8-byte
/// aligned, and be valid for an 8-byte read-modify-write for the duration of
/// the call — i.e. inside a live device allocation, not null, dangling, freed,
/// or a host pointer.
///
/// A *generic* pointer is fine, and is the normal case: a device `*mut f64`
/// from `cudaMalloc` is generic and refers to global memory, which PTX permits
/// (`atom` accepts generic addressing "where the address points to `.global`
/// or `.shared` space"). What is undefined is a location in the wrong state
/// space — `.local`, `.shared`, `.const`, or `.param` — reached through such a
/// pointer, since this instruction is `.global`-qualified.
///
/// Concurrent *non-atomic* loads or stores to `addr` are a data race, not
/// resolved contention. Concurrent atomics are defined, but only when their
/// scopes are mutually inclusive: this form carries the default `.gpu` scope,
/// so a `.cta`-scoped or cross-device atomic on the same location is **not**
/// atomic with respect to it. This function is not a private coherence domain —
/// CUDA's `atomicAdd` on `double` over the same address is well defined
/// alongside it.
///
/// Requires sm_60 or later, where the f64 form is native.
///
/// Callers establish the bound, not this function. The in-tree caller
/// (`sat-kernels`' gradient accumulation, `warp-types-sat/sat-kernels/src/lib.rs`)
/// indexes `grad` by clause variable id, and the host side bounds every such id
/// against `ClauseDataSoA::num_vars` before launch (as `len >= max + 1`, not by
/// scanning each id) — see the asserts in
/// `warp-types-sat/src/gpu_launcher.rs`. An unchecked variable id here writes
/// outside the gradient allocation with no diagnostic.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn atomic_add_f64(addr: *mut f64, val: f64) -> f64 {
    let result: f64;
    core::arch::asm!(
        "atom.global.add.f64 {result}, [{addr}], {val};",
        result = out(reg64) result,
        addr = in(reg64) addr,
        val = in(reg64) val,
    );
    result
}

/// Butterfly shuffle: exchange with lane (lane_id XOR lane_mask).
/// PTX: `shfl.sync.bfly.b32`
///
/// # Safety
///
/// `mask` (the PTX membermask) must name the executing lane, and every lane
/// named in `mask` must be converged and executing this same instruction —
/// naming a lane that does not execute it is undefined behavior (PTX
/// `shfl.sync`). Data read from a source lane not named in `mask` is
/// undefined. The typed wrappers (`Warp<All>`, `Tile`, `shuffle_xor_within`)
/// establish this contract; call this directly only with a proven mask.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn shfl_sync_bfly_i32(mask: u32, val: i32, lane_mask: u32) -> i32 {
    let result: i32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.bfly.b32 {result}, {val}, {lane_mask}, 31, {mask};",
            result = out(reg32) result,
            val = in(reg32) val,
            lane_mask = in(reg32) lane_mask,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Shuffle down: lane[i] reads from lane[i + delta].
/// PTX: `shfl.sync.down.b32`
///
/// # Safety
///
/// Same membermask contract as [`shfl_sync_bfly_i32`]: `mask` must name the
/// executing lane and only lanes converged on this instruction.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn shfl_sync_down_i32(mask: u32, val: i32, delta: u32) -> i32 {
    let result: i32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.down.b32 {result}, {val}, {delta}, 31, {mask};",
            result = out(reg32) result,
            val = in(reg32) val,
            delta = in(reg32) delta,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Shuffle up: lane[i] reads from lane[i - delta].
/// PTX: `shfl.sync.up.b32`
///
/// # Safety
///
/// Same membermask contract as [`shfl_sync_bfly_i32`]: `mask` must name the
/// executing lane and only lanes converged on this instruction.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn shfl_sync_up_i32(mask: u32, val: i32, delta: u32) -> i32 {
    let result: i32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.up.b32 {result}, {val}, {delta}, 0, {mask};",
            result = out(reg32) result,
            val = in(reg32) val,
            delta = in(reg32) delta,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Indexed shuffle: lane[i] reads from lane[src_lane].
/// PTX: `shfl.sync.idx.b32`
///
/// # Safety
///
/// Same membermask contract as [`shfl_sync_bfly_i32`]: `mask` must name the
/// executing lane and only lanes converged on this instruction.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn shfl_sync_idx_i32(mask: u32, val: i32, src_lane: u32) -> i32 {
    let result: i32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.idx.b32 {result}, {val}, {src_lane}, 31, {mask};",
            result = out(reg32) result,
            val = in(reg32) val,
            src_lane = in(reg32) src_lane,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Butterfly shuffle confined to a segment of `width` lanes.
/// PTX: `shfl.sync.bfly.b32` with `c = ((32 - width) << 8) | 0x1F`
///
/// Used by `Tile<SIZE>` to confine shuffles within tile boundaries.
/// `width` must be a power of 2 in {4, 8, 16, 32}.
///
/// # Safety
///
/// Same membermask contract as [`shfl_sync_bfly_i32`]: `mask` must name the
/// executing lane and only lanes converged on this instruction.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn shfl_sync_bfly_i32_width(mask: u32, val: i32, lane_mask: u32, width: u32) -> i32 {
    assert!(
        width.is_power_of_two() && (4..=32).contains(&width),
        "shfl_sync_bfly width {width} must be a power of two in 4..=32"
    );
    let c = ((32 - width) << 8) | 0x1F;
    let result: i32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.bfly.b32 {result}, {val}, {lane_mask}, {c}, {mask};",
            result = out(reg32) result,
            val = in(reg32) val,
            lane_mask = in(reg32) lane_mask,
            c = in(reg32) c,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Shuffle down confined to a segment of `width` lanes.
/// PTX: `shfl.sync.down.b32` with `c = ((32 - width) << 8) | (width - 1)`
///
/// # Safety
///
/// Same membermask contract as [`shfl_sync_bfly_i32`]: `mask` must name the
/// executing lane and only lanes converged on this instruction.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn shfl_sync_down_i32_width(mask: u32, val: i32, delta: u32, width: u32) -> i32 {
    assert!(
        width.is_power_of_two() && (4..=32).contains(&width),
        "shfl_sync_down width {width} must be a power of two in 4..=32"
    );
    let c = ((32 - width) << 8) | (width - 1);
    let result: i32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.down.b32 {result}, {val}, {delta}, {c}, {mask};",
            result = out(reg32) result,
            val = in(reg32) val,
            delta = in(reg32) delta,
            c = in(reg32) c,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Shuffle up confined to a segment of `width` lanes.
/// PTX: `shfl.sync.up.b32` with `c = ((32 - width) << 8)`
///
/// # Safety
///
/// Same membermask contract as [`shfl_sync_bfly_i32`]: `mask` must name the
/// executing lane and only lanes converged on this instruction.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn shfl_sync_up_i32_width(mask: u32, val: i32, delta: u32, width: u32) -> i32 {
    assert!(
        width.is_power_of_two() && (4..=32).contains(&width),
        "shfl_sync_up width {width} must be a power of two in 4..=32"
    );
    let c = (32 - width) << 8;
    let result: i32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.up.b32 {result}, {val}, {delta}, {c}, {mask};",
            result = out(reg32) result,
            val = in(reg32) val,
            delta = in(reg32) delta,
            c = in(reg32) c,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Ballot: each thread votes, returns bitmask of votes.
/// PTX: `vote.sync.ballot.b32`
///
/// Uses the setp/selp workaround for Rust's missing `pred` register class:
/// declare `.reg .pred` inside the asm block, convert to/from u32 via setp,
/// pass everything through reg32. Same pattern as Rust-CUDA's `cuda_std`.
///
/// # Safety
///
/// `mask` (the PTX membermask) must name the executing lane, and every lane
/// named in `mask` must be converged and executing this same instruction —
/// otherwise behavior is undefined (PTX `vote.sync`). `Warp<All>::ballot`
/// establishes this by requiring a full-warp witness before passing
/// `0xFFFFFFFF`.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn ballot_sync(mask: u32, predicate: bool) -> u32 {
    let result: u32;
    let pred_u32 = predicate as u32;
    unsafe {
        core::arch::asm!(
            "{{",
            ".reg .pred %p_vote;",
            "setp.ne.u32 %p_vote, {pred_in}, 0;",
            "vote.sync.ballot.b32 {result}, %p_vote, {mask};",
            "}}",
            pred_in = in(reg32) pred_u32,
            result = out(reg32) result,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Warp barrier synchronization.
/// PTX: `bar.warp.sync`
///
/// # Safety
///
/// `mask` must name the executing lane, and every lane named in `mask` must
/// eventually execute a `bar.warp.sync` with the same mask — otherwise the
/// barrier deadlocks or has undefined behavior (PTX `bar.warp.sync`).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub unsafe fn syncwarp(mask: u32) {
    unsafe {
        core::arch::asm!(
            "bar.warp.sync {mask};",
            mask = in(reg32) mask,
        );
    }
}

/// Thread fence (global memory ordering).
/// PTX: `membar.gl`
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn threadfence() {
    unsafe {
        core::arch::asm!("membar.gl;");
    }
}

// ============================================================================
// AMD GCN intrinsics (amdgcn target)
// ============================================================================

// AMD GPUs use DPP (Data-Parallel Primitives) for intra-wavefront communication.
// Key instructions:
//   - ds_permute_b32 / ds_bpermute_b32: arbitrary lane permutation via LDS
//   - v_mov_b32 with DPP modifiers: for regular patterns (row_shl, row_xmask, etc.)
//   - v_readlane_b32 / v_writelane_b32: scalar ↔ vector lane access
//
// AMD wavefronts are 64 lanes (CDNA) or 32/64 (RDNA wave32/wave64 mode).
// The exec mask is 64-bit (s[exec_lo:exec_hi]).
//
// These stubs will be filled when amdgcn target support is available in Rust.

/// AMD DPP row XOR: each lane exchanges with lane (lane_id XOR mask).
/// Equivalent to NVIDIA's shfl.sync.bfly — butterfly pattern.
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
pub fn dpp_row_xor_i32(val: i32, xor_mask: u32) -> i32 {
    // TODO: implement via inline asm when amdgcn asm support is stable
    // v_mov_b32 with DPP modifier row_xmask:<mask>
    let _ = xor_mask;
    val // placeholder
}

/// AMD ds_bpermute: lane[i] reads from lane[src_lane].
/// Equivalent to NVIDIA's shfl.sync.idx.
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
pub fn ds_bpermute_i32(val: i32, src_lane_x4: u32) -> i32 {
    // ds_bpermute_b32 uses byte offset (lane * 4)
    let _ = src_lane_x4;
    val // placeholder
}

/// AMD exec mask: 64-bit mask of active lanes.
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
pub fn exec_mask() -> u64 {
    0xFFFFFFFFFFFFFFFF // placeholder
}

// ============================================================================
// GpuShuffle trait — type-safe dispatch for shuffle intrinsics
// ============================================================================

/// Trait for types that can be shuffled via GPU intrinsics.
///
/// On nvptx64: maps to actual `shfl.sync.*.b32` instructions.
/// On amdgpu: will map to DPP row_xmask / ds_bpermute (not yet implemented).
/// On other targets: provides CPU emulation (identity for single-thread).
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be shuffled across GPU lanes",
    label = "GpuShuffle is implemented for i32, u32, f32, i64, u64, f64, bool — use one of these types",
    note = "larger types require two shuffles; implement GpuShuffle manually for custom types"
)]
pub trait GpuShuffle: crate::gpu_sealed::GpuSealed + Copy + 'static {
    /// Butterfly shuffle: exchange with lane (lane_id XOR mask).
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self;

    /// Butterfly shuffle with an explicit membermask (converged-lane set).
    ///
    /// On nvptx64 the membermask is passed straight to `shfl.sync.bfly.b32`;
    /// on CPU it is ignored (single-thread identity, same as `gpu_shfl_xor`).
    ///
    /// # Contract (nvptx64)
    ///
    /// `membermask` must name the executing lane and only lanes converged on
    /// this call, and every source lane `laneid ^ xor_mask` read by a named
    /// lane must itself be named. The typed entry point
    /// [`Warp::shuffle_xor_within`](crate::warp::Warp::shuffle_xor_within)
    /// establishes this by asserting the XOR permutation preserves the active
    /// set `S` and passing `S::MASK`. (The trait is sealed — this contract is
    /// discharged inside the crate, not by downstream implementors.)
    fn gpu_shfl_xor_masked(self, xor_mask: u32, membermask: u32) -> Self;

    /// Shuffle down: read from lane (lane_id + delta).
    fn gpu_shfl_down(self, delta: u32) -> Self;

    /// Shuffle up: read from lane (lane_id - delta).
    fn gpu_shfl_up(self, delta: u32) -> Self;

    /// Indexed shuffle: read from specific lane.
    fn gpu_shfl_idx(self, src_lane: u32) -> Self;

    /// Butterfly shuffle confined to a segment of `width` lanes.
    ///
    /// Used by `Tile<SIZE>` to confine shuffles within tile boundaries.
    /// Default delegates to full-warp shuffle (correct for CPU identity).
    fn gpu_shfl_xor_width(self, xor_mask: u32, _width: u32) -> Self {
        self.gpu_shfl_xor(xor_mask)
    }

    /// Shuffle down confined to a segment of `width` lanes.
    fn gpu_shfl_down_width(self, delta: u32, _width: u32) -> Self {
        self.gpu_shfl_down(delta)
    }

    /// Shuffle up confined to a segment of `width` lanes.
    fn gpu_shfl_up_width(self, delta: u32, _width: u32) -> Self {
        self.gpu_shfl_up(delta)
    }
}

// SAFETY (applies to every `unsafe { shfl_sync_* }` call below): the
// full-warp membermask 0xFFFFFFFF is only reachable through the typed
// wrappers (`Warp<All>`, `Tile` — both witness full-warp convergence), so
// every named lane is converged and executing the instruction. The masked
// variant forwards its caller's membermask under the `gpu_shfl_xor_masked`
// contract (discharged by `shuffle_xor_within`'s preservation assert).
#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for i32 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        // SAFETY: full-warp membermask; see impl-level comment.
        unsafe { shfl_sync_bfly_i32(0xFFFFFFFF, self, xor_mask) }
    }
    #[inline(always)]
    fn gpu_shfl_xor_masked(self, xor_mask: u32, membermask: u32) -> Self {
        // SAFETY: caller's membermask under the trait contract; see
        // impl-level comment.
        unsafe { shfl_sync_bfly_i32(membermask, self, xor_mask) }
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        // SAFETY: full-warp membermask; see impl-level comment.
        unsafe { shfl_sync_down_i32(0xFFFFFFFF, self, delta) }
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        // SAFETY: full-warp membermask; see impl-level comment.
        unsafe { shfl_sync_up_i32(0xFFFFFFFF, self, delta) }
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        // SAFETY: full-warp membermask; see impl-level comment.
        unsafe { shfl_sync_idx_i32(0xFFFFFFFF, self, src_lane) }
    }
    #[inline(always)]
    fn gpu_shfl_xor_width(self, xor_mask: u32, width: u32) -> Self {
        // SAFETY: full-warp membermask; see impl-level comment.
        unsafe { shfl_sync_bfly_i32_width(0xFFFFFFFF, self, xor_mask, width) }
    }
    #[inline(always)]
    fn gpu_shfl_down_width(self, delta: u32, width: u32) -> Self {
        // SAFETY: full-warp membermask; see impl-level comment.
        unsafe { shfl_sync_down_i32_width(0xFFFFFFFF, self, delta, width) }
    }
    #[inline(always)]
    fn gpu_shfl_up_width(self, delta: u32, width: u32) -> Self {
        // SAFETY: full-warp membermask; see impl-level comment.
        unsafe { shfl_sync_up_i32_width(0xFFFFFFFF, self, delta, width) }
    }
}

// f32 shares the same b32 instruction (reinterpret bits)
#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for f32 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        f32::from_bits((self.to_bits() as i32).gpu_shfl_xor(xor_mask) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_xor_masked(self, xor_mask: u32, membermask: u32) -> Self {
        f32::from_bits((self.to_bits() as i32).gpu_shfl_xor_masked(xor_mask, membermask) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        f32::from_bits((self.to_bits() as i32).gpu_shfl_down(delta) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        f32::from_bits((self.to_bits() as i32).gpu_shfl_up(delta) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        f32::from_bits((self.to_bits() as i32).gpu_shfl_idx(src_lane) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_xor_width(self, xor_mask: u32, width: u32) -> Self {
        f32::from_bits((self.to_bits() as i32).gpu_shfl_xor_width(xor_mask, width) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_down_width(self, delta: u32, width: u32) -> Self {
        f32::from_bits((self.to_bits() as i32).gpu_shfl_down_width(delta, width) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_up_width(self, delta: u32, width: u32) -> Self {
        f32::from_bits((self.to_bits() as i32).gpu_shfl_up_width(delta, width) as u32)
    }
}

#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for u32 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        (self as i32).gpu_shfl_xor(xor_mask) as u32
    }
    #[inline(always)]
    fn gpu_shfl_xor_masked(self, xor_mask: u32, membermask: u32) -> Self {
        (self as i32).gpu_shfl_xor_masked(xor_mask, membermask) as u32
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        (self as i32).gpu_shfl_down(delta) as u32
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        (self as i32).gpu_shfl_up(delta) as u32
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        (self as i32).gpu_shfl_idx(src_lane) as u32
    }
    #[inline(always)]
    fn gpu_shfl_xor_width(self, xor_mask: u32, width: u32) -> Self {
        (self as i32).gpu_shfl_xor_width(xor_mask, width) as u32
    }
    #[inline(always)]
    fn gpu_shfl_down_width(self, delta: u32, width: u32) -> Self {
        (self as i32).gpu_shfl_down_width(delta, width) as u32
    }
    #[inline(always)]
    fn gpu_shfl_up_width(self, delta: u32, width: u32) -> Self {
        (self as i32).gpu_shfl_up_width(delta, width) as u32
    }
}

// ============================================================================
// 64-bit types: two-pass shuffle (split into high/low 32-bit halves)
//
// GPU shuffle instructions are 32-bit. For i64/f64/u64, we split into
// two 32-bit halves, shuffle each independently, and reassemble.
// The type system ensures both halves are shuffled together — you can't
// accidentally shuffle only the low half and leave the high half stale.
// ============================================================================

#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for i64 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        let bits = self as u64;
        let lo = (bits as i32).gpu_shfl_xor(xor_mask) as u32;
        let hi = ((bits >> 32) as i32).gpu_shfl_xor(xor_mask) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_xor_masked(self, xor_mask: u32, membermask: u32) -> Self {
        let bits = self as u64;
        let lo = (bits as i32).gpu_shfl_xor_masked(xor_mask, membermask) as u32;
        let hi = ((bits >> 32) as i32).gpu_shfl_xor_masked(xor_mask, membermask) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        let bits = self as u64;
        let lo = (bits as i32).gpu_shfl_down(delta) as u32;
        let hi = ((bits >> 32) as i32).gpu_shfl_down(delta) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        let bits = self as u64;
        let lo = (bits as i32).gpu_shfl_up(delta) as u32;
        let hi = ((bits >> 32) as i32).gpu_shfl_up(delta) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        let bits = self as u64;
        let lo = (bits as i32).gpu_shfl_idx(src_lane) as u32;
        let hi = ((bits >> 32) as i32).gpu_shfl_idx(src_lane) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_xor_width(self, xor_mask: u32, width: u32) -> Self {
        let bits = self as u64;
        let lo = (bits as i32).gpu_shfl_xor_width(xor_mask, width) as u32;
        let hi = ((bits >> 32) as i32).gpu_shfl_xor_width(xor_mask, width) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_down_width(self, delta: u32, width: u32) -> Self {
        let bits = self as u64;
        let lo = (bits as i32).gpu_shfl_down_width(delta, width) as u32;
        let hi = ((bits >> 32) as i32).gpu_shfl_down_width(delta, width) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_up_width(self, delta: u32, width: u32) -> Self {
        let bits = self as u64;
        let lo = (bits as i32).gpu_shfl_up_width(delta, width) as u32;
        let hi = ((bits >> 32) as i32).gpu_shfl_up_width(delta, width) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
}

#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for u64 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        (self as i64).gpu_shfl_xor(xor_mask) as u64
    }
    #[inline(always)]
    fn gpu_shfl_xor_masked(self, xor_mask: u32, membermask: u32) -> Self {
        (self as i64).gpu_shfl_xor_masked(xor_mask, membermask) as u64
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        (self as i64).gpu_shfl_down(delta) as u64
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        (self as i64).gpu_shfl_up(delta) as u64
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        (self as i64).gpu_shfl_idx(src_lane) as u64
    }
    #[inline(always)]
    fn gpu_shfl_xor_width(self, xor_mask: u32, width: u32) -> Self {
        (self as i64).gpu_shfl_xor_width(xor_mask, width) as u64
    }
    #[inline(always)]
    fn gpu_shfl_down_width(self, delta: u32, width: u32) -> Self {
        (self as i64).gpu_shfl_down_width(delta, width) as u64
    }
    #[inline(always)]
    fn gpu_shfl_up_width(self, delta: u32, width: u32) -> Self {
        (self as i64).gpu_shfl_up_width(delta, width) as u64
    }
}

#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for f64 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        f64::from_bits((self.to_bits() as i64).gpu_shfl_xor(xor_mask) as u64)
    }
    #[inline(always)]
    fn gpu_shfl_xor_masked(self, xor_mask: u32, membermask: u32) -> Self {
        f64::from_bits((self.to_bits() as i64).gpu_shfl_xor_masked(xor_mask, membermask) as u64)
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        f64::from_bits((self.to_bits() as i64).gpu_shfl_down(delta) as u64)
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        f64::from_bits((self.to_bits() as i64).gpu_shfl_up(delta) as u64)
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        f64::from_bits((self.to_bits() as i64).gpu_shfl_idx(src_lane) as u64)
    }
    #[inline(always)]
    fn gpu_shfl_xor_width(self, xor_mask: u32, width: u32) -> Self {
        f64::from_bits((self.to_bits() as i64).gpu_shfl_xor_width(xor_mask, width) as u64)
    }
    #[inline(always)]
    fn gpu_shfl_down_width(self, delta: u32, width: u32) -> Self {
        f64::from_bits((self.to_bits() as i64).gpu_shfl_down_width(delta, width) as u64)
    }
    #[inline(always)]
    fn gpu_shfl_up_width(self, delta: u32, width: u32) -> Self {
        f64::from_bits((self.to_bits() as i64).gpu_shfl_up_width(delta, width) as u64)
    }
}

// CPU fallback: single-thread, shuffle returns own value (identity).
//
// **Caveat:** This makes reduce_sum accidentally correct (butterfly doubling)
// but makes inclusive_sum/exclusive_sum incorrect (produces reduce result,
// not prefix). Tests that rely on scan semantics must be gated behind
// `#[cfg(target_arch)]` or use the Platform trait's multi-lane CpuSimd emulator.
macro_rules! impl_cpu_gpu_shuffle {
    ($($t:ty),+) => {
        $(
            #[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
            impl GpuShuffle for $t {
                fn gpu_shfl_xor(self, _: u32) -> Self { self }
                // Membermask is ignored on CPU: single-thread identity.
                fn gpu_shfl_xor_masked(self, _: u32, _: u32) -> Self { self }
                fn gpu_shfl_down(self, _: u32) -> Self { self }
                fn gpu_shfl_up(self, _: u32) -> Self { self }
                fn gpu_shfl_idx(self, _: u32) -> Self { self }
            }
        )+
    }
}

impl_cpu_gpu_shuffle!(i32, f32, u32, i64, u64, f64);

// bool: encode as u32 0/1 for GPU shuffle, identity on CPU.
#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for bool {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        (self as i32).gpu_shfl_xor(xor_mask) != 0
    }
    #[inline(always)]
    fn gpu_shfl_xor_masked(self, xor_mask: u32, membermask: u32) -> Self {
        (self as i32).gpu_shfl_xor_masked(xor_mask, membermask) != 0
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        (self as i32).gpu_shfl_down(delta) != 0
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        (self as i32).gpu_shfl_up(delta) != 0
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        (self as i32).gpu_shfl_idx(src_lane) != 0
    }
    #[inline(always)]
    fn gpu_shfl_xor_width(self, xor_mask: u32, width: u32) -> Self {
        (self as i32).gpu_shfl_xor_width(xor_mask, width) != 0
    }
    #[inline(always)]
    fn gpu_shfl_down_width(self, delta: u32, width: u32) -> Self {
        (self as i32).gpu_shfl_down_width(delta, width) != 0
    }
    #[inline(always)]
    fn gpu_shfl_up_width(self, delta: u32, width: u32) -> Self {
        (self as i32).gpu_shfl_up_width(delta, width) != 0
    }
}

impl_cpu_gpu_shuffle!(bool);
