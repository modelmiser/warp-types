//! GPU intrinsics for nvptx64 targets.
//!
//! Provides actual PTX instructions for shuffle, ballot, and sync operations.
//! Gated behind `#[cfg(target_arch = "nvptx64")]` — on other targets, these
//! are not available and the CPU emulation in `shuffle.rs` is used instead.
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

/// Get the current thread's X index within the block.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn thread_id_x() -> u32 {
    let id: u32;
    unsafe { core::arch::asm!("mov.u32 {}, %tid.x;", out(reg32) id) };
    id
}

/// Butterfly shuffle: exchange with lane (lane_id XOR lane_mask).
/// PTX: `shfl.sync.bfly.b32`
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn shfl_sync_bfly_i32(mask: u32, val: i32, lane_mask: u32) -> i32 {
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
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn shfl_sync_down_i32(mask: u32, val: i32, delta: u32) -> i32 {
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
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn shfl_sync_up_i32(mask: u32, val: i32, delta: u32) -> i32 {
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
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn shfl_sync_idx_i32(mask: u32, val: i32, src_lane: u32) -> i32 {
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

/// Ballot: each thread votes, returns bitmask of votes.
/// PTX: `vote.sync.ballot.b32`
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn ballot_sync(mask: u32, predicate: bool) -> u32 {
    let result: u32;
    let pred_u32 = predicate as u32;
    unsafe {
        core::arch::asm!(
            "vote.sync.ballot.b32 {result}, {pred}, {mask};",
            result = out(reg32) result,
            pred = in(reg32) pred_u32,
            mask = in(reg32) mask,
        );
    }
    result
}

/// Warp barrier synchronization.
/// PTX: `bar.warp.sync`
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn syncwarp(mask: u32) {
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
    label = "GpuShuffle is implemented for i32, u32, f32 — use one of these types",
    note = "larger types (i64, f64) require two shuffles; implement GpuShuffle manually for custom types"
)]
pub trait GpuShuffle: Copy + 'static {
    /// Butterfly shuffle: exchange with lane (lane_id XOR mask).
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self;

    /// Shuffle down: read from lane (lane_id + delta).
    fn gpu_shfl_down(self, delta: u32) -> Self;

    /// Shuffle up: read from lane (lane_id - delta).
    fn gpu_shfl_up(self, delta: u32) -> Self;

    /// Indexed shuffle: read from specific lane.
    fn gpu_shfl_idx(self, src_lane: u32) -> Self;
}

#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for i32 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        shfl_sync_bfly_i32(0xFFFFFFFF, self, xor_mask)
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        shfl_sync_down_i32(0xFFFFFFFF, self, delta)
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        shfl_sync_up_i32(0xFFFFFFFF, self, delta)
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        shfl_sync_idx_i32(0xFFFFFFFF, self, src_lane)
    }
}

// f32 shares the same b32 instruction (reinterpret bits)
#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for f32 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        f32::from_bits(shfl_sync_bfly_i32(0xFFFFFFFF, self.to_bits() as i32, xor_mask) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        f32::from_bits(shfl_sync_down_i32(0xFFFFFFFF, self.to_bits() as i32, delta) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        f32::from_bits(shfl_sync_up_i32(0xFFFFFFFF, self.to_bits() as i32, delta) as u32)
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        f32::from_bits(shfl_sync_idx_i32(0xFFFFFFFF, self.to_bits() as i32, src_lane) as u32)
    }
}

#[cfg(target_arch = "nvptx64")]
impl GpuShuffle for u32 {
    #[inline(always)]
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self {
        shfl_sync_bfly_i32(0xFFFFFFFF, self as i32, xor_mask) as u32
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        shfl_sync_down_i32(0xFFFFFFFF, self as i32, delta) as u32
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        shfl_sync_up_i32(0xFFFFFFFF, self as i32, delta) as u32
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        shfl_sync_idx_i32(0xFFFFFFFF, self as i32, src_lane) as u32
    }
}

// CPU fallback: single-thread, shuffle returns own value (identity)
#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
impl GpuShuffle for i32 {
    fn gpu_shfl_xor(self, _: u32) -> Self { self }
    fn gpu_shfl_down(self, _: u32) -> Self { self }
    fn gpu_shfl_up(self, _: u32) -> Self { self }
    fn gpu_shfl_idx(self, _: u32) -> Self { self }
}

#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
impl GpuShuffle for f32 {
    fn gpu_shfl_xor(self, _: u32) -> Self { self }
    fn gpu_shfl_down(self, _: u32) -> Self { self }
    fn gpu_shfl_up(self, _: u32) -> Self { self }
    fn gpu_shfl_idx(self, _: u32) -> Self { self }
}

#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
impl GpuShuffle for u32 {
    fn gpu_shfl_xor(self, _: u32) -> Self { self }
    fn gpu_shfl_down(self, _: u32) -> Self { self }
    fn gpu_shfl_up(self, _: u32) -> Self { self }
    fn gpu_shfl_idx(self, _: u32) -> Self { self }
}
