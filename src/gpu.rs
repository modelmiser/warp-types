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

/// Butterfly shuffle confined to a segment of `width` lanes.
/// PTX: `shfl.sync.bfly.b32` with `c = ((32 - width) << 8) | 0x1F`
///
/// Used by `Tile<SIZE>` to confine shuffles within tile boundaries.
/// `width` must be a power of 2 in {4, 8, 16, 32}.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn shfl_sync_bfly_i32_width(mask: u32, val: i32, lane_mask: u32, width: u32) -> i32 {
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
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn shfl_sync_down_i32_width(mask: u32, val: i32, delta: u32, width: u32) -> i32 {
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
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn shfl_sync_up_i32_width(mask: u32, val: i32, delta: u32, width: u32) -> i32 {
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
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn ballot_sync(mask: u32, predicate: bool) -> u32 {
    let result: u32;
    let pred_u32 = predicate as u32;
    unsafe {
        // vote.sync.ballot.b32 requires a predicate register (%p), not a
        // general-purpose register (%r).  Convert via setp first.
        core::arch::asm!(
            "setp.ne.u32 {pred_p}, {pred_in}, 0;",
            "vote.sync.ballot.b32 {result}, {pred_p}, {mask};",
            pred_in = in(reg32) pred_u32,
            pred_p = out(pred) _,
            result = out(reg32) result,
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
    label = "GpuShuffle is implemented for i32, u32, f32, i64, u64, f64, bool — use one of these types",
    note = "larger types require two shuffles; implement GpuShuffle manually for custom types"
)]
pub trait GpuShuffle: crate::active_set::sealed::Sealed + Copy + 'static {
    /// Butterfly shuffle: exchange with lane (lane_id XOR mask).
    fn gpu_shfl_xor(self, xor_mask: u32) -> Self;

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
    #[inline(always)]
    fn gpu_shfl_xor_width(self, xor_mask: u32, width: u32) -> Self {
        shfl_sync_bfly_i32_width(0xFFFFFFFF, self, xor_mask, width)
    }
    #[inline(always)]
    fn gpu_shfl_down_width(self, delta: u32, width: u32) -> Self {
        shfl_sync_down_i32_width(0xFFFFFFFF, self, delta, width)
    }
    #[inline(always)]
    fn gpu_shfl_up_width(self, delta: u32, width: u32) -> Self {
        shfl_sync_up_i32_width(0xFFFFFFFF, self, delta, width)
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
        let lo = shfl_sync_bfly_i32(0xFFFFFFFF, bits as i32, xor_mask) as u32;
        let hi = shfl_sync_bfly_i32(0xFFFFFFFF, (bits >> 32) as i32, xor_mask) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        let bits = self as u64;
        let lo = shfl_sync_down_i32(0xFFFFFFFF, bits as i32, delta) as u32;
        let hi = shfl_sync_down_i32(0xFFFFFFFF, (bits >> 32) as i32, delta) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        let bits = self as u64;
        let lo = shfl_sync_up_i32(0xFFFFFFFF, bits as i32, delta) as u32;
        let hi = shfl_sync_up_i32(0xFFFFFFFF, (bits >> 32) as i32, delta) as u32;
        ((hi as u64) << 32 | lo as u64) as i64
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        let bits = self as u64;
        let lo = shfl_sync_idx_i32(0xFFFFFFFF, bits as i32, src_lane) as u32;
        let hi = shfl_sync_idx_i32(0xFFFFFFFF, (bits >> 32) as i32, src_lane) as u32;
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
        shfl_sync_bfly_i32(0xFFFFFFFF, self as i32, xor_mask) != 0
    }
    #[inline(always)]
    fn gpu_shfl_down(self, delta: u32) -> Self {
        shfl_sync_down_i32(0xFFFFFFFF, self as i32, delta) != 0
    }
    #[inline(always)]
    fn gpu_shfl_up(self, delta: u32) -> Self {
        shfl_sync_up_i32(0xFFFFFFFF, self as i32, delta) != 0
    }
    #[inline(always)]
    fn gpu_shfl_idx(self, src_lane: u32) -> Self {
        shfl_sync_idx_i32(0xFFFFFFFF, self as i32, src_lane) != 0
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
