# Cold Review Round 11 — Code Correctness Only

**Date:** 2026-03-17
**Scope:** 42 source files, 5 parallel cold agents + 1 triage verification agent
**Focus:** Code correctness only (no style, docs wording, naming, idioms)

## Method

5 agents divided by functional area, each reading `lib.rs` first (public API + types), then assigned files. Zero shared context between agents. Triage agent cross-referenced all findings against actual source code.

| Agent | Scope | Files | Result |
|-------|-------|-------|--------|
| 1 | Core type system | lib.rs, warp.rs, active_set.rs, diverge.rs, merge.rs, fence.rs | 0 bugs |
| 2 | Operations + hardware | shuffle.rs, data.rs, block.rs, gpu.rs, platform.rs, cub.rs, tile.rs | 2 GENUINE, 1 DOC-ONLY |
| 3 | Extensions + sub-crates | sort.rs, dynamic.rs, gradual.rs, proof.rs, macros, builder, kernel, examples | 1 GENUINE, 1 DOC-ONLY→FP, 1 UNCERTAIN→GENUINE |
| 4 | Research A + reproduce | 10 research files + 5 reproduce files | 0 bugs |
| 5 | Research B + examples | 14 research files + 7 real-world examples | 0 bugs |

## Verified Findings

### GENUINE — sort.rs:50-67 — compare_swap lacks lane-identity direction

`compare_swap` always keeps the smaller value (`if my <= partner`). On GPU, both lanes in a shuffle_xor pair independently evaluate this — both take the minimum, destroying the maximum. The `_stage_mask` parameter is accepted but unused.

**Mitigating factor:** Thoroughly documented in comments (lines 37-48, 88-89). The example kernel at `examples/gpu-project/my-kernels/src/lib.rs:110-193` has the correct direction-aware algorithm.

**Impact:** Data destruction on GPU. CPU tests pass because identity shuffle means `my == partner`.

### GENUINE — tile.rs:111-139 — sub-warp tile shuffles cross tile boundaries

All `Tile<SIZE>` shuffle methods delegate to `GpuShuffle` trait methods, which emit `shfl.sync.*.b32` with `c=31` (full 32-lane segment) and `membermask=0xFFFFFFFF`. For sub-warp tiles (SIZE < 32), the PTX `c` operand should be `(SIZE-1)` to confine the shuffle within tile-sized segments.

**Impact:** On GPU hardware, `Tile<4>`, `Tile<8>`, and `Tile<16>` shuffle/reduce produce wrong results by reading across tile boundaries. `Tile<32>` is unaffected. CPU tests pass because identity shuffle.

**Fix:** Tile-aware shuffle variants that pass `(SIZE-1)` as the clamp value.

### GENUINE — platform.rs:190-197 — CpuSimd ballot overflow for WIDTH > 64

`CpuSimd<WIDTH>::ballot` uses `mask |= 1u64 << i` in a loop over `0..WIDTH`. For WIDTH > 64, shifts by i >= 64 panic in debug or produce 0 in release. No compile-time guard prevents `CpuSimd<128>`.

**Impact:** Any `CpuSimd<N>` with N > 64 silently produces incorrect ballot results in release, panics in debug.

**Fix:** Const assert `WIDTH <= 64`, or wider mask type.

### GENUINE (low) — warp-types-builder:396-406 — fragile TOML name parsing

`find_ptx_file` searches for the first line starting with `name` in Cargo.toml, which could match a `[dependencies]` entry before `[package]`.

**Impact:** Build-time only. Wrong PTX file name lookup, confusing error message. Unlikely in practice (standard Cargo.toml has `[package]` first).

**Fix:** Track TOML section or use a proper TOML parser.

### DOC-ONLY — tile.rs:53-64 — TILE_MASK constants assume lane-0 origin

`TILE_MASK` is defined but never read anywhere. Constants assume tile origin at lane 0 (e.g., `0xFF` for `Tile<8>` covers lanes 0-7 only).

**Impact:** Dead code. No runtime effect.

### FALSE POSITIVE — sort.rs:88 — doc "ascending-only" characterization

Agent 3 claimed docs say "ascending-only" but actual behavior is data destruction. Triage found the docs accurately describe the limitation — "ascending-only" is the design intent, immediately followed by explanation that this makes it incorrect for intermediate bitonic stages.

## Summary

| Severity | Count | Files |
|----------|-------|-------|
| GENUINE | 3 | sort.rs, tile.rs, platform.rs |
| GENUINE (low) | 1 | warp-types-builder |
| DOC-ONLY | 1 | tile.rs (dead code) |
| FALSE POSITIVE | 1 | sort.rs (docs are accurate) |
| **Clean files** | **36/42** | |

## Root Cause Pattern

The two highest-severity bugs (sort.rs, tile.rs) share a root cause: **CPU test paths use identity shuffles**, so both lanes see the same value and all tests pass. The bugs only manifest when two distinct lanes exchange real data on GPU hardware. This is an inherent limitation of testing type-level GPU state encoding on CPU.
