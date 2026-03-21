# Changelog

## [0.3.0] — 2026-03-20

### Added
- `research` feature flag — gates experimental `research/` module (25 modules, 12K lines of design-space prototypes). Always compiled during `cargo test` but excluded from default `cargo doc` and downstream builds
- Prelude: added `LaneId`, `WarpId`, `warp_kernel` to `prelude` module
- Lean 4: all four §5.1 loop typing rules mechanized (LOOP-UNIFORM, LOOP-CONVERGENT, LOOP-VARYING, LOOP-PHASED) with full progress, preservation, and substitution coverage
- Lean 4: nested merge mechanized — `IsComplement s1 s2 parent` (generalized from `IsComplementAll`)
- Lean 4: `letPair` linear pair destructor with full metatheory
- C++20 interop header (`include/warp_types.h`) — concepts, requires clauses, CUDA/HIP/host-only modes
- CMake example project for C++ host + Rust PTX workflow

### Fixed
- **warp64 completion**: `cub::reduce`, `bitonic_sort` (3 variants), and `shuffle::Permutation` algebra all now handle 64-lane wavefronts correctly — previously hardcoded 32-lane constants produced silently wrong results under `--features warp64`
- Permutation masks use `WARP_SIZE - 1` instead of `0x1F`; rotate ops use `WARP_SIZE` instead of `32`
- `FullButterfly` type alias includes `ButterflyStage5 = Xor<32>` under warp64
- `shuffle_by` accepts `[T; 64]` under warp64
- Aliasing UB in research `coalescing.rs` (`WarpPtrMut` Clone, `store` taking `&` instead of `&mut`)
- FFI 64-lane: `warp_types.h` `ComplementOf` concept generalized via `ComplementWithin`
- `GpuWarp32::shuffle` wraps mod 32 (hardware behavior), not clamp
- Stale doc counts, version references, and terminology across README, paper, blog, tutorial

### Changed
- API encapsulation: `Role`, `BlockId`, `ThreadId` fields now private; `DynWarp` no longer derives `Clone`
- `GpuValue` sealed separately from `ActiveSet` (distinct sealing concerns)
- `proof` module gated behind `cfg(any(test, feature = "formal-proof"))` (was always-compiled)
- Paper terminology: remaining "session types" → "linear typestate" in §5.3, §10

## [0.2.0] — 2026-03-18

### Added
- **SimWarp**: Multi-lane warp simulator with real shuffle semantics — butterfly reduce, bitonic sort, tile-confined shuffle, Hillis-Steele scan, 64-lane AMD reduce all verified with actual lane exchange

### Fixed
- `SimWarp::shuffle_idx` OOB behavior: now wraps modularly via `% WIDTH` (matching GPU `shfl.sync.idx` semantics); previously returned lane 0's value
- Bitonic sort `stage_mask` must be constant per stage (`2^k`), not `2 * xor_mask`
- 64-lane AMD path: 3 bugs in wide-warp shuffle/reduce
- Hard-seal `Sealed` trait: `_sealed()` now requires `SealToken` (pub(crate)), making external impls impossible (previously could be bypassed via default method body)
- `ValidTileSize` sealed via `Sealed` supertrait — external impls of invalid tile sizes blocked
- `DynWarp::from_mask` width inference: correctly auto-detects 32/64-lane from mask value
- `PortableVector::extract/insert`: `debug_assert` + direct index (no `% WIDTH` wrapping that silently masked OOB)
- `CpuSimd<0>`: const assert prevents empty-iterator panics in `reduce_sum/max/min`
- `butterfly_reduce_sum`: WIDTH-generic with const power-of-2 assertion (no longer hardcodes `[16,8,4,2,1]`)
- `DynWarp::reduce_sum_scalar`: uses `count_ones()` not hardcoded `* 32`
- `DynWarp::ballot`: rejects 64-lane warps (u32 return incompatible with >32 lanes)
- Paper: corrected LLVM IR example (butterfly is `shl`, not identity)
- 100+ documentation, terminology, and test count corrections across 23 cold review rounds

### Changed
- Terminology: "session types" → "linear typestate" throughout — the mechanism is typestate over a Boolean lattice, not session types proper
- Paper condensed from ~25 to ~19 pages — peripheral sections trimmed, core type system (§3) preserved at full size
- Removed Sol references — standalone warp-types project
- GPU kernels re-verified on RTX 4000 SFF Ada (compute 8.9, 2026-03-17)

### Documentation
- Tutorial updated for crates.io dependency (not path dep)

## [0.1.0] — 2026-03-18

Initial release on crates.io.

- Linear typestate for GPU warp divergence safety
- Zero-overhead implementation via Rust phantom types
- `Warp<S>`, `ComplementOf`, `merge`, `diverge` — core type system
- `DynWarp` gradual typing bridge (runtime → compile-time promotion)
- Platform abstraction (`CpuSimd`, `GpuWarp32`)
- Fence-divergence type-state machine
- Proc macros: `warp_sets!`, `#[warp_kernel]`
- Build-time GPU compilation: `WarpBuilder`
- 369 tests (291 unit + 50 example + 28 doc)
- Lean 4 mechanization: progress, preservation, substitution lemma (zero sorry, zero axioms)
- 5 bug untypability proofs (cuda-samples#398, CCCL#854, PIConGPU#2514, LLVM#155682, shuffle-after-diverge)
