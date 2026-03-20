# warp-types TODO — Cold Review Findings (2026-03-20)

Non-code findings from `--scope all` cold review (13 categories, 5 waves).
Code fixes committed (`e2b0094ef`). Paper/doc/API fixes across 3 commits.

## Paper — Submission-Blocking

### Terminology Drift (MODERATE)
- [x] `paper/README.md:34` — Contribution #2 updated to "linear typestate over active-set lattice"
- [x] `paper/extensions.md:211` — Updated to "linear typestate for warp divergence"
- [x] `paper/README.md:28` — Page count updated to ~19

### Unsupported Claims (MODERATE)
- [x] `paper/evaluation.md:174` — Annotation burden corrected: 16.7% mean (11.3%–25.3%), methodology documented
- [x] `blog/post.md:16` — Hedged: "a key building block" not "the reason"

### Paper-Code Correspondence (MODERATE)
- [x] `shuffle.rs` vs §3.3 BALLOT — Paper updated to match code (Warp<All> only), relaxation documented as future work
- [x] Lean merge scope — Documented in §4.8: Lean only models IsComplementAll, nested merge not yet mechanized
- [x] `proof.rs:207` — Now checks BOTH disjointness AND covering (paper requires both)
- [x] `shuffle.rs:120` vs §3.1 reduce_sum — Documented as simplification in §3.1 (Uniform<T> via butterfly)

### Formalism (MODERATE)
- [x] `Metatheory.lean:43` Step.mergeVal — Documented in §4.8 as merge-scope restriction
- [x] `paper/metatheory.md:143` Linearity — Documented in §4.8: mechanism is sound, standalone theorems are future work
- [x] `proof.rs:291` — step() now returns None (stuck) instead of panicking on ill-typed terms

### Prior Art Gaps (MODERATE)
- [x] ISPC characterization sharpened — acknowledges foreach_active, distinguishes compile-time vs runtime
- [x] AMD DPP + Intel subgroup operations — new paragraph in §8.5
- [x] Vulkan/SPIR-V subgroup operations — new paragraph in §8.5

### Reproducibility (MODERATE)
- [x] `reproduce/compare_ptx.sh` — $CUDA_ARCH env var, documented architectures
- [x] `reproduce/run.sh` — $POWER_LIMIT/$CLOCK_MHZ env vars, hardware docs
- [x] `README.md` — Zero overhead verification points to compare_ptx.sh + symbol name

### Novelty Claims (tightened)
- [x] Abstract — acknowledges existing runtime approaches before stating type-level contribution
- [x] Conclusion — "the concept is not new; the type-level guarantee is"
- [x] Related work summary — "Positioning" paragraph replaces "Our unique contribution"
- [x] Introduction — "most sophisticated" → "representative" for Hazy

## Paper — Non-Blocking but Important

### Claims Accuracy (LOW)
- [x] `README.md:129` — Gradual typing test count updated: 21 → 25
- [x] `paper/future-and-conclusion.md:38` — gradual.rs test count updated: 16 → 25
- [x] `paper/future-and-conclusion.md:36` — protocol inference: verified correct at 14
- [x] `INSIGHTS.md:165` — LLVM IR claim corrected (shl, not ret)
- [ ] `blog/post.md:73` — "Identical output for typed vs. untyped" imprecise — true for PTX target, not CPU LLVM IR

### Stale Content
- [x] `README.md:134-176` — Project structure tree: added simwarp.rs
- [x] `INTEGRATION.md` — Path deps updated to crates.io
- [x] `cub.rs:241` — inclusive_sum IS now #[deprecated] after our fix, comment is now correct

## API Design — Future Release (v0.3.0)

These are semver-sensitive and should be batched:

- [ ] `data.rs:146` — `Role` has `pub` fields bypassing constructor validation
- [ ] `block.rs:134` — `BlockId(pub u32)` inconsistent with LaneId/WarpId
- [ ] `block.rs:137-141` — `ThreadId` all-public fields
- [ ] `gradual.rs:105` — `DynWarp` derives `Clone`, widening affine escape hatch
- [ ] `lib.rs:153` — `GpuValue` reuses `ActiveSet::sealed::Sealed` for sealing
- [ ] `lib.rs:65-66` — `proof` module pub under feature, name-shadows ActiveSet
- [ ] `lib.rs:236-244` — Prelude missing BallotResult, Fenced, GlobalRegion, etc.

## Missing Tests — Linearity

- [x] `warp.rs` — compile_fail doctest for Warp::clone() rejection (Lemma 4.8)
- [ ] No test artifact for `#[must_use]` warning on warp discard (Lemma 4.9) — Rust warnings can't be tested via compile_fail

## Prior Art — LOW Priority

- [x] NVIDIA compute-sanitizer synccheck — new paragraph in §8.5
- [x] Cooperative Groups — expanded to full paragraph with runtime vs compile-time distinction
- [x] Halide/TVM scheduling — new paragraph in §8.5
- [x] OOPSLA 2023 lockstep verification — new paragraph in §8.5
- [x] Hazy superlative removed

## Status

**38/45 complete.** Remaining 7 are v0.3.0 API breaking changes (parked for next release).
