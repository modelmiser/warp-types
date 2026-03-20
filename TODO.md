# warp-types TODO — Cold Review Findings (2026-03-20)

Non-code findings from `--scope all` cold review (13 categories, 5 waves).
Code fixes already committed (`e2b0094ef`). These require paper/doc/API work.

## Paper — Submission-Blocking

### Terminology Drift (MODERATE)
- [x] `paper/README.md:34` — Contribution #2 updated to "linear typestate over active-set lattice"
- [x] `paper/extensions.md:211` — Updated to "linear typestate for warp divergence"
- [x] `paper/README.md:28` — Page count updated to ~19

### Unsupported Claims (MODERATE)
- [x] `paper/evaluation.md:174` — Annotation burden corrected: 16.7% mean (11.3%–25.3%), methodology documented (counted lines with type annotations across 8 examples)
- [ ] `blog/post.md:16` — "They're the reason GPU matrix multiply reaches 90%+ of peak throughput" — causal overclaim about shuffles and GEMM. Hedge or cite.

### Paper-Code Correspondence (MODERATE)
- [ ] `shuffle.rs` vs §3.3 BALLOT rule — Paper allows `ballot(Warp<S>, pred)` for any `S`; code restricts to `Warp<All>`. Either relax the code or update the paper's typing rule. Undocumented over-restriction.
- [ ] Lean `Metatheory.lean:43` vs §4.2 — Lean merge only models `IsComplementAll` (always produces `Warp<All>`). Paper claims general merge `S1∪S2` and nested merge. Metatheory claims (Theorem 4.2) are broader than what is mechanized. Document the gap or extend Lean.
- [ ] `proof.rs:207` — Merge type-check only verifies disjointness, not covering. Paper's merge premise requires both. Fix proof.rs or document as known simplification.
- [ ] `shuffle.rs:120` vs §3.1 — Paper says reductions return `SingleLane<T, 0>`; code returns `Uniform<T>` (butterfly gives all lanes the sum). Document as simplification.

### Formalism (MODERATE)
- [ ] `Metatheory.lean:43` — `Step.mergeVal` hardcodes `ActiveSet.all` as result. Would break under nested-merge generalization. Document or fix.
- [ ] `paper/metatheory.md:143` — Linearity (Lemmas 4.8/4.9) claimed but no standalone Lean theorems. Add or document as future work.
- [ ] `proof.rs:291` — `step()` panics on ill-typed terms instead of returning `None` (getting stuck). Masks violations in the test harness.

### Prior Art Gaps (MODERATE)
- [x] `related-work.md:80` — ISPC characterization sharpened: acknowledges `foreach_active` and runtime active set; distinguishes compile-time (ours) vs runtime (ISPC) tracking
- [ ] `related-work.md` — AMD DPP/Intel subgroup operations need individual treatment (paper claims portability, uses u64 masks).
- [ ] `related-work.md` — Vulkan/SPIR-V subgroup operations missing. Most widely deployed subgroup API. `subgroupBallot` has the same convergence requirement.

### Reproducibility (MODERATE)
- [x] `reproduce/compare_ptx.sh:12` — Now uses `$CUDA_ARCH` env var with fallback to sm_89, documents common architectures
- [x] `reproduce/run.sh:8-9` — Now uses `$POWER_LIMIT` and `$CLOCK_MHZ` env vars, documents hardware specificity
- [x] `README.md:124` — Zero overhead verification now points to `compare_ptx.sh` and names the symbol to search for

## Paper — Non-Blocking but Important

### Claims Accuracy (LOW)
- [x] `README.md:129` — Gradual typing test count updated: 21 → 25
- [x] `paper/future-and-conclusion.md:38` — gradual.rs test count updated: 16 → 25
- [ ] `paper/future-and-conclusion.md:36` — protocol inference: says 14 tests, actual 14 (verified — was correct)
- [ ] `INSIGHTS.md:165` — Claims `zero_overhead_butterfly` compiles to `ret i32 %data`; actual is `shl i32 %data, 5`.
- [ ] `blog/post.md:73` — "Identical output for typed vs. untyped" imprecise — true for PTX target, not CPU LLVM IR.

### Stale Content
- [x] `README.md:134-176` — Project structure tree: added `simwarp.rs`
- [x] `INTEGRATION.md:14,78,167` — Path deps updated to `warp-types = "0.2"`
- [ ] `cub.rs:241` — Test comment claims `inclusive_sum` is `#[deprecated]` — now true after our fix, verify comment matches.

## API Design — Future Release

These are semver-sensitive and should be batched for v0.3.0:

- [ ] `data.rs:146` — `Role` has `pub` fields bypassing constructor validation. Make private in 0.3.0.
- [ ] `block.rs:134` — `BlockId(pub u32)` inconsistent with `LaneId`/`WarpId` (private fields + constructors).
- [ ] `block.rs:137-141` — `ThreadId` all-public fields, same issue.
- [ ] `gradual.rs:105` — `DynWarp` derives `Clone`, widening affine escape hatch. Consider removing Clone in 0.3.0.
- [ ] `lib.rs:153` — `GpuValue` reuses `ActiveSet::sealed::Sealed` for sealing, coupling two concerns in public API surface.
- [ ] `lib.rs:65-66` — `proof` module `pub` under `formal-proof` feature but not documented in re-exports. `proof::ActiveSet` name-shadows `active_set::ActiveSet`.
- [ ] `lib.rs:236-244` — Prelude missing `BallotResult`, `Fenced`, `GlobalRegion`, `Role`, `SharedRegion`, `Permutation`.

## Missing Tests — Linearity

- [ ] No compile-fail test for `warp.clone()` rejection (Lemma 4.8).
- [ ] No test artifact for `#[must_use]` warning on warp discard (Lemma 4.9).

## Prior Art — LOW Priority

- [ ] NVIDIA `compute-sanitizer --tool synccheck` in related work (not just evaluation).
- [ ] Cooperative Groups deserves a paragraph in related work (not just a clause).
- [ ] Halide/TVM scheduling as design-level alternative (cited as bug source but not as related work).
- [ ] OOPSLA 2023 "Verifying SIMT Programs via Lockstep Execution" — directly in scope.
- [ ] `introduction.md:24` — "most sophisticated persistent thread program" is subjective superlative.
