# Cold Review Prompt v4

Fresh scaffold for `--scope all` cold review of warp-types.

---

Run parallel cold reviews against warp-types — all applicable categories,
hierarchical dispatch.

SETUP:
- Each agent reads lib.rs first (public API + types)
- Zero-sized phantom types are intentional (type-level state encoding)
- CPU shuffles are IDENTITY (return self) — bugs only manifest when two
  distinct lanes exchange real values
- SimWarp (src/simwarp.rs) provides real lane exchange for algorithm testing
- Skip: style, docs wording, naming, "could be more idiomatic"

PROJECT STRUCTURE:
- Workspace: warp-types (core), warp-types-macros, warp-types-kernel,
  warp-types-builder
- Core src/: 18 modules (active_set, block, cub, data, diverge, dynamic,
  fence, gpu, gradual, merge, platform, proof, shuffle, simwarp, sort,
  tile, warp, lib)
- Research: src/research/ — 27 experimental modules (self-contained
  prototypes, not public API)
- Sub-crates: warp-types-macros/ (proc macros), warp-types-kernel/
  (kernel attribute), warp-types-builder/ (build pipeline)
- Examples: 8 real-bug reproductions from NVIDIA, PyTorch, OpenCV, LLVM, etc.
- Reproduce: GPU demo scripts + CUDA comparison code
- Paper: 12-section research paper (paper/*.md)
- Lean: Formal metatheory proofs (lean/WarpTypes/{Basic,Metatheory}.lean)
- Tutorial: tutorial/README.md
- Blog: blog/post.md

FILE ASSIGNMENT (for flat dispatch — hierarchical dispatch overrides per
category):
- Agent 1 (CORE): gpu.rs, sort.rs, shuffle.rs, cub.rs, tile.rs,
  platform.rs, simwarp.rs
- Agent 2 (TYPE SYSTEM): lib.rs, warp.rs, active_set.rs, diverge.rs,
  merge.rs, fence.rs, proof.rs, data.rs, warp-types-macros/,
  warp-types-kernel/, warp-types-builder/
- Agent 3 (RUNTIME + BOUNDARY): dynamic.rs, gradual.rs, block.rs,
  examples/**, reproduce/**, src/research/** (boundary scan only)

KNOWN PATTERNS (already fixed — don't re-flag):
- CpuSimd::shuffle_down used modular wrap instead of GPU-style clamp (platform.rs:shuffle_down)
- Tile::shuffle_xor and shuffle_down used debug_assert for mask/delta bounds (tile.rs:shuffle_xor, shuffle_down)
- DynDiverge::diverge_dynamic used debug_assert for stray-bit mask validation (dynamic.rs:diverge_dynamic)
- DynWarp::from_mask had ambiguous width inference for 32-bit masks; added from_mask_32/from_mask_64 (gradual.rs:from_mask)
- WarpPtrMut derived Clone, enabling aliased mutable writes via shared ref (research/coalescing.rs:WarpPtrMut)

KNOWN UNTESTED (accepted — don't re-flag):
- [DOCUMENTED] Warp::kernel_entry() can be called multiple times, bypassing linear typestate (warp.rs — fundamental affine vs linear limitation)
- [PAPER-SCOPE] shuffle_xor_within is in paper §3.3 formal rules but only in research module, not public API
- [PAPER-SCOPE] Loop typing rules (§5.1: LOOP-UNIFORM, LOOP-CONVERGENT, LOOP-VARYING, LOOP-PHASED) have no implementation
- [LEAN-SCOPE] Lean mergeVal hardcodes ActiveSet.all — nested merge (merge_within) has no formal backing (documented §4.8)
- [LEAN-SCOPE] Lean substitution relies on value restriction for capture avoidance (sound but fragile)
- [LEAN-SCOPE] Lean/Rust correspondence gap — independent formalizations, proofs don't directly certify Rust code

REVIEW PROTOCOL (per file):
1. Read end-to-end
2. Standard check: arithmetic, off-by-one, state machine, invariant
3. ADVERSARIAL CHECK: For each public safe function, try to construct
   a safe-code call sequence that violates its documented invariant.
   If sealed traits block you, verify the seal is complete.
4. CALLER-INVARIANT CHECK: For each public function, grep all callers.
   Verify each caller's assumed invariant is guaranteed in all code paths.
5. CROSS-FILE INVARIANT: List invariants your files EXPORT that files
   outside your assignment depend on.
6. TEST COVERAGE: Classify untested paths as:
   - UNTESTED-HARD: cfg-gated or requires hardware (not actionable now)
   - UNTESTED-TEST: could have a test but doesn't (actionable)
   - UNTESTED-FEATURE: feature gap, not a missing test (informational only)

SEVERITY:
- CRITICAL: soundness hole, silent wrong results, UB in safe code
- MODERATE: panic on valid input, wrong result in edge case, API misuse
  not prevented
- LOW: doc/code mismatch, unreachable panic, defensive check missing

REPORTING (structured, per file):
  CLEAN: filename.rs
  BUG: filename.rs:line [CRITICAL|MODERATE|LOW] — summary
    Evidence (2-3 lines) + affected callers
  UNTESTED-TEST: filename.rs:line — description
  UNTESTED-FEATURE: filename.rs:line — description (why not testable as-is)
  UNTESTED-HARD: filename.rs:line — description (why untestable)
  EXPORT: invariant description → consuming files

End with:
  N CRITICAL, N MODERATE, N LOW
  N UNTESTED-TEST, N UNTESTED-FEATURE, N UNTESTED-HARD
  N EXPORT invariants
