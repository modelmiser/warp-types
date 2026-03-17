# Cold Review Prompt v3

Use `/cold-review:run github/warp-types` or paste into a fresh session.

---

Run 3 parallel cold reviews against ./github/warp-types/ — code correctness
only.

SETUP:
- Each agent reads lib.rs first (public API + types)
- Zero-sized phantom types are intentional (type-level state encoding)
- CPU shuffles are IDENTITY (return self) — look for bugs that only
  manifest when two distinct lanes exchange real values
- Skip: style, docs wording, naming, "could be more idiomatic"

FILE ASSIGNMENT (EXCLUSIVE — review ONLY your assigned files):
- Agent 1 (CORE): gpu.rs, sort.rs, shuffle.rs, cub.rs, tile.rs, platform.rs
- Agent 2 (TYPE SYSTEM): lib.rs, warp.rs, active_set.rs, diverge.rs,
  merge.rs, fence.rs, proof.rs, data.rs, warp-types-macros/,
  warp-types-kernel/, warp-types-builder/
- Agent 3 (RUNTIME + BOUNDARY): dynamic.rs, gradual.rs, block.rs,
  simwarp.rs, examples/**, reproduce/**. For src/research/**: scan-only
  for boundary violations (research types leaking into core API, incorrect
  imports, unsound interactions between research Copy warps and core
  non-Copy warps). Do NOT full-review research module internals — they
  are self-contained prototypes proven stable across rounds 13-14.

KNOWN PATTERNS (already fixed — don't re-flag):
- Copy on Warp<S> was removed — linearity is enforced
- ComplementOf/ActiveSet are sealed — external impls blocked
- ballot PTX uses setp.ne.u32 for pred register conversion
- bitonic sort stage_mask is constant per stage (2*xor_mask was fixed)
- DynWarp::reduce_sum_scalar uses count_ones() not hardcoded *32
- DynWarp::ballot rejects 64-lane warps (u32 return incompatible)
- butterfly_reduce_sum has const assert for power-of-2 WIDTH
- inclusive_sum/exclusive_sum are documented broken (no lane_id guard)
- Tile shuffles use width-confined PTX encoding (c parameter verified)
- DynWarp::merge checks full_mask equality
- DynWarp::from_mask auto-detects 32/64-lane width from mask value
- PortableVector::extract/insert use debug_assert + direct index (no % WIDTH wrapping)
- SimWarp width-confined shuffles assert power-of-2 width in 1..=WIDTH
- SimWarp::shuffle_idx clamps OOB src_lane (GPU semantics, not % WIDTH wrapping)
- ValidTileSize is sealed via Sealed supertrait — external impls blocked
- Sealed trait is hard-sealed: _sealed() has no default body, requires SealToken (pub(crate)) — external impls impossible
- sort.rs/cub.rs tests are type-system validation (CPU identity no-op); simwarp tests algorithm correctness
- README test counts match actual (291 unit + 50 example + 28 doc = 369)
- lib.rs module overview includes all public modules including simwarp
- butterfly_reduce is WIDTH-generic (dynamic mask computation, not hardcoded [16,8,4,2,1])
- README test counts all match actual (291 unit, 25 gradual, 369 total)
- builder Cargo.toml fragile TOML name parsing — ACCEPTED (fallback mitigates)
- CpuSimd<WIDTH>::reduce_sum/max/min have const assert WIDTH > 0 — prevents empty-iterator panic
- block.rs ReductionSession is type-state ordering demo (CPU identity no-op, same class as sort.rs/cub.rs)

KNOWN UNTESTED (accepted — don't re-flag):
- shuffle.rs: ballot has no GPU codepath (CPU-only on all targets) — FEATURE GAP
- proof.rs: type_safety_check step limit (MAX_STEPS=1000) — ACCEPTED RISK
- All PTX inline assembly, AMD DPP stubs, warp_kernel macro, builder pipeline — HARDWARE-GATED
- All GPU-targeting examples/reproduce files — HARDWARE-GATED
- GpuWarp64 Platform impl (placeholder, no AMD support yet) — FEATURE GAP
- 64-bit BallotResult (mentioned in docs, not implemented) — FEATURE GAP
- CpuSimd::shuffle_down wraps vs GpuWarp32 clamps — ACCEPTED (trait allows both)
- CpuSimd::shuffle_xor % WIDTH incorrect for non-power-of-2 — ACCEPTED (no callers use non-pow2)
- cub.rs: reduce with non-commutative op silently wrong — ACCEPTED (matches CUB's contract)
- exclusive_sum ignores identity parameter — ACCEPTED (deprecated, documented broken)

SIMWARP COVERAGE (verified with real lane exchange — skip these sequences):
- Butterfly reduce (5-step XOR)
- Bitonic sort (15-step CAS with direction)
- Tile-confined shuffle (width parameter confinement)
- Tile reduce (per-tile butterfly)
- Hillis-Steele scan (correct + bug demonstration)
- 64-lane AMD reduce (6-step XOR)

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
- MODERATE: panic on valid input, wrong result in edge case, API misuse not
  prevented
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
