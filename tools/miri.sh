#!/bin/bash
# Run warp-types' host-side unsafe under Miri.
# From the repo root:  tools/miri.sh [extra cargo test args]
#
# WHAT THIS COVERS, AND WHAT IT CANNOT
#
# Miri interprets HOST code. The repo's unsafe splits three ways:
#
#   warp-types-sat  watch/analyze/bcp/trail/clause — the pointer-based BCP
#                   compaction loop, the arena `get_unchecked` accessors, and
#                   the trail's unchecked position map. This is the real
#                   host-side memory-unsafety surface and it is what this
#                   script exists for.
#   warp-types      research::coalescing — the only host-side raw-pointer
#                   derefs in the root crate (safe load/store fns behind an
#                   unsafe constructor).
#   NOT COVERED     src/gpu.rs's 33 unsafe sites are all
#                   `#[cfg(target_arch = "nvptx64")]`, and reproduce/ plus
#                   examples/gpu-project sit outside the workspace. None of
#                   it compiles for the host, so Miri cannot reach any of it
#                   — those sites are verified only by running on real
#                   hardware. gpu_launcher.rs needs CUDA for the same reason.
#
# THE LIMIT WORTH REMEMBERING: Miri reports UB on paths the tests EXECUTE. It
# cannot invent a call. An unsafe fn no test reaches stays unverified however
# long this runs.
#
# WHY THE SKIPS
#   add_clause_rejects_oversized_clause / arena_overflow
#       allocate against the 2^31-word arena cap. Minutes-to-hours under
#       interpretation to re-check pure arithmetic; the native suite covers
#       them on every run.
#   bench::
#       bench_suite_runs solves whole 3-SAT instances. Same reasoning.
#   Without these two skips the run does not finish in CI, and a Miri job
#   that times out is a Miri job that gets deleted.
#
# READING THE OUTPUT: libtest's "finished in N.NNs" is IDENTICAL across runs
# because Miri emulates a deterministic clock. That is not a cached run —
# check the wallclock this script prints instead.
set -u
cd "$(dirname "$0")/.." || exit 2

SAT_FILTER=(watch:: analyze:: bcp:: trail:: clause::
            --skip add_clause_rejects_oversized --skip arena_overflow)
EXTRA=("$@")   # anything the caller passed through, e.g. --no-fail-fast

run() { # run <label> <miriflags>
  local label=$1 flags=$2 t0 rc
  echo "== $label =="
  t0=$(date +%s)
  MIRIFLAGS="$flags" rustup run nightly cargo miri test \
    -p warp-types-sat --lib "${EXTRA[@]+"${EXTRA[@]}"}" -- "${SAT_FILTER[@]}"
  rc=$?
  MIRIFLAGS="$flags" rustup run nightly cargo miri test \
    -p warp-types --lib "${EXTRA[@]+"${EXTRA[@]}"}" -- research::coalescing
  rc=$(( rc | $? ))
  echo "-- $label wallclock $(( $(date +%s) - t0 ))s, exit $rc"
  echo
  return $rc
}

run "Stacked Borrows (default model)" "-Zmiri-ignore-leaks"
sb=$?
run "Tree Borrows (the other candidate model)" "-Zmiri-ignore-leaks -Zmiri-tree-borrows"
tb=$?

echo "stacked-borrows exit=$sb  tree-borrows exit=$tb"
exit $(( sb | tb ))
