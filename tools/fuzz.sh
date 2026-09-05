#!/bin/bash
# Fuzz the SAT crate's outside-facing input paths.
#   tools/fuzz.sh [seconds-per-target]     (default 60)
#
# WHY THESE TWO TARGETS
#   dimacs               DIMACS is the only input this crate takes from outside
#                        itself, and it ESTABLISHES the precondition
#                        (`max_variable() < num_vars`) that `solve*` ASSERTS
#                        three call frames later. The target parses, then
#                        solves with a tiny budget: a file that panics deep in
#                        the solver instead of being rejected at the boundary is
#                        the finding, not merely a parser crash.
#   solver_differential  `solve` (scan-every-clause BCP) vs `solve_watched`
#                        (two-watched-literal hot loop with raw-pointer
#                        compaction and unchecked arena indexing) must agree,
#                        AND any Sat model is checked against the clauses the
#                        harness itself built. Agreement alone cannot catch both
#                        being wrong together — they share `ClauseDb`.
#                        This generalises `watch::tests::watched_agrees_with_
#                        original_bcp`, which pins ten fixed seeds.
#
# BOTH ORACLES WERE WATCHED TO FAIL (2026-09-04) before the clean runs were
# believed: dropping a clause from one side trips the agreement assert, and
# inverting the model-check polarity trips the model assert. A fuzz target whose
# oracle cannot fire reports "N million runs, no findings" forever.
#
# IN CI since 2026-09-05 (ci.yml `fuzz` job), via `taiki-e/install-action@cargo-fuzz`
# — a prebuilt binary, because `cargo install cargo-fuzz` is a multi-minute build
# on a cold runner and a slow job gets deleted rather than fixed. CI runs the
# deterministic seed replay (`-runs=0`) as a hard gate, then this script at 60s
# per target, and uploads artifacts/ on failure so a CI-only finding stays
# reproducible.
#
# SEEDS: cargo-fuzz gitignores its own corpus/, so a fresh clone would start
# cold every time. `seeds/<target>/` is committed and passed as a second corpus
# directory — libFuzzer reads it and writes new finds to corpus/, so the seeds
# stay a fixed regression set rather than growing without review.
#
# COVERAGE NOTE: a 90 s dimacs run reached ~1079 edges with 210 of 944 corpus
# entries parsing successfully — i.e. it does get past the error path into the
# solver. Check that ratio before trusting a clean run: a parser target that
# never produces valid input is clean for the wrong reason.
#
# CI also runs `cargo check` on this crate (see ci.yml). That step stays even
# though the fuzz job compiles the targets too: `cargo check` is seconds and runs
# on every push, so compile-rot is still caught the cheap way.
set -u
cd "$(dirname "$0")/../warp-types-sat/fuzz" || exit 2
T=${1:-60}
rc=0
for target in dimacs solver_differential; do
  echo "== $target (${T}s) =="
  cargo +nightly fuzz run "$target" "corpus/$target" "seeds/$target" \
    -- -max_total_time="$T" || rc=1
  echo
done
exit $rc
