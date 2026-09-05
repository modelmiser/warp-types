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
# NOT IN CI, deliberately — `cargo install cargo-fuzz` is a multi-minute build
# on a cold runner, and a slow job gets deleted rather than fixed. See TODO.md
# for the open decision (prebuilt-binary action vs. accepting the cost).
#
# COVERAGE NOTE: corpus/ and artifacts/ are gitignored by cargo-fuzz's own
# template, so each run starts cold. A 90 s dimacs run reached ~1079 edges with
# 210 of 944 corpus entries parsing successfully — i.e. it does get past the
# error path into the solver. Check that ratio before trusting a clean run.
set -u
cd "$(dirname "$0")/../warp-types-sat/fuzz" || exit 2
T=${1:-60}
rc=0
for target in dimacs solver_differential; do
  echo "== $target (${T}s) =="
  cargo +nightly fuzz run "$target" -- -max_total_time="$T" || rc=1
  echo
done
exit $rc
