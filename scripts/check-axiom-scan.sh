#!/bin/bash
# Guard: lean/axiom-scan.txt is what lean/AxiomScan.lean prints today.
#
# The READMEs and the paper say every theorem constant of the library depends
# only on Lean's three standard axioms. Until 2026-11-25 that sentence rested on
# a scan whose script and output were not in the repository. This re-runs the
# scan and compares it with the committed output; the scan itself exits non-zero
# if any theorem depends on another axiom (a `sorry` shows up as `sorryAx`).
#
# Needs `lake` (elan). Without it the check is SKIPPED, loudly — a skip is not a
# pass; CI (.github/workflows/lean.yml) runs the same comparison.
set -euo pipefail
cd "$(dirname "$0")/../lean"
if ! command -v lake >/dev/null; then
  echo "axiom scan SKIPPED: no lake on PATH (not checked here; lean.yml checks it in CI)"
  exit 0
fi
lake build --quiet >/dev/null 2>&1 || { echo "axiom scan: lake build failed"; exit 1; }
out=$(mktemp "${TMPDIR:-/tmp}/axiom-scan.XXXXXX")
trap 'rm -f "$out"' EXIT
if ! lake env lean AxiomScan.lean > "$out" 2>&1; then
  echo "AXIOM SCAN FAILED:"; tail -5 "$out"; exit 1
fi
if ! diff -u axiom-scan.txt "$out"; then
  echo "AXIOM SCAN DRIFT: lean/axiom-scan.txt is stale. Regenerate: cd lean && lake env lean AxiomScan.lean > axiom-scan.txt"
  echo "…and update the count wherever the docs quote it (README.md, lean/README.md, paper)."
  exit 1
fi
echo "axiom scan OK ($(grep -E '^total theorem constants' axiom-scan.txt))"
