#!/bin/bash
# check-doc-counts.sh — catch stale test/theorem counts in docs before push
#
# Checks .md files for the specific count formats used in our docs:
#   "N unit + N example + N doc tests (N total)"
#   "N named theorems"
#   "N unit tests"
#
# Excludes: CHANGELOG.md (historical), .review/ (local), INSIGHTS.md (local)
#
# Called by pre-push hook. Standalone: bash scripts/check-doc-counts.sh

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- Collect actual counts ---
UNIT=$(cargo test --workspace --lib --quiet 2>&1 | grep "^test result:" | head -1 | sed -n 's/.*ok\. \([0-9]*\) passed.*/\1/p')
DOC=$(cargo test --doc --quiet 2>&1 | grep "^test result:" | sed -n 's/.*ok\. \([0-9]*\) passed.*/\1/p')

EXAMPLE=0
while IFS= read -r line; do
    n=$(echo "$line" | sed -n 's/.*ok\. \([0-9]*\) passed.*/\1/p')
    [ -n "$n" ] && EXAMPLE=$((EXAMPLE + n))
done < <(cargo test --examples --quiet 2>&1 | grep "^test result:")

TOTAL=$((UNIT + DOC + EXAMPLE))

THEOREMS_BASIC=$(grep -c "^theorem" lean/WarpTypes/Basic.lean 2>/dev/null || echo 0)
THEOREMS_META=$(grep -c "^theorem" lean/WarpTypes/Metatheory.lean 2>/dev/null || echo 0)
THEOREMS=$((THEOREMS_BASIC + THEOREMS_META))

echo "doc-counts: ${UNIT} unit, ${EXAMPLE} example, ${DOC} doc (${TOTAL} total), ${THEOREMS} Lean theorems"

# --- Check .md files for stale counts ---
# Patterns are specific to the formats actually used in our docs.
# This avoids false positives from "21 documented bugs" matching "doc".

STALE_FILE=$(mktemp)
trap 'rm -f "$STALE_FILE"' EXIT

MD_FILES=$(find . -maxdepth 3 -name '*.md' \
    -not -path './.review/*' \
    -not -path './.git/*' \
    -not -path './target/*' \
    -not -path './INSIGHTS.md' \
    -not -path './CHANGELOG.md' \
    | sort)

for file in $MD_FILES; do
    # Pattern: "N unit +" or "N unit test" or "N unit," (test summary contexts)
    grep -nE '[0-9]+ unit [+,t]' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ unit' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && [ "$found" != "$UNIT" ] && echo "STALE: ${file}:${ln} — unit tests: says ${found}, actual ${UNIT}"
    done

    # Pattern: "N doc test" or "N doc (" or "N doc)" (NOT "documented")
    grep -nE '[0-9]+ doc[ )(t]' "$file" 2>/dev/null | grep -v 'documented' | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ doc' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && [ "$found" != "$DOC" ] && echo "STALE: ${file}:${ln} — doc tests: says ${found}, actual ${DOC}"
    done

    # Pattern: "N example test" or "N example +" (NOT "8 worked examples" or "8 real-bug")
    grep -nE '[0-9]+ example [+t]' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ example' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && [ "$found" != "$EXAMPLE" ] && echo "STALE: ${file}:${ln} — example tests: says ${found}, actual ${EXAMPLE}"
    done

    # Pattern: "(N total)" — parenthesized total
    grep -nE '\([0-9]+ total\)' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ total' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && [ "$found" != "$TOTAL" ] && echo "STALE: ${file}:${ln} — total tests: says ${found}, actual ${TOTAL}"
    done

    # Pattern: "N named theorem" (Lean theorem count)
    grep -nE '[0-9]+ named theorem' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ named' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && [ "$found" != "$THEOREMS" ] && echo "STALE: ${file}:${ln} — Lean theorems: says ${found}, actual ${THEOREMS}"
    done

done 2>&1 | tee "$STALE_FILE"

COUNT=$(grep -c "^STALE:" "$STALE_FILE" 2>/dev/null || true)
COUNT=${COUNT:-0}
# Strip any whitespace from wc output
COUNT=$(echo "$COUNT" | tr -d '[:space:]')

if [ "$COUNT" -gt 0 ]; then
    echo ""
    echo "FAIL: ${COUNT} stale doc count(s). Update docs before pushing."
    exit 1
else
    echo "OK: all doc counts match reality."
fi
