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

# Map spelled-out number words to integers. The paper prose uses words
# ("fourteen compile-fail doctests"), which is exactly where the count drifted
# out of sync with the digit breakdown — so the guard must read words too.
word2num() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        zero) echo 0 ;; one) echo 1 ;; two) echo 2 ;; three) echo 3 ;; four) echo 4 ;;
        five) echo 5 ;; six) echo 6 ;; seven) echo 7 ;; eight) echo 8 ;; nine) echo 9 ;;
        ten) echo 10 ;; eleven) echo 11 ;; twelve) echo 12 ;; thirteen) echo 13 ;;
        fourteen) echo 14 ;; fifteen) echo 15 ;; sixteen) echo 16 ;; seventeen) echo 17 ;;
        eighteen) echo 18 ;; nineteen) echo 19 ;; twenty) echo 20 ;;
        *) return 1 ;;
    esac
}

# --- Collect actual counts ---
# Unit tests are counted MAIN-CRATE-ONLY (-p warp-types) by design: 326 is the
# figure the docs assert. The workspace lib total is 608, which no doc claims.
UNIT=$(cargo test -p warp-types --lib --quiet 2>&1 | grep "^test result:" | sed -n 's/.*ok\. \([0-9]*\) passed.*/\1/p')

# Doc tests are counted WORKSPACE-WIDE, because that is what CI runs
# (`cargo test --workspace --doc` in .github/workflows/ci.yml). An unscoped
# `cargo test --doc` sees the root crate only (31) and would mark CI's real
# number (37) as stale. One "test result:" line per crate, so sum them —
# same shape as the EXAMPLE loop below. Only `passed` is counted; the 3
# `ignore`d doctests are collected but never run, so they are not tests
# that passed.
DOC=0
while IFS= read -r line; do
    n=$(echo "$line" | sed -n 's/.*ok\. \([0-9]*\) passed.*/\1/p')
    [ -n "$n" ] && DOC=$((DOC + n))
done < <(cargo test --workspace --doc --quiet 2>&1 | grep "^test result:")

# Compile-fail vs runnable ("doc examples") breakdown of the doc tests.
# Classify each doctest rustdoc ACTUALLY collects (from its own inventory) by
# reading the fence at the reported line. This is authoritative: a raw grep of
# ```compile_fail across the workspace overcounts — it includes research-module
# and example fences that rustdoc does not collect (29 grep-hits vs 16 real).
# Workspace-scoped to match DOC above; rustdoc reports member-crate paths
# relative to the workspace root, so the `-f` test resolves from here.
# Uses `< <(...)` (not a pipe) so the increment persists in this shell.
DOC_CF=0
while IFS= read -r entry; do
    f=${entry%% - *}
    ln=$(printf '%s\n' "$entry" | sed -n 's/.*(line \([0-9]*\)).*/\1/p')
    [ -n "$f" ] && [ -n "$ln" ] && [ -f "$f" ] || continue
    fence=$(sed -n "${ln}p" "$f" | sed 's/^[[:space:]]*\/\/[/!][[:space:]]*//')
    case "$fence" in '```compile_fail'*) DOC_CF=$((DOC_CF + 1)) ;; esac
done < <(cargo test --workspace --doc -- --list 2>/dev/null | grep -E '\(line [0-9]+\): test$')
DOC_EXAMPLES=$((DOC - DOC_CF))

EXAMPLE=0
while IFS= read -r line; do
    n=$(echo "$line" | sed -n 's/.*ok\. \([0-9]*\) passed.*/\1/p')
    [ -n "$n" ] && EXAMPLE=$((EXAMPLE + n))
done < <(cargo test --examples --quiet 2>&1 | grep "^test result:")

TOTAL=$((UNIT + DOC + EXAMPLE))

THEOREMS_BASIC=$(grep -c "^theorem" lean/WarpTypes/Basic.lean 2>/dev/null || echo 0)
THEOREMS_META=$(grep -c "^theorem" lean/WarpTypes/Metatheory.lean 2>/dev/null || echo 0)
THEOREMS=$((THEOREMS_BASIC + THEOREMS_META))

echo "doc-counts: ${UNIT} unit, ${EXAMPLE} example, ${DOC} doc = ${DOC_CF} compile-fail + ${DOC_EXAMPLES} examples (${TOTAL} total), ${THEOREMS} Lean theorems"

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
    | sort | while IFS= read -r f; do
        # Skip gitignored files: DEVLOG/TODO/audit-docs are local-only and
        # legitimately reference workspace-scoped counts (e.g. 512) that don't
        # match the main-crate-scope number (317) this script computes.
        git check-ignore -q "$f" 2>/dev/null || printf '%s\n' "$f"
      done)

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

    # Pattern: "(N total)" — parenthesized total — or the bare leading form
    # "N tests (" (blog/post.md: "413 tests (326 unit + ...)"), which states the
    # same total outside any parentheses and would otherwise drift silently.
    grep -nE '\([0-9]+ total\)|[0-9]+ tests \(' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '\([0-9]+ total\)' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] || found=$(echo "$rest" | grep -oE '[0-9]+ tests \(' | head -1 | grep -oE '^[0-9]+') || true
        [ -n "$found" ] && [ "$found" != "$TOTAL" ] && echo "STALE: ${file}:${ln} — total tests: says ${found}, actual ${TOTAL}"
    done

    # Pattern: "N compile-fail" / "<word> compile-fail" (breakdown + paper prose)
    grep -niE '([0-9]+|[a-z]+) compile-fail' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        tok=$(echo "$rest" | grep -oiE '([0-9]+|[a-z]+) compile-fail' | head -1 | grep -oiE '^[0-9a-z]+')
        [ -n "$tok" ] || continue
        if echo "$tok" | grep -qE '^[0-9]+$'; then num=$tok; else num=$(word2num "$tok") || continue; fi
        [ -n "$num" ] && [ "$num" != "$DOC_CF" ] && echo "STALE: ${file}:${ln} — compile-fail doctests: says '${tok}', actual ${DOC_CF}"
    done

    # Pattern: "N doc examples" / "<word> doc examples" (breakdown)
    grep -niE '([0-9]+|[a-z]+) doc example' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        tok=$(echo "$rest" | grep -oiE '([0-9]+|[a-z]+) doc example' | head -1 | grep -oiE '^[0-9a-z]+')
        [ -n "$tok" ] || continue
        if echo "$tok" | grep -qE '^[0-9]+$'; then num=$tok; else num=$(word2num "$tok") || continue; fi
        [ -n "$num" ] && [ "$num" != "$DOC_EXAMPLES" ] && echo "STALE: ${file}:${ln} — doc examples: says '${tok}', actual ${DOC_EXAMPLES}"
    done

    # Pattern: "N named theorem" / "N Lean 4 theorem" / "N Lean theorem".
    # The prose form ("31 Lean 4 theorems") is how the blog states it, and it
    # was invisible to the "named"-only pattern — and wrong.
    grep -nE '[0-9]+ (named|Lean( 4)?) theorem' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ (named|Lean( 4)?) theorem' | head -1 | grep -oE '^[0-9]+') || true
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
