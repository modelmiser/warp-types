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
# Per-line record of every number an existing pattern actually extracted:
# "<file>\t<line>\t<number>". The unguarded lint below diffs count-vocabulary
# numbers against this, so a pattern's silence becomes visible instead of
# passing for approval. Words are recorded as their integer value, so
# "sixteen" and "16" are the same fact.
VALIDATED_FILE=$(mktemp)
trap 'rm -f "$STALE_FILE" "$VALIDATED_FILE"' EXIT

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

{
for file in $MD_FILES; do
    # Pattern: "N unit +" or "N unit test" or "N unit," (test summary contexts)
    grep -nE '[0-9]+ unit [+,t]' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ unit' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$UNIT" ] && echo "STALE: ${file}:${ln} — unit tests: says ${found}, actual ${UNIT}"
    done

    # Pattern: "N doc test" or "N doc (" or "N doc)" (NOT "documented")
    grep -nE '[0-9]+ doc[ )(t]' "$file" 2>/dev/null | grep -v 'documented' | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ doc' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$DOC" ] && echo "STALE: ${file}:${ln} — doc tests: says ${found}, actual ${DOC}"
    done

    # Pattern: "N example test" or "N example +" (NOT "8 worked examples" or "8 real-bug")
    grep -nE '[0-9]+ example [+t]' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ example' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$EXAMPLE" ] && echo "STALE: ${file}:${ln} — example tests: says ${found}, actual ${EXAMPLE}"
    done

    # Pattern: "(N total)" — parenthesized total — or the bare leading form
    # "N tests (" (blog/post.md: "413 tests (326 unit + ...)"), which states the
    # same total outside any parentheses and would otherwise drift silently.
    grep -nE '\([0-9]+ total\)|[0-9]+ tests \(' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '\([0-9]+ total\)' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] || found=$(echo "$rest" | grep -oE '[0-9]+ tests \(' | head -1 | grep -oE '^[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$TOTAL" ] && echo "STALE: ${file}:${ln} — total tests: says ${found}, actual ${TOTAL}"
    done

    # Pattern: "N compile-fail" / "<word> compile-fail" (breakdown + paper prose)
    grep -niE '([0-9]+|[a-z]+) compile-fail' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        tok=$(echo "$rest" | grep -oiE '([0-9]+|[a-z]+) compile-fail' | head -1 | grep -oiE '^[0-9a-z]+')
        [ -n "$tok" ] || continue
        if echo "$tok" | grep -qE '^[0-9]+$'; then num=$tok; else num=$(word2num "$tok") || continue; fi
        [ -n "$num" ] && printf '%s\t%s\t%s\n' "$file" "$ln" "$num" >> "$VALIDATED_FILE"
        [ -n "$num" ] && [ "$num" != "$DOC_CF" ] && echo "STALE: ${file}:${ln} — compile-fail doctests: says '${tok}', actual ${DOC_CF}"
    done

    # Pattern: "N doc examples" / "<word> doc examples" (breakdown)
    grep -niE '([0-9]+|[a-z]+) doc example' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        tok=$(echo "$rest" | grep -oiE '([0-9]+|[a-z]+) doc example' | head -1 | grep -oiE '^[0-9a-z]+')
        [ -n "$tok" ] || continue
        if echo "$tok" | grep -qE '^[0-9]+$'; then num=$tok; else num=$(word2num "$tok") || continue; fi
        [ -n "$num" ] && printf '%s\t%s\t%s\n' "$file" "$ln" "$num" >> "$VALIDATED_FILE"
        [ -n "$num" ] && [ "$num" != "$DOC_EXAMPLES" ] && echo "STALE: ${file}:${ln} — doc examples: says '${tok}', actual ${DOC_EXAMPLES}"
    done

    # Pattern: "N named theorem" / "N Lean 4 theorem" / "N Lean theorem".
    # The prose form ("31 Lean 4 theorems") is how the blog states it, and it
    # was invisible to the "named"-only pattern — and wrong.
    grep -nE '[0-9]+ (named|Lean( 4)?) theorem' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ (named|Lean( 4)?) theorem' | head -1 | grep -oE '^[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$THEOREMS" ] && echo "STALE: ${file}:${ln} — Lean theorems: says ${found}, actual ${THEOREMS}"
    done

done

# --- Unguarded count lint (deny-by-default over count vocabulary) ---
# The patterns above can only fail on counts they already know how to read; a
# count phrased any other way passes silently, and silence is indistinguishable
# from approval. This pass inverts that for the vocabulary the guard owns:
# every number sitting within three words of test/doctest/example(-as-test)/
# compile-fail/theorem/total must have been extracted by some pattern above,
# or be explicitly waived with "<!-- unguarded: <reason> -->" on the line.
#
# Deliberately NOT deny-by-default over all numerals: the corpus is full of
# line counts, percentages, lane widths and section numbers, and flagging those
# collapses into false positives. Three exclusions keep it honest rather than
# noisy: hyphenated compounds ("three-domain", "16-tile") are adjectives, not
# counts; "example(s)" only counts when test-adjacent ("N example tests",
# "N doc examples"), so prose like "8 worked bug examples" is out of scope;
# and identifier-position numerals are not quantities ("Lean 4" is a version --
# which is why the theorem pattern above already spells it as a qualifier --
# "Figure 7", percentages, "~170", and numerals inside `code spans`).
awk -v VF="$VALIDATED_FILE" '
function clean(s) { gsub(/^[^0-9A-Za-z]+/, "", s); gsub(/[^0-9A-Za-z-]+$/, "", s); return s }
function low(s) { return tolower(clean(s)) }
function num(t,   c) {
    if (t ~ /[`%~]/) return -1
    c = clean(t)
    if (c ~ /-/) return -1
    if (c ~ /^[0-9]+$/) return c + 0
    c = tolower(c)
    if (c in W) return W[c]
    return -1
}
function counted(t) { return (t ~ /^(tests?|doctests?|theorems?|examples?|compile-fail)$/) }
BEGIN {
    split("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty", A, " ")
    for (i = 1; i <= 21; i++) W[A[i]] = i - 1
    while ((getline l < VF) > 0) { split(l, f, "\t"); V[f[1] SUBSEP f[2] SUBSEP (f[3] + 0)] = 1 }
}
{
    waived = 0; reason = ""
    if (match($0, /<!-- unguarded:[^>]*-->/)) {
        waived = 1
        reason = substr($0, RSTART, RLENGTH)
        sub(/^<!-- unguarded:[ \t]*/, "", reason)
        sub(/[ \t]*-->$/, "", reason)
    }
    n = split($0, T, /[ \t]+/)
    for (i = 1; i <= n; i++) {
        v = num(T[i])
        if (v < 0) continue
        if (i > 1 && low(T[i-1]) ~ /^(lean|version|v|figure|fig|table|section|chapter|appendix|lemma|theorem)$/) continue
        if ((FILENAME SUBSEP FNR SUBSEP v) in V) continue
        for (j = i + 1; j <= i + 3 && j <= n; j++) {
            if (num(T[j]) >= 0) break          # the next numeral heads its own phrase
            t = low(T[j])
            if (t == "example" || t == "examples") {
                if (!(low(T[j-1]) == "doc" || low(T[j+1]) ~ /^tests?$/)) continue
            } else if (t == "total") {
                # "total" names no count of its own; what is being totalled must
                # be guard vocabulary ("N total", "N doc tests total") and not
                # something else that happens to be summed ("52 cells total").
                ok = 1
                for (k = i + 1; k < j; k++) if (!counted(low(T[k]))) ok = 0
                if (!ok) continue
            } else if (t !~ /^(tests?|doctests?|theorems?|compile-fail)$/) continue
            phrase = ""
            for (k = i; k <= j; k++) phrase = phrase (k > i ? " " : "") T[k]
            if (waived) printf "WAIVED: %s:%d — \"%s\" — %s\n", FILENAME, FNR, phrase, reason
            else printf "UNGUARDED: %s:%d — \"%s\" — no pattern validates this count; guard it or waive it\n", FILENAME, FNR, phrase
            break
        }
    }
}' $MD_FILES
} 2>&1 | tee "$STALE_FILE"

COUNT=$(grep -c "^STALE:" "$STALE_FILE" 2>/dev/null || true)
COUNT=${COUNT:-0}
# Strip any whitespace from wc output
COUNT=$(echo "$COUNT" | tr -d '[:space:]')

UNGUARDED_COUNT=$(grep -c "^UNGUARDED:" "$STALE_FILE" 2>/dev/null || true)
UNGUARDED_COUNT=$(echo "${UNGUARDED_COUNT:-0}" | tr -d '[:space:]')

if [ "$COUNT" -gt 0 ] || [ "$UNGUARDED_COUNT" -gt 0 ]; then
    echo ""
    [ "$COUNT" -gt 0 ] && echo "FAIL: ${COUNT} stale doc count(s). Update docs before pushing."
    [ "$UNGUARDED_COUNT" -gt 0 ] && echo "FAIL: ${UNGUARDED_COUNT} unguarded count(s). Guard the phrasing or waive with '<!-- unguarded: <reason> -->'."
    exit 1
else
    echo "OK: all doc counts match reality."
fi
