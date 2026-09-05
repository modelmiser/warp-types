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

# Main-crate doc tests, as distinct from the workspace-wide DOC above. The docs
# quote both: 37 workspace-wide, of which 31 are the main crate's. Without an
# actual for the 31 it was an unguarded numeral sitting next to guarded ones,
# which is the shape that drifts.
DOC_MAIN=$(cargo test -p warp-types --doc --quiet 2>&1 | grep "^test result:" | sed -n 's/.*ok\. \([0-9]*\) passed.*/\1/p')
DOC_MAIN=${DOC_MAIN:-0}

# The main-crate-only figure a reader can actually reproduce with one command
# (`cargo test -p warp-types --lib --doc --examples`). TOTAL mixes scopes —
# main-crate unit+example plus WORKSPACE doc — so it is a real sum of stated
# parts but is not any single invocation's output. Publishing both is what makes
# the composite reconcilable instead of merely disclosed.
TOTAL_MAIN=$((UNIT + DOC_MAIN + EXAMPLE))

THEOREMS_BASIC=$(grep -c "^theorem" lean/WarpTypes/Basic.lean 2>/dev/null || echo 0)
THEOREMS_META=$(grep -c "^theorem" lean/WarpTypes/Metatheory.lean 2>/dev/null || echo 0)
THEOREMS=$((THEOREMS_BASIC + THEOREMS_META))

# Bug untypability proofs: theorems named `bugN_...` in the Lean sources. This
# is the only unambiguous definition in the repo — a grep for "untypable" in
# theorem names returns 6, and they are a DIFFERENT set (Fence/Reduce/Csp domain
# rules), so counting those would guard the wrong number and look like a guard.
BUG_PROOFS=$(grep -rhE '^theorem bug[0-9]+_' lean/WarpTypes/*.lean 2>/dev/null | wc -l | tr -d '[:space:]')

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
        [ -n "$found" ] && printf '%s\t%s\t%s\ttest\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$UNIT" ] && echo "STALE: ${file}:${ln} — unit tests: says ${found}, actual ${UNIT}"
    done

    # Pattern: "N doc test" or "N doc (" or "N doc)" (NOT "documented")
    grep -nE '[0-9]+ doc[ )(t]' "$file" 2>/dev/null | grep -v 'documented' | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ doc' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\ttest\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$DOC" ] && echo "STALE: ${file}:${ln} — doc tests: says ${found}, actual ${DOC}"
    done

    # Pattern: "N example test" or "N example +" (NOT "8 worked examples" or "8 real-bug")
    grep -nE '[0-9]+ example [+t]' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ example' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\texample\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$EXAMPLE" ] && echo "STALE: ${file}:${ln} — example tests: says ${found}, actual ${EXAMPLE}"
    done

    # Pattern: "(N total)" — parenthesized total — or the bare leading form
    # "N tests (" (blog/post.md: "413 tests (326 unit + ...)"), which states the
    # same total outside any parentheses and would otherwise drift silently.
    grep -nE '\([0-9]+ total\)|[0-9]+ tests \(' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '\([0-9]+ total\)' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] || found=$(echo "$rest" | grep -oE '[0-9]+ tests \(' | head -1 | grep -oE '^[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\ttotal\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && printf '%s\t%s\t%s\ttest\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$TOTAL" ] && echo "STALE: ${file}:${ln} — total tests: says ${found}, actual ${TOTAL}"
    done

    # Pattern: "N compile-fail" / "<word> compile-fail" (breakdown + paper prose)
    grep -niE '([0-9]+|[a-z]+) compile-fail' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        tok=$(echo "$rest" | grep -oiE '([0-9]+|[a-z]+) compile-fail' | head -1 | grep -oiE '^[0-9a-z]+')
        [ -n "$tok" ] || continue
        if echo "$tok" | grep -qE '^[0-9]+$'; then num=$tok; else num=$(word2num "$tok") || continue; fi
        [ -n "$num" ] && printf '%s\t%s\t%s\tcompile-fail\n' "$file" "$ln" "$num" >> "$VALIDATED_FILE"
        [ -n "$num" ] && [ "$num" != "$DOC_CF" ] && echo "STALE: ${file}:${ln} — compile-fail doctests: says '${tok}', actual ${DOC_CF}"
    done

    # Pattern: "N doc examples" / "<word> doc examples" (breakdown)
    grep -niE '([0-9]+|[a-z]+) doc example' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        tok=$(echo "$rest" | grep -oiE '([0-9]+|[a-z]+) doc example' | head -1 | grep -oiE '^[0-9a-z]+')
        [ -n "$tok" ] || continue
        if echo "$tok" | grep -qE '^[0-9]+$'; then num=$tok; else num=$(word2num "$tok") || continue; fi
        [ -n "$num" ] && printf '%s\t%s\t%s\texample\n' "$file" "$ln" "$num" >> "$VALIDATED_FILE"
        [ -n "$num" ] && [ "$num" != "$DOC_EXAMPLES" ] && echo "STALE: ${file}:${ln} — doc examples: says '${tok}', actual ${DOC_EXAMPLES}"
    done

    # Pattern: "N bug proofs" / "N (bug|mechanized) untypability proofs".
    # The modifiers are enumerated, not wildcarded, so prose like "zero extra
    # proof" and "Zero new proofs" stays out of scope rather than being counted
    # as an inventory claim it never was.
    grep -niE '([0-9]+|[a-z]+) (bug|mechanized) (untypability )?proofs?|([0-9]+|[a-z]+) untypability proofs?' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        tok=$(echo "$rest" | grep -oiE '([0-9]+|[a-z]+) (bug|mechanized) (untypability )?proofs?|([0-9]+|[a-z]+) untypability proofs?' | head -1 | grep -oiE '^[0-9a-z]+')
        [ -n "$tok" ] || continue
        if echo "$tok" | grep -qE '^[0-9]+$'; then num=$tok; else num=$(word2num "$tok") || continue; fi
        [ -n "$num" ] && printf '%s\t%s\t%s\tproof\n' "$file" "$ln" "$num" >> "$VALIDATED_FILE"
        [ -n "$num" ] && [ "$num" != "$BUG_PROOFS" ] && echo "STALE: ${file}:${ln} — bug untypability proofs: says '${tok}', actual ${BUG_PROOFS}"
    done

    # Pattern: "the N in the main crate" — the main-crate share of the
    # workspace doc-test count, quoted alongside it in the paper.
    grep -nE 'the [0-9]+ in the main crate' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE 'the [0-9]+ in the main crate' | head -1 | grep -oE '[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\ttest\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$DOC_MAIN" ] && echo "STALE: ${file}:${ln} — main-crate doc tests: says ${found}, actual ${DOC_MAIN}"
    done

    # Pattern: "N main-crate total" — the one-command reproducible figure.
    grep -nE '[0-9]+ main-crate total' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ main-crate total' | head -1 | grep -oE '^[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\ttotal\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$TOTAL_MAIN" ] && echo "STALE: ${file}:${ln} — main-crate total: says ${found}, actual ${TOTAL_MAIN}"
    done

    # Pattern: "N named theorem" / "N Lean 4 theorem" / "N Lean theorem".
    # The prose form ("31 Lean 4 theorems") is how the blog states it, and it
    # was invisible to the "named"-only pattern — and wrong.
    grep -nE '[0-9]+ (named|Lean( 4)?) theorem' "$file" 2>/dev/null | while IFS=: read -r ln rest; do
        found=$(echo "$rest" | grep -oE '[0-9]+ (named|Lean( 4)?) theorem' | head -1 | grep -oE '^[0-9]+') || true
        [ -n "$found" ] && printf '%s\t%s\t%s\ttheorem\n' "$file" "$ln" "$found" >> "$VALIDATED_FILE"
        [ -n "$found" ] && [ "$found" != "$THEOREMS" ] && echo "STALE: ${file}:${ln} — Lean theorems: says ${found}, actual ${THEOREMS}"
    done

done

# --- Unguarded count lint (deny-by-default over count vocabulary) ---
# The patterns above can only fail on counts they already know how to read; a
# count phrased any other way passes silently, and silence is indistinguishable
# from approval. This pass inverts that for the vocabulary the guard owns:
# every number sitting within three words of test/doctest/example(-as-test)/
# compile-fail/theorem/total must have been extracted by some pattern above,
# or be explicitly waived with "<!-- unguarded: <count> — <reason> -->".
# The waiver must NAME the count it covers (digits or a number-word, comma
# separated for several). A waiver that names none covers the whole line,
# so a number added to that line later inherits an unrelated reason and is
# never checked again — that is now a hard failure (WAIVER-UNSCOPED).
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
    if (index(t, "§") > 0) return -1        # "(§3)" is a cross-reference, not a count
    c = clean(t)
    if (c ~ /-/) return -1
    if (c ~ /^[0-9]+$/) return c + 0
    c = tolower(c)
    if (c in W) return W[c]
    return -1
}
function counted(t) { return (t ~ /^(tests?|doctests?|theorems?|examples?|proofs?|compile-fail)$/) }
# Canonical vocabulary term, so a validated count is keyed to WHAT it counts and
# not just to its value. Without this, "37 doc tests and 37 theorems" needed only
# one of the two validated: the key was (file, line, value), so approving either
# 37 approved both. Singular/plural and doctest/test collapse to one term.
function canon(s) { sub(/s$/, "", s); if (s == "doctest") s = "test"; return s }
BEGIN {
    split("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty", A, " ")
    for (i = 1; i <= 21; i++) W[A[i]] = i - 1
    while ((getline l < VF) > 0) { split(l, f, "\t"); V[f[1] SUBSEP f[2] SUBSEP (f[3] + 0) SUBSEP f[4]] = 1 }
}
{
    waived = 0; unscoped = 0; reason = ""
    split("", WV)
    scan = $0
    if (match($0, /<!-- unguarded:[^>]*-->/)) {
        waived = 1
        ws = RSTART; wl = RLENGTH
        reason = substr($0, ws, wl)
        # The waiver comment is metadata, not prose: drop it from the text that
        # gets scanned. Otherwise the count named in the waiver is itself read as
        # a claim ("unguarded: 14 — module-scoped test count" parses as "14 test"),
        # and the guard reports numbers it invented.
        scan = substr($0, 1, ws - 1) " " substr($0, ws + wl)
        sub(/^<!-- unguarded:[ \t]*/, "", reason)
        sub(/[ \t]*-->$/, "", reason)
        # A waiver must NAME the counts it covers. Leading numerals (digits or
        # number-words, comma separated) are its scope; the rest is the reason.
        # Without this the waiver covered the whole LINE, so a count added later
        # to an already-waived line inherited a stale, unrelated reason and was
        # never checked again — a hole that widens silently and looks like a
        # clean run. An unscoped waiver is now a failure, not a pass.
        nscope = 0
        while (match(reason, /^[ \t]*([0-9]+|[A-Za-z]+)[ \t]*,?/)) {
            tok = substr(reason, RSTART, RLENGTH)
            gsub(/[ \t,]/, "", tok)
            wv = num(tok)
            if (wv < 0) break
            WV[wv] = 1
            nscope++
            reason = substr(reason, RSTART + RLENGTH)
        }
        if (nscope == 0) unscoped = 1
        else sub(/^[ \t]*(\xe2\x80\x94|--|-|:)[ \t]*/, "", reason)
    }
    n = split(scan, T, /[ \t]+/)
    ft = 0
    for (q = 1; q <= n; q++) if (T[q] != "") { ft = q; break }
    for (i = 1; i <= n; i++) {
        v = num(T[i])
        if (v < 0) continue
        if (T[i] ~ /:$/) continue                          # a label: "Tier 1: ...", "Bug 4: ..."
        if (i == ft && T[i] ~ /^[0-9]+[.)]$/) continue     # an ordered-list marker: "2. **A soundness proof**"
        if (i > 1 && low(T[i-1]) ~ /^(lean|version|v|figure|fig|table|section|chapter|appendix|lemma|theorem)$/) continue
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
            } else if (t !~ /^(tests?|doctests?|theorems?|proofs?|compile-fail)$/) continue
            if ((FILENAME SUBSEP FNR SUBSEP v SUBSEP canon(t)) in V) break
            phrase = ""
            for (k = i; k <= j; k++) phrase = phrase (k > i ? " " : "") T[k]
            if (waived && unscoped) printf "WAIVER-UNSCOPED: %s:%d — \"%s\" — waiver names no count; write \"<!-- unguarded: %s — <reason> -->\"\n", FILENAME, FNR, phrase, clean(T[i])
            else if (waived && (v in WV)) printf "WAIVED: %s:%d — \"%s\" — %s\n", FILENAME, FNR, phrase, reason
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

# An unscoped waiver is a failure, not a pass. It is the shape that rots: it
# covers every count on its line, including ones written long after the reason.
UNSCOPED_COUNT=$(grep -c "^WAIVER-UNSCOPED:" "$STALE_FILE" 2>/dev/null || true)
UNSCOPED_COUNT=$(echo "${UNSCOPED_COUNT:-0}" | tr -d '[:space:]')

if [ "$COUNT" -gt 0 ] || [ "$UNGUARDED_COUNT" -gt 0 ] || [ "$UNSCOPED_COUNT" -gt 0 ]; then
    echo ""
    [ "$COUNT" -gt 0 ] && echo "FAIL: ${COUNT} stale doc count(s). Update docs before pushing."
    [ "$UNGUARDED_COUNT" -gt 0 ] && echo "FAIL: ${UNGUARDED_COUNT} unguarded count(s). Guard the phrasing or waive with '<!-- unguarded: <count> — <reason> -->'."
    [ "$UNSCOPED_COUNT" -gt 0 ] && echo "FAIL: ${UNSCOPED_COUNT} waiver(s) name no count. A waiver must say WHICH number it covers: '<!-- unguarded: 407 — <reason> -->'."
    exit 1
else
    echo "OK: all doc counts match reality."
fi
