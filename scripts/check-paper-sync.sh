#!/usr/bin/env bash
# Guard: every numbered section file must be identical to its span of paper/paper.md.
# The fork this prevents shipped contradictory Cooperative-Groups/ISPC verdicts and
# placeholder references in the DOI'd assembled paper (found at the 2026-08-13 gate).
set -u
cd "$(dirname "$0")/.."
fail=0
python3 - <<'PY' || fail=1
import re, sys
p = open('paper/paper.md').read()
secs = re.split(r'(?m)^(?=# \d+\. )', p)
mapping = {1:'introduction.md',2:'background.md',3:'core-type-system.md',4:'metatheory.md',5:'extensions.md',6:'implementation.md',7:'evaluation.md',8:'related-work.md'}
fac_parts = []
bad = []
for s in secs:
    m = re.match(r'# (\d+)\.', s)
    if not m: continue
    n = int(m.group(1))
    body = re.sub(r'\n---\s*$', '', s.rstrip()) + '\n'
    if n in mapping:
        if open('paper/'+mapping[n]).read() != body:
            bad.append(mapping[n])
    elif n in (9,10):
        fac_parts.append(body)
if open('paper/future-and-conclusion.md').read() != '\n---\n\n'.join(fac_parts):
    bad.append('future-and-conclusion.md')
if bad:
    print("PAPER SYNC DRIFT (section file != paper.md span):", ", ".join(bad))
    print("Fix: edit paper/paper.md (canonical) and regenerate, or vice versa — never one side.")
    sys.exit(1)
print("paper sync OK (9 files match paper.md)")
PY
exit $fail
