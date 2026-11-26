#!/usr/bin/env bash
# Guard: every numbered section file must be identical to its span of paper/paper.md,
# the spans must make up ALL of paper.md (sections 1..10, nothing before, nothing else),
# and every block marked `<!-- quote:paper -->` … `<!-- /quote -->` in a paper/*.md file
# that is not a section file must appear verbatim in paper.md.
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
# Until 2026-11-25 text before section 1, and any section numbered past 10, belonged
# to no file and was skipped, with "9 files match" printed all the same.
nums = []
for s in secs:
    m = re.match(r'# (\d+)\.', s)
    if not m:
        if s.strip():
            bad.append("paper.md has %d bytes before '# 1.' that belong to no section file" % len(s))
        continue
    nums.append(int(m.group(1)))
if nums != list(range(1, 11)):
    bad.append("paper.md's top-level sections are numbered %s, expected 1..10" % nums)
checked = 0
for s in secs:
    m = re.match(r'# (\d+)\.', s)
    if not m: continue
    n = int(m.group(1))
    body = re.sub(r'\n---\s*$', '', s.rstrip()) + '\n'
    if n in mapping:
        checked += 1
        if open('paper/'+mapping[n]).read() != body:
            bad.append(mapping[n])
    elif n in (9,10):
        fac_parts.append(body)
checked += 1
if open('paper/future-and-conclusion.md').read() != '\n---\n\n'.join(fac_parts):
    bad.append('future-and-conclusion.md')
if checked != 9 and not bad:
    bad.append("only %d of 9 section files were compared" % checked)
# Quotes of the paper in the hand-kept files (appendix, README): verbatim or fail.
import glob, os
section_files = set(mapping.values()) | {'future-and-conclusion.md', 'paper.md'}
norm = lambda t: ' '.join(t.split())
pn = norm(p)
quotes = 0
for f in sorted(glob.glob('paper/*.md')):
    if os.path.basename(f) in section_files: continue
    t = open(f).read()
    opens = t.count('<!-- quote:paper -->'); blocks = re.findall(r'(?s)<!-- quote:paper -->\n(.*?)<!-- /quote -->', t)
    if opens != len(blocks):
        bad.append("%s: %d quote markers opened, %d closed" % (f, opens, len(blocks)))
    for b in blocks:
        quotes += 1
        q = norm('\n'.join(re.sub(r'^> ?', '', l) for l in b.split('\n')))
        if not q or q not in pn:
            bad.append("%s: a block marked as a quote of the paper is not in paper.md: %r…" % (f, q[:60]))
if bad:
    print("PAPER SYNC DRIFT:", "; ".join(bad))
    print("Fix: edit paper/paper.md (canonical) and regenerate, or vice versa — never one side.")
    sys.exit(1)
print("paper sync OK (9 section files match paper.md, which they cover entirely; %d marked quote(s) verbatim)" % quotes)
PY
exit $fail
