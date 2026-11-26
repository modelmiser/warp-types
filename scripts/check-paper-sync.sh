#!/usr/bin/env bash
# Guard, three parts:
#  1. every numbered section file is identical to its span of paper/paper.md;
#  2. those spans make up all of paper.md: its top-level sections are numbered exactly 1..10 and nothing
#     but whitespace precedes section 1 (compared up to trailing whitespace and the `---` separator that
#     ends a section, as part 1 always was);
#  3. quotes of the paper in the hand-kept paper files. A block
#         <!-- quote:paper §7.5 -->
#         > text
#         <!-- /quote -->
#     must be, whitespace aside, one WHOLE prose paragraph of paper.md (not a list, table, code or rule,
#     and at least 200 characters) lying under the numbered heading named in the marker — not a fragment
#     of one (a quote cut off before its qualifications is the drift this is for). A marker must be alone
#     on its line; anything that looks like one but is not exactly one is an error; and the files in EXPECT
#     must quote exactly the listed sections, so deleting or misspelling a marker fails too. What part 3
#     cannot see: prose in those files that paraphrases the paper without a marker, and WHICH paragraph of
#     a section is quoted (any whole paragraph of §1.2 satisfies "§1.2").
# The fork part 1 prevents shipped contradictory Cooperative-Groups/ISPC verdicts and placeholder
# references in the DOI'd assembled paper (found at the 2026-08-13 gate). Parts 2 and 3 date from
# 2026-11-25: text before section 1 or a section 11 passed unnoticed, and paper/empirical-evidence.md
# paraphrased §1.2 without the qualifications the paper had since added.
set -u
cd "$(dirname "$0")/.."
fail=0
python3 - <<'PY' || fail=1
import glob, os, re, sys
p = open('paper/paper.md').read()
# A `# N. ` line inside a code fence would be taken for a section; it then breaks the 1..10 numbering
# below and fails, which is the safe direction.
secs = re.split(r'(?m)^(?=# \d+\. )', p)
mapping = {1:'introduction.md',2:'background.md',3:'core-type-system.md',4:'metatheory.md',5:'extensions.md',6:'implementation.md',7:'evaluation.md',8:'related-work.md'}
EXPECT = {'empirical-evidence.md': ['1.2', '7.5']}   # hand-kept file -> the sections it must quote, in order
bad = []

def read(name):
    try:
        return open('paper/' + name).read()
    except OSError as e:
        bad.append("%s cannot be read (%s)" % (name, e.strerror))
        return None

# ---- parts 1 and 2
nums, fac_parts = [], []
for s in secs:
    m = re.match(r'# (\d+)\.', s)
    if not m:
        if s.strip():
            bad.append("paper.md has %d bytes before '# 1.' that belong to no section file" % len(s))
        continue
    n = int(m.group(1)); nums.append(n)
    body = re.sub(r'\n---\s*$', '', s.rstrip()) + '\n'
    if n in mapping:
        t = read(mapping[n])
        if t is not None and t != body:
            bad.append(mapping[n])
    elif n in (9, 10):
        fac_parts.append(body)
if nums != list(range(1, 11)):
    bad.append("paper.md's top-level sections are numbered %s, expected 1..10" % nums)
t = read('future-and-conclusion.md')
if t is not None and t != '\n---\n\n'.join(fac_parts):
    bad.append('future-and-conclusion.md')

# ---- part 3
norm = lambda x: ' '.join(x.split())
# Prose paragraphs of paper.md, read line by line: runs of non-blank lines outside code fences that are
# not headings, list items, table rows, block quotes or rules. Each is filed under the last NUMBERED
# heading above it; an unnumbered heading at the same or a higher level ends that section (References and
# Acknowledgments are not "§10"), a deeper unnumbered one (`### Limitations` under 6.9) does not.
paras = []          # (normalised paragraph, section number or None)
cur, cur_level, buf, fenced = None, 0, [], False
def flush():
    if buf:
        first = buf[0].lstrip()
        if not re.match(r'([-*+>|]|\d+[.)] |---)', first):
            paras.append((norm(' '.join(buf)), cur))
        del buf[:]
for line in p.split('\n'):
    if line.lstrip().startswith('```'):
        flush(); fenced = not fenced; continue
    if fenced: continue
    h = re.match(r'(#{1,6}) (.*)', line)
    if h:
        flush()
        level = len(h.group(1)); num = re.match(r'(\d+(?:\.\d+)*)[. ]', h.group(2))
        if num: cur, cur_level = num.group(1), level
        elif level <= cur_level: cur, cur_level = None, 0
        continue
    if not line.strip(): flush(); continue
    buf.append(line)
flush()
OPEN = re.compile(r'<!-- quote:paper §(\d+(?:\.\d+)*) -->')
section_files = set(mapping.values()) | {'future-and-conclusion.md', 'paper.md'}
quotes = 0
for f in sorted(glob.glob('paper/*.md')):
    name = os.path.basename(f)
    if name in section_files: continue
    t = read(name)
    if t is None: continue
    # anything that could be meant as a marker, whatever its case or spelling, must be exactly one,
    # alone on its line
    for mk in re.findall(r'(?i)<!--[^>]*(?:quot|qout|:\s*paper|paper\s*§)[^>]*-->', t):
        if not (OPEN.fullmatch(mk) or mk == '<!-- /quote -->'):
            bad.append("%s: %r looks like a quote marker but is not one" % (f, mk))
    blocks = re.findall(r'(?ms)^<!-- quote:paper §(\d+(?:\.\d+)*) -->\n(.*?)\n<!-- /quote -->$', t)
    n_open = len(OPEN.findall(t)); n_close = t.count('<!-- /quote -->')
    if not (n_open == n_close == len(blocks)):
        bad.append("%s: %d quote markers opened, %d closed, %d well-formed blocks (a marker must be alone on its line)" % (f, n_open, n_close, len(blocks)))
    if name in EXPECT and [sec for sec, _ in blocks] != EXPECT[name]:
        bad.append("%s must quote §%s, in that order; it quotes %s" % (f, ', §'.join(EXPECT[name]), [sec for sec, _ in blocks]))
    for sec, b in blocks:
        quotes += 1
        q = norm('\n'.join(re.sub(r'^> ?', '', l) for l in b.split('\n')))
        hits = [s for (pp, s) in paras if pp == q]
        if len(q) < 200 or not hits:
            bad.append("%s: the block quoting §%s is not a whole prose paragraph of paper.md: %r…" % (f, sec, q[:60]))
        elif sec not in hits:
            bad.append("%s: the block says §%s; that paragraph is under §%s in paper.md" % (f, sec, ', §'.join(map(str, hits))))
for name in EXPECT:
    if not os.path.exists('paper/' + name):
        bad.append("%s is missing (it must quote §%s)" % (name, ', §'.join(EXPECT[name])))
if bad:
    print("PAPER SYNC DRIFT:", "; ".join(bad))
    print("Fix: edit paper/paper.md (canonical) and regenerate, or vice versa — never one side.")
    sys.exit(1)
print("paper sync OK (9 section files match paper.md and cover it; %d marked quote(s) are whole paragraphs of the section they name)" % quotes)
PY
exit $fail
