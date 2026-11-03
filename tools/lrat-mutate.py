#!/usr/bin/env python3
"""lrat-mutate.py — LRAT-level negative controls for the proof checkers.

The certification campaign's negatives were DRAT corruptions; `drat-trim -L`
then either repaired them (33 of 66 became valid LRATs) or emitted a proof
without the empty clause (the other 33) — so the archive's rejections only ever
exercised NoEmptyClause, never the hint logic (cross-vendor panel, 2026-11-02).
This script corrupts GENUINE LRAT files at the hint level, so that every
checker's per-lemma reasoning is what decides:

  hintdrop   remove one hint from a lemma in the middle of the proof
  litflip    negate one literal of a lemma in the middle of the proof
  fwdhint    replace a hint by the id of the proof's LAST lemma (a forward hint)
  useafterdel  delete a clause right before a lemma whose hints use it

A mutant is not guaranteed invalid (a dropped hint may be redundant); this is
an AGREEMENT corpus, not a rejection corpus — every checker must give the same
verdict on each file, and the verdict distribution is reported.

usage: lrat-mutate.py --archive DIR[,DIR..] --out DIR [--lrat-check BIN] [--cake-lpr BIN] [--ours BIN]
  DIR layout: DIR/cnf/<stem>.cnf, DIR/proof/<stem>.lrat (genuine = stem without '.').
  Output: OUT/cnf/<stem>.cnf (copied), OUT/proof/<stem>.<kind>.lrat, OUT/results.csv.
"""
import argparse, csv, os, random, shutil, subprocess, sys

def parse(text):
    steps = []
    for ln in text.split("\n"):
        t = ln.split()
        if not t:
            continue
        if len(t) >= 2 and t[1] == "d":
            steps.append(("d", int(t[0]), [int(x) for x in t[2:-1]]))
        else:
            z = t.index("0")
            steps.append(("a", int(t[0]), [int(x) for x in t[1:z]], [int(x) for x in t[z + 1:-1]]))
    return steps

def fmt(steps):
    out = []
    for s in steps:
        if s[0] == "d":
            out.append(f"{s[1]} d {' '.join(map(str, s[2]))} 0")
        else:
            out.append(f"{s[1]} {' '.join(map(str, s[2]))} 0 {' '.join(map(str, s[3]))} 0")
    return "\n".join(out) + "\n"

def mutate(steps, kind, rng):
    steps = [list(s) for s in steps]
    adds = [i for i, s in enumerate(steps) if s[0] == "a" and s[2] and s[3]]
    if not adds:
        return None
    i = adds[len(adds) // 2]
    if kind == "hintdrop":
        if len(steps[i][3]) < 2:
            return None
        steps[i][3] = steps[i][3][:]
        del steps[i][3][rng.randrange(len(steps[i][3]))]
    elif kind == "litflip":
        steps[i][2] = steps[i][2][:]
        k = rng.randrange(len(steps[i][2]))
        steps[i][2][k] = -steps[i][2][k]
    elif kind == "fwdhint":
        last = max(s[1] for s in steps if s[0] == "a")
        if last <= steps[i][1]:
            return None
        steps[i][3] = steps[i][3][:]
        steps[i][3][rng.randrange(len(steps[i][3]))] = last
    elif kind == "useafterdel":
        # pick a hint h of lemma i that is not deleted before i, and delete it just before i
        deleted = set()
        for s in steps[:i]:
            if s[0] == "d":
                deleted.update(s[2])
        cands = [h for h in steps[i][3] if h > 0 and h not in deleted]
        if not cands:
            return None
        h = rng.choice(cands)
        steps.insert(i, ["d", steps[i][1], [h]])
    else:
        raise ValueError(kind)
    return steps

def run(cmd, timeout=600):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lrat-check", default=None)
    ap.add_argument("--cake-lpr", default=None)
    ap.add_argument("--ours", default=None, help="examples/lrat_check binary")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "cnf"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "proof"), exist_ok=True)
    rows = []
    for d in a.archive.split(","):
        for f in sorted(os.listdir(os.path.join(d, "proof"))):
            stem = f[:-5]
            if not f.endswith(".lrat") or "." in stem:
                continue
            cnf = os.path.join(d, "cnf", stem + ".cnf")
            if not os.path.exists(cnf):
                continue
            shutil.copy(cnf, os.path.join(a.out, "cnf", stem + ".cnf"))
            steps = parse(open(os.path.join(d, "proof", f)).read())
            for kind in ("hintdrop", "litflip", "fwdhint", "useafterdel"):
                m = mutate(steps, kind, rng)
                if m is None:
                    continue
                name = f"{stem}.{kind}"
                lp = os.path.join(a.out, "proof", name + ".lrat")
                open(lp, "w").write(fmt(m))
                row = {"name": name, "kind": kind, "lrat_check": "", "cake_lpr": "", "ours": ""}
                if a.lrat_check:
                    rc, out = run([a.lrat_check, cnf, lp])
                    row["lrat_check"] = "VERIFIED" if "c VERIFIED" in out and rc == 0 else "NOT"
                if a.cake_lpr:
                    rc, out = run([a.cake_lpr, cnf, lp])
                    row["cake_lpr"] = "VERIFIED" if "s VERIFIED" in out else "NOT"
                if a.ours:
                    rc, out = run([a.ours, cnf, lp, "1"])
                    v = [l for l in out.split("\n") if l.startswith("c sequential:")]
                    row["ours"] = (v[0].split(":", 1)[1].strip() if v else f"ERR rc={rc}")[:80]
                rows.append(row)
                print(f"{name}: lrat-check={row['lrat_check']} cake_lpr={row['cake_lpr']} ours={row['ours']}", flush=True)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "kind", "lrat_check", "cake_lpr", "ours"])
        w.writeheader(); w.writerows(rows)
    # summary
    from collections import Counter
    c = Counter()
    dis = 0
    for r in rows:
        ours_v = "VERIFIED" if r["ours"].startswith("Verified") else "NOT"
        c[(r["kind"], r["lrat_check"], ours_v)] += 1
        if r["lrat_check"] and r["lrat_check"] != ours_v:
            dis += 1
    print("\nkind, lrat-check, ours: count")
    for k, n in sorted(c.items()):
        print(f"  {k}: {n}")
    print(f"mutants: {len(rows)}  disagreements(lrat-check vs ours): {dis}")

if __name__ == "__main__":
    main()
