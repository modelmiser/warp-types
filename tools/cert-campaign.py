#!/usr/bin/env python3
"""Certification campaign for warp-types-sat's UNSAT answers (research/drat-certification.md).

Independent of the crate on purpose: it drives the `solve` BINARY on FILES, checks SAT models against the
original file bytes with its own DIMACS reader, and sends every UNSAT proof through external checkers:

    drat-trim (default)      -> s VERIFIED
    drat-trim -v -W          -> exits non-zero on the first ignored deletion (unit-deletion independence)
    drat-trim -f             -> forward mode, a second DRAT semantics
    drat-trim -L -> lrat-check (and cake_lpr when given)

kissat is run on every instance as a DEBUGGER: a disagreement is recorded and adjudicated by certificate /
model, never by majority. Negative controls corrupt genuine proofs and record which checkers notice.

Kill conditions (pre-registered): any genuine proof rejected by any checker, or any SAT model failing the
file-bytes check -> write the summary and exit 2. Coverage (UNSAT proofs with >= 1 `d` line) below 50 % ->
exit 3 (campaign invalid). Everything else -> exit 0.

    python3 tools/cert-campaign.py --out ~/staging/<dir> --solve target/release/solve \
        --drat-trim PATH --lrat-check PATH [--cake-lpr PATH] [--kissat PATH] [--cnfuzz PATH] \
        [--reduce-interval 50] [--seeds 8]
"""
import argparse, csv, hashlib, os, random, re, statistics, subprocess, sys, time
from pathlib import Path

# ----------------------------------------------------------------------------- generators (seeded)
def w(path, nv, clauses):
    with open(path, "w") as f:
        f.write(f"p cnf {nv} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")

def gen_r3sat(nv, ratio, seed, path):
    r = random.Random(seed); cl = []
    for _ in range(int(nv * ratio)):
        vs = r.sample(range(1, nv + 1), 3)
        cl.append([v if r.random() < 0.5 else -v for v in vs])
    w(path, nv, cl)

def gen_planted(nv, ratio, seed, path):
    r = random.Random(seed); sol = [r.random() < 0.5 for _ in range(nv)]; cl = []
    while len(cl) < int(nv * ratio):
        vs = r.sample(range(1, nv + 1), 3)
        c = [v if r.random() < 0.5 else -v for v in vs]
        if any((l > 0) == sol[abs(l) - 1] for l in c):
            cl.append(c)
    w(path, nv, cl)

def gen_php(n, path):
    v = lambda p, h: p * n + h + 1; cl = []
    for p in range(n + 1): cl.append([v(p, h) for h in range(n)])
    for h in range(n):
        for p in range(n + 1):
            for q in range(p + 1, n + 1): cl.append([-v(p, h), -v(q, h)])
    w(path, (n + 1) * n, cl)

def gen_parity(k, path):
    # x1 xor x2 = 1, x2 xor x3 = 1, ... (SAT: alternate)
    cl = []
    for i in range(1, k + 1):
        cl += [[i, i + 1], [-i, -(i + 1)]]
    w(path, k + 1, cl)

def xor_clauses(vars_, parity):
    """CNF for XOR(vars_) == parity (direct encoding, vars_ small)."""
    out = []; n = len(vars_)
    for mask in range(1 << n):
        bits = [(mask >> i) & 1 for i in range(n)]
        if sum(bits) % 2 != parity:  # forbid assignments with the wrong parity
            out.append([-(v) if b else v for v, b in zip(vars_, bits)])
    return out

def gen_tseitin(nv_graph, seed, path):
    """Tseitin on a random 3-regular graph with one odd-charged vertex: UNSAT."""
    r = random.Random(seed)
    while True:  # pairing model; retry until simple
        stubs = [v for v in range(nv_graph) for _ in range(3)]; r.shuffle(stubs)
        edges = [tuple(sorted((stubs[i], stubs[i + 1]))) for i in range(0, len(stubs), 2)]
        if all(a != b for a, b in edges) and len(set(edges)) == len(edges): break
    eid = {e: i + 1 for i, e in enumerate(edges)}
    inc = {v: [] for v in range(nv_graph)}
    for e in edges:
        inc[e[0]].append(eid[e]); inc[e[1]].append(eid[e])
    cl = []
    for v in range(nv_graph):
        cl += xor_clauses(inc[v], 1 if v == 0 else 0)
    w(path, len(edges), cl)

# ----------------------------------------------------------------------------- helpers
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

def run(cmd, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr, time.time() - t
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT", time.time() - t

def verdict(out):
    """drat-trim: `s VERIFIED` / `s NOT VERIFIED`; lrat-check: `c VERIFIED` or `c FAILED`/`c ERROR` and
    exit 0 either way (so the exit code is useless); cake_lpr: `s VERIFIED UNSAT` or an error message."""
    out = out.replace("\r", "\n")
    if "s NOT VERIFIED" in out: return "NOT VERIFIED"
    if "s VERIFIED" in out or "\nc VERIFIED" in out or out.startswith("c VERIFIED"): return "VERIFIED"
    if "c FAILED" in out or "c ERROR" in out: return "NOT VERIFIED"
    if "s UNSATISFIABLE" in out: return "UNSAT"
    if "s SATISFIABLE" in out: return "SAT"
    if "s UNKNOWN" in out: return "UNKNOWN"
    if out == "TIMEOUT": return "TIMEOUT"
    return "NONE"

def read_cnf_bytes(path):
    """Own DIMACS reader over the raw bytes: header, clauses, stops at '%'."""
    clauses, cur, nv = [], [], 0
    for line in Path(path).read_bytes().decode().split("\n"):
        t = line.strip()
        if not t or t[0] == "c": continue
        if t[0] == "%": break
        if t.startswith("p"):
            nv = int(t.split()[2]); continue
        for tok in t.split():
            v = int(tok)
            if v == 0: clauses.append(cur); cur = []
            else: cur.append(v)
    if cur: clauses.append(cur)
    return nv, clauses

def model_ok(cnf_path, solver_out):
    lits = set()
    for line in solver_out.split("\n"):
        if line.startswith("v"):
            lits.update(int(x) for x in line[1:].split() if x != "0")
    nv, clauses = read_cnf_bytes(cnf_path)
    if not lits: return False, "no v-line"
    bad = [c for c in clauses if not any(l in lits for l in c)]
    return (not bad), f"{len(clauses) - len(bad)}/{len(clauses)} satisfied"

def decode_bdrat(data):
    """Binary DRAT (drat-trim's layout) -> the equivalent text, one step per line. Independent of the
    crate's parser on purpose: a second reading of the same bytes."""
    out, pos, n = [], 0, len(data)
    while pos < n:
        pre = data[pos]; pos += 1
        if pre not in (97, 100): raise ValueError(f"byte {pos-1}: prefix {pre}")
        lits = []
        while True:
            code, shift = 0, 0
            while True:
                if pos >= n: raise ValueError(f"byte {pos}: truncated step")
                b = data[pos]; pos += 1
                code |= (b & 0x7f) << shift; shift += 7
                if not b & 0x80: break
            if code == 0: break
            if code == 1: raise ValueError(f"byte {pos}: literal code 1")
            lits.append(-(code >> 1) if code & 1 else code >> 1)
        out.append(("d " if pre == 100 else "") + " ".join(map(str, lits + [0])))
    return "\n".join(out) + "\n"

def corrupt(proof_text, kind):
    lines = [l for l in proof_text.split("\n") if l.strip()]
    if kind == "flip":  # flip the first literal of a lemma one third of the way in (the last lemmas
        # are often RUP either way, the formula being contradictory by then)
        adds = [i for i, l in enumerate(lines) if not l.startswith("d") and l.strip() != "0"]
        if adds:
            i = adds[len(adds) // 3]
            toks = lines[i].split(); toks[0] = str(-int(toks[0])); lines[i] = " ".join(toks)
    elif kind == "drop0":
        lines = [l for l in lines if l.strip() != "0"]
    elif kind == "delorig":  # pretend the first original clause was deleted before anything
        lines = ["d __ORIG__"] + lines
    return "\n".join(lines) + "\n"

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--solve", required=True)
    ap.add_argument("--drat-trim", required=True); ap.add_argument("--lrat-check", required=True)
    ap.add_argument("--cake-lpr"); ap.add_argument("--kissat"); ap.add_argument("--cnfuzz")
    ap.add_argument("--reduce-interval", type=int, default=50); ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--solve-timeout", type=int, default=120); ap.add_argument("--check-timeout", type=int, default=300)
    ap.add_argument("--negatives", type=int, default=6); ap.add_argument("--timing-reps", type=int, default=5)
    ap.add_argument("--timing-only", action="store_true", help="skip the corpus; reuse CNFs already in --out")
    ap.add_argument("--drat-binary", action="store_true", help="solve with --drat-binary; stats and negatives via decode_bdrat")
    a = ap.parse_args()
    out = Path(a.out); (out / "cnf").mkdir(parents=True, exist_ok=True); (out / "proof").mkdir(exist_ok=True)

    # corpus
    inst = []
    for n in (150, 200, 250):
        for s in range(a.seeds): inst.append((f"r3sat_n{n}_s{s}", lambda p, n=n, s=s: gen_r3sat(n, 4.26, 1000 + s, p)))
    for n in (6, 7, 8): inst.append((f"php{n}", lambda p, n=n: gen_php(n, p)))
    for k in (20, 40, 80): inst.append((f"parity{k}", lambda p, k=k: gen_parity(k, p)))
    for n in (20, 30, 40):
        for s in range(max(2, a.seeds // 2)): inst.append((f"tseitin_v{n}_s{s}", lambda p, n=n, s=s: gen_tseitin(n, 2000 + s, p)))
    for n in (200, 400):
        for s in range(max(2, a.seeds // 2)): inst.append((f"planted_n{n}_s{s}", lambda p, n=n, s=s: gen_planted(n, 4.0, 3000 + s, p)))
    if a.cnfuzz:
        for s in range(a.seeds * 4):
            def g(p, s=s):
                r = subprocess.run([a.cnfuzz, str(4000 + s)], capture_output=True, text=True, timeout=60)
                Path(p).write_text(r.stdout)
            inst.append((f"cnfuzz_s{s}", g))

    rows, negs, kills = [], [], []
    cols = ["name", "cnf_sha", "vars", "clauses", "solve", "solve_s", "kissat", "agree", "model",
            "proof_sha", "proof_lines", "d_lines", "unit_d_lines", "dt_default", "dt_s", "dt_vW_exit", "dt_vW_ignoring",
            "dt_forward", "lrat_lines", "lrat_check", "cake_lpr"]
    for name, gen in ([] if a.timing_only else inst):
        cnf = out / "cnf" / f"{name}.cnf"; gen(cnf)
        nv, clauses = read_cnf_bytes(cnf)
        row = dict.fromkeys(cols, ""); row.update(name=name, cnf_sha=sha(cnf), vars=nv, clauses=len(clauses))
        proof = out / "proof" / f"{name}.drat"
        rc, so, dt = run([a.solve, "--drat", str(proof)] + (["--drat-binary"] if a.drat_binary else []) + ["--reduce-interval", str(a.reduce_interval), str(cnf)], a.solve_timeout)
        row["solve"] = verdict(so); row["solve_s"] = f"{dt:.3f}"
        if a.kissat:
            krc, ko, _ = run([a.kissat, "-q", str(cnf)], a.solve_timeout)
            row["kissat"] = verdict(ko)
            row["agree"] = "" if row["solve"] in ("TIMEOUT", "UNKNOWN") or row["kissat"] in ("TIMEOUT", "UNKNOWN") else str(row["solve"] == row["kissat"])
            if row["agree"] == "False": kills.append(f"{name}: solve={row['solve']} kissat={row['kissat']} (adjudicate by certificate/model)")
        if row["solve"] == "SAT":
            ok, why = model_ok(cnf, so); row["model"] = f"{'OK' if ok else 'FAIL'} {why}"
            if not ok: kills.append(f"{name}: SAT model fails file-bytes check: {why}")
        elif row["solve"] == "UNSAT":
            text = decode_bdrat(proof.read_bytes()) if a.drat_binary else proof.read_text()
            lines = [l for l in text.split("\n") if l.strip()]
            row["proof_sha"] = sha(proof); row["proof_lines"] = len(lines)
            row["d_lines"] = sum(1 for l in lines if l.startswith("d "))
            row["unit_d_lines"] = sum(1 for l in lines if l.startswith("d ") and len(l.split()) == 3)
            rc, o, t = run([a.drat_trim, str(cnf), str(proof)], a.check_timeout); row["dt_default"] = verdict(o); row["dt_s"] = f"{t:.3f}"
            rc2, o2, _ = run([a.drat_trim, str(cnf), str(proof), "-v", "-W"], a.check_timeout)
            row["dt_vW_exit"] = rc2; row["dt_vW_ignoring"] = o2.count("ignoring deletion")
            rc3, o3, _ = run([a.drat_trim, str(cnf), str(proof), "-f"], a.check_timeout); row["dt_forward"] = verdict(o3)
            lrat = out / "proof" / f"{name}.lrat"
            run([a.drat_trim, str(cnf), str(proof), "-L", str(lrat)], a.check_timeout)
            if lrat.exists():
                row["lrat_lines"] = sum(1 for _ in open(lrat))
                rc4, o4, _ = run([a.lrat_check, str(cnf), str(lrat)], a.check_timeout); row["lrat_check"] = verdict(o4)
                if a.cake_lpr:
                    rc5, o5, _ = run([a.cake_lpr, str(cnf), str(lrat)], a.check_timeout); row["cake_lpr"] = verdict(o5)
            for k in ("dt_default", "dt_forward", "lrat_check", "cake_lpr"):
                if row[k] and row[k] != "VERIFIED": kills.append(f"{name}: {k} = {row[k]} on a genuine proof")
            if row["unit_d_lines"] or row["dt_vW_ignoring"]: kills.append(f"{name}: unit deletions present/ignored ({row['unit_d_lines']}, {row['dt_vW_ignoring']})")
            # negative controls on the first few UNSAT proofs
            if len(negs) < a.negatives * 3:
                first = " ".join(map(str, clauses[0]))
                for kind in ("flip", "drop0", "delorig"):
                    bad = out / "proof" / f"{name}.{kind}.drat"  # text, decoded from the binary proof when --drat-binary
                    bad.write_text(corrupt(text, kind).replace("__ORIG__", first + " 0"))
                    n1 = verdict(run([a.drat_trim, str(cnf), str(bad)], a.check_timeout)[1])
                    n2 = verdict(run([a.drat_trim, str(cnf), str(bad), "-f"], a.check_timeout)[1])
                    bl = out / "proof" / f"{name}.{kind}.lrat"; run([a.drat_trim, str(cnf), str(bad), "-L", str(bl)], a.check_timeout)
                    n3 = verdict(run([a.lrat_check, str(cnf), str(bl)], a.check_timeout)[1]) if bl.exists() else "NO-LRAT"
                    n4 = verdict(run([a.cake_lpr, str(cnf), str(bl)], a.check_timeout)[1]) if (a.cake_lpr and bl.exists()) else ""
                    negs.append(dict(name=name, kind=kind, dt_default=n1, dt_forward=n2, lrat_check=n3, cake_lpr=n4))
        rows.append(row)
        print(f"{name:22s} {row['solve']:8s} kissat={row['kissat']:8s} d={row['d_lines']!s:5s} dt={row['dt_default']:13s} fwd={row['dt_forward']:13s} lrat={row['lrat_check']:13s} cake={row['cake_lpr']}", flush=True)
        if kills:
            print("KILL:", kills[-1], flush=True); break

    # timing: no proof vs /dev/null vs file, on instances that delete
    timing = []
    for name in ("php7", "php8", f"r3sat_n200_s0", f"r3sat_n250_s0"):
        cnf = out / "cnf" / f"{name}.cnf"
        if not cnf.exists(): continue
        for mode, extra in (("none", ["--reduce-interval", str(a.reduce_interval)]), ("devnull", ["--drat", "/dev/null", "--reduce-interval", str(a.reduce_interval)]),
                            ("file", ["--drat", str(out / "proof" / f"{name}.timing.drat"), "--reduce-interval", str(a.reduce_interval)]),
                            ("bin-devnull", ["--drat", "/dev/null", "--drat-binary", "--reduce-interval", str(a.reduce_interval)]),
                            ("bin-file", ["--drat", str(out / "proof" / f"{name}.timing.bdrat"), "--drat-binary", "--reduce-interval", str(a.reduce_interval)])):
            ts = [run([a.solve] + extra + [str(cnf)], a.solve_timeout)[2] for _ in range(a.timing_reps)]
            timing.append(dict(name=name, mode=mode, median_s=f"{statistics.median(ts):.4f}", min_s=f"{min(ts):.4f}", reps=a.timing_reps))

    # write
    if a.timing_only:
        with open(out / "timing.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["name", "mode", "median_s", "min_s", "reps"]); wr.writeheader(); wr.writerows(timing)
        print(open(out / "timing.csv").read()); return
    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
    with open(out / "negatives.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["name", "kind", "dt_default", "dt_forward", "lrat_check", "cake_lpr"]); wr.writeheader(); wr.writerows(negs)
    with open(out / "timing.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["name", "mode", "median_s", "min_s", "reps"]); wr.writeheader(); wr.writerows(timing)
    unsat = [r for r in rows if r["solve"] == "UNSAT"]; sat = [r for r in rows if r["solve"] == "SAT"]
    cov = sum(1 for r in unsat if int(r["d_lines"] or 0) > 0) / len(unsat) if unsat else 0.0
    summ = [f"# cert-campaign summary", f"instances {len(rows)}: UNSAT {len(unsat)}, SAT {len(sat)}, other {len(rows)-len(unsat)-len(sat)}",
            f"reduce_interval {a.reduce_interval}; d-line coverage {cov:.1%} ({sum(1 for r in unsat if int(r['d_lines'] or 0) > 0)}/{len(unsat)})",
            ("proofs: BINARY DRAT (stats/negatives via decode_bdrat)" if a.drat_binary else "proofs: text DRAT"),
            f"drat-trim default VERIFIED {sum(1 for r in unsat if r['dt_default']=='VERIFIED')}/{len(unsat)}; forward {sum(1 for r in unsat if r['dt_forward']=='VERIFIED')}/{len(unsat)}; lrat-check {sum(1 for r in unsat if r['lrat_check']=='VERIFIED')}/{len(unsat)}; cake_lpr {sum(1 for r in unsat if r['cake_lpr']=='VERIFIED')}/{sum(1 for r in unsat if r['cake_lpr'])}",
            f"unit d-lines total {sum(int(r['unit_d_lines'] or 0) for r in unsat)}; -v -W ignoring total {sum(int(r['dt_vW_ignoring'] or 0) for r in unsat)}",
            f"SAT models OK {sum(1 for r in sat if r['model'].startswith('OK'))}/{len(sat)}; kissat disagreements {sum(1 for r in rows if r['agree']=='False')}",
            f"negative controls: " + "; ".join(f"{k}: dt {sum(1 for n in negs if n['kind']==k and n['dt_default']=='NOT VERIFIED')}/{sum(1 for n in negs if n['kind']==k)} rejected, lrat-check {sum(1 for n in negs if n['kind']==k and n['lrat_check']!='VERIFIED')}/{sum(1 for n in negs if n['kind']==k)} rejected" for k in ("flip", "drop0", "delorig")),
            f"kills: {kills if kills else 'none'}"]
    (out / "summary.md").write_text("\n".join(summ) + "\n"); print("\n".join(summ))
    if kills: sys.exit(2)
    if cov < 0.5: sys.exit(3)

if __name__ == "__main__":
    main()
