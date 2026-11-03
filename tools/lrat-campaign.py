#!/usr/bin/env python3
"""lrat-campaign.py — the solver's own LRAT against the DRAT + drat-trim route.

For every UNSAT CNF under the given archive dirs (DIR/cnf/*.cnf; the proof dir
is ignored), run:

  A  solve --lrat P.lrat                 time, bytes, additions/hints
     lrat-check / cake_lpr / ours        verdicts on P.lrat
  B  solve --drat P.drat                 time
     drat-trim P.cnf P.drat -L P.tl.lrat time, trimmed-LRAT bytes
     lrat-check on P.tl.lrat             the route the certification arc used
  C  solve (no proof)                    the baseline for the overhead ratio

Writes OUT/results.csv and prints the pre-registered quantities
(research/lrat-from-solver.md §4): P1 verdicts, P2 overhead ratio
(lrat - none) / (drat - none), P3 bytes ratio ours / drat-trim's LRAT,
P4 end-to-end time ratio (B total) / (A solve + lrat-check).

usage: lrat-campaign.py --archive DIR[,DIR..] --out DIR --solve BIN --drat-trim BIN
                        --lrat-check BIN [--cake-lpr BIN] [--ours BIN] [--timeout S]
"""
import argparse, csv, os, subprocess, sys, time

def run(cmd, timeout):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return time.perf_counter() - t0, p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return timeout, 124, "TIMEOUT"

def verdict_line(out, prefix="s "):
    for l in out.split("\n"):
        if l.startswith(prefix):
            return l.strip()
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solve", required=True)
    ap.add_argument("--drat-trim", required=True)
    ap.add_argument("--lrat-check", required=True)
    ap.add_argument("--cake-lpr")
    ap.add_argument("--ours", help="examples/lrat_check binary")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--only", help="substring filter on instance names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for d in a.archive.split(","):
        cdir = os.path.join(d, "cnf")
        for f in sorted(os.listdir(cdir)):
            if not f.endswith(".cnf"):
                continue
            stem = f[:-4]
            if a.only and a.only not in stem:
                continue
            cnf = os.path.join(cdir, f)
            tag = f"{os.path.basename(os.path.normpath(d))}_{stem}"
            base = os.path.join(a.out, tag)
            row = {"name": tag, "vars": "", "clauses": ""}
            # C: baseline
            t_none, rc, out = run([a.solve, cnf], a.timeout)
            v = verdict_line(out)
            row["verdict"] = v
            row["t_none"] = f"{t_none:.3f}"
            if "UNSATISFIABLE" not in v:
                print(f"{tag}: {v or 'no verdict'} — skipped", flush=True)
                rows.append(row); continue
            # A: solver LRAT
            lrat = base + ".lrat"
            t_lrat, rc, out = run([a.solve, "--lrat", lrat, cnf], a.timeout)
            row["t_lrat"] = f"{t_lrat:.3f}"
            row["lrat_bytes"] = os.path.getsize(lrat) if os.path.exists(lrat) else ""
            stat = [l for l in out.split("\n") if "LRAT proof written" in l]
            row["lrat_stats"] = stat[0].split("(", 1)[1].rstrip(")") if stat else ""
            t_lc, rc, out = run([a.lrat_check, cnf, lrat], a.timeout)
            row["lrat_check"] = "VERIFIED" if rc == 0 and "c VERIFIED" in out else "NOT"
            row["t_lrat_check"] = f"{t_lc:.3f}"
            if a.cake_lpr:
                t_ck, rc, out = run([a.cake_lpr, cnf, lrat], a.timeout)
                row["cake_lpr"] = "VERIFIED" if "s VERIFIED" in out else "NOT"
                row["t_cake_lpr"] = f"{t_ck:.3f}"
            if a.ours:
                t_o, rc, out = run([a.ours, cnf, lrat, "4"], a.timeout)
                row["ours"] = verdict_line(out)
                row["t_ours"] = f"{t_o:.3f}"
            # B: DRAT route
            drat = base + ".drat"
            t_drat, rc, out = run([a.solve, "--drat", drat, cnf], a.timeout)
            row["t_drat"] = f"{t_drat:.3f}"
            row["drat_bytes"] = os.path.getsize(drat) if os.path.exists(drat) else ""
            tl = base + ".tl.lrat"
            t_trim, rc, out = run([a.drat_trim, cnf, drat, "-L", tl], a.timeout)
            row["t_drat_trim"] = f"{t_trim:.3f}"
            row["drat_trim"] = "VERIFIED" if "s VERIFIED" in out else "NOT"
            row["tl_bytes"] = os.path.getsize(tl) if os.path.exists(tl) else ""
            if os.path.exists(tl):
                t_lc2, rc, out = run([a.lrat_check, cnf, tl], a.timeout)
                row["t_lrat_check_tl"] = f"{t_lc2:.3f}"
            else:
                row["t_lrat_check_tl"] = ""
            # derived
            try:
                row["p2_overhead_ratio"] = f"{(t_lrat - t_none) / max(t_drat - t_none, 1e-6):.2f}"
                row["p3_bytes_ratio"] = f"{row['lrat_bytes'] / max(row['tl_bytes'], 1):.2f}" if row["tl_bytes"] else ""
                a_total = t_lrat + t_lc
                b_total = t_drat + t_trim + (float(row["t_lrat_check_tl"]) if row["t_lrat_check_tl"] else 0.0)
                row["p4_speedup"] = f"{b_total / max(a_total, 1e-6):.2f}"
            except Exception:
                pass
            print(f"{tag}: lrat-check={row['lrat_check']} cake_lpr={row.get('cake_lpr','')} ours={row.get('ours','')} "
                  f"solve {t_none:.2f}s lrat {t_lrat:.2f}s drat {t_drat:.2f}s trim {t_trim:.2f}s "
                  f"bytes {row['lrat_bytes']}/{row['tl_bytes']} P4 {row.get('p4_speedup','')}", flush=True)
            for p in (drat, tl):
                try: os.remove(p)
                except OSError: pass
            rows.append(row)
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    done = [r for r in rows if r.get("lrat_check")]
    print(f"\ninstances: {len(done)}  lrat-check VERIFIED: {sum(r['lrat_check']=='VERIFIED' for r in done)}"
          f"  cake_lpr VERIFIED: {sum(r.get('cake_lpr')=='VERIFIED' for r in done)}")
    import statistics as st
    for k in ("p2_overhead_ratio", "p3_bytes_ratio", "p4_speedup"):
        xs = [float(r[k]) for r in done if r.get(k)]
        if xs:
            print(f"{k}: median {st.median(xs):.2f}  min {min(xs):.2f}  max {max(xs):.2f}")

if __name__ == "__main__":
    main()
