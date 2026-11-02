#!/usr/bin/env python3
"""Run warp-types-bmc and warp-types-pdr on a directory of AIGER instances, compare against an ABC reference
CSV, and certify every PDR invariant with tools/check-invariant.py (own encoding + kissat).

    python3 tools/bmc-pdr-campaign.py --aigs DIR --abc-csv ref/abc.csv --out DIR \
        --bmc target/release/bmc --pdr target/release/pdr --kissat PATH [--timeout 60] [--max-kb 20]

Pre-registered kill conditions (research/bmc-pdr-aiger.md): a counterexample-depth mismatch with ABC bmc3, a
verdict contradiction with ABC, or a PDR invariant the independent check rejects → stop after writing the summary,
exit 2. Otherwise exit 0.
"""
import argparse, csv, subprocess, sys, time
from pathlib import Path

def run(cmd, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr, time.time() - t
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT", time.time() - t

def parse_verdict(out):
    for line in out.split("\n"):
        if line.startswith("s "):
            t = line.split(); kv = dict(x.split("=", 1) for x in t[2:] if "=" in x)
            return t[1], kv
    return "NONE", {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aigs", required=True); ap.add_argument("--abc-csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bmc", required=True); ap.add_argument("--pdr", required=True); ap.add_argument("--kissat", required=True)
    ap.add_argument("--timeout", type=int, default=60); ap.add_argument("--max-kb", type=int, default=20)
    ap.add_argument("--max-depth", type=int, default=1000); ap.add_argument("--max-frames", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out); (out / "inv").mkdir(parents=True, exist_ok=True)
    abc = {r["name"]: r for r in csv.DictReader(open(a.abc_csv))}
    files = sorted(p for p in Path(a.aigs).glob("*.aig") if p.stat().st_size < a.max_kb * 1024)
    cols = ["name", "bytes", "abc_bmc3", "abc_bmc3_depth", "abc_bmc3_s", "abc_pdr", "abc_pdr_depth", "abc_pdr_s",
            "our_bmc", "our_bmc_depth", "our_bmc_s", "our_pdr", "our_pdr_depth_or_frame", "our_pdr_clauses", "our_pdr_s",
            "inv_check", "depth_match", "verdict_consistent"]
    rows, kills = [], []
    for f in files:
        name = f.stem; ref = abc.get(name, {})
        row = dict.fromkeys(cols, ""); row.update(name=name, bytes=f.stat().st_size,
            abc_bmc3=ref.get("bmc3_verdict", ""), abc_bmc3_depth=ref.get("bmc3_depth", ""), abc_bmc3_s=ref.get("bmc3_s", ""),
            abc_pdr=ref.get("pdr_verdict", ""), abc_pdr_depth=ref.get("pdr_depth", ""), abc_pdr_s=ref.get("pdr_s", ""))
        # our BMC
        rc, so, se, t = run([a.bmc, str(f), "--max-depth", str(a.max_depth)], a.timeout)
        v, kv = parse_verdict(so) if rc != -9 else ("TIMEOUT", {})
        row["our_bmc"] = v; row["our_bmc_depth"] = kv.get("depth", ""); row["our_bmc_s"] = f"{t:.2f}"
        # our PDR
        inv = out / "inv" / f"{name}.inv"
        rc, so, se, t = run([a.pdr, str(f), "--max-frames", str(a.max_frames), "--invariant", str(inv)], a.timeout)
        pv, pkv = parse_verdict(so) if rc != -9 else ("TIMEOUT", {})
        row["our_pdr"] = pv; row["our_pdr_depth_or_frame"] = pkv.get("depth", pkv.get("frame", "")); row["our_pdr_clauses"] = pkv.get("clauses", ""); row["our_pdr_s"] = f"{t:.2f}"
        if pv == "UNSAT" and inv.exists():
            rc, so, se, t = run([sys.executable, str(Path(__file__).with_name("check-invariant.py")), "--aig", str(f), "--inv", str(inv), "--kissat", a.kissat], 600)
            row["inv_check"] = so.strip().split("\n")[-1] if so.strip() else f"ERROR {se.strip()[:80]}"
            if not row["inv_check"].startswith("CERTIFIED"):
                kills.append(f"{name}: PDR invariant REJECTED by independent check: {row['inv_check']}")
        # controls
        ours_sat_depths = [d for d in (row["our_bmc_depth"] if row["our_bmc"] == "SAT" else "",) if d]
        if row["our_bmc"] == "SAT" and row["abc_bmc3"] == "SAT":
            row["depth_match"] = str(row["our_bmc_depth"] == row["abc_bmc3_depth"])
            if row["depth_match"] == "False":
                kills.append(f"{name}: BMC depth mismatch ours={row['our_bmc_depth']} abc={row['abc_bmc3_depth']}")
        if row["our_pdr"] == "SAT" and row["abc_bmc3"] == "SAT":
            if row["our_pdr_depth_or_frame"] != row["abc_bmc3_depth"]:
                kills.append(f"{name}: PDR counterexample depth {row['our_pdr_depth_or_frame']} vs abc bmc3 {row['abc_bmc3_depth']}")
        abc_says_sat = "SAT" in (row["abc_bmc3"], row["abc_pdr"]); abc_says_unsat = row["abc_pdr"] == "UNSAT"
        ours_says_sat = "SAT" in (row["our_bmc"], row["our_pdr"]); ours_says_unsat = row["our_pdr"] == "UNSAT"
        if (abc_says_sat and ours_says_unsat) or (abc_says_unsat and ours_says_sat):
            row["verdict_consistent"] = "False"; kills.append(f"{name}: verdict contradiction abc=({row['abc_bmc3']},{row['abc_pdr']}) ours=({row['our_bmc']},{row['our_pdr']})")
        elif (abc_says_sat or abc_says_unsat) and (ours_says_sat or ours_says_unsat):
            row["verdict_consistent"] = "True"
        rows.append(row)
        print(f"{name:26s} bmc {row['our_bmc']:12s}@{row['our_bmc_depth']:>4s} {row['our_bmc_s']:>6s}s | pdr {row['our_pdr']:9s}@{row['our_pdr_depth_or_frame']:>4s} {row['our_pdr_s']:>6s}s {row['inv_check'][:9]:9s} | abc bmc3 {row['abc_bmc3']}@{row['abc_bmc3_depth']} pdr {row['abc_pdr']}@{row['abc_pdr_depth']} | match={row['depth_match']} cons={row['verdict_consistent']}", flush=True)
        if kills:
            print("KILL:", kills[-1], flush=True); break
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    n = len(rows)
    bmc_sat = [r for r in rows if r["our_bmc"] == "SAT"]; abc_sat = [r for r in rows if r["abc_bmc3"] == "SAT"]
    pdr_unsat = [r for r in rows if r["our_pdr"] == "UNSAT"]; abc_unsat = [r for r in rows if r["abc_pdr"] == "UNSAT"]
    both_sat = [r for r in rows if r["our_bmc"] == "SAT" and r["abc_bmc3"] == "SAT"]
    def ratio(rs, ok, ak):
        xs = [float(r[ok]) / max(float(r[ak]), 1e-3) for r in rs if r[ok] and r[ak]]
        xs.sort(); return (f"median {xs[len(xs)//2]:.1f}x, min {xs[0]:.1f}x, max {xs[-1]:.1f}x") if xs else "n/a"
    summ = [f"# bmc-pdr-campaign summary ({n} instances < {a.max_kb} KB, timeout {a.timeout}s)",
            f"BMC: ours SAT {len(bmc_sat)} (ABC bmc3 SAT {len(abc_sat)}); depth agreement on the {len(both_sat)} both-SAT: {sum(1 for r in both_sat if r['depth_match']=='True')}/{len(both_sat)}; ours BOUNDED_SAFE {sum(1 for r in rows if r['our_bmc']=='BOUNDED_SAFE')}, TIMEOUT {sum(1 for r in rows if r['our_bmc']=='TIMEOUT')}",
            f"BMC time ours/ABC on both-SAT: {ratio(both_sat, 'our_bmc_s', 'abc_bmc3_s')}",
            f"PDR: ours UNSAT {len(pdr_unsat)} (ABC pdr UNSAT {len(abc_unsat)}), ours SAT {sum(1 for r in rows if r['our_pdr']=='SAT')}, EXHAUSTED {sum(1 for r in rows if r['our_pdr']=='EXHAUSTED')}, TIMEOUT {sum(1 for r in rows if r['our_pdr']=='TIMEOUT')}",
            f"PDR invariants certified by independent check: {sum(1 for r in pdr_unsat if r['inv_check'].startswith('CERTIFIED'))}/{len(pdr_unsat)}; empty invariants (P inductive): {sum(1 for r in pdr_unsat if r['our_pdr_clauses']=='0')}",
            f"PDR time ours/ABC on both-UNSAT: {ratio([r for r in pdr_unsat if r['abc_pdr']=='UNSAT'], 'our_pdr_s', 'abc_pdr_s')}",
            f"verdict contradictions: {sum(1 for r in rows if r['verdict_consistent']=='False')}; kills: {kills if kills else 'none'}"]
    (out / "summary.md").write_text("\n".join(summ) + "\n"); print("\n".join(summ))
    sys.exit(2 if kills else 0)

if __name__ == "__main__":
    main()
