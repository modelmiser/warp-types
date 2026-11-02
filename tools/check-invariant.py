#!/usr/bin/env python3
"""Independent check of a PDR inductive invariant for an AIGER safety instance.

Own binary/ASCII AIGER reader, own Tseitin encoding, kissat as the solver. Nothing from the warp-types crates
is in the path except the invariant text (`pdr --invariant OUT`: clauses over AIGER literals, 0-terminated,
the constant-true variable written as literal 1).

Three queries, each must be UNSAT:
  (1) initiation:  Init ∧ C ∧ ¬Inv
  (2) consecution: Inv ∧ C ∧ T ∧ C' ∧ ¬Inv'
  (3) safety:      Inv ∧ C ∧ bad
where C are the AIGER invariant constraints, T the latch updates, and every AND gate is defined at both steps.
Inputs are free at each step (they are state in the crates' lowering; an invariant may mention them).

    python3 tools/check-invariant.py --aig FILE --inv FILE --kissat PATH [--keep DIR]
Exit 0 = all three UNSAT (certified); 1 = some query SAT (invariant NOT inductive or NOT safe); 2 = error.
"""
import argparse, subprocess, sys, tempfile, os
from pathlib import Path

def parse_aiger(data: bytes):
    pos = 0
    def line():
        nonlocal pos
        end = data.index(b"\n", pos); s = data[pos:end].decode(); pos = end + 1; return s.strip()
    hdr = line().split()
    if hdr[0] not in ("aig", "aag"): raise ValueError("bad header")
    binary = hdr[0] == "aig"
    M, I, L, O, A = map(int, hdr[1:6]); B = int(hdr[6]) if len(hdr) > 6 else 0; C = int(hdr[7]) if len(hdr) > 7 else 0
    J = int(hdr[8]) if len(hdr) > 8 else 0; F = int(hdr[9]) if len(hdr) > 9 else 0
    inputs = [2 * (k + 1) for k in range(I)] if binary else [int(line()) for _ in range(I)]
    latches = []
    for k in range(L):
        t = line().split()
        if binary: lit = 2 * (I + k + 1); rest = t
        else: lit = int(t[0]); rest = t[1:]
        nxt = int(rest[0]); r = int(rest[1]) if len(rest) > 1 else 0
        reset = "0" if r == 0 else "1" if r == 1 else "x"
        latches.append((lit, nxt, reset))
    outputs = [int(line()) for _ in range(O)]
    bad = [int(line()) for _ in range(B)]
    cons = [int(line()) for _ in range(C)]
    if J or F:
        js = [int(line()) for _ in range(J)]
        for n in js:
            for _ in range(n): line()
        for _ in range(F): line()
    ands = []
    if binary:
        def delta():
            nonlocal pos
            x = 0; sh = 0
            while True:
                ch = data[pos]; pos += 1
                x |= (ch & 0x7f) << sh
                if not ch & 0x80: return x
                sh += 7
        for k in range(A):
            lhs = 2 * (I + L + k + 1); r0 = lhs - delta(); r1 = r0 - delta(); ands.append((lhs, r0, r1))
    else:
        for _ in range(A):
            a, b, c = map(int, line().split()); ands.append((a, b, c))
    if len(bad) == 1: badlit = bad[0]
    elif not bad and len(outputs) == 1: badlit = outputs[0]
    else: raise ValueError(f"need exactly one bad/output, got B={len(bad)} O={len(outputs)}")
    return dict(M=M, inputs=inputs, latches=latches, ands=ands, cons=cons, bad=badlit)

class Cnf:
    def __init__(self, M):
        self.M = M; self.clauses = []; self.nvars = 2 * M  # step 0: 1..M, step 1: M+1..2M
    def new_var(self):
        self.nvars += 1; return self.nvars
    def lit(self, aiger_lit, step):
        """AIGER literal → DIMACS literal at step 0/1. Returns None for constant-true (clause satisfied) and
        'FALSE' sentinel for constant-false."""
        idx = aiger_lit >> 1; neg = aiger_lit & 1
        if idx == 0:
            return "TRUE" if neg else "FALSE"   # literal 1 is true, literal 0 is false
        v = idx + step * self.M
        return -v if neg else v
    def add(self, lits):
        out = []
        for l in lits:
            if l == "TRUE": return          # clause satisfied
            if l == "FALSE": continue       # literal drops out
            out.append(l)
        self.clauses.append(out)
    def gate_defs(self, aig, step):
        for lhs, r0, r1 in aig["ands"]:
            g, a, b = self.lit(lhs, step), self.lit(r0, step), self.lit(r1, step)
            self.add([neg(g), a]); self.add([neg(g), b]); self.add([g, neg(a), neg(b)])
        for c in aig["cons"]:
            self.add([self.lit(c, step)])
    def init(self, aig):
        for lit, nxt, reset in aig["latches"]:
            if reset == "0": self.add([neg(self.lit(lit, 0))])
            elif reset == "1": self.add([self.lit(lit, 0)])
    def transition(self, aig):
        for lit, nxt, reset in aig["latches"]:
            l1 = self.lit(lit, 1); nx = self.lit(nxt, 0)
            self.add([neg(l1), nx]); self.add([l1, neg(nx)])
    def inv(self, inv_clauses, step):
        for c in inv_clauses:
            self.add([self.lit(l, step) for l in c])
    def not_inv(self, inv_clauses, step):
        sels = []
        for c in inv_clauses:
            lits = [self.lit(l, step) for l in c]
            if any(l == "TRUE" for l in lits): continue          # cannot be falsified
            lits = [l for l in lits if l != "FALSE"]
            t = self.new_var(); sels.append(t)
            for l in lits: self.clauses.append([-t, -l])
        self.clauses.append(sels)   # empty → UNSAT (¬Inv impossible): fine
    def write(self, path):
        with open(path, "w") as f:
            f.write(f"p cnf {self.nvars} {len(self.clauses)}\n")
            for c in self.clauses: f.write(" ".join(map(str, c)) + " 0\n")

def neg(l):
    if l == "TRUE": return "FALSE"
    if l == "FALSE": return "TRUE"
    return -l

def read_inv(path):
    out = []
    for line in Path(path).read_text().split("\n"):
        t = line.split()
        if not t or t[0] == "c": continue
        lits = [int(x) for x in t]
        assert lits[-1] == 0
        out.append(lits[:-1])
    return out

def kissat(bin_, cnf):
    p = subprocess.run([bin_, "-q", cnf], capture_output=True, text=True)
    return {10: "SAT", 20: "UNSAT"}.get(p.returncode, f"rc={p.returncode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aig", required=True); ap.add_argument("--inv", required=True)
    ap.add_argument("--kissat", required=True); ap.add_argument("--keep")
    a = ap.parse_args()
    try:
        aig = parse_aiger(Path(a.aig).read_bytes())
    except Exception as e:
        print(f"error: {e}"); sys.exit(2)
    inv = read_inv(a.inv)
    d = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="check-inv-"))
    d.mkdir(parents=True, exist_ok=True)
    results = {}
    # (1) initiation
    c = Cnf(aig["M"]); c.gate_defs(aig, 0); c.init(aig); c.not_inv(inv, 0); c.write(d / "init.cnf")
    results["initiation"] = kissat(a.kissat, str(d / "init.cnf"))
    # (2) consecution
    c = Cnf(aig["M"]); c.gate_defs(aig, 0); c.gate_defs(aig, 1); c.transition(aig); c.inv(inv, 0); c.not_inv(inv, 1); c.write(d / "cons.cnf")
    results["consecution"] = kissat(a.kissat, str(d / "cons.cnf"))
    # (3) safety
    c = Cnf(aig["M"]); c.gate_defs(aig, 0); c.inv(inv, 0); c.add([c.lit(aig["bad"], 0)]); c.write(d / "safe.cnf")
    results["safety"] = kissat(a.kissat, str(d / "safe.cnf"))
    ok = all(v == "UNSAT" for v in results.values())
    print(f"{'CERTIFIED' if ok else 'REJECTED'} inv_clauses={len(inv)} " + " ".join(f"{k}={v}" for k, v in results.items()))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
