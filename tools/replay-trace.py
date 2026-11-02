#!/usr/bin/env python3
"""Replay a counterexample trace through an AIG, independently of the crates.

Own AIGER reader (binary/ASCII), own simulator. The trace (`bmc --trace` / `pdr --trace`) gives the inputs at every
step and the latch values at step 0 (`step aiger-literal value` lines). We start from the AIGER reset values
(taking the trace's step-0 value only for uninitialised latches), simulate step by step with the trace's inputs,
and require: every invariant constraint true at every step 0..K, and the bad literal true at step K.

    python3 tools/replay-trace.py --aig FILE --trace FILE
Exit 0 = REPLAYED (a genuine counterexample of length K); 1 = REJECTED; 2 = error.
"""
import argparse, sys
from pathlib import Path

def parse_aiger(data: bytes):
    pos = 0
    def line():
        nonlocal pos
        end = data.index(b"\n", pos); s = data[pos:end].decode(); pos = end + 1; return s.strip()
    hdr = line().split()
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
        latches.append((lit, nxt, "0" if r == 0 else "1" if r == 1 else "x"))
    outputs = [int(line()) for _ in range(O)]; bad = [int(line()) for _ in range(B)]; cons = [int(line()) for _ in range(C)]
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
                ch = data[pos]; pos += 1; x |= (ch & 0x7f) << sh
                if not ch & 0x80: return x
                sh += 7
        for k in range(A):
            lhs = 2 * (I + L + k + 1); r0 = lhs - delta(); r1 = r0 - delta(); ands.append((lhs, r0, r1))
    else:
        for _ in range(A):
            a, b, c = map(int, line().split()); ands.append((a, b, c))
    badlit = bad[0] if len(bad) == 1 else outputs[0] if (not bad and len(outputs) == 1) else None
    if badlit is None: raise ValueError("need exactly one bad/output")
    return dict(M=M, inputs=inputs, latches=latches, ands=ands, cons=cons, bad=badlit)

def read_trace(path):
    steps = {}
    for line in Path(path).read_text().split("\n"):
        t = line.split()
        if not t or t[0] == "c": continue
        k, lit, v = int(t[0]), int(t[1]), int(t[2]); steps.setdefault(k, {})[lit] = v
    return steps

def simulate(aig, steps):
    K = max(steps) if steps else 0
    val = {}  # variable index -> 0/1 for the current step
    def lit(l):
        if l == 0: return 0
        if l == 1: return 1
        v = val[l >> 1]; return v ^ (l & 1)
    # latches at step 0
    for lit_, nxt, reset in aig["latches"]:
        idx = lit_ >> 1
        if reset == "0": val[idx] = 0
        elif reset == "1": val[idx] = 1
        else:
            if lit_ not in steps.get(0, {}): return False, f"uninitialised latch {lit_} has no step-0 value in the trace"
            val[idx] = steps[0][lit_]
    for k in range(K + 1):
        for i in aig["inputs"]:
            if i not in steps.get(k, {}): return False, f"input {i} missing at step {k}"
            val[i >> 1] = steps[k][i]
        for lhs, r0, r1 in aig["ands"]:
            val[lhs >> 1] = lit(r0) & lit(r1)
        for c in aig["cons"]:
            if lit(c) != 1: return False, f"constraint {c} violated at step {k}"
        if k == K:
            return (lit(aig["bad"]) == 1), f"bad={lit(aig['bad'])} at step {K}"
        nxt_vals = [(l >> 1, lit(n)) for l, n, _ in aig["latches"]]
        for idx, v in nxt_vals: val[idx] = v
    return False, "empty trace"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aig", required=True); ap.add_argument("--trace", required=True)
    a = ap.parse_args()
    try:
        aig = parse_aiger(Path(a.aig).read_bytes()); steps = read_trace(a.trace)
    except Exception as e:
        print(f"error: {e}"); sys.exit(2)
    ok, why = simulate(aig, steps)
    print(f"{'REPLAYED' if ok else 'REJECTED'} depth={max(steps) if steps else 0} {why}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
