# bmc-pdr-campaign summary (119 instances < 20 KB, our timeout 60s per crate; the ABC reference used its own caps — see abc.csv/run-abc.sh — so coverage is not like-for-like)
BMC: ours SAT 0 (ABC bmc3 SAT 26); depth agreement on the 0 both-SAT: 0/0; ours BOUNDED_SAFE 0, TIMEOUT 0
BMC counterexamples replayed through the AIG by tools/replay-trace.py: 0/0; PDR counterexamples replayed: 11/11
BMC time ours/ABC on both-SAT: n/a
PDR: ours UNSAT 33 (ABC pdr UNSAT 55), ours SAT 11, EXHAUSTED 0, TIMEOUT 75
PDR invariants certified by independent check: 33/33; fences that are the property clause alone (P inductive, frame of 1 clause): 3
PDR time ours/ABC on both-UNSAT: median 16.5x, min 0.0x, max 507.2x (ABC times include ~40 ms process start-up; ratios on sub-0.1 s instances are not meaningful)
verdict contradictions: 0; kills: none
