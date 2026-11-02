# bmc-pdr-campaign summary (119 instances < 20 KB, our timeout 60s per crate; the ABC reference used its own caps — see abc.csv/run-abc.sh — so coverage is not like-for-like)
BMC: ours SAT 21 (ABC bmc3 SAT 26); depth agreement on the 21 both-SAT: 21/21; ours BOUNDED_SAFE 9, TIMEOUT 89
BMC counterexamples replayed through the AIG by tools/replay-trace.py: 21/21; PDR counterexamples replayed: 4/4
BMC time ours/ABC on both-SAT: median 0.2x, min 0.0x, max 22.9x (ABC times include ~40 ms process start-up; ratios on sub-0.1 s instances are not meaningful)
PDR: ours UNSAT 20 (ABC pdr UNSAT 55), ours SAT 4, EXHAUSTED 0, TIMEOUT 95
PDR invariants certified by independent check: 20/20; fences that are the property clause alone (P inductive, frame of 1 clause): 13
PDR time ours/ABC on both-UNSAT: median 185.1x, min 0.0x, max 1148.9x (ABC times include ~40 ms process start-up; ratios on sub-0.1 s instances are not meaningful)
verdict contradictions: 0; kills: none
