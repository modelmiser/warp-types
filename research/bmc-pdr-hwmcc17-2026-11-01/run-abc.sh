#!/bin/bash
# ABC reference: bmc3 and pdr on every small HWMCC17 instance, 30 s each.
# Output CSV: name,bytes,bmc3_verdict,bmc3_depth,bmc3_s,pdr_verdict,pdr_depth,pdr_s
cd ~/staging/2026-11-01-hwmcc
echo "name,bytes,bmc3_verdict,bmc3_depth,bmc3_s,pdr_verdict,pdr_depth,pdr_s" > ref/abc.csv
for f in $(find . -maxdepth 1 -name '*.aig' -size -20k | sort); do
  n=$(basename $f .aig); b=$(stat -c %s $f)
  s=$(date +%s.%N); o=$(timeout 40 yosys-abc -c "read_aiger $f; fold; bmc3 -F 1000 -T 30 -v" 2>&1); e=$(date +%s.%N)
  t1=$(echo "$e - $s" | bc)
  if echo "$o" | grep -q "asserted in frame"; then v1=SAT; d1=$(echo "$o" | grep -o "asserted in frame [0-9]*" | head -1 | awk '{print $4}')
  elif echo "$o" | grep -qi "No output asserted in [0-9]* frames"; then v1=UNKNOWN; d1=$(echo "$o" | grep -o "No output asserted in [0-9]* frames" | awk '{print $5}')
  else v1=NONE; d1=; fi
  s=$(date +%s.%N); o=$(timeout 40 yosys-abc -c "read_aiger $f; fold; pdr -T 30 -v" 2>&1); e=$(date +%s.%N)
  t2=$(echo "$e - $s" | bc)
  if echo "$o" | grep -q "asserted in frame"; then v2=SAT; d2=$(echo "$o" | grep -o "asserted in frame [0-9]*" | head -1 | awk '{print $4}')
  elif echo "$o" | grep -qi "Property proved"; then v2=UNSAT; d2=$(echo "$o" | grep -oE "Invariant F\[ *[0-9]+\]" | head -1 | grep -oE "[0-9]+")
  else v2=UNKNOWN; d2=; fi
  echo "$n,$b,$v1,$d1,$t1,$v2,$d2,$t2" >> ref/abc.csv
  echo "$n $v1@$d1 ${t1}s | $v2@$d2 ${t2}s"
done
echo DONE
