#!/bin/bash
# runpod-lrat-chain.sh — the whole certified chain on one GPU machine:
#   solve --lrat-binary  →  lrat-trim  →  device check (and lrat-check / cake_lpr on the same box)
#
# Expects (uploaded by the driver):
#   /tmp/warp-types.tgz        git archive of the tree under test
#   /tmp/chain/cnf/*.cnf       UNSAT instances (the certification corpus)
#   /tmp/chain/lrat-check.c    compiled here with gcc
#   /tmp/chain/cake_lpr        static-enough binary (libc only)
#   /tmp/chain/big.{cnf,lrat,blrat}  one large EXTERNAL (kissat + drat-trim) proof for
#                              the same-machine lrat-check / cake_lpr / device comparison
# Writes /tmp/chain-results.txt and /tmp/chain-results.csv. Prints "=== DONE ===" last.
set -uo pipefail
R=/tmp/chain-results.txt; CSV=/tmp/chain-results.csv
: > "$R"
log() { echo "$*" | tee -a "$R"; }
log "=== warp-types certified chain on RunPod ==="
log "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ! command -v nvcc &>/dev/null && [ -x /usr/local/cuda/bin/nvcc ]; then export PATH="/usr/local/cuda/bin:$PATH"; fi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee -a "$R"
nproc | sed 's/^/host cores: /' | tee -a "$R"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.' | head -1); export SM_ARCH="sm_${ARCH}"
if ! command -v rustup &>/dev/null; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none >/dev/null 2>&1; fi
source "$HOME/.cargo/env"
mkdir -p /tmp/warp-types && tar -xzf /tmp/warp-types.tgz -C /tmp/warp-types && cd /tmp/warp-types
rustup component add rust-src >/dev/null 2>&1; rustup target add nvptx64-nvidia-cuda >/dev/null 2>&1
rustc --version | tee -a "$R"
log "=== Build ==="
if ! cargo build --release -p warp-types-sat --features gpu --bin solve --bin lrat-trim --example lrat_check 2>&1 | tail -2 | tee -a "$R"; then log "BUILD FAILED"; log "=== DONE ==="; exit 1; fi
PTX=$(ls -t target/release/build/warp-types-sat/*/out/kernels.ptx | head -1)
log "ptx: $(grep -m1 '^.target' "$PTX") externs=$(grep -c '\.extern' "$PTX")"
gcc /tmp/chain/lrat-check.c -std=c99 -O2 -o /tmp/chain/lrat-check 2>&1 | tail -2; chmod +x /tmp/chain/cake_lpr
SOLVE=target/release/solve; TRIM=target/release/lrat-trim; CHK=target/release/examples/lrat_check
LC=/tmp/chain/lrat-check; CAKE=/tmp/chain/cake_lpr
DT=$(( $(nproc) < 32 ? $(nproc) : 32 ))
now() { date +%s.%N; }
el() { echo "$2 - $1" | bc; }

echo "name,t_solve,bytes_untrimmed,t_trim,bytes_trimmed,t_decode,t_device,gpu_verdict,t_lrat_check,lrat_check,t_cake,cake" > "$CSV"
log "=== (A) Solver → binary LRAT → trim → device check, per instance (decode threads $DT) ==="
mkdir -p /tmp/chain/out
for c in /tmp/chain/cnf/*.cnf; do
    n=$(basename "$c" .cnf); o=/tmp/chain/out/$n
    t0=$(now); "$SOLVE" --lrat "$o.blrat" --lrat-binary "$c" > "$o.solve.txt" 2>&1; t1=$(now)
    grep -q "^s UNSATISFIABLE" "$o.solve.txt" || { log "$n: not UNSAT here ($(grep '^s ' "$o.solve.txt"))"; continue; }
    b0=$(stat -c %s "$o.blrat")
    t2=$(now); "$TRIM" "$o.blrat" "$o.t.blrat" --binary > "$o.trim.txt" 2>&1; "$TRIM" "$o.blrat" "$o.t.lrat" >/dev/null 2>&1; t3=$(now)
    b1=$(stat -c %s "$o.t.blrat")
    g=$(LRAT_GPU=1 LRAT_GPU_BLOCKS=4096 LRAT_SKIP_SLOW=1 LRAT_DECODE_THREADS=$DT "$CHK" "$c" "$o.t.blrat" 4 2>&1)
    td=$(echo "$g" | grep -oE 'flat-decode\([0-9]+\) [0-9.]+' | grep -oE '[0-9.]+$' | head -1)
    tg=$(echo "$g" | grep -oE 'device [0-9.]+s' | grep -oE '[0-9.]+' | head -1)
    gv=$(echo "$g" | grep -E '^s ' | head -1 | cut -c3-)
    t4=$(now); lcout=$("$LC" "$c" "$o.t.lrat" 2>&1); t5=$(now)
    lcv=$([ $? -eq 0 ] && echo "$lcout" | grep -q "c VERIFIED" && echo VERIFIED || echo NOT)
    t6=$(now); ck=$("$CAKE" "$c" "$o.t.blrat" 2>&1); t7=$(now)
    ckv=$(echo "$ck" | grep -q "s VERIFIED" && echo VERIFIED || echo NOT)
    echo "$n,$(el $t0 $t1),$b0,$(el $t2 $t3),$b1,${td:-},${tg:-},${gv:-none},$(el $t4 $t5),$lcv,$(el $t6 $t7),$ckv" >> "$CSV"
    echo "$n: solve $(el $t0 $t1)s trim $(el $t2 $t3)s bytes $b0->$b1 decode ${td:-?}s device ${tg:-?}s gpu=${gv:-none} lrat-check=$lcv ($(el $t4 $t5)s) cake=$ckv ($(el $t6 $t7)s)" | tee -a "$R"
    rm -f "$o.blrat" "$o.t.lrat"
done

log "=== (B) One large external proof, same machine: lrat-check (text) / cake_lpr (binary) / device (binary) ==="
c=/tmp/chain/big.cnf
t0=$(now); "$LC" "$c" /tmp/chain/big.lrat > /tmp/chain/big.lc.txt 2>&1; t1=$(now); log "lrat-check (text, $(stat -c %s /tmp/chain/big.lrat) bytes): $(el $t0 $t1)s $(grep -c 'c VERIFIED' /tmp/chain/big.lc.txt)"
t0=$(now); "$CAKE" "$c" /tmp/chain/big.blrat > /tmp/chain/big.ck.txt 2>&1; t1=$(now); log "cake_lpr (binary, $(stat -c %s /tmp/chain/big.blrat) bytes): $(el $t0 $t1)s $(grep -c 's VERIFIED' /tmp/chain/big.ck.txt)"
LRAT_GPU=1 LRAT_GPU_BLOCKS=4096 LRAT_SKIP_SLOW=1 LRAT_DECODE_THREADS=$DT "$CHK" "$c" /tmp/chain/big.blrat 4 2>&1 | grep -E "^c read|^c flat|^c gpu|^s " | tee -a "$R"
log "=== DONE ==="
