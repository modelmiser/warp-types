#!/bin/bash
# runpod-lrat.sh — measure the warp-typed LRAT checker on a RunPod GPU.
#
# Expects (uploaded by the driver):
#   /tmp/warp-types.tgz          git archive of the tree under test
#   /tmp/lrat-corpus/            *.cnf + *.blrat (binary LRAT) for the timing runs,
#                                archive-proofs.tgz (run*/cnf, run*/proof text LRATs)
# Writes /tmp/lrat-results.txt. Prints "=== DONE ===" last; the driver requires it.
#
# Steps: GPU info; Rust under the repo pin; build the gpu feature + example;
# (A) agreement: every archived proof through the device path, verdict must
#     equal the CPU reference's (the example exits 3 on any disagreement);
# (B) timing: each binary proof, blocks ∈ {1024, 4096, 16384}, CPU passes
#     skipped except the independent checker (the comparison point).
set -uo pipefail
R=/tmp/lrat-results.txt
: > "$R"
log() { echo "$*" | tee -a "$R"; }
log "=== warp-types LRAT checker on RunPod ==="
log "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ! command -v nvcc &>/dev/null && [ -x /usr/local/cuda/bin/nvcc ]; then export PATH="/usr/local/cuda/bin:$PATH"; fi
log "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | tee -a "$R"
nproc | sed 's/^/host cores: /' | tee -a "$R"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.' | head -1)
export SM_ARCH="sm_${ARCH}"
log "SM_ARCH=$SM_ARCH"

log "=== Rust ==="
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none >/dev/null 2>&1
fi
source "$HOME/.cargo/env"
mkdir -p /tmp/warp-types && tar -xzf /tmp/warp-types.tgz -C /tmp/warp-types
cd /tmp/warp-types
rustup show active-toolchain 2>&1 | tee -a "$R"
rustup component add rust-src >/dev/null 2>&1
rustup target add nvptx64-nvidia-cuda >/dev/null 2>&1
rustc --version | tee -a "$R"

log "=== Build (gpu feature, $SM_ARCH) ==="
t0=$(date +%s)
if ! cargo build --release -p warp-types-sat --features gpu --example lrat_check 2>&1 | tail -3 | tee -a "$R"; then
    log "BUILD FAILED"; log "=== DONE ==="; exit 1
fi
log "build time: $(( $(date +%s) - t0 ))s"
BIN=/tmp/warp-types/target/release/examples/lrat_check

log "=== (0) Diagnostic: one full run on the smallest binary proof (stdout+stderr, backtrace on) ==="
f=$(ls /tmp/lrat-corpus/*.blrat | xargs ls -S | tail -1); base=$(basename "$f" .blrat)
RUST_BACKTRACE=1 LRAT_GPU=1 LRAT_GPU_BLOCKS=1024 LRAT_SKIP_SLOW=1 "$BIN" /tmp/lrat-corpus/$base.cnf "$f" 4 2>&1 | tail -40 | tee -a "$R"
log "diagnostic exit: ${PIPESTATUS[0]}"

log "=== (A) Agreement on the archive: device path vs CPU reference ==="
mkdir -p /tmp/lrat-archive && tar -xzf /tmp/lrat-corpus/archive-proofs.tgz -C /tmp/lrat-archive
n=0; ok=0; dis=0; ver=0; err=0
for d in /tmp/lrat-archive/*/; do
    for p in "$d"proof/*.lrat; do
        [ -f "$p" ] || continue
        b=$(basename "$p"); base=${b%%.*}; c="$d"cnf/$base.cnf
        [ -f "$c" ] || continue
        out=$(LRAT_GPU=1 LRAT_GPU_BLOCKS=1024 LRAT_SKIP_SLOW=1 "$BIN" "$c" "$p" 4 2>&1)
        rc=$?
        n=$((n+1))
        # A run counts as agreeing ONLY if it printed a verdict line; a crash is an error, not agreement.
        if ! echo "$out" | grep -q "^s "; then err=$((err+1)); [ $err -le 3 ] && echo "ERROR $b (exit $rc): $(echo "$out" | grep -E 'panicked|Error|error' | head -2 | tr '\n' ' ')" | tee -a "$R"
        elif echo "$out" | grep -q "DISAGREE"; then dis=$((dis+1)); echo "DISAGREE $b: $(echo "$out" | grep -E '^c gpu|^s ' | tr '\n' ' ')" | tee -a "$R"
        else ok=$((ok+1)); fi
        echo "$out" | grep -q "^s VERIFIED" && ver=$((ver+1))
    done
done
log "archive: $n proofs, $ok agree, $dis disagree, ${err:-0} errors, $ver verified"
# Note: text LRATs take the parse+flatten route; the device path uses the same flat layout.

log "=== (B) Timing on binary proofs ==="
for f in /tmp/lrat-corpus/*.blrat; do
    base=$(basename "$f" .blrat); c=/tmp/lrat-corpus/$base.cnf
    log "--- $base ($(stat -c %s "$f") bytes) ---"
    LRAT_GPU=1 LRAT_GPU_BLOCKS=1024,4096,16384 LRAT_SKIP_SLOW=1 "$BIN" "$c" "$f" "$(nproc)" 2>&1 | grep -E "^c read|^c flat|^c gpu|^s |panicked|Error|error" | tee -a "$R"
done
log "=== DONE ==="
