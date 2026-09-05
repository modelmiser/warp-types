#!/bin/bash
# runpod-h200.sh — Full warp-types GPU verification on RunPod H200 SXM
#
# Paste this into a RunPod H200 SXM terminal.
# Takes ~10-15 minutes (mostly Rust install + compile).
#
# What it verifies:
#   1. GPU info (confirm H200, compute 9.0)
#   2. Shuffle semantics (wrap-mod-32, clamp, overflow)
#   3. Zero-overhead PTX (typed vs untyped identical)
#   4. Full demo (buggy CUDA → compile error → typed fix on GPU)
#   5. Rust test suite on GPU hardware
#   6. PTX REGENERATION under the repo pin, then execution of what it produced
#      (step 5b) — the two committed .ptx files are ISA 6.0 from an unrecorded
#      toolchain while the pin emits 7.0, so they must be regenerated and re-run
#      TOGETHER or the paper's hardware numbers lose their artifact
#   7. The `gpu` feature (step 5c) — the only non-default feature no CI job
#      covers, and the only way `gpu_launcher.rs`'s bounds asserts ever execute
#
# The toolchain is NOT pinned here; `rust-toolchain.toml` governs. $RESULTS
# records what was actually used, because a provenance run that misnames its own
# compiler is worse than none.
#
# Output goes to /tmp/h200-results.txt for easy copy-paste.

set -e

RESULTS="/tmp/h200-results.txt"
echo "=== warp-types H200 SXM Verification ===" | tee "$RESULTS"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 0a: PATH ---
# The stock RunPod `*-cuda*-devel` images ship nvcc at /usr/local/cuda/bin but do
# not put it on PATH, in login OR non-login shells. Steps 4 and 5 call nvcc
# directly, so without this the script dies on a machine that has a perfectly
# good CUDA toolkit. Verified on a pytorch:2.4.0-cuda12.4.1-devel H200 pod,
# 2026-09-05.
if ! command -v nvcc &>/dev/null && [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

# --- Step 0: GPU Info ---
echo "=== Step 0: GPU Info ===" | tee -a "$RESULTS"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | tee -a "$RESULTS"
nvcc --version | tail -2 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 1: Install Rust nightly ---
echo "=== Step 1: Installing Rust ===" | tee -a "$RESULTS"
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
    source "$HOME/.cargo/env"
else
    echo "Rust already installed" | tee -a "$RESULTS"
fi
# NO explicit pin here. `rust-toolchain.toml` in the repo governs, and it also
# declares the components (rust-src, clippy, rustfmt) and the nvptx64 target, so
# rustup installs them on the first cargo invocation inside the clone. This
# script used to `rustup default nightly-2026-04-03`, three months behind the
# repo's pin — ineffective inside the repo (the toolchain file wins, verified
# 2026-09-05) but it installed an unused toolchain and made $RESULTS name the
# wrong one, which for a provenance run is the whole point.
#
# rust-src is still added to generic `nightly` as a fallback: WarpBuilder now
# discovers the kernel crate's pin (fixed 2026-09-05 — it used to hardcode
# RUSTUP_TOOLCHAIN="nightly", which outranks rust-toolchain.toml and unpinned
# every kernel build), but falls back to plain `nightly` for crates with no pin.
# The pinned toolchain gets its components from the toolchain file itself.
rustup component add rust-src --toolchain nightly 2>/dev/null || true
echo "" | tee -a "$RESULTS"
echo "--- toolchains (recorded so results cannot misattribute) ---" | tee -a "$RESULTS"
echo "rustup default : $(rustup default 2>/dev/null)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 2: Clone repo ---
echo "=== Step 2: Clone ===" | tee -a "$RESULTS"
cd /tmp
if [ -d warp-types ]; then
    cd warp-types && git pull
else
    git clone https://github.com/modelmiser/warp-types.git
    cd warp-types
fi
echo "Commit: $(git rev-parse --short HEAD)" | tee -a "$RESULTS"
echo "repo pin      : $(grep '^channel' rust-toolchain.toml)" | tee -a "$RESULTS"
echo "rustc in repo : $(rustc --version)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 3: Rust tests (CPU) ---
echo "=== Step 3: Rust Tests (CPU) ===" | tee -a "$RESULTS"
cargo test --workspace --lib 2>&1 | grep "test result" | tee -a "$RESULTS"
cargo test --workspace --doc 2>&1 | grep "test result" | tee -a "$RESULTS"
# --workspace, matching CI: an unscoped --doc sees the root crate only (31 vs 37).
echo "" | tee -a "$RESULTS"

# --- Step 4: GPU shuffle semantics ---
echo "=== Step 4: GPU Shuffle Semantics ===" | tee -a "$RESULTS"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.' | head -1)
echo "Compiling for sm_${ARCH}..." | tee -a "$RESULTS"
cd reproduce
nvcc -arch=sm_${ARCH} -o /tmp/gpu_semantics_test gpu_semantics_test.cu
/tmp/gpu_semantics_test 2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 5: Zero-overhead PTX ---
echo "=== Step 5: Zero-Overhead PTX ===" | tee -a "$RESULTS"
CUDA_ARCH=sm_${ARCH} bash compare_ptx.sh 2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 5b: Regenerate the committed PTX, then RUN it ---
#
# Why this step exists. `typed_butterfly_kernel.ptx` and `reduce7_typed.ptx` are
# committed and are what `reproduce/host` loads on the GPU — so they are the
# artifact the paper's H200 / RTX 4000 Ada numbers actually ran. Both were built
# by an unrecorded toolchain and emit `.version 6.0`; the current pin emits 7.0
# (59 and 121 lines differ). Regenerating alone would be WORSE than leaving them:
# it would silently replace the evidence with something never executed.
#
# So regenerate and re-run in the same session, and commit both together or
# neither. If the host harness fails after regeneration, restore the .ptx from
# git and report — do not commit a half-updated pair.
echo "=== Step 5b: PTX Regeneration + Execution ===" | tee -a "$RESULTS"
cd /tmp/warp-types/reproduce
for k in typed_butterfly_kernel reduce7_typed rust_ptx_typed; do
    cp "$k.ptx" "/tmp/$k.ptx.committed" 2>/dev/null || true
    rustc --target nvptx64-nvidia-cuda --emit=asm -O --edition 2021 "$k.rs" -o "$k.ptx" 2>/dev/null
    OLD_ISA=$(grep -m1 '^.version' "/tmp/$k.ptx.committed" 2>/dev/null || echo "(none)")
    NEW_ISA=$(grep -m1 '^.version' "$k.ptx")
    if diff -q "/tmp/$k.ptx.committed" "$k.ptx" >/dev/null 2>&1; then
        echo "$k: UNCHANGED ($NEW_ISA)" | tee -a "$RESULTS"
    else
        echo "$k: REGENERATED  committed[$OLD_ISA] -> pinned[$NEW_ISA]" | tee -a "$RESULTS"
    fi
done
echo "" | tee -a "$RESULTS"

# Execute the regenerated PTX on this GPU. These are the numbers that make the
# regeneration committable — without them the new .ptx has no more provenance
# than the old.
echo "--- host harness on the REGENERATED ptx ---" | tee -a "$RESULTS"
cd /tmp/warp-types/reproduce/host
cargo run --release 2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 5c: The gpu feature, which no CI job covers ---
echo "=== Step 5c: gpu feature (CUDA-only, untested anywhere else) ===" | tee -a "$RESULTS"
cd /tmp/warp-types
cargo test -p warp-types-sat --features gpu 2>&1 | grep -E "test result|error|panicked" | tee -a "$RESULTS"
# This is the only place `gpu_launcher.rs`'s num_vars bounds asserts can fire;
# they have never executed. A panic here is a finding, not a flake.
echo "" | tee -a "$RESULTS"

# --- Step 6: Full demo (bug → error → fix) ---
echo "=== Step 6: Full Demo ===" | tee -a "$RESULTS"
cd /tmp/warp-types
bash reproduce/demo.sh 2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Summary ---
echo "=== DONE ===" | tee -a "$RESULTS"
echo "Results saved to $RESULTS" | tee -a "$RESULTS"
echo "Copy with: cat $RESULTS"
