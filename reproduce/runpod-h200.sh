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
#
# Output goes to /tmp/h200-results.txt for easy copy-paste.

set -e

RESULTS="/tmp/h200-results.txt"
echo "=== warp-types H200 SXM Verification ===" | tee "$RESULTS"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 0: GPU Info ---
echo "=== Step 0: GPU Info ===" | tee -a "$RESULTS"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 1: Install Rust nightly ---
echo "=== Step 1: Installing Rust ===" | tee -a "$RESULTS"
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
    source "$HOME/.cargo/env"
else
    echo "Rust already installed" | tee -a "$RESULTS"
fi
rustup install nightly-2026-03-19
rustup default nightly-2026-03-19
# rust-src needed for -Z build-std=core (cross-compile core for nvptx64).
# Install for both pinned and generic nightly (WarpBuilder uses generic).
rustup component add rust-src --toolchain nightly-2026-03-19
rustup component add rust-src --toolchain nightly 2>/dev/null || true
rustc --version | tee -a "$RESULTS"
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
echo "" | tee -a "$RESULTS"

# --- Step 3: Rust tests (CPU) ---
echo "=== Step 3: Rust Tests (CPU) ===" | tee -a "$RESULTS"
cargo test --workspace --lib 2>&1 | grep "test result" | tee -a "$RESULTS"
cargo test --doc 2>&1 | grep "test result" | tee -a "$RESULTS"
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

# --- Step 6: Full demo (bug → error → fix) ---
echo "=== Step 6: Full Demo ===" | tee -a "$RESULTS"
cd /tmp/warp-types
bash reproduce/demo.sh 2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Summary ---
echo "=== DONE ===" | tee -a "$RESULTS"
echo "Results saved to $RESULTS" | tee -a "$RESULTS"
echo "Copy with: cat $RESULTS"
