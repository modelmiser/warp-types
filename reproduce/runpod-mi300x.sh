#!/bin/bash
# runpod-mi300x.sh --- Full warp-types GPU verification on RunPod MI300X
#
# Paste this into a RunPod MI300X terminal (or any AMD GPU pod with ROCm).
# Takes ~8-12 minutes (mostly Rust install + hipcc compile).
#
# What it verifies:
#   1. GPU info (confirm MI300X, gfx942, wavefront 64)
#   2. Rust CPU test suite (same as runpod-h200.sh --- mechanization unchanged across vendors)
#   3. HIP kernel suite (amd_mi300x_verify.hip):
#      a. 64-lane butterfly reduction (6 stages needed, not NVIDIA's 5)
#      b. 5-stage reduction bug (NVIDIA pattern on 64-wide --- halves don't cross)
#      c. Diverged shuffle (even/odd lane bug class)
#      d. Diverged half-wavefront (32 of 64 active)
#      e. Wavefront ballot (confirms 64-wide width)
#
# Output goes to /tmp/mi300x-results.txt for easy copy-paste.
#
# Companion to runpod-h200.sh:
#   - H200 script: NVIDIA 32-lane, CUDA/PTX, reduce7 bug class
#   - MI300X script: AMD 64-lane, HIP/GCN, 5-vs-6-stage + wide-wavefront bugs
# Both demonstrate that the paper's type system is width-parametric
# (PSet n at n=32 for NVIDIA, n=64 for AMD).

set -e

RESULTS="/tmp/mi300x-results.txt"
echo "=== warp-types MI300X Verification ===" | tee "$RESULTS"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 0: GPU Info ---
echo "=== Step 0: GPU Info ===" | tee -a "$RESULTS"
if command -v rocm-smi &>/dev/null; then
    rocm-smi --showproductname --showdriverversion | tee -a "$RESULTS"
else
    echo "WARNING: rocm-smi not found. HIP tests will likely fail." | tee -a "$RESULTS"
fi
if command -v hipcc &>/dev/null; then
    hipcc --version 2>&1 | head -3 | tee -a "$RESULTS"
else
    echo "ERROR: hipcc not found. Install ROCm (apt install rocm-hip-sdk on Ubuntu)." | tee -a "$RESULTS"
    exit 1
fi
echo "" | tee -a "$RESULTS"

# --- Step 1: Install Rust nightly (pinned) ---
echo "=== Step 1: Installing Rust ===" | tee -a "$RESULTS"
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    source "$HOME/.cargo/env"
else
    echo "rustup already installed" | tee -a "$RESULTS"
fi
# Pin matches rust-toolchain.toml in the repo.
# No explicit pin: `rust-toolchain.toml` governs and declares its own
# components. This used to force nightly-2026-04-03, three months behind the
# repo's pin — ineffective inside the clone but it named the wrong toolchain in
# the results. Same fix as runpod-h200.sh (2026-09-05).
rustc --version | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 2: Clone repo ---
echo "=== Step 2: Clone ===" | tee -a "$RESULTS"
cd /tmp
if [ -d warp-types ]; then
    cd warp-types && git pull --ff-only
else
    git clone https://github.com/modelmiser/warp-types.git
    cd warp-types
fi
echo "Commit: $(git rev-parse --short HEAD)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 3: Rust tests (CPU) ---
# The mechanization and the CPU-side type-system tests are vendor-agnostic ---
# they verify that the Rust type system rejects untypable programs at compile
# time regardless of which GPU vendor's intrinsics the kernels target.
echo "=== Step 3: Rust Tests (CPU, vendor-agnostic) ===" | tee -a "$RESULTS"
cargo test --workspace --lib 2>&1 | grep "test result" | tee -a "$RESULTS"
cargo test --doc 2>&1 | grep "test result" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 4: HIP kernel suite on MI300X ---
echo "=== Step 4: HIP Kernel Suite (AMD MI300X) ===" | tee -a "$RESULTS"
cd reproduce
# hipcc handles arch autodetection; gfx942 is the MI300X target.
# --offload-arch can be overridden if the pod has a different AMD part.
OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx942}"
echo "Compiling amd_mi300x_verify.hip for --offload-arch=${OFFLOAD_ARCH}..." | tee -a "$RESULTS"
hipcc --offload-arch="${OFFLOAD_ARCH}" -O2 -o /tmp/amd_verify amd_mi300x_verify.hip
echo "Running..." | tee -a "$RESULTS"
/tmp/amd_verify 2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Summary ---
echo "=== DONE ===" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
echo "Expected outcomes (see amd_mi300x_verify.hip for test definitions):" | tee -a "$RESULTS"
echo "  Wavefront size:       64" | tee -a "$RESULTS"
echo "  ballot(1) popcount:   64 (all lanes active)" | tee -a "$RESULTS"
echo "  Test 1 (6-stage):     PASS --- every lane reads sum(0..63) = 2016" | tee -a "$RESULTS"
echo "  Test 2 (5-stage bug): CONFIRMED --- lane 0 reads 496 (half-sum), not 2016" | tee -a "$RESULTS"
echo "  Test 3 (diverged):    demonstrates even-only shuffle reads inactive lanes" | tee -a "$RESULTS"
echo "  Test 4 (half-active): demonstrates 32-of-64 active-lane divergence" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
echo "Paper claim: all four test patterns that produce wrong answers or undefined" | tee -a "$RESULTS"
echo "results on MI300X are *untypable* in the Rust type system. The type checker" | tee -a "$RESULTS"
echo "rejects them at compile time; see warp-types-kernel's ActiveSet-indexed APIs." | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
echo "Results saved to $RESULTS" | tee -a "$RESULTS"
echo "Copy with: cat $RESULTS"
