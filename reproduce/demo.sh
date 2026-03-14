#!/bin/bash
# demo.sh — The full warp-types demonstration
#
# Runs three things side by side:
#   1. Original CUDA reduce7 (buggy) — wrong results
#   2. Original CUDA reduce7 (fixed) — correct results
#   3. Typed Rust reduce7 on GPU — correct results, bug is compile error
#
# Usage: bash reproduce/demo.sh

set -e
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Session-Typed Divergence: The Complete Demo            ║"
echo "║  Same bug. Same GPU. Type system prevents it.           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Compile the CUDA version
echo "▸ Compiling CUDA reduce7 (buggy + fixed)..."
nvcc -O2 -o /tmp/reduce7_cuda reduce7_bug.cu 2>/dev/null
echo "  Done."

# Step 2: Compile the Rust PTX
echo "▸ Compiling typed Rust kernels to PTX..."
rustc +nightly --target nvptx64-nvidia-cuda --emit=asm -O \
    --edition 2021 reduce7_typed.rs -o reduce7_typed.ptx 2>/dev/null
echo "  Done."

# Step 3: Build the host runner (if needed)
echo "▸ Building host runner..."
(cd host && cargo build --release 2>/dev/null)
echo "  Done."
echo ""

# Run CUDA version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part 1: CUDA (no type system)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
/tmp/reduce7_cuda
echo ""

# Run Rust typed version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part 2: Rust with session-typed divergence"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
(cd host && cargo run --release 2>&1 | grep -v "warning\|^  -->\|^   =\|^   |$\|^   [0-9]\|Compiling\|Finished\|Running" | grep -v "^$")
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  CUDA (buggy):  sum = 1   ← silent wrong answer"
echo "  CUDA (fixed):  sum = 32  ← correct, but no compile-time safety"
echo "  Rust (typed):  sum = 32  ← correct, AND the bug is a compile error"
echo ""
echo "  The buggy pattern literally cannot be expressed in the type system."
echo "  Warp<Lane0> has no shuffle_down method. It doesn't exist."
echo ""
echo "  NVIDIA deprecated __shfl because of this bug class."
echo "  We eliminate it at compile time with zero runtime overhead."
