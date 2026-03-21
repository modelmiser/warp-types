#!/bin/bash
# demo.sh — The complete warp-types demonstration
#
# Three beats:
#   1. THE BUG:   Run buggy CUDA code → wrong answer
#   2. THE ERROR: Try the buggy pattern in Rust → compiler rejects it
#   3. THE FIX:   Run typed Rust code on GPU → correct answer
#
# Usage: bash reproduce/demo.sh

set -e

# Resolve paths once at the top
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  warp-types: Type-Safe Warp Programming via Linear Typestate   ║"
echo "║  Same bug. Same GPU. The type system prevents it.              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ================================================================
# Beat 1: THE BUG
# ================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Beat 1: THE BUG"
echo "CUDA compiles the buggy pattern. GPU produces wrong answer."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Pattern: diverge to lane 0, then shuffle_down."
echo "  Lane 0 reads from lane 16 — but lane 16 didn't participate."
echo "  Its register value is whatever was there. Undefined behavior."
echo ""

if command -v nvcc &>/dev/null && [ -f "$SCRIPT_DIR/reduce7_bug.cu" ]; then
    echo "▸ Compiling CUDA reduce7 (buggy + fixed)..."
    nvcc -O2 -o /tmp/reduce7_cuda "$SCRIPT_DIR/reduce7_bug.cu" 2>/dev/null
    echo ""
    /tmp/reduce7_cuda
else
    if ! command -v nvcc &>/dev/null; then
        echo "  [nvcc not found — showing expected output]"
    else
        echo "  [CUDA source not included (requires NVIDIA CUDA Samples) — showing expected output]"
    fi
    echo ""
    echo "  === Buggy reduce7 (CUDA, partial mask) ==="
    echo "    Input:    [1, 1, 1, ..., 1]  (32 ones)"
    echo "    Expected: 32"
    echo "    Got:      1"
    echo "    Result:   WRONG"
    echo ""
    echo "  === Fixed reduce7 (CUDA, full mask) ==="
    echo "    Input:    [1, 1, 1, ..., 1]  (32 ones)"
    echo "    Expected: 32"
    echo "    Got:      32"
    echo "    Result:   correct"
fi
echo ""

# ================================================================
# Beat 2: THE ERROR (the centerpiece)
# ================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Beat 2: THE COMPILER ERROR"
echo "The same pattern in Rust. The compiler catches it."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Code:"
echo ""
echo "    let warp: Warp<All> = Warp::kernel_entry();"
echo "    let (evens, _odds) = warp.diverge_even_odd();"
echo "    evens.shuffle_xor(data, 1);  // <-- the bug"
echo ""
echo "  Compiler output:"
echo ""

# Actually compile it and capture the error
cp "$SCRIPT_DIR/buggy_pattern.rs" "$PROJECT_ROOT/examples/buggy_pattern.rs" 2>/dev/null || true
COMPILE_OUTPUT=$(cd "$PROJECT_ROOT" && cargo check --example buggy_pattern 2>&1 || true)
rm -f "$PROJECT_ROOT/examples/buggy_pattern.rs"

# Show just the relevant error
echo "$COMPILE_OUTPUT" | grep -A5 "no method named" | head -8 | sed 's/^/    /'
echo ""
echo "  The method was found for Warp<All> — not Warp<Even>."
echo "  The buggy pattern is not checked at runtime."
echo "  It does not exist in the type system."
echo ""

# ================================================================
# Beat 3: THE FIX
# ================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Beat 3: THE FIX"
echo "Typed Rust kernels on real GPU. Correct results."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if command -v nvidia-smi &>/dev/null; then
    echo "▸ Compiling typed Rust kernels to PTX..."
    cd "$PROJECT_ROOT/examples/gpu-project"
    cargo run --release 2>&1 | grep -E "GPU:|Result:|Input:|Expected:|Got:" | sed 's/^/  /'
else
    echo "  [No GPU detected — showing expected output]"
    echo "  GPU: NVIDIA RTX 4000 SFF Ada Generation"
    echo ""
    echo "  Test 1: butterfly_reduce     → PASS (32)"
    echo "  Test 2: diverge_merge_reduce → PASS (496)"
    echo "  Test 3: reduce_n             → PASS (32)"
    echo "  Test 4: bitonic_sort_i32     → PASS ([0,1,...,31])"
fi
echo ""

# ================================================================
# Summary
# ================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  CUDA:                Compiles the bug. Produces wrong answer."
echo "  Rust (warp-types):   Rejects the bug. Produces correct answer."
echo ""
echo "  The type system:"
echo "    ✓ Catches shuffle-from-inactive-lane at compile time"
echo "    ✓ Zero runtime overhead (PhantomData erasure)"
echo "    ✓ Verified at MIR, LLVM IR, and PTX levels"
echo "    ✓ Real GPU execution on NVIDIA H200 SXM and RTX 4000 Ada"
echo ""
echo "  21 documented bugs across 16 real-world projects."
echo "  31 Lean theorems. 395 Rust tests. One type system."
echo ""
echo "  github.com/modelmiser/warp-types"
echo ""
