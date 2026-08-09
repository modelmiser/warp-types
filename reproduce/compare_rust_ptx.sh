#!/bin/bash
# compare_rust_ptx.sh — Prove zero-overhead: typed Rust vs untyped Rust on nvptx64
#
# Compiles actual Rust type system code (PhantomData, trait bounds,
# ComplementOf, diverge/merge) to NVIDIA PTX and compares with
# untyped equivalents. Byte-identical PTX = zero overhead.
#
# Unlike the CUDA comparison (compare_ptx.sh), this compiles ACTUAL
# Rust type system machinery to PTX, not just comments.
#
# Requires: rustc nightly with nvptx64-nvidia-cuda target
# Install:  rustup target add nvptx64-nvidia-cuda --toolchain nightly
#
# Usage: bash reproduce/compare_rust_ptx.sh

set -e
cd "$(dirname "$0")"

# Ensure nvptx64-nvidia-cuda target is installed
if ! rustup +nightly target list --installed | grep -q nvptx64-nvidia-cuda; then
    echo "Installing nvptx64-nvidia-cuda target..."
    rustup target add nvptx64-nvidia-cuda --toolchain nightly
fi

SRC="rust_ptx_typed.rs"
PTX="rust_ptx_typed.ptx"

echo "=== Compiling Rust to PTX (nvptx64-nvidia-cuda, -O) ==="
rustc +nightly --target nvptx64-nvidia-cuda --emit=asm -O "${SRC}" -o "${PTX}" 2>&1 | grep -v warning || true
echo "Generated: ${PTX}"
echo ""

# Extract function body between .visible .func ... { ... }
extract_func() {
    local name="$1"
    local ptx="$2"
    awk "
        /^\\.visible.*${name}/ { capture=1 }
        capture { print }
        capture && /^}/ { exit }
    " "${ptx}" | sed -E "s/${name}[_a-zA-Z0-9]*/FUNC/g"
}

# Assert an extracted function body is non-empty and actually contains the
# normalized function name. An empty or mismatched extraction would make the
# diff below pass vacuously — fail loudly instead.
assert_extracted() {
    local file="$1"
    local desc="$2"
    if [ ! -s "${file}" ] || ! grep -q "FUNC" "${file}"; then
        echo "ERROR: extraction of ${desc} failed (empty or missing function body: ${file})" >&2
        echo "PTX symbol names or layout may have changed — inspect ${PTX}." >&2
        exit 1
    fi
}

echo "=== Butterfly: typed vs untyped ==="
extract_func "butterfly_typed" "${PTX}" > /tmp/rust_ptx_typed.txt
extract_func "butterfly_untyped" "${PTX}" > /tmp/rust_ptx_untyped.txt
assert_extracted /tmp/rust_ptx_typed.txt "butterfly_typed"
assert_extracted /tmp/rust_ptx_untyped.txt "butterfly_untyped"

if diff -q /tmp/rust_ptx_typed.txt /tmp/rust_ptx_untyped.txt > /dev/null 2>&1; then
    echo "IDENTICAL PTX (butterfly)"
else
    echo "DIFFERENT PTX (butterfly) — unexpected"
    diff /tmp/rust_ptx_typed.txt /tmp/rust_ptx_untyped.txt || true
fi

echo ""
echo "=== Diverge/merge: typed vs untyped ==="
extract_func "diverge_merge_typed" "${PTX}" > /tmp/rust_ptx_dm_typed.txt
extract_func "diverge_merge_untyped" "${PTX}" > /tmp/rust_ptx_dm_untyped.txt
assert_extracted /tmp/rust_ptx_dm_typed.txt "diverge_merge_typed"
assert_extracted /tmp/rust_ptx_dm_untyped.txt "diverge_merge_untyped"

if diff -q /tmp/rust_ptx_dm_typed.txt /tmp/rust_ptx_dm_untyped.txt > /dev/null 2>&1; then
    echo "IDENTICAL PTX (diverge/merge)"
else
    echo "DIFFERENT PTX (diverge/merge) — unexpected"
    diff /tmp/rust_ptx_dm_typed.txt /tmp/rust_ptx_dm_untyped.txt || true
fi

echo ""
echo "=== Full PTX for butterfly_typed ==="
awk '/^\.visible.*butterfly_typed/,/^}/' "${PTX}"
echo ""
echo "=== Full PTX for diverge_merge_typed ==="
awk '/^\.visible.*diverge_merge_typed/,/^}/' "${PTX}"
