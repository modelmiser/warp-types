#!/bin/bash
# compare_ptx.sh — Verify zero-overhead: typed vs untyped butterfly PTX
#
# Compiles typed_vs_untyped.cu to PTX, extracts function bodies,
# normalizes names, and diffs. Identical PTX = zero overhead.
#
# Usage: bash reproduce/compare_ptx.sh

set -e
cd "$(dirname "$0")"

ARCH="sm_89"  # Ada Lovelace (RTX 4000 SFF Ada)
SRC="typed_vs_untyped.cu"
PTX="typed_vs_untyped.ptx"

echo "=== Compiling to PTX (nvcc -ptx -arch=${ARCH} -O2) ==="
nvcc -ptx -arch="${ARCH}" -O2 "${SRC}" -o "${PTX}"
echo "Generated: ${PTX}"
echo ""

# Extract the .func block for a given mangled name.
# Matches: .func (.param ...) _ZNNfuncnamei( ... ) { ... }
# Stops at the first closing brace at column 0.
# Normalizes mangled names and parameter names for comparison.
extract_func() {
    local mangled="$1"
    local ptx_file="$2"
    awk "
        /^\\.func.*${mangled}/ { capture=1 }
        capture { print }
        capture && /^}/ { exit }
    " "${ptx_file}" \
    | sed -E "s/${mangled}[^ ]*/FUNC/g; s/${mangled}/FUNC/g"
}

# Find the mangled names
MANGLED_U=$(grep -oP '_Z\d+butterfly_untypedi' "${PTX}" | head -1)
MANGLED_T=$(grep -oP '_Z\d+butterfly_typedi' "${PTX}" | head -1)

echo "=== Mangled names ==="
echo "  untyped: ${MANGLED_U}"
echo "  typed:   ${MANGLED_T}"
echo ""

echo "=== Extracting .func bodies ==="
extract_func "${MANGLED_U}" "${PTX}" > /tmp/ptx_untyped.txt
extract_func "${MANGLED_T}" "${PTX}" > /tmp/ptx_typed.txt

LINES_U=$(wc -l < /tmp/ptx_untyped.txt)
LINES_T=$(wc -l < /tmp/ptx_typed.txt)
echo "  butterfly_untyped: ${LINES_U} PTX lines"
echo "  butterfly_typed:   ${LINES_T} PTX lines"
echo ""

echo "=== Comparing ==="
if diff -q /tmp/ptx_untyped.txt /tmp/ptx_typed.txt > /dev/null 2>&1; then
    echo "IDENTICAL PTX"
    echo ""
    echo "The typestate annotations produce zero additional PTX instructions."
    echo "Type system overhead: 0 instructions, 0 registers, 0 bytes."
else
    echo "DIFFERENT PTX (unexpected)"
    echo ""
    diff /tmp/ptx_untyped.txt /tmp/ptx_typed.txt || true
fi

echo ""
echo "=== Full PTX for butterfly_untyped ==="
awk "/^\\.func.*${MANGLED_U}/,/^}/" "${PTX}"
