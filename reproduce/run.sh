#!/bin/bash
# Reproduce cuda-samples #398 on local GPU
# Run with: sudo bash reproduce/run.sh
#
# Hardware note: power limit and clock values below are for RTX 4000 SFF Ada.
# On other GPUs, these may be outside valid range — the script continues
# gracefully if nvidia-smi rejects them. Set POWER_LIMIT and CLOCK_MHZ
# env vars to override, or remove the power/clock lines entirely.

set -e

POWER_LIMIT="${POWER_LIMIT:-30}"
CLOCK_MHZ="${CLOCK_MHZ:-210}"

echo "=== GPU Power/Thermal Setup ==="
nvidia-smi -pl "$POWER_LIMIT" 2>/dev/null && echo "Power limit: ${POWER_LIMIT}W" || echo "Could not set power limit (non-fatal)"
nvidia-smi -lgc "$CLOCK_MHZ","$CLOCK_MHZ" 2>/dev/null && echo "Clocks locked: ${CLOCK_MHZ} MHz" || echo "Could not lock clocks (non-fatal)"
echo ""

echo "=== Compiling ==="
cd "$(dirname "$0")"
nvcc -O2 -o reduce7_bug reduce7_bug.cu
echo "Compiled successfully"
echo ""

echo "=== Running Reproduction ==="
./reduce7_bug
echo ""

echo "=== Restoring GPU defaults ==="
nvidia-smi -rgc 2>/dev/null && echo "Clocks restored" || echo "Could not restore clocks"
nvidia-smi -pl 50 2>/dev/null && echo "Power limit restored to 50W" || echo "Could not restore power"
