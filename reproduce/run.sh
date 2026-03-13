#!/bin/bash
# Reproduce cuda-samples #398 on local GPU
# Run with: sudo bash reproduce/run.sh

set -e

echo "=== GPU Power/Thermal Setup ==="
nvidia-smi -pl 30 2>/dev/null && echo "Power limit: 30W" || echo "Could not set power limit"
nvidia-smi -lgc 210,210 2>/dev/null && echo "Clocks locked: 210 MHz" || echo "Could not lock clocks"
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
