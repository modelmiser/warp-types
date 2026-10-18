#!/usr/bin/env bash
# Guard: the membermask and width operands the typed API promises actually reach
# the emitted `shfl.sync` instructions. Builds tools/ptxprobe for the device
# target (no GPU needed; the nvptx CI job already has the target) and greps the
# PTX. Found by the 2026-10-18 unsafe audit: the "membermask reaches shfl.sync"
# half of src/shuffle.rs's membermask test was marked UNTESTED-HARD for months
# when it was a ten-second grep.
#
# Operand values: Even::MASK 0x55555555 = 1431655765; Odd::MASK 0xAAAAAAAA =
# -1431655766 as a PTX b32 immediate; Tile<8> down c = ((32-8)<<8)|7 = 6151.
# Immediates go through a register (`in(reg32)`), so we look for the `mov.b32`
# that feeds the operand, not for the shfl line alone.
set -euo pipefail
cd "$(dirname "$0")/.."
out="$PWD/target/ptxprobe-check"          # under target/ (gitignored, on disk: /tmp is a RAM tmpfs here)
( cd tools/ptxprobe && CARGO_TARGET_DIR="$out" cargo rustc --release --quiet \
    --target nvptx64-nvidia-cuda -- --emit=asm >/dev/null )
ptx=$(find "$out" -name 'ptxprobe-*.s' -newer tools/ptxprobe/src/lib.rs | head -1)
[ -n "$ptx" ] || ptx=$(find "$out" -name 'ptxprobe-*.s' | head -1)
[ -n "$ptx" ] || { echo "no PTX emitted"; exit 1; }
fail=0
section() { awk -v f="$1" '$0 ~ "\\.entry " f {p=1} p {print} p && /^}/ {exit}' "$ptx"; }

check() {  # name, regex, description
  if section "$1" | grep -qE "$2"; then echo "ok   $1: $3"; else echo "FAIL $1: $3"; fail=1; fi
}
# membermask reaches the bfly for both sub-warps
check probe_within_even 'mov\.b32[[:space:]]+%r[0-9]+, 1431655765;' 'Even::MASK feeds an operand'
check probe_within_even 'shfl\.sync\.bfly\.b32' 'a shfl.sync.bfly is emitted'
check probe_within_odd  'mov\.b32[[:space:]]+%r[0-9]+, -1431655766;' 'Odd::MASK feeds an operand'
# the lane guard: the within-shuffle must be under a predicated branch on %laneid
check probe_within_even 'mov\.u32[[:space:]]+%r[0-9]+, %laneid;' 'lane_id is read (guard present)'
check probe_within_even '@%p[0-9]+ bra' 'the shuffle sits behind a branch'
# tiles: full-warp membermask, width-confined c
check probe_tile8_down  'mov\.b32[[:space:]]+%r[0-9]+, 6151;' 'Tile<8> down c operand = 0x1807'
check probe_tile8_down  'mov\.b32[[:space:]]+%r[0-9]+, -1;' 'Tile shuffle keeps the full-warp membermask'
check probe_tile8_down  'shfl\.sync\.down\.b32' 'a shfl.sync.down is emitted'
# 64-bit: two b32 shuffles, c = 0 for up
n=$(section probe_up_i64 | grep -cE 'shfl\.sync\.up\.b32' || true)
if [ "$n" = 2 ]; then echo "ok   probe_up_i64: two shfl.sync.up.b32 (lo/hi)"; else echo "FAIL probe_up_i64: expected 2 shfl.sync.up.b32, got $n"; fail=1; fi
check probe_up_i64 'shfl\.sync\.up\.b32[[:space:]]+%r[0-9]+, %r[0-9]+, %r[0-9]+, 0, %r[0-9]+;' 'up uses c = 0'
if [ $fail = 0 ]; then echo "ptx operands OK"; else echo "PTX OPERAND CHECK FAILED"; exit 1; fi
