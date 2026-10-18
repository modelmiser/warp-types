#!/usr/bin/env bash
# Guard: the membermask and width operands the typed API promises actually reach
# the emitted `shfl.sync` instructions, and `Warp<Empty>::shuffle_xor_within` is
# rejected by the DEVICE build. Builds tools/ptxprobe for nvptx64 (no GPU needed;
# the nvptx CI job already has the target) and reads the PTX.
#
# What this establishes, and what it does not. For each probe it finds the
# `shfl.sync.*` line, reads the REGISTER in its membermask (5th) operand and
# requires that register to be loaded with the expected constant in the same
# entry (`mov.b32 %rM, <mask>`): operand dataflow for the mask, one hop. For the
# lane guard it requires a `%laneid` read and a predicated branch in the entry.
# It does not prove the shfl is dominated by the guard's taken edge; kernel 6 of
# examples/gpu-project checks the lane values on hardware.
#
# Found by the 2026-10-18 unsafe audit: the "membermask reaches shfl.sync" half
# of src/shuffle.rs's membermask test was marked UNTESTED-HARD for months when it
# was a grep. The 2026-10-18 panel then refuted this script's first version: it
# searched for `ptxprobe-*.s` and took the first hit, the stale-artifact bug the
# builder fix was about. Now rustc writes the asm to a path we name (`--emit asm=`).
#
# Operand values: Even::MASK 0x55555555 = 1431655765; Odd::MASK 0xAAAAAAAA =
# -1431655766 as a PTX b32 immediate; Tile<8> down c = ((32-8)<<8)|7 = 6151.
set -euo pipefail
cd "$(dirname "$0")/.."
out="$PWD/target/ptxprobe-check"          # under target/ (gitignored, on disk: /tmp is a RAM tmpfs here)
mkdir -p "$out"
ptx="$out/probe.s"
rm -f "$ptx"
# A cached build re-runs nothing and writes nothing (the first pre-push with this
# script failed with "no PTX emitted" for exactly that reason), so evict the probe.
( cd tools/ptxprobe && CARGO_TARGET_DIR="$out/target" cargo clean --quiet -p ptxprobe --release \
    --target nvptx64-nvidia-cuda 2>/dev/null || true )
ir="$out/probe.ll"
rm -f "$ir"
( cd tools/ptxprobe && CARGO_TARGET_DIR="$out/target" cargo rustc --release --quiet \
    --target nvptx64-nvidia-cuda -- --emit "asm=$ptx,llvm-ir=$ir" >/dev/null )
[ -s "$ptx" ] || { echo "no PTX emitted at $ptx"; exit 1; }
[ -s "$ir" ] || { echo "no LLVM IR emitted at $ir"; exit 1; }
fail=0
section() { awk -v f="$1" '$0 ~ "\\.entry " f {p=1} p {print} p && /^}/ {exit}' "$ptx"; }

check() {  # name, regex, description
  if section "$1" | grep -qE "$2"; then echo "ok   $1: $3"; else echo "FAIL $1: $3"; fail=1; fi
}
# membermask dataflow: the shfl's 5th operand register must be loaded with the constant
mask_check() {  # entry, shfl-kind, expected immediate, description
  local sec reg
  sec=$(section "$1")
  reg=$(echo "$sec" | grep -oE "shfl\.sync\.$2\.b32[[:space:]]+%r[0-9]+(\|%p[0-9]+)?,[[:space:]]*%r[0-9]+,[[:space:]]*(%r[0-9]+|-?[0-9]+),[[:space:]]*(%r[0-9]+|-?[0-9]+),[[:space:]]*(%r[0-9]+|-?[0-9]+);" \
        | head -1 | sed -E 's/.*,[[:space:]]*(%r[0-9]+|-?[0-9]+);$/\1/')
  if [ -z "$reg" ]; then echo "FAIL $1: no shfl.sync.$2 found"; fail=1; return; fi
  if [ "$reg" = "$3" ]; then
    echo "ok   $1: membermask immediate $3 ($4)"
  elif echo "$sec" | grep -qE "mov\.b32[[:space:]]+\\$reg, $3;"; then
    echo "ok   $1: membermask $reg = $3 ($4)"
  else
    echo "FAIL $1: membermask operand $reg is neither $3 nor a register loaded with it ($4)"; fail=1
  fi
}
mask_check probe_within_even bfly 1431655765 'Even::MASK feeds the bfly'
mask_check probe_within_odd  bfly -1431655766 'Odd::MASK feeds the bfly'
mask_check probe_tile8_down  down -1 'Tile shuffle keeps the full-warp membermask'
# the lane guard: %laneid read and a predicated branch in the within entries
check probe_within_even 'mov\.u32[[:space:]]+%r[0-9]+, %laneid;' 'lane_id is read (guard present)'
check probe_within_even '@%p[0-9]+ bra' 'a predicated branch is present (guard)'
check probe_within_odd  'mov\.u32[[:space:]]+%r[0-9]+, %laneid;' 'lane_id is read (guard present)'
# tiles: width-confined c
check probe_tile8_down  'shfl\.sync\.down\.b32[[:space:]]+%r[0-9]+, %r[0-9]+, (%r[0-9]+|1), (6151|%r[0-9]+), (%r[0-9]+|-1);' 'Tile<8> down carries c = 0x1807'
if ! section probe_tile8_down | grep -qE "6151"; then echo "FAIL probe_tile8_down: c = 6151 absent"; fail=1; fi
# 64-bit: two b32 shuffles, c = 0 for up
n=$(section probe_up_i64 | grep -cE 'shfl\.sync\.up\.b32' || true)
if [ "$n" = 2 ]; then echo "ok   probe_up_i64: two shfl.sync.up.b32 (lo/hi)"; else echo "FAIL probe_up_i64: expected 2 shfl.sync.up.b32, got $n"; fail=1; fi
check probe_up_i64 'shfl\.sync\.up\.b32[[:space:]]+%r[0-9]+, %r[0-9]+, (%r[0-9]+|[0-9]+), 0, (%r[0-9]+|-1);' 'up uses c = 0'

# Link completeness: a non-generic device function that is not inlined shows up
# as `.extern .func` and fails PTX JIT with CUDA_ERROR_INVALID_PTX (2026-10-18:
# `Warp<All>::ballot` after its body stopped being inline asm). No GPU needed.
if grep -qE '^\s*\.extern\s+\.func' "$ptx"; then
  echo "FAIL ptx: unresolved .extern .func — a device function was not inlined:"; grep -E '^\s*\.extern\s+\.func' "$ptx" | head -3; fail=1
else
  echo "ok   ptx: no .extern .func (every device function inlined)"
fi
check probe_ballot_and_wrapping 'vote\.sync\.ballot\.b32' 'ballot lowers to vote.sync.ballot'
if command -v ptxas >/dev/null; then
  if ptxas -arch=sm_70 "$ptx" -o /dev/null 2>"$out/ptxas.log"; then echo "ok   ptxas: assembles for sm_70"; else echo "FAIL ptxas:"; head -3 "$out/ptxas.log"; fail=1; fi
fi

# Convergence: every warp collective must be an llvm.nvvm intrinsic call that LLVM
# declares `convergent`. Inline asm is not convergent, so an asm `shfl`/`vote`/
# `bar.warp` could legally be duplicated into divergent control flow.
if grep -qE 'asm sideeffect[^"]*"[^"]*(shfl\.sync|vote\.|bar\.warp)' "$ir"; then
  echo "FAIL ir: a warp collective is still inline asm (not convergent)"; fail=1
else
  echo "ok   ir: no warp collective is inline asm"
fi
nconv=0
for d in $(grep -oE '^declare [^@]*@llvm\.nvvm\.(shfl\.sync|vote\.ballot\.sync|bar\.warp\.sync)[^ ]*' "$ir" | grep -oE '@llvm[^ (]+' | sort -u); do
  grp=$(grep -E "^declare [^@]*$(printf '%s' "$d" | sed 's/[.]/\\./g')\(" "$ir" | grep -oE '#[0-9]+$' | head -1)
  if [ -n "$grp" ] && grep -E "^attributes $grp " "$ir" | grep -q convergent; then
    echo "ok   ir: $d is convergent"; nconv=$((nconv+1))
  else
    echo "FAIL ir: $d is not declared convergent (attrs $grp)"; fail=1
  fi
done
if [ "$nconv" -lt 2 ]; then echo "FAIL ir: expected at least the bfly and down intrinsics, found $nconv"; fail=1; fi

# Warp<Empty>::shuffle_xor_within must FAIL the device build (post-monomorphization
# E0080 from the inline const assert). This is the nvptx half the host compile_fail
# doctest cannot give; the probe kernel is reached from a codegen root (an .entry).
elog="$out/empty-probe.log"
if ( cd tools/ptxprobe && CARGO_TARGET_DIR="$out/target" cargo rustc --release --quiet \
      --features empty-probe --target nvptx64-nvidia-cuda -- --emit "asm=$out/empty.s" ) >"$elog" 2>&1; then
  echo "FAIL empty-probe: Warp<Empty>::shuffle_xor_within COMPILED for nvptx64"; fail=1
elif grep -qE "E0080|has no lanes to shuffle" "$elog"; then
  echo "ok   empty-probe: Warp<Empty>::shuffle_xor_within rejected on nvptx64 (E0080)"
else
  echo "FAIL empty-probe: device build failed for another reason:"; tail -5 "$elog"; fail=1
fi
# A type alias to a slice behind a raw pointer must fail the macro's const size
# assertion on the device build (the syntactic thin-pointee check cannot see
# through the alias; 2026-10-18 panel).
alog="$out/alias-probe.log"
if ( cd tools/ptxprobe && CARGO_TARGET_DIR="$out/target" cargo rustc --release --quiet \
      --features alias-probe --target nvptx64-nvidia-cuda -- --emit "asm=$out/alias.s" ) >"$alog" 2>&1; then
  echo "FAIL alias-probe: #[warp_kernel] accepted a fat pointer behind a type alias"; fail=1
elif grep -qE "fat pointer" "$alog"; then
  echo "ok   alias-probe: fat pointer behind a type alias rejected by #[warp_kernel]"
else
  echo "FAIL alias-probe: device build failed for another reason:"; tail -5 "$alog"; fail=1
fi
if [ $fail = 0 ]; then echo "ptx operands OK"; else echo "PTX OPERAND CHECK FAILED"; exit 1; fi
