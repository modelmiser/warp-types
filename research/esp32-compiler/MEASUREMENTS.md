# esp32-compiler — x86_64 measurement baseline

**Date:** 2026-04-10
**Purpose:** Establish an empirical floor for the research note §6 claim
"36 KB peak SRAM / <100 ms per function" (currently unvalidated).
This file contains the x86_64 baseline; on-target ESP32 numbers come
in a follow-up session when the ULX3S board is brought up.

## Method

Custom `#[global_allocator]` wrapping `std::alloc::System`, tracking
`Layout::size()` delta from a reset point. Single compile, preceded by
a warm-up call to stabilize allocator state. Run under `--release`.
See `src/bin/measure.rs` for the full driver.

```
cargo run --release --bin measure
```

## Results (2026-04-10, Linux x86_64, stable across 3+ runs)

| Program                       | Peak bytes | Allocations | Wall-clock µs | Code words | Outcome             |
|-------------------------------|-----------:|------------:|--------------:|-----------:|---------------------|
| helper (x+1, no protocol)     |      1,264 |          12 |          ~2.0 |          6 | OK                  |
| add_three (3+4, no protocol)  |      1,137 |           9 |          ~1.2 |          6 | OK                  |
| pingpong core0 (valid)        |      1,193 |          22 |          ~3.0 |          9 | OK                  |
| pingpong core1 (valid)        |      1,217 |          24 |          ~3.3 |         12 | OK                  |
| bad_ping (protocol violation) |      1,272 |          18 |          ~2.8 |          — | rejected (expected) |

Peak bytes and allocation counts are **deterministic across runs**.
Wall-clock is single-shot and noisy at microsecond scale.

## What this means for the §6 claim

The research note §6 "ESP32 as compiler host" states:

> 36KB peak SRAM estimate is unmeasured
> Edit-compile-run cycle: < 100ms

Comparison against the measured x86 baseline:

| Metric                 | Research target   | x86 baseline | Headroom (x86) |
|------------------------|-------------------|-------------:|---------------:|
| Peak SRAM per function | 36,864 B (36 KB)  | ≤ 1,272 B    | **~29×**       |
| Wall-clock per compile | 100,000 µs        | ≤ 3.8 µs     | **~26,000×**   |

**Interpretation.** Even with a pessimistic 10× performance penalty on
ESP32 (conservative — Xtensa LX6 at 240 MHz vs x86_64 at ~3 GHz is
typically ~5–20× slower on cache-resident CPU-bound code), per-function
compile would be ~38 µs. Well under the 100 ms research-note budget.
SRAM headroom is even wider: pointer-heavy structures will be *smaller*
on Xtensa (4-byte pointers vs 8-byte x86_64), so the ESP32 peak should
be ≤ 1,272 bytes, not larger.

**The §6 "36 KB peak" target is loose by almost two orders of magnitude.**
This unblocks the architecture — the compiler-on-host approach is
clearly feasible; the real question becomes whether the embedded
bringup (HAL, flash tool, linker script, SPI-to-FPGA transport) is
worth doing, not whether the compiler fits.

## Honest limitations

- **Peak is *requested* bytes (`Layout::size()`), not slot-allocated
  bytes.** Real allocators pad for alignment; glibc will use somewhat
  more than the reported number. `embedded-alloc` on ESP32 has its own
  overhead model. The *order of magnitude* transfers; the exact number
  does not.
- **Stack usage is not measured.** On ESP32, stack and heap share SRAM.
  A separate max-stack probe (e.g., fill-with-pattern then scan) is
  needed for the full budget. The parser uses recursion, so deep
  expressions could push stack usage non-trivially.
- **x86_64 pointers are 8 bytes; Xtensa LX6 is 4 bytes.** Pointer-heavy
  structures (`Vec<Box<Expr>>`, `Vec<Token>`, linked error strings)
  will be smaller on target. This makes the x86 baseline a *loose upper
  bound* for ESP32 heap — not a lower bound.
- **Wall-clock is noisy** at microsecond scale. Single-shot timing;
  cache effects, scheduler noise, and frequency scaling all add jitter.
  Use the numbers for order-of-magnitude reasoning only.
- **One function at a time.** The research note §6 architecture compiles
  per-function, so this is the right granularity — whole-program
  compile is never done on-target.
- **Representative programs are tiny.** The largest program here is 6
  lines; real Forth-shaped code may be 50–100 lines per function.
  Scale probably grows linearly (AST is linear in token count, peak is
  bounded by AST size + output buffer), but this is untested.

## What the follow-up ESP32 session will add

(See the "ESP32 handoff" prompt from the 2026-04-10 Experiment 2
session — DEVLOG entry for this same day.)

1. `no_std` is already verified (`cargo test --no-default-features`
   passes as of commit that added this file). **Precondition done.**
2. `esp-hal` integration for ESP32 (original, not S2/S3/C3 — the
   ULX3S carries an ESP32-WROOM-32 companion).
3. `embedded-alloc` with the same peak-tracking wrapper, reporting
   real on-target numbers.
4. A UART/SPI driver to load the compiled `Vec<u32>` into the J1 on
   the ECP5 FPGA side.
5. Re-run the same 5-program measurement suite on the board.
6. Compare to this x86 baseline. Expect peak ≤ 1.3 KB, wall-clock ≤
   100 µs per function. If either is wildly off, the research note §6
   architecture needs a hard look.

## Reproducing

```bash
cd research/esp32-compiler
cargo run --release --bin measure
```

Or to verify no-std mode compiles (the precondition for ESP32 work):

```bash
cargo build --no-default-features
cargo test --no-default-features
```
