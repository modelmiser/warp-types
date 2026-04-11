# esp32-blinky

Minimum-viable ESP32-WROOM bringup for the warp-types research note
§6 "ESP32 as Compiler Host" — **Experiment 2** precondition.

This crate does nothing interesting on its own: it prints a UART
heartbeat once per second. Its purpose is to prove the toolchain +
flash + UART path before porting `research/esp32-compiler/` to run
on the same board and collect the on-target measurement baseline
for MEASUREMENTS.md.

## Target hardware

- **Board:** ULX3S v3.0.8 (LFE5U-85F ECP5 + ESP32-WROOM-32 companion)
- **ESP32 access:** through a ULX3S passthrough FPGA bitstream
  (`ulx3s_bin/fpga/passthru/passthru-v20-85f/ulx3s_85f_passthru.bit`)
  loaded to FPGA SRAM via `fujprog -j sram`. The FT231X USB UART
  is bridged to ESP32 UART0 by that bitstream.
- **USB:** `/dev/ttyUSB0` (requires the `80-fpga-ulx3s.rules` udev
  rule to be re-applied with `--action=add` after install; see
  "Gotchas" below).

## Toolchain

- **Rust:** Xtensa fork `1.93.0-nightly (2b43689c5 2026-01-27) (1.93.0.0)`
- **Installer:** `espup v0.16.0` (installed via `cargo install espup --locked`)
- **Install command:** `espup install --targets esp32`
- **Shell env:** `. ~/export-esp.sh` must be sourced in every shell
  before `cargo build`
- **Flasher:** `espflash v4.3.0` (`cargo install espflash --locked`)
- **Read-flash / backup:** `esptool v5.2.0` from a userspace venv at
  `/home/cjb/Claude/.venv-esp-tools/bin/esptool`

## Dependency pinning (all from crates.io)

| Crate                  | Version | Why                               |
|------------------------|---------|-----------------------------------|
| esp-hal                | 1.0.0   | First stable 1.0 of the HAL       |
| esp-backtrace          | 0.18.1  | panic handler + exception vectors |
| esp-println            | 0.16.1  | UART `println!` macro             |
| esp-bootloader-esp-idf | 0.4.0   | `esp_app_desc!` macro for boot    |
| critical-section       | 1.2     | no longer bundled by esp-hal 1.0  |

## Build + flash

```sh
. ~/export-esp.sh
cd research/esp32-blinky
cargo build --release

# Load passthrough bitstream first (one-time per power cycle):
fujprog -j sram path/to/ulx3s_85f_passthru.bit

# Then flash the blinky:
cargo run --release   # runs espflash flash --monitor via .cargo/config.toml runner
```

Expected UART monitor output:

```
ESP32 blinky: hello from warp-types experiment 2
tick 0
tick 1
tick 2
...
```

## Binary footprint (as of first build, esp-hal 1.0.0)

```
   text    data     bss     dec     hex
  22325    1228  195380  218933   35735
```

Flash image after `espflash save-image`: 77,888 bytes (1.89% of 4 MB).

## Gotchas (load-bearing; removing any of these breaks the build or link)

### 1. `[workspace]` stanza in `Cargo.toml`

Makes this crate a **standalone mini-workspace** so the top-level
`warp-types` workspace doesn't pull it in (which would force stable
Rust onto this crate). Same pattern as `research/esp32-compiler/`.

### 2. `[unstable] build-std = ["core", "alloc"]` in `.cargo/config.toml`

The esp rustup fork ships `rustc` + `rust-src` but **no precompiled
`rust-std` for xtensa-esp32-none-elf** (the manifest only has
`x86_64-unknown-linux-gnu`). `core` and `alloc` are built from source
on demand. Without this stanza you get `E0463: can't find crate for 'core'`
and the error message misleadingly suggests `rustup target add` which
does not work for Xtensa targets.

### 3. `rustflags = ["-C", "link-arg=-nostartfiles"]` in `.cargo/config.toml`

Prevents `xtensa-esp-elf-gcc`'s own `crt*` startup files from clashing
with `xtensa-lx-rt`'s vector table and `_start` entry point.

### 4. `build.rs` with `println!("cargo:rustc-link-arg=-Tlinkall.x");`

**The load-bearing one.** `linkall.x` is the linker script that
esp-hal's build script places in the link search path; it resolves
all the peripheral IRQ vector symbols (`UART2`, `TIMER1`, `WDT`, ...).
Without this line in `build.rs`, you get ~80 "undefined reference"
errors at link time with no hint that a linker script is missing.
Cargo library build scripts cannot inject link args into their
dependents, so the *binary* crate has to do it.

### 5. Explicit `critical-section = "1.2"` dependency

esp-hal 1.0 no longer bundles a critical-section implementation.
Binary crates must provide one (esp-hal's peripherals will link to
it). Without this dep the build fails at link time.

### 6. udev rule `--action=add` re-trigger

The `/etc/udev/rules.d/80-fpga-ulx3s.rules` rule sets
`MODE="0666"`, but `MODE=`, `OWNER=`, `GROUP=` are **only applied on
`add` events, not `change` events** (systemd-udev docs). After
editing or installing the rule, re-trigger with:

```sh
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=tty
```

Without `--action=add`, the mode stays at whatever the original
`add` event set it to (typically `0664`), and the `test -w
/dev/ttyUSB0` check fails silently.

### 7. `openFPGALoader --board ulx3s` (not the default cable)

openFPGALoader's default cable is `ft2232` (the dual-channel FT2232H
used by Digilent boards). The ULX3S has a **single-channel FT231X**
(`0403:6015`). Running `openFPGALoader --detect` without
`--board ulx3s` (or `--cable ft231X`) reports misleading "device
not found" errors even though the board is plugged in and
enumerated.

## Status

- ✓ Phase A (toolchain + host access) — complete
- ✓ Phase C code (Cargo.toml, build.rs, main.rs, config) — complete
- ✓ `cargo build --release` succeeds
- ⚠ Phase B — **BLOCKED**: the passthru bitstream from
  `ulx3s-bin/fpga/passthru/passthru-v20-85f/ulx3s_85f_passthru.bit`
  does not route the FT231X UART to the ESP32's UART0 — a raw `cat
  /dev/ttyUSB0` at 115200 captures zero bytes over 2 seconds, and
  `esptool` sync fails with `Invalid head of packet (0x08)` in every
  reset/baud combination tried. Probably the `_serial2` RTL variant,
  not the `_wifi` variant needed for `esptool`. See DEVLOG.md
  "2026-04-10 — esp32-blinky Phase B BLOCKED" for the full diagnostic
  trail and handoff prompt, and INSIGHTS.md #N+38 for the passthru
  variant taxonomy.
- ⏸ Phase C flash + verify UART output — blocked on Phase B

### Recommended path to unblock

Compile `ulx3s_v20_passthru_wifi.vhd` from emard/ulx3s-passthru source
against `constraints/ulx3s_v20.lpf` using ghdl + yosys + nextpnr-ecp5
+ project-trellis. Or: use a different ESP32-WROOM-32 dev board with
native USB-UART for the on-target measurement baseline; Experiment 2
doesn't strictly require ULX3S FPGA adjacency (only Experiment 3
does, for the SPI→FPGA roundtrip).

## Next (Experiment 2 proper)

Once this blinky is running on hardware, port `research/esp32-compiler/`
to `xtensa-esp32-none-elf` using `embedded-alloc::Heap` with the same
peak-tracking wrapper as `research/esp32-compiler/src/bin/measure.rs`.
Re-run the 5-program measurement suite on-target and append the
results to `research/esp32-compiler/MEASUREMENTS.md` under a new
"On-target results (ULX3S ESP32-WROOM-32)" section. Compare against
the x86 baseline (peak ≤ 1,272 B / ≤ 3.8 µs from the 2026-04-10 run).
