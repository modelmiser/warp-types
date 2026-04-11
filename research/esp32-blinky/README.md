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

Flash image after `espflash save-image`: 77,888 bytes (0.47% of 16 MB).
Factory flash is actually 16 MB on this board (manufacturer `0xa1`,
device `0x4018`) — not 4 MB as originally assumed. Factory backup
therefore needs `esptool read_flash 0 0x1000000`, not `0x400000`.

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
- ✓ Phase B (passthru bitstream + esptool sync) — **UNBLOCKED 2026-04-11**
- ✓ Phase C code (Cargo.toml, build.rs, main.rs, config) — complete
- ✓ `cargo build --release` succeeds
- ⏸ Phase C flash + verify UART output — pending factory flash backup
  + `cargo run --release`

### Phase B resolution

The working bitstream is built from `emard/ulx3s-passthru`
`rtl/ulx3s_v20_passthru_wifi.vhd` with three modifications:

1. Strip `use work.f32c_pack.all;` (vestigial, unused)
2. Strip `library ecp5u; use ecp5u.components.all;` (vestigial, unused)
3. Pass `-fsynopsys -fexplicit` to `ghdl` (for `std_logic_unsigned` +
   overloaded `=` operator resolution)

Build recipe (oss-cad-suite must be sourced):

```sh
yosys -m ghdl -p "ghdl -fsynopsys -fexplicit ulx3s_v20_passthru_wifi.vhd \
                       -e ulx3s_passthru_wifi; \
                  synth_ecp5 -top ulx3s_passthru_wifi -json passthru.json"
nextpnr-ecp5 --85k --package CABGA381 --lpf-allow-unconstrained \
  --json passthru.json --lpf ulx3s_v20.lpf --textcfg passthru.config
ecppack --compress passthru.config passthru.bit
openFPGALoader --board ulx3s passthru.bit
```

Output: ~281 KB bitstream, 52 cells total (13 CCU2C + 5 LUT4 + 34
TRELLIS_FF), clean timing margin, 2 benign unconstrained-IO warnings
for unused `wifi_gpio2` and `ftdi_ndsr`.

**Critical gotcha that blocked the previous session:** the ulx3s-bin
prebuilt 85F passthru bitstreams (both Nov 2018 and Feb 2019 builds)
produce *identical* `Invalid head of packet (0x08)` failure mode. So
does a freshly built one. The bitstream was never the bug. **The real
blocker was ESP32 persistent state** — the ESP32 had been running some
firmware that didn't respond to soft reset (DTR/RTS → EN toggle). Only
a physical USB cable unplug-and-replug (which hard-power-cycles the
board) cleared it. After replug, esptool syncs on the 6th SYNC attempt
(first 5 timeout while the boot ROM is warming up) and flash_id
completes cleanly.

**If you see `Invalid head of packet (0x08)` on ESP32 bringup:**
always try a physical USB replug first, before investigating
bitstream variants. Soft reset does not clear stuck firmware.

## Next (Experiment 2 proper)

Once this blinky is running on hardware, port `research/esp32-compiler/`
to `xtensa-esp32-none-elf` using `embedded-alloc::Heap` with the same
peak-tracking wrapper as `research/esp32-compiler/src/bin/measure.rs`.
Re-run the 5-program measurement suite on-target and append the
results to `research/esp32-compiler/MEASUREMENTS.md` under a new
"On-target results (ULX3S ESP32-WROOM-32)" section. Compare against
the x86 baseline (peak ≤ 1,272 B / ≤ 3.8 µs from the 2026-04-10 run).
