//! ESP32 blinky for warp-types research-note §6 Experiment 2.
//!
//! Minimum viable bringup. Verifies:
//!   (1) Xtensa Rust 1.93.0.0 toolchain produces a working image
//!   (2) `espflash flash --monitor` path works end-to-end
//!   (3) The ULX3S passthru-v20-85f bitstream routes FT231X UART
//!       to the ESP32-WROOM UART0 TX pin correctly
//!
//! Success criterion (prints this to UART monitor after flash):
//!
//!     ESP32 blinky: hello from warp-types experiment 2
//!     tick 0
//!     tick 1
//!     ...
//!
//! Once this runs, the next step is to bring up `esp32-compiler` on
//! this same hardware and run the 5-program measurement suite (that's
//! Experiment 2 proper; this file is the precondition).

#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{main, time::Instant};

esp_bootloader_esp_idf::esp_app_desc!();

#[main]
fn main() -> ! {
    let _peripherals = esp_hal::init(esp_hal::Config::default());

    esp_println::println!("ESP32 blinky: hello from warp-types experiment 2");

    let mut tick: u32 = 0;
    loop {
        esp_println::println!("tick {}", tick);
        tick = tick.wrapping_add(1);

        // Busy-wait 1 second — sleep/timer APIs are out-of-scope for
        // a bringup-only blinky. Real measurement code (esp32-compiler
        // port) will use proper timer peripherals.
        let start = Instant::now();
        while start.elapsed().as_millis() < 1000 {}
    }
}
