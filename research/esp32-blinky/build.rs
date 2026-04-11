// The load-bearing part: inject `linkall.x` into the final link step.
// This linker script is provided by esp-hal's build script via a
// `rustc-link-search` path; without it, all peripheral IRQ vector
// symbols (UART2, TIMER1, WDT, ...) are undefined at link time.
// Library build scripts can't add -T flags to their dependents, so
// the *binary* crate has to do it here.
fn main() {
    println!("cargo:rustc-link-arg=-Tlinkall.x");
}
