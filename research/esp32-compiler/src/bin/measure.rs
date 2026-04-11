//! Peak-allocation and wall-clock measurement driver.
//!
//! Installs a custom `#[global_allocator]` wrapping `System` that tracks
//! the delta between a reset point and the peak live-bytes value during
//! a single `compile_function` call. Produces an x86_64 baseline for
//! Experiment 2 of complemented-typestate-framework.md §8.
//!
//! ## Honest limitations (for the paper)
//!
//! * **Peak = request bytes, not allocator slot bytes.** We track
//!   `Layout::size()`, which is what Rust requests; the actual glibc
//!   slot may be larger due to padding/alignment. ESP32 (usually
//!   `embedded-alloc`) has different overhead, so the NUMBER does not
//!   transfer — only the *shape*.
//! * **Stack is not measured.** On ESP32 the stack and heap share SRAM;
//!   a separate max-stack probe is needed for a total-SRAM figure.
//! * **Pointers are 8 bytes on x86_64, 4 bytes on Xtensa LX6 (ESP32).**
//!   Pointer-heavy structures like `Vec<Box<Expr>>` will be
//!   *smaller* on target, so this baseline is a **loose upper bound**
//!   on ESP32 heap footprint — not a lower bound.
//! * **Debug vs release matters.** Run with `--release` for numbers
//!   that match production. Debug has extra debug-info and differs.
//! * **Wall-clock is noisy at microsecond scale.** Each program is run
//!   once; don't over-index on small deltas.
//!
//! ## What the numbers DO tell us
//!
//! The x86 baseline bounds the order of magnitude of ESP32 requirements.
//! If compile_function uses 10 KB here, ESP32 will likely use 4-10 KB
//! (lower pointer cost, possibly same or slightly better allocator).
//! If it used 1 MB here, ESP32 would certainly not fit. The research
//! note's §6 "36 KB peak SRAM" claim is testable against this bound.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use esp32_compiler::compile_function;

// ---------------------------------------------------------------------------
// Custom peak-tracking allocator
// ---------------------------------------------------------------------------

struct PeakAlloc;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static BASELINE: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let new =
                CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            // Peak update via CAS loop (single-threaded measurement driver,
            // so contention is not a concern — CAS is just defensive).
            let mut peak = PEAK.load(Ordering::Relaxed);
            while new > peak {
                match PEAK.compare_exchange_weak(
                    peak,
                    new,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(prev) => peak = prev,
                }
            }
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        System.dealloc(p, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: PeakAlloc = PeakAlloc;

/// Reset the measurement window. Records the CURRENT live bytes as the
/// new BASELINE and resets PEAK := CURRENT. After this call,
/// `peak_delta()` returns the maximum live-bytes rise above the baseline.
fn reset_peak() {
    let c = CURRENT.load(Ordering::Relaxed);
    BASELINE.store(c, Ordering::Relaxed);
    PEAK.store(c, Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
}

fn peak_delta() -> usize {
    let peak = PEAK.load(Ordering::Relaxed);
    let baseline = BASELINE.load(Ordering::Relaxed);
    peak.saturating_sub(baseline)
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Row {
    label: &'static str,
    peak_bytes: usize,
    allocations: usize,
    wall_us: f64,
    code_words: Option<usize>,
    outcome: &'static str,
}

fn measure(label: &'static str, source: &str) -> Row {
    // Warm-up call to stabilize allocator state (first call may do
    // one-time thread-local init that would pollute the peak).
    let _ = compile_function(source);
    reset_peak();

    let start = Instant::now();
    let result = compile_function(source);
    let elapsed = start.elapsed();

    let peak_bytes = peak_delta();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let wall_us = elapsed.as_secs_f64() * 1e6;

    match result {
        Ok(r) => Row {
            label,
            peak_bytes,
            allocations,
            wall_us,
            code_words: Some(r.code.len()),
            outcome: "OK",
        },
        Err(_) => Row {
            label,
            peak_bytes,
            allocations,
            wall_us,
            code_words: None,
            outcome: "rejected (expected)",
        },
    }
}

fn main() {
    // Five representative programs covering: trivial function, constant
    // arithmetic, full PingPong core0, full PingPong core1, and the
    // intentional protocol violation error path (so we see the cost of
    // rejecting a bad program vs. accepting a good one).
    let programs: &[(&'static str, &'static str)] = &[
        (
            "helper (x+1, no protocol)",
            r#"fn helper(x: u32) -> u32 { return x + 1; }"#,
        ),
        (
            "add_three (3+4, no protocol)",
            r#"fn add_three() -> u32 { return 3 + 4; }"#,
        ),
        (
            "pingpong core0 (valid)",
            r#"fn ping(val: u32) : core0 -> u32 {
                send val ch01;
                recv reply ch10;
                return reply;
            }"#,
        ),
        (
            "pingpong core1 (valid)",
            r#"fn pong() : core1 -> u32 {
                recv msg ch01;
                let result = msg + 1;
                send result ch10;
                return result;
            }"#,
        ),
        (
            "bad_ping (protocol violation)",
            r#"fn bad_ping(val: u32) : core0 -> u32 {
                recv reply ch10;
                send val ch01;
                return reply;
            }"#,
        ),
    ];

    println!("# esp32-compiler x86_64 measurement baseline");
    println!();
    println!(
        "**Host:** Linux x86_64, `cargo run --release --bin measure`  "
    );
    println!("**Allocator:** custom `#[global_allocator]` over `System`  ");
    println!(
        "**What's measured:** `Layout::size()` peak delta during a single "
    );
    println!("`compile_function` call, after a warm-up call to stabilize state.");
    println!();
    println!("## Results");
    println!();
    println!(
        "| Program                         | Peak bytes | Allocations | Wall-clock µs | Code words | Outcome              |"
    );
    println!(
        "|---------------------------------|-----------:|------------:|--------------:|-----------:|----------------------|"
    );
    for (label, src) in programs {
        let r = measure(label, src);
        let code = match r.code_words {
            Some(n) => format!("{n}"),
            None => "—".into(),
        };
        println!(
            "| {:31} | {:>10} | {:>11} | {:>13.2} | {:>10} | {:20} |",
            r.label, r.peak_bytes, r.allocations, r.wall_us, code, r.outcome
        );
    }
    println!();
    println!("## Honest limitations");
    println!();
    println!(
        "- Peak is *requested* bytes (`Layout::size()`), not slot-allocated bytes."
    );
    println!(
        "- Stack usage is not measured. ESP32 shares SRAM between stack and heap; a"
    );
    println!(
        "  separate max-stack probe is needed for the full on-target budget."
    );
    println!(
        "- x86_64 pointers are 8 bytes; Xtensa LX6 (ESP32) is 4 bytes. Pointer-heavy"
    );
    println!(
        "  structures (`Vec<Box<Expr>>`) will be *smaller* on target, so these"
    );
    println!(
        "  numbers are a **loose upper bound** on the ESP32 heap footprint."
    );
    println!(
        "- Wall-clock at microsecond scale is noisy; one-shot, not averaged."
    );
    println!(
        "- `compile_function` compiles *one function at a time* — the per-function"
    );
    println!(
        "  bound is what matters for the research note §6 architecture."
    );
}
