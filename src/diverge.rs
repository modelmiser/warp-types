//! Diverge operations: splitting a warp by predicate.
//!
//! Divergence is the first half of the warp typestate pattern.
//! When a warp diverges, it splits into two sub-warps with complementary
//! active sets. The type system tracks exactly which lanes are active.
//!
//! # Compile-Time Safety
//!
//! After divergence, `Warp<Even>` cannot call `shuffle_xor` — the method
//! simply doesn't exist on that type. This prevents the most common class
//! of GPU divergence bugs.
//!
//! ## Bug: Shuffle after diverge (caught at compile time)
//!
//! ```compile_fail
//! use warp_types::*;
//!
//! fn buggy_shuffle() {
//!     let warp: Warp<All> = Warp::kernel_entry();
//!     let (evens, _odds) = warp.diverge_even_odd();
//!     let data = data::PerLane::new(42i32);
//!     // BUG: shuffle_xor doesn't exist on Warp<Even>
//!     let _ = evens.shuffle_xor(data, 1);
//! }
//! ```

use crate::warp::Warp;
use crate::active_set::*;

// ============================================================================
// Diverge methods on Warp<All>
// ============================================================================

impl Warp<All> {
    /// Diverge into even/odd lanes.
    ///
    /// Returns two warps with COMPLEMENTARY active sets.
    /// The type system tracks that Even and Odd together make All.
    pub fn diverge_even_odd(self) -> (Warp<Even>, Warp<Odd>) {
        (Warp::new(), Warp::new())
    }

    /// Diverge into low/high halves.
    pub fn diverge_halves(self) -> (Warp<LowHalf>, Warp<HighHalf>) {
        (Warp::new(), Warp::new())
    }

    /// Extract lane 0 (e.g., for reduction result handling).
    pub fn extract_lane0(self) -> (Warp<Lane0>, Warp<NotLane0>) {
        (Warp::new(), Warp::new())
    }
}

// ============================================================================
// Nested diverge: further split already-diverged warps
// ============================================================================

impl Warp<Even> {
    /// Split even lanes into low-even and high-even.
    ///
    /// Even ∩ LowHalf = lanes 0, 2, 4, 6, 8, 10, 12, 14
    /// Even ∩ HighHalf = lanes 16, 18, 20, 22, 24, 26, 28, 30
    pub fn diverge_halves(self) -> (Warp<EvenLow>, Warp<EvenHigh>) {
        (Warp::new(), Warp::new())
    }
}

impl Warp<Odd> {
    /// Split odd lanes into low-odd and high-odd.
    pub fn diverge_halves(self) -> (Warp<OddLow>, Warp<OddHigh>) {
        (Warp::new(), Warp::new())
    }
}

impl Warp<LowHalf> {
    /// Split low-half lanes into even-low and odd-low.
    pub fn diverge_even_odd(self) -> (Warp<EvenLow>, Warp<OddLow>) {
        (Warp::new(), Warp::new())
    }
}

impl Warp<HighHalf> {
    /// Split high-half lanes into even-high and odd-high.
    pub fn diverge_even_odd(self) -> (Warp<EvenHigh>, Warp<OddHigh>) {
        (Warp::new(), Warp::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diverge_even_odd() {
        let all: Warp<All> = Warp::new();
        let (evens, odds) = all.diverge_even_odd();
        assert_eq!(evens.active_set_name(), "Even");
        assert_eq!(odds.active_set_name(), "Odd");
        assert_eq!(evens.population(), 16);
        assert_eq!(odds.population(), 16);
    }

    #[test]
    fn test_diverge_halves() {
        let all: Warp<All> = Warp::new();
        let (low, high) = all.diverge_halves();
        assert_eq!(low.active_mask(), 0x0000FFFF);
        assert_eq!(high.active_mask(), 0xFFFF0000);
    }

    #[test]
    fn test_extract_lane0() {
        let all: Warp<All> = Warp::new();
        let (lane0, rest) = all.extract_lane0();
        assert_eq!(lane0.population(), 1);
        assert_eq!(rest.population(), 31);
    }

    #[test]
    fn test_nested_diverge_even() {
        let all: Warp<All> = Warp::new();
        let (evens, _odds) = all.diverge_even_odd();
        let (even_low, even_high) = evens.diverge_halves();
        assert_eq!(even_low.active_mask(), 0x00005555);
        assert_eq!(even_high.active_mask(), 0x55550000);
        assert_eq!(even_low.population(), 8);
        assert_eq!(even_high.population(), 8);
    }

    #[test]
    fn test_nested_diverge_odd() {
        let all: Warp<All> = Warp::new();
        let (_evens, odds) = all.diverge_even_odd();
        let (odd_low, odd_high) = odds.diverge_halves();
        assert_eq!(odd_low.active_mask(), 0x0000AAAA);
        assert_eq!(odd_high.active_mask(), 0xAAAA0000);
    }

    #[test]
    fn test_path_independence() {
        // EvenLow can be reached via Even→halves or LowHalf→even_odd
        let all1: Warp<All> = Warp::new();
        let (evens, _) = all1.diverge_even_odd();
        let (via_even, _) = evens.diverge_halves();

        let all2: Warp<All> = Warp::new();
        let (low, _) = all2.diverge_halves();
        let (via_low, _) = low.diverge_even_odd();

        // Same type, same mask
        assert_eq!(via_even.active_mask(), via_low.active_mask());
    }
}
