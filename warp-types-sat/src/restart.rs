//! Luby restart policy for CDCL solvers.
//!
//! Periodic restarts prevent the solver from getting trapped in unproductive
//! subtrees. The Luby sequence provides a universally optimal restart strategy
//! (Luby, Sinclair & Zuckerman, 1993).
//!
//! Sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
//! Each value is multiplied by a base interval (default: 100 conflicts).

/// Luby restart policy.
pub struct LubyRestarts {
    /// Conflicts since last restart.
    conflicts_since: u64,
    /// Current Luby sequence index (1-indexed).
    index: u32,
    /// Base interval multiplier.
    base: u64,
}

impl LubyRestarts {
    /// Create a new Luby restart policy.
    ///
    /// `base` is multiplied by the Luby sequence value to get the conflict
    /// interval. Standard values: 100 (MiniSat), 512 (Glucose).
    pub fn new(base: u64) -> Self {
        LubyRestarts {
            conflicts_since: 0,
            index: 1,
            base,
        }
    }

    /// Record a conflict. Returns true if a restart should happen now.
    pub fn on_conflict(&mut self) -> bool {
        self.conflicts_since += 1;
        let limit = self.base * luby(self.index);
        if self.conflicts_since >= limit {
            self.conflicts_since = 0;
            self.index += 1;
            true
        } else {
            false
        }
    }
}

/// Compute the i-th Luby sequence value (1-indexed).
///
/// Recursive definition:
/// - If i = 2^k - 1 for some k, return 2^(k-1)
/// - Otherwise, strip the largest complete block and recurse
fn luby(i: u32) -> u64 {
    debug_assert!(i >= 1, "luby sequence is 1-indexed");
    let mut k = 1u32;
    while (1u64 << k) - 1 < i as u64 {
        k += 1;
    }
    if (1u64 << k) - 1 == i as u64 {
        1u64 << (k - 1)
    } else {
        luby(i - ((1u64 << (k - 1)) - 1) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luby_sequence_first_15() {
        let expected = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(luby(i as u32 + 1), e, "luby({}) should be {}", i + 1, e);
        }
    }

    #[test]
    fn restart_fires_at_base() {
        let mut r = LubyRestarts::new(10);
        // First Luby value is 1, so restart at 10 conflicts.
        for _ in 0..9 {
            assert!(!r.on_conflict());
        }
        assert!(r.on_conflict());
    }

    #[test]
    fn restart_sequence_intervals() {
        let mut r = LubyRestarts::new(1);
        // Luby: 1, 1, 2, 1, 1, 2, 4
        let expected_intervals = [1, 1, 2, 1, 1, 2, 4];
        for &interval in &expected_intervals {
            for _ in 0..interval - 1 {
                assert!(!r.on_conflict());
            }
            assert!(r.on_conflict());
        }
    }

    #[test]
    fn large_index() {
        // Luby(15) = 8, Luby(16) = 1 (start of new block)
        assert_eq!(luby(15), 8);
        assert_eq!(luby(16), 1);
        assert_eq!(luby(31), 16);
    }
}
