//! Soundness proof sketch — see [`crate::proof`] for the canonical version.
//!
//! This module previously contained a duplicate of `src/proof.rs`.
//! It now re-exports from the canonical location to avoid divergence.

// Re-export from the canonical proof module when it's compiled.
// (proof is gated behind #[cfg(any(test, feature = "formal-proof"))])
