//! # warp-types-invariant
//!
//! State-machine induction combinators for warp-types verification,
//! packaged as a Lean 4 library.
//!
//! The primary artifact of this crate is the Lean project at `lean/`, not
//! the Rust surface. This `lib.rs` exists so the crate can participate in
//! the warp-types Cargo workspace and be published to crates.io for
//! discoverability, but it contains no code.
//!
//! ## What this crate provides
//!
//! Two foundational induction combinators for proving
//! `∀ n, P (step^[n] s₀)`-shaped state-machine invariants:
//!
//! - **`iterate_invariant`** — autonomous step function
//!   (`f : α → α`, conclude `∀ n, P (f^[n] s₀)`).
//! - **`foldl_invariant`** — input-driven step function
//!   (`step : S → I → S`, conclude
//!   `∀ inputs : List I, P (inputs.foldl step s₀)`).
//!
//! Plus two derived corollaries used in hardware-model proofs:
//!
//! - **`iterate_fixpoint`** — if `f s₀ = s₀` then `∀ n, f^[n] s₀ = s₀`
//!   (reset-state fixpoint, common in pipelined RTL).
//! - **`foldl_constant`** — if `P` ignores trajectory, `P s₀ → P (foldl …)`
//!   simplifies to the base case (used by `maskBounded`-style obligations
//!   where the predicate is purely typing-witnessed).
//!
//! ## Use from a Lean project
//!
//! Add a Lake dependency in your `lakefile.toml`:
//!
//! ```toml
//! [[require]]
//! name = "WarpTypesInvariant"
//! path = "path/to/warp-types/warp-types-invariant/lean"
//! ```
//!
//! Then import in your `.lean` files:
//!
//! ```text
//! import WarpTypesInvariant
//!
//! example (s₀ : MyState) (h : reset s₀ = s₀) :
//!     ∀ n, reset^[n] s₀ = s₀ := WarpTypesInvariant.iterate_fixpoint h
//! ```
//!
//! ## Why a crate and not a standalone Lake package?
//!
//! This crate is a workspace member of [`warp-types`] so that versioning,
//! licensing, and release cadence stay consistent across the sibling family.
//! The Lean project inside `lean/` is the actual verification artifact.
//!
//! [`warp-types`]: https://crates.io/crates/warp-types
