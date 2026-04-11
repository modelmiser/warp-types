//! # warp-types-overflow
//!
//! Rust arithmetic overflow-freedom lemmas for warp-types verification,
//! packaged as a Lean 4 library.
//!
//! The primary artifact of this crate is the Lean project at `lean/`, not
//! the Rust surface. This `lib.rs` exists so the crate can participate in
//! the warp-types Cargo workspace and be published to crates.io for
//! discoverability, but it contains no code.
//!
//! ## What this crate provides
//!
//! Five verified lemmas covering the overflow-freedom obligation class:
//!
//! - **`add_no_wrap`** — `a.toNat + b.toNat < 2^n → (a + b).toNat = a.toNat + b.toNat`
//! - **`mul_no_wrap`** — `a.toNat * b.toNat < 2^n → (a * b).toNat = a.toNat * b.toNat`
//! - **`sub_no_wrap`** — `b.toNat ≤ a.toNat → (a - b).toNat = a.toNat - b.toNat`
//! - **`add_half_range`** — corollary: two values each less than half the range
//!   cannot overflow when added
//! - **`value_in_range`** — trivial wrapper over `BitVec.isLt`:
//!   `a.toNat < 2^n`
//!
//! These are the obligations a Rust verification plugin would emit for
//! annotations like `#[sol_verify(overflow_free)] fn transfer(...)`.
//!
//! ## Use from a Lean project
//!
//! Add a Lake dependency in your `lakefile.toml`:
//!
//! ```toml
//! [[require]]
//! name = "WarpTypesOverflow"
//! path = "path/to/warp-types/warp-types-overflow/lean"
//! ```
//!
//! Then import in your `.lean` files:
//!
//! ```text
//! import WarpTypesOverflow
//!
//! example (a b : BitVec 32) (h : a.toNat + b.toNat < 2 ^ 32) :
//!     (a + b).toNat = a.toNat + b.toNat :=
//!   WarpTypesOverflow.add_no_wrap h
//! ```
//!
//! ## Why a crate and not a standalone Lake package?
//!
//! This crate is a workspace member of [`warp-types`] so that versioning,
//! licensing, and release cadence stay consistent across the sibling family.
//! The Lean project inside `lean/` is the actual verification artifact.
//!
//! [`warp-types`]: https://crates.io/crates/warp-types
