//! # warp-types-bitwise
//!
//! Bitvector tautology and mask-algebra lemmas for warp-types verification,
//! packaged as a Lean 4 library.
//!
//! The primary artifact of this crate is the Lean project at `lean/`, not
//! the Rust surface. This `lib.rs` exists so the crate can participate in
//! the warp-types Cargo workspace and be published to crates.io for
//! discoverability, but it contains no code.
//!
//! ## What this crate provides
//!
//! 14 verified lemmas covering the bitwise-tautology obligation class:
//!
//! - **Rust-shaped** (5): `mask_idempotent`, `disjoint_masks`,
//!   `field_insert_read`, `counter_mask_valid`, `disjoint_update`
//! - **CUDA-shaped** (6): `ballot_split`, `all_sync_split`,
//!   `any_sync_monotone`, `mask_produces_subset`, `child_within_parent`,
//!   `syncwarp_safe`
//! - **Verilog-shaped** (3): `else_complement`, `else_disjoint_from_taken`,
//!   `rtl_else_xor`
//!
//! Three of these lemmas (`ballot_split`, `all_sync_split`,
//! `any_sync_monotone`) are fold-algebra theorems that may relocate to
//! `warp-types-ballot` when that sibling crate is built.
//!
//! ## Use from a Lean project
//!
//! Add a Lake dependency in your `lakefile.toml`:
//!
//! ```toml
//! [[require]]
//! name = "WarpTypesBitwise"
//! path = "path/to/warp-types/warp-types-bitwise/lean"
//! ```
//!
//! Then import in your `.lean` files:
//!
//! ```text
//! import WarpTypesBitwise
//! ```
//!
//! ## Why a crate and not a standalone Lake package?
//!
//! This crate is a workspace member of [`warp-types`] so that versioning,
//! licensing, and release cadence stay consistent across the sibling family.
//! The Lean project inside `lean/` is the actual verification artifact.
//!
//! [`warp-types`]: https://crates.io/crates/warp-types
