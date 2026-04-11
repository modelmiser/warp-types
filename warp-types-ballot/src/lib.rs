//! # warp-types-ballot
//!
//! GPU warp-vote fold-algebra lemmas for warp-types verification,
//! packaged as a Lean 4 library.
//!
//! The primary artifact of this crate is the Lean project at `lean/`, not
//! the Rust surface. This `lib.rs` exists so the crate can participate in
//! the warp-types Cargo workspace and be published to crates.io for
//! discoverability, but it contains no code.
//!
//! ## What this crate provides
//!
//! Seven lemmas covering the GPU warp-vote API — `__ballot_sync`,
//! `__all_sync`, `__any_sync` — modeled as fold algebra over lists of
//! per-lane masks:
//!
//! ### Ballot (OR-fold)
//! - **`ballot_nil`** — empty ballot is zero
//! - **`ballot_singleton`** — singleton ballot is the lane value
//! - **`ballot_split`** — OR-fold distributes over list append
//!
//! ### All-sync (AND-fold starting from allOnes)
//! - **`all_sync_nil`** — empty all-sync is allOnes
//! - **`all_sync_singleton`** — singleton all-sync is the lane value
//! - **`all_sync_split`** — AND-fold distributes over list append
//!
//! ### Any-sync
//! - **`any_sync_monotone`** — if the OR of two masks is non-zero, at
//!   least one is non-zero
//!
//! Three of these (`ballot_split`, `all_sync_split`, `any_sync_monotone`)
//! are structurally identical to the fold-algebra theorems already in
//! `warp-types-bitwise`'s `CUDA.lean` module; the v0.1.0 release
//! duplicates them here per the sibling-plan's "each crate standalone"
//! discipline, and a family-wide refactor scheduled after ballot lands
//! will hoist the shared helpers into a common module.
//!
//! The four `*_nil` and `*_singleton` completions are new in ballot —
//! they fill out the boundary cases of the fold API so that consumers
//! can case-analyze lists (nil / singleton / cons / append) without
//! dropping to raw `List.foldl` reduction.
//!
//! ## Use from a Lean project
//!
//! Add a Lake dependency in your `lakefile.toml`:
//!
//! ```toml
//! [[require]]
//! name = "WarpTypesBallot"
//! path = "path/to/warp-types/warp-types-ballot/lean"
//! ```
//!
//! Then import in your `.lean` files:
//!
//! ```text
//! import WarpTypesBallot
//!
//! example (L R : List (BitVec 32)) :
//!     List.foldl (· ||| ·) 0 (L ++ R) =
//!     List.foldl (· ||| ·) 0 L ||| List.foldl (· ||| ·) 0 R :=
//!   WarpTypesBallot.ballot_split L R
//! ```
//!
//! ## Why a crate and not a standalone Lake package?
//!
//! This crate is a workspace member of [`warp-types`] so that versioning,
//! licensing, and release cadence stay consistent across the sibling family.
//! The Lean project inside `lean/` is the actual verification artifact.
//!
//! [`warp-types`]: https://crates.io/crates/warp-types
