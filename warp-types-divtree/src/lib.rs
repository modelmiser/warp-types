//! # warp-types-divtree
//!
//! Nested divergence partition tree for warp-types verification, packaged
//! as a Lean 4 library.
//!
//! The primary artifact of this crate is the Lean project at `lean/`, not
//! the Rust surface. This `lib.rs` exists so the crate can participate in
//! the warp-types Cargo workspace and be published to crates.io for
//! discoverability, but it contains no code.
//!
//! ## What this crate provides
//!
//! A formalization of the *diverge tree* — a binary tree whose internal
//! nodes split a parent bitmask into complementary halves via a predicate
//! — together with four soundness theorems proving that a well-formed
//! tree's leaves form a disjoint cover of its root:
//!
//! - **`leaves_cover_root`** — OR-folding all leaves recovers the root
//!   mask (the coverage half of the partition property).
//! - **`leaves_pairwise_disjoint`** — all leaf masks are pairwise
//!   disjoint (the disjointness half of the partition property).
//! - **`leaf_subset_root`** — every leaf is an AND-subset of the root
//!   (a lemma used by `leaves_pairwise_disjoint` and useful on its own).
//! - **`leaves_length`** — a tree with `k` internal nodes has `k + 1`
//!   leaves (size invariant for static capacity bounds).
//!
//! Plus the inductive type `DivTree`, the `WellFormed` predicate, and
//! the `root` / `leaves` / `nodeCount` accessors.
//!
//! ## Use from a Lean project
//!
//! Add a Lake dependency in your `lakefile.toml`:
//!
//! ```toml
//! [[require]]
//! name = "WarpTypesDivtree"
//! path = "path/to/warp-types/warp-types-divtree/lean"
//! ```
//!
//! Then import in your `.lean` files:
//!
//! ```text
//! import WarpTypesDivtree
//!
//! open WarpTypesDivtree
//!
//! example (t : DivTree 32) (h : DivTree.WellFormed t) :
//!     List.foldl (· ||| ·) 0 t.leaves = t.root :=
//!   DivTree.leaves_cover_root h
//! ```
//!
//! ## Why a crate and not a standalone Lake package?
//!
//! This crate is a workspace member of [`warp-types`] so that versioning,
//! licensing, and release cadence stay consistent across the sibling family.
//! The Lean project inside `lean/` is the actual verification artifact.
//!
//! [`warp-types`]: https://crates.io/crates/warp-types
