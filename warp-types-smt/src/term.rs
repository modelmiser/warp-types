//! Term representation for QF_EUF.
//!
//! Arena-based with hash-consing: each structurally unique term is stored once,
//! referenced by [`TermId`]. This enables O(1) identity checks and efficient
//! union-find indexing in the congruence closure engine.
//!
//! # Term kinds
//!
//! - **Variable**: a named constant of a given sort (0-ary function application)
//! - **Apply**: function application `f(t₁, ..., tₙ)` where `f` is an
//!   uninterpreted function symbol and each `tᵢ` is a [`TermId`]

use std::collections::HashMap;

// ============================================================================
// Identifiers
// ============================================================================

/// Opaque sort identifier (index into the session's sort table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SortId(pub(crate) u32);

/// Opaque function symbol identifier (index into the session's function table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncId(pub(crate) u32);

/// Opaque term identifier (index into [`TermArena`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TermId(pub(crate) u32);

impl TermId {
    /// Raw index for union-find and array indexing.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// ============================================================================
// Sort and function declarations
// ============================================================================

/// A declared sort (opaque — no internal structure in QF_EUF).
#[derive(Debug, Clone)]
pub struct Sort {
    /// Sort name (e.g. "S", "T").
    pub name: String,
}

/// A declared uninterpreted function symbol.
#[derive(Debug, Clone)]
pub struct FuncDecl {
    /// Function name (e.g. "f", "g").
    pub name: String,
    /// Argument sort signature.
    pub arg_sorts: Vec<SortId>,
    /// Return sort.
    pub ret_sort: SortId,
}

// ============================================================================
// Term structure
// ============================================================================

/// The internal structure of a term.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TermKind {
    /// A variable (named constant of a given sort).
    Variable { name: String, sort: SortId },
    /// Function application: `f(t₁, ..., tₙ)`.
    Apply { func: FuncId, args: Vec<TermId> },
}

/// Arena entry for a term.
#[derive(Debug, Clone)]
pub struct TermEntry {
    /// Structural content of this term.
    pub kind: TermKind,
    /// Sort of this term.
    pub sort: SortId,
}

// ============================================================================
// Term arena with hash-consing
// ============================================================================

/// Arena-based term storage with hash-consing.
///
/// Each structurally unique term is stored exactly once. `intern()` returns
/// the existing [`TermId`] if the term already exists, or allocates a new
/// entry if it doesn't. This guarantees that `t1 == t2` iff the terms are
/// structurally identical — no deep comparison needed.
pub struct TermArena {
    /// Flat array of term entries, indexed by TermId.
    terms: Vec<TermEntry>,
    /// Hash-cons table: maps TermKind → TermId for deduplication.
    dedup: HashMap<TermKind, TermId>,
}

impl TermArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        TermArena {
            terms: Vec::new(),
            dedup: HashMap::new(),
        }
    }

    /// Intern a term: return the existing ID if structurally identical,
    /// or allocate a new entry.
    pub fn intern(&mut self, kind: TermKind, sort: SortId) -> TermId {
        if let Some(&id) = self.dedup.get(&kind) {
            return id;
        }
        let id = TermId(self.terms.len() as u32);
        self.dedup.insert(kind.clone(), id);
        self.terms.push(TermEntry { kind, sort });
        id
    }

    /// Look up a term entry by ID.
    ///
    /// # Panics
    /// Debug-panics if the ID is out of bounds.
    pub fn get(&self, id: TermId) -> &TermEntry {
        &self.terms[id.index()]
    }

    /// Number of interned terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Default for TermArena {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_consing_deduplicates() {
        let mut arena = TermArena::new();
        let s = SortId(0);
        let kind = TermKind::Variable {
            name: "x".into(),
            sort: s,
        };
        let t1 = arena.intern(kind.clone(), s);
        let t2 = arena.intern(kind, s);
        assert_eq!(t1, t2);
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn distinct_terms_get_distinct_ids() {
        let mut arena = TermArena::new();
        let s = SortId(0);
        let a = arena.intern(
            TermKind::Variable {
                name: "a".into(),
                sort: s,
            },
            s,
        );
        let b = arena.intern(
            TermKind::Variable {
                name: "b".into(),
                sort: s,
            },
            s,
        );
        assert_ne!(a, b);
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn function_application_interning() {
        let mut arena = TermArena::new();
        let s = SortId(0);
        let f = FuncId(0);
        let a = arena.intern(
            TermKind::Variable {
                name: "a".into(),
                sort: s,
            },
            s,
        );
        let fa1 = arena.intern(
            TermKind::Apply {
                func: f,
                args: vec![a],
            },
            s,
        );
        let fa2 = arena.intern(
            TermKind::Apply {
                func: f,
                args: vec![a],
            },
            s,
        );
        assert_eq!(fa1, fa2);
        assert_eq!(arena.len(), 2); // "a" and "f(a)"
    }
}
