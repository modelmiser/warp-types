//! SMT formula representation and Boolean abstraction.
//!
//! An [`SmtFormula`] is a Boolean combination of equality atoms over
//! uninterpreted function terms. The abstraction layer transforms formulas
//! into a propositional CNF skeleton (for the SAT solver) plus an [`AtomMap`]
//! (for the EUF theory solver to map SAT variables back to equality atoms).

use crate::term::{TermArena, TermId, TermKind};
use warp_types_sat::bcp::ClauseDb;
use warp_types_sat::literal::Lit;

// ============================================================================
// Formula type
// ============================================================================

/// An SMT formula: Boolean combination of equality atoms.
#[derive(Debug, Clone)]
pub enum SmtFormula {
    /// Equality atom: `(= t₁ t₂)`.
    Eq(TermId, TermId),
    /// Disequality: `(distinct t₁ t₂)`. Sugar for `Not(Eq(t₁, t₂))`.
    Neq(TermId, TermId),
    /// Boolean negation.
    Not(Box<SmtFormula>),
    /// Conjunction.
    And(Vec<SmtFormula>),
    /// Disjunction.
    Or(Vec<SmtFormula>),
    /// Implication: `p → q`.
    Implies(Box<SmtFormula>, Box<SmtFormula>),
}

// ============================================================================
// Atom map (theory ↔ SAT variable bridge)
// ============================================================================

/// Opaque equality-atom identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtomId(pub(crate) u32);

/// Bidirectional mapping between equality atoms and SAT variables.
///
/// Each equality atom `(= t₁ t₂)` is assigned a unique SAT variable.
/// The atom map enables the EUF theory solver to interpret SAT assignments
/// as equality/disequality assertions, and to produce SAT-level clauses
/// for theory lemmas.
pub struct AtomMap {
    /// `atom_to_var[atom_id]` = SAT variable for this atom.
    pub(crate) atom_to_var: Vec<u32>,
    /// `var_to_atom[var]` = Some((t1, t2)) if this SAT variable represents
    /// an equality atom, None if it's a Tseitin auxiliary.
    pub(crate) var_to_atom: Vec<Option<(TermId, TermId)>>,
    /// Deduplication: `(min(t1,t2), max(t1,t2))` → AtomId.
    pub(crate) eq_to_atom: std::collections::HashMap<(TermId, TermId), AtomId>,
    /// Next SAT variable to allocate.
    pub(crate) next_var: u32,
}

impl AtomMap {
    /// Create an empty atom map.
    pub fn new() -> Self {
        AtomMap {
            atom_to_var: Vec::new(),
            var_to_atom: Vec::new(),
            eq_to_atom: std::collections::HashMap::new(),
            next_var: 0,
        }
    }

    /// Get or create an atom for `(= t1 t2)`. Returns the AtomId and SAT variable.
    /// Uses canonical ordering `(min, max)` so `(= a b)` and `(= b a)` share an atom.
    pub fn get_or_create(&mut self, t1: TermId, t2: TermId) -> (AtomId, u32) {
        let key = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        if let Some(&atom_id) = self.eq_to_atom.get(&key) {
            return (atom_id, self.atom_to_var[atom_id.0 as usize]);
        }
        let atom_id = AtomId(self.atom_to_var.len() as u32);
        let var = self.alloc_var();
        self.atom_to_var.push(var);
        // Ensure var_to_atom is large enough
        while self.var_to_atom.len() <= var as usize {
            self.var_to_atom.push(None);
        }
        self.var_to_atom[var as usize] = Some(key);
        self.eq_to_atom.insert(key, atom_id);
        (atom_id, var)
    }

    /// Allocate a fresh SAT variable (for Tseitin auxiliaries).
    pub fn alloc_var(&mut self) -> u32 {
        let v = self.next_var;
        self.next_var += 1;
        v
    }

    /// Number of SAT variables allocated.
    pub fn num_vars(&self) -> u32 {
        self.next_var
    }

    /// Number of equality atoms.
    pub fn num_atoms(&self) -> u32 {
        self.atom_to_var.len() as u32
    }

    /// SAT variable for an atom.
    pub fn var_for_atom(&self, atom: AtomId) -> u32 {
        self.atom_to_var[atom.0 as usize]
    }

    /// Look up the equality atom for a SAT variable, if any.
    pub fn atom_for_var(&self, var: u32) -> Option<(TermId, TermId)> {
        self.var_to_atom.get(var as usize).copied().flatten()
    }
}

impl Default for AtomMap {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Boolean abstraction (Tseitin transformation)
// ============================================================================

/// Result of abstracting SMT formulas into propositional CNF.
pub struct Abstraction {
    /// Propositional CNF clause database (ready for SAT solver).
    pub db: ClauseDb,
    /// Total number of SAT variables (atoms + Tseitin auxiliaries).
    pub num_vars: u32,
    /// Bidirectional map between equality atoms and SAT variables.
    pub atom_map: AtomMap,
}

/// Transform SMT formulas into a propositional CNF skeleton.
///
/// Each equality atom `(= t₁ t₂)` gets a dedicated SAT variable.
/// Boolean connectives are encoded via Tseitin transformation:
/// each sub-formula gets an auxiliary variable, and implication
/// clauses enforce equivalence between the auxiliary and the sub-formula.
///
/// After encoding, **argument pair purification** creates equality atoms
/// for the arguments of matching function applications. If the formula
/// contains `(f(a), f(b))` as an equality atom, the atom `(a, b)` is
/// created so cross-theory equalities (e.g., BV evaluating `a = b`)
/// can be communicated through the combining solver. This iterates
/// to fixpoint for nested applications.
pub fn abstract_formulas(formulas: &[SmtFormula], arena: &TermArena) -> Abstraction {
    let mut atom_map = AtomMap::new();
    let mut db = ClauseDb::new();

    for formula in formulas {
        let top_var = tseitin_encode(formula, &mut atom_map, &mut db);
        // Assert the top-level formula is true
        db.add_clause(vec![Lit::pos(top_var)]);
    }

    // Purify: ensure argument-pair atoms exist for congruent applications.
    purify_argument_pairs(&mut atom_map, arena);

    // Pad var_to_atom to cover all allocated variables
    while atom_map.var_to_atom.len() < atom_map.next_var as usize {
        atom_map.var_to_atom.push(None);
    }

    Abstraction {
        num_vars: atom_map.num_vars(),
        db,
        atom_map,
    }
}

/// Create equality atoms for argument pairs of matching function applications.
///
/// If the atom map has `(f(a₁..aₙ), f(b₁..bₙ))`, creates atoms for each
/// `(aᵢ, bᵢ)` where `aᵢ ≠ bᵢ`. This enables cross-theory equality sharing:
/// the combining solver can only propagate equalities that have SAT atoms.
///
/// Handles both `Apply` (uninterpreted functions) and `BvOp` (bitvector
/// operations). Iterates to fixpoint for nested applications.
fn purify_argument_pairs(atom_map: &mut AtomMap, arena: &TermArena) {
    loop {
        let mut new_pairs: Vec<(TermId, TermId)> = Vec::new();
        let num_atoms = atom_map.num_atoms();

        for atom_idx in 0..num_atoms {
            let atom_id = AtomId(atom_idx);
            let var = atom_map.var_for_atom(atom_id);
            let Some((t1, t2)) = atom_map.atom_for_var(var) else {
                continue;
            };
            if let Some((args1, args2)) = matching_congruence_args(arena, t1, t2) {
                for (&ai, &bi) in args1.iter().zip(args2.iter()) {
                    if ai != bi {
                        let key = if ai <= bi { (ai, bi) } else { (bi, ai) };
                        if !atom_map.eq_to_atom.contains_key(&key) {
                            new_pairs.push((ai, bi));
                        }
                    }
                }
            }
        }

        if new_pairs.is_empty() {
            break;
        }
        new_pairs.sort();
        new_pairs.dedup();
        for (a, b) in new_pairs {
            atom_map.get_or_create(a, b);
        }
    }
}

/// If two terms are applications of the same function (or same BvOp with
/// matching width), return their argument lists for congruence pairing.
fn matching_congruence_args<'a>(
    arena: &'a TermArena,
    t1: TermId,
    t2: TermId,
) -> Option<(&'a [TermId], &'a [TermId])> {
    let k1 = &arena.get(t1).kind;
    let k2 = &arena.get(t2).kind;
    match (k1, k2) {
        (TermKind::Apply { func: f1, args: a1 }, TermKind::Apply { func: f2, args: a2 })
            if f1 == f2 && a1.len() == a2.len() =>
        {
            Some((a1, a2))
        }
        (
            TermKind::BvOp {
                op: o1,
                width: w1,
                args: a1,
            },
            TermKind::BvOp {
                op: o2,
                width: w2,
                args: a2,
            },
        ) if o1 == o2 && w1 == w2 && a1.len() == a2.len() => Some((a1, a2)),
        _ => None,
    }
}

/// Recursively Tseitin-encode a formula. Returns the SAT variable
/// representing this sub-formula (true iff the sub-formula is satisfied).
fn tseitin_encode(formula: &SmtFormula, atom_map: &mut AtomMap, db: &mut ClauseDb) -> u32 {
    match formula {
        SmtFormula::Eq(t1, t2) => {
            let (_atom_id, var) = atom_map.get_or_create(*t1, *t2);
            var
        }

        SmtFormula::Neq(t1, t2) => {
            // Neq(t1, t2) = Not(Eq(t1, t2))
            let (_atom_id, eq_var) = atom_map.get_or_create(*t1, *t2);
            let neq_var = atom_map.alloc_var();
            // neq_var ↔ ¬eq_var
            // (neq_var → ¬eq_var): (¬neq_var ∨ ¬eq_var)
            db.add_clause(vec![Lit::neg(neq_var), Lit::neg(eq_var)]);
            // (¬eq_var → neq_var): (eq_var ∨ neq_var)
            db.add_clause(vec![Lit::pos(eq_var), Lit::pos(neq_var)]);
            neq_var
        }

        SmtFormula::Not(inner) => {
            let inner_var = tseitin_encode(inner, atom_map, db);
            let not_var = atom_map.alloc_var();
            // not_var ↔ ¬inner_var
            // (not_var → ¬inner_var): (¬not_var ∨ ¬inner_var)
            db.add_clause(vec![Lit::neg(not_var), Lit::neg(inner_var)]);
            // (¬inner_var → not_var): (inner_var ∨ not_var)
            db.add_clause(vec![Lit::pos(inner_var), Lit::pos(not_var)]);
            not_var
        }

        SmtFormula::And(children) => {
            if children.is_empty() {
                // Empty conjunction is true — allocate a variable forced true
                let var = atom_map.alloc_var();
                db.add_clause(vec![Lit::pos(var)]);
                return var;
            }
            let child_vars: Vec<u32> = children
                .iter()
                .map(|c| tseitin_encode(c, atom_map, db))
                .collect();
            let and_var = atom_map.alloc_var();
            // and_var → (c₁ ∧ c₂ ∧ ... ∧ cₙ)
            // For each cᵢ: (¬and_var ∨ cᵢ)
            for &cv in &child_vars {
                db.add_clause(vec![Lit::neg(and_var), Lit::pos(cv)]);
            }
            // (c₁ ∧ c₂ ∧ ... ∧ cₙ) → and_var
            // (¬c₁ ∨ ¬c₂ ∨ ... ∨ ¬cₙ ∨ and_var)
            let mut clause: Vec<Lit> = child_vars.iter().map(|&cv| Lit::neg(cv)).collect();
            clause.push(Lit::pos(and_var));
            db.add_clause(clause);
            and_var
        }

        SmtFormula::Or(children) => {
            if children.is_empty() {
                // Empty disjunction is false — allocate a variable forced false
                let var = atom_map.alloc_var();
                db.add_clause(vec![Lit::neg(var)]);
                return var;
            }
            let child_vars: Vec<u32> = children
                .iter()
                .map(|c| tseitin_encode(c, atom_map, db))
                .collect();
            let or_var = atom_map.alloc_var();
            // or_var → (c₁ ∨ c₂ ∨ ... ∨ cₙ)
            // (¬or_var ∨ c₁ ∨ c₂ ∨ ... ∨ cₙ)
            let mut clause: Vec<Lit> = child_vars.iter().map(|&cv| Lit::pos(cv)).collect();
            clause.insert(0, Lit::neg(or_var));
            db.add_clause(clause);
            // For each cᵢ: (cᵢ → or_var) = (¬cᵢ ∨ or_var)
            for &cv in &child_vars {
                db.add_clause(vec![Lit::neg(cv), Lit::pos(or_var)]);
            }
            or_var
        }

        SmtFormula::Implies(lhs, rhs) => {
            // p → q  ≡  ¬p ∨ q
            let p_var = tseitin_encode(lhs, atom_map, db);
            let q_var = tseitin_encode(rhs, atom_map, db);
            let imp_var = atom_map.alloc_var();
            // imp_var → (¬p ∨ q)
            // (¬imp_var ∨ ¬p ∨ q)
            db.add_clause(vec![Lit::neg(imp_var), Lit::neg(p_var), Lit::pos(q_var)]);
            // (¬p ∨ q) → imp_var
            // Contrapositive: ¬imp_var → (p ∧ ¬q)
            // So: (imp_var ∨ p) and (imp_var ∨ ¬q)
            db.add_clause(vec![Lit::pos(imp_var), Lit::pos(p_var)]);
            db.add_clause(vec![Lit::pos(imp_var), Lit::neg(q_var)]);
            imp_var
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::{SortId, TermKind};

    fn tid(n: u32) -> TermId {
        TermId(n)
    }

    /// Dummy arena with `n` Variable entries — purification sees no Apply/BvOp
    /// terms, so no new atoms are created. Allows formula.rs tests (which use
    /// raw TermIds) to pass the arena to `abstract_formulas`.
    fn dummy_arena(n: usize) -> TermArena {
        let mut arena = TermArena::new();
        let s = SortId(0);
        for i in 0..n {
            arena.intern(
                TermKind::Variable {
                    name: format!("t{i}"),
                    sort: s,
                },
                s,
            );
        }
        arena
    }

    #[test]
    fn single_equality_atom() {
        let formulas = vec![SmtFormula::Eq(tid(0), tid(1))];
        let abs = abstract_formulas(&formulas, &dummy_arena(2));
        // One equality atom → one SAT variable (var 0)
        // Plus one unit clause asserting it true
        assert_eq!(abs.atom_map.num_atoms(), 1);
        assert_eq!(abs.num_vars, 1);
        // The atom should map to var 0
        assert_eq!(abs.atom_map.var_for_atom(AtomId(0)), 0);
        assert_eq!(abs.atom_map.atom_for_var(0), Some((tid(0), tid(1))));
    }

    #[test]
    fn disequality_creates_auxiliary() {
        let formulas = vec![SmtFormula::Neq(tid(0), tid(1))];
        let abs = abstract_formulas(&formulas, &dummy_arena(4));
        // One equality atom (var 0) + one Neq auxiliary (var 1)
        assert_eq!(abs.atom_map.num_atoms(), 1);
        assert_eq!(abs.num_vars, 2);
    }

    #[test]
    fn conjunction_encoding() {
        let formulas = vec![SmtFormula::And(vec![
            SmtFormula::Eq(tid(0), tid(1)),
            SmtFormula::Eq(tid(2), tid(3)),
        ])];
        let abs = abstract_formulas(&formulas, &dummy_arena(4));
        // 2 equality atoms (vars 0, 1) + 1 And auxiliary (var 2)
        assert_eq!(abs.atom_map.num_atoms(), 2);
        assert_eq!(abs.num_vars, 3);
    }

    #[test]
    fn disjunction_encoding() {
        let formulas = vec![SmtFormula::Or(vec![
            SmtFormula::Eq(tid(0), tid(1)),
            SmtFormula::Eq(tid(2), tid(3)),
        ])];
        let abs = abstract_formulas(&formulas, &dummy_arena(4));
        // 2 equality atoms + 1 Or auxiliary
        assert_eq!(abs.atom_map.num_atoms(), 2);
        assert_eq!(abs.num_vars, 3);
    }

    #[test]
    fn shared_atom_deduplication() {
        // Two assertions sharing the same equality atom
        let formulas = vec![
            SmtFormula::Eq(tid(0), tid(1)),
            SmtFormula::Eq(tid(1), tid(0)), // reversed — same canonical pair
        ];
        let abs = abstract_formulas(&formulas, &dummy_arena(4));
        // Only one atom despite two formulas (canonical ordering dedup)
        assert_eq!(abs.atom_map.num_atoms(), 1);
        assert_eq!(abs.num_vars, 1);
    }

    #[test]
    fn implication_encoding() {
        let formulas = vec![SmtFormula::Implies(
            Box::new(SmtFormula::Eq(tid(0), tid(1))),
            Box::new(SmtFormula::Eq(tid(2), tid(3))),
        )];
        let abs = abstract_formulas(&formulas, &dummy_arena(4));
        // 2 equality atoms + 1 Implies auxiliary
        assert_eq!(abs.atom_map.num_atoms(), 2);
        assert_eq!(abs.num_vars, 3);
    }

    #[test]
    fn nested_formula() {
        // (a = b) ∧ ¬(c = d)
        let formulas = vec![SmtFormula::And(vec![
            SmtFormula::Eq(tid(0), tid(1)),
            SmtFormula::Not(Box::new(SmtFormula::Eq(tid(2), tid(3)))),
        ])];
        let abs = abstract_formulas(&formulas, &dummy_arena(4));
        // 2 atoms + 1 Not aux + 1 And aux = 4 vars
        assert_eq!(abs.atom_map.num_atoms(), 2);
        assert_eq!(abs.num_vars, 4);
    }
}
