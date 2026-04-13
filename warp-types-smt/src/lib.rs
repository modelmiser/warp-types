//! Phase-typed SMT solver for QF_EUF (Quantifier-Free Equality with
//! Uninterpreted Functions).
//!
//! Encodes the SMT workflow in Rust's type system following the typestate
//! pattern from `warp_types`. Phase transitions consume the old state and
//! produce the new — invalid transitions are compile errors.
//!
//! # SMT Phase Machine
//!
//! ```text
//! with_session(|session| {
//!   // session: SmtSession<'s, Init>
//!   let (session, s) = session.declare_sort("S");
//!   let (session, f) = session.declare_fun("f", &[s], s);
//!   let (session, a) = session.var("a", s);
//!   let (session, b) = session.var("b", s);
//!   let declared = session.finish_declarations();
//!   // declared: SmtSession<'s, Declared>
//!   let asserted = declared
//!       .assert_formula(SmtFormula::Eq(a, b))
//!       .finish_assertions();
//!   // asserted: SmtSession<'s, Asserted>
//!   asserted.check_sat()
//! })
//! ```
//!
//! # Built on warp-types-sat
//!
//! Uses the CDCL SAT solver from `warp-types-sat` as the Boolean backbone.
//! The EUF congruence closure engine implements `TheorySolver`, integrating
//! via the DPLL(T) protocol: check after BCP fixpoint, lazy explanation
//! during conflict analysis, backtrackable union-find.

pub mod euf;
pub mod formula;
pub mod phase;
pub mod session;
pub mod solver;
pub mod term;

pub use euf::EufSolver;
pub use formula::{AtomId, AtomMap, SmtFormula};
pub use phase::{Asserted, Declared, Init, Phase, Sat, Unknown, Unsat};
pub use session::{with_session, SmtSession};
pub use solver::SmtResult;
pub use term::{FuncDecl, FuncId, Sort, SortId, TermArena, TermId, TermKind};
