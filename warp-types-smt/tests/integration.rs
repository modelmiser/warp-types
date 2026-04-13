//! End-to-end integration tests for warp-types-smt.
//!
//! Each test exercises the full DPLL(T) pipeline:
//! session API → formula abstraction → EUF theory solver → SAT solver.

use warp_types_smt::{with_session, SmtFormula, SmtResult};

// ============================================================================
// Helper: build common test scenarios through the session API
// ============================================================================

/// Run a test with one sort "S", optional function "f: S → S",
/// and a formula builder that receives (sort, func_id, term_ids...).
fn check_with_vars_and_fun(
    var_names: &[&str],
    need_func: bool,
    build: impl for<'s> FnOnce(
        &[warp_types_smt::TermId],
        Option<warp_types_smt::FuncId>,
        &mut Vec<SmtFormula>,
    ),
) -> SmtResult {
    with_session(|session| {
        // Declare sort S
        let (session, s) = session.declare_sort("S");

        // Optionally declare f: S → S
        let (session, f_opt) = if need_func {
            let (session, f) = session.declare_fun("f", &[s], s);
            (session, Some(f))
        } else {
            (session, None)
        };

        // Declare variables
        let mut sess = session;
        let mut term_ids = Vec::new();
        for &name in var_names {
            let (s2, tid) = sess.var(name, s);
            sess = s2;
            term_ids.push(tid);
        }

        // Build function applications if needed
        // The caller will build them using apply in the formulas directly
        // — but we need to create f(a), f(b), etc. in the arena first
        let mut extra_terms = Vec::new();
        if let Some(f) = f_opt {
            for &t in &term_ids {
                let (s2, ft) = sess.apply(f, &[t]);
                sess = s2;
                extra_terms.push(ft);
            }
        }

        // Combine all terms: [vars..., f(vars)...]
        let all_terms: Vec<_> = term_ids.iter().chain(extra_terms.iter()).copied().collect();

        let mut formulas = Vec::new();
        build(&all_terms, f_opt, &mut formulas);

        let declared = sess.finish_declarations();
        let mut asserted = declared;
        for formula in formulas {
            asserted = asserted.assert_formula(formula);
        }
        asserted.finish_assertions().check_sat()
    })
}

// ============================================================================
// Test 1: Trivial SAT — no assertions
// ============================================================================

#[test]
fn trivial_sat_no_assertions() {
    let result = with_session(|session| {
        let (session, _s) = session.declare_sort("S");
        let declared = session.finish_declarations();
        let asserted = declared.finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Sat);
}

// ============================================================================
// Test 2: Simple equality — SAT
// ============================================================================

#[test]
fn simple_equality_sat() {
    // Assert: a = b. This is satisfiable (just make a and b the same).
    let result = check_with_vars_and_fun(&["a", "b"], false, |terms, _, formulas| {
        formulas.push(SmtFormula::Eq(terms[0], terms[1]));
    });
    assert_eq!(result, SmtResult::Sat);
}

// ============================================================================
// Test 3: Congruence UNSAT — a = b, f(a) ≠ f(b)
// ============================================================================

#[test]
fn congruence_unsat() {
    // Assert: a = b AND f(a) ≠ f(b)
    // UNSAT: if a = b then f(a) = f(b) by congruence
    let result = check_with_vars_and_fun(&["a", "b"], true, |terms, _, formulas| {
        // terms: [a, b, f(a), f(b)]
        let (a, b, fa, fb) = (terms[0], terms[1], terms[2], terms[3]);
        formulas.push(SmtFormula::And(vec![
            SmtFormula::Eq(a, b),
            SmtFormula::Neq(fa, fb),
        ]));
    });
    assert_eq!(result, SmtResult::Unsat);
}

// ============================================================================
// Test 4: Transitivity UNSAT — a = b, b = c, f(a) ≠ f(c)
// ============================================================================

#[test]
fn transitivity_unsat() {
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, c) = session.var("c", s);
        let (session, fa) = session.apply(f, &[a]);
        let (session, fc) = session.apply(f, &[c]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(a, b),
                SmtFormula::Eq(b, c),
                SmtFormula::Neq(fa, fc),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Unsat);
}

// ============================================================================
// Test 5: Diamond UNSAT — a = b, a = c, f(b) ≠ f(c)
// ============================================================================

#[test]
fn diamond_unsat() {
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, c) = session.var("c", s);
        let (session, fb) = session.apply(f, &[b]);
        let (session, fc) = session.apply(f, &[c]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(a, b),
                SmtFormula::Eq(a, c),
                SmtFormula::Neq(fb, fc),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Unsat);
}

// ============================================================================
// Test 6: Nested congruence UNSAT — a = b, f(f(a)) ≠ f(f(b))
// ============================================================================

#[test]
fn nested_congruence_unsat() {
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, fa) = session.apply(f, &[a]);
        let (session, fb) = session.apply(f, &[b]);
        let (session, ffa) = session.apply(f, &[fa]);
        let (session, ffb) = session.apply(f, &[fb]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(a, b),
                SmtFormula::Neq(ffa, ffb),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Unsat);
}

// ============================================================================
// Test 7: Boolean disjunction SAT — (a = b OR c = d), f(a) ≠ f(b)
// ============================================================================

#[test]
fn disjunction_sat() {
    // (a = b OR c = d) AND f(a) ≠ f(b)
    // SAT: pick c = d (which satisfies the disjunction), a ≠ b (which satisfies f(a) ≠ f(b))
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, c) = session.var("c", s);
        let (session, d) = session.var("d", s);
        let (session, fa) = session.apply(f, &[a]);
        let (session, fb) = session.apply(f, &[b]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Or(vec![SmtFormula::Eq(a, b), SmtFormula::Eq(c, d)]),
                SmtFormula::Neq(fa, fb),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Sat);
}

// ============================================================================
// Test 8: Boolean disjunction UNSAT
// (a = b OR a = c), f(a) ≠ f(b), f(a) ≠ f(c) → UNSAT
// ============================================================================

#[test]
fn disjunction_unsat() {
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, c) = session.var("c", s);
        let (session, fa) = session.apply(f, &[a]);
        let (session, fb) = session.apply(f, &[b]);
        let (session, fc) = session.apply(f, &[c]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::Or(vec![
                SmtFormula::Eq(a, b),
                SmtFormula::Eq(a, c),
            ]))
            .assert_formula(SmtFormula::Neq(fa, fb))
            .assert_formula(SmtFormula::Neq(fa, fc))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Unsat);
}

// ============================================================================
// Test 9: Pure equality SAT — a = b, b = c (no disequalities)
// ============================================================================

#[test]
fn pure_equality_sat() {
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, c) = session.var("c", s);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::Eq(a, b))
            .assert_formula(SmtFormula::Eq(b, c))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Sat);
}

// ============================================================================
// Test 10: Self-equality — a = a (trivially SAT)
// ============================================================================

#[test]
fn self_equality_sat() {
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, a) = session.var("a", s);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::Eq(a, a))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Sat);
}

// ============================================================================
// Test 11: Implication — (a = b) → (f(a) = f(b)) — tautology, SAT
// ============================================================================

#[test]
fn implication_tautology_sat() {
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, fa) = session.apply(f, &[a]);
        let (session, fb) = session.apply(f, &[b]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::Implies(
                Box::new(SmtFormula::Eq(a, b)),
                Box::new(SmtFormula::Eq(fa, fb)),
            ))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Sat);
}

// ============================================================================
// Test 12: Constants (0-ary functions) — a = b, b ≠ a → UNSAT
// ============================================================================

#[test]
fn contradiction_unsat() {
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::Eq(a, b))
            .assert_formula(SmtFormula::Neq(a, b))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Unsat);
}
