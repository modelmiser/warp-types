//! End-to-end integration tests for warp-types-smt.
//!
//! Each test exercises the full DPLL(T) pipeline:
//! session API → formula abstraction → EUF theory solver → SAT solver.

use warp_types_smt::{with_session, BvOpKind, SmtFormula, SmtResult};

// ============================================================================
// Helper: build common test scenarios through the session API
// ============================================================================

/// Run a test with one sort "S", optional function "f: S → S",
/// and a formula builder that receives (sort, func_id, term_ids...).
fn check_with_vars_and_fun(
    var_names: &[&str],
    need_func: bool,
    build: impl FnOnce(&[warp_types_smt::TermId], Option<warp_types_smt::FuncId>, &mut Vec<SmtFormula>),
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

// ============================================================================
// BV operator tests — exercise the constant-propagation BV module on each
// of the operators added in this commit (Not, Sub, Extract, Concat). Each
// asserts a concrete ground instance and checks that `check_sat_bv` reaches
// UNSAT when the computed value conflicts with a disequality.
// ============================================================================

#[test]
fn bvnot_detects_conflict() {
    // x = 0b01010, y = 0b10101, bvnot(x) ≠ y → UNSAT (bvnot(0b01010) = 0b10101)
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, x) = session.var("x", s);
        let (session, y) = session.var("y", s);
        let (session, five_val) = session.bv_const(5, 0b01010, s);
        let (session, twenty_one) = session.bv_const(5, 0b10101, s);
        let (session, not_x) = session.bv_op(BvOpKind::Not, 5, &[x], s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, five_val),
                SmtFormula::Eq(y, twenty_one),
                SmtFormula::Neq(not_x, y),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Unsat);
}

#[test]
fn bvsub_detects_conflict() {
    // x = 5, y = 2, z = 3, bvsub(x, y) ≠ z → UNSAT (5 - 2 = 3 in 5-bit)
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, x) = session.var("x", s);
        let (session, y) = session.var("y", s);
        let (session, z) = session.var("z", s);
        let (session, five_) = session.bv_const(5, 5, s);
        let (session, two) = session.bv_const(5, 2, s);
        let (session, three) = session.bv_const(5, 3, s);
        let (session, sub) = session.bv_op(BvOpKind::Sub, 5, &[x, y], s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, five_),
                SmtFormula::Eq(y, two),
                SmtFormula::Eq(z, three),
                SmtFormula::Neq(sub, z),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Unsat);
}

#[test]
fn bvextract_detects_conflict() {
    // x = 0b1101_0110 (8-bit), extract[3:1](x) = 0b011 (3-bit)
    // Assert extract(x) ≠ 0b011 → UNSAT.
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV");
        let (session, x) = session.var("x", s);
        let (session, full) = session.bv_const(8, 0b1101_0110, s);
        let (session, expected) = session.bv_const(3, 0b011, s);
        let (session, ext) = session.bv_extract(3, 1, x, s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, full),
                SmtFormula::Neq(ext, expected),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Unsat);
}

// ============================================================================
// Regression tests from cold review
// ============================================================================

// Finding A: BvOp terms participate in congruence closure (term.rs documents
// them as "uninterpreted by EUF (congruence only)").

#[test]
fn bvop_congruence_unsat_bv() {
    // a = b ∧ bvadd(a,c) ≠ bvadd(b,c) → UNSAT by congruence over the BV op
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, c) = session.var("c", s);
        let (session, add_ac) = session.bv_op(BvOpKind::Add, 5, &[a, c], s);
        let (session, add_bc) = session.bv_op(BvOpKind::Add, 5, &[b, c], s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(a, b),
                SmtFormula::Neq(add_ac, add_bc),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Unsat);
}

#[test]
fn bvop_congruence_unsat_euf_only() {
    // Same formula under plain check_sat: congruence alone suffices.
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, c) = session.var("c", s);
        let (session, add_ac) = session.bv_op(BvOpKind::Add, 5, &[a, c], s);
        let (session, add_bc) = session.bv_op(BvOpKind::Add, 5, &[b, c], s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(a, b),
                SmtFormula::Neq(add_ac, add_bc),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    assert_eq!(result, SmtResult::Unsat);
}

// Finding B: equality chaining a term to two different constants must conflict.

#[test]
fn bv_conflicting_constants_unsat() {
    // x = 3 ∧ x = 4 (5-bit) → UNSAT
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, x) = session.var("x", s);
        let (session, three) = session.bv_const(5, 3, s);
        let (session, four) = session.bv_const(5, 4, s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, three),
                SmtFormula::Eq(x, four),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Unsat);
}

// Finding B (adjacent instance): a BvOp forced to a value contradicting its
// ground evaluation must conflict.

#[test]
fn bv_forced_op_value_unsat() {
    // x = 3 ∧ bvadd(x,1) = 3 → UNSAT (3 + 1 = 4 ≠ 3)
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, x) = session.var("x", s);
        let (session, three) = session.bv_const(5, 3, s);
        let (session, one) = session.bv_const(5, 1, s);
        let (session, add) = session.bv_op(BvOpKind::Add, 5, &[x, one], s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, three),
                SmtFormula::Eq(add, three),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Unsat);
}

// Finding C: a solve path that requires a decision, a theory conflict, a
// backjump, and subsequent theory reasoning on the other branch.

#[test]
fn bv_decision_backtrack_sat() {
    // (x = 3 ∨ x = 4) ∧ bvadd(x,1) ≠ 4 → SAT (x = 4 works: 4+1 = 5 ≠ 4).
    // The x = 3 branch conflicts (3+1 = 4), forcing a backjump; stale module
    // state on the second branch previously produced non-falsified conflict
    // clauses.
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, x) = session.var("x", s);
        let (session, three) = session.bv_const(5, 3, s);
        let (session, four) = session.bv_const(5, 4, s);
        let (session, one) = session.bv_const(5, 1, s);
        let (session, add) = session.bv_op(BvOpKind::Add, 5, &[x, one], s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Or(vec![SmtFormula::Eq(x, three), SmtFormula::Eq(x, four)]),
                SmtFormula::Neq(add, four),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Sat);
}

// Finding F: bv_const masks the value to the declared width.

#[test]
fn bv_const_masked_to_width_unsat() {
    // 34 mod 2^5 = 2, so x = 1 ∧ bvadd(x,x) ≠ 34 is UNSAT (1+1 = 2 = 34 & 31).
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, x) = session.var("x", s);
        let (session, one) = session.bv_const(5, 1, s);
        let (session, big) = session.bv_const(5, 34, s);
        let (session, add) = session.bv_op(BvOpKind::Add, 5, &[x, x], s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, one),
                SmtFormula::Neq(add, big),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Unsat);
}

#[test]
#[should_panic(expected = "bv_const: width")]
fn bv_const_rejects_width_over_64() {
    with_session(|session| {
        let (session, s) = session.declare_sort("BV");
        let (_session, _t) = session.bv_const(65, 0, s);
    });
}

// Finding I: bv_extract validates hi < 64 (u64-backed values).

#[test]
#[should_panic(expected = "bv_extract: hi")]
fn bv_extract_rejects_hi_64() {
    with_session(|session| {
        let (session, s) = session.declare_sort("BV");
        let (session, x) = session.var("x", s);
        let (_session, _t) = session.bv_extract(64, 0, x, s);
    });
}

// Finding H: TermIds carry no brand — an out-of-range id from another
// session must fail loudly at the consuming entry point, not silently alias.

#[test]
#[should_panic(expected = "apply: TermId")]
fn cross_session_term_id_fails_loudly() {
    let foreign = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, _a) = session.var("a", s);
        let (_session, b) = session.var("b", s);
        b // TermId(1) — escapes the session (ids are unbranded indices)
    });
    with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        // This session's arena is empty: TermId(1) is out of range.
        let (_session, _t) = session.apply(f, &[foreign]);
    });
}

#[test]
fn bvconcat_detects_conflict() {
    // a = 0b101 (3-bit), b = 0b010 (3-bit), concat(a, b) = 0b101_010 (6-bit)
    // Assert concat ≠ 0b101010 → UNSAT.
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("BV");
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, hi_val) = session.bv_const(3, 0b101, s);
        let (session, lo_val) = session.bv_const(3, 0b010, s);
        let (session, expected) = session.bv_const(6, 0b101_010, s);
        let (session, cat) = session.bv_concat(a, 3, b, 3, s);
        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(a, hi_val),
                SmtFormula::Eq(b, lo_val),
                SmtFormula::Neq(cat, expected),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    assert_eq!(result, SmtResult::Unsat);
}
