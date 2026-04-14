//! SMT solver demo.
//!
//! Demonstrates the phase-typed SMT solver on QF_EUF and QF_UFBV examples,
//! including Nelson-Oppen cross-theory reasoning.

use warp_types_smt::{with_session, BvOpKind, SmtFormula, SmtResult};

fn main() {
    println!("warp-types-smt: Phase-typed SMT Solver\n");

    // ── EUF examples ──

    println!("── 1: a = b, f(a) = f(b) ──");
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, fa) = session.apply(f, &[a]);
        let (session, fb) = session.apply(f, &[b]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(a, b),
                SmtFormula::Eq(fa, fb),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    print_result(&result);

    println!("── 2: a = b, f(a) ≠ f(b) — congruence violation ──");
    let result = with_session(|session| {
        let (session, s) = session.declare_sort("S");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, a) = session.var("a", s);
        let (session, b) = session.var("b", s);
        let (session, fa) = session.apply(f, &[a]);
        let (session, fb) = session.apply(f, &[b]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(a, b),
                SmtFormula::Neq(fa, fb),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    print_result(&result);

    println!("── 3: a = b, b = c, f(a) ≠ f(c) — transitivity chain ──");
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
    print_result(&result);

    println!("── 4: (a = b ∨ c = d), f(a) ≠ f(b) — Boolean + theory ──");
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
    print_result(&result);

    // ── Cross-theory: EUF + BV (Nelson-Oppen combination) ──

    println!("\n── 5: x = 3, y = 4, bvadd(x,1) ≠ y — BV constant eval ──");
    println!("  (SAT with EUF only, UNSAT with BV — bvadd is interpreted)");
    let result_euf = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, x) = session.var("x", s);
        let (session, y) = session.var("y", s);
        let (session, three) = session.bv_const(5, 3, s);
        let (session, four) = session.bv_const(5, 4, s);
        let (session, one) = session.bv_const(5, 1, s);
        let (session, add_x_1) = session.bv_op(BvOpKind::Add, 5, &[x, one], s);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, three),
                SmtFormula::Eq(y, four),
                SmtFormula::Neq(add_x_1, y),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    let result_bv = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, x) = session.var("x", s);
        let (session, y) = session.var("y", s);
        let (session, three) = session.bv_const(5, 3, s);
        let (session, four) = session.bv_const(5, 4, s);
        let (session, one) = session.bv_const(5, 1, s);
        let (session, add_x_1) = session.bv_op(BvOpKind::Add, 5, &[x, one], s);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, three),
                SmtFormula::Eq(y, four),
                SmtFormula::Neq(add_x_1, y),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    print!("  EUF only: ");
    print_result(&result_euf);
    print!("  EUF + BV: ");
    print_result(&result_bv);

    println!("── 6: x = 3, y = 4, f(bvadd(x,1)) ≠ f(y) — BV + congruence ──");
    println!("  (BV evaluates bvadd(3,1) = 4 = y → EUF congruence: f(bvadd(x,1)) = f(y))");
    let result_euf = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, x) = session.var("x", s);
        let (session, y) = session.var("y", s);
        let (session, three) = session.bv_const(5, 3, s);
        let (session, four) = session.bv_const(5, 4, s);
        let (session, one) = session.bv_const(5, 1, s);
        let (session, add_x_1) = session.bv_op(BvOpKind::Add, 5, &[x, one], s);
        let (session, f_add) = session.apply(f, &[add_x_1]);
        let (session, f_y) = session.apply(f, &[y]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, three),
                SmtFormula::Eq(y, four),
                SmtFormula::Neq(f_add, f_y),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    let result_bv = with_session(|session| {
        let (session, s) = session.declare_sort("BV5");
        let (session, f) = session.declare_fun("f", &[s], s);
        let (session, x) = session.var("x", s);
        let (session, y) = session.var("y", s);
        let (session, three) = session.bv_const(5, 3, s);
        let (session, four) = session.bv_const(5, 4, s);
        let (session, one) = session.bv_const(5, 1, s);
        let (session, add_x_1) = session.bv_op(BvOpKind::Add, 5, &[x, one], s);
        let (session, f_add) = session.apply(f, &[add_x_1]);
        let (session, f_y) = session.apply(f, &[y]);

        let declared = session.finish_declarations();
        let asserted = declared
            .assert_formula(SmtFormula::And(vec![
                SmtFormula::Eq(x, three),
                SmtFormula::Eq(y, four),
                SmtFormula::Neq(f_add, f_y),
            ]))
            .finish_assertions();
        asserted.check_sat_bv()
    });
    print!("  EUF only: ");
    print_result(&result_euf);
    print!("  EUF + BV: ");
    print_result(&result_bv);
}

fn print_result(result: &SmtResult) {
    match result {
        SmtResult::Sat => println!("SAT"),
        SmtResult::Unsat => println!("UNSAT"),
        SmtResult::Unknown => println!("UNKNOWN"),
    }
}
