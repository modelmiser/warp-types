//! SMT command-line tool.
//!
//! Demonstrates the phase-typed SMT solver on QF_EUF examples.

use warp_types_smt::{with_session, SmtFormula, SmtResult};

fn main() {
    println!("warp-types-smt: Phase-typed SMT Solver for QF_EUF\n");

    // ── Example 1: SAT — a = b, f(a) = f(b) ──
    println!("── Example 1: a = b, f(a) = f(b) ──");
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

    // ── Example 2: UNSAT — a = b, f(a) ≠ f(b) (congruence violation) ──
    println!("\n── Example 2: a = b, f(a) ≠ f(b) — congruence violation ──");
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

    // ── Example 3: UNSAT — a = b, b = c, f(a) ≠ f(c) (transitivity + congruence) ──
    println!("\n── Example 3: a = b, b = c, f(a) ≠ f(c) — transitivity chain ──");
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

    // ── Example 4: SAT — (a = b OR c = d), f(a) ≠ f(b) — Boolean reasoning ──
    println!("\n── Example 4: (a = b ∨ c = d), f(a) ≠ f(b) — Boolean + theory ──");
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
                SmtFormula::Or(vec![
                    SmtFormula::Eq(a, b),
                    SmtFormula::Eq(c, d),
                ]),
                SmtFormula::Neq(fa, fb),
            ]))
            .finish_assertions();
        asserted.check_sat()
    });
    print_result(&result);
}

fn print_result(result: &SmtResult) {
    match result {
        SmtResult::Sat => println!("  SAT"),
        SmtResult::Unsat => println!("  UNSAT"),
        SmtResult::Unknown => println!("  UNKNOWN"),
    }
}
