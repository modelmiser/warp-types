//! smt-crosscheck — our verdicts against an outside SMT solver on random instances.
//!
//! Usage: smt-crosscheck --euf N --bv N [--seed S] [--yices PATH] [--out DIR]
//!
//! Generates random QF_UF instances (checked with `check_sat`, which is
//! complete for EUF) and random QF_BV instances (checked with `check_sat_bv`,
//! whose `Sat` is incomplete by design: ground-only, no bit-blasting), exports
//! each as SMT-LIB 2, asks `yices-smt2`, and compares:
//!
//!   EUF  : verdicts must be equal — any difference is a kill.
//!   BV   : our `Unsat` must be yices `unsat` — a kill otherwise; our `Sat`
//!          against yices `unsat` is the documented incompleteness, counted.
//!
//! This is a *verdict* cross-check against a reference, not a certificate:
//! nothing here can be handed to a checker. Pre-registration:
//! `research/smt-crosscheck.md`.
use std::{env, fs, process};
use warp_types_smt::{with_session, BvOpKind, SmtFormula, SmtResult, TermId};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

fn yices(bin: &str, path: &str) -> String {
    match process::Command::new(bin).arg(path).output() {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.lines()
                .map(|l| l.trim())
                .find(|l| *l == "sat" || *l == "unsat" || *l == "unknown")
                .unwrap_or("error")
                .to_string()
        }
        Err(e) => format!("error: {e}"),
    }
}

fn ours(r: &SmtResult) -> &'static str {
    match r {
        SmtResult::Sat => "sat",
        SmtResult::Unsat => "unsat",
        SmtResult::Unknown => "unknown",
    }
}

/// Random literal-level formula over the given terms.
fn random_formula(rng: &mut Rng, terms: &[TermId], n_lits: usize) -> SmtFormula {
    let mut lits = Vec::new();
    for _ in 0..n_lits {
        let a = rng.pick(terms);
        let b = rng.pick(terms);
        let lit = if rng.below(2) == 0 {
            SmtFormula::Eq(a, b)
        } else {
            SmtFormula::Neq(a, b)
        };
        lits.push(lit);
    }
    // Some structure: a disjunction and an implication among the conjuncts.
    if lits.len() >= 4 {
        let d = SmtFormula::Or(vec![lits.pop().unwrap(), lits.pop().unwrap()]);
        let p = lits.pop().unwrap();
        let c = lits.pop().unwrap();
        lits.push(d);
        lits.push(SmtFormula::Implies(Box::new(p), Box::new(c)));
    }
    SmtFormula::And(lits)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut n_euf = 200usize;
    let mut n_bv = 200usize;
    let mut seed = 1u64;
    let mut yices_bin = "yices-smt2".to_string();
    let mut out_dir = "smt-crosscheck-out".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--euf" => {
                i += 1;
                n_euf = args[i].parse().expect("--euf N");
            }
            "--bv" => {
                i += 1;
                n_bv = args[i].parse().expect("--bv N");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("--seed S");
            }
            "--yices" => {
                i += 1;
                yices_bin = args[i].clone();
            }
            "--out" => {
                i += 1;
                out_dir = args[i].clone();
            }
            a => {
                eprintln!("unknown argument {a}");
                process::exit(1);
            }
        }
        i += 1;
    }
    fs::create_dir_all(&out_dir).expect("out dir");
    let mut csv = String::from("kind,index,ours,yices,class\n");
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let (mut euf_agree, mut euf_kill, mut euf_unknown) = (0, 0, 0);
    let (mut bv_agree, mut bv_kill, mut bv_incomplete, mut bv_unknown) = (0, 0, 0, 0);

    for k in 0..n_euf {
        let n_vars = 3 + rng.below(4);
        let n_lits = 3 + rng.below(5);
        let (res, text) = with_session(|session| {
            let (session, s) = session.declare_sort("S");
            let (mut session, f) = session.declare_fun("f", &[s], s);
            let (s2, g) = session.declare_fun("g", &[s, s], s);
            session = s2;
            let mut terms: Vec<TermId> = Vec::new();
            for v in 0..n_vars {
                let (s2, t) = session.var(&format!("x{v}"), s);
                session = s2;
                terms.push(t);
            }
            let base = terms.clone();
            for _ in 0..(2 + rng.below(5)) {
                let a = rng.pick(&terms);
                let t = if rng.below(2) == 0 {
                    let (s2, t) = session.apply(f, &[a]);
                    session = s2;
                    t
                } else {
                    let b = rng.pick(&base);
                    let (s2, t) = session.apply(g, &[a, b]);
                    session = s2;
                    t
                };
                terms.push(t);
            }
            let formula = random_formula(&mut rng, &terms, n_lits);
            let asserted = session
                .finish_declarations()
                .assert_formula(formula)
                .finish_assertions();
            let text = asserted.to_smtlib2().expect("export");
            (asserted.check_sat(), text)
        });
        let path = format!("{out_dir}/euf_{k}.smt2");
        fs::write(&path, &text).expect("write");
        let y = yices(&yices_bin, &path);
        let o = ours(&res);
        let class = if o == "unknown" || y == "unknown" || y.starts_with("error") {
            euf_unknown += 1;
            "unknown"
        } else if o == y {
            euf_agree += 1;
            "agree"
        } else {
            euf_kill += 1;
            "KILL"
        };
        csv.push_str(&format!("euf,{k},{o},{y},{class}\n"));
        if class == "KILL" {
            eprintln!("KILL: euf_{k}: ours {o}, yices {y} ({path})");
        }
    }

    for k in 0..n_bv {
        let width = rng.pick(&[4u32, 8]);
        let n_vars = 2 + rng.below(3);
        let n_lits = 2 + rng.below(4);
        let (res, text) = with_session(|session| {
            let (mut session, s) = session.declare_sort(&format!("BV{width}"));
            let mut terms: Vec<TermId> = Vec::new();
            for v in 0..n_vars {
                let (s2, t) = session.var(&format!("v{v}"), s);
                session = s2;
                terms.push(t);
            }
            for _ in 0..(2 + rng.below(3)) {
                let c = rng.next() & ((1u64 << width) - 1);
                let (s2, t) = session.bv_const(width, c, s);
                session = s2;
                terms.push(t);
            }
            for _ in 0..(2 + rng.below(4)) {
                let a = rng.pick(&terms);
                let b = rng.pick(&terms);
                let op = rng.pick(&[
                    BvOpKind::Add,
                    BvOpKind::And,
                    BvOpKind::Or,
                    BvOpKind::Xor,
                    BvOpKind::Sub,
                    BvOpKind::Not,
                ]);
                let (s2, t) = match op {
                    BvOpKind::Not => session.bv_op(op, width, &[a], s),
                    _ => session.bv_op(op, width, &[a, b], s),
                };
                session = s2;
                terms.push(t);
            }
            let formula = random_formula(&mut rng, &terms, n_lits);
            let asserted = session
                .finish_declarations()
                .assert_formula(formula)
                .finish_assertions();
            let text = asserted.to_smtlib2().expect("export");
            (asserted.check_sat_bv(), text)
        });
        let path = format!("{out_dir}/bv_{k}.smt2");
        fs::write(&path, &text).expect("write");
        let y = yices(&yices_bin, &path);
        let o = ours(&res);
        let class = if o == "unknown" || y == "unknown" || y.starts_with("error") {
            bv_unknown += 1;
            "unknown"
        } else if o == y {
            bv_agree += 1;
            "agree"
        } else if o == "unsat" {
            bv_kill += 1;
            "KILL"
        } else {
            bv_incomplete += 1;
            "incomplete"
        };
        csv.push_str(&format!("bv,{k},{o},{y},{class}\n"));
        if class == "KILL" {
            eprintln!("KILL: bv_{k}: ours {o}, yices {y} ({path})");
        }
    }
    fs::write(format!("{out_dir}/results.csv"), csv).expect("csv");
    println!("EUF: {n_euf} instances — agree {euf_agree}, KILL {euf_kill}, unknown {euf_unknown}");
    println!(
        "BV : {n_bv} instances — agree {bv_agree}, incomplete (ours sat, yices unsat) {bv_incomplete}, KILL (ours unsat, yices sat) {bv_kill}, unknown {bv_unknown}"
    );
    if euf_kill + bv_kill > 0 {
        process::exit(1);
    }
}
