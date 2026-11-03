//! smt-crosscheck — our verdicts against an outside SMT solver on random instances.
//!
//! Usage: smt-crosscheck --euf N --bv N [--seed S] [--yices PATH] [--out DIR]
//!                       [--bitblast] [--lrat-check BIN] [--cake-lpr BIN]
//!
//! `--bitblast` also decides every BV instance through the bit-blaster
//! (`SmtSession::bitblast`, complete), which must equal Yices on all of them;
//! with `--lrat-check` / `--cake-lpr`, every bit-blasted UNSAT writes its
//! DIMACS and LRAT and the checkers must accept them (kill otherwise).
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
    let mut bitblast = false;
    let mut lrat_check: Option<String> = None;
    let mut cake_lpr: Option<String> = None;
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
            "--bitblast" => bitblast = true,
            "--lrat-check" => {
                i += 1;
                lrat_check = Some(args[i].clone());
            }
            "--cake-lpr" => {
                i += 1;
                cake_lpr = Some(args[i].clone());
            }
            a => {
                eprintln!("unknown argument {a}");
                process::exit(1);
            }
        }
        i += 1;
    }
    fs::create_dir_all(&out_dir).expect("out dir");
    let mut csv = String::from("kind,index,ours,yices,class,bitblast,bb_class,proof\n");
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let (mut euf_agree, mut euf_kill, mut euf_unknown) = (0, 0, 0);
    let (mut bv_agree, mut bv_kill, mut bv_incomplete, mut bv_unknown) = (0, 0, 0, 0);
    let (mut bb_agree, mut bb_kill, mut proofs_ok, mut proofs_bad) = (0, 0, 0, 0);

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
        csv.push_str(&format!("euf,{k},{o},{y},{class},,,\n"));
        if class == "KILL" {
            eprintln!("KILL: euf_{k}: ours {o}, yices {y} ({path})");
        }
    }

    for k in 0..n_bv {
        let width = rng.pick(&[4u32, 8]);
        let n_vars = 2 + rng.below(3);
        let n_lits = 2 + rng.below(4);
        let (res, text, blasted) = with_session(|session| {
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
            let blasted = if bitblast {
                let bb = asserted.bitblast().expect("bitblast");
                let mut proof = Vec::new();
                let (r, _) = bb.solve_lrat(&mut proof, false).expect("solve");
                Some((r, bb.dimacs(), proof))
            } else {
                None
            };
            (asserted.check_sat_bv(), text, blasted)
        });
        let path = format!("{out_dir}/bv_{k}.smt2");
        fs::write(&path, &text).expect("write");
        let y = yices(&yices_bin, &path);
        let o = ours(&res);
        // Bit-blasted verdict and, on UNSAT, its certificate.
        let (bb_v, bb_class, proof_class) = match &blasted {
            None => (String::new(), String::new(), String::new()),
            Some((r, dimacs, proof)) => {
                let v = match r {
                    warp_types_sat::solver::SolveResult::Sat(_) => "sat",
                    warp_types_sat::solver::SolveResult::Unsat => "unsat",
                    warp_types_sat::solver::SolveResult::Unknown => "unknown",
                };
                let cls = if y == "unknown" || y.starts_with("error") || v == "unknown" {
                    "unknown"
                } else if v == y {
                    bb_agree += 1;
                    "agree"
                } else {
                    bb_kill += 1;
                    eprintln!("KILL: bv_{k}: bit-blasted {v}, yices {y} ({path})");
                    "KILL"
                };
                let mut pc = String::new();
                if v == "unsat" && (lrat_check.is_some() || cake_lpr.is_some()) {
                    let cnf_path = format!("{out_dir}/bv_{k}.cnf");
                    let lrat_path = format!("{out_dir}/bv_{k}.lrat");
                    fs::write(&cnf_path, dimacs).expect("write cnf");
                    fs::write(&lrat_path, proof).expect("write lrat");
                    let mut ok = true;
                    if let Some(lc) = &lrat_check {
                        let o = process::Command::new(lc)
                            .arg(&cnf_path)
                            .arg(&lrat_path)
                            .output()
                            .expect("lrat-check");
                        let good = o.status.success()
                            && String::from_utf8_lossy(&o.stdout).contains("c VERIFIED");
                        ok &= good;
                        pc.push_str(if good { "lc:ok " } else { "lc:NOT " });
                    }
                    if let Some(ck) = &cake_lpr {
                        let o = process::Command::new(ck)
                            .arg(&cnf_path)
                            .arg(&lrat_path)
                            .output()
                            .expect("cake_lpr");
                        let good = String::from_utf8_lossy(&o.stdout).contains("s VERIFIED");
                        ok &= good;
                        pc.push_str(if good { "cake:ok" } else { "cake:NOT" });
                    }
                    if ok {
                        proofs_ok += 1;
                    } else {
                        proofs_bad += 1;
                        eprintln!("KILL: bv_{k}: proof rejected ({pc})");
                    }
                }
                (v.to_string(), cls.to_string(), pc)
            }
        };
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
        csv.push_str(&format!(
            "bv,{k},{o},{y},{class},{bb_v},{bb_class},{proof_class}\n"
        ));
        if class == "KILL" {
            eprintln!("KILL: bv_{k}: ours {o}, yices {y} ({path})");
        }
    }
    fs::write(format!("{out_dir}/results.csv"), csv).expect("csv");
    println!("EUF: {n_euf} instances — agree {euf_agree}, KILL {euf_kill}, unknown {euf_unknown}");
    println!(
        "BV : {n_bv} instances — agree {bv_agree}, incomplete (ours sat, yices unsat) {bv_incomplete}, KILL (ours unsat, yices sat) {bv_kill}, unknown {bv_unknown}"
    );
    if bitblast {
        println!(
            "BV bit-blasted: agree with yices {bb_agree}, KILL {bb_kill}; UNSAT proofs accepted {proofs_ok}, rejected {proofs_bad}"
        );
    }
    if euf_kill + bv_kill + bb_kill + proofs_bad > 0 {
        process::exit(1);
    }
}
