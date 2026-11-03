//! SMT-LIB 2 export of a session's declarations and assertions, so that an
//! outside solver can be asked the same question (`smt-crosscheck`).
//!
//! The crate's sorts carry no width; a sort is exported as `(_ BitVec w)` when
//! every `BvConst` / `BvOp` term of that sort agrees on a width `w`, and as an
//! uninterpreted sort otherwise — which is exactly how the crate's own EUF
//! treats it. Symbols are `|quoted|` so any name survives.

use crate::formula::SmtFormula;
use crate::session::SmtEnv;
use crate::term::{BvOpKind, SortId, TermId, TermKind};
use std::collections::HashMap;
use std::fmt::Write;

/// Why an environment could not be exported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportError {
    /// Two bitvector terms of the same sort disagree on the width.
    InconsistentWidth { sort: String, a: u32, b: u32 },
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::InconsistentWidth { sort, a, b } => {
                write!(f, "sort {sort} is used at widths {a} and {b}")
            }
        }
    }
}

fn q(name: &str) -> String {
    format!("|{}|", name.replace('|', "_"))
}

/// Bitvector width per sort, inferred from the terms.
fn widths(env: &SmtEnv) -> Result<HashMap<u32, u32>, ExportError> {
    let mut w: HashMap<u32, u32> = HashMap::new();
    for i in 0..env.arena.len() {
        let e = env.arena.get(TermId(i as u32));
        let width = match &e.kind {
            TermKind::BvConst { width, .. } | TermKind::BvOp { width, .. } => *width,
            _ => continue,
        };
        match w.get(&e.sort.0) {
            Some(&have) if have != width => {
                return Err(ExportError::InconsistentWidth {
                    sort: env.sorts[e.sort.0 as usize].name.clone(),
                    a: have,
                    b: width,
                })
            }
            _ => {
                w.insert(e.sort.0, width);
            }
        }
    }
    Ok(w)
}

fn sort_name(env: &SmtEnv, s: SortId) -> String {
    q(&env.sorts[s.0 as usize].name)
}

fn term(env: &SmtEnv, t: TermId, out: &mut String) {
    let e = env.arena.get(t);
    match &e.kind {
        TermKind::Variable { name, .. } => out.push_str(&q(name)),
        TermKind::Apply { func, args } => {
            let f = &env.func_decls[func.0 as usize];
            if args.is_empty() {
                out.push_str(&q(&f.name));
            } else {
                out.push('(');
                out.push_str(&q(&f.name));
                for &a in args {
                    out.push(' ');
                    term(env, a, out);
                }
                out.push(')');
            }
        }
        TermKind::BvConst { width, value } => {
            let _ = write!(out, "(_ bv{value} {width})");
        }
        TermKind::BvOp { op, args, .. } => {
            let name = match op {
                BvOpKind::Add => "bvadd",
                BvOpKind::And => "bvand",
                BvOpKind::Or => "bvor",
                BvOpKind::Xor => "bvxor",
                BvOpKind::Not => "bvnot",
                BvOpKind::Sub => "bvsub",
                BvOpKind::Concat => "concat",
                BvOpKind::Extract { hi, lo } => {
                    let _ = write!(out, "((_ extract {hi} {lo}) ");
                    term(env, args[0], out);
                    out.push(')');
                    return;
                }
            };
            // Variadic ops are folded left so the text is unambiguous.
            if args.len() <= 2 {
                out.push('(');
                out.push_str(name);
                for &a in args {
                    out.push(' ');
                    term(env, a, out);
                }
                out.push(')');
            } else {
                let mut acc = String::new();
                acc.push_str(&format!("({name} "));
                term(env, args[0], &mut acc);
                acc.push(' ');
                term(env, args[1], &mut acc);
                acc.push(')');
                for &a in &args[2..] {
                    let mut next = format!("({name} {acc} ");
                    term(env, a, &mut next);
                    next.push(')');
                    acc = next;
                }
                out.push_str(&acc);
            }
        }
    }
}

fn formula(env: &SmtEnv, f: &SmtFormula, out: &mut String) {
    match f {
        SmtFormula::Eq(a, b) => {
            out.push_str("(= ");
            term(env, *a, out);
            out.push(' ');
            term(env, *b, out);
            out.push(')');
        }
        SmtFormula::Neq(a, b) => {
            out.push_str("(distinct ");
            term(env, *a, out);
            out.push(' ');
            term(env, *b, out);
            out.push(')');
        }
        SmtFormula::Not(g) => {
            out.push_str("(not ");
            formula(env, g, out);
            out.push(')');
        }
        SmtFormula::And(gs) if gs.is_empty() => out.push_str("true"),
        SmtFormula::Or(gs) if gs.is_empty() => out.push_str("false"),
        SmtFormula::And(gs) | SmtFormula::Or(gs) => {
            out.push_str(if matches!(f, SmtFormula::And(_)) {
                "(and"
            } else {
                "(or"
            });
            for g in gs {
                out.push(' ');
                formula(env, g, out);
            }
            out.push(')');
        }
        SmtFormula::Implies(p, c) => {
            out.push_str("(=> ");
            formula(env, p, out);
            out.push(' ');
            formula(env, c, out);
            out.push(')');
        }
    }
}

/// The whole problem as an SMT-LIB 2 script ending in `(check-sat)`.
pub(crate) fn export(env: &SmtEnv) -> Result<String, ExportError> {
    let w = widths(env)?;
    let mut has_uf = false;
    let mut has_bv = !w.is_empty();
    for i in 0..env.arena.len() {
        match &env.arena.get(TermId(i as u32)).kind {
            TermKind::Apply { .. } => has_uf = true,
            TermKind::BvConst { .. } | TermKind::BvOp { .. } => has_bv = true,
            TermKind::Variable { sort, .. } if !w.contains_key(&sort.0) => has_uf = true,
            _ => {}
        }
    }
    let logic = match (has_uf, has_bv) {
        (true, true) => "QF_UFBV",
        (false, true) => "QF_BV",
        _ => "QF_UF",
    };
    let mut out = String::new();
    let _ = writeln!(out, "(set-logic {logic})");
    for (i, s) in env.sorts.iter().enumerate() {
        match w.get(&(i as u32)) {
            Some(width) => {
                let _ = writeln!(out, "(define-sort {} () (_ BitVec {width}))", q(&s.name));
            }
            None => {
                let _ = writeln!(out, "(declare-sort {} 0)", q(&s.name));
            }
        }
    }
    for f in &env.func_decls {
        let args: Vec<String> = f.arg_sorts.iter().map(|&s| sort_name(env, s)).collect();
        let _ = writeln!(
            out,
            "(declare-fun {} ({}) {})",
            q(&f.name),
            args.join(" "),
            sort_name(env, f.ret_sort)
        );
    }
    for i in 0..env.arena.len() {
        let e = env.arena.get(TermId(i as u32));
        if let TermKind::Variable { name, sort } = &e.kind {
            let _ = writeln!(out, "(declare-const {} {})", q(name), sort_name(env, *sort));
        }
    }
    for a in &env.assertions {
        out.push_str("(assert ");
        formula(env, a, &mut out);
        out.push_str(")\n");
    }
    out.push_str("(check-sat)\n");
    Ok(out)
}
