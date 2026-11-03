//! Bit-blasting: QF_BV through the certified SAT path.
//!
//! Every bitvector term becomes a vector of SAT literals (bit 0 = least
//! significant), every equality atom the conjunction of bit equivalences,
//! and the formula is Tseitin-encoded into a `warp-types-sat` clause set.
//! The verdict is then complete for QF_BV — unlike the ground-only module in
//! [`crate::bv`] — and on UNSAT the solver's own LRAT proof of the
//! bit-blasted CNF is available for `lrat-check` / `cake_lpr`.
//!
//! What a proof certifies: that the CNF written beside it is unsatisfiable.
//! The translation from terms to that CNF is this module's and is not itself
//! certified; the Yices cross-check on the same instances is what stands
//! against a translation bug. Pre-registration: `research/smt-bitblast.md`.

use crate::formula::SmtFormula;
use crate::session::SmtEnv;
use crate::term::{BvOpKind, TermId, TermKind};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write;
use warp_types_sat::bcp::ClauseDb;
use warp_types_sat::literal::Lit;
use warp_types_sat::solver::{
    solve_watched_lrat_with, solve_watched_with, LratStats, SolveResult, DEFAULT_REDUCE_INTERVAL,
};

/// Why a session could not be bit-blasted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitBlastError {
    /// An uninterpreted function application: not QF_BV.
    Uninterpreted(String),
    /// A variable of a sort with no bitvector width in this session.
    UnknownWidth(String),
    /// Operands of different widths where equal widths are required.
    WidthMismatch { op: String, expected: u32, got: u32 },
}

impl std::fmt::Display for BitBlastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitBlastError::Uninterpreted(n) => write!(f, "uninterpreted function {n}: not QF_BV"),
            BitBlastError::UnknownWidth(s) => write!(f, "sort {s} has no bitvector width"),
            BitBlastError::WidthMismatch { op, expected, got } => {
                write!(f, "{op}: width {got} where {expected} was required")
            }
        }
    }
}

/// The bit-blasted problem: a clause set plus the bookkeeping to write it
/// out and to build a solver database from it.
pub struct BitBlasted {
    /// Clauses, in order (the DIMACS and the solver's database agree on it,
    /// so an LRAT proof's original-clause ids are the DIMACS line numbers).
    pub clauses: Vec<Vec<Lit>>,
    /// Variables used (0-based in `Lit`, 1-based in DIMACS).
    pub num_vars: u32,
    /// Number of bitvector terms encoded (for the size bound).
    pub terms: usize,
    /// Sum of term widths (for the size bound).
    pub bits: u64,
}

struct Blaster {
    next_var: u32,
    clauses: Vec<Vec<Lit>>,
    true_lit: Lit,
    memo: HashMap<TermId, Vec<Lit>>,
    bits: u64,
    terms: usize,
}

impl Blaster {
    fn fresh(&mut self) -> Lit {
        let v = self.next_var;
        self.next_var += 1;
        Lit::pos(v)
    }
    fn constant(&self, b: bool) -> Lit {
        if b {
            self.true_lit
        } else {
            self.true_lit.complement()
        }
    }
    fn is_const(&self, l: Lit) -> Option<bool> {
        if l.var() == self.true_lit.var() {
            Some(l.is_negated() == self.true_lit.is_negated())
        } else {
            None
        }
    }
    fn and2(&mut self, a: Lit, b: Lit) -> Lit {
        match (self.is_const(a), self.is_const(b)) {
            (Some(false), _) | (_, Some(false)) => return self.constant(false),
            (Some(true), _) => return b,
            (_, Some(true)) => return a,
            _ => {}
        }
        if a == b {
            return a;
        }
        if a == b.complement() {
            return self.constant(false);
        }
        let g = self.fresh();
        self.clauses.push(vec![g.complement(), a]);
        self.clauses.push(vec![g.complement(), b]);
        self.clauses.push(vec![g, a.complement(), b.complement()]);
        g
    }
    fn or2(&mut self, a: Lit, b: Lit) -> Lit {
        self.and2(a.complement(), b.complement()).complement()
    }
    fn xor2(&mut self, a: Lit, b: Lit) -> Lit {
        match (self.is_const(a), self.is_const(b)) {
            (Some(false), _) => return b,
            (_, Some(false)) => return a,
            (Some(true), _) => return b.complement(),
            (_, Some(true)) => return a.complement(),
            _ => {}
        }
        if a == b {
            return self.constant(false);
        }
        if a == b.complement() {
            return self.constant(true);
        }
        let g = self.fresh();
        self.clauses.push(vec![g.complement(), a, b]);
        self.clauses
            .push(vec![g.complement(), a.complement(), b.complement()]);
        self.clauses.push(vec![g, a.complement(), b]);
        self.clauses.push(vec![g, a, b.complement()]);
        g
    }
    fn and_all(&mut self, lits: &[Lit]) -> Lit {
        let mut acc = self.constant(true);
        for &l in lits {
            acc = self.and2(acc, l);
        }
        acc
    }
    fn or_all(&mut self, lits: &[Lit]) -> Lit {
        let mut acc = self.constant(false);
        for &l in lits {
            acc = self.or2(acc, l);
        }
        acc
    }
    /// Ripple-carry `a + b + cin`, result truncated to `a.len()` bits.
    fn add(&mut self, a: &[Lit], b: &[Lit], cin: Lit) -> Vec<Lit> {
        let mut out = Vec::with_capacity(a.len());
        let mut carry = cin;
        for i in 0..a.len() {
            let x = self.xor2(a[i], b[i]);
            let s = self.xor2(x, carry);
            let c1 = self.and2(a[i], b[i]);
            let c2 = self.and2(x, carry);
            carry = self.or2(c1, c2);
            out.push(s);
        }
        out
    }

    fn term(
        &mut self,
        env: &SmtEnv,
        widths: &HashMap<u32, u32>,
        t: TermId,
    ) -> Result<Vec<Lit>, BitBlastError> {
        if let Some(v) = self.memo.get(&t) {
            return Ok(v.clone());
        }
        let e = env.arena.get(t);
        let bits: Vec<Lit> = match &e.kind {
            TermKind::Variable { sort, .. } => {
                let w = *widths.get(&sort.0).ok_or_else(|| {
                    BitBlastError::UnknownWidth(env.sorts[sort.0 as usize].name.clone())
                })?;
                (0..w).map(|_| self.fresh()).collect()
            }
            TermKind::Apply { func, .. } => {
                return Err(BitBlastError::Uninterpreted(
                    env.func_decls[func.0 as usize].name.clone(),
                ))
            }
            TermKind::BvConst { width, value } => (0..*width)
                .map(|i| self.constant((value >> i) & 1 == 1))
                .collect(),
            TermKind::BvOp { op, width, args } => {
                let w = *width;
                let mut argbits = Vec::with_capacity(args.len());
                for &a in args {
                    argbits.push(self.term(env, widths, a)?);
                }
                let same = |name: &str, bits: &[Vec<Lit>]| -> Result<(), BitBlastError> {
                    for b in bits {
                        if b.len() as u32 != w {
                            return Err(BitBlastError::WidthMismatch {
                                op: name.into(),
                                expected: w,
                                got: b.len() as u32,
                            });
                        }
                    }
                    Ok(())
                };
                match op {
                    BvOpKind::Add => {
                        same("bvadd", &argbits)?;
                        let zero = self.constant(false);
                        let mut acc = argbits[0].clone();
                        for b in &argbits[1..] {
                            acc = self.add(&acc, b, zero);
                        }
                        acc
                    }
                    BvOpKind::Sub => {
                        same("bvsub", &argbits)?;
                        let nb: Vec<Lit> = argbits[1].iter().map(|l| l.complement()).collect();
                        let one = self.constant(true);
                        self.add(&argbits[0], &nb, one)
                    }
                    BvOpKind::And | BvOpKind::Or | BvOpKind::Xor => {
                        same("bitwise", &argbits)?;
                        let mut acc = argbits[0].clone();
                        for b in &argbits[1..] {
                            acc = (0..w as usize)
                                .map(|i| match op {
                                    BvOpKind::And => self.and2(acc[i], b[i]),
                                    BvOpKind::Or => self.or2(acc[i], b[i]),
                                    _ => self.xor2(acc[i], b[i]),
                                })
                                .collect();
                        }
                        acc
                    }
                    BvOpKind::Not => {
                        same("bvnot", &argbits)?;
                        argbits[0].iter().map(|l| l.complement()).collect()
                    }
                    BvOpKind::Extract { hi, lo } => {
                        let src = &argbits[0];
                        if *hi as usize >= src.len() {
                            return Err(BitBlastError::WidthMismatch {
                                op: "extract".into(),
                                expected: hi + 1,
                                got: src.len() as u32,
                            });
                        }
                        src[*lo as usize..=*hi as usize].to_vec()
                    }
                    BvOpKind::Concat => {
                        // First argument is the high bits.
                        let mut v = argbits[1].clone();
                        v.extend_from_slice(&argbits[0]);
                        if v.len() as u32 != w {
                            return Err(BitBlastError::WidthMismatch {
                                op: "concat".into(),
                                expected: w,
                                got: v.len() as u32,
                            });
                        }
                        v
                    }
                }
            }
        };
        self.bits += bits.len() as u64;
        self.terms += 1;
        self.memo.insert(t, bits.clone());
        Ok(bits)
    }

    fn eq(
        &mut self,
        env: &SmtEnv,
        widths: &HashMap<u32, u32>,
        a: TermId,
        b: TermId,
    ) -> Result<Lit, BitBlastError> {
        let x = self.term(env, widths, a)?;
        let y = self.term(env, widths, b)?;
        if x.len() != y.len() {
            return Err(BitBlastError::WidthMismatch {
                op: "=".into(),
                expected: x.len() as u32,
                got: y.len() as u32,
            });
        }
        let eqs: Vec<Lit> = (0..x.len())
            .map(|i| self.xor2(x[i], y[i]).complement())
            .collect();
        Ok(self.and_all(&eqs))
    }

    fn formula(
        &mut self,
        env: &SmtEnv,
        widths: &HashMap<u32, u32>,
        f: &SmtFormula,
    ) -> Result<Lit, BitBlastError> {
        Ok(match f {
            SmtFormula::Eq(a, b) => self.eq(env, widths, *a, *b)?,
            SmtFormula::Neq(a, b) => self.eq(env, widths, *a, *b)?.complement(),
            SmtFormula::Not(g) => self.formula(env, widths, g)?.complement(),
            SmtFormula::And(gs) => {
                let mut ls = Vec::with_capacity(gs.len());
                for g in gs {
                    ls.push(self.formula(env, widths, g)?);
                }
                self.and_all(&ls)
            }
            SmtFormula::Or(gs) => {
                let mut ls = Vec::with_capacity(gs.len());
                for g in gs {
                    ls.push(self.formula(env, widths, g)?);
                }
                self.or_all(&ls)
            }
            SmtFormula::Implies(p, c) => {
                let np = self.formula(env, widths, p)?.complement();
                let c = self.formula(env, widths, c)?;
                self.or2(np, c)
            }
        })
    }
}

/// Bit-blast the session's assertions.
pub(crate) fn bitblast(env: &SmtEnv) -> Result<BitBlasted, BitBlastError> {
    let widths = crate::smtlib::widths(env).map_err(|e| match e {
        crate::smtlib::ExportError::InconsistentWidth { sort, a, b } => {
            BitBlastError::WidthMismatch {
                op: format!("sort {sort}"),
                expected: a,
                got: b,
            }
        }
    })?;
    let mut b = Blaster {
        next_var: 1,
        clauses: vec![vec![Lit::pos(0)]],
        true_lit: Lit::pos(0),
        memo: HashMap::new(),
        bits: 0,
        terms: 0,
    };
    for a in &env.assertions {
        let root = b.formula(env, &widths, a)?;
        b.clauses.push(vec![root]);
    }
    Ok(BitBlasted {
        clauses: b.clauses,
        num_vars: b.next_var,
        terms: b.terms,
        bits: b.bits,
    })
}

impl BitBlasted {
    /// A fresh solver database holding the clauses in order.
    pub fn db(&self) -> ClauseDb {
        let mut db = ClauseDb::new();
        for c in &self.clauses {
            db.add_clause(c.clone());
        }
        db
    }

    /// The CNF as DIMACS text (clause `i` of `clauses` is line `i` of the body,
    /// which is what an LRAT proof's original ids refer to).
    pub fn dimacs(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "p cnf {} {}", self.num_vars, self.clauses.len());
        for c in &self.clauses {
            for &l in c {
                let v = l.var() as i64 + 1;
                let _ = write!(s, "{} ", if l.is_negated() { -v } else { v });
            }
            s.push_str("0\n");
        }
        s
    }

    /// Solve without a proof.
    pub fn solve(&self) -> SolveResult {
        solve_watched_with(self.db(), self.num_vars, DEFAULT_REDUCE_INTERVAL)
    }

    /// Solve writing an LRAT proof (text, or binary) to `out`.
    pub fn solve_lrat(
        &self,
        out: &mut dyn Write,
        binary: bool,
    ) -> std::io::Result<(SolveResult, LratStats)> {
        solve_watched_lrat_with(
            self.db(),
            self.num_vars,
            out,
            DEFAULT_REDUCE_INTERVAL,
            binary,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{with_session, BvOpKind, SmtResult};
    use warp_types_sat::lrat::{check_sequential, parse_cnf_ints, parse_text};

    fn verdict(r: &SolveResult) -> SmtResult {
        match r {
            SolveResult::Sat(_) => SmtResult::Sat,
            SolveResult::Unsat => SmtResult::Unsat,
            SolveResult::Unknown => SmtResult::Unknown,
        }
    }

    /// The ground module's documented incompleteness witness: the bit-blaster
    /// decides it.
    #[test]
    fn width_one_witness_is_unsat() {
        let (ground, blasted) = with_session(|s| {
            let (s, bv) = s.declare_sort("BV1");
            let (s, x) = s.var("x", bv);
            let (s, zero) = s.bv_const(1, 0, bv);
            let (s, one) = s.bv_const(1, 1, bv);
            let asserted = s
                .finish_declarations()
                .assert_formula(SmtFormula::And(vec![
                    SmtFormula::Neq(x, zero),
                    SmtFormula::Neq(x, one),
                ]))
                .finish_assertions();
            let bb = asserted.bitblast().unwrap();
            (asserted.check_sat_bv(), verdict(&bb.solve()))
        });
        assert_eq!(
            ground,
            SmtResult::Sat,
            "the ground module is incomplete here, by design"
        );
        assert_eq!(blasted, SmtResult::Unsat);
    }

    /// Arithmetic against u64 on random constants, every op, widths 1..=8.
    #[test]
    fn ops_agree_with_u64_arithmetic() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..300 {
            let w = 1 + (rnd() % 8) as u32;
            let mask = (1u64 << w) - 1;
            let (a, b) = (rnd() & mask, rnd() & mask);
            let op = [
                BvOpKind::Add,
                BvOpKind::Sub,
                BvOpKind::And,
                BvOpKind::Or,
                BvOpKind::Xor,
                BvOpKind::Not,
            ][(rnd() % 6) as usize];
            let expect = match op {
                BvOpKind::Add => (a + b) & mask,
                BvOpKind::Sub => a.wrapping_sub(b) & mask,
                BvOpKind::And => a & b,
                BvOpKind::Or => a | b,
                BvOpKind::Xor => a ^ b,
                _ => !a & mask,
            };
            // Claim the WRONG value too: must be unsat.
            let wrong = (expect + 1) & mask;
            for (claim, want) in [(expect, SmtResult::Sat), (wrong, SmtResult::Unsat)] {
                let got = with_session(|s| {
                    let (s, bv) = s.declare_sort("BV");
                    let (s, ta) = s.bv_const(w, a, bv);
                    let (s, tb) = s.bv_const(w, b, bv);
                    let (s, tc) = s.bv_const(w, claim, bv);
                    let (s, t) = match op {
                        BvOpKind::Not => s.bv_op(op, w, &[ta], bv),
                        _ => s.bv_op(op, w, &[ta, tb], bv),
                    };
                    let asserted = s
                        .finish_declarations()
                        .assert_formula(SmtFormula::Eq(t, tc))
                        .finish_assertions();
                    verdict(&asserted.bitblast().unwrap().solve())
                });
                assert_eq!(got, want, "{op:?} w={w} a={a} b={b} claim={claim}");
            }
        }
    }

    /// Extract and concat wiring, and a variable solved for.
    #[test]
    fn extract_concat_and_variables() {
        let got = with_session(|s| {
            let (s, bv8) = s.declare_sort("BV8");
            let (s, bv4) = s.declare_sort("BV4");
            let (s, x) = s.var("x", bv8);
            let (s, hi) = s.bv_extract(7, 4, x, bv4);
            let (s, lo) = s.bv_extract(3, 0, x, bv4);
            let (s, swapped) = s.bv_concat(lo, 4, hi, 4, bv8);
            let (s, c) = s.bv_const(8, 0xA5, bv8);
            let (s, d) = s.bv_const(8, 0x5A, bv8);
            let asserted = s
                .finish_declarations()
                .assert_formula(SmtFormula::And(vec![
                    SmtFormula::Eq(x, c),
                    SmtFormula::Neq(swapped, d),
                ]))
                .finish_assertions();
            verdict(&asserted.bitblast().unwrap().solve())
        });
        assert_eq!(got, SmtResult::Unsat);
    }

    /// The LRAT of a bit-blasted UNSAT is accepted by the crate's checker
    /// against the DIMACS, and a corrupted copy is not.
    #[test]
    fn unsat_proof_checks_and_corruption_is_caught() {
        let (dimacs, proof) = with_session(|s| {
            let (s, bv) = s.declare_sort("BV4");
            let (s, x) = s.var("x", bv);
            let (s, y) = s.var("y", bv);
            let (s, one) = s.bv_const(4, 1, bv);
            let (s, xp1) = s.bv_op(BvOpKind::Add, 4, &[x, one], bv);
            let (s, yp1) = s.bv_op(BvOpKind::Add, 4, &[y, one], bv);
            let asserted = s
                .finish_declarations()
                .assert_formula(SmtFormula::And(vec![
                    SmtFormula::Eq(xp1, yp1),
                    SmtFormula::Neq(x, y),
                ]))
                .finish_assertions();
            let bb = asserted.bitblast().unwrap();
            let mut out = Vec::new();
            let (r, _) = bb.solve_lrat(&mut out, false).unwrap();
            assert_eq!(verdict(&r), SmtResult::Unsat);
            (bb.dimacs(), String::from_utf8(out).unwrap())
        });
        let cnf = parse_cnf_ints(&dimacs);
        let p = parse_text(&proof).unwrap();
        assert!(check_sequential(&cnf, &p).is_verified());
        // Drop one hint from the middle lemma: rejected.
        let mut lines: Vec<String> = proof.lines().map(String::from).collect();
        let mid = lines.len() / 2;
        let toks: Vec<&str> = lines[mid].split_whitespace().collect();
        if let Some(z) = toks.iter().position(|&t| t == "0") {
            if toks.len() > z + 2 && toks[1] != "d" {
                let mut t2 = toks.clone();
                t2.remove(z + 1);
                lines[mid] = t2.join(" ");
                let p2 = parse_text(&lines.join("\n")).unwrap();
                assert!(
                    !check_sequential(&cnf, &p2).is_verified(),
                    "corruption not caught"
                );
            }
        }
    }
}
