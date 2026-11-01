//! DRAT proof logging for UNSAT answers.
//!
//! A CDCL solver's `Unsat` is only as trustworthy as the solver. A DRAT proof
//! (Wetzler, Heule & Hunt 2014) makes the answer checkable by an independent
//! tool: every clause the solver learns is written as it is added, every
//! clause it drops is written as a deletion, and the empty clause closes the
//! proof. Each learned clause of 1-UIP conflict analysis is a RUP clause
//! (reverse unit propagation) with respect to the clauses still in the
//! database, so the plain DRAT text format suffices; no RAT steps are emitted.
//!
//! Where the solver logs (`solver::solve_cdcl_core_inner`):
//! - the learned clause of every conflict, before it is added to the database;
//! - every clause `reduce_learned` tombstones, before compaction frees it;
//! - the empty clause at every `Unsat` return;
//! - theory conflict clauses added to the database (DPLL(T) only). These are
//!   *not* RUP over the CNF, so a proof from a theory-driven solve will not
//!   check with a DRAT checker; the log is complete as a trace, not as a proof.
//!   Use [`crate::solver::solve_watched_drat`] (pure SAT) for a checkable proof.
//!
//! Variables are written 1-based as in DIMACS: `Lit` variable `v` is `v + 1`,
//! negated literals carry a minus sign. The proof refers to the variables of
//! the *input* formula, so it is checked against the DIMACS file the clause
//! database was parsed from.
//!
//! [`check_drat`] is a forward RUP checker used by the tests: it is an
//! independent reimplementation of the check (unit propagation over the
//! clause set, not the solver's watched BCP), but it lives in this crate; an
//! external checker (`drat-trim`, `cake_lpr`) remains the real test.

use crate::literal::Lit;
use std::io::{self, Write};

/// Receiver for proof steps. The solver core calls these as it runs.
pub trait ProofLog {
    /// A clause was added (learned, or a theory lemma).
    fn add(&mut self, lits: &[Lit]);
    /// A clause was removed from the database.
    fn delete(&mut self, lits: &[Lit]);
    /// The solver concluded UNSAT: the empty clause is derivable.
    fn empty(&mut self) {
        self.add(&[]);
    }
}

/// No proof: every step is a no-op, and the compiler removes the calls when
/// the core is used through the ordinary entry points.
pub struct NoProof;

impl ProofLog for NoProof {
    #[inline]
    fn add(&mut self, _lits: &[Lit]) {}
    #[inline]
    fn delete(&mut self, _lits: &[Lit]) {}
}

/// DIMACS integer for a literal: variable `v` is `v + 1`, negated is negative.
#[inline]
pub fn dimacs_int(lit: Lit) -> i64 {
    let v = lit.var() as i64 + 1;
    if lit.is_negated() {
        -v
    } else {
        v
    }
}

/// Writes DRAT text to any `Write`. I/O errors are kept in the writer and
/// returned by [`DratWriter::finish`]; the solver never sees them.
pub struct DratWriter<W: Write> {
    out: W,
    err: Option<io::Error>,
    /// Steps written: (additions, deletions). Additions count the empty clause.
    pub additions: u64,
    pub deletions: u64,
}

impl<W: Write> DratWriter<W> {
    pub fn new(out: W) -> Self {
        DratWriter {
            out,
            err: None,
            additions: 0,
            deletions: 0,
        }
    }

    fn line(&mut self, prefix: &str, lits: &[Lit]) {
        if self.err.is_some() {
            return;
        }
        let mut s = String::with_capacity(prefix.len() + lits.len() * 6 + 2);
        s.push_str(prefix);
        for &l in lits {
            s.push_str(&dimacs_int(l).to_string());
            s.push(' ');
        }
        s.push_str("0\n");
        if let Err(e) = self.out.write_all(s.as_bytes()) {
            self.err = Some(e);
        }
    }

    /// Flush and hand back the writer, or the first I/O error that occurred.
    pub fn finish(mut self) -> io::Result<W> {
        if let Some(e) = self.err.take() {
            return Err(e);
        }
        self.out.flush()?;
        Ok(self.out)
    }
}

impl<W: Write> ProofLog for DratWriter<W> {
    fn add(&mut self, lits: &[Lit]) {
        self.additions += 1;
        self.line("", lits);
    }
    fn delete(&mut self, lits: &[Lit]) {
        self.deletions += 1;
        self.line("d ", lits);
    }
}

/// One parsed DRAT step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DratStep {
    Add(Vec<i64>),
    Delete(Vec<i64>),
}

/// Parse DRAT text (one step per line, `d` prefix for deletions, `0` terminated).
pub fn parse_drat(text: &str) -> Result<Vec<DratStep>, String> {
    let mut steps = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        let (is_del, body) = match line.strip_prefix('d') {
            Some(rest) => (true, rest),
            None => (false, line),
        };
        let mut lits = Vec::new();
        let mut terminated = false;
        for tok in body.split_whitespace() {
            let v: i64 = tok
                .parse()
                .map_err(|_| format!("line {}: bad literal {tok:?}", n + 1))?;
            if v == 0 {
                terminated = true;
                break;
            }
            lits.push(v);
        }
        if !terminated {
            return Err(format!("line {}: missing 0 terminator", n + 1));
        }
        steps.push(if is_del {
            DratStep::Delete(lits)
        } else {
            DratStep::Add(lits)
        });
    }
    Ok(steps)
}

/// Forward RUP check of a DRAT proof against a CNF given as DIMACS integer
/// clauses. Returns `Ok(())` iff every added clause is RUP with respect to the
/// clauses present at that point and the proof derives the empty clause.
///
/// Deletions of clauses not present are reported as errors (a solver that
/// deletes what it does not have is logging wrongly). Checking is
/// quadratic-ish and meant for test-sized instances.
pub fn check_drat(cnf: &[Vec<i64>], proof: &[DratStep]) -> Result<(), String> {
    // Active clause multiset, keyed by sorted literals.
    use std::collections::HashMap;
    let key = |c: &[i64]| {
        let mut k = c.to_vec();
        k.sort_unstable();
        k.dedup();
        k
    };
    let mut active: HashMap<Vec<i64>, usize> = HashMap::new();
    for c in cnf {
        *active.entry(key(c)).or_insert(0) += 1;
    }
    let mut derived_empty = cnf.iter().any(|c| c.is_empty());

    for (i, step) in proof.iter().enumerate() {
        match step {
            DratStep::Delete(c) => {
                let k = key(c);
                match active.get_mut(&k) {
                    Some(n) if *n > 0 => {
                        *n -= 1;
                        if *n == 0 {
                            active.remove(&k);
                        }
                    }
                    _ => return Err(format!("step {}: deletes absent clause {c:?}", i + 1)),
                }
            }
            DratStep::Add(c) => {
                if !rup(&active, c) {
                    return Err(format!("step {}: clause {c:?} is not RUP", i + 1));
                }
                if c.is_empty() {
                    derived_empty = true;
                    break;
                }
                *active.entry(key(c)).or_insert(0) += 1;
            }
        }
    }
    if derived_empty {
        Ok(())
    } else {
        Err("proof does not derive the empty clause".into())
    }
}

/// Is `clause` RUP w.r.t. `active`: does asserting its negation and running
/// unit propagation to fixpoint over `active` produce a conflict?
fn rup(active: &std::collections::HashMap<Vec<i64>, usize>, clause: &[i64]) -> bool {
    use std::collections::HashSet;
    let mut assigned: HashSet<i64> = HashSet::new(); // literals assumed true
    for &l in clause {
        if assigned.contains(&l) {
            return true; // negation of a tautology is contradictory
        }
        assigned.insert(-l);
    }
    loop {
        let mut progress = false;
        for c in active.keys() {
            let mut unassigned: Option<i64> = None;
            let mut satisfied = false;
            let mut count = 0;
            for &l in c {
                if assigned.contains(&l) {
                    satisfied = true;
                    break;
                }
                if !assigned.contains(&-l) {
                    count += 1;
                    unassigned = Some(l);
                }
            }
            if satisfied {
                continue;
            }
            match count {
                0 => return true, // falsified clause: conflict
                1 => {
                    assigned.insert(unassigned.unwrap());
                    progress = true;
                }
                _ => {}
            }
        }
        if !progress {
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{generate_3sat_phase_transition, pigeonhole_dimacs};
    use crate::dimacs::parse_dimacs_str;
    use crate::solver::{solve_watched, solve_watched_drat, SolveResult};

    fn cnf_ints(text: &str) -> Vec<Vec<i64>> {
        text.lines()
            .filter(|l| !l.starts_with('p') && !l.starts_with('c') && !l.trim().is_empty())
            .map(|l| {
                l.split_whitespace()
                    .map(|t| t.parse::<i64>().unwrap())
                    .take_while(|&v| v != 0)
                    .collect()
            })
            .collect()
    }

    fn drat_for(text: &str) -> (SolveResult, String) {
        let inst = parse_dimacs_str(text).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        let r = solve_watched_drat(inst.db, inst.num_vars, &mut buf).unwrap();
        (r, String::from_utf8(buf).unwrap())
    }

    #[test]
    fn dimacs_mapping() {
        assert_eq!(dimacs_int(Lit::pos(0)), 1);
        assert_eq!(dimacs_int(Lit::neg(0)), -1);
        assert_eq!(dimacs_int(Lit::neg(41)), -42);
    }

    #[test]
    fn writer_format() {
        let mut w = DratWriter::new(Vec::new());
        w.add(&[Lit::pos(0), Lit::neg(2)]);
        w.delete(&[Lit::neg(1)]);
        w.empty();
        let out = String::from_utf8(w.finish().unwrap()).unwrap();
        assert_eq!(out, "1 -3 0\nd -2 0\n0\n");
    }

    #[test]
    fn checker_accepts_a_hand_proof_and_rejects_a_bad_one() {
        // (a ∨ b) (¬a ∨ b) (a ∨ ¬b) (¬a ∨ ¬b): learn b, learn ¬b, empty.
        let cnf = vec![vec![1, 2], vec![-1, 2], vec![1, -2], vec![-1, -2]];
        let good = parse_drat("2 0\n-2 0\n0\n").unwrap();
        assert_eq!(check_drat(&cnf, &good), Ok(()));
        // Over the satisfiable (a ∨ b) (¬a ∨ b), the unit a is not RUP:
        // assuming ¬a makes b unit and nothing conflicts.
        let sat_cnf = vec![vec![1, 2], vec![-1, 2]];
        let bad = parse_drat("1 0\n").unwrap();
        assert!(check_drat(&sat_cnf, &bad).unwrap_err().contains("not RUP"));
        // Over the UNSAT four-clause set every clause is RUP, including a.
        let also_fine = parse_drat("1 0\n0\n").unwrap();
        assert_eq!(check_drat(&cnf, &also_fine), Ok(()));
        let no_empty = parse_drat("2 0\n").unwrap();
        assert!(check_drat(&cnf, &no_empty).is_err());
        let bad_del = parse_drat("d 5 6 0\n").unwrap();
        assert!(check_drat(&cnf, &bad_del).unwrap_err().contains("absent"));
    }

    #[test]
    fn pigeonhole_proofs_check() {
        for n in 3..=6 {
            let text = pigeonhole_dimacs(n);
            let (r, proof) = drat_for(&text);
            assert_eq!(r, SolveResult::Unsat, "PHP({n})");
            let steps = parse_drat(&proof).unwrap();
            assert!(steps.len() > 1, "PHP({n}) proof has {} steps", steps.len());
            assert_eq!(check_drat(&cnf_ints(&text), &steps), Ok(()), "PHP({n})");
        }
    }

    #[test]
    fn pigeonhole_7_proof_checks_with_deletions() {
        // PHP(7) runs past the 2000-conflict reduce threshold, so the proof
        // carries deletions; the checker must still accept it.
        let text = pigeonhole_dimacs(7);
        let (r, proof) = drat_for(&text);
        assert_eq!(r, SolveResult::Unsat);
        let steps = parse_drat(&proof).unwrap();
        let dels = steps
            .iter()
            .filter(|s| matches!(s, DratStep::Delete(_)))
            .count();
        assert!(dels > 0, "expected deletions in the PHP(7) proof");
        assert_eq!(check_drat(&cnf_ints(&text), &steps), Ok(()));
    }

    #[test]
    fn random_unsat_proofs_check() {
        // Random 3-SAT at the phase transition; keep the UNSAT ones.
        let mut checked = 0;
        for seed in 0..40u64 {
            let db = generate_3sat_phase_transition(60, seed);
            let text = crate::bench::db_to_dimacs(&db, 60);
            let inst = parse_dimacs_str(&text).unwrap();
            if solve_watched(inst.db, inst.num_vars) != SolveResult::Unsat {
                continue;
            }
            let (r, proof) = drat_for(&text);
            assert_eq!(r, SolveResult::Unsat);
            let steps = parse_drat(&proof).unwrap();
            assert_eq!(check_drat(&cnf_ints(&text), &steps), Ok(()), "seed {seed}");
            checked += 1;
        }
        assert!(
            checked >= 5,
            "only {checked} UNSAT instances among 40 seeds"
        );
    }

    #[test]
    fn sat_answer_logs_no_empty_clause() {
        let (r, proof) = drat_for("p cnf 3 2\n1 2 0\n-1 3 0\n");
        assert!(matches!(r, SolveResult::Sat(_)));
        assert!(!proof.lines().any(|l| l.trim() == "0"));
    }

    #[test]
    fn empty_input_clause_is_a_one_line_proof() {
        // Built directly rather than parsed, so this test exercises the
        // core's own empty-clause check and does not depend on the parser
        // (which keeps empty clauses since 2026-11-01; dimacs.rs has its
        // own test for that).
        let mut db = crate::bcp::ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::pos(1)]);
        db.add_clause(vec![]);
        let mut buf: Vec<u8> = Vec::new();
        let r = solve_watched_drat(db, 2, &mut buf).unwrap();
        assert_eq!(r, SolveResult::Unsat);
        assert_eq!(String::from_utf8(buf).unwrap(), "0\n");
    }

    /// Checks PHP(6) and PHP(7) proofs with an external DRAT checker when one
    /// is available (`DRAT_TRIM=/path/to/drat-trim` or `drat-trim` on PATH);
    /// otherwise prints that it was skipped. The in-crate checker is not the
    /// real test; this is.
    #[test]
    fn external_drat_trim_if_present() {
        let bin = std::env::var("DRAT_TRIM").ok().or_else(|| {
            std::env::var_os("PATH").and_then(|p| {
                std::env::split_paths(&p)
                    .map(|d| d.join("drat-trim"))
                    .find(|f| f.is_file())
                    .map(|f| f.to_string_lossy().into_owned())
            })
        });
        let Some(bin) = bin else {
            eprintln!("external_drat_trim_if_present: SKIPPED (no drat-trim; set DRAT_TRIM)");
            return;
        };
        let dir = std::env::temp_dir().join(format!("warp-types-sat-drat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in [6u32, 7] {
            let text = pigeonhole_dimacs(n);
            let (r, proof) = drat_for(&text);
            assert_eq!(r, SolveResult::Unsat);
            let cnf = dir.join(format!("php{n}.cnf"));
            let prf = dir.join(format!("php{n}.drat"));
            std::fs::write(&cnf, &text).unwrap();
            std::fs::write(&prf, &proof).unwrap();
            let out = std::process::Command::new(&bin)
                .arg(&cnf)
                .arg(&prf)
                .output()
                .expect("run drat-trim");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                stdout.contains("s VERIFIED"),
                "drat-trim did not verify PHP({n}):\n{stdout}\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
