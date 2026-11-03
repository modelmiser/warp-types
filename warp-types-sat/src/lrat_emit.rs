//! LRAT emission from the CDCL core: clause ids and RUP hint chains.
//!
//! The chain for a learned clause is derived from the FINAL clause, not from
//! the order of resolution. Under the assumption "every literal of the
//! learned clause is false", every literal of every clause the derivation
//! touches must be either assumed or derived; so walk the implication graph
//! from the conflict clause and, for each literal that is false on the trail
//! and whose variable is not in the learned clause, take its reason clause and
//! recurse. Emit the reasons sorted by the trail position of the literal each
//! one implies, then the conflict clause. Trail order is implication order
//! (`Trail::position_of`), so every reason is unit when its turn comes and
//! the conflict clause is falsified last. Clause minimization needs no
//! special case — a literal it removed is simply "not in the learned clause"
//! and gets derived — and neither do level-0 literals (their unit reasons are
//! collected like any other). Pre-registration: `research/lrat-from-solver.md`.

use crate::bcp::{CRef, ClauseDb};
use crate::literal::Lit;
use crate::trail::{Reason, Trail};

/// `CRef → LRAT id`, for the clauses currently in the database. A `CRef` is
/// an arena offset, so a plain vector indexed by it (0 = no id) replaces the
/// hash map that was the single largest cost of the chain walk (every reason
/// of every conflict is looked up once).
pub(crate) struct LratIds {
    ids: Vec<u64>,
}

impl LratIds {
    /// Originals get `1..=n` in database (= file) order.
    pub(crate) fn new(db: &ClauseDb) -> Self {
        let mut ids = vec![0u64; db.arena_words() + 1];
        for (i, cref) in db.iter_crefs().enumerate() {
            ids[cref as usize] = i as u64 + 1;
        }
        LratIds { ids }
    }
    #[inline]
    pub(crate) fn id(&self, c: CRef) -> u64 {
        let id = self.ids.get(c as usize).copied().unwrap_or(0);
        debug_assert!(
            id != 0,
            "lrat: clause {c} has no id (added without logging?)"
        );
        id
    }
    pub(crate) fn insert(&mut self, c: CRef, id: u64) {
        let c = c as usize;
        if self.ids.len() <= c {
            self.ids.resize(c + 1 + self.ids.len() / 2, 0);
        }
        self.ids[c] = id;
    }
    pub(crate) fn remove(&mut self, c: CRef) -> Option<u64> {
        let slot = self.ids.get_mut(c as usize)?;
        let id = std::mem::take(slot);
        (id != 0).then_some(id)
    }
    /// After database compaction: `remap` is `(old, new)` for every surviving
    /// clause (the solver applies the same list to the trail's reasons).
    pub(crate) fn remap(&mut self, remap: &[(CRef, CRef)]) {
        let max_new = remap.iter().map(|&(_, n)| n as usize).max().unwrap_or(0);
        let mut next = vec![0u64; max_new + 1];
        let mut moved = 0usize;
        for &(old, new) in remap {
            let id = self.ids.get(old as usize).copied().unwrap_or(0);
            if id != 0 {
                next[new as usize] = id;
                moved += 1;
            }
        }
        debug_assert_eq!(
            moved,
            self.ids.iter().filter(|&&i| i != 0).count(),
            "lrat: clause(s) with ids were not in the compaction remap"
        );
        self.ids = next;
    }
}

/// Scratch for [`rup_chain`], reused across conflicts.
pub(crate) struct ChainScratch {
    seen: Vec<bool>,
    touched: Vec<u32>,
    stack: Vec<CRef>,
    out: Vec<(usize, u64)>,
    /// Literals met whose reason is a theory propagation or a decision
    /// outside the learned clause: the chain is incomplete. Counted, not
    /// panicked on, so a theory solve still produces a (rejectable) proof.
    pub(crate) unjustified_literals: u64,
}

impl ChainScratch {
    pub(crate) fn new(num_vars: usize) -> Self {
        ChainScratch {
            seen: vec![false; num_vars],
            touched: Vec::new(),
            stack: Vec::new(),
            out: Vec::new(),
            unjustified_literals: 0,
        }
    }
}

/// The RUP chain (clause ids) that refutes `learned` from the current trail,
/// ending with `conflict`. `learned` may be empty (the final conflict at
/// level 0).
pub(crate) fn rup_chain(
    trail: &Trail,
    db: &ClauseDb,
    conflict: CRef,
    learned: &[Lit],
    ids: &LratIds,
    sc: &mut ChainScratch,
) -> Vec<u64> {
    if sc.seen.len() < trail.num_vars() {
        sc.seen.resize(trail.num_vars(), false);
    }
    sc.touched.clear();
    sc.stack.clear();
    sc.out.clear();
    for &l in learned {
        let v = l.var() as usize;
        if !sc.seen[v] {
            sc.seen[v] = true;
            sc.touched.push(v as u32);
        }
    }
    sc.stack.push(conflict);
    while let Some(c) = sc.stack.pop() {
        for &l in db.clause(c).literals {
            let v = l.var() as usize;
            if sc.seen[v] {
                continue;
            }
            sc.seen[v] = true;
            sc.touched.push(v as u32);
            // `l` is false on the trail (the clause's implied literal, if this
            // is a reason, was marked before the clause was pushed).
            let pos = trail.position_of(v as u32);
            match pos.map(|p| trail.entries()[p].reason) {
                Some(Reason::Propagation(r)) => {
                    sc.out.push((pos.expect("assigned"), ids.id(r)));
                    sc.stack.push(r);
                }
                Some(Reason::Decision) | None => {
                    // A decision (or an unassigned variable) that the learned
                    // clause does not cover: 1-UIP analysis guarantees this
                    // cannot happen for pure SAT.
                    debug_assert!(
                        false,
                        "lrat: variable {v} is a decision outside the learned clause"
                    );
                    sc.unjustified_literals += 1;
                }
                Some(Reason::TheoryPropagation(_)) => sc.unjustified_literals += 1,
            }
        }
    }
    sc.out.sort_unstable_by_key(|&(pos, _)| pos);
    let mut chain: Vec<u64> = Vec::with_capacity(sc.out.len() + 1);
    chain.extend(sc.out.iter().map(|&(_, id)| id));
    chain.push(ids.id(conflict));
    for &v in &sc.touched {
        sc.seen[v as usize] = false;
    }
    chain
}

#[cfg(test)]
mod tests {
    use crate::bench::{parity_chain_dimacs, pigeonhole_dimacs};
    use crate::dimacs::parse_dimacs_str;
    use crate::lrat::{
        check_independent, check_sequential, parse_binary, parse_cnf_ints, parse_text, Verdict,
    };
    use crate::lrat_warp::check_warp;
    use crate::proof::fault::{arm, Fault};
    use crate::solver::{solve_watched_lrat_with, SolveResult};

    /// Random 3-SAT at a ratio past the threshold; most instances are UNSAT.
    fn random_3sat(n: u32, m: u32, seed: u64) -> String {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut s = format!("p cnf {n} {m}\n");
        for _ in 0..m {
            for _ in 0..3 {
                let v = (rnd() % n as u64) as i64 + 1;
                let l = if rnd() % 2 == 0 { v } else { -v };
                s.push_str(&format!("{l} "));
            }
            s.push_str("0\n");
        }
        s
    }

    /// Solve with LRAT logging; on UNSAT, every checker in the crate must
    /// verify the proof (text and binary); returns the verdicts' agreement.
    fn solve_and_check(text: &str, reduce_interval: u64) -> Option<Verdict> {
        let cnf = parse_cnf_ints(text);
        let mut buf = Vec::new();
        let inst = parse_dimacs_str(text).unwrap();
        let (r, st) =
            solve_watched_lrat_with(inst.db, inst.num_vars, &mut buf, reduce_interval, false)
                .unwrap();
        let SolveResult::Unsat = r else { return None };
        assert_eq!(st.unjustified, 0, "pure SAT: every clause has a chain");
        let proof = parse_text(std::str::from_utf8(&buf).unwrap()).expect("emitted LRAT parses");
        let seq = check_sequential(&cnf, &proof);
        let ind = check_independent(&cnf, &proof, 3);
        let warp = check_warp(&cnf, &proof);
        assert_eq!(ind, seq, "sequential vs independent");
        assert_eq!(warp, seq, "sequential vs warp");
        // Binary emission decodes to the same proof.
        let mut bin = Vec::new();
        let inst = parse_dimacs_str(text).unwrap();
        let (r2, _) =
            solve_watched_lrat_with(inst.db, inst.num_vars, &mut bin, reduce_interval, true)
                .unwrap();
        assert_eq!(r2, r);
        let pb = parse_binary(&bin).expect("emitted binary LRAT parses");
        assert_eq!(pb, proof, "binary and text emissions are the same proof");
        Some(seq)
    }

    #[test]
    fn pigeonhole_proofs_verify() {
        for n in 3..=7 {
            let v = solve_and_check(&pigeonhole_dimacs(n), 50).expect("PHP is UNSAT");
            assert!(v.is_verified(), "PHP({n}): {v:?}");
        }
    }

    #[test]
    fn parity_proofs_verify() {
        for k in 3..=12 {
            if let Some(v) = solve_and_check(&parity_chain_dimacs(k), 50) {
                assert!(v.is_verified(), "parity({k}): {v:?}");
            }
        }
    }

    #[test]
    fn random_3sat_proofs_verify_with_deletions_and_compaction() {
        let mut unsat = 0;
        for seed in 0..40u64 {
            // ratio 4.6: mostly UNSAT at n = 60; interval 20 forces reductions.
            if let Some(v) = solve_and_check(&random_3sat(60, 276, seed), 20) {
                assert!(v.is_verified(), "seed {seed}: {v:?}");
                unsat += 1;
            }
        }
        assert!(unsat >= 10, "need UNSAT instances to test ({unsat})");
    }

    #[test]
    fn empty_original_clause_and_unit_conflicts() {
        // An empty original clause: `a id 0 <its id> 0`.
        let v = solve_and_check("p cnf 2 3\n1 2 0\n0\n-1 0\n", 50).unwrap();
        assert!(v.is_verified(), "{v:?}");
        // Unit clauses that conflict at level 0.
        let v = solve_and_check("p cnf 2 4\n1 0\n-1 2 0\n-2 0\n1 2 0\n", 50).unwrap();
        assert!(v.is_verified(), "{v:?}");
        // Two-variable complete contradiction.
        let v = solve_and_check("p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n", 50).unwrap();
        assert!(v.is_verified(), "{v:?}");
    }

    /// Each LRAT fault must flip the in-crate verdict (the oracle is watched
    /// to fail). PHP(6) has well over ten learned clauses.
    #[test]
    fn lrat_faults_are_caught() {
        let text = pigeonhole_dimacs(6);
        let cnf = parse_cnf_ints(&text);
        for f in [
            Fault::DropHint,
            Fault::ReverseHints,
            Fault::WrongId,
            Fault::FlipLiteral,
        ] {
            let _armed = arm(f);
            let inst = parse_dimacs_str(&text).unwrap();
            let mut buf = Vec::new();
            let (r, _) =
                solve_watched_lrat_with(inst.db, inst.num_vars, &mut buf, 50, false).unwrap();
            assert_eq!(r, SolveResult::Unsat);
            let s = std::str::from_utf8(&buf).unwrap();
            let v = match parse_text(s) {
                Ok(p) => check_sequential(&cnf, &p),
                Err(_) => continue, // a parse error is a rejection too
            };
            assert!(!v.is_verified(), "{f:?}: the fault was not caught: {v:?}");
        }
        // And without a fault the same solve verifies (the faults are the
        // difference, not the instance).
        let v = solve_and_check(&text, 50).unwrap();
        assert!(v.is_verified());
    }
}
