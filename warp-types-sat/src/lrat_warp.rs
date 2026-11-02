//! Warp-per-lemma LRAT checking on `SimWarp` — the organisation the GPU
//! kernel uses, run on the CPU with real 32-lane semantics.
//!
//! The lemma-independent formulation of [`crate::lrat::check_independent`]
//! is reorganised for a warp: one warp checks one lemma; for each hint clause
//! the 32 lanes stride over its literals, each lane counts how many of its
//! literals are *not* falsified under the lemma's running assignment, a
//! butterfly reduction sums those counts across the warp, and a ballot of
//! "this lane holds a free literal" picks the lane that owns the unit when
//! the total is one, whose literal is then broadcast to the warp by
//! `shuffle_idx`. Total zero is the conflict that makes the lemma RUP; total
//! two or more is a failure. The running assignment is a fixed-capacity
//! open-addressing table of literal codes — the shape it will have in shared
//! memory on the device — and overflowing it is a distinct verdict, never a
//! pass.
//!
//! Everything device-facing is flat and index-based ([`FlatProof`]): clause
//! literals in one array with start offsets, lemma literal and hint ranges,
//! hints as *dense clause indices* (not ids), a deletion step per clause.
//! The host builds it once from a [`Proof`]; the warp code reads only it.
//!
//! The checker must agree with the reference on every input; the tests hold
//! it to that on the hand cases and, when `LRAT_ARCHIVE_DIRS` is set, on the
//! archive. Fault injection (`cfg(test)`) shows each oracle failing.

use crate::lrat::{Proof, Reason, Step, Verdict};
use std::collections::HashMap;
use warp_types::simwarp::{butterfly_reduce, SimWarp};

const WARP: usize = 32;

/// Device-friendly proof layout. All indices are dense (0-based); `u32::MAX`
/// in `deleted_at` means "never deleted".
#[derive(Debug, Clone)]
pub struct FlatProof {
    /// All clause literals, originals first then lemmas in proof order.
    pub lits: Vec<i32>,
    /// `clause_start[c]..clause_start[c+1]` is clause `c`'s literal range.
    pub clause_start: Vec<u32>,
    /// Step index at which clause `c` is deleted, or `u32::MAX`.
    pub deleted_at: Vec<u32>,
    /// Per lemma: dense clause index of the lemma itself.
    pub lemma_clause: Vec<u32>,
    /// Per lemma: proof step index (for the deletion test).
    pub lemma_step: Vec<u32>,
    /// Per lemma: external id (for reporting).
    pub lemma_id: Vec<u64>,
    /// `hint_start[i]..hint_start[i+1]` is lemma `i`'s hint range.
    pub hint_start: Vec<u32>,
    /// Hints as dense clause indices.
    pub hints: Vec<u32>,
    /// Index (into lemmas) of the first empty lemma, if any.
    pub first_empty: Option<u32>,
    /// Number of original clauses (dense indices `0..n_original`).
    pub n_original: u32,
}

/// Why flattening refused a proof (these are checked sequentially on the
/// host because they are about the step sequence, not about any lemma).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlattenError {
    IdNotIncreasing { step: usize, id: u64 },
    UnknownHint { step: usize, id: u64, hint: u64 },
    BadDeletion { step: usize, id: u64 },
    RatUnsupported { step: usize, id: u64 },
}

impl FlattenError {
    /// The reference checker's verdict for the same defect.
    pub fn as_verdict(&self) -> Verdict {
        match *self {
            FlattenError::IdNotIncreasing { step, id } => Verdict::Rejected {
                step,
                id,
                reason: Reason::IdNotIncreasing,
            },
            FlattenError::UnknownHint { step, id, hint } => Verdict::Rejected {
                step,
                id,
                reason: Reason::UnknownHint(hint),
            },
            FlattenError::BadDeletion { step, id } => Verdict::Rejected {
                step,
                id,
                reason: Reason::BadDeletion(id),
            },
            FlattenError::RatUnsupported { step, id } => Verdict::Unsupported { step, id },
        }
    }
}

/// Build the flat layout. Hints that name an id never added at all are
/// refused here (the warp code has no id table); hints that name a *later*
/// id or a deleted clause are left to the per-lemma check, which has the
/// dense index order and the deletion steps to decide them.
pub fn flatten(cnf: &[Vec<i32>], proof: &Proof) -> Result<FlatProof, FlattenError> {
    let mut fp = FlatProof {
        lits: Vec::new(),
        clause_start: vec![0],
        deleted_at: Vec::new(),
        lemma_clause: Vec::new(),
        lemma_step: Vec::new(),
        lemma_id: Vec::new(),
        hint_start: vec![0],
        hints: Vec::new(),
        first_empty: None,
        n_original: cnf.len() as u32,
    };
    let mut dense_of: HashMap<u64, u32> = HashMap::with_capacity(cnf.len() * 2);
    for (i, c) in cnf.iter().enumerate() {
        fp.lits.extend_from_slice(c);
        fp.clause_start.push(fp.lits.len() as u32);
        fp.deleted_at.push(u32::MAX);
        dense_of.insert(i as u64 + 1, i as u32);
    }
    let mut last_id = cnf.len() as u64;
    // Hints may name ids that are added *later* in the file (a forward hint,
    // rejected per lemma) — resolve ids in a first pass so the dense index
    // exists, then fill hints in a second pass.
    for (step_ix, step) in proof.steps.iter().enumerate() {
        match step {
            Step::Delete { .. } => {}
            Step::Add { id, lits, hints } => {
                if *id <= last_id {
                    return Err(FlattenError::IdNotIncreasing {
                        step: step_ix,
                        id: *id,
                    });
                }
                if hints.iter().any(|&h| h < 0) {
                    return Err(FlattenError::RatUnsupported {
                        step: step_ix,
                        id: *id,
                    });
                }
                last_id = *id;
                let dense = fp.clause_start.len() as u32 - 1;
                dense_of.insert(*id, dense);
                fp.lits.extend_from_slice(lits);
                fp.clause_start.push(fp.lits.len() as u32);
                fp.deleted_at.push(u32::MAX);
                fp.lemma_clause.push(dense);
                fp.lemma_step.push(step_ix as u32);
                fp.lemma_id.push(*id);
                if lits.is_empty() && fp.first_empty.is_none() {
                    fp.first_empty = Some(fp.lemma_clause.len() as u32 - 1);
                }
            }
        }
    }
    // Deletions and hints.
    let mut lemma_ix = 0usize;
    for (step_ix, step) in proof.steps.iter().enumerate() {
        match step {
            Step::Delete { ids } => {
                for &d in ids {
                    let Some(&dense) = dense_of.get(&d) else {
                        return Err(FlattenError::BadDeletion {
                            step: step_ix,
                            id: d,
                        });
                    };
                    // A deletion of something added later in the file, or
                    // deleted twice, is a bad deletion in the reference too.
                    let added_at = if dense < fp.n_original {
                        0
                    } else {
                        fp.lemma_step[(dense - fp.n_original) as usize] as usize
                    };
                    if added_at > step_ix || fp.deleted_at[dense as usize] != u32::MAX {
                        return Err(FlattenError::BadDeletion {
                            step: step_ix,
                            id: d,
                        });
                    }
                    fp.deleted_at[dense as usize] = step_ix as u32;
                }
            }
            Step::Add { id, hints, .. } => {
                for &h in hints {
                    let Some(&dense) = dense_of.get(&(h as u64)) else {
                        return Err(FlattenError::UnknownHint {
                            step: step_ix,
                            id: *id,
                            hint: h as u64,
                        });
                    };
                    fp.hints.push(dense);
                }
                fp.hint_start.push(fp.hints.len() as u32);
                lemma_ix += 1;
                if Some(lemma_ix as u32 - 1) == fp.first_empty {
                    // Lemmas after the first empty one are not checked.
                    break;
                }
            }
        }
    }
    // Truncate lemma arrays to those up to the first empty lemma.
    if let Some(fe) = fp.first_empty {
        let keep = fe as usize + 1;
        fp.lemma_clause.truncate(keep);
        fp.lemma_step.truncate(keep);
        fp.lemma_id.truncate(keep);
        fp.hint_start.truncate(keep + 1);
    }
    Ok(fp)
}

/// Per-lemma outcome from the warp check, as a small code (device-friendly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LemmaCode {
    Ok = 0,
    NotUnit = 1,
    NoConflict = 2,
    DeletedHint = 3,
    ForwardHint = 4,
    /// The per-warp assignment table filled up: no verdict from this path.
    TableOverflow = 5,
}

/// Fixed-capacity open-addressing set of literal codes (the shared-memory
/// shape). Keys are `code(l) + 1` so that 0 is empty.
struct LitTable {
    slots: Vec<u32>,
    count: usize,
}

impl LitTable {
    const CAP: usize = 4096;
    fn new() -> Self {
        LitTable {
            slots: vec![0; Self::CAP],
            count: 0,
        }
    }
    #[inline]
    fn code(l: i32) -> u32 {
        ((l.unsigned_abs()) << 1) | u32::from(l < 0)
    }
    #[inline]
    fn hash(k: u32) -> usize {
        (k.wrapping_mul(0x9E37_79B1) as usize) & (Self::CAP - 1)
    }
    /// Insert the code of `l` (marking `l` false). Returns false on overflow.
    fn insert(&mut self, l: i32) -> bool {
        if self.count >= Self::CAP - 1 {
            return false;
        }
        let k = Self::code(l) + 1;
        let mut i = Self::hash(k);
        loop {
            if self.slots[i] == 0 {
                self.slots[i] = k;
                self.count += 1;
                return true;
            }
            if self.slots[i] == k {
                return true;
            }
            i = (i + 1) & (Self::CAP - 1);
        }
    }
    /// Is `l` falsified (its code present)?
    #[inline]
    fn falsified(&self, l: i32) -> bool {
        let k = Self::code(l) + 1;
        let mut i = Self::hash(k);
        loop {
            let s = self.slots[i];
            if s == 0 {
                return false;
            }
            if s == k {
                return true;
            }
            i = (i + 1) & (Self::CAP - 1);
        }
    }
    fn clear(&mut self) {
        if self.count > 0 {
            self.slots.iter_mut().for_each(|s| *s = 0);
            self.count = 0;
        }
    }
}

/// Test-only fault injection for the warp checker. No-ops outside tests.
pub mod fault {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    #[repr(u8)]
    pub enum Fault {
        None = 0,
        /// The free-literal total is read one too low: a two-free hint
        /// looks like a unit (and a unit like a conflict).
        CountOffByOne = 1,
        /// The deletion test is skipped: deleted hints are admitted.
        SkipDeletionTest = 2,
        /// The lane owning the unit is taken as lane 0 regardless of the
        /// ballot: the wrong literal gets propagated.
        WrongUnitLane = 3,
    }
    #[cfg(test)]
    thread_local! {
        static FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    }
    #[cfg(test)]
    pub fn arm(f: Fault) -> Armed {
        FAULT.with(|c| c.set(f as u8));
        Armed
    }
    #[cfg(test)]
    pub struct Armed;
    #[cfg(test)]
    impl Drop for Armed {
        fn drop(&mut self) {
            FAULT.with(|c| c.set(0));
        }
    }
    #[inline]
    pub fn active(f: Fault) -> bool {
        #[cfg(test)]
        {
            FAULT.with(|c| c.get()) == f as u8
        }
        #[cfg(not(test))]
        {
            let _ = f;
            false
        }
    }
}

/// Check lemma `i` of `fp` with one simulated warp.
fn warp_check_lemma(fp: &FlatProof, i: usize, table: &mut LitTable) -> LemmaCode {
    use fault::{active, Fault};
    table.clear();
    let lemma_dense = fp.lemma_clause[i] as usize;
    let lemma_step = fp.lemma_step[i];
    let (ls, le) = (
        fp.clause_start[lemma_dense] as usize,
        fp.clause_start[lemma_dense + 1] as usize,
    );
    for &l in &fp.lits[ls..le] {
        if !table.insert(l) {
            return LemmaCode::TableOverflow;
        }
    }
    let (hs, he) = (fp.hint_start[i] as usize, fp.hint_start[i + 1] as usize);
    for &h in &fp.hints[hs..he] {
        let h = h as usize;
        // Admissibility (warp-uniform: every lane evaluates the same test).
        if h >= lemma_dense {
            return LemmaCode::ForwardHint;
        }
        if !active(Fault::SkipDeletionTest) && fp.deleted_at[h] < lemma_step {
            return LemmaCode::DeletedHint;
        }
        let (cs, ce) = (fp.clause_start[h] as usize, fp.clause_start[h + 1] as usize);
        let clause = &fp.lits[cs..ce];
        // Each lane strides over the clause: count its free literals and
        // remember one of them.
        let per_lane = SimWarp::<(u32, i32)>::new(|lane| {
            let mut free = 0u32;
            let mut last = 0i32;
            let mut k = lane as usize;
            while k < clause.len() {
                if !table.falsified(clause[k]) {
                    free += 1;
                    last = clause[k];
                }
                k += WARP;
            }
            (free, last)
        });
        // Warp total of free literals (butterfly: every lane ends with the sum).
        let counts = SimWarp::<u32>::new(|lane| per_lane.lane(lane as usize).0);
        let total = butterfly_reduce(&counts, |a, b| a + b).lane(0);
        let total = if active(Fault::CountOffByOne) {
            total.saturating_sub(1)
        } else {
            total
        };
        match total {
            0 => return LemmaCode::Ok, // conflict: the lemma is RUP
            1 => {
                // Ballot: which lane owns the free literal? Then broadcast it.
                let mut ballot = 0u32;
                for lane in 0..WARP {
                    if counts.lane(lane) > 0 {
                        ballot |= 1 << lane;
                    }
                }
                let src = if active(Fault::WrongUnitLane) {
                    0
                } else {
                    ballot.trailing_zeros()
                };
                let unit = per_lane.shuffle_idx(src).lane(0).1;
                if !table.insert(-unit) {
                    return LemmaCode::TableOverflow;
                }
            }
            _ => return LemmaCode::NotUnit,
        }
    }
    LemmaCode::NoConflict
}

/// Check every lemma with the warp formulation and combine into a verdict
/// identical in shape to the reference's. `TableOverflow` on any lemma is
/// reported as that lemma's rejection with reason `NoConflict` — wrong in
/// spirit, so it is surfaced separately by [`check_warp_codes`]; the
/// agreement tests require that no archive proof overflows.
pub fn check_warp(cnf: &[Vec<i32>], proof: &Proof) -> Verdict {
    let fp = match flatten(cnf, proof) {
        Ok(fp) => fp,
        Err(e) => return e.as_verdict(),
    };
    let codes = check_warp_codes(&fp);
    combine(&fp, &codes)
}

/// Per-lemma codes (what the device returns).
pub fn check_warp_codes(fp: &FlatProof) -> Vec<LemmaCode> {
    let mut table = LitTable::new();
    (0..fp.lemma_clause.len())
        .map(|i| warp_check_lemma(fp, i, &mut table))
        .collect()
}

/// Host-side combination: earliest failing lemma wins; verified iff all
/// lemmas up to the first empty one are `Ok` and such a lemma exists.
pub fn combine(fp: &FlatProof, codes: &[LemmaCode]) -> Verdict {
    for (i, &c) in codes.iter().enumerate() {
        if c == LemmaCode::Ok {
            continue;
        }
        let step = fp.lemma_step[i] as usize;
        let id = fp.lemma_id[i];
        let hint_at = |pred: &dyn Fn(usize) -> bool| -> u64 {
            let (hs, he) = (fp.hint_start[i] as usize, fp.hint_start[i + 1] as usize);
            fp.hints[hs..he]
                .iter()
                .map(|&h| h as usize)
                .find(|&h| pred(h))
                .map(|h| dense_id(fp, h))
                .unwrap_or(0)
        };
        let reason = match c {
            LemmaCode::Ok => unreachable!(),
            LemmaCode::NotUnit => Reason::NotUnit(0),
            LemmaCode::NoConflict | LemmaCode::TableOverflow => Reason::NoConflict,
            LemmaCode::DeletedHint => {
                Reason::DeletedHint(hint_at(&|h| fp.deleted_at[h] < fp.lemma_step[i]))
            }
            LemmaCode::ForwardHint => {
                Reason::ForwardHint(hint_at(&|h| h >= fp.lemma_clause[i] as usize))
            }
        };
        return Verdict::Rejected { step, id, reason };
    }
    match fp.first_empty {
        Some(_) => Verdict::Verified {
            lemmas: codes.len(),
            deletions: fp.deleted_at.iter().filter(|&&d| d != u32::MAX).count(),
        },
        None => Verdict::NoEmptyClause {
            lemmas: codes.len(),
        },
    }
}

/// External id of a dense clause index.
fn dense_id(fp: &FlatProof, dense: usize) -> u64 {
    if (dense as u32) < fp.n_original {
        dense as u64 + 1
    } else {
        fp.lemma_id
            .get(dense - fp.n_original as usize)
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lrat::{check_sequential, parse_cnf_ints, parse_text};

    const CNF: &str = "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n";

    /// The two verdicts must agree up to the hint id inside `NotUnit`
    /// (the warp code reports 0 there; the reference names the hint).
    fn same(a: &Verdict, b: &Verdict) -> bool {
        match (a, b) {
            (
                Verdict::Rejected {
                    step: s1,
                    id: i1,
                    reason: Reason::NotUnit(_),
                },
                Verdict::Rejected {
                    step: s2,
                    id: i2,
                    reason: Reason::NotUnit(_),
                },
            ) => s1 == s2 && i1 == i2,
            _ => a == b,
        }
    }

    fn agree(cnf: &str, proof: &str) -> Verdict {
        let c = parse_cnf_ints(cnf);
        let p = parse_text(proof).unwrap();
        let r = check_sequential(&c, &p);
        let w = check_warp(&c, &p);
        assert!(same(&r, &w), "reference {r:?} vs warp {w:?}");
        w
    }

    #[test]
    fn warp_agrees_on_hand_cases() {
        assert!(agree(CNF, "5 2 0 1 2 0\n6 0 5 3 4 0\n").is_verified());
        assert!(agree(CNF, "5 2 0 2 1 0\n6 0 5 3 4 0\n").is_verified());
        assert!(agree(CNF, "5 2 0 1 2 0\n5 d 1 2 0\n6 0 5 3 4 0\n").is_verified());
        assert!(!agree(CNF, "5 2 0 1 2 0\n5 d 3 0\n6 0 5 3 4 0\n").is_verified());
        assert!(!agree(CNF, "5 0 3 4 0\n").is_verified());
        assert!(!agree(CNF, "5 2 0 1 0\n6 0 5 3 4 0\n").is_verified());
        assert!(!agree(CNF, "5 -2 0 1 2 0\n6 0 5 3 4 0\n").is_verified());
        assert!(!agree(CNF, "5 2 0 1 2 0\n").is_verified());
        assert!(!agree(CNF, "5 2 0 1 6 0\n6 0 5 3 4 0\n").is_verified());
        assert!(!agree(CNF, "4 2 0 1 2 0\n").is_verified());
        assert!(!agree(CNF, "5 2 0 -1 2 0\n6 0 5 3 4 0\n").is_verified());
    }

    #[test]
    fn wide_clause_strides_over_all_lanes() {
        // A clause of 70 literals (more than two per lane) that is unit on
        // its last literal: lanes 0..31 see literal indices k, k+32, k+64.
        let n = 70;
        let mut cnf = String::from("p cnf 70 71\n");
        // Clause 1: x1 ∨ … ∨ x70. Clauses 2..70: ¬x1 … ¬x69 (units).
        cnf.push_str(&(1..=n).map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
        cnf.push_str(" 0\n");
        for v in 1..n {
            cnf.push_str(&format!("-{v} 0\n"));
        }
        cnf.push_str("-70 0\n"); // clause 71: ¬x70 → UNSAT overall
                                 // Lemma 72 = () : hints 2..70 make x1..x69 false, clause 1 then unit
                                 // x70, clause 71 conflicts.
        let hints: Vec<String> = (2..=70).map(|h| h.to_string()).collect();
        let proof = format!("72 0 {} 1 71 0\n", hints.join(" "));
        assert!(agree(&cnf, &proof).is_verified());
    }

    #[test]
    fn faults_are_caught_on_the_hand_proof() {
        let c = parse_cnf_ints(CNF);
        // Valid except that lemma 6 uses clause 3 after it was deleted.
        let good = parse_text("5 2 0 1 2 0\n5 d 3 0\n6 0 5 3 4 0\n").unwrap();
        // Build a proof whose verdict each fault must change.
        let p = parse_text("5 2 0 1 2 0\n6 0 5 3 4 0\n").unwrap();
        assert!(check_warp(&c, &p).is_verified());
        {
            let _a = fault::arm(fault::Fault::CountOffByOne);
            // Lemma 5's first hint (clause 1 under ¬b) is unit on a; off by
            // one reads it as a conflict and accepts lemma 5 early — then
            // lemma 6 is reached with the same shortcut: still "verified".
            // So this fault is invisible on a proof that IS valid; it must
            // show on an invalid one: the non-unit hint case.
            let bad = parse_text("5 0 3 4 0\n").unwrap();
            let v = check_warp(&c, &bad);
            assert!(
                v.is_verified(),
                "fault should make the non-unit proof pass: {v:?}"
            );
        }
        {
            let _a = fault::arm(fault::Fault::SkipDeletionTest);
            let v = check_warp(&c, &good);
            assert!(
                v.is_verified(),
                "skipping the deletion test must accept the deleted hint: {v:?}"
            );
            drop(_a);
            assert!(!check_warp(&c, &good).is_verified());
        }
        {
            let _a = fault::arm(fault::Fault::WrongUnitLane);
            // Needs a hint whose unit sits on a lane other than 0: the wide
            // clause test's proof. Reuse it.
            let n = 70;
            let mut cnf = String::from("p cnf 70 71\n");
            cnf.push_str(&(1..=n).map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
            cnf.push_str(" 0\n");
            for v in 1..n {
                cnf.push_str(&format!("-{v} 0\n"));
            }
            cnf.push_str("-70 0\n");
            let hints: Vec<String> = (2..=70).map(|h| h.to_string()).collect();
            let proof = format!("72 0 {} 1 71 0\n", hints.join(" "));
            let c2 = parse_cnf_ints(&cnf);
            let p2 = parse_text(&proof).unwrap();
            let v = check_warp(&c2, &p2);
            assert!(
                !v.is_verified(),
                "wrong unit lane must break the wide-clause proof: {v:?}"
            );
        }
    }

    /// Archive agreement: reference vs warp on every proof under
    /// `LRAT_ARCHIVE_DIRS` (skips when unset).
    #[test]
    fn archive_agreement_reference_vs_warp() {
        let Ok(dirs) = std::env::var("LRAT_ARCHIVE_DIRS") else {
            eprintln!("archive_agreement_reference_vs_warp: SKIPPED (set LRAT_ARCHIVE_DIRS)");
            return;
        };
        let mut n = 0;
        let mut overflow = 0;
        for dir in dirs.split(':') {
            let dir = std::path::Path::new(dir);
            let Ok(rd) = std::fs::read_dir(dir.join("proof")) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("lrat") {
                    continue;
                }
                let stem = p.file_name().unwrap().to_str().unwrap().to_string();
                let base = stem.split('.').next().unwrap();
                let Ok(cnf_text) =
                    std::fs::read_to_string(dir.join("cnf").join(format!("{base}.cnf")))
                else {
                    continue;
                };
                let Ok(proof_text) = std::fs::read_to_string(&p) else {
                    continue;
                };
                let Ok(proof) = parse_text(&proof_text) else {
                    continue;
                };
                let cnf = parse_cnf_ints(&cnf_text);
                let r = check_sequential(&cnf, &proof);
                let w = check_warp(&cnf, &proof);
                if let Ok(fp) = flatten(&cnf, &proof) {
                    overflow += check_warp_codes(&fp)
                        .iter()
                        .filter(|&&c| c == LemmaCode::TableOverflow)
                        .count();
                }
                assert!(same(&r, &w), "{stem}: reference {r:?} vs warp {w:?}");
                n += 1;
            }
        }
        eprintln!(
            "archive_agreement_reference_vs_warp: {n} proofs agree; table overflows {overflow}"
        );
        assert!(n > 0);
        assert_eq!(overflow, 0, "the fixed-capacity table overflowed");
    }
}
