//! LRAT proof checking — the reference (sequential) checker and the
//! lemma-independent formulation the warp version is built on.
//!
//! An LRAT proof (Cruz-Filipe, Heule, Hunt, Kaufmann, Schneider-Kamp, CADE
//! 2017) is a list of steps over clause *ids*: original clauses are ids
//! `1..=n` in file order; each `a` step adds lemma `id` with literals and the
//! ids of the *hint* clauses that, taken in order under the assignment that
//! falsifies the lemma, are each unit or conflicting — reverse unit
//! propagation with the propagation order written down; each `d` step deletes
//! ids. A proof is verified when an empty lemma is checked successfully.
//!
//! Semantics follow `lrat-check` (drat-trim repository, commit 2e3b2dc0),
//! with one deliberate narrowing: **RAT lemmas (negative hints) are not
//! supported and come back as [`Verdict::Unsupported`]**, never as verified.
//! Every proof this stack produces is RUP-only (`drat-trim -L` reports 0 RAT
//! lemmas on all of them), and the warp kernel is RUP-only by design.
//!
//! # Why lemmas can be checked independently
//!
//! Checking lemma `i` reads the literals of its hint clauses and nothing
//! else; it does not depend on the *result* of checking any other lemma. The
//! sequential constraints are only (a) ids strictly increase, (b) a hint must
//! name a clause with a smaller id that has not been deleted before lemma
//! `i`. Both are decidable from a prepass that records, for every id, the
//! position of the step that deletes it. So: prepass, then every lemma in
//! parallel, then conjoin — [`check_independent`] is that formulation on the
//! CPU, and the SimWarp / GPU versions reorganise exactly it. The two
//! checkers must agree on every input; the tests hold them to that.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// One proof step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Add lemma `id` with `lits` (DIMACS integers) justified by `hints`
    /// (clause ids; negative = RAT pivot hint, unsupported here).
    Add {
        id: u64,
        lits: Vec<i32>,
        hints: Vec<i64>,
    },
    /// Delete clause ids (the step id on the `d` line is ignored, as in
    /// `lrat-check`).
    Delete { ids: Vec<u64> },
}

/// A parsed proof.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Proof {
    pub steps: Vec<Step>,
}

/// Largest variable index any parser accepts. The reference checkers size a
/// literal mask by the largest variable, so an unbounded literal (`i32::MIN`)
/// would be a 34 GB allocation on malformed input (cross-vendor panel,
/// 2026-11-02). 2^26 variables is a 1 GB mask at worst, proportional to a
/// problem that size.
pub const MAX_VAR: u32 = 1 << 26;
/// Largest gap between consecutive lemma ids any parser accepts; the flat
/// decoders index ids densely, so one huge id would be one huge allocation.
pub const MAX_ID_GAP: u64 = 1 << 26;

/// Parse error (format problems; not a proof failure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRAT parse error at line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for ParseError {}

/// Parse text LRAT: `id lits 0 hints 0` and `id d ids 0`; comment lines
/// starting with `c` are skipped.
pub fn parse_text(text: &str) -> Result<Proof, ParseError> {
    let mut steps = Vec::new();
    let mut last_add: Option<u64> = None;
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        let mut toks = line.split_whitespace();
        let id: u64 = toks
            .next()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| ParseError {
                line: n + 1,
                msg: "missing or bad step id".into(),
            })?;
        let rest: Vec<&str> = toks.collect();
        if rest.first() == Some(&"d") {
            let mut ids = Vec::new();
            for t in &rest[1..] {
                let v: i64 = t.parse().map_err(|_| ParseError {
                    line: n + 1,
                    msg: format!("bad id {t:?}"),
                })?;
                if v == 0 {
                    break;
                }
                if v < 0 {
                    return Err(ParseError {
                        line: n + 1,
                        msg: "negative id in deletion".into(),
                    });
                }
                ids.push(v as u64);
            }
            steps.push(Step::Delete { ids });
            continue;
        }
        let mut lits = Vec::new();
        let mut hints = Vec::new();
        let mut in_hints = false;
        let mut terminated = false;
        for t in &rest {
            let v: i64 = t.parse().map_err(|_| ParseError {
                line: n + 1,
                msg: format!("bad number {t:?}"),
            })?;
            if v == 0 {
                if in_hints {
                    terminated = true;
                    break;
                }
                in_hints = true;
                continue;
            }
            if in_hints {
                hints.push(v);
            } else {
                let l = i32::try_from(v).map_err(|_| ParseError {
                    line: n + 1,
                    msg: format!("literal {v} out of range"),
                })?;
                if l.unsigned_abs() > MAX_VAR {
                    return Err(ParseError {
                        line: n + 1,
                        msg: format!(
                            "variable {} beyond the checker bound {MAX_VAR}",
                            l.unsigned_abs()
                        ),
                    });
                }
                lits.push(l);
            }
        }
        if !in_hints || !terminated {
            return Err(ParseError {
                line: n + 1,
                msg: "add step must be `id lits 0 hints 0`".into(),
            });
        }
        if let Some(prev) = last_add {
            if id > prev && id - prev > MAX_ID_GAP {
                return Err(ParseError {
                    line: n + 1,
                    msg: format!("id gap {} beyond the checker bound {MAX_ID_GAP}", id - prev),
                });
            }
        }
        last_add = Some(id);
        steps.push(Step::Add { id, lits, hints });
    }
    Ok(Proof { steps })
}

/// Binary LRAT, as written by CaDiCaL/lrat-trim and read by cake_lpr and
/// Lean's `bv_decide` (whose parser was used as the reference for the byte
/// layout): a step is the tag byte `a` or `d`; numbers are base-128 varints
/// (7 bits per byte, low group first, high bit = continue) of `2·|x| + (x<0)`,
/// so ids are encoded like positive literals; an `a` step is
/// `id lits… 0 hints… 0`, a `d` step is `ids… 0`. A RAT hint block (negative
/// pivot then ids) is parsed and reported as a negative hint.
pub fn parse_binary(bytes: &[u8]) -> Result<Proof, ParseError> {
    let mut pos = 0usize;
    let mut steps = Vec::new();
    fn varint(bytes: &[u8], pos: &mut usize) -> Result<u64, ParseError> {
        let mut x: u64 = 0;
        let mut shift = 0u32;
        loop {
            let b = *bytes.get(*pos).ok_or(ParseError {
                line: *pos,
                msg: "truncated varint".into(),
            })?;
            *pos += 1;
            if shift > 63 {
                return Err(ParseError {
                    line: *pos,
                    msg: "varint too long".into(),
                });
            }
            x |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(x);
            }
            shift += 7;
        }
    }
    let signed = |m: u64| -> i64 {
        let mag = (m >> 1) as i64;
        if m & 1 == 1 {
            -mag
        } else {
            mag
        }
    };
    let mut last_add: Option<u64> = None;
    while pos < bytes.len() {
        let tag = bytes[pos];
        pos += 1;
        match tag {
            b'a' => {
                let id = varint(bytes, &mut pos)? >> 1;
                let mut lits = Vec::new();
                loop {
                    let m = varint(bytes, &mut pos)?;
                    if m == 0 {
                        break;
                    }
                    let v = signed(m);
                    let l = i32::try_from(v).map_err(|_| ParseError {
                        line: pos,
                        msg: format!("literal {v} out of range"),
                    })?;
                    if l.unsigned_abs() > MAX_VAR {
                        return Err(ParseError {
                            line: pos,
                            msg: format!(
                                "variable {} beyond the checker bound {MAX_VAR}",
                                l.unsigned_abs()
                            ),
                        });
                    }
                    lits.push(l);
                }
                let mut hints = Vec::new();
                loop {
                    let m = varint(bytes, &mut pos)?;
                    if m == 0 {
                        break;
                    }
                    hints.push(signed(m));
                }
                if let Some(prev) = last_add {
                    if id > prev && id - prev > MAX_ID_GAP {
                        return Err(ParseError {
                            line: pos,
                            msg: format!(
                                "id gap {} beyond the checker bound {MAX_ID_GAP}",
                                id - prev
                            ),
                        });
                    }
                }
                last_add = Some(id);
                steps.push(Step::Add { id, lits, hints });
            }
            b'd' => {
                let mut ids = Vec::new();
                loop {
                    let m = varint(bytes, &mut pos)?;
                    if m == 0 {
                        break;
                    }
                    ids.push(m >> 1);
                }
                steps.push(Step::Delete { ids });
            }
            other => {
                return Err(ParseError {
                    line: pos,
                    msg: format!("expected 'a' or 'd', found byte {other}"),
                })
            }
        }
    }
    Ok(Proof { steps })
}

/// Write a proof in binary LRAT (see [`parse_binary`]).
pub fn write_binary(proof: &Proof) -> Vec<u8> {
    fn put(out: &mut Vec<u8>, mut m: u64) {
        loop {
            let b = (m & 0x7f) as u8;
            m >>= 7;
            if m == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }
    let enc = |x: i64| -> u64 { ((x.unsigned_abs()) << 1) | u64::from(x < 0) };
    let mut out = Vec::new();
    for s in &proof.steps {
        match s {
            Step::Add { id, lits, hints } => {
                out.push(b'a');
                put(&mut out, id << 1);
                for &l in lits {
                    put(&mut out, enc(l as i64));
                }
                put(&mut out, 0);
                for &h in hints {
                    put(&mut out, enc(h));
                }
                put(&mut out, 0);
            }
            Step::Delete { ids } => {
                out.push(b'd');
                for &d in ids {
                    put(&mut out, d << 1);
                }
                put(&mut out, 0);
            }
        }
    }
    out
}

/// Parse either form: binary if the first byte is `a` or `d` and the input
/// is not valid text LRAT (text LRAT starts with a digit or `c`).
pub fn parse_any(bytes: &[u8]) -> Result<Proof, ParseError> {
    match bytes.first() {
        Some(b'a') | Some(b'd') => parse_binary(bytes),
        _ => parse_text(std::str::from_utf8(bytes).map_err(|_| ParseError {
            line: 0,
            msg: "not UTF-8 and not binary LRAT".into(),
        })?),
    }
}

/// Parse a DIMACS CNF into clauses of integers (own reader: `c` lines
/// skipped, `%` ends the input, header ignored beyond presence).
pub fn parse_cnf_ints(text: &str) -> Vec<Vec<i32>> {
    let mut clauses = Vec::new();
    let mut cur = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('c') || t.starts_with('p') {
            continue;
        }
        if t.starts_with('%') {
            break;
        }
        for tok in t.split_whitespace() {
            let v: i32 = match tok.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v == 0 {
                clauses.push(std::mem::take(&mut cur));
            } else {
                cur.push(v);
            }
        }
    }
    if !cur.is_empty() {
        clauses.push(cur);
    }
    clauses
}

/// Why a lemma failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reason {
    /// Lemma id not greater than every earlier id.
    IdNotIncreasing,
    /// A hint names an id that does not exist (yet).
    UnknownHint(u64),
    /// A hint names a clause deleted before this lemma.
    DeletedHint(u64),
    /// A hint names an id not smaller than the lemma's own id.
    ForwardHint(u64),
    /// A hint clause had two or more non-falsified literals: not unit.
    NotUnit(u64),
    /// The hints ran out without a conflict.
    NoConflict,
    /// A deletion named an id that does not exist or is already deleted.
    BadDeletion(u64),
    /// The warp paths' per-lemma assignment table (4,095 literals) overflowed:
    /// that path gives no verdict on this lemma.
    TableOverflow,
    /// No code was written for this lemma (device did not run, or the code
    /// vector is the wrong length): fail closed.
    Unchecked,
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reason::IdNotIncreasing => write!(f, "id not increasing"),
            Reason::UnknownHint(h) => write!(f, "hint {h} does not exist"),
            Reason::DeletedHint(h) => write!(f, "hint {h} was deleted"),
            Reason::ForwardHint(h) => write!(f, "hint {h} is not earlier than the lemma"),
            Reason::TableOverflow => write!(
                f,
                "assignment table (4,095 literals) overflowed on this path"
            ),
            Reason::Unchecked => write!(f, "lemma was not checked (no code written)"),
            Reason::NotUnit(h) => write!(f, "hint {h} is neither unit nor conflicting"),
            Reason::NoConflict => write!(f, "hints exhausted without a conflict"),
            Reason::BadDeletion(h) => write!(f, "deletion of unknown or already-deleted id {h}"),
        }
    }
}

/// The checker's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// An empty lemma was derived and every lemma up to it checked.
    Verified { lemmas: usize, deletions: usize },
    /// A step failed. `step` is the 0-based index in `Proof::steps`.
    Rejected {
        step: usize,
        id: u64,
        reason: Reason,
    },
    /// A RAT lemma (negative hint) was met; this checker does not do RAT.
    Unsupported { step: usize, id: u64 },
    /// Every step checked but no empty lemma was derived.
    NoEmptyClause { lemmas: usize },
}

impl Verdict {
    pub fn is_verified(&self) -> bool {
        matches!(self, Verdict::Verified { .. })
    }
}

/// Literal → index into a mask array: `2*|l| + (l < 0)`.
#[inline]
fn code(l: i32) -> usize {
    ((l.unsigned_abs() as usize) << 1) | usize::from(l < 0)
}

/// Clause arena keyed by id, plus deletion positions.
struct Arena {
    /// id → literals; original clauses are ids `1..=n`.
    clauses: HashMap<u64, Vec<i32>>,
    max_var: usize,
}

/// Check one lemma against the arena. `mask`/`stamp` are the caller's
/// scratch (reused across lemmas; `stamp` must be fresh per call).
/// `admissible(hint_id)` decides existence/ordering/deletion.
fn check_lemma(
    arena: &Arena,
    lits: &[i32],
    hints: &[i64],
    mask: &mut [u64],
    stamp: u64,
    mut admissible: impl FnMut(u64) -> Result<(), Reason>,
) -> Result<(), Reason> {
    for &l in lits {
        mask[code(l)] = stamp; // the lemma's literals are false
    }
    for &h in hints {
        // Negative hints are RAT pivots: refused upstream.
        let h = h as u64;
        admissible(h)?;
        let clause = arena.clauses.get(&h).ok_or(Reason::UnknownHint(h))?;
        let mut unit: Option<i32> = None;
        let mut free = 0u32;
        for &l in clause {
            if mask[code(l)] == stamp {
                continue; // falsified
            }
            free += 1;
            if free > 1 {
                return Err(Reason::NotUnit(h));
            }
            unit = Some(l);
        }
        match unit {
            None => return Ok(()),             // conflict: the lemma is RUP
            Some(u) => mask[code(-u)] = stamp, // u must be true: ¬u is false
        }
    }
    Err(Reason::NoConflict)
}

fn build_arena(cnf: &[Vec<i32>]) -> Arena {
    let mut clauses = HashMap::with_capacity(cnf.len() * 2);
    let mut max_var = 0usize;
    for (i, c) in cnf.iter().enumerate() {
        for &l in c {
            max_var = max_var.max(l.unsigned_abs() as usize);
        }
        clauses.insert(i as u64 + 1, c.clone());
    }
    Arena { clauses, max_var }
}

fn max_var_of_proof(proof: &Proof, base: usize) -> usize {
    let mut m = base;
    for s in &proof.steps {
        if let Step::Add { lits, .. } = s {
            for &l in lits {
                m = m.max(l.unsigned_abs() as usize);
            }
        }
    }
    m
}

/// The sequential reference checker: steps in order, deletions applied as
/// they come, `lrat-check`'s semantics (strict on deletions of unknown ids
/// and on lemmas whose hints end without a conflict).
pub fn check_sequential(cnf: &[Vec<i32>], proof: &Proof) -> Verdict {
    let mut arena = build_arena(cnf);
    let mv = max_var_of_proof(proof, arena.max_var);
    let mut mask = vec![0u64; 2 * (mv + 1) + 2];
    let mut stamp = 0u64;
    let mut last_id = cnf.len() as u64;
    let mut lemmas = 0usize;
    let mut deletions = 0usize;
    let mut empty_seen = false;
    // Every id ever added: distinguishes a deleted hint from one never added.
    let mut added: HashSet<u64> = (1..=cnf.len() as u64).collect();
    for (step_ix, step) in proof.steps.iter().enumerate() {
        match step {
            Step::Delete { ids } => {
                for &d in ids {
                    if arena.clauses.remove(&d).is_none() {
                        return Verdict::Rejected {
                            step: step_ix,
                            id: d,
                            reason: Reason::BadDeletion(d),
                        };
                    }
                    deletions += 1;
                }
            }
            Step::Add { id, lits, hints } => {
                if *id <= last_id {
                    return Verdict::Rejected {
                        step: step_ix,
                        id: *id,
                        reason: Reason::IdNotIncreasing,
                    };
                }
                if hints.iter().any(|&h| h < 0) {
                    return Verdict::Unsupported {
                        step: step_ix,
                        id: *id,
                    };
                }
                stamp += 1;
                let lemma_id = *id;
                let r = check_lemma(&arena, lits, hints, &mut mask, stamp, |h| {
                    if h >= lemma_id {
                        Err(Reason::ForwardHint(h))
                    } else {
                        Ok(())
                    }
                });
                // An unknown hint under sequential checking is either never
                // added or already deleted; report it as deleted when it was
                // once a valid id, else unknown.
                if let Err(reason) = r {
                    let reason = match reason {
                        Reason::UnknownHint(h) if added.contains(&h) => Reason::DeletedHint(h),
                        other => other,
                    };
                    return Verdict::Rejected {
                        step: step_ix,
                        id: *id,
                        reason,
                    };
                }
                lemmas += 1;
                last_id = *id;
                added.insert(*id);
                // `lrat-check`'s rule: every step is checked, the empty clause
                // may come anywhere, and a bad step AFTER it still fails the
                // proof (its main loop breaks on error, not on the empty
                // clause; the verdict is "empty seen and no error").
                if lits.is_empty() {
                    empty_seen = true;
                }
                arena.clauses.insert(*id, lits.clone());
            }
        }
    }
    if empty_seen {
        Verdict::Verified { lemmas, deletions }
    } else {
        Verdict::NoEmptyClause { lemmas }
    }
}

/// Prepass product for the lemma-independent formulation.
struct Prepass {
    arena: Arena,
    /// id → index of the step that deletes it (`usize::MAX` if never).
    deleted_at: HashMap<u64, usize>,
    /// Index of the first empty lemma that must verify, if any.
    first_empty: Option<usize>,
    /// Steps (indices) that are lemmas, in order, up to and including the
    /// first empty one.
    lemma_steps: Vec<usize>,
    max_var: usize,
    /// Structural verdict at a step every kept lemma precedes (see `prepass`).
    early: Option<Verdict>,
}

/// The step sequence alone can decide a verdict (an id out of order, a bad
/// deletion, a RAT step) at some step E; lemmas before E must still be
/// checked, and the verdict is the EARLIEST failing step — the sequential
/// reference's and `lrat-check`'s answer. So the prepass records that
/// verdict instead of returning it, and keeps only the lemmas before E.
fn prepass(cnf: &[Vec<i32>], proof: &Proof) -> Prepass {
    let mut arena = build_arena(cnf);
    let mut deleted_at: HashMap<u64, usize> = HashMap::new();
    let mut last_id = cnf.len() as u64;
    let mut lemma_steps = Vec::new();
    let mut first_empty = None;
    let mut early = None;
    'steps: for (step_ix, step) in proof.steps.iter().enumerate() {
        match step {
            Step::Delete { ids } => {
                for &d in ids {
                    if !arena.clauses.contains_key(&d) || deleted_at.contains_key(&d) {
                        early = Some(Verdict::Rejected {
                            step: step_ix,
                            id: d,
                            reason: Reason::BadDeletion(d),
                        });
                        break 'steps;
                    }
                    deleted_at.insert(d, step_ix);
                }
            }
            Step::Add { id, lits, hints } => {
                if *id <= last_id {
                    early = Some(Verdict::Rejected {
                        step: step_ix,
                        id: *id,
                        reason: Reason::IdNotIncreasing,
                    });
                    break;
                }
                if hints.iter().any(|&h| h < 0) {
                    early = Some(Verdict::Unsupported {
                        step: step_ix,
                        id: *id,
                    });
                    break;
                }
                last_id = *id;
                lemma_steps.push(step_ix);
                arena.clauses.insert(*id, lits.clone());
                if lits.is_empty() && first_empty.is_none() {
                    first_empty = Some(step_ix);
                }
            }
        }
    }
    let max_var = max_var_of_proof(proof, arena.max_var);
    Prepass {
        arena,
        deleted_at,
        first_empty,
        lemma_steps,
        max_var,
        early,
    }
}

/// Check one lemma step of a prepassed proof. Independent of every other
/// lemma: reads only the arena, the deletion table and this step.
fn check_lemma_independent(
    pp: &Prepass,
    proof: &Proof,
    step_ix: usize,
    mask: &mut [u64],
    stamp: u64,
) -> Result<(), (u64, Reason)> {
    let Step::Add { id, lits, hints } = &proof.steps[step_ix] else {
        return Ok(());
    };
    let lemma_id = *id;
    check_lemma(&pp.arena, lits, hints, mask, stamp, |h| {
        if h >= lemma_id {
            return Err(Reason::ForwardHint(h));
        }
        if let Some(&d) = pp.deleted_at.get(&h) {
            if d < step_ix {
                return Err(Reason::DeletedHint(h));
            }
        }
        Ok(())
    })
    .map_err(|r| (lemma_id, r))
}

/// The lemma-independent checker: prepass, then every lemma on its own,
/// `threads` of them at a time (1 = sequential order, still independent).
/// Must agree with [`check_sequential`] on every input.
pub fn check_independent(cnf: &[Vec<i32>], proof: &Proof, threads: usize) -> Verdict {
    let pp = prepass(cnf, proof);
    let n = pp.lemma_steps.len();
    let threads = threads.max(1).min(n.max(1));
    let chunk = n.div_ceil(threads);
    let mut firsts: Vec<Option<(usize, u64, Reason)>> = vec![None; threads];
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (t, slot) in firsts.iter_mut().enumerate() {
            let pp = &pp;
            let lo = (t * chunk).min(n);
            let hi = ((t + 1) * chunk).min(n);
            handles.push(s.spawn(move || {
                let mut mask = vec![0u64; 2 * (pp.max_var + 1) + 2];
                let mut stamp = 0u64;
                for &step_ix in &pp.lemma_steps[lo..hi] {
                    stamp += 1;
                    if let Err((id, r)) =
                        check_lemma_independent(pp, proof, step_ix, &mut mask, stamp)
                    {
                        *slot = Some((step_ix, id, r));
                        return;
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("checker thread panicked");
        }
    });
    // Earliest failing step wins (so the verdict is deterministic) — among
    // the lemma failures and the prepass's structural verdict alike.
    if let Some((step, id, reason)) = firsts.into_iter().flatten().min_by_key(|x| x.0) {
        return Verdict::Rejected { step, id, reason };
    }
    if let Some(e) = pp.early {
        return e;
    }
    let deletions = pp.deleted_at.len();
    match pp.first_empty {
        Some(_) => Verdict::Verified {
            lemmas: n,
            deletions,
        },
        None => Verdict::NoEmptyClause { lemmas: n },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (a ∨ b) (¬a ∨ b) (a ∨ ¬b) (¬a ∨ ¬b): ids 1..4. Lemma 5 = (b) by hints
    // 1,2 (assume ¬b: clause 1 → a unit; clause 2 → conflict). Lemma 6 = ()
    // by hints 5,3,4 (b unit; 3 → a; 4 → conflict).
    const CNF: &str = "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n";
    const PROOF: &str = "5 2 0 1 2 0\n6 0 5 3 4 0\n";

    fn both(cnf: &str, proof: &str) -> (Verdict, Verdict) {
        let c = parse_cnf_ints(cnf);
        let p = parse_text(proof).unwrap();
        (check_sequential(&c, &p), check_independent(&c, &p, 3))
    }

    #[test]
    fn binary_roundtrip_and_check() {
        let p = parse_text("5 2 0 1 2 0\n5 d 1 2 0\n6 0 5 3 4 0\n").unwrap();
        let bin = write_binary(&p);
        assert_eq!(bin[0], b'a');
        let back = parse_binary(&bin).unwrap();
        assert_eq!(back, p);
        let via_any = parse_any(&bin).unwrap();
        assert_eq!(via_any, p);
        let cnf = parse_cnf_ints(CNF);
        assert_eq!(
            check_sequential(&cnf, &back),
            Verdict::Verified {
                lemmas: 2,
                deletions: 2
            }
        );
        // Varint boundary: ids and literals above 127 need two bytes.
        let big = Proof {
            steps: vec![Step::Add {
                id: 70000,
                lits: vec![-300, 128],
                hints: vec![69999],
            }],
        };
        assert_eq!(parse_binary(&write_binary(&big)).unwrap(), big);
    }

    #[test]
    fn parses_add_and_delete() {
        let p = parse_text("5 2 0 1 2 0\n5 d 1 2 0\n6 0 5 3 4 0\n").unwrap();
        assert_eq!(p.steps.len(), 3);
        assert!(matches!(&p.steps[1], Step::Delete { ids } if ids == &vec![1, 2]));
    }

    #[test]
    fn hand_proof_verifies_both_ways() {
        let (s, i) = both(CNF, PROOF);
        assert_eq!(
            s,
            Verdict::Verified {
                lemmas: 2,
                deletions: 0
            }
        );
        assert_eq!(i, s);
    }

    #[test]
    fn deletion_before_use_is_rejected_both_ways() {
        // Delete clause 3 before lemma 6 needs it.
        let (s, i) = both(CNF, "5 2 0 1 2 0\n5 d 3 0\n6 0 5 3 4 0\n");
        assert!(
            matches!(
                s,
                Verdict::Rejected {
                    id: 6,
                    reason: Reason::DeletedHint(3),
                    ..
                }
            ),
            "{s:?}"
        );
        assert_eq!(i, s);
    }

    #[test]
    fn deletion_after_use_is_fine() {
        let (s, i) = both(CNF, "5 2 0 1 2 0\n5 d 1 2 0\n6 0 5 3 4 0\n");
        assert_eq!(
            s,
            Verdict::Verified {
                lemmas: 2,
                deletions: 2
            }
        );
        assert_eq!(i, s);
    }

    #[test]
    fn non_unit_hint_is_rejected() {
        // Under ¬b both hint orders for lemma 5 are fine (clause 2 is unit
        // on a once b is false). A genuinely non-unit hint: the empty lemma
        // straight from clauses 3 and 4 — clause 3 (a ∨ ¬b) has two free
        // literals under the empty assignment.
        let (s, i) = both(CNF, "5 2 0 2 1 0\n6 0 5 3 4 0\n");
        assert_eq!(
            s,
            Verdict::Verified {
                lemmas: 2,
                deletions: 0
            }
        );
        assert_eq!(i, s);
        let (s, i) = both(CNF, "5 0 3 4 0\n");
        assert!(
            matches!(
                s,
                Verdict::Rejected {
                    id: 5,
                    reason: Reason::NotUnit(3),
                    ..
                }
            ),
            "{s:?}"
        );
        assert_eq!(i, s);
    }

    #[test]
    fn missing_conflict_is_rejected() {
        let (s, i) = both(CNF, "5 2 0 1 0\n6 0 5 3 4 0\n");
        assert!(matches!(
            s,
            Verdict::Rejected {
                id: 5,
                reason: Reason::NoConflict,
                ..
            }
        ));
        assert_eq!(i, s);
    }

    #[test]
    fn flipped_literal_is_rejected() {
        let (s, i) = both(CNF, "5 -2 0 1 2 0\n6 0 5 3 4 0\n");
        assert!(matches!(s, Verdict::Rejected { .. }));
        assert_eq!(i, s);
    }

    #[test]
    fn no_empty_clause_is_not_verified() {
        let (s, i) = both(CNF, "5 2 0 1 2 0\n");
        assert_eq!(s, Verdict::NoEmptyClause { lemmas: 1 });
        assert_eq!(i, s);
    }

    #[test]
    fn forward_and_non_increasing_ids_are_rejected() {
        let (s, _) = both(CNF, "5 2 0 1 6 0\n6 0 5 3 4 0\n");
        assert!(
            matches!(
                s,
                Verdict::Rejected {
                    reason: Reason::ForwardHint(6),
                    ..
                }
            ),
            "{s:?}"
        );
        let (s, i) = both(CNF, "4 2 0 1 2 0\n");
        assert!(matches!(
            s,
            Verdict::Rejected {
                reason: Reason::IdNotIncreasing,
                ..
            }
        ));
        assert_eq!(i, s);
    }

    #[test]
    fn rat_hint_is_unsupported_not_verified() {
        let (s, i) = both(CNF, "5 2 0 -1 2 0\n6 0 5 3 4 0\n");
        assert!(matches!(s, Verdict::Unsupported { id: 5, .. }));
        assert_eq!(i, s);
    }

    /// Agreement with lrat-check and cake_lpr over an archive of proofs:
    /// `LRAT_ARCHIVE_DIRS` = colon-separated run directories each holding
    /// `cnf/<name>.cnf` and `proof/<name>[.tag].lrat`; `LRAT_CHECK` and
    /// optionally `CAKE_LPR` name the external checkers. Skips when unset.
    #[test]
    fn archive_agreement_with_external_checkers() {
        let Ok(dirs) = std::env::var("LRAT_ARCHIVE_DIRS") else {
            eprintln!("archive_agreement: SKIPPED (set LRAT_ARCHIVE_DIRS, LRAT_CHECK[, CAKE_LPR])");
            return;
        };
        let lrat_check =
            std::env::var("LRAT_CHECK").expect("LRAT_CHECK must be set with LRAT_ARCHIVE_DIRS");
        let cake = std::env::var("CAKE_LPR").ok();
        let mut n = 0;
        let mut agreed = 0;
        let mut verified = 0;
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
                let stem = p.file_name().unwrap().to_str().unwrap();
                let base = stem.split('.').next().unwrap();
                let cnf_path = dir.join("cnf").join(format!("{base}.cnf"));
                let Ok(cnf_text) = std::fs::read_to_string(&cnf_path) else {
                    continue;
                };
                let Ok(proof_text) = std::fs::read_to_string(&p) else {
                    continue;
                };
                let cnf = parse_cnf_ints(&cnf_text);
                let ours = match parse_text(&proof_text) {
                    Ok(pr) => {
                        let s = check_sequential(&cnf, &pr);
                        let i = check_independent(&cnf, &pr, 4);
                        assert_eq!(s, i, "{stem}: sequential vs independent disagree");
                        s.is_verified()
                    }
                    Err(_) => false,
                };
                let ext = std::process::Command::new(&lrat_check)
                    .arg(&cnf_path)
                    .arg(&p)
                    .output()
                    .expect("run lrat-check");
                let theirs = String::from_utf8_lossy(&ext.stdout).contains("c VERIFIED");
                n += 1;
                if ours == theirs {
                    agreed += 1;
                } else {
                    eprintln!("DISAGREE {stem}: ours={ours} lrat-check={theirs}");
                }
                if let Some(c) = &cake {
                    let o = std::process::Command::new(c)
                        .arg(&cnf_path)
                        .arg(&p)
                        .output()
                        .expect("run cake_lpr");
                    let cv = String::from_utf8_lossy(&o.stdout).contains("s VERIFIED");
                    assert_eq!(ours, cv, "{stem}: ours vs cake_lpr");
                }
                verified += usize::from(ours);
            }
        }
        eprintln!("archive_agreement: {agreed}/{n} agree with lrat-check; {verified} verified");
        assert!(n > 0, "no proofs found under LRAT_ARCHIVE_DIRS");
        assert_eq!(agreed, n);
    }
}
