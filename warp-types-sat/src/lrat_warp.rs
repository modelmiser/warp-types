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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// A verdict the host reached from the step SEQUENCE alone (an id out of
    /// order, a bad deletion, a hint naming an id never added, a RAT step),
    /// at some step E. Only lemmas with step < E are in the arrays; the
    /// per-lemma check still runs on them, and [`combine`] reports whichever
    /// failure comes first — the earliest failing step, which is what
    /// `lrat-check` reports when it stops. (Found by the hint-level
    /// negatives, 2026-11-02: on a use-after-delete the sequential reference
    /// names the lemma, a prepass-first path named the later double
    /// deletion.)
    pub early: Option<Verdict>,
}

/// Step index of a Rejected / Unsupported verdict.
fn verdict_step(v: &Verdict) -> Option<usize> {
    match v {
        Verdict::Rejected { step, .. } | Verdict::Unsupported { step, .. } => Some(*step),
        _ => None,
    }
}

/// Drop lemmas from index `keep` on (after an early verdict at a step the
/// `keep`-th lemma does not precede), and their clauses: the layout after an
/// early verdict is exactly "the originals and the kept lemmas", whichever
/// route built it (flatten adds every lemma's clause before it knows where
/// it stops; the decoders stop as they go — this makes the two equal).
fn truncate_lemmas(fp: &mut FlatProof, keep: usize) {
    fp.lemma_clause.truncate(keep);
    fp.lemma_step.truncate(keep);
    fp.lemma_id.truncate(keep);
    if fp.hint_start.len() > keep + 1 {
        let h = fp.hint_start[keep] as usize;
        fp.hint_start.truncate(keep + 1);
        fp.hints.truncate(h);
    }
    let n_clauses = fp.n_original as usize + keep;
    if fp.clause_start.len() > n_clauses + 1 {
        fp.clause_start.truncate(n_clauses + 1);
        fp.lits.truncate(fp.clause_start[n_clauses] as usize);
        fp.deleted_at.truncate(n_clauses);
    }
    if let Some(fe) = fp.first_empty {
        if fe as usize >= keep {
            fp.first_empty = None;
        }
    }
}

/// A hint of lemma `l` (external id `lemma_id`) that names no kept clause:
/// a forward hint when its id is not below the lemma's (the admissibility
/// test the reference applies first), unknown otherwise.
fn missing_hint_reason(hint: u64, lemma_id: u64) -> Reason {
    if hint >= lemma_id {
        Reason::ForwardHint(hint)
    } else {
        Reason::UnknownHint(hint)
    }
}

/// Build the flat layout. Hints that name an id never added at all are
/// decided here (the warp code has no id table) and become `early`; hints
/// that name a *later* id or a deleted clause are left to the per-lemma
/// check, which has the dense index order and the deletion steps.
pub fn flatten(cnf: &[Vec<i32>], proof: &Proof) -> FlatProof {
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
        early: None,
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
    let mut stop = proof.steps.len(); // steps >= stop are after the early verdict
    for (step_ix, step) in proof.steps.iter().enumerate() {
        match step {
            Step::Delete { .. } => {}
            Step::Add { id, lits, hints } => {
                if *id <= last_id {
                    fp.early = Some(Verdict::Rejected {
                        step: step_ix,
                        id: *id,
                        reason: Reason::IdNotIncreasing,
                    });
                    stop = step_ix;
                    break;
                }
                if hints.iter().any(|&h| h < 0) {
                    fp.early = Some(Verdict::Unsupported {
                        step: step_ix,
                        id: *id,
                    });
                    stop = step_ix;
                    break;
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
    'steps: for (step_ix, step) in proof.steps.iter().enumerate().take(stop) {
        match step {
            Step::Delete { ids } => {
                for &d in ids {
                    let bad = |fp: &mut FlatProof| {
                        fp.early = Some(Verdict::Rejected {
                            step: step_ix,
                            id: d,
                            reason: Reason::BadDeletion(d),
                        });
                    };
                    let Some(&dense) = dense_of.get(&d) else {
                        bad(&mut fp);
                        break 'steps;
                    };
                    // A deletion of something added later in the file, or
                    // deleted twice, is a bad deletion in the reference too.
                    let added_at = if dense < fp.n_original {
                        0
                    } else {
                        fp.lemma_step[(dense - fp.n_original) as usize] as usize
                    };
                    if added_at > step_ix || fp.deleted_at[dense as usize] != u32::MAX {
                        bad(&mut fp);
                        break 'steps;
                    }
                    fp.deleted_at[dense as usize] = step_ix as u32;
                }
            }
            Step::Add { id, hints, .. } => {
                for &h in hints {
                    let Some(&dense) = dense_of.get(&(h as u64)) else {
                        fp.early = Some(Verdict::Rejected {
                            step: step_ix,
                            id: *id,
                            reason: missing_hint_reason(h as u64, *id),
                        });
                        // This lemma is the failing one: keep only earlier ones.
                        fp.hints.truncate(fp.hint_start[lemma_ix] as usize);
                        break 'steps;
                    };
                    fp.hints.push(dense);
                }
                fp.hint_start.push(fp.hints.len() as u32);
                lemma_ix += 1;
            }
        }
    }
    // Third pass: a kept lemma whose hint names a clause that will be
    // dropped (added after the early verdict's step) fails here, the same
    // way the decoders — which never saw that id — report it.
    let limit = fp.n_original as usize + lemma_ix;
    let mut keep = lemma_ix;
    'lemmas: for l in 0..lemma_ix {
        let (hs, he) = (fp.hint_start[l] as usize, fp.hint_start[l + 1] as usize);
        for &d in &fp.hints[hs..he] {
            if d as usize >= limit {
                let hint = dense_id(&fp, d as usize);
                fp.early = Some(Verdict::Rejected {
                    step: fp.lemma_step[l] as usize,
                    id: fp.lemma_id[l],
                    reason: missing_hint_reason(hint, fp.lemma_id[l]),
                });
                keep = l;
                break 'lemmas;
            }
        }
    }
    // Lemmas the second pass did not reach (after the early verdict's step)
    // are dropped; the ones kept are all checked, including any after the
    // first empty clause (`lrat-check`'s rule; see `check_sequential`).
    truncate_lemmas(&mut fp, keep);
    fp
}

impl FlatProof {
    /// Decode binary LRAT straight into the flat layout — no per-step
    /// allocation, ids mapped through a dense vector. This is the host-side
    /// cost that the device path pays; the text/`Proof` route exists for the
    /// reference checker and the tests. Semantics identical to
    /// [`flatten`]`(cnf, &parse_binary(bytes))`.
    pub fn from_binary(
        cnf: &[Vec<i32>],
        bytes: &[u8],
    ) -> Result<FlatProof, crate::lrat::ParseError> {
        use crate::lrat::ParseError;
        let err = |pos: usize, msg: &str| ParseError {
            line: pos,
            msg: msg.to_string(),
        };
        let n_original = cnf.len() as u32;
        let mut fp = FlatProof {
            lits: Vec::with_capacity(bytes.len() / 2),
            clause_start: Vec::with_capacity(bytes.len() / 8),
            deleted_at: Vec::with_capacity(bytes.len() / 8),
            lemma_clause: Vec::new(),
            lemma_step: Vec::new(),
            lemma_id: Vec::new(),
            hint_start: vec![0],
            hints: Vec::with_capacity(bytes.len() / 4),
            first_empty: None,
            n_original,
            early: None,
        };
        fp.clause_start.push(0);
        for c in cnf {
            fp.lits.extend_from_slice(c);
            fp.clause_start.push(fp.lits.len() as u32);
            fp.deleted_at.push(u32::MAX);
        }
        // id -> dense index + 1 (0 = absent), grown as ids appear.
        let mut dense_of: Vec<u32> = vec![0; cnf.len() + 1];
        for i in 0..cnf.len() {
            dense_of[i + 1] = i as u32 + 1;
        }
        let mut last_id = cnf.len() as u64;
        let mut pos = 0usize;
        let mut step_ix = 0usize;
        #[inline]
        fn varint(bytes: &[u8], pos: &mut usize) -> Option<u64> {
            let mut x: u64 = 0;
            let mut shift = 0u32;
            loop {
                let b = *bytes.get(*pos)?;
                *pos += 1;
                x |= ((b & 0x7f) as u64) << shift;
                if b & 0x80 == 0 {
                    return Some(x);
                }
                shift += 7;
                if shift > 63 {
                    return None;
                }
            }
        }
        // Hints are recorded as raw ids and resolved after the whole file is
        // read: a hint may (illegally) name a later id, and the per-lemma
        // check must be the one to reject it, exactly as `flatten` does.
        let mut raw_hints: Vec<u64> = Vec::with_capacity(bytes.len() / 4);
        // Structural failures (`early`) stop the decode: nothing after the
        // failing step can change the verdict, which is the earliest failure.
        while pos < bytes.len() && fp.early.is_none() {
            let tag = bytes[pos];
            pos += 1;
            match tag {
                b'a' => {
                    let id = varint(bytes, &mut pos).ok_or_else(|| err(pos, "truncated id"))? >> 1;
                    if id <= last_id {
                        fp.early = Some(Verdict::Rejected {
                            step: step_ix,
                            id,
                            reason: Reason::IdNotIncreasing,
                        });
                        break;
                    }
                    if id - last_id > crate::lrat::MAX_ID_GAP {
                        return Err(err(pos, "id gap beyond the checker bound"));
                    }
                    last_id = id;
                    let dense = fp.clause_start.len() as u32 - 1;
                    if dense_of.len() <= id as usize {
                        dense_of.resize(id as usize + 1, 0);
                    }
                    dense_of[id as usize] = dense + 1;
                    let lit_start = fp.lits.len();
                    loop {
                        let m =
                            varint(bytes, &mut pos).ok_or_else(|| err(pos, "truncated literal"))?;
                        if m == 0 {
                            break;
                        }
                        let mag = (m >> 1) as i64;
                        let v = if m & 1 == 1 { -mag } else { mag };
                        let l = i32::try_from(v).map_err(|_| err(pos, "literal out of range"))?;
                        if l == 0 {
                            return Err(err(pos, "literal 0 (encoded 1) names no variable"));
                        }
                        if l.unsigned_abs() > crate::lrat::MAX_VAR {
                            return Err(err(pos, "variable beyond the checker bound"));
                        }
                        fp.lits.push(l);
                    }
                    fp.clause_start.push(fp.lits.len() as u32);
                    fp.deleted_at.push(u32::MAX);
                    let hint_lo = raw_hints.len();
                    let mut rat = false;
                    loop {
                        let m =
                            varint(bytes, &mut pos).ok_or_else(|| err(pos, "truncated hint"))?;
                        if m == 0 {
                            break;
                        }
                        rat |= m & 1 == 1;
                        raw_hints.push(m >> 1);
                    }
                    if rat {
                        raw_hints.truncate(hint_lo);
                        fp.early = Some(Verdict::Unsupported { step: step_ix, id });
                        break;
                    }
                    fp.hint_start.push(raw_hints.len() as u32);
                    fp.lemma_clause.push(dense);
                    fp.lemma_step.push(step_ix as u32);
                    fp.lemma_id.push(id);
                    if fp.lits.len() == lit_start && fp.first_empty.is_none() {
                        fp.first_empty = Some(fp.lemma_clause.len() as u32 - 1);
                    }
                }
                b'd' => loop {
                    let m =
                        varint(bytes, &mut pos).ok_or_else(|| err(pos, "truncated deletion"))?;
                    if m == 0 {
                        break;
                    }
                    let id = (m >> 1) as usize;
                    let dense = dense_of.get(id).copied().unwrap_or(0);
                    if dense == 0 || fp.deleted_at[(dense - 1) as usize] != u32::MAX {
                        fp.early = Some(Verdict::Rejected {
                            step: step_ix,
                            id: id as u64,
                            reason: Reason::BadDeletion(id as u64),
                        });
                        break;
                    }
                    fp.deleted_at[(dense - 1) as usize] = step_ix as u32;
                },
                _ => return Err(err(pos, "expected 'a' or 'd'")),
            }
            step_ix += 1;
        }
        fp.hints.reserve(raw_hints.len());
        for (k, &h) in raw_hints.iter().enumerate() {
            let dense = dense_of.get(h as usize).copied().unwrap_or(0);
            if dense == 0 {
                // The lemma holding this hint fails here; it precedes any
                // `early` from the step loop (those lemmas were never kept).
                let l = fp.hint_start.partition_point(|&s| s as usize <= k) - 1;
                fp.early = Some(Verdict::Rejected {
                    step: fp.lemma_step[l] as usize,
                    id: fp.lemma_id[l],
                    reason: missing_hint_reason(h, fp.lemma_id[l]),
                });
                truncate_lemmas(&mut fp, l);
                return Ok(fp);
            }
            fp.hints.push(dense - 1);
        }
        // Normalise: a lemma that ended in an early verdict may have pushed
        // its clause before the verdict was known.
        let keep = fp.lemma_clause.len();
        truncate_lemmas(&mut fp, keep);
        Ok(fp)
    }
}

// ============================================================================
// Parallel binary decode.
//
// The device check is ~0.3 s on a 941 MB proof; the one-thread decode above is
// 3.4–5.9 s, so the decode is the pipeline (research/gpu-lrat-checker.md,
// Stage 4). Binary LRAT has no index, but it is self-synchronising: in a
// minimal varint encoding the byte 0x00 occurs ONLY as a terminator, so a step
// can start only at a byte `a`/`d` that follows a 0x00 (or at offset 0). Not
// every such byte starts a step — a hint id of 50 encodes as 0x64 = `d` — so a
// worker that starts at a wrong candidate parses garbage for a step or two and
// then lands on the true step boundaries. The design makes that harmless:
//
//   phase 1 (parallel)   each worker parses from the first candidate in its
//                        slice to the first step boundary at or past the slice
//                        end, recording every step's start offset;
//   phase 2 (sequential) walk the TRUE chain: start at offset 0 (worker 0's
//                        chain is true by construction); at each slice boundary
//                        continue from the true chain's current offset — if the
//                        next worker recorded a step at exactly that offset its
//                        chain is the true chain from there on (same parser,
//                        same bytes, same start: same result), otherwise parse
//                        steps one at a time until it is, or until that slice
//                        is exhausted;
//   phase 3 (parallel)   copy literals/hints into the flat arrays at prefix
//                        offsets; resolve hint ids through the dense map.
//
// Correctness therefore does not rest on the 0x00 property at all: the output
// is, for ANY byte string, what the one-thread decoder produces (the tests
// hold the two equal on valid proofs, on every truncation and on every
// single-bit flip of a small proof). The property only decides how much work
// phase 2 has to do. One step of phase 2 per boundary is the normal case.
// ============================================================================

/// One parsed step in a worker's local record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawStep {
    /// Byte offset of the tag.
    pub(crate) pos: u64,
    /// `a`: the id. `d`: unused.
    pub(crate) id: u64,
    /// `a`: offset just past the id varint (error position for a bad id).
    pub(crate) pos_after_id: u64,
    /// `a`: end (exclusive) of this step's literals in the worker's `lits`.
    pub(crate) lit_end: u32,
    /// `a`: end (exclusive) of this step's raw hints in the worker's `hints`.
    pub(crate) hint_end: u32,
    /// `d`: end (exclusive) of this step's ids in the worker's `dels`.
    pub(crate) del_end: u32,
    pub(crate) tag: u8,
    /// `a`: a negative (RAT) hint was present — an `Unsupported` verdict at
    /// this step, decided in phase 3a in step order.
    pub(crate) rat: bool,
}

/// A worker's output: steps plus the flat local payloads they index.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct RawChunk {
    pub(crate) steps: Vec<RawStep>,
    pub(crate) lits: Vec<i32>,
    pub(crate) hints: Vec<u64>,
    /// (id, offset just past its varint) per deletion.
    pub(crate) dels: Vec<(u64, u64)>,
    /// Where the chain stopped with a format error, if it did.
    pub(crate) error: Option<crate::lrat::ParseError>,
    /// Offset just past the last complete step (the chain's current offset).
    pub(crate) end: u64,
    /// If the chain ended with a format error inside an `a` step whose id had
    /// already been read: (id, offset after the id). The one-thread decoder
    /// checks the id BEFORE reading on, so a bad id outranks the later format
    /// error; phase 3a replays that order.
    pub(crate) partial_add: Option<(u64, u64)>,
    /// Start in `dels` of the ids read by the interrupted `d` step, for the
    /// same reason (each id is checked as it is read).
    pub(crate) partial_del_lo: usize,
}

#[inline]
fn varint_at(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let mut x: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = *bytes.get(*pos)?;
        *pos += 1;
        x |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some(x);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

/// Parse one step at `pos` into `out`. Returns the offset after it. Error
/// positions are exactly the one-thread decoder's (offset after the failing
/// read) so the two agree on malformed input too.
fn parse_step(
    bytes: &[u8],
    pos: usize,
    out: &mut RawChunk,
) -> Result<usize, crate::lrat::ParseError> {
    use crate::lrat::ParseError;
    let err = |pos: usize, msg: &str| ParseError {
        line: pos,
        msg: msg.to_string(),
    };
    let start = pos;
    let mut pos = pos;
    let tag = bytes[pos];
    pos += 1;
    out.partial_add = None;
    match tag {
        b'a' => {
            let id = varint_at(bytes, &mut pos).ok_or_else(|| err(pos, "truncated id"))? >> 1;
            let pos_after_id = pos;
            out.partial_add = Some((id, pos_after_id as u64));
            loop {
                let m = varint_at(bytes, &mut pos).ok_or_else(|| err(pos, "truncated literal"))?;
                if m == 0 {
                    break;
                }
                let mag = (m >> 1) as i64;
                let v = if m & 1 == 1 { -mag } else { mag };
                let l = i32::try_from(v).map_err(|_| err(pos, "literal out of range"))?;
                if l == 0 {
                    return Err(err(pos, "literal 0 (encoded 1) names no variable"));
                }
                if l.unsigned_abs() > crate::lrat::MAX_VAR {
                    return Err(err(pos, "variable beyond the checker bound"));
                }
                out.lits.push(l);
            }
            let mut rat = false;
            loop {
                let m = varint_at(bytes, &mut pos).ok_or_else(|| err(pos, "truncated hint"))?;
                if m == 0 {
                    break;
                }
                rat |= m & 1 == 1;
                out.hints.push(m >> 1);
            }
            out.steps.push(RawStep {
                pos: start as u64,
                id,
                pos_after_id: pos_after_id as u64,
                lit_end: out.lits.len() as u32,
                hint_end: out.hints.len() as u32,
                del_end: out.dels.len() as u32,
                tag,
                rat,
            });
        }
        b'd' => {
            loop {
                let m = varint_at(bytes, &mut pos).ok_or_else(|| err(pos, "truncated deletion"))?;
                if m == 0 {
                    break;
                }
                out.dels.push((m >> 1, pos as u64));
            }
            out.steps.push(RawStep {
                pos: start as u64,
                id: 0,
                pos_after_id: 0,
                lit_end: out.lits.len() as u32,
                hint_end: out.hints.len() as u32,
                del_end: out.dels.len() as u32,
                tag,
                rat: false,
            });
        }
        _ => return Err(err(pos, "expected 'a' or 'd'")),
    }
    out.partial_add = None;
    Ok(pos)
}

/// Parse steps from `start` until the chain reaches `stop` or beyond, or an
/// error, or the end of input.
fn parse_run(bytes: &[u8], start: usize, stop: usize) -> RawChunk {
    let mut out = RawChunk {
        end: start as u64,
        ..Default::default()
    };
    let mut pos = start;
    while pos < stop && pos < bytes.len() {
        // A step that fails mid-way leaves partial payload behind; roll it back
        // so the record stays consistent (only the steps vector is consumed).
        let (nl, nh, nd) = (out.lits.len(), out.hints.len(), out.dels.len());
        match parse_step(bytes, pos, &mut out) {
            Ok(next) => {
                pos = next;
                out.end = pos as u64;
            }
            Err(e) => {
                out.lits.truncate(nl);
                out.hints.truncate(nh);
                out.partial_del_lo = nd; // dels[nd..] belong to the interrupted step
                out.error = Some(e);
                break;
            }
        }
    }
    out
}

/// First plausible step start in `lo..hi`: offset 0, or an `a`/`d` byte that
/// follows a 0x00. `None` when the slice has no candidate (the previous
/// worker's chain then simply runs through it).
fn first_candidate(bytes: &[u8], lo: usize, hi: usize) -> Option<usize> {
    if lo == 0 {
        return Some(0);
    }
    (lo..hi.min(bytes.len())).find(|&p| bytes[p - 1] == 0 && (bytes[p] == b'a' || bytes[p] == b'd'))
}

/// A segment of the true chain: steps `step_lo..step_hi` of `chunks[chunk]`.
struct Segment {
    chunk: usize,
    step_lo: usize,
    step_hi: usize,
}

impl FlatProof {
    /// [`FlatProof::from_binary`] on `threads` threads. Identical output and
    /// identical errors for every input (see the module comment above);
    /// `threads <= 1` is the one-thread decoder. Peak memory is about twice the
    /// one-thread decoder's (payloads are decoded into per-worker buffers, then
    /// copied).
    pub fn from_binary_par(
        cnf: &[Vec<i32>],
        bytes: &[u8],
        threads: usize,
    ) -> Result<FlatProof, crate::lrat::ParseError> {
        if threads <= 1 || bytes.len() < 1 << 16 {
            return FlatProof::from_binary(cnf, bytes);
        }
        Self::from_binary_par_chunked(cnf, bytes, threads, bytes.len().div_ceil(threads))
    }

    /// The worker slice size is a parameter so the tests can force many
    /// boundaries onto a small proof.
    pub(crate) fn from_binary_par_chunked(
        cnf: &[Vec<i32>],
        bytes: &[u8],
        threads: usize,
        chunk: usize,
    ) -> Result<FlatProof, crate::lrat::ParseError> {
        if bytes.is_empty() {
            return FlatProof::from_binary(cnf, bytes);
        }
        let bounds = slice_bounds(bytes.len(), chunk);
        let chunks = parse_slices_host(bytes, &bounds, threads);
        stitch_chunks(cnf, bytes, &bounds, chunks, threads)
    }
}

/// Slice `len` bytes into `[lo, hi)` ranges of `chunk` bytes (the last one
/// shorter). Shared by the host workers and the device path so both stitch
/// over the same boundaries.
pub(crate) fn slice_bounds(len: usize, chunk: usize) -> Vec<(usize, usize)> {
    let chunk = chunk.max(1);
    let n_slices = len.div_ceil(chunk);
    (0..n_slices)
        .map(|i| (i * chunk, ((i + 1) * chunk).min(len)))
        .collect()
}

/// Phase 1 on host threads: the speculative parse of every slice.
pub(crate) fn parse_slices_host(
    bytes: &[u8],
    bounds: &[(usize, usize)],
    threads: usize,
) -> Vec<RawChunk> {
    let n_slices = bounds.len();
    let mut chunks: Vec<RawChunk> = (0..n_slices).map(|_| RawChunk::default()).collect();
    std::thread::scope(|sc| {
        let threads = threads.min(n_slices).max(1);
        let per = n_slices.div_ceil(threads);
        let mut rest: &mut [RawChunk] = &mut chunks;
        let mut first = 0usize;
        while !rest.is_empty() {
            let take = per.min(rest.len());
            let (mine, tail) = rest.split_at_mut(take);
            rest = tail;
            let lo = first;
            first += take;
            let bounds = &bounds;
            sc.spawn(move || {
                for (k, out) in mine.iter_mut().enumerate() {
                    let (slo, shi) = bounds[lo + k];
                    *out = match first_candidate(bytes, slo, shi) {
                        Some(start) => parse_run(bytes, start, shi),
                        None => RawChunk {
                            end: slo as u64,
                            ..Default::default()
                        },
                    };
                }
            });
        }
    });
    chunks
}

/// Phases 2–3 over phase-1 records, whoever produced them (host threads or the
/// device): follow the true chain across slice boundaries, do the sequential
/// bookkeeping, copy payloads and resolve hints. `chunks[i]` must be exactly
/// `parse_run(bytes, start_i, bounds[i].1)` for the slice's first candidate
/// (or empty with `end = bounds[i].0` when the slice has none) — that is the
/// whole contract a producer has to meet.
pub(crate) fn stitch_chunks(
    cnf: &[Vec<i32>],
    bytes: &[u8],
    bounds: &[(usize, usize)],
    chunks: Vec<RawChunk>,
    threads: usize,
) -> Result<FlatProof, crate::lrat::ParseError> {
    let plan = stitch_plan(cnf, bytes, bounds, chunks, threads)?;
    stitch_payloads_host(cnf, plan, threads)
}

/// A true-chain segment to copy in phase 3b: chunk-local literal and hint
/// ranges of `chunk` (an index into chunks ++ bridges).
pub(crate) struct Copy {
    pub(crate) chunk: usize,
    pub(crate) lit_lo: usize,
    pub(crate) lit_hi: usize,
    pub(crate) hint_lo: usize,
    pub(crate) hint_hi: usize,
}

/// Phases 2 and 3a done: everything about the proof except its literal and
/// hint payloads, plus what phase 3b needs to place them — on the host
/// ([`stitch_payloads_host`]) or on the device (`decode_device`).
pub(crate) struct StitchPlan {
    /// `lits` and `hints` empty; every other field final (before the
    /// bad-hint truncation phase 3b may still apply).
    pub(crate) fp: FlatProof,
    pub(crate) copies: Vec<Copy>,
    pub(crate) dense_of: Vec<u32>,
    pub(crate) lit_total: usize,
    pub(crate) hint_total: usize,
    pub(crate) orig_lits: usize,
    pub(crate) chunks: Vec<RawChunk>,
    pub(crate) bridges: Vec<RawChunk>,
    /// Phase 3a ran on the optimistic parallel walk ([`plan_3a_parallel`]);
    /// `false` = the sequential walk (one thread, or the proof showed a
    /// violation and the sequential walk decided it).
    // Read by the device route (feature `gpu`) and by the tests.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fast: bool,
}

/// What the optimistic walk produces: the plan without the chunks.
struct Fast3a {
    fp: FlatProof,
    copies: Vec<Copy>,
    dense_of: Vec<u32>,
    lit_total: usize,
    hint_total: usize,
}

/// Per true-chain segment, from the validating pass.
#[derive(Clone, Copy)]
struct SegSum {
    lit_lo: usize,
    lit_hi: usize,
    hint_lo: usize,
    hint_hi: usize,
    del_lo: usize,
}

/// Per group of segments (one thread's share), from the validating pass.
#[derive(Clone, Copy)]
struct GroupSum {
    adds: usize,
    first_id: u64,
    last_id: u64,
    /// Group-local index of the first lemma with no literals.
    empty_at: Option<usize>,
}

/// Split `s` into consecutive pieces ending at each of `cuts` (ascending,
/// absolute indices into `s`; the last piece runs to the end).
fn split_at_cuts<'a, T>(mut s: &'a mut [T], cuts: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut at = 0usize;
    for &c in cuts {
        let (a, b) = s.split_at_mut(c - at);
        out.push(a);
        s = b;
        at = c;
    }
    out.push(s);
    out
}

/// Phase 3a on threads, for proofs that turn out to have nothing to reject.
///
/// Three passes over the true chain, each split into contiguous groups of
/// segments: (A) validate — ids strictly increasing, the gap bound, no RAT
/// hint — and size every segment; (B) write the lemma arrays and `dense_of`
/// at prefix offsets (ids strictly increasing makes each group's `dense_of`
/// range disjoint, so the pieces are plain `split_at_mut` slices); (C) the
/// deletions, against atomic `deleted_at`: the id must name a clause added at
/// an EARLIER step, and a swap that finds a step already there is a second
/// deletion. Any of those, anywhere, returns `None` and the caller runs the
/// sequential walk, which alone decides early verdicts and format errors —
/// so on every proof the output is the sequential walk's (the tests hold the
/// two equal).
fn plan_3a_parallel(
    cnf: &[Vec<i32>],
    all: &[&RawChunk],
    segments: &[Segment],
    threads: usize,
) -> Option<Fast3a> {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    let segs: Vec<&Segment> = segments.iter().filter(|s| s.step_hi > s.step_lo).collect();
    let n_orig = cnf.len();
    let orig_lits: usize = cnf.iter().map(|c| c.len()).sum();
    // Groups of segments by step count.
    let total_steps: usize = segs.iter().map(|s| s.step_hi - s.step_lo).sum();
    let per = total_steps.div_ceil(threads.max(1)).max(1);
    let mut groups: Vec<(usize, usize)> = Vec::new();
    {
        let (mut lo, mut acc) = (0usize, 0usize);
        for (i, sg) in segs.iter().enumerate() {
            acc += sg.step_hi - sg.step_lo;
            if acc >= per {
                groups.push((lo, i + 1));
                lo = i + 1;
                acc = 0;
            }
        }
        if lo < segs.len() || groups.is_empty() {
            groups.push((lo, segs.len()));
        }
    }
    let steps_of = |sg: &Segment| &all[sg.chunk].steps[sg.step_lo..sg.step_hi];
    let prev_of = |sg: &Segment| {
        if sg.step_lo == 0 {
            None
        } else {
            Some(all[sg.chunk].steps[sg.step_lo - 1])
        }
    };
    let bad = AtomicBool::new(false);

    // ---- pass A: validate + size ----
    let pass_a: Vec<Option<(Vec<SegSum>, GroupSum)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = groups
            .iter()
            .map(|&(a, b)| {
                let segs = &segs[a..b];
                let bad = &bad;
                sc.spawn(move || {
                    let mut sums = Vec::with_capacity(segs.len());
                    let mut g = GroupSum {
                        adds: 0,
                        first_id: 0,
                        last_id: 0,
                        empty_at: None,
                    };
                    for sg in segs {
                        if bad.load(Ordering::Relaxed) {
                            return None;
                        }
                        let prev = prev_of(sg);
                        let lit_lo = prev.map_or(0, |p| p.lit_end as usize);
                        let hint_lo = prev.map_or(0, |p| p.hint_end as usize);
                        let del_lo = prev.map_or(0, |p| p.del_end as usize);
                        let (mut lit_hi, mut hint_hi, mut prev_lit_end) = (lit_lo, hint_lo, lit_lo);
                        for st in steps_of(sg) {
                            if st.tag != b'a' {
                                continue;
                            }
                            if st.rat {
                                bad.store(true, Ordering::Relaxed);
                                return None;
                            }
                            if g.adds > 0
                                && (st.id <= g.last_id
                                    || st.id - g.last_id > crate::lrat::MAX_ID_GAP)
                            {
                                bad.store(true, Ordering::Relaxed);
                                return None;
                            }
                            if g.adds == 0 {
                                g.first_id = st.id;
                            }
                            g.last_id = st.id;
                            if st.lit_end as usize == prev_lit_end && g.empty_at.is_none() {
                                g.empty_at = Some(g.adds);
                            }
                            prev_lit_end = st.lit_end as usize;
                            lit_hi = st.lit_end as usize;
                            hint_hi = st.hint_end as usize;
                            g.adds += 1;
                        }
                        sums.push(SegSum {
                            lit_lo,
                            lit_hi,
                            hint_lo,
                            hint_hi,
                            del_lo,
                        });
                    }
                    Some((sums, g))
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("plan pass A"))
            .collect()
    });
    let mut sums: Vec<SegSum> = Vec::with_capacity(segs.len());
    let mut gsums: Vec<GroupSum> = Vec::with_capacity(groups.len());
    for r in pass_a {
        let (s, g) = r?;
        sums.extend(s);
        gsums.push(g);
    }

    // ---- across groups: ids, prefixes, copies ----
    let mut last_id = n_orig as u64;
    let mut id_cut: Vec<usize> = Vec::with_capacity(groups.len());
    let mut lemma_base: Vec<usize> = Vec::with_capacity(groups.len() + 1);
    let mut step_base: Vec<u32> = Vec::with_capacity(groups.len());
    let mut acc_base: Vec<u32> = Vec::with_capacity(groups.len());
    let mut hint_base: Vec<usize> = Vec::with_capacity(groups.len());
    let mut first_empty: Option<u32> = None;
    let (mut lemmas, mut steps_acc, mut lit_acc, mut hint_acc) = (0usize, 0u32, orig_lits, 0usize);
    let mut copies: Vec<Copy> = Vec::with_capacity(segs.len());
    for (gi, &(a, b)) in groups.iter().enumerate() {
        let g = gsums[gi];
        id_cut.push(last_id as usize + 1);
        if g.adds > 0 {
            if g.first_id <= last_id || g.first_id - last_id > crate::lrat::MAX_ID_GAP {
                return None;
            }
            last_id = g.last_id;
        }
        if first_empty.is_none() {
            first_empty = g.empty_at.map(|e| (lemmas + e) as u32);
        }
        lemma_base.push(lemmas);
        step_base.push(steps_acc);
        acc_base.push(lit_acc as u32);
        hint_base.push(hint_acc);
        lemmas += g.adds;
        for (sg, sm) in segs[a..b].iter().zip(&sums[a..b]) {
            steps_acc += (sg.step_hi - sg.step_lo) as u32;
            lit_acc += sm.lit_hi - sm.lit_lo;
            hint_acc += sm.hint_hi - sm.hint_lo;
            copies.push(Copy {
                chunk: sg.chunk,
                lit_lo: sm.lit_lo,
                lit_hi: sm.lit_hi,
                hint_lo: sm.hint_lo,
                hint_hi: sm.hint_hi,
            });
        }
    }
    lemma_base.push(lemmas);

    // ---- allocate; the originals' part serially ----
    let mut clause_start = vec![0u32; n_orig + 1 + lemmas];
    let mut acc = 0u32;
    for (i, c) in cnf.iter().enumerate() {
        acc += c.len() as u32;
        clause_start[i + 1] = acc;
    }
    let mut dense_of = vec![0u32; (n_orig + 1).max(last_id as usize + 1)];
    for i in 0..n_orig {
        dense_of[i + 1] = i as u32 + 1;
    }
    let mut lemma_clause = vec![0u32; lemmas];
    let mut lemma_step = vec![0u32; lemmas];
    let mut lemma_id = vec![0u64; lemmas];
    let mut hint_start = vec![0u32; lemmas + 1];

    // ---- pass B: write ----
    {
        let lcuts = &lemma_base[1..groups.len()];
        let cs_cuts: Vec<usize> = lcuts.iter().map(|&l| n_orig + 1 + l).collect();
        let hs_cuts: Vec<usize> = lcuts.iter().map(|&l| 1 + l).collect();
        let cs = split_at_cuts(&mut clause_start, &cs_cuts);
        let lc = split_at_cuts(&mut lemma_clause, lcuts);
        let ls = split_at_cuts(&mut lemma_step, lcuts);
        let li = split_at_cuts(&mut lemma_id, lcuts);
        let hs = split_at_cuts(&mut hint_start, &hs_cuts);
        let dn = split_at_cuts(&mut dense_of, &id_cut[1..]);
        std::thread::scope(|sc| {
            let pieces = cs.into_iter().zip(lc).zip(ls).zip(li).zip(hs).zip(dn);
            for (gi, (((((cs, lc), ls), li), hs), dn)) in pieces.enumerate() {
                let (a, b) = groups[gi];
                let segs = &segs[a..b];
                let sums = &sums[a..b];
                let (cs_off, lc_off, hs_off) = if gi == 0 {
                    (n_orig + 1, 0, 1)
                } else {
                    (0, 0, 0)
                };
                let dn_off = if gi == 0 { 0 } else { id_cut[gi] };
                let lb = lemma_base[gi];
                let mut step_ix = step_base[gi];
                let mut acc = acc_base[gi];
                let mut hint_seg_base = hint_base[gi];
                sc.spawn(move || {
                    let mut l = 0usize;
                    for (sg, sm) in segs.iter().zip(sums) {
                        let mut prev_lit_end = sm.lit_lo;
                        for st in steps_of(sg) {
                            if st.tag == b'a' {
                                let dense = (n_orig + lb + l) as u32;
                                dn[st.id as usize - dn_off] = dense + 1;
                                acc += (st.lit_end as usize - prev_lit_end) as u32;
                                prev_lit_end = st.lit_end as usize;
                                cs[cs_off + l] = acc;
                                lc[lc_off + l] = dense;
                                ls[lc_off + l] = step_ix;
                                li[lc_off + l] = st.id;
                                hs[hs_off + l] =
                                    (hint_seg_base + (st.hint_end as usize - sm.hint_lo)) as u32;
                                l += 1;
                            }
                            step_ix += 1;
                        }
                        hint_seg_base += sm.hint_hi - sm.hint_lo;
                    }
                });
            }
        });
    }

    // ---- pass C: deletions ----
    let deleted_at: Vec<AtomicU32> = vec![u32::MAX; n_orig + lemmas]
        .into_iter()
        .map(AtomicU32::new)
        .collect();
    let ok = std::thread::scope(|sc| {
        let hs: Vec<_> = groups
            .iter()
            .enumerate()
            .map(|(gi, &(a, b))| {
                let segs = &segs[a..b];
                let sums = &sums[a..b];
                let (dense_of, lemma_step, deleted_at, bad) =
                    (&dense_of, &lemma_step, &deleted_at, &bad);
                let mut step_ix = step_base[gi];
                sc.spawn(move || {
                    for (sg, sm) in segs.iter().zip(sums) {
                        if bad.load(Ordering::Relaxed) {
                            return false;
                        }
                        let c = all[sg.chunk];
                        let mut cursor = sm.del_lo;
                        for st in steps_of(sg) {
                            if st.tag != b'a' {
                                let end = st.del_end as usize;
                                for &(id, _) in &c.dels[cursor..end] {
                                    let dense = dense_of.get(id as usize).copied().unwrap_or(0);
                                    let fine = dense != 0
                                        && ((dense as usize) <= n_orig
                                            || lemma_step[dense as usize - 1 - n_orig] < step_ix)
                                        && deleted_at[dense as usize - 1]
                                            .swap(step_ix, Ordering::Relaxed)
                                            == u32::MAX;
                                    if !fine {
                                        bad.store(true, Ordering::Relaxed);
                                        return false;
                                    }
                                }
                                cursor = end;
                            }
                            step_ix += 1;
                        }
                    }
                    true
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("plan pass C"))
            .fold(true, |x, y| x & y)
    });
    if !ok {
        return None;
    }
    let deleted_at: Vec<u32> = deleted_at.into_iter().map(AtomicU32::into_inner).collect();
    Some(Fast3a {
        fp: FlatProof {
            lits: Vec::new(),
            clause_start,
            deleted_at,
            lemma_clause,
            lemma_step,
            lemma_id,
            hint_start,
            hints: Vec::new(),
            first_empty,
            n_original: n_orig as u32,
            early: None,
        },
        copies,
        dense_of,
        lit_total: lit_acc,
        hint_total: hint_acc,
    })
}

/// Phases 2 and 3a: follow the true chain across slice boundaries (bridging
/// where a worker was mis-synchronised), then the sequential bookkeeping over
/// steps (ids increasing, dense map, deletions, first empty clause, earliest
/// structural verdict).
pub(crate) fn stitch_plan(
    cnf: &[Vec<i32>],
    bytes: &[u8],
    bounds: &[(usize, usize)],
    chunks: Vec<RawChunk>,
    threads: usize,
) -> Result<StitchPlan, crate::lrat::ParseError> {
    use crate::lrat::ParseError;
    let err = |pos: usize, msg: &str| ParseError {
        line: pos,
        msg: msg.to_string(),
    };
    let n_slices = bounds.len();
    {
        // ---- phase 2: follow the true chain across slice boundaries ----
        let mut segments: Vec<Segment> = Vec::with_capacity(n_slices * 2);
        let mut bridges: Vec<RawChunk> = Vec::new();
        let mut cur = 0usize; // true chain's current offset
        let mut pending_error: Option<ParseError> = None;
        let mut error_chunk: Option<usize> = None; // index into `all` below
        for (i, c) in chunks.iter().enumerate() {
            let (_, shi) = bounds[i];
            if cur >= shi {
                continue; // an earlier chain already covered this slice
            }
            // Does this worker's record contain a step starting at `cur`?
            let mut at = c.steps.binary_search_by(|s| s.pos.cmp(&(cur as u64))).ok();
            if at.is_none() {
                // Bridge: parse steps one at a time until we land on a recorded
                // step start or leave the slice.
                let mut b = RawChunk {
                    end: cur as u64,
                    ..Default::default()
                };
                let mut pos = cur;
                loop {
                    if pos >= bytes.len() || pos >= shi {
                        break;
                    }
                    if let Ok(ix) = c.steps.binary_search_by(|s| s.pos.cmp(&(pos as u64))) {
                        at = Some(ix);
                        break;
                    }
                    let (nl, nh, nd) = (b.lits.len(), b.hints.len(), b.dels.len());
                    match parse_step(bytes, pos, &mut b) {
                        Ok(next) => {
                            pos = next;
                            b.end = pos as u64;
                        }
                        Err(e) => {
                            b.lits.truncate(nl);
                            b.hints.truncate(nh);
                            b.partial_del_lo = nd;
                            b.error = Some(e);
                            break;
                        }
                    }
                }
                let had_error = b.error.clone();
                cur = b.end as usize;
                if !b.steps.is_empty() || b.error.is_some() {
                    let n = b.steps.len();
                    bridges.push(b);
                    if n > 0 {
                        segments.push(Segment {
                            chunk: n_slices + bridges.len() - 1,
                            step_lo: 0,
                            step_hi: n,
                        });
                    }
                }
                if let Some(e) = had_error {
                    pending_error = Some(e);
                    error_chunk = Some(n_slices + bridges.len() - 1);
                    break;
                }
            }
            if let Some(ix) = at {
                if ix < c.steps.len() {
                    segments.push(Segment {
                        chunk: i,
                        step_lo: ix,
                        step_hi: c.steps.len(),
                    });
                }
                cur = c.end as usize;
                if let Some(e) = &c.error {
                    pending_error = Some(e.clone());
                    error_chunk = Some(i);
                    break;
                }
            }
        }
        let all: Vec<&RawChunk> = chunks.iter().chain(bridges.iter()).collect();

        // ---- phase 3a, optimistic: on threads, for a proof with nothing to
        // reject; anything else falls through to the sequential walk ----
        if pending_error.is_none() && threads > 1 {
            if let Some(f) = plan_3a_parallel(cnf, &all, &segments, threads) {
                drop(all);
                let orig_lits: usize = cnf.iter().map(|c| c.len()).sum();
                return Ok(StitchPlan {
                    fp: f.fp,
                    copies: f.copies,
                    dense_of: f.dense_of,
                    lit_total: f.lit_total,
                    hint_total: f.hint_total,
                    orig_lits,
                    chunks,
                    bridges,
                    fast: true,
                });
            }
        }

        // ---- phase 3a: sequential bookkeeping over the true chain ----
        // (ids increasing, dense map, deletions, first empty clause). This is
        // one cheap pass over STEPS, not over bytes.
        let n_original = cnf.len() as u32;
        let mut n_lemmas = 0usize;
        let mut n_dels = 0usize;
        for seg in &segments {
            let c = all[seg.chunk];
            for s in &c.steps[seg.step_lo..seg.step_hi] {
                if s.tag == b'a' {
                    n_lemmas += 1;
                } else {
                    n_dels += 1;
                }
            }
        }
        let _ = n_dels;
        let mut fp = FlatProof {
            lits: Vec::new(),
            clause_start: Vec::with_capacity(cnf.len() + n_lemmas + 1),
            deleted_at: Vec::with_capacity(cnf.len() + n_lemmas),
            lemma_clause: Vec::with_capacity(n_lemmas),
            lemma_step: Vec::with_capacity(n_lemmas),
            lemma_id: Vec::with_capacity(n_lemmas),
            hint_start: Vec::with_capacity(n_lemmas + 1),
            hints: Vec::new(),
            first_empty: None,
            n_original,
            early: None,
        };
        let orig_lits: usize = cnf.iter().map(|c| c.len()).sum();
        fp.clause_start.push(0);
        let mut acc = 0u32;
        for c in cnf {
            acc += c.len() as u32;
            fp.clause_start.push(acc);
            fp.deleted_at.push(u32::MAX);
        }
        fp.hint_start.push(0);
        let mut dense_of: Vec<u32> = vec![0; cnf.len() + 1];
        for i in 0..cnf.len() {
            dense_of[i + 1] = i as u32 + 1;
        }
        let mut last_id = cnf.len() as u64;
        let mut step_ix = 0u32;
        // Per true-chain segment: (chunk, step range actually consumed, lit
        // range, hint range) — phase 3b copies these in parallel.
        let mut copies: Vec<Copy> = Vec::with_capacity(segments.len());
        let mut lit_total = orig_lits;
        let mut hint_total = 0usize;
        let mut consumed_error = pending_error.take();
        'segs: for seg in &segments {
            let c = all[seg.chunk];
            let steps = &c.steps[seg.step_lo..seg.step_hi];
            if steps.is_empty() {
                continue;
            }
            let lit_lo = if seg.step_lo == 0 {
                0
            } else {
                c.steps[seg.step_lo - 1].lit_end as usize
            };
            let hint_lo = if seg.step_lo == 0 {
                0
            } else {
                c.steps[seg.step_lo - 1].hint_end as usize
            };
            let mut del_cursor = if seg.step_lo == 0 {
                0
            } else {
                c.steps[seg.step_lo - 1].del_end as usize
            };
            let mut lit_hi = lit_lo;
            let mut hint_hi = hint_lo;
            let mut prev_lit_end = lit_lo;
            let mut stopped = false;
            for s in steps {
                if s.tag == b'a' {
                    if s.id <= last_id {
                        fp.early = Some(Verdict::Rejected {
                            step: step_ix as usize,
                            id: s.id,
                            reason: Reason::IdNotIncreasing,
                        });
                        stopped = true;
                        break;
                    }
                    if s.id - last_id > crate::lrat::MAX_ID_GAP {
                        return Err(err(
                            s.pos_after_id as usize,
                            "id gap beyond the checker bound",
                        ));
                    }
                    if s.rat {
                        fp.early = Some(Verdict::Unsupported {
                            step: step_ix as usize,
                            id: s.id,
                        });
                        stopped = true;
                        break;
                    }
                    last_id = s.id;
                    let dense = fp.clause_start.len() as u32 - 1;
                    if dense_of.len() <= s.id as usize {
                        dense_of.resize(s.id as usize + 1, 0);
                    }
                    dense_of[s.id as usize] = dense + 1;
                    let n_lits = s.lit_end as usize - prev_lit_end;
                    prev_lit_end = s.lit_end as usize;
                    lit_hi = s.lit_end as usize;
                    hint_hi = s.hint_end as usize;
                    acc += n_lits as u32;
                    fp.clause_start.push(acc);
                    fp.deleted_at.push(u32::MAX);
                    fp.hint_start
                        .push((hint_total + (hint_hi - hint_lo)) as u32);
                    fp.lemma_clause.push(dense);
                    fp.lemma_step.push(step_ix);
                    fp.lemma_id.push(s.id);
                    if n_lits == 0 && fp.first_empty.is_none() {
                        fp.first_empty = Some(fp.lemma_clause.len() as u32 - 1);
                    }
                } else {
                    let end = s.del_end as usize;
                    for &(id, _pos_after) in &c.dels[del_cursor..end] {
                        let dense = dense_of.get(id as usize).copied().unwrap_or(0);
                        if dense == 0 || fp.deleted_at[(dense - 1) as usize] != u32::MAX {
                            fp.early = Some(Verdict::Rejected {
                                step: step_ix as usize,
                                id,
                                reason: Reason::BadDeletion(id),
                            });
                            stopped = true;
                            break;
                        }
                        fp.deleted_at[(dense - 1) as usize] = step_ix;
                    }
                    if stopped {
                        break;
                    }
                    del_cursor = end;
                }
                step_ix += 1;
            }
            copies.push(Copy {
                chunk: seg.chunk,
                lit_lo,
                lit_hi,
                hint_lo,
                hint_hi,
            });
            lit_total += lit_hi - lit_lo;
            hint_total += hint_hi - hint_lo;
            if stopped {
                // Nothing after the early verdict is decoded — including a
                // format error the one-thread decoder would never reach.
                consumed_error = None;
                break 'segs;
            }
        }
        if let Some(e) = consumed_error {
            // The interrupted step: its id / its deletions were checked by the
            // one-thread decoder before the byte that failed.
            // (A structural failure inside the interrupted step would be an
            // `early` verdict at that step, reached before the format error.)
            let mut early_in_partial = false;
            if let Some(c) = error_chunk.map(|ci| all[ci]) {
                if let Some((id, pos_after_id)) = c.partial_add {
                    if id <= last_id {
                        fp.early = Some(Verdict::Rejected {
                            step: step_ix as usize,
                            id,
                            reason: Reason::IdNotIncreasing,
                        });
                        early_in_partial = true;
                    } else if id - last_id > crate::lrat::MAX_ID_GAP {
                        return Err(err(
                            pos_after_id as usize,
                            "id gap beyond the checker bound",
                        ));
                    }
                }
                if !early_in_partial {
                    for &(id, _pos_after) in &c.dels[c.partial_del_lo.min(c.dels.len())..] {
                        let dense = dense_of.get(id as usize).copied().unwrap_or(0);
                        if dense == 0 || fp.deleted_at[(dense - 1) as usize] != u32::MAX {
                            fp.early = Some(Verdict::Rejected {
                                step: step_ix as usize,
                                id,
                                reason: Reason::BadDeletion(id),
                            });
                            early_in_partial = true;
                            break;
                        }
                        fp.deleted_at[(dense - 1) as usize] = step_ix;
                    }
                }
            }
            if !early_in_partial {
                return Err(e);
            }
        }
        drop(all);
        Ok(StitchPlan {
            fp,
            copies,
            dense_of,
            lit_total,
            hint_total,
            orig_lits,
            chunks,
            bridges,
            fast: false,
        })
    }
}

/// Phase 3b on host threads: copy the true chain's literals and hints into
/// the flat arrays at their prefix offsets, resolving hint ids through the
/// dense map; the first hint naming an id never added fails its lemma.
pub(crate) fn stitch_payloads_host(
    cnf: &[Vec<i32>],
    plan: StitchPlan,
    threads: usize,
) -> Result<FlatProof, crate::lrat::ParseError> {
    let StitchPlan {
        mut fp,
        copies,
        dense_of,
        lit_total,
        hint_total,
        orig_lits,
        chunks,
        bridges,
        fast: _,
    } = plan;
    let all: Vec<&RawChunk> = chunks.iter().chain(bridges.iter()).collect();
    {
        fp.lits = vec![0i32; lit_total];
        fp.hints = vec![0u32; hint_total];
        {
            let mut off = 0usize;
            for c in cnf {
                fp.lits[off..off + c.len()].copy_from_slice(c);
                off += c.len();
            }
        }
        // Global index of the first hint naming an id never added (usize::MAX
        // = none) and that id.
        let mut bad_hint = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let mut bad_hint_id = std::sync::atomic::AtomicU64::new(0);
        {
            let lits_out = &mut fp.lits;
            let hints_out = &mut fp.hints;
            let dense_of = &dense_of;
            let bad = &mut bad_hint;
            let bad_id = &mut bad_hint_id;
            // Hand each copy its own disjoint destination slice.
            let mut lit_slices: Vec<&mut [i32]> = Vec::with_capacity(copies.len());
            let mut hint_slices: Vec<&mut [u32]> = Vec::with_capacity(copies.len());
            {
                let mut rest_l: &mut [i32] = &mut lits_out[orig_lits..];
                let mut rest_h: &mut [u32] = &mut hints_out[..];
                for cp in &copies {
                    let (a, b) = rest_l.split_at_mut(cp.lit_hi - cp.lit_lo);
                    lit_slices.push(a);
                    rest_l = b;
                    let (a, b) = rest_h.split_at_mut(cp.hint_hi - cp.hint_lo);
                    hint_slices.push(a);
                    rest_h = b;
                }
            }
            // Each job's global hint offset, for reporting the first bad one.
            let mut hint_offsets: Vec<usize> = Vec::with_capacity(copies.len());
            {
                let mut off = 0usize;
                for cp in &copies {
                    hint_offsets.push(off);
                    off += cp.hint_hi - cp.hint_lo;
                }
            }
            let jobs: Vec<(&Copy, usize, &mut [i32], &mut [u32])> = copies
                .iter()
                .zip(hint_offsets)
                .zip(lit_slices)
                .zip(hint_slices)
                .map(|(((cp, off), ls), hs)| (cp, off, ls, hs))
                .collect();
            // At most `threads` workers, each taking a contiguous run of jobs
            // (the tests force hundreds of tiny slices; one thread per slice
            // would be a thread storm).
            let per = jobs.len().div_ceil(threads.max(1)).max(1);
            let mut it = jobs.into_iter();
            std::thread::scope(|sc| {
                let bad: &std::sync::atomic::AtomicUsize = bad;
                let bad_id: &std::sync::atomic::AtomicU64 = bad_id;
                let all = &all;
                loop {
                    let group: Vec<_> = it.by_ref().take(per).collect();
                    if group.is_empty() {
                        break;
                    }
                    sc.spawn(move || {
                        use std::sync::atomic::Ordering::Relaxed;
                        for (cp, off, ls, hs) in group {
                            let c = all[cp.chunk];
                            ls.copy_from_slice(&c.lits[cp.lit_lo..cp.lit_hi]);
                            for (k, (dst, &h)) in hs
                                .iter_mut()
                                .zip(&c.hints[cp.hint_lo..cp.hint_hi])
                                .enumerate()
                            {
                                let d = dense_of.get(h as usize).copied().unwrap_or(0);
                                if d == 0 {
                                    let g = off + k;
                                    // Keep the smallest global index (and its id).
                                    let mut cur = bad.load(Relaxed);
                                    while g < cur {
                                        match bad.compare_exchange_weak(cur, g, Relaxed, Relaxed) {
                                            Ok(_) => {
                                                bad_id.store(h, Relaxed);
                                                break;
                                            }
                                            Err(seen) => cur = seen,
                                        }
                                    }
                                }
                                *dst = d.wrapping_sub(1);
                            }
                        }
                    });
                }
            });
        }
        let first_bad = *bad_hint.get_mut();
        if first_bad != usize::MAX {
            // Race on bad_id: two threads may have raced on different indices;
            // re-read the id at the winning index to be exact.
            let l = fp.hint_start.partition_point(|&s| s as usize <= first_bad) - 1;
            let mut h = *bad_hint_id.get_mut();
            // Re-derive from the source chunk to be independent of the race.
            {
                let mut g = 0usize;
                for cp in &copies {
                    let n = cp.hint_hi - cp.hint_lo;
                    if first_bad < g + n {
                        h = all[cp.chunk].hints[cp.hint_lo + (first_bad - g)];
                        break;
                    }
                    g += n;
                }
            }
            fp.early = Some(Verdict::Rejected {
                step: fp.lemma_step[l] as usize,
                id: fp.lemma_id[l],
                reason: missing_hint_reason(h, fp.lemma_id[l]),
            });
            truncate_lemmas(&mut fp, l);
        }
        let keep = fp.lemma_clause.len();
        truncate_lemmas(&mut fp, keep);
        Ok(fp)
    }
}

#[cfg(feature = "gpu")]
#[path = "lrat_decode_device.rs"]
pub mod decode_device;
#[path = "lrat_decode_model.rs"]
pub mod decode_model;

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
    /// No code was written for this lemma. The device code buffer is
    /// pre-filled with this value, so a lemma the kernel never reached fails
    /// closed instead of reading as `Ok` (= 0).
    Unwritten = 6,
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
    let fp = flatten(cnf, proof);
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

/// Device path (feature `gpu`): codes from `sat-kernels::lrat_check_lemmas`,
/// combined on the host exactly as the SimWarp codes are.
#[cfg(feature = "gpu")]
pub fn check_warp_gpu(
    ctx: &crate::gpu_launcher::GpuContext,
    cnf: &[Vec<i32>],
    proof: &Proof,
    blocks: u32,
) -> Result<Verdict, crate::gpu_launcher::GpuError> {
    let fp = flatten(cnf, proof);
    let raw = ctx.check_lrat_codes(&fp, blocks)?;
    let codes: Vec<LemmaCode> = raw
        .iter()
        .map(|&c| match c {
            0 => LemmaCode::Ok,
            1 => LemmaCode::NotUnit,
            2 => LemmaCode::NoConflict,
            3 => LemmaCode::DeletedHint,
            4 => LemmaCode::ForwardHint,
            5 => LemmaCode::TableOverflow,
            _ => LemmaCode::Unwritten,
        })
        .collect();
    Ok(combine(&fp, &codes))
}

/// Host-side combination: earliest failing lemma wins; verified iff all
/// lemmas up to the first empty one are `Ok` and such a lemma exists.
pub fn combine(fp: &FlatProof, codes: &[LemmaCode]) -> Verdict {
    if codes.len() != fp.lemma_clause.len() {
        // A short or long code vector means some lemma has no verdict.
        return Verdict::Rejected {
            step: 0,
            id: 0,
            reason: Reason::Unchecked,
        };
    }
    for (i, &c) in codes.iter().enumerate() {
        if c == LemmaCode::Ok {
            continue;
        }
        let step = fp.lemma_step[i] as usize;
        let id = fp.lemma_id[i];
        let hint_at = |pred: &dyn Fn(usize) -> bool| -> u64 {
            let (hs, he) = (fp.hint_start[i] as usize, fp.hint_start[i + 1] as usize);
            if fp.hints.len() < he {
                // Payloads not on the host (the device-resident route): the
                // reason stands, the hint id is reported as unknown (0).
                return 0;
            }
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
            LemmaCode::NoConflict => Reason::NoConflict,
            LemmaCode::TableOverflow => Reason::TableOverflow,
            LemmaCode::Unwritten => Reason::Unchecked,
            LemmaCode::DeletedHint => {
                Reason::DeletedHint(hint_at(&|h| fp.deleted_at[h] < fp.lemma_step[i]))
            }
            LemmaCode::ForwardHint => {
                Reason::ForwardHint(hint_at(&|h| h >= fp.lemma_clause[i] as usize))
            }
        };
        let lemma_fail = Verdict::Rejected { step, id, reason };
        // The structural verdict, if any, is at a later step than every kept
        // lemma — but say so by comparison, not by construction.
        return match &fp.early {
            Some(e) if verdict_step(e) < Some(step) => e.clone(),
            _ => lemma_fail,
        };
    }
    if let Some(e) = &fp.early {
        return e.clone();
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

    #[test]
    fn flat_decode_matches_flatten() {
        use crate::lrat::write_binary;
        let c = parse_cnf_ints(CNF);
        for text in [
            "5 2 0 1 2 0\n6 0 5 3 4 0\n",
            "5 2 0 1 2 0\n5 d 1 2 0\n6 0 5 3 4 0\n",
            "5 2 0 1 2 0\n5 d 3 0\n6 0 5 3 4 0\n",
            "5 2 0 1 6 0\n6 0 5 3 4 0\n",
            "5 2 0 1 2 0\n",
        ] {
            let p = parse_text(text).unwrap();
            let via_flatten = flatten(&c, &p);
            let via_decode = FlatProof::from_binary(&c, &write_binary(&p)).unwrap();
            assert_eq!(via_decode.early, via_flatten.early, "{text:?}");
            assert_eq!(via_decode.lits, via_flatten.lits, "{text:?}");
            assert_eq!(via_decode.clause_start, via_flatten.clause_start);
            assert_eq!(via_decode.deleted_at, via_flatten.deleted_at);
            assert_eq!(via_decode.lemma_clause, via_flatten.lemma_clause);
            assert_eq!(via_decode.lemma_step, via_flatten.lemma_step);
            assert_eq!(via_decode.hint_start, via_flatten.hint_start);
            assert_eq!(via_decode.hints, via_flatten.hints);
            assert_eq!(via_decode.first_empty, via_flatten.first_empty);
            let a = combine(&via_flatten, &check_warp_codes(&via_flatten));
            let b = combine(&via_decode, &check_warp_codes(&via_decode));
            assert_eq!(a, b);
        }
        // What flatten decides from the step sequence, the flat decoder
        // decides the same way (an `early` verdict, not a parse error).
        for (text, want) in [
            (
                "4 2 0 1 2 0\n",
                Verdict::Rejected {
                    step: 0,
                    id: 4,
                    reason: Reason::IdNotIncreasing,
                },
            ),
            ("5 2 0 -1 2 0\n", Verdict::Unsupported { step: 0, id: 5 }),
            (
                "5 2 0 1 9 0\n",
                Verdict::Rejected {
                    step: 0,
                    id: 5,
                    reason: Reason::ForwardHint(9),
                },
            ),
            (
                "5 d 9 0\n",
                Verdict::Rejected {
                    step: 0,
                    id: 9,
                    reason: Reason::BadDeletion(9),
                },
            ),
        ] {
            let p = parse_text(text).unwrap();
            let f = flatten(&c, &p);
            assert_eq!(f.early, Some(want.clone()), "{text:?}: flatten");
            let d = FlatProof::from_binary(&c, &write_binary(&p)).unwrap();
            assert_flat_eq(&d, &f, text);
            assert_eq!(
                combine(&f, &check_warp_codes(&f)),
                want,
                "{text:?}: combine"
            );
            assert_eq!(check_sequential(&c, &p), want, "{text:?}: reference");
        }
    }

    /// A hint below the lemma's id that was never added is `UnknownHint`
    /// (above or equal: `ForwardHint`, the test the reference applies first).
    #[test]
    fn unknown_hint_below_lemma_id() {
        use crate::lrat::{check_independent, check_sequential, write_binary};
        let c = parse_cnf_ints(CNF);
        let p = parse_text("6 2 0 1 5 0\n").unwrap();
        let want = Verdict::Rejected {
            step: 0,
            id: 6,
            reason: Reason::UnknownHint(5),
        };
        assert_eq!(check_sequential(&c, &p), want);
        assert_eq!(check_independent(&c, &p, 2), want);
        let f = flatten(&c, &p);
        assert_eq!(f.early, Some(want.clone()));
        assert_eq!(combine(&f, &check_warp_codes(&f)), want);
        let d = FlatProof::from_binary(&c, &write_binary(&p)).unwrap();
        assert_flat_eq(&d, &f, "unknown hint");
        let dp = FlatProof::from_binary_par_chunked(&c, &write_binary(&p), 2, 3).unwrap();
        assert_flat_eq(&dp, &f, "unknown hint (par)");
    }

    /// Earliest failing step, whichever kind: a lemma using a deleted clause
    /// precedes the later double deletion; a bad deletion precedes a later
    /// failing lemma. All paths, same answer (the hint-level negatives found
    /// the prepass-first paths naming the wrong step, 2026-11-02).
    #[test]
    fn earliest_failure_wins_across_kinds() {
        use crate::lrat::{check_independent, check_sequential, write_binary};
        let c = parse_cnf_ints(CNF);
        for (text, want) in [
            // delete 1, lemma 6 uses 1 (step 2 fails), then 1 deleted again (step 3).
            (
                "5 2 0 1 2 0\n5 d 1 0\n6 0 5 1 4 0\n7 d 1 0\n",
                Verdict::Rejected {
                    step: 2,
                    id: 6,
                    reason: Reason::DeletedHint(1),
                },
            ),
            // bad deletion at step 1 precedes the failing lemma at step 2.
            (
                "5 2 0 1 2 0\n5 d 9 0\n6 0 5 1 4 0\n",
                Verdict::Rejected {
                    step: 1,
                    id: 9,
                    reason: Reason::BadDeletion(9),
                },
            ),
            // failing lemma at step 0 precedes a RAT step at step 1.
            (
                "5 2 0 1 3 0\n6 1 0 -2 0\n",
                Verdict::Rejected {
                    step: 0,
                    id: 5,
                    reason: Reason::NotUnit(3),
                },
            ),
            // unknown hint in lemma at step 1 precedes an id-order error at step 2.
            (
                "5 2 0 1 2 0\n6 1 0 99 0\n3 0 5 3 4 0\n",
                Verdict::Rejected {
                    step: 1,
                    id: 6,
                    reason: Reason::ForwardHint(99),
                },
            ),
        ] {
            let p = parse_text(text).unwrap();
            assert_eq!(check_sequential(&c, &p), want, "{text:?}: sequential");
            assert_eq!(check_independent(&c, &p, 2), want, "{text:?}: independent");
            let f = flatten(&c, &p);
            let w = combine(&f, &check_warp_codes(&f));
            // The warp path does not know the literal in NotUnit.
            let same = match (&w, &want) {
                (
                    Verdict::Rejected {
                        step: a,
                        id: b,
                        reason: Reason::NotUnit(_),
                    },
                    Verdict::Rejected {
                        step: x,
                        id: y,
                        reason: Reason::NotUnit(_),
                    },
                ) => a == x && b == y,
                _ => w == want,
            };
            assert!(same, "{text:?}: warp {w:?} vs {want:?}");
            let bin = write_binary(&p);
            let d = FlatProof::from_binary(&c, &bin).unwrap();
            assert_flat_eq(&d, &f, text);
            let dp = FlatProof::from_binary_par_chunked(&c, &bin, 3, 4).unwrap();
            assert_flat_eq(&dp, &f, text);
        }
    }

    pub(super) fn assert_flat_eq(a: &FlatProof, b: &FlatProof, what: &str) {
        assert_eq!(a.early, b.early, "{what}: early");
        assert_eq!(a.lits, b.lits, "{what}: lits");
        assert_eq!(a.clause_start, b.clause_start, "{what}: clause_start");
        assert_eq!(a.deleted_at, b.deleted_at, "{what}: deleted_at");
        assert_eq!(a.lemma_clause, b.lemma_clause, "{what}: lemma_clause");
        assert_eq!(a.lemma_step, b.lemma_step, "{what}: lemma_step");
        assert_eq!(a.lemma_id, b.lemma_id, "{what}: lemma_id");
        assert_eq!(a.hint_start, b.hint_start, "{what}: hint_start");
        assert_eq!(a.hints, b.hints, "{what}: hints");
        assert_eq!(a.first_empty, b.first_empty, "{what}: first_empty");
        assert_eq!(a.n_original, b.n_original, "{what}: n_original");
    }

    /// Same Result from both decoders — Ok equal field by field, Err equal
    /// (position and message).
    fn assert_same_result(
        cnf: &[Vec<i32>],
        bytes: &[u8],
        threads: usize,
        chunk: usize,
        what: &str,
    ) {
        let a = FlatProof::from_binary(cnf, bytes);
        let b = FlatProof::from_binary_par_chunked(cnf, bytes, threads, chunk);
        match (a, b) {
            (Ok(a), Ok(b)) => assert_flat_eq(&a, &b, what),
            (Err(a), Err(b)) => assert_eq!(a, b, "{what}: error"),
            (a, b) => panic!(
                "{what}: sequential {:?} vs parallel {:?}",
                a.map(|_| ()),
                b.map(|_| ())
            ),
        }
    }

    /// A format-valid proof over 60 clauses whose lemmas' FIRST hint is often
    /// clause 50 — encoded 0x64 = `d`, right after the literals' 0x00
    /// terminator: the one byte pattern that a worker can mistake for a step
    /// start. Deletions and multi-byte ids/literals are mixed in.
    pub(super) fn synthetic_proof(with_empty: bool) -> (Vec<Vec<i32>>, Vec<u8>) {
        let cnf: Vec<Vec<i32>> = (0..60)
            .map(|i| vec![(i % 19) + 1, -((i % 23) + 2)])
            .collect();
        let mut steps = Vec::new();
        let mut id = 61u64;
        let mut live: Vec<u64> = (1..=60).collect();
        let mut x = 12345u32;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for k in 0..400u32 {
            let n_lits = (rnd() % 4) as usize; // 0..3 literals (0 only if with_empty at the end)
            let n_lits = if n_lits == 0 { 1 } else { n_lits };
            let lits: Vec<i32> = (0..n_lits)
                .map(|_| {
                    let v = (rnd() % 300) as i32 + 1; // multi-byte varints past 63
                    if rnd() % 2 == 0 {
                        v
                    } else {
                        -v
                    }
                })
                .collect();
            let mut hints: Vec<i64> = Vec::new();
            if k % 3 != 2 {
                hints.push(50);
            }
            for _ in 0..(rnd() % 4) {
                hints.push(live[(rnd() as usize) % live.len()] as i64);
            }
            steps.push(Step::Add { id, lits, hints });
            live.push(id);
            id += 1 + (rnd() % 3) as u64; // gaps in the id sequence
            if k % 5 == 4 && live.len() > 55 {
                let mut ids = Vec::new();
                for _ in 0..1 + rnd() % 3 {
                    let ix = (rnd() as usize) % live.len();
                    ids.push(live.swap_remove(ix));
                }
                steps.push(Step::Delete { ids });
            }
        }
        if with_empty {
            steps.push(Step::Add {
                id,
                lits: vec![],
                hints: vec![50, 61],
            });
            // A step after the empty clause is still decoded (and checked).
            steps.push(Step::Delete { ids: vec![live[0]] });
        }
        (cnf, crate::lrat::write_binary(&Proof { steps }))
    }

    /// `lrat-check`'s rule: a bad step AFTER the empty clause still fails the
    /// proof, and a RAT step after it is still Unsupported — on every path.
    #[test]
    fn steps_after_the_empty_clause_are_checked_on_every_path() {
        use crate::lrat::{check_independent, check_sequential, write_binary};
        let c = parse_cnf_ints(CNF);
        for (text, want_verified) in [
            ("5 2 0 1 2 0\n6 0 5 3 4 0\n7 1 0 99 0\n", false), // unknown hint after ⊥
            ("5 2 0 1 2 0\n6 0 5 3 4 0\n7 1 0 1 0\n", false),  // no conflict after ⊥
            ("5 2 0 1 2 0\n6 0 5 3 4 0\n7 1 0 5 3 0\n", true), // a valid step after ⊥
            ("5 2 0 1 2 0\n6 0 5 3 4 0\n7 d 5 0\n8 1 0 5 3 0\n", false), // deleted hint after ⊥
        ] {
            let p = parse_text(text).unwrap();
            let seq = check_sequential(&c, &p);
            let ind = check_independent(&c, &p, 2);
            assert_eq!(
                seq.is_verified(),
                want_verified,
                "{text:?}: sequential {seq:?}"
            );
            assert_eq!(ind, seq, "{text:?}: independent");
            let fp = flatten(&c, &p);
            let w = combine(&fp, &check_warp_codes(&fp));
            assert_eq!(w.is_verified(), want_verified, "{text:?}: warp {w:?}");
            assert_eq!(w, seq, "{text:?}: warp vs sequential");
            let fd = FlatProof::from_binary(&c, &write_binary(&p)).unwrap();
            assert_flat_eq(&fd, &fp, text);
            let fpar = FlatProof::from_binary_par_chunked(&c, &write_binary(&p), 2, 3).unwrap();
            assert_flat_eq(&fpar, &fp, text);
        }
        // RAT after ⊥: Unsupported everywhere, never Verified.
        let p = parse_text("5 2 0 1 2 0\n6 0 5 3 4 0\n7 1 0 -1 0\n").unwrap();
        assert!(matches!(
            check_sequential(&c, &p),
            Verdict::Unsupported { .. }
        ));
        assert!(matches!(
            check_independent(&c, &p, 2),
            Verdict::Unsupported { .. }
        ));
        let f = flatten(&c, &p);
        assert!(matches!(
            f.early,
            Some(Verdict::Unsupported { step: 2, id: 7 })
        ));
        assert!(matches!(
            combine(&f, &check_warp_codes(&f)),
            Verdict::Unsupported { .. }
        ));
        let d = FlatProof::from_binary(&c, &write_binary(&p)).unwrap();
        assert_flat_eq(&d, &f, "rat after empty");
    }

    /// The SimWarp table refuses the 4,096th literal; the verdict names the
    /// capacity, not a proof defect.
    #[test]
    fn table_overflow_is_reported_as_capacity() {
        let cnf: Vec<Vec<i32>> = vec![vec![1]];
        let lits: Vec<i32> = (1..=4096).collect();
        let p = Proof {
            steps: vec![Step::Add {
                id: 2,
                lits,
                hints: vec![1],
            }],
        };
        let fp = flatten(&cnf, &p);
        let codes = check_warp_codes(&fp);
        assert_eq!(codes[0], LemmaCode::TableOverflow);
        assert!(matches!(
            combine(&fp, &codes),
            Verdict::Rejected {
                reason: Reason::TableOverflow,
                ..
            }
        ));
        // 4,095 literals fit.
        let p = Proof {
            steps: vec![Step::Add {
                id: 2,
                lits: (1..=4095).collect(),
                hints: vec![1],
            }],
        };
        let fp = flatten(&cnf, &p);
        assert_ne!(check_warp_codes(&fp)[0], LemmaCode::TableOverflow);
    }

    /// `combine` fails closed on a code vector of the wrong length and on an
    /// unwritten code (the device buffer's pre-fill).
    #[test]
    fn combine_fails_closed_on_missing_codes() {
        let c = parse_cnf_ints(CNF);
        let p = parse_text("5 2 0 1 2 0\n6 0 5 3 4 0\n").unwrap();
        let fp = flatten(&c, &p);
        assert!(combine(&fp, &check_warp_codes(&fp)).is_verified());
        assert!(matches!(
            combine(&fp, &[LemmaCode::Ok]),
            Verdict::Rejected {
                reason: Reason::Unchecked,
                ..
            }
        ));
        assert!(matches!(
            combine(&fp, &[LemmaCode::Ok, LemmaCode::Unwritten]),
            Verdict::Rejected {
                reason: Reason::Unchecked,
                ..
            }
        ));
    }

    /// Parser bounds: an id gap or a variable beyond the bound is a parse
    /// error in every parser, never an allocation.
    #[test]
    fn parser_bounds_reject_huge_ids_and_variables() {
        use crate::lrat::{parse_binary, write_binary, MAX_ID_GAP, MAX_VAR};
        let c = parse_cnf_ints(CNF);
        let far = 5 + MAX_ID_GAP + 1;
        let text = format!("5 2 0 1 2 0\n{far} 0 5 3 4 0\n");
        assert!(parse_text(&text).is_err());
        let p = Proof {
            steps: vec![
                Step::Add {
                    id: 5,
                    lits: vec![2],
                    hints: vec![1, 2],
                },
                Step::Add {
                    id: far,
                    lits: vec![],
                    hints: vec![5, 3, 4],
                },
            ],
        };
        let bin = write_binary(&p);
        assert!(parse_binary(&bin).is_err());
        assert!(FlatProof::from_binary(&c, &bin).is_err()); // a parse error, not a verdict
        assert_same_result(&c, &bin, 2, 3, "id gap");
        let big = MAX_VAR as i64 + 1;
        assert!(parse_text(&format!("5 {big} 0 1 2 0\n")).is_err());
        let p = Proof {
            steps: vec![Step::Add {
                id: 5,
                lits: vec![big as i32],
                hints: vec![1],
            }],
        };
        let bin = write_binary(&p);
        assert!(parse_binary(&bin).is_err());
        assert!(FlatProof::from_binary(&c, &bin).is_err());
        assert_same_result(&c, &bin, 2, 3, "big var");
        // Literal 0 — encoded 1, "−0" — names no variable and has no text
        // form; every binary parser refuses it (accepted until 2026-11-04).
        // `a 5 | 2 −0 0 | 1 2 0`, and the same with the 0 first.
        for lits in [&[0x04u8, 0x01][..], &[0x01, 0x04][..]] {
            let mut bin = vec![b'a', 0x0a];
            bin.extend_from_slice(lits);
            bin.extend_from_slice(&[0x00, 0x02, 0x04, 0x00]);
            assert!(parse_binary(&bin).is_err(), "{lits:?}");
            assert!(FlatProof::from_binary(&c, &bin).is_err(), "{lits:?}");
            assert_same_result(&c, &bin, 2, 3, "literal 0");
        }
    }

    #[test]
    fn par_decode_equals_sequential_on_valid_proofs() {
        for with_empty in [false, true] {
            let (cnf, bytes) = synthetic_proof(with_empty);
            // The 'd' mimic really is in the stream: 0x00 then 0x64 somewhere.
            assert!(bytes.windows(2).any(|w| w == [0, b'd']));
            for (threads, chunk) in [
                (2, 5),
                (3, 7),
                (4, 64),
                (8, 1),
                (16, 1000),
                (3, bytes.len() + 10),
            ] {
                assert_same_result(
                    &cnf,
                    &bytes,
                    threads,
                    chunk,
                    &format!("valid t={threads} c={chunk} e={with_empty}"),
                );
            }
            let p = crate::lrat::parse_binary(&bytes).unwrap();
            let via_flatten = flatten(&cnf, &p);
            let via_par = FlatProof::from_binary_par_chunked(&cnf, &bytes, 4, 9).unwrap();
            assert_flat_eq(&via_par, &via_flatten, "par vs flatten");
        }
    }

    /// Phases 1–3a on host threads with the given phase-3a thread count.
    fn plan_with(
        cnf: &[Vec<i32>],
        bytes: &[u8],
        chunk: usize,
        threads: usize,
    ) -> Result<StitchPlan, crate::lrat::ParseError> {
        let bounds = slice_bounds(bytes.len(), chunk);
        let chunks = parse_slices_host(bytes, &bounds, 3);
        stitch_plan(cnf, bytes, &bounds, chunks, threads)
    }

    /// Two plans equal in everything phase 3b consumes.
    fn assert_plan_eq(a: &StitchPlan, b: &StitchPlan, what: &str) {
        assert_flat_eq(&a.fp, &b.fp, what);
        let cp = |p: &StitchPlan| -> Vec<(usize, usize, usize, usize, usize)> {
            p.copies
                .iter()
                .map(|c| (c.chunk, c.lit_lo, c.lit_hi, c.hint_lo, c.hint_hi))
                .collect()
        };
        assert_eq!(cp(a), cp(b), "{what}: copies");
        assert_eq!(a.dense_of, b.dense_of, "{what}: dense_of");
        assert_eq!(a.lit_total, b.lit_total, "{what}: lit_total");
        assert_eq!(a.hint_total, b.hint_total, "{what}: hint_total");
        assert_eq!(a.orig_lits, b.orig_lits, "{what}: orig_lits");
    }

    /// The optimistic parallel phase 3a against the sequential walk on valid
    /// proofs: taken (not fallen back from), and equal in every output.
    #[test]
    fn plan_parallel_equals_sequential_walk_on_valid_proofs() {
        for with_empty in [false, true] {
            let (cnf, bytes) = synthetic_proof(with_empty);
            for chunk in [1, 5, 13, 64, bytes.len() + 3] {
                let seq = plan_with(&cnf, &bytes, chunk, 1).unwrap();
                assert!(!seq.fast, "one thread runs the sequential walk");
                for threads in [2, 3, 8, 64] {
                    let par = plan_with(&cnf, &bytes, chunk, threads).unwrap();
                    let what = format!("e={with_empty} c={chunk} t={threads}");
                    assert!(par.fast, "{what}: valid proof fell back");
                    assert_plan_eq(&par, &seq, &what);
                }
            }
        }
    }

    /// Every kind of thing phase 3a rejects, placed mid-proof among valid
    /// filler so it lands in a later group: the parallel walk must hand the
    /// proof to the sequential walk (`fast == false`), and the result —
    /// verdict or format error — must be the sequential walk's.
    #[test]
    fn plan_parallel_falls_back_on_every_rejection() {
        use crate::lrat::{write_binary, MAX_ID_GAP};
        let cnf: Vec<Vec<i32>> = (0..60).map(|i| vec![i + 1]).collect();
        let filler = |from: u64, n: u64| -> Vec<Step> {
            (0..n)
                .flat_map(|k| {
                    let id = from + k;
                    let mut v = vec![Step::Add {
                        id,
                        lits: vec![(k % 50) as i32 + 1],
                        hints: vec![1, 2],
                    }];
                    if k % 4 == 3 {
                        v.push(Step::Delete { ids: vec![id - 1] });
                    }
                    v
                })
                .collect()
        };
        let add = |id: u64, hints: Vec<i64>| Step::Add {
            id,
            lits: vec![3],
            hints,
        };
        let del = |ids: Vec<u64>| Step::Delete { ids };
        // (name, steps placed between 300 filler adds (ids 61..361) and the tail)
        let cases: Vec<(&str, Vec<Step>)> = vec![
            (
                "id not increasing",
                vec![add(400, vec![1]), add(400, vec![1])],
            ),
            ("id decreasing", vec![add(400, vec![1]), add(399, vec![1])]),
            ("RAT hint", vec![add(400, vec![1, -2, 3])]),
            ("delete never-added id", vec![del(vec![365])]),
            (
                "delete twice across steps",
                vec![del(vec![7]), del(vec![7])],
            ),
            ("delete twice in one step", vec![del(vec![8, 8])]),
            (
                "delete before its add",
                vec![del(vec![400]), add(400, vec![1])],
            ),
            ("delete a lemma twice", vec![del(vec![100]), del(vec![100])]),
            (
                "id gap beyond the bound",
                vec![add(400 + MAX_ID_GAP + 1, vec![1])],
            ),
        ];
        for (name, mid) in cases {
            // A valid tail just above every id used in the middle (within the
            // gap bound, so the tail alone never forces the fallback).
            let top = mid
                .iter()
                .filter_map(|s| match s {
                    Step::Add { id, .. } => Some(*id),
                    _ => None,
                })
                .max()
                .unwrap_or(0)
                .max(360);
            let mut steps = filler(61, 300);
            steps.extend(mid);
            steps.extend(
                filler(top + 1, 50)
                    .into_iter()
                    .filter(|s| !matches!(s, Step::Delete { ids } if ids[0] <= top)),
            );
            let bytes = write_binary(&Proof { steps });
            // Many thread counts, so that at some of them the two halves of a
            // violation (an id pair, a deletion pair) sit in different groups.
            for chunk in [1, 7, 64] {
                let seq = plan_with(&cnf, &bytes, chunk, 1);
                for threads in [2, 3, 4, 5, 7, 16, 64, 256] {
                    let par = plan_with(&cnf, &bytes, chunk, threads);
                    let what = format!("{name} c={chunk} t={threads}");
                    match (&seq, &par) {
                        (Ok(a), Ok(b)) => {
                            assert!(!b.fast, "{what}: the parallel walk accepted it");
                            assert!(a.fp.early.is_some(), "{what}: no verdict at all");
                            assert_plan_eq(b, a, &what);
                        }
                        (Err(a), Err(b)) => assert_eq!(a, b, "{what}: error"),
                        _ => panic!("{what}: Ok/Err differ"),
                    }
                }
            }
        }
    }

    /// Every prefix of the proof: both decoders return the same Result.
    #[test]
    fn par_decode_equals_sequential_on_every_truncation() {
        let (cnf, bytes) = synthetic_proof(true);
        for cut in 0..bytes.len() {
            assert_same_result(&cnf, &bytes[..cut], 3, 11, &format!("truncate at {cut}"));
        }
    }

    /// Every single-bit flip of the proof (non-minimal varints, stray zero
    /// bytes, bogus tags, ids out of order, RAT hints …): same Result.
    #[test]
    fn par_decode_equals_sequential_on_every_bit_flip() {
        let (cnf, bytes) = synthetic_proof(false);
        let mut m = bytes.clone();
        for i in 0..bytes.len() {
            for bit in 0..8 {
                m[i] ^= 1 << bit;
                assert_same_result(&cnf, &m, 3, 13, &format!("flip byte {i} bit {bit}"));
                m[i] ^= 1 << bit;
            }
        }
    }

    /// The parallel decoder's own fault: skipping the bridge (trusting a
    /// worker's first recorded step blindly) must be caught by the valid-proof
    /// test above — demonstrated here by hand: a worker started on the 'd'
    /// mimic yields a different step list.
    #[test]
    fn mimic_candidate_parses_differently() {
        let (_, bytes) = synthetic_proof(false);
        let p = bytes.windows(2).position(|w| w == [0, b'd']).unwrap() + 1;
        let wrong = parse_run(&bytes, p, p + 1);
        assert_eq!(
            wrong.steps[0].tag, b'd',
            "the mimic parses as a deletion step"
        );
        // and the true chain has no step starting there
        let truth = parse_run(&bytes, 0, bytes.len());
        assert!(truth
            .steps
            .binary_search_by(|s| s.pos.cmp(&(p as u64)))
            .is_err());
    }

    /// Archive agreement: reference vs warp on every proof under
    /// `LRAT_ARCHIVE_DIRS` (skips when unset).
    /// Archive: the optimistic parallel phase 3a against the sequential walk
    /// on every proof under `LRAT_ARCHIVE_DIRS` (skips when unset), with a
    /// tally of which walk ran against the reference verdict. A proof whose
    /// failure is a bad hint passes phase 3a — the check decides it — so
    /// "parallel and rejected" is expected; "sequential and verified" is not.
    #[test]
    fn archive_plan_parallel_equals_sequential_walk() {
        let Ok(dirs) = std::env::var("LRAT_ARCHIVE_DIRS") else {
            eprintln!(
                "archive_plan_parallel_equals_sequential_walk: SKIPPED (set LRAT_ARCHIVE_DIRS)"
            );
            return;
        };
        // (fast, verified) → count
        let mut tally = std::collections::BTreeMap::<(bool, bool), usize>::new();
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
                let (Ok(cnf_text), Ok(proof_text)) = (
                    std::fs::read_to_string(dir.join("cnf").join(format!("{base}.cnf"))),
                    std::fs::read_to_string(&p),
                ) else {
                    continue;
                };
                let Ok(proof) = parse_text(&proof_text) else {
                    continue;
                };
                let cnf = parse_cnf_ints(&cnf_text);
                let bin = crate::lrat::write_binary(&proof);
                let verified = check_sequential(&cnf, &proof).is_verified();
                let mut fast_any = None;
                for chunk in [64, 4096] {
                    let seq = plan_with(&cnf, &bin, chunk, 1);
                    let par = plan_with(&cnf, &bin, chunk, 4);
                    let what = format!("{stem} c={chunk}");
                    match (&seq, &par) {
                        (Ok(a), Ok(b)) => {
                            assert_plan_eq(b, a, &what);
                            assert!(fast_any.is_none() || fast_any == Some(b.fast), "{what}");
                            fast_any = Some(b.fast);
                        }
                        (Err(a), Err(b)) => {
                            assert_eq!(a, b, "{what}: error");
                            fast_any = Some(false);
                        }
                        _ => panic!("{what}: Ok/Err differ"),
                    }
                }
                *tally.entry((fast_any.unwrap(), verified)).or_default() += 1;
            }
        }
        eprintln!("archive_plan_parallel_equals_sequential_walk: (parallel, verified) → proofs: {tally:?}");
        assert!(
            !tally.contains_key(&(false, true)),
            "a verified proof fell back"
        );
    }

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
                {
                    let fp = flatten(&cnf, &proof);
                    let codes = check_warp_codes(&fp);
                    overflow += codes
                        .iter()
                        .filter(|&&c| c == LemmaCode::TableOverflow)
                        .count();
                    // The direct binary decoder must produce the same layout
                    // and the same verdict as flatten(parse(..)).
                    let bin = crate::lrat::write_binary(&proof);
                    let fd = FlatProof::from_binary(&cnf, &bin)
                        .expect("flat decode of a flatten-able proof");
                    let fpar = FlatProof::from_binary_par_chunked(
                        &cnf,
                        &bin,
                        4,
                        bin.len().div_ceil(4).max(1),
                    )
                    .unwrap_or_else(|e| panic!("{stem}: from_binary_par: {e:?}"));
                    assert_flat_eq(&fpar, &fd, &stem);
                    assert_eq!(fd.hints, fp.hints, "{stem}: flat decode hints differ");
                    assert_eq!(
                        fd.deleted_at, fp.deleted_at,
                        "{stem}: flat decode deletions differ"
                    );
                    assert_eq!(
                        combine(&fd, &check_warp_codes(&fd)),
                        combine(&fp, &codes),
                        "{stem}: flat verdict"
                    );
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
