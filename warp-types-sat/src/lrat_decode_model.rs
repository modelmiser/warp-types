//! SimWarp model of the device-side binary LRAT decode
//! (`research/gpu-lrat-decode.md` §1): one warp per slice, 32 bytes per
//! window, one byte per lane.
//!
//! The contract is the host decoder's phase-1 contract: for a slice `[lo, hi)`
//! the record must be exactly `parse_run(bytes, start, hi)` from the slice's
//! first candidate. The warp formulation reaches it without a sequential
//! byte walk:
//!
//! * **tokens** — a byte with the high bit clear ends a token (a varint or the
//!   one-byte tag). `ballot(term)` gives the window's token ends; the lane at a
//!   token's last byte reconstructs its value from at most 10 bytes.
//! * **roles** — a five-state machine over tokens (`TAG → ID → LITS → HINTS →
//!   TAG` for `a`, `TAG → DELS → TAG` for `d`). Each terminator lane holds its
//!   token's transition table (6 states × 3 bits); an inclusive scan composes
//!   them, so every lane knows the state before its token from the state at
//!   window entry. This is the parallel finite-state-machine-by-scan
//!   construction.
//! * **slots** — output positions are `running + popcount(ballot & lanes below
//!   me)` per role; no sequential counter.
//!
//! Anything the one-thread decoder would turn into a `ParseError` (a bad tag, a
//! varint longer than 10 bytes, a literal out of range, bytes running out
//! mid-step) makes the slice *dirty*: the model returns `None` and the caller
//! re-parses that slice with `parse_run`. So the model has to agree with the
//! host only on slices it calls clean — and the tests check exactly that, slice
//! by slice, on every truncation and every bit flip of the synthetic proof, so
//! that "dirty" cannot hide a wrong record.
//!
//! The scan is the lane-guarded Hillis–Steele. The core `Warp::inclusive_sum`
//! lacked that guard until 2026-11-04 and only sums; the composition scan is
//! written here, tested against a sequential fold, and shown to fail without
//! the guard.

use super::{first_candidate, parse_run, RawChunk, RawStep};
use warp_types::simwarp::SimWarp;

/// Lanes per warp (the model is 32-wide like the kernel).
pub const WARP: usize = 32;
/// Longest varint the one-thread decoder accepts (`shift > 63` after it).
const MAX_VARINT: usize = 10;

const S_TAG: u32 = 0;
const S_ID: u32 = 1;
const S_LITS: u32 = 2;
const S_HINTS: u32 = 3;
const S_DELS: u32 = 4;
const S_ERR: u32 = 5;
const N_STATES: u32 = 6;

/// Token classes: 0 = value zero, 1 = the one-byte tag `a`, 2 = `d`, 3 = other.
const C_ZERO: u32 = 0;
const C_A: u32 = 1;
const C_D: u32 = 2;
const C_OTHER: u32 = 3;

/// One transition of the role machine.
#[inline]
pub(crate) const fn transition(state: u32, class: u32) -> u32 {
    match state {
        S_TAG => match class {
            C_A => S_ID,
            C_D => S_DELS,
            _ => S_ERR,
        },
        S_ID => S_LITS,
        S_LITS => {
            if class == C_ZERO {
                S_HINTS
            } else {
                S_LITS
            }
        }
        S_HINTS => {
            if class == C_ZERO {
                S_TAG
            } else {
                S_HINTS
            }
        }
        S_DELS => {
            if class == C_ZERO {
                S_TAG
            } else {
                S_DELS
            }
        }
        _ => S_ERR,
    }
}

/// A token's whole transition table: 3 bits per source state, 18 bits.
#[inline]
pub(crate) const fn table_for(class: u32) -> u32 {
    let mut t = 0u32;
    let mut s = 0u32;
    while s < N_STATES {
        t |= transition(s, class) << (3 * s);
        s += 1;
    }
    t
}

/// The identity table (a lane holding no token end).
#[inline]
pub(crate) const fn table_identity() -> u32 {
    let mut t = 0u32;
    let mut s = 0u32;
    while s < N_STATES {
        t |= s << (3 * s);
        s += 1;
    }
    t
}

#[inline]
pub(crate) const fn apply(table: u32, state: u32) -> u32 {
    (table >> (3 * state)) & 7
}

/// `then ∘ first`: apply `first`, then `then`.
#[inline]
pub(crate) const fn compose(first: u32, then: u32) -> u32 {
    let mut t = 0u32;
    let mut s = 0u32;
    while s < N_STATES {
        t |= apply(then, apply(first, s)) << (3 * s);
        s += 1;
    }
    t
}

/// Ballot: bit `i` set when lane `i`'s predicate holds.
#[inline]
pub(crate) fn ballot(pred: &[bool; WARP]) -> u32 {
    let mut b = 0u32;
    for (i, &p) in pred.iter().enumerate() {
        if p {
            b |= 1 << i;
        }
    }
    b
}

/// `%lanemask_lt`: the lanes below `lane`.
#[inline]
pub(crate) const fn lanes_below(lane: u32) -> u32 {
    (1u32 << lane) - 1
}

/// Inclusive scan of transition tables under composition (lane `i` ends with
/// `f_i ∘ … ∘ f_0`), Hillis–Steele on `shuffle_up` with the lane guard.
pub(crate) fn scan_compose(tables: [u32; WARP]) -> [u32; WARP] {
    scan_compose_guarded(tables, true)
}

/// The same scan with the guard switchable, so a test can show that the
/// unguarded form (what `Warp::inclusive_sum` did through 0.4.3) is wrong.
pub(crate) fn scan_compose_guarded(tables: [u32; WARP], guard: bool) -> [u32; WARP] {
    let mut v = SimWarp::<u32>::from_array(tables);
    let mut d = 1u32;
    while (d as usize) < WARP {
        let up = v.shuffle_up(d);
        v = v.map(|lane, x| {
            if !guard || lane as u32 >= d {
                compose(up.lane(lane), x)
            } else {
                x
            }
        });
        d *= 2;
    }
    let mut out = [0u32; WARP];
    for (i, o) in out.iter_mut().enumerate() {
        *o = v.lane(i);
    }
    out
}

/// What the warp carries from one window to the next (all warp-uniform).
struct Carry {
    /// Role state after the last kept token.
    state: u32,
    /// Start offset of the token in progress.
    tok_start: usize,
    /// Steps started so far (a step's record index is its start order).
    steps: usize,
    lits: usize,
    hints: usize,
    dels: usize,
    /// A negative hint seen in the open `a` step.
    rat: bool,
}

/// Decode one slice with the warp formulation. `Some(record)` equals
/// `parse_run(bytes, start, hi)`; `None` = dirty (the caller falls back).
///
/// The lane loops index several per-lane arrays by `lane` on purpose: each
/// body is what one lane of the kernel does, and the kernel is read against
/// it line by line.
#[allow(clippy::needless_range_loop)]
pub(crate) fn decode_slice_model(bytes: &[u8], lo: usize, hi: usize) -> Option<RawChunk> {
    let len = bytes.len();
    let hi = hi.min(len);
    // ---- the speculative start: offset 0, or an `a`/`d` after a 0x00 ----
    let start = if lo == 0 {
        0
    } else {
        let mut found = None;
        let mut base = lo;
        while base < hi && found.is_none() {
            let mut cand = [false; WARP];
            for (lane, c) in cand.iter_mut().enumerate() {
                let p = base + lane;
                *c = p < hi && bytes[p - 1] == 0 && (bytes[p] == b'a' || bytes[p] == b'd');
            }
            let b = ballot(&cand);
            if b != 0 {
                found = Some(base + b.trailing_zeros() as usize);
            }
            base += WARP;
        }
        match found {
            Some(s) => s,
            None => {
                return Some(RawChunk {
                    end: lo as u64,
                    ..Default::default()
                })
            }
        }
    };
    let identity = table_identity();
    let mut out = RawChunk {
        end: start as u64,
        ..Default::default()
    };
    let mut carry = Carry {
        state: S_TAG,
        tok_start: start,
        steps: 0,
        lits: 0,
        hints: 0,
        dels: 0,
        rat: false,
    };
    let mut base = start;
    loop {
        if base >= len {
            // Bytes ran out before a step boundary at or past `hi`: the
            // one-thread decoder reports a truncation here.
            return None;
        }
        // ---- per-lane byte and token end ----
        let mut term = [false; WARP];
        for (lane, t) in term.iter_mut().enumerate() {
            let p = base + lane;
            *t = p < len && bytes[p] & 0x80 == 0;
        }
        let term_b = ballot(&term);
        if term_b == 0 {
            // 32 continuation bytes: the varint is longer than 10 bytes.
            return None;
        }
        // ---- each terminator lane: token start, value, class ----
        let mut value = [0u64; WARP];
        let mut class = [C_OTHER; WARP];
        let mut tok_len = [0usize; WARP];
        let mut tables = [identity; WARP];
        for lane in 0..WARP {
            if !term[lane] {
                continue;
            }
            let p = base + lane;
            let below = term_b & lanes_below(lane as u32);
            let tstart = if below != 0 {
                base + (31 - below.leading_zeros() as usize) + 1
            } else {
                carry.tok_start
            };
            let n = p - tstart + 1;
            tok_len[lane] = n;
            if n > MAX_VARINT {
                return None;
            }
            let mut x = 0u64;
            for (k, &b) in bytes[tstart..=p].iter().enumerate() {
                x |= ((b & 0x7f) as u64) << (7 * k);
            }
            value[lane] = x;
            let b = bytes[p];
            class[lane] = if n == 1 && b == b'a' {
                C_A
            } else if n == 1 && b == b'd' {
                C_D
            } else if x == 0 {
                C_ZERO
            } else {
                C_OTHER
            };
            tables[lane] = table_for(class[lane]);
        }
        // ---- the role scan ----
        let incl = scan_compose(tables);
        let mut before = [S_TAG; WARP];
        for lane in 0..WARP {
            let excl = if lane == 0 { identity } else { incl[lane - 1] };
            before[lane] = apply(excl, carry.state);
        }
        // ---- role ballots ----
        let mut is_start = [false; WARP];
        let mut is_id = [false; WARP];
        let mut is_lit = [false; WARP];
        let mut is_hint = [false; WARP];
        let mut is_rat = [false; WARP];
        let mut is_del = [false; WARP];
        let mut is_end = [false; WARP];
        let mut is_err = [false; WARP];
        for lane in 0..WARP {
            if !term[lane] {
                continue;
            }
            let s = before[lane];
            let c = class[lane];
            let v = value[lane];
            match s {
                S_TAG => {
                    if c == C_A || c == C_D {
                        is_start[lane] = true;
                    } else {
                        is_err[lane] = true;
                    }
                }
                S_ID => is_id[lane] = true,
                S_LITS => {
                    if v != 0 {
                        is_lit[lane] = true;
                        let mag = (v >> 1) as i64;
                        let signed = if v & 1 == 1 { -mag } else { mag };
                        match i32::try_from(signed) {
                            Ok(l) if l.unsigned_abs() <= crate::lrat::MAX_VAR => {}
                            _ => is_err[lane] = true,
                        }
                    }
                }
                S_HINTS => {
                    if v != 0 {
                        is_hint[lane] = true;
                        is_rat[lane] = v & 1 == 1;
                    } else {
                        is_end[lane] = true;
                    }
                }
                S_DELS => {
                    if v != 0 {
                        is_del[lane] = true;
                    } else {
                        is_end[lane] = true;
                    }
                }
                _ => is_err[lane] = true,
            }
        }
        // ---- the cut: the first step end at or past `hi` ends the slice ----
        let end_b = ballot(&is_end);
        let mut cut: Option<usize> = None;
        {
            let mut b = end_b;
            while b != 0 {
                let lane = b.trailing_zeros() as usize;
                if base + lane + 1 >= hi {
                    cut = Some(lane);
                    break;
                }
                b &= b - 1;
            }
        }
        let keep = match cut {
            Some(lane) => lanes_below(lane as u32) | (1 << lane),
            None => u32::MAX,
        };
        if ballot(&is_err) & keep != 0 {
            return None;
        }
        let start_b = ballot(&is_start) & keep;
        let id_b = ballot(&is_id) & keep;
        let lit_b = ballot(&is_lit) & keep;
        let hint_b = ballot(&is_hint) & keep;
        let rat_b = ballot(&is_rat) & keep;
        let del_b = ballot(&is_del) & keep;
        let end_b = end_b & keep;
        // ---- writes, slot = running + popcount(ballot & lanes below) ----
        let n_start = start_b.count_ones() as usize;
        let n_lit = lit_b.count_ones() as usize;
        let n_hint = hint_b.count_ones() as usize;
        let n_del = del_b.count_ones() as usize;
        out.steps.resize(
            carry.steps + n_start,
            RawStep {
                pos: 0,
                id: 0,
                pos_after_id: 0,
                lit_end: 0,
                hint_end: 0,
                del_end: 0,
                tag: 0,
                rat: false,
            },
        );
        out.lits.resize(carry.lits + n_lit, 0);
        out.hints.resize(carry.hints + n_hint, 0);
        out.dels.resize(carry.dels + n_del, (0, 0));
        for lane in 0..WARP {
            if keep & (1 << lane) == 0 {
                break;
            }
            let p = base + lane;
            let lb = lanes_below(lane as u32);
            if start_b & (1 << lane) != 0 {
                let k = carry.steps + (start_b & lb).count_ones() as usize;
                out.steps[k].pos = p as u64;
                out.steps[k].tag = bytes[p];
            }
            // The step this lane belongs to: the last start at or below it,
            // else the one carried in from an earlier window.
            let my_step = carry.steps + (start_b & (lb | (1 << lane))).count_ones() as usize;
            if id_b & (1 << lane) != 0 {
                let k = my_step - 1;
                out.steps[k].id = value[lane] >> 1;
                out.steps[k].pos_after_id = (p + 1) as u64;
            }
            if lit_b & (1 << lane) != 0 {
                let v = value[lane];
                let mag = (v >> 1) as i64;
                let signed = if v & 1 == 1 { -mag } else { mag };
                out.lits[carry.lits + (lit_b & lb).count_ones() as usize] = signed as i32;
            }
            if hint_b & (1 << lane) != 0 {
                out.hints[carry.hints + (hint_b & lb).count_ones() as usize] = value[lane] >> 1;
            }
            if del_b & (1 << lane) != 0 {
                out.dels[carry.dels + (del_b & lb).count_ones() as usize] =
                    (value[lane] >> 1, (p + 1) as u64);
            }
            if end_b & (1 << lane) != 0 {
                let k = my_step - 1;
                // Rat bits of this step: those since its start lane (or since
                // the window began, with the carried flag).
                let since = match (start_b & (lb | (1 << lane))).count_ones() {
                    0 => u32::MAX,
                    _ => {
                        let s = 31 - (start_b & (lb | (1 << lane))).leading_zeros();
                        !lanes_below(s) & !(1 << s)
                    }
                };
                let carried = if (start_b & lb) == 0 {
                    carry.rat
                } else {
                    false
                };
                out.steps[k].rat = carried || (rat_b & since & lb) != 0;
                out.steps[k].lit_end = (carry.lits + (lit_b & lb).count_ones() as usize) as u32;
                out.steps[k].hint_end = (carry.hints + (hint_b & lb).count_ones() as usize) as u32;
                out.steps[k].del_end = (carry.dels + (del_b & lb).count_ones() as usize) as u32;
            }
        }
        // ---- carry to the next window ----
        carry.steps += n_start;
        carry.lits += n_lit;
        carry.hints += n_hint;
        carry.dels += n_del;
        if let Some(lane) = cut {
            out.end = (base + lane + 1) as u64;
            return Some(out);
        }
        carry.state = apply(incl[WARP - 1], carry.state);
        carry.tok_start = base + (31 - term_b.leading_zeros() as usize) + 1;
        // Rat flag of the step left open: bits after its start lane (or all
        // bits with the carried flag when it started earlier).
        carry.rat = if start_b != 0 {
            let s = 31 - start_b.leading_zeros();
            (rat_b & !lanes_below(s) & !(1 << s)) != 0
        } else {
            carry.rat || rat_b != 0
        };
        base += WARP;
    }
}

/// Phase 1 by the model for every slice, with the host fallback on dirty
/// slices. Returns the records and how many slices fell back.
pub(crate) fn parse_slices_model(
    bytes: &[u8],
    bounds: &[(usize, usize)],
) -> (Vec<RawChunk>, usize) {
    let mut dirty = 0usize;
    let chunks = bounds
        .iter()
        .map(|&(lo, hi)| match decode_slice_model(bytes, lo, hi) {
            Some(c) => c,
            None => {
                dirty += 1;
                host_record(bytes, lo, hi)
            }
        })
        .collect();
    (chunks, dirty)
}

/// What the host's phase 1 produces for one slice.
pub(crate) fn host_record(bytes: &[u8], lo: usize, hi: usize) -> RawChunk {
    match first_candidate(bytes, lo, hi) {
        Some(start) => parse_run(bytes, start, hi),
        None => RawChunk {
            end: lo as u64,
            ..Default::default()
        },
    }
}

impl super::FlatProof {
    /// [`FlatProof::from_binary`](super::FlatProof::from_binary) through the
    /// warp model's phase 1 (slices of `chunk` bytes) and the shared stitch.
    pub fn from_binary_model(
        cnf: &[Vec<i32>],
        bytes: &[u8],
        chunk: usize,
    ) -> Result<super::FlatProof, crate::lrat::ParseError> {
        if bytes.is_empty() {
            return super::FlatProof::from_binary(cnf, bytes);
        }
        let bounds = super::slice_bounds(bytes.len(), chunk);
        let (chunks, _dirty) = parse_slices_model(bytes, &bounds);
        super::stitch_chunks(cnf, bytes, &bounds, chunks, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{assert_flat_eq, synthetic_proof};
    use super::super::{slice_bounds, FlatProof};
    use super::*;

    fn fold(tables: &[u32]) -> u32 {
        tables
            .iter()
            .fold(table_identity(), |acc, &t| compose(acc, t))
    }

    #[test]
    fn tables_round_trip_the_transition_function() {
        for c in 0..4 {
            let t = table_for(c);
            for s in 0..N_STATES {
                assert_eq!(apply(t, s), transition(s, c));
            }
        }
        for s in 0..N_STATES {
            assert_eq!(apply(table_identity(), s), s);
        }
    }

    /// The guarded scan equals the sequential fold on random tables; the
    /// unguarded one (the deprecated core method's shape) does not.
    #[test]
    fn compose_scan_is_a_fold_and_fails_without_the_guard() {
        let mut seed = 0x9E37_79B9u32;
        let mut wrong_once = false;
        for _ in 0..200 {
            let mut tables = [0u32; WARP];
            for t in tables.iter_mut() {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *t = match seed % 5 {
                    0 => table_identity(),
                    k => table_for(k - 1),
                };
            }
            let incl = scan_compose(tables);
            for i in 0..WARP {
                assert_eq!(incl[i], fold(&tables[..=i]), "lane {i}");
            }
            let bad = scan_compose_guarded(tables, false);
            wrong_once |= bad != incl;
        }
        assert!(wrong_once, "the unguarded scan must be seen to be wrong");
    }

    /// Per slice, the model's record is the host's — or the model says dirty
    /// and the host's own record had an error. Never a clean, different one.
    fn assert_slices_match(bytes: &[u8], chunk: usize, what: &str) -> (usize, usize) {
        let bounds = slice_bounds(bytes.len(), chunk);
        let (mut clean, mut dirty) = (0usize, 0usize);
        for &(lo, hi) in &bounds {
            let host = host_record(bytes, lo, hi);
            match decode_slice_model(bytes, lo, hi) {
                Some(m) => {
                    assert_eq!(m, host, "{what}: slice {lo}..{hi}");
                    clean += 1;
                }
                None => {
                    assert!(
                        host.error.is_some(),
                        "{what}: slice {lo}..{hi} dirty but the host parsed it cleanly: {host:?}"
                    );
                    dirty += 1;
                }
            }
        }
        (clean, dirty)
    }

    fn assert_same_result_model(cnf: &[Vec<i32>], bytes: &[u8], chunk: usize, what: &str) {
        let a = FlatProof::from_binary(cnf, bytes);
        let b = FlatProof::from_binary_model(cnf, bytes, chunk);
        match (a, b) {
            (Ok(a), Ok(b)) => assert_flat_eq(&a, &b, what),
            (Err(a), Err(b)) => assert_eq!(a, b, "{what}: error"),
            (a, b) => panic!(
                "{what}: sequential {:?} vs model {:?}",
                a.map(|_| ()),
                b.map(|_| ())
            ),
        }
    }

    const CHUNKS: [usize; 8] = [1, 3, 5, 7, 9, 13, 64, 1000];

    #[test]
    fn model_matches_host_slices_on_valid_proofs() {
        for with_empty in [false, true] {
            let (cnf, bytes) = synthetic_proof(with_empty);
            for &chunk in &CHUNKS {
                let what = format!("valid c={chunk} e={with_empty}");
                let (clean, dirty) = assert_slices_match(&bytes, chunk, &what);
                // A mis-synchronised start can parse garbage into an error, so
                // some dirt is legitimate — but on a valid proof the first
                // slice is never dirty, and most are clean.
                assert!(clean > dirty, "{what}: {clean} clean, {dirty} dirty");
                assert!(decode_slice_model(&bytes, 0, chunk.min(bytes.len())).is_some());
                assert_same_result_model(&cnf, &bytes, chunk, &what);
            }
        }
    }

    #[test]
    fn model_matches_host_slices_on_every_truncation() {
        let (cnf, bytes) = synthetic_proof(true);
        for cut in 0..bytes.len() {
            for &chunk in &[11usize, 64] {
                let what = format!("truncate at {cut} c={chunk}");
                assert_slices_match(&bytes[..cut], chunk, &what);
                assert_same_result_model(&cnf, &bytes[..cut], chunk, &what);
            }
        }
    }

    fn bit_flips(bits: &[u32], chunks: &[usize]) {
        let (cnf, bytes) = synthetic_proof(false);
        let mut m = bytes.clone();
        for i in 0..bytes.len() {
            for &bit in bits {
                m[i] ^= 1 << bit;
                for &chunk in chunks {
                    let what = format!("flip byte {i} bit {bit} c={chunk}");
                    assert_slices_match(&m, chunk, &what);
                    assert_same_result_model(&cnf, &m, chunk, &what);
                }
                m[i] ^= 1 << bit;
            }
        }
    }

    /// The default suite flips the two bits that change the most — bit 7
    /// (tokenisation) and bit 0 (sign / zero) — at every byte. The model
    /// simulates 32 lanes on one thread, so the full flip battery is ≈ 100×
    /// the host decoder's and lives in the ignored test below
    /// (`cargo test --release -- --ignored`); it passed on 2026-11-03.
    #[test]
    fn model_matches_host_slices_on_bit_flips() {
        bit_flips(&[7, 0], &[13]);
    }

    #[test]
    #[ignore = "exhaustive: every bit of every byte at two slice sizes; run with --release -- --ignored"]
    fn model_matches_host_slices_on_every_bit_flip() {
        bit_flips(&[0, 1, 2, 3, 4, 5, 6, 7], &[7, 13]);
    }

    /// Long varints (up to 10 bytes) and one of 11 bytes: the model decodes
    /// the former like the host and calls the latter dirty.
    #[test]
    fn long_varints_decode_and_too_long_is_dirty() {
        // One lemma: id 5, literal with a 10-byte (non-minimal) encoding, hint 1.
        let mut bytes = vec![b'a', 10u8 /* id 5 → 2*5 */];
        let lit: u64 = 2 * 7; // +7
        for k in 0..9 {
            bytes.push((((lit >> (7 * k)) & 0x7f) as u8) | 0x80);
        }
        bytes.push(((lit >> 63) & 0x7f) as u8); // 10th byte, terminator
        bytes.extend_from_slice(&[0, 2, 0]); // end lits; hint 1; end hints
        let cnf = vec![vec![1, 2], vec![-1], vec![-2], vec![7, 8]];
        let host = host_record(&bytes, 0, bytes.len());
        assert!(host.error.is_none(), "{host:?}");
        assert_eq!(decode_slice_model(&bytes, 0, bytes.len()), Some(host));
        assert_same_result_model(&cnf, &bytes, 4, "10-byte literal");
        // 11 bytes: dirty, and the host errors.
        let mut long = vec![b'a', 10u8];
        long.extend(std::iter::repeat(0x80u8).take(10));
        long.push(0);
        long.extend_from_slice(&[0, 2, 0]);
        assert!(decode_slice_model(&long, 0, long.len()).is_none());
        assert!(host_record(&long, 0, long.len()).error.is_some());
        assert_same_result_model(&cnf, &long, 4, "11-byte literal");
    }

    /// A RAT hint anywhere in a long hint list — a step that spans several
    /// windows, with the negative hint in a window that neither starts nor
    /// ends the step — must reach the record as `rat = true` (the host turns
    /// it into an `Unsupported` verdict in phase 3a). Found as a gap by a
    /// surviving mutant (the rat carry dropped) on 2026-11-03.
    #[test]
    fn rat_hint_anywhere_in_a_step_spanning_windows() {
        let cnf: Vec<Vec<i32>> = (1..=200).map(|v| vec![v]).collect();
        for k in [0usize, 1, 20, 40, 60, 79] {
            let hints: Vec<i64> = (0..80)
                .map(|h| {
                    if h == k {
                        -((h + 1) as i64)
                    } else {
                        (h + 1) as i64
                    }
                })
                .collect();
            let p = crate::lrat::Proof {
                steps: vec![crate::lrat::Step::Add {
                    id: 201,
                    lits: vec![7, 8, 9],
                    hints,
                }],
            };
            let bytes = crate::lrat::write_binary(&p);
            assert!(bytes.len() > 2 * WARP, "the step must span three windows");
            for &chunk in &[9usize, 40, 4096] {
                let what = format!("rat at {k} c={chunk}");
                let (clean, _) = assert_slices_match(&bytes, chunk, &what);
                assert!(clean >= 1, "{what}");
                let host = host_record(&bytes, 0, bytes.len());
                assert!(host.steps[0].rat, "{what}: host");
                assert_same_result_model(&cnf, &bytes, chunk, &what);
            }
        }
    }

    /// A token or a step straddling a 32-byte window: the carry is exercised
    /// by a proof whose literals push steps across window boundaries.
    #[test]
    fn steps_and_tokens_straddle_windows() {
        // 40 lemmas of 7 two-byte literals each: every window boundary falls
        // inside some step, many inside a two-byte varint.
        let cnf: Vec<Vec<i32>> = (1..=300).map(|v| vec![v]).collect();
        let mut p = crate::lrat::Proof { steps: Vec::new() };
        for i in 0..40u64 {
            p.steps.push(crate::lrat::Step::Add {
                id: 301 + i,
                lits: (0..7).map(|k| 100 + (i as i32 * 7 + k) % 150).collect(),
                hints: vec![1, 2],
            });
        }
        let bytes = crate::lrat::write_binary(&p);
        assert!(bytes.len() > 400);
        for &chunk in &[33usize, 64, 100, 4096] {
            let what = format!("straddle c={chunk}");
            let (clean, _) = assert_slices_match(&bytes, chunk, &what);
            assert!(clean >= 1);
            assert_same_result_model(&cnf, &bytes, chunk, &what);
        }
    }
}
