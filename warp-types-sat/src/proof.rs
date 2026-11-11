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
//! # Why every logged lemma is RUP (the invariant, stated 2026-11-01)
//!
//! A clause is RUP (reverse unit propagation) w.r.t. a clause set if asserting
//! its negation and propagating to fixpoint yields a conflict. For the
//! pure-SAT core:
//!
//! 1. Each learned clause is the 1-UIP resolvent of the conflict clause and
//!    trail reasons, every one of which is present in the log at that moment:
//!    original clauses are never deleted, and learned reasons are locked (3).
//! 2. Resolution keeps level-0 literals in the clause; minimization then
//!    removes a literal only when every path through its reason chain ends in
//!    a clause literal or a level-0 literal. Dropping a level-0 literal is
//!    RUP only if the checker can re-derive that level-0 unit, so **the
//!    level-0 reasons must still be in the log**. The `abstract_levels`
//!    bit-filter in `analyze.rs` is an early abort only; aliasing of levels
//!    makes it less effective, never more permissive.
//! 3. `reduce_learned` runs only after a restart has backtracked to level 0,
//!    so the only reasons alive during a reduce are level-0 reasons, and
//!    `build_locked_set` locks every propagation reason on the trail. Learned
//!    units have LBD 1, and clauses with LBD ≤ 2 are never candidates. Hence no
//!    clause a later lemma depends on through the trail is ever deleted, and
//!    no single-literal `d` line is ever written (which also keeps the proofs
//!    independent of `drat-trim`'s decision to ignore unit deletions).
//! 4. The initial-BCP conflict and a conflict at level 0 emit the empty clause
//!    directly: unit propagation over the logged set reproduces the solver's
//!    level-0 trail, so `0` is RUP.
//! 5. Compaction remaps clause references, not literals; deletions are logged
//!    from the tombstoned clause before `compact` frees it.
//!
//! Each point has a falsifier in [`fault`] / the tests: the lock emptied, a
//! literal flipped, the empty clause suppressed, a phantom deletion, a wrong
//! deletion. The strict in-crate checker must reject every one; which of them
//! `drat-trim` also rejects is recorded in `research/drat-certification.md`,
//! because a checker that accepts a corrupted proof is a fact about the
//! checker's semantics, not a pass.
//!
//! [`check_drat`] is a forward RUP checker used by the tests: it is an
//! independent reimplementation of the check (unit propagation over the
//! clause set, not the solver's watched BCP), but it lives in this crate; an
//! external checker (`drat-trim`, `cake_lpr`) remains the real test.

use crate::literal::Lit;
use std::io::{self, Write};

/// Test-only fault injection for the proof log. Every function here is a
/// no-op (and compiles to nothing) outside `cfg(test)`; under test, a
/// thread-local selects at most one fault, so the solver on this thread writes
/// a deliberately corrupted log while behaving normally otherwise. Used to
/// watch each checker fail (see the module docs).
pub mod fault {
    use super::ProofLog;
    use crate::bcp::{CRef, ClauseDb};
    use crate::literal::Lit;

    /// The corruptions. See `research/drat-certification.md` §3.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    #[repr(u8)]
    pub enum Fault {
        None = 0,
        /// `build_locked_set` returns the empty set: reasons may be deleted.
        EmptyLock = 1,
        /// The 10th learned clause is logged with its first literal negated.
        FlipLiteral = 2,
        /// `proof.empty()` is suppressed: the log never derives `0`.
        OmitFinalEmpty = 3,
        /// At the first reduce, a `d` line for a clause that was never added.
        PhantomDelete = 4,
        /// At the first reduce, `d` lines for the clauses that were *kept*
        /// (still in use) instead of the ones deleted.
        WrongDelete = 5,
        /// At every reduce, `d` lines for the LOCKED clauses (trail reasons)
        /// as if they had been deleted, while the solver keeps them. Isolates
        /// the proof-side question from solver integrity: if a level-0
        /// reason's deletion is logged, does the checker notice?
        LogLockedDelete = 6,
        /// At every reduce, the locked LEARNED clauses are really deleted
        /// (tombstoned regardless of lock and LBD). The solver's compaction
        /// remap must then refuse with a panic: this watches the integrity
        /// guard fail on purpose.
        DeleteLocked = 7,
        /// LRAT: the first hint of the 10th learned clause's chain is dropped
        /// (the chain no longer propagates to a conflict).
        DropHint = 8,
        /// LRAT: the 10th learned clause's chain is written in reverse (the
        /// last reason is not unit first).
        ReverseHints = 9,
        /// LRAT: the 10th learned clause is written with id + 1 (later hints
        /// name an id that does not exist / ids go out of order).
        WrongId = 10,
    }

    #[cfg(test)]
    thread_local! {
        static FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
        static LEARNED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static REDUCES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static LOCKED_LOGGED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// Arm `f` for the current thread; the returned guard disarms it.
    #[cfg(test)]
    pub fn arm(f: Fault) -> Armed {
        FAULT.with(|c| c.set(f as u8));
        LEARNED.with(|c| c.set(0));
        REDUCES.with(|c| c.set(0));
        LOCKED_LOGGED.with(|c| c.set(0));
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

    /// Log a learned clause, applying `FlipLiteral` to the 10th one.
    #[inline]
    pub fn log_learned(proof: &mut dyn ProofLog, learned: &[Lit]) {
        #[cfg(test)]
        {
            let n = LEARNED.with(|c| {
                let v = c.get() + 1;
                c.set(v);
                v
            });
            if active(Fault::FlipLiteral) && n == 10 && !learned.is_empty() {
                let mut flipped = learned.to_vec();
                flipped[0] = flipped[0].complement();
                proof.add(&flipped);
                return;
            }
        }
        proof.add(learned);
    }

    /// LRAT twin of [`log_learned`]: a learned clause with its RUP chain.
    /// Applies `FlipLiteral`, `DropHint`, `ReverseHints` and `WrongId` to the
    /// 10th learned clause. Returns the id the log assigned.
    #[inline]
    pub fn log_learned_hints(proof: &mut dyn ProofLog, learned: &[Lit], hints: &[u64]) -> u64 {
        #[cfg(test)]
        {
            let n = LEARNED.with(|c| {
                let v = c.get() + 1;
                c.set(v);
                v
            });
            if n == 10 {
                if active(Fault::FlipLiteral) && !learned.is_empty() {
                    let mut flipped = learned.to_vec();
                    flipped[0] = flipped[0].complement();
                    return proof.add_with_hints(&flipped, hints);
                }
                if active(Fault::DropHint) && !hints.is_empty() {
                    return proof.add_with_hints(learned, &hints[1..]);
                }
                if active(Fault::ReverseHints) {
                    let mut rev = hints.to_vec();
                    rev.reverse();
                    return proof.add_with_hints(learned, &rev);
                }
                if active(Fault::WrongId) {
                    return proof.add_with_wrong_id(learned, hints);
                }
            }
        }
        proof.add_with_hints(learned, hints)
    }

    /// Hook at a reduce, before the genuine `d` lines are written.
    #[inline]
    pub fn on_reduce(
        db: &ClauseDb,
        locked: &std::collections::HashSet<CRef>,
        deleted: &[CRef],
        proof: &mut dyn ProofLog,
    ) {
        #[cfg(test)]
        {
            let n = REDUCES.with(|c| {
                let v = c.get() + 1;
                c.set(v);
                v
            });
            if active(Fault::LogLockedDelete) {
                let mut locked: Vec<CRef> = locked.iter().copied().collect();
                locked.sort_unstable();
                for &c in &locked {
                    proof.delete(db.clause(c).literals);
                }
                LOCKED_LOGGED.with(|x| x.set(x.get() + locked.len() as u64));
            }
            if n != 1 {
                return;
            }
            if active(Fault::PhantomDelete) {
                // A clause over the two highest variables that no solve of a
                // test instance adds: 50 literals wide, all positive.
                let nv = db.max_variable();
                let phantom: Vec<Lit> = (0..nv.min(49)).map(Lit::pos).collect();
                proof.delete(&phantom);
            }
            if active(Fault::WrongDelete) {
                // Delete (in the log only) every live learned clause that was
                // NOT deleted — the ones the solver keeps using.
                let kept: Vec<CRef> = db
                    .iter_crefs()
                    .filter(|c| !deleted.contains(c) && !db.clause(*c).literals.is_empty())
                    .collect();
                let start = kept.len().saturating_sub(deleted.len().max(8));
                for &c in &kept[start..] {
                    proof.delete(db.clause(c).literals);
                }
            }
        }
        #[cfg(not(test))]
        {
            let _ = (db, locked, deleted, proof);
        }
    }

    /// Number of locked-clause deletions `LogLockedDelete` has written on this
    /// thread since `arm`. Zero means the fault was vacuous on that instance.
    #[cfg(test)]
    pub fn locked_logged() -> u64 {
        LOCKED_LOGGED.with(|c| c.get())
    }
}

/// Receiver for proof steps. The solver core calls these as it runs.
///
/// The three defaulted `*_hints` / `*_ids` methods are the LRAT face: a log
/// that answers `wants_hints()` gets clause ids and RUP chains from the
/// solver (see `lrat_emit`); a DRAT log never sees them.
pub trait ProofLog {
    /// A clause was added (learned, or a theory lemma).
    fn add(&mut self, lits: &[Lit]);
    /// A clause was removed from the database.
    fn delete(&mut self, lits: &[Lit]);
    /// The solver concluded UNSAT: the empty clause is derivable.
    fn empty(&mut self) {
        if fault::active(fault::Fault::OmitFinalEmpty) {
            return;
        }
        self.add(&[]);
    }
    /// Does this log want clause ids and hint chains (LRAT)?
    fn wants_hints(&self) -> bool {
        false
    }
    /// A clause added with the ids of the clauses whose unit propagation,
    /// in this order, refutes its negation. Returns the id assigned to the
    /// new clause (0 for logs that do not track ids).
    fn add_with_hints(&mut self, lits: &[Lit], _hints: &[u64]) -> u64 {
        self.add(lits);
        0
    }
    /// Test hook for `fault::Fault::WrongId`; DRAT logs ignore it.
    #[doc(hidden)]
    fn add_with_wrong_id(&mut self, lits: &[Lit], hints: &[u64]) -> u64 {
        self.add_with_hints(lits, hints)
    }
    /// Clauses removed, by id (LRAT); DRAT logs use [`ProofLog::delete`].
    fn delete_ids(&mut self, _ids: &[u64]) {}
    /// UNSAT, with the chain refuting the empty clause.
    fn empty_with_hints(&mut self, hints: &[u64]) {
        if fault::active(fault::Fault::OmitFinalEmpty) {
            return;
        }
        self.add_with_hints(&[], hints);
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

/// Append a decimal integer without allocating (the writers format tens of
/// millions of numbers; `to_string()` per number was ~1 s per 250 MB).
#[inline]
fn push_int(s: &mut String, v: i64) {
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    let neg = v < 0;
    let mut n = v.unsigned_abs();
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    if neg {
        i -= 1;
        buf[i] = b'-';
    }
    // SAFETY-free: the bytes are ASCII digits and '-'.
    s.push_str(std::str::from_utf8(&buf[i..]).expect("ascii"));
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

/// The literal code of the binary DRAT / LRAT layouts: `2·|v| + [v < 0]` for
/// the DIMACS integer `v` (drat-trim's `read_lit`, cake_lpr's reader).
#[inline]
pub fn binary_lit(lit: Lit) -> u64 {
    let v = dimacs_int(lit);
    2 * v.unsigned_abs() + u64::from(v < 0)
}

/// 7-bit little-endian varint (continuation bit 0x80), the number encoding of
/// binary DRAT and binary LRAT.
pub(crate) fn push_varint(buf: &mut Vec<u8>, mut x: u64) {
    loop {
        let b = (x & 0x7f) as u8;
        x >>= 7;
        if x == 0 {
            buf.push(b);
            return;
        }
        buf.push(b | 0x80);
    }
}

/// Writes DRAT — text, or the binary layout drat-trim reads (`-i`, or
/// detected from the first bytes): a step is the byte `a` (97) or `d` (100),
/// then each literal as a varint of [`binary_lit`], then a 0 byte. I/O errors
/// are kept in the writer and returned by [`DratWriter::finish`]; the solver
/// never sees them.
pub struct DratWriter<W: Write> {
    out: W,
    err: Option<io::Error>,
    binary: bool,
    /// Reused step buffer (binary).
    bbuf: Vec<u8>,
    /// Steps written: (additions, deletions). Additions count the empty clause.
    pub additions: u64,
    pub deletions: u64,
}

impl<W: Write> DratWriter<W> {
    /// Text DRAT.
    pub fn new(out: W) -> Self {
        Self::with_format(out, false)
    }

    /// Binary DRAT.
    pub fn new_binary(out: W) -> Self {
        Self::with_format(out, true)
    }

    /// Text (`binary == false`) or binary DRAT.
    pub fn with_format(out: W, binary: bool) -> Self {
        DratWriter {
            out,
            err: None,
            binary,
            bbuf: Vec::new(),
            additions: 0,
            deletions: 0,
        }
    }

    fn step(&mut self, delete: bool, lits: &[Lit]) {
        if self.err.is_some() {
            return;
        }
        if self.binary {
            let mut b = std::mem::take(&mut self.bbuf);
            b.clear();
            b.push(if delete { b'd' } else { b'a' });
            for &l in lits {
                push_varint(&mut b, binary_lit(l));
            }
            b.push(0);
            if let Err(e) = self.out.write_all(&b) {
                self.err = Some(e);
            }
            self.bbuf = b;
            return;
        }
        let prefix = if delete { "d " } else { "" };
        let mut s = String::with_capacity(prefix.len() + lits.len() * 6 + 2);
        s.push_str(prefix);
        for &l in lits {
            push_int(&mut s, dimacs_int(l));
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
        self.step(false, lits);
    }
    fn delete(&mut self, lits: &[Lit]) {
        self.deletions += 1;
        self.step(true, lits);
    }
}

/// Writes LRAT (text, or binary in the layout `lrat::parse_binary` reads) to
/// any `Write`. Ids: originals `1..=n_original` in file order (the DIMACS
/// loader keeps every clause, empty ones included, in order), then every
/// added clause in emission order. I/O errors are kept and returned by
/// [`LratWriter::finish`]. A clause added WITHOUT hints (a theory lemma, or a
/// DRAT-style `add`) is written with an empty chain and counted in
/// `unjustified`: no checker will accept it, and this writer does not pretend
/// otherwise. See `research/lrat-from-solver.md`.
pub struct LratWriter<W: Write> {
    out: W,
    err: Option<io::Error>,
    binary: bool,
    next_id: u64,
    last_id: u64,
    /// Reused line buffers (text / binary).
    sbuf: String,
    bbuf: Vec<u8>,
    /// Steps written: additions (the empty clause included), deletions (ids),
    /// additions without a chain.
    pub additions: u64,
    pub deletions: u64,
    pub unjustified: u64,
    /// Hints written in total (the proof's size driver).
    pub hints: u64,
}

impl<W: Write> LratWriter<W> {
    pub fn new(out: W, n_original: usize, binary: bool) -> Self {
        LratWriter {
            out,
            err: None,
            binary,
            next_id: n_original as u64 + 1,
            last_id: n_original as u64,
            sbuf: String::new(),
            bbuf: Vec::new(),
            additions: 0,
            deletions: 0,
            unjustified: 0,
            hints: 0,
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        if self.err.is_some() {
            return;
        }
        if let Err(e) = self.out.write_all(bytes) {
            self.err = Some(e);
        }
    }

    fn add_line(&mut self, id: u64, lits: &[Lit], hints: &[u64]) {
        self.additions += 1;
        self.hints += hints.len() as u64;
        if hints.is_empty() {
            self.unjustified += 1;
        }
        if self.binary {
            let mut b = std::mem::take(&mut self.bbuf);
            b.clear();
            b.push(b'a');
            push_varint(&mut b, 2 * id);
            for &l in lits {
                push_varint(&mut b, binary_lit(l));
            }
            b.push(0);
            for &h in hints {
                push_varint(&mut b, 2 * h);
            }
            b.push(0);
            self.write(&b);
            self.bbuf = b;
        } else {
            let mut s = std::mem::take(&mut self.sbuf);
            s.clear();
            push_int(&mut s, id as i64);
            for &l in lits {
                s.push(' ');
                push_int(&mut s, dimacs_int(l));
            }
            s.push_str(" 0");
            for &h in hints {
                s.push(' ');
                push_int(&mut s, h as i64);
            }
            s.push_str(" 0\n");
            self.write(s.as_bytes());
            self.sbuf = s;
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

impl<W: Write> ProofLog for LratWriter<W> {
    fn add(&mut self, lits: &[Lit]) {
        self.add_with_hints(lits, &[]);
    }
    fn delete(&mut self, _lits: &[Lit]) {
        // LRAT deletes by id; the solver uses `delete_ids` when `wants_hints`.
        // A literal-only delete (a DRAT fault hook) has no id here: ignored.
    }
    fn wants_hints(&self) -> bool {
        true
    }
    fn add_with_hints(&mut self, lits: &[Lit], hints: &[u64]) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.last_id = id;
        self.add_line(id, lits, hints);
        id
    }
    fn add_with_wrong_id(&mut self, lits: &[Lit], hints: &[u64]) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.last_id = id;
        self.add_line(id + 1, lits, hints); // written wrong, tracked right
        id
    }
    fn delete_ids(&mut self, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        self.deletions += ids.len() as u64;
        if self.binary {
            let mut b = Vec::with_capacity(2 + ids.len() * 4);
            b.push(b'd');
            for &i in ids {
                push_varint(&mut b, 2 * i);
            }
            b.push(0);
            self.write(&b);
        } else {
            // The step id on a `d` line is ignored by every checker; the last
            // added id is the convention.
            let mut s = std::mem::take(&mut self.sbuf);
            s.clear();
            push_int(&mut s, self.last_id as i64);
            s.push_str(" d");
            for &i in ids {
                s.push(' ');
                push_int(&mut s, i as i64);
            }
            s.push_str(" 0\n");
            self.write(s.as_bytes());
            self.sbuf = s;
        }
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

/// Parse binary DRAT (drat-trim's layout, see [`DratWriter`]). Every step
/// must start with `a` (97) or `d` (100) and end with a 0 byte; the literal
/// code 1 ("−0") names no variable and is rejected, as the binary LRAT
/// decoders reject it (drat-trim's `read_lit` would read it as a terminator).
pub fn parse_drat_binary(bytes: &[u8]) -> Result<Vec<DratStep>, String> {
    let mut steps = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let at = pos;
        let delete = match bytes[pos] {
            b'a' => false,
            b'd' => true,
            b => {
                return Err(format!(
                    "byte {at}: step prefix {b}, expected 97 (a) or 100 (d)"
                ))
            }
        };
        pos += 1;
        let mut lits = Vec::new();
        loop {
            let mut code: u64 = 0;
            let mut shift = 0u32;
            loop {
                let Some(&b) = bytes.get(pos) else {
                    return Err(format!(
                        "byte {pos}: proof ends inside the step at byte {at}"
                    ));
                };
                pos += 1;
                if shift >= 63 {
                    return Err(format!(
                        "byte {pos}: varint too long in the step at byte {at}"
                    ));
                }
                code |= u64::from(b & 0x7f) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            if code == 0 {
                break;
            }
            if code == 1 {
                return Err(format!(
                    "byte {pos}: literal code 1 (\"-0\") in the step at byte {at}"
                ));
            }
            let v = i64::try_from(code >> 1).map_err(|_| {
                format!("byte {pos}: variable out of range in the step at byte {at}")
            })?;
            lits.push(if code & 1 == 1 { -v } else { v });
        }
        steps.push(if delete {
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
    use crate::solver::{
        solve_watched, solve_watched_drat, solve_watched_drat_format, solve_watched_drat_with,
        SolveResult,
    };

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
    fn writer_format_binary() {
        // drat-trim's layout: `a`/`d`, varint(2·|v| + [v < 0]) per literal, 0.
        let mut w = DratWriter::new_binary(Vec::new());
        w.add(&[Lit::pos(0), Lit::neg(2)]); // 1 -3
        w.delete(&[Lit::neg(1)]); // d -2
        w.add(&[Lit::pos(99)]); // 100 → code 200 → two varint bytes
        w.empty();
        let out = w.finish().unwrap();
        assert_eq!(
            out,
            vec![b'a', 2, 7, 0, b'd', 5, 0, b'a', 0xC8, 0x01, 0, b'a', 0]
        );
        assert_eq!(
            parse_drat_binary(&out).unwrap(),
            vec![
                DratStep::Add(vec![1, -3]),
                DratStep::Delete(vec![-2]),
                DratStep::Add(vec![100]),
                DratStep::Add(vec![]),
            ]
        );
        assert_eq!(binary_lit(Lit::pos(0)), 2);
        assert_eq!(binary_lit(Lit::neg(0)), 3);
        assert_eq!(binary_lit(Lit::neg(41)), 85);
    }

    #[test]
    fn parse_drat_binary_rejects_bad_bytes() {
        let e = parse_drat_binary(&[b'b', 2, 0]).unwrap_err();
        assert!(e.contains("prefix 98"), "{e}");
        let e = parse_drat_binary(&[b'a', 2]).unwrap_err();
        assert!(e.contains("ends inside"), "{e}");
        let e = parse_drat_binary(&[b'a', 2, 0, b'd']).unwrap_err();
        assert!(e.contains("ends inside"), "{e}");
        let e = parse_drat_binary(&[b'a', 1, 0]).unwrap_err();
        assert!(e.contains("literal code 1"), "{e}");
        let e = parse_drat_binary(&[
            b'a', 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0,
        ])
        .unwrap_err();
        assert!(e.contains("too long"), "{e}");
        assert_eq!(parse_drat_binary(&[]).unwrap(), vec![]);
    }

    /// The same instance solved twice, once per layout; the search is
    /// deterministic, so the two proofs must be the same steps.
    fn drat_both_layouts(text: &str, interval: u64) -> (SolveResult, Vec<u8>, String) {
        let inst = parse_dimacs_str(text).unwrap();
        let mut bin: Vec<u8> = Vec::new();
        let r =
            solve_watched_drat_format(inst.db, inst.num_vars, &mut bin, interval, true).unwrap();
        let inst = parse_dimacs_str(text).unwrap();
        let mut txt: Vec<u8> = Vec::new();
        let r2 =
            solve_watched_drat_format(inst.db, inst.num_vars, &mut txt, interval, false).unwrap();
        assert_eq!(r, r2);
        (r, bin, String::from_utf8(txt).unwrap())
    }

    #[test]
    fn binary_and_text_drat_are_the_same_proof() {
        // PHP(3–7) at interval 50 (deletions from PHP(6) up), then the random
        // UNSAT set: step-for-step equal, strict checker accepts, binary smaller.
        let mut cases: Vec<(String, String)> = (3..=7)
            .map(|n| (format!("PHP({n})"), pigeonhole_dimacs(n)))
            .collect();
        for seed in 0..40u64 {
            let db = generate_3sat_phase_transition(60, seed);
            let text = crate::bench::db_to_dimacs(&db, 60);
            let inst = parse_dimacs_str(&text).unwrap();
            if solve_watched(inst.db, inst.num_vars) == SolveResult::Unsat {
                cases.push((format!("r3sat seed {seed}"), text));
            }
        }
        assert!(cases.len() >= 10, "{} cases", cases.len());
        let mut with_deletions = 0;
        for (name, text) in &cases {
            let (r, bin, txt) = drat_both_layouts(text, 50);
            assert_eq!(r, SolveResult::Unsat, "{name}");
            let b = parse_drat_binary(&bin).unwrap_or_else(|e| panic!("{name}: {e}"));
            let t = parse_drat(&txt).unwrap();
            assert_eq!(b, t, "{name}: binary and text proofs differ");
            assert_eq!(check_drat(&cnf_ints(text), &b), Ok(()), "{name}");
            assert!(
                bin.len() < txt.len(),
                "{name}: binary {} bytes, text {}",
                bin.len(),
                txt.len()
            );
            if b.iter().any(|s| matches!(s, DratStep::Delete(_))) {
                with_deletions += 1;
            }
            eprintln!(
                "{name}: {} steps, binary {} B / text {} B = {:.2}",
                b.len(),
                bin.len(),
                txt.len(),
                bin.len() as f64 / txt.len() as f64
            );
        }
        assert!(
            with_deletions >= 2,
            "only {with_deletions} proofs carry deletions"
        );
    }

    #[test]
    fn binary_proofs_verify_with_drat_trim() {
        // Autodetected (no flag) and forced (`-i`); PHP(6) at interval 50 has
        // deletions. Skips only when no drat-trim is configured.
        let Some(_) = drat_trim_bin() else {
            eprintln!("drat-trim not configured: skipped");
            return;
        };
        for (n, interval) in [(6u32, 50u64), (7, 2000)] {
            let text = pigeonhole_dimacs(n);
            let (r, bin, _) = drat_both_layouts(&text, interval);
            assert_eq!(r, SolveResult::Unsat);
            let auto = drat_trim_output(&format!("bin-php{n}"), &text, &bin, &[]).unwrap();
            assert!(
                auto.contains("turning on binary mode"),
                "PHP({n}): drat-trim did not detect the binary layout:\n{auto}"
            );
            assert!(
                auto.contains("s VERIFIED") && !auto.contains("s NOT VERIFIED"),
                "PHP({n}) autodetect:\n{auto}"
            );
            let forced = drat_trim_output(&format!("bin-php{n}-i"), &text, &bin, &["-i"]).unwrap();
            assert!(
                forced.contains("s VERIFIED") && !forced.contains("s NOT VERIFIED"),
                "PHP({n}) -i:\n{forced}"
            );
        }
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

    /// Locate drat-trim: `DRAT_TRIM` (must then work — no silent skip) or PATH.
    fn drat_trim_bin() -> Option<String> {
        std::env::var("DRAT_TRIM").ok().or_else(|| {
            std::env::var_os("PATH").and_then(|p| {
                std::env::split_paths(&p)
                    .map(|d| d.join("drat-trim"))
                    .find(|f| f.is_file())
                    .map(|f| f.to_string_lossy().into_owned())
            })
        })
    }

    /// Run drat-trim on proof BYTES with extra args; returns its whole stdout,
    /// or `None` when no checker is configured.
    fn drat_trim_output(tag: &str, cnf: &str, proof: &[u8], extra: &[&str]) -> Option<String> {
        let bin = drat_trim_bin()?;
        let dir =
            std::env::temp_dir().join(format!("warp-types-sat-drat-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let c = dir.join("i.cnf");
        let d = dir.join("i.drat");
        std::fs::write(&c, cnf).unwrap();
        std::fs::write(&d, proof).unwrap();
        let out = std::process::Command::new(&bin)
            .arg(&c)
            .arg(&d)
            .args(extra)
            .output()
            .expect("DRAT_TRIM is set or drat-trim is on PATH but it did not run");
        Some(String::from_utf8_lossy(&out.stdout).replace('\r', "\n"))
    }

    /// Run drat-trim with extra args; returns its `s ...` verdict line, or
    /// `None` when no checker is configured.
    fn drat_trim_verdict(tag: &str, cnf: &str, proof: &str, extra: &[&str]) -> Option<String> {
        let bin = drat_trim_bin()?;
        let dir =
            std::env::temp_dir().join(format!("warp-types-sat-drat-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let c = dir.join("i.cnf");
        let d = dir.join("i.drat");
        std::fs::write(&c, cnf).unwrap();
        std::fs::write(&d, proof).unwrap();
        let out = std::process::Command::new(&bin)
            .arg(&c)
            .arg(&d)
            .args(extra)
            .output()
            .expect("DRAT_TRIM is set or drat-trim is on PATH but it did not run");
        let text = String::from_utf8_lossy(&out.stdout).replace('\r', "\n");
        Some(if text.contains("s NOT VERIFIED") {
            "s NOT VERIFIED".to_string()
        } else if text.contains("s VERIFIED") {
            "s VERIFIED".to_string()
        } else {
            format!("(no verdict; exit {:?})", out.status.code())
        })
    }

    #[test]
    fn reduce_interval_knob_brings_deletions_into_php6() {
        // Default interval: PHP(6) never reaches a reduce, so no `d` lines.
        let text = pigeonhole_dimacs(6);
        let (r, proof) = drat_for(&text);
        assert_eq!(r, SolveResult::Unsat);
        assert!(
            !proof.lines().any(|l| l.starts_with("d ")),
            "default has no deletions"
        );
        // Interval 50: deletions happen and the proof still checks.
        let inst = parse_dimacs_str(&text).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        let r = solve_watched_drat_with(inst.db, inst.num_vars, &mut buf, 50).unwrap();
        assert_eq!(r, SolveResult::Unsat);
        let proof = String::from_utf8(buf).unwrap();
        let dels = proof.lines().filter(|l| l.starts_with("d ")).count();
        assert!(dels > 0, "interval 50 should delete on PHP(6)");
        assert!(
            !proof
                .lines()
                .any(|l| { l.starts_with("d ") && l.split_whitespace().count() == 3 }),
            "no single-literal deletion may ever be written"
        );
        let steps = parse_drat(&proof).unwrap();
        assert_eq!(check_drat(&cnf_ints(&text), &steps), Ok(()));
        eprintln!("reduce_interval=50 on PHP(6): {dels} deletions; proof checks");
    }

    /// Each fault must be rejected by the strict checker; drat-trim's verdicts
    /// are recorded (not asserted) because some faults are invisible to its
    /// semantics by design. See research/drat-certification.md §3.
    fn run_fault_on(f: fault::Fault, text: &str) -> (Option<SolveResult>, String, u64) {
        let inst = parse_dimacs_str(text).unwrap();
        let _armed = fault::arm(f);
        let mut buf: Vec<u8> = Vec::new();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve_watched_drat_with(inst.db, inst.num_vars, &mut buf, 50).unwrap()
        }))
        .ok();
        let logged = fault::locked_logged();
        (r, String::from_utf8(buf).unwrap(), logged)
    }

    fn run_fault(f: fault::Fault) -> (Option<SolveResult>, String) {
        let (r, p, _) = run_fault_on(f, &pigeonhole_dimacs(6));
        (r, p)
    }

    fn strict_verdict_on(text: &str, proof: &str) -> Result<(), String> {
        let steps = parse_drat(proof).map_err(|e| format!("parse: {e}"))?;
        check_drat(&cnf_ints(text), &steps)
    }

    fn strict_verdict(proof: &str) -> Result<(), String> {
        strict_verdict_on(&pigeonhole_dimacs(6), proof)
    }

    fn record(f: fault::Fault, r: &Option<SolveResult>, proof: &str, strict: &Result<(), String>) {
        record_on("PHP(6)", &pigeonhole_dimacs(6), f, r, proof, strict);
    }

    fn record_on(
        name: &str,
        text: &str,
        f: fault::Fault,
        r: &Option<SolveResult>,
        proof: &str,
        strict: &Result<(), String>,
    ) {
        let dt = drat_trim_verdict(&format!("{f:?}-{name}"), text, proof, &[]);
        let dtf = drat_trim_verdict(&format!("{f:?}-{name}-f"), text, proof, &["-f"]);
        eprintln!(
            "FAULT {f:?} on {name}: solver={} strict={} drat-trim={} drat-trim-f={}",
            match r {
                Some(x) => format!("{x:?}").chars().take(5).collect::<String>(),
                None => "PANIC".into(),
            },
            match strict {
                Ok(()) => "ACCEPT".to_string(),
                Err(e) => format!("REJECT({e})"),
            },
            dt.as_deref().unwrap_or("n/a"),
            dtf.as_deref().unwrap_or("n/a"),
        );
    }

    #[test]
    fn fault_none_is_clean() {
        let (r, proof) = run_fault(fault::Fault::None);
        assert_eq!(r, Some(SolveResult::Unsat));
        assert_eq!(strict_verdict(&proof), Ok(()));
    }

    #[test]
    fn fault_flip_literal_is_rejected() {
        let (r, proof) = run_fault(fault::Fault::FlipLiteral);
        let v = strict_verdict(&proof);
        record(fault::Fault::FlipLiteral, &r, &proof, &v);
        assert!(v.as_ref().unwrap_err().contains("not RUP"), "{v:?}");
    }

    #[test]
    fn fault_omit_final_empty_is_rejected() {
        let (r, proof) = run_fault(fault::Fault::OmitFinalEmpty);
        let v = strict_verdict(&proof);
        record(fault::Fault::OmitFinalEmpty, &r, &proof, &v);
        assert_eq!(r, Some(SolveResult::Unsat));
        assert!(v.as_ref().unwrap_err().contains("empty clause"), "{v:?}");
    }

    #[test]
    fn fault_phantom_delete_is_rejected() {
        let (r, proof) = run_fault(fault::Fault::PhantomDelete);
        let v = strict_verdict(&proof);
        record(fault::Fault::PhantomDelete, &r, &proof, &v);
        assert!(v.as_ref().unwrap_err().contains("absent"), "{v:?}");
    }

    #[test]
    fn fault_wrong_delete_is_rejected() {
        let (r, proof) = run_fault(fault::Fault::WrongDelete);
        let v = strict_verdict(&proof);
        record(fault::Fault::WrongDelete, &r, &proof, &v);
        assert!(
            v.is_err(),
            "deleting live clauses from the log must break a later lemma: {v:?}"
        );
    }

    /// A small corpus on which the lock-related faults are tried. PHP(6)
    /// alone turned out to be vacuous for them (2026-11-01: no locked
    /// LBD > 2 reason at any reduce), so the test sweeps.
    fn lock_corpus() -> Vec<(String, String)> {
        let mut v = vec![
            ("PHP(6)".to_string(), pigeonhole_dimacs(6)),
            ("PHP(7)".to_string(), pigeonhole_dimacs(7)),
        ];
        for seed in 0..40u64 {
            let db = generate_3sat_phase_transition(60, seed);
            let text = crate::bench::db_to_dimacs(&db, 60);
            let inst = parse_dimacs_str(&text).unwrap();
            if solve_watched(inst.db, inst.num_vars) == SolveResult::Unsat {
                v.push((format!("r3sat60/s{seed}"), text));
            }
        }
        for seed in 0..6u64 {
            let db = generate_3sat_phase_transition(120, seed);
            let text = crate::bench::db_to_dimacs(&db, 120);
            let inst = parse_dimacs_str(&text).unwrap();
            if solve_watched(inst.db, inst.num_vars) == SolveResult::Unsat {
                v.push((format!("r3sat120/s{seed}"), text));
            }
        }
        v
    }

    #[test]
    fn fault_empty_lock_is_vacuous_on_small_instances_but_never_unsound() {
        // A forgotten lock. Finding (2026-11-01): on this corpus it never
        // bites — the LBD ≤ 2 keep rule retains every level-0 reason that
        // occurs, so no live reason is ever in the deleted half — which means
        // small-instance tests would NOT catch a forgotten lock. Recorded in
        // research/drat-certification.md. What this test asserts is the
        // soundness half: whenever the solver completes, the proof is valid.
        let mut panics = 0;
        let mut completed_valid = 0;
        let mut completed_invalid = 0;
        for (name, text) in lock_corpus() {
            let (r, proof, _) = run_fault_on(fault::Fault::EmptyLock, &text);
            match r {
                None => panics += 1,
                Some(_) => {
                    let v = strict_verdict_on(&text, &proof);
                    if v.is_ok() {
                        completed_valid += 1;
                    } else {
                        completed_invalid += 1;
                        record_on(&name, &text, fault::Fault::EmptyLock, &r, &proof, &v);
                    }
                }
            }
        }
        eprintln!(
            "FAULT EmptyLock over corpus: panics={panics} completed_valid={completed_valid} \
             completed_invalid={completed_invalid}"
        );
        assert_eq!(
            completed_invalid, 0,
            "a completed solve must still log a valid proof"
        );
    }

    #[test]
    fn fault_delete_locked_trips_the_compaction_guard() {
        // Really delete live reasons: `Trail::remap_reasons` must panic on
        // every instance where the locked set was non-empty at a reduce.
        let mut panics = 0;
        let mut vacuous = 0;
        let mut survived = Vec::new();
        for (name, text) in lock_corpus() {
            let (_, _, logged) = run_fault_on(fault::Fault::LogLockedDelete, &text);
            let (r, _, _) = run_fault_on(fault::Fault::DeleteLocked, &text);
            match (logged, r) {
                (0, _) => vacuous += 1,
                (_, None) => panics += 1,
                (_, Some(_)) => survived.push(name),
            }
        }
        eprintln!(
            "FAULT DeleteLocked over corpus: panics={panics} vacuous={vacuous} survived={survived:?}"
        );
        assert!(panics > 0, "the compaction guard never fired");
        assert!(
            survived.is_empty(),
            "live reasons were deleted and the solver did not refuse: {survived:?}"
        );
    }

    #[test]
    fn fault_log_locked_delete_proof_semantics() {
        // Deletion of locked (level-0 reason) clauses is LOGGED but not done.
        // Prediction: strict checker REJECTS wherever a later lemma needs such
        // a reason; drat-trim default ACCEPTS (ignores pseudo-unit deletions).
        let mut vacuous = 0;
        let mut rejected = 0;
        let mut accepted_nonvacuous = 0;
        let mut dt_accepts = 0;
        let mut dt_runs = 0;
        for (name, text) in lock_corpus() {
            let (r, proof, logged) = run_fault_on(fault::Fault::LogLockedDelete, &text);
            assert_eq!(
                r,
                Some(SolveResult::Unsat),
                "{name}: solver must be unaffected"
            );
            if logged == 0 {
                vacuous += 1;
                continue;
            }
            let v = strict_verdict_on(&text, &proof);
            if v.is_err() {
                rejected += 1;
            } else {
                accepted_nonvacuous += 1;
            }
            if let Some(dt) = drat_trim_verdict(&format!("LLD-{name}"), &text, &proof, &[]) {
                dt_runs += 1;
                if dt == "s VERIFIED" {
                    dt_accepts += 1;
                }
            }
            eprintln!(
                "FAULT LogLockedDelete on {name}: logged={logged} strict={}",
                if v.is_ok() { "ACCEPT" } else { "REJECT" }
            );
        }
        eprintln!(
            "FAULT LogLockedDelete over corpus: vacuous={vacuous} strict_rejected={rejected} \
             strict_accepted_nonvacuous={accepted_nonvacuous} drat-trim accepted {dt_accepts}/{dt_runs}"
        );
        assert!(
            rejected > 0,
            "logging the deletion of level-0 reasons never broke a proof on the corpus"
        );
    }

    /// Checks PHP(6) and PHP(7) proofs with an external DRAT checker when one
    /// is available (`DRAT_TRIM=/path/to/drat-trim` or `drat-trim` on PATH);
    /// otherwise prints that it was skipped. The in-crate checker is not the
    /// real test; this is.
    #[test]
    fn external_drat_trim_if_present() {
        let Some(bin) = drat_trim_bin() else {
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
