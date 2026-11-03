//! AIGER reader and lowering into a [`TransitionSystem`].
//!
//! Reads the binary (`aig`) and ASCII (`aag`) AIGER formats, version 1.0 and
//! the 1.9 extensions this crate can use (bad-state literals `B`, invariant
//! constraints `C`, latch reset values). Justice and fairness sections are
//! parsed past but not supported.
//!
//! # Lowering
//!
//! Every AIG variable (index `1..=M`) becomes one state variable `index - 1`,
//! and one extra state variable `M` stands for the constant `true`, so AIGER
//! literals map one-to-one onto [`Lit`]s. Then, with `n = M + 1`:
//!
//! - **inputs** are unconstrained (free at every step);
//! - **latches** get an initial-state clause from their reset value (none when
//!   uninitialised) and two transition clauses `l' ↔ next(l)`;
//! - **AND gates** `g = a ∧ b` are defined by three clauses in the initial
//!   state (step 0) and on *both* sides of the transition relation, so every
//!   time step carries the definition;
//! - **invariant constraints** are asserted in the initial state and on both
//!   sides of the transition relation;
//! - the single **bad** literal `b` (or output 0 when the file predates `B`)
//!   becomes the one property clause `¬b`.
//!
//! This is the plain unrolling semantics of AIGER safety checking: a trace is
//! a counterexample iff it starts in a reset state, follows the gates and
//! latches, satisfies the constraints at every step, and reaches `b`.
//! Inputs count as state here, which is sound (they are free) and means a PDR
//! invariant may mention them.

use crate::model::TransitionSystem;
use std::fmt;
use warp_types_sat::literal::Lit;

/// A parsed AIG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aig {
    /// Maximum variable index `M`.
    pub max_var: u32,
    /// Input literals (always `2 * index`), in order.
    pub inputs: Vec<u32>,
    /// Latches: `(literal, next-state literal, reset)`.
    pub latches: Vec<Latch>,
    /// Output literals.
    pub outputs: Vec<u32>,
    /// Bad-state literals (AIGER 1.9 `B`).
    pub bad: Vec<u32>,
    /// Invariant-constraint literals (AIGER 1.9 `C`).
    pub constraints: Vec<u32>,
    /// AND gates: `(lhs, rhs0, rhs1)`, lhs even.
    pub ands: Vec<(u32, u32, u32)>,
}

/// One latch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Latch {
    pub lit: u32,
    pub next: u32,
    pub reset: Reset,
}

/// A latch's reset value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reset {
    Zero,
    One,
    /// Uninitialised: any value at step 0.
    Undefined,
}

/// Parse error.
#[derive(Debug)]
pub enum AigerError {
    Format(String),
    Io(std::io::Error),
}

impl fmt::Display for AigerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AigerError::Format(s) => write!(f, "AIGER format error: {s}"),
            AigerError::Io(e) => write!(f, "AIGER read error: {e}"),
        }
    }
}

impl std::error::Error for AigerError {}

impl From<std::io::Error> for AigerError {
    fn from(e: std::io::Error) -> Self {
        AigerError::Io(e)
    }
}

fn err<T>(s: impl Into<String>) -> Result<T, AigerError> {
    Err(AigerError::Format(s.into()))
}

/// Read a line (without the newline) from `bytes` starting at `*pos`.
fn read_line<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a str, AigerError> {
    if *pos >= bytes.len() {
        return err("unexpected end of file");
    }
    let start = *pos;
    while *pos < bytes.len() && bytes[*pos] != b'\n' {
        *pos += 1;
    }
    let line = std::str::from_utf8(&bytes[start..*pos])
        .map_err(|_| AigerError::Format("non-UTF-8 header line".into()))?;
    if *pos < bytes.len() {
        *pos += 1; // consume '\n'
    }
    Ok(line.trim_end_matches('\r'))
}

fn parse_u32(tok: &str) -> Result<u32, AigerError> {
    tok.parse::<u32>()
        .map_err(|_| AigerError::Format(format!("bad number {tok:?}")))
}

/// Binary-AIGER delta decoding (7 bits per byte, MSB = continue).
fn decode_delta(bytes: &[u8], pos: &mut usize) -> Result<u32, AigerError> {
    let mut x: u32 = 0;
    let mut shift = 0u32;
    loop {
        if *pos >= bytes.len() {
            return err("truncated AND section");
        }
        let ch = bytes[*pos];
        *pos += 1;
        if shift >= 32 {
            return err("delta overflow in AND section");
        }
        x |= ((ch & 0x7f) as u32) << shift;
        if ch & 0x80 == 0 {
            return Ok(x);
        }
        shift += 7;
    }
}

/// Parse AIGER bytes, binary or ASCII.
pub fn parse(bytes: &[u8]) -> Result<Aig, AigerError> {
    let mut pos = 0usize;
    let header = read_line(bytes, &mut pos)?;
    let toks: Vec<&str> = header.split_whitespace().collect();
    if toks.len() < 6 || (toks[0] != "aig" && toks[0] != "aag") {
        return err(format!("bad header {header:?}"));
    }
    let binary = toks[0] == "aig";
    let nums: Vec<u32> = toks[1..]
        .iter()
        .map(|t| parse_u32(t))
        .collect::<Result<_, _>>()?;
    let (m, i, l, o, a) = (nums[0], nums[1], nums[2], nums[3], nums[4]);
    let b = nums.get(5).copied().unwrap_or(0);
    let c = nums.get(6).copied().unwrap_or(0);
    let j = nums.get(7).copied().unwrap_or(0);
    let f = nums.get(8).copied().unwrap_or(0);
    if m != i + l + a {
        return err(format!("M={m} but I+L+A={}", i + l + a));
    }

    let mut aig = Aig {
        max_var: m,
        inputs: Vec::with_capacity(i as usize),
        latches: Vec::with_capacity(l as usize),
        outputs: Vec::with_capacity(o as usize),
        bad: Vec::with_capacity(b as usize),
        constraints: Vec::with_capacity(c as usize),
        ands: Vec::with_capacity(a as usize),
    };

    // Inputs
    if binary {
        for k in 0..i {
            aig.inputs.push(2 * (k + 1));
        }
    } else {
        for _ in 0..i {
            aig.inputs
                .push(parse_u32(read_line(bytes, &mut pos)?.trim())?);
        }
    }
    // Latches
    for k in 0..l {
        let line = read_line(bytes, &mut pos)?;
        let t: Vec<&str> = line.split_whitespace().collect();
        let (lit, rest) = if binary {
            (2 * (i + k + 1), &t[..])
        } else {
            if t.is_empty() {
                return err("empty latch line");
            }
            (parse_u32(t[0])?, &t[1..])
        };
        if rest.is_empty() {
            return err("latch without next-state literal");
        }
        let next = parse_u32(rest[0])?;
        let reset = match rest.get(1) {
            None => Reset::Zero,
            Some(r) => {
                let r = parse_u32(r)?;
                if r == 0 {
                    Reset::Zero
                } else if r == 1 {
                    Reset::One
                } else if r == lit {
                    Reset::Undefined
                } else {
                    return err(format!(
                        "latch {lit}: reset {r} is neither 0, 1 nor the latch"
                    ));
                }
            }
        };
        aig.latches.push(Latch { lit, next, reset });
    }
    // Outputs, bad, constraints, justice (skipped), fairness (skipped)
    for _ in 0..o {
        aig.outputs
            .push(parse_u32(read_line(bytes, &mut pos)?.trim())?);
    }
    for _ in 0..b {
        aig.bad.push(parse_u32(read_line(bytes, &mut pos)?.trim())?);
    }
    for _ in 0..c {
        aig.constraints
            .push(parse_u32(read_line(bytes, &mut pos)?.trim())?);
    }
    if j > 0 || f > 0 {
        // Justice: J lines each "n" then n literal lines; fairness: F lines.
        let mut justice_lits = Vec::new();
        for _ in 0..j {
            justice_lits.push(parse_u32(read_line(bytes, &mut pos)?.trim())?);
        }
        for n in justice_lits {
            for _ in 0..n {
                read_line(bytes, &mut pos)?;
            }
        }
        for _ in 0..f {
            read_line(bytes, &mut pos)?;
        }
    }
    // ANDs
    if binary {
        for k in 0..a {
            let lhs = 2 * (i + l + k + 1);
            let d0 = decode_delta(bytes, &mut pos)?;
            let rhs0 = lhs
                .checked_sub(d0)
                .ok_or_else(|| AigerError::Format(format!("AND {lhs}: delta0 {d0} too large")))?;
            let d1 = decode_delta(bytes, &mut pos)?;
            let rhs1 = rhs0
                .checked_sub(d1)
                .ok_or_else(|| AigerError::Format(format!("AND {lhs}: delta1 {d1} too large")))?;
            aig.ands.push((lhs, rhs0, rhs1));
        }
    } else {
        for _ in 0..a {
            let line = read_line(bytes, &mut pos)?;
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() != 3 {
                return err(format!("bad AND line {line:?}"));
            }
            aig.ands
                .push((parse_u32(t[0])?, parse_u32(t[1])?, parse_u32(t[2])?));
        }
    }
    // Validate literal ranges.
    let max_lit = 2 * m + 1;
    let check = |lit: u32, what: &str| -> Result<(), AigerError> {
        if lit > max_lit {
            err(format!("{what} literal {lit} exceeds 2M+1 = {max_lit}"))
        } else {
            Ok(())
        }
    };
    for &x in &aig.outputs {
        check(x, "output")?;
    }
    for &x in &aig.bad {
        check(x, "bad")?;
    }
    for &x in &aig.constraints {
        check(x, "constraint")?;
    }
    for la in &aig.latches {
        check(la.next, "latch next")?;
    }
    for &(lhs, r0, r1) in &aig.ands {
        check(lhs, "AND lhs")?;
        check(r0, "AND rhs0")?;
        check(r1, "AND rhs1")?;
        if lhs & 1 == 1 {
            return err(format!("AND lhs {lhs} is negated"));
        }
    }
    Ok(aig)
}

/// Read and parse an AIGER file.
pub fn read_file(path: impl AsRef<std::path::Path>) -> Result<Aig, AigerError> {
    parse(&std::fs::read(path)?)
}

/// 64-bit FNV-1a of a file's bytes, as `pdr --invariant` records it and
/// `aig-witness` checks it: an invariant file names the model it is about,
/// and a witness built against a different file of the same name is refused
/// instead of being certified for the wrong circuit. Not cryptographic — a
/// mix-up detector, which is the failure this guards against.
pub fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The `fnv1a64 <hex>` recorded in an invariant file's comment, if any.
pub fn invariant_file_fingerprint(text: &str) -> Option<u64> {
    for line in text.lines().take_while(|l| l.starts_with('c')) {
        if let Some(i) = line.find("fnv1a64 ") {
            return u64::from_str_radix(line[i + 8..].split_whitespace().next()?, 16).ok();
        }
    }
    None
}

/// How AIGER literals map onto the lowered system's state variables.
///
/// `var_of[index]` is the state variable of AIG variable `index` (`None` when
/// the node was dropped by cone-of-influence reduction); `index_of[var]` is
/// the inverse. The constant-true variable is the last one.
#[derive(Debug, Clone)]
pub struct Lowering {
    /// Number of state variables (kept nodes + 1).
    pub num_state_vars: u32,
    /// The state variable that is constrained to `true`.
    pub const_var: u32,
    var_of: Vec<Option<u32>>,
    index_of: Vec<u32>,
}

impl Lowering {
    /// Full lowering: every AIG variable `1..=M` is state variable `index - 1`.
    pub fn full(max_var: u32) -> Self {
        Self::from_keep(&(0..=max_var as usize).map(|i| i != 0).collect::<Vec<_>>())
    }

    /// Lowering over the AIG variables with `keep[index] == true`.
    pub fn from_keep(keep: &[bool]) -> Self {
        let mut var_of = vec![None; keep.len()];
        let mut index_of = Vec::new();
        for (index, &k) in keep.iter().enumerate().skip(1) {
            if k {
                var_of[index] = Some(index_of.len() as u32);
                index_of.push(index as u32);
            }
        }
        let const_var = index_of.len() as u32;
        Lowering {
            num_state_vars: const_var + 1,
            const_var,
            var_of,
            index_of,
        }
    }

    /// Number of AIG variables kept (excluding the constant).
    pub fn kept(&self) -> usize {
        self.index_of.len()
    }

    /// Is AIG variable `index` represented?
    pub fn keeps(&self, index: u32) -> bool {
        index == 0 || self.var_of.get(index as usize).is_some_and(|v| v.is_some())
    }

    /// AIGER literal → current-state [`Lit`], `None` for a dropped node.
    pub fn try_lit(&self, aiger_lit: u32) -> Option<Lit> {
        let index = aiger_lit >> 1;
        let neg = aiger_lit & 1 == 1;
        if index == 0 {
            // AIGER literal 0 is FALSE and literal 1 is TRUE: the polarity is
            // the reverse of an ordinary variable's, because the constant
            // variable is constrained to true.
            return Some(if neg {
                Lit::pos(self.const_var)
            } else {
                Lit::neg(self.const_var)
            });
        }
        let var = (*self.var_of.get(index as usize)?)?;
        Some(if neg { Lit::neg(var) } else { Lit::pos(var) })
    }

    /// AIGER literal → current-state [`Lit`].
    ///
    /// # Panics
    /// If the literal's variable was dropped by cone-of-influence reduction.
    pub fn lit(&self, aiger_lit: u32) -> Lit {
        self.try_lit(aiger_lit)
            .unwrap_or_else(|| panic!("AIGER literal {aiger_lit} is outside the lowered cone"))
    }
    /// AIGER literal → next-state [`Lit`] (shifted by `num_state_vars`).
    pub fn next_lit(&self, aiger_lit: u32) -> Lit {
        let l = self.lit(aiger_lit);
        let v = l.var() + self.num_state_vars;
        if l.is_negated() {
            Lit::neg(v)
        } else {
            Lit::pos(v)
        }
    }
    /// State variable → AIGER literal; the constant variable maps to
    /// literal `1` (`0` when negated).
    pub fn aiger_lit(&self, lit: Lit) -> u32 {
        let base = if lit.var() == self.const_var {
            1
        } else {
            2 * self.index_of[lit.var() as usize]
        };
        if lit.is_negated() {
            base ^ 1
        } else {
            base
        }
    }
}

/// Write a counterexample trace in a replayable form: for every step the
/// value of every AIGER input, and at step 0 the value of every latch (needed
/// only for uninitialised latches, written for all). One line per
/// assignment: `<step> <aiger-literal> <0|1>`. `tools/replay-trace.py`
/// re-simulates the AIG from reset with these inputs and checks that the bad
/// literal is true at the last step and the constraints hold throughout.
/// Parse an invariant file as `pdr --invariant` writes it: comment lines
/// start with `c`; every other line is one clause over AIGER literals,
/// `0`-terminated.
pub fn parse_invariant_file(text: &str) -> Result<Vec<Vec<u32>>, AigerError> {
    let mut clauses = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('c') {
            continue;
        }
        let mut clause = Vec::new();
        for tok in t.split_whitespace() {
            let v: u32 = tok.parse().map_err(|_| {
                AigerError::Format(format!("invariant line {}: bad literal {tok:?}", n + 1))
            })?;
            if v == 0 {
                break;
            }
            clause.push(v);
        }
        clauses.push(clause);
    }
    Ok(clauses)
}

/// A Certifaiger witness for an inductive invariant: the model itself, with
/// its bad signal replaced by `¬(Inv ∧ P)`, where `Inv` is the conjunction of
/// `clauses` (AIGER literals of the model) and `P` the conjunction of the
/// model's negated bad signals (outputs, for pre-1.9 files). Inputs, latches,
/// resets, constraints and gates are kept verbatim and in order, so the
/// default identity mapping applies and Certifaiger's simulation checks are
/// trivial; its inductive checks (`R' → P'`, `F' ∧ P'_s → P'_t`) are exactly
/// initiation and consecution of `Inv ∧ P`, and its safety check `P' → P`
/// holds by construction. ASCII AIGER 1.9 (`aag M I L O A B C`), one bad.
/// See `research/bmc-pdr-aiger.md` and Froleyks, Yu & Biere, CAV25.
/// AND-gate builder over a model's literals, with constant folding.
struct GateBuilder {
    next_var: u32,
    ands: Vec<(u32, u32, u32)>,
}

impl GateBuilder {
    fn and2(&mut self, a: u32, b: u32) -> u32 {
        match (a, b) {
            (0, _) | (_, 0) => 0,
            (1, x) | (x, 1) => x,
            _ if a == b => a,
            _ if a == (b ^ 1) => 0,
            _ => {
                self.next_var += 1;
                let lhs = 2 * self.next_var;
                self.ands.push((lhs, a.max(b), a.min(b)));
                lhs
            }
        }
    }
    fn and_all(&mut self, lits: &[u32]) -> u32 {
        lits.iter().fold(1u32, |acc, &l| self.and2(acc, l))
    }
}

pub fn write_witness(aig: &Aig, clauses: &[Vec<u32>], model_name: &str) -> String {
    let mut g = GateBuilder {
        next_var: aig.max_var,
        ands: aig.ands.clone(),
    };
    let bads: &[u32] = if aig.bad.is_empty() {
        &aig.outputs
    } else {
        &aig.bad
    };
    let mut conj: Vec<u32> = Vec::with_capacity(clauses.len() + bads.len());
    for c in clauses {
        // clause = ¬ AND(¬l)
        let negs: Vec<u32> = c.iter().map(|&l| l ^ 1).collect();
        let not_c = g.and_all(&negs);
        conj.push(not_c ^ 1);
    }
    for &b in bads {
        conj.push(b ^ 1);
    }
    let p_prime = g.and_all(&conj);
    let bad_prime = p_prime ^ 1;
    let (next_var, ands) = (g.next_var, g.ands);
    let mut out = String::new();
    out.push_str(&format!(
        "aag {} {} {} 0 {} 1 {}\n",
        next_var,
        aig.inputs.len(),
        aig.latches.len(),
        ands.len(),
        aig.constraints.len()
    ));
    for &i in &aig.inputs {
        out.push_str(&format!("{i}\n"));
    }
    for l in &aig.latches {
        match l.reset {
            Reset::Zero => out.push_str(&format!("{} {} 0\n", l.lit, l.next)),
            Reset::One => out.push_str(&format!("{} {} 1\n", l.lit, l.next)),
            Reset::Undefined => out.push_str(&format!("{} {} {}\n", l.lit, l.next, l.lit)),
        }
    }
    out.push_str(&format!("{bad_prime}\n"));
    for &c in &aig.constraints {
        out.push_str(&format!("{c}\n"));
    }
    for &(lhs, r0, r1) in &ands {
        out.push_str(&format!("{lhs} {r0} {r1}\n"));
    }
    out.push_str(&format!(
        "c\nWITNESS b0 {model_name}\ninductive invariant from warp-types-pdr: {} clause(s) + property; model kept verbatim (identity mapping)\n",
        clauses.len()
    ));
    out
}

pub fn write_trace(aig: &Aig, lw: &Lowering, trace: &[Vec<bool>], path: &str, out: &mut String) {
    // A node outside the lowered cone cannot influence the property or the
    // constraints, so any value will do for the replay: write 0.
    let val = |step: &Vec<bool>, aiger_lit: u32| -> u8 {
        match lw.try_lit(aiger_lit) {
            Some(l) => {
                let v = step[l.var() as usize];
                u8::from(if l.is_negated() { !v } else { v })
            }
            None => 0,
        }
    };
    out.push_str(&format!(
        "c counterexample trace for {path}: depth {}; lines are `step aiger-literal value`\n",
        trace.len().saturating_sub(1)
    ));
    if let Some(first) = trace.first() {
        for la in &aig.latches {
            out.push_str(&format!("0 {} {}\n", la.lit, val(first, la.lit)));
        }
    }
    for (k, step) in trace.iter().enumerate() {
        for &i in &aig.inputs {
            out.push_str(&format!("{k} {i} {}\n", val(step, i)));
        }
    }
}

/// Which literal the lowering treats as the bad state.
fn bad_literal(aig: &Aig) -> Result<u32, AigerError> {
    match (aig.bad.len(), aig.outputs.len()) {
        (1, _) => Ok(aig.bad[0]),
        (0, 1) => Ok(aig.outputs[0]),
        (0, 0) => err("no bad literal and no output"),
        (nb, no) => err(format!(
            "expected exactly one bad literal (or one output), found B={nb} O={no}"
        )),
    }
}

/// The cone of influence of the bad literal and the invariant constraints:
/// `keep[index]` is true for every AIG variable that can reach them through
/// AND inputs or latch next-state literals. Inputs and gates outside the cone
/// cannot affect any verdict.
pub fn cone_of_influence(aig: &Aig) -> Result<Vec<bool>, AigerError> {
    let bad = bad_literal(aig)?;
    let m = aig.max_var as usize;
    let mut and_of: Vec<Option<(u32, u32)>> = vec![None; m + 1];
    for &(lhs, r0, r1) in &aig.ands {
        and_of[(lhs >> 1) as usize] = Some((r0, r1));
    }
    let mut next_of: Vec<Option<u32>> = vec![None; m + 1];
    for la in &aig.latches {
        next_of[(la.lit >> 1) as usize] = Some(la.next);
    }
    let mut keep = vec![false; m + 1];
    let mut stack: Vec<u32> = Vec::new();
    let push = |lit: u32, keep: &mut Vec<bool>, stack: &mut Vec<u32>| {
        let i = (lit >> 1) as usize;
        if i != 0 && !keep[i] {
            keep[i] = true;
            stack.push(i as u32);
        }
    };
    push(bad, &mut keep, &mut stack);
    for &c in &aig.constraints {
        push(c, &mut keep, &mut stack);
    }
    while let Some(i) = stack.pop() {
        if let Some((r0, r1)) = and_of[i as usize] {
            push(r0, &mut keep, &mut stack);
            push(r1, &mut keep, &mut stack);
        } else if let Some(nx) = next_of[i as usize] {
            push(nx, &mut keep, &mut stack);
        }
    }
    Ok(keep)
}

/// Lower an AIG into a [`TransitionSystem`] (see the module docs for the
/// encoding), keeping every AIG variable. Errors if the AIG does not have
/// exactly one bad literal (or, for pre-1.9 files, exactly one output).
pub fn lower(aig: &Aig) -> Result<(TransitionSystem, Lowering), AigerError> {
    bad_literal(aig)?;
    lower_with(aig, Lowering::full(aig.max_var))
}

/// Lower only the cone of influence of the bad literal and the constraints
/// (see [`cone_of_influence`]). Same verdicts and counterexample depths as
/// [`lower`]; fewer state variables.
pub fn lower_coi(aig: &Aig) -> Result<(TransitionSystem, Lowering), AigerError> {
    let keep = cone_of_influence(aig)?;
    lower_with(aig, Lowering::from_keep(&keep))
}

fn lower_with(aig: &Aig, lw: Lowering) -> Result<(TransitionSystem, Lowering), AigerError> {
    let bad = bad_literal(aig)?;
    let n = lw.num_state_vars;
    let mut sys = TransitionSystem::new(n);

    // Constant true: initial, and on both sides of T (the current-side unit
    // keeps PDR from exploring predecessor states with the constant false;
    // added 2026-11-02 after the panel — semantically neutral, see the log).
    sys.add_initial(vec![Lit::pos(lw.const_var)]);
    sys.add_transition(vec![Lit::pos(lw.const_var)]);
    sys.add_transition(vec![Lit::pos(lw.const_var + n)]);

    // Latches.
    for la in &aig.latches {
        if !lw.keeps(la.lit >> 1) {
            continue;
        }
        match la.reset {
            Reset::Zero => sys.add_initial(vec![lw.lit(la.lit).complement()]),
            Reset::One => sys.add_initial(vec![lw.lit(la.lit)]),
            Reset::Undefined => {}
        }
        let l_next = lw.next_lit(la.lit);
        let nx = lw.lit(la.next);
        sys.add_transition(vec![l_next.complement(), nx]);
        sys.add_transition(vec![l_next, nx.complement()]);
    }

    // AND gates: g ↔ a ∧ b at step 0 (initial) and on both sides of T.
    for &(lhs, r0, r1) in &aig.ands {
        if !lw.keeps(lhs >> 1) {
            continue;
        }
        for side in 0..3 {
            let (g, a, b) = match side {
                0 => (lw.lit(lhs), lw.lit(r0), lw.lit(r1)),
                1 => (lw.lit(lhs), lw.lit(r0), lw.lit(r1)),
                _ => (lw.next_lit(lhs), lw.next_lit(r0), lw.next_lit(r1)),
            };
            let c1 = vec![g.complement(), a];
            let c2 = vec![g.complement(), b];
            let c3 = vec![g, a.complement(), b.complement()];
            if side == 0 {
                sys.add_initial(c1);
                sys.add_initial(c2);
                sys.add_initial(c3);
            } else {
                sys.add_transition(c1);
                sys.add_transition(c2);
                sys.add_transition(c3);
            }
        }
    }

    // Invariant constraints: every step.
    for &c in &aig.constraints {
        sys.add_initial(vec![lw.lit(c)]);
        sys.add_transition(vec![lw.lit(c)]);
        sys.add_transition(vec![lw.next_lit(c)]);
    }

    // Property: ¬bad.
    sys.add_property(vec![lw.lit(bad).complement()]);
    Ok((sys, lw))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1-latch toggle with a bad signal that can never fire; the witness
    /// keeps the model verbatim and adds `¬(Inv ∧ P)` as its single bad.
    #[test]
    fn witness_keeps_model_and_adds_invariant_gates() {
        // aag: M=3 I=1 L=1 O=0 A=1 B=1: latch 2 (reset 0) next = 6 = AND(¬2, 4); bad = 2 ∧ ¬2 = const 0? keep it simple: bad = 0.
        let model = "aag 3 1 1 0 1 1\n4\n2 6 0\n0\n6 3 4\n";
        let aig = parse(model.as_bytes()).unwrap();
        let clauses = vec![vec![3u32], vec![3, 4]]; // ¬latch ; (¬latch ∨ input)
        let w = write_witness(&aig, &clauses, "model.aag");
        let waig = parse(w.as_bytes()).unwrap();
        assert_eq!(waig.inputs, aig.inputs);
        assert_eq!(waig.latches, aig.latches);
        assert_eq!(&waig.ands[..aig.ands.len()], &aig.ands[..]);
        assert_eq!(waig.bad.len(), 1);
        assert!(waig.ands.len() > aig.ands.len());
        assert!(w.contains("WITNESS b0 model.aag"));
        assert_eq!(
            parse_invariant_file("c comment\n3 0\n3 4 0\n").unwrap(),
            clauses
        );
    }
    use crate::checker::{check, BmcResult};

    /// 2-bit counter in ASCII AIGER: latches l0 (bit 0), l1 (bit 1);
    /// l0' = ¬l0; l1' = l0 xor l1; bad = l0 ∧ l1. Counterexample at depth 3.
    const COUNTER_AAG: &str = "aag 7 0 2 0 5 1\n\
        2 3\n\
        4 14\n\
        6\n\
        6 4 2\n\
        8 5 3\n\
        10 5 2\n\
        12 4 3\n\
        14 13 11\n";
    // Gates: 6 = l0 ∧ l1 (bad); 8 = ¬l0 ∧ ¬l1; 10 = l0 ∧ ¬l1; 12 = ¬l0 ∧ l1;
    // 14 = ¬10 ∧ ¬12 = ¬(l0 xor l1)... careful: xor = 10 ∨ 12 = ¬(¬10 ∧ ¬12) = ¬14,
    // so l1' should be 15. Use 15 below instead (test fixes the literal).

    fn counter() -> Aig {
        let text = COUNTER_AAG.replace("4 14\n", "4 15\n");
        parse(text.as_bytes()).unwrap()
    }

    #[test]
    fn parses_ascii_counter() {
        let aig = counter();
        assert_eq!(aig.max_var, 7);
        assert_eq!(aig.latches.len(), 2);
        assert_eq!(aig.latches[0].next, 3);
        assert_eq!(aig.latches[1].next, 15);
        assert_eq!(aig.bad, vec![6]);
        assert_eq!(aig.ands.len(), 5);
    }

    #[test]
    fn counter_bmc_finds_depth_3() {
        let (sys, _) = lower(&counter()).unwrap();
        match check(&sys, 10, 0) {
            BmcResult::CounterexampleFound { depth, .. } => assert_eq!(depth, 3),
            r => panic!("expected counterexample, got {r:?}"),
        }
    }

    #[test]
    fn binary_roundtrip_matches_ascii() {
        // Encode the same counter in binary AIGER by hand: header, latch
        // lines, bad line, then delta-coded ANDs.
        let aig = counter();
        let mut bytes = b"aig 7 0 2 0 5 1\n3\n15\n6\n".to_vec();
        for &(lhs, r0, r1) in &aig.ands {
            for mut d in [lhs - r0, r0 - r1] {
                loop {
                    let byte = (d & 0x7f) as u8;
                    d >>= 7;
                    if d == 0 {
                        bytes.push(byte);
                        break;
                    }
                    bytes.push(byte | 0x80);
                }
            }
        }
        let bin = parse(&bytes).unwrap();
        assert_eq!(bin, aig);
    }

    #[test]
    fn lowering_literal_map_roundtrips() {
        let lw = Lowering::full(7);
        for aiger_lit in 0..16u32 {
            assert_eq!(
                lw.aiger_lit(lw.lit(aiger_lit)),
                aiger_lit,
                "lit {aiger_lit}"
            );
        }
        // Constants: literal 0 is false, literal 1 is true (the constant
        // variable is asserted true, so false is its negation).
        assert_eq!(lw.lit(0), Lit::neg(7));
        assert_eq!(lw.lit(1), Lit::pos(7));
    }

    #[test]
    fn constant_bad_literal_behaves() {
        // bad = literal 1 (true): counterexample at depth 0.
        let aig = parse(b"aag 0 0 0 0 0 1\n1\n").unwrap();
        let (sys, _) = lower(&aig).unwrap();
        assert!(matches!(
            check(&sys, 3, 0),
            BmcResult::CounterexampleFound { depth: 0, .. }
        ));
        // bad = literal 0 (false): never.
        let aig = parse(b"aag 0 0 0 0 0 1\n0\n").unwrap();
        let (sys, _) = lower(&aig).unwrap();
        assert!(matches!(check(&sys, 3, 0), BmcResult::BoundedSafe { .. }));
    }

    #[test]
    fn constraint_can_make_bad_unreachable() {
        // Same counter, but constrain ¬l1 at every step: the counter can never
        // show bit 1, so bad (l0 ∧ l1) is unreachable within any bound.
        let text = "aag 7 0 2 0 5 1 1\n2 3\n4 15\n6\n5\n6 4 2\n8 5 3\n10 5 2\n12 4 3\n14 13 11\n";
        let aig = parse(text.as_bytes()).unwrap();
        assert_eq!(aig.constraints, vec![5]);
        let (sys, _) = lower(&aig).unwrap();
        assert!(matches!(check(&sys, 8, 0), BmcResult::BoundedSafe { .. }));
    }

    #[test]
    fn constant_unit_on_both_sides_of_t() {
        let (sys, lw) = lower(&counter()).unwrap();
        let n = sys.num_state_vars();
        let has = |v: u32| {
            sys.transition()
                .iter()
                .any(|c| c.lits.len() == 1 && c.lits[0] == Lit::pos(v))
        };
        assert!(has(lw.const_var));
        assert!(has(lw.const_var + n));
    }

    #[test]
    fn trace_writer_lists_inputs_per_step_and_latches_at_zero() {
        // Counter has 2 latches and no inputs: 2 lines, both step 0.
        let aig = counter();
        let (sys, lw) = lower(&aig).unwrap();
        let trace = match check(&sys, 10, 0) {
            BmcResult::CounterexampleFound { trace, .. } => trace,
            r => panic!("{r:?}"),
        };
        let mut out = String::new();
        write_trace(&aig, &lw, &trace, "counter", &mut out);
        let lines: Vec<&str> = out.lines().filter(|l| !l.starts_with('c')).collect();
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().all(|l| l.starts_with("0 ")));
        // Reset is 0 for both latches, so step-0 latch values are 0.
        assert!(lines.iter().all(|l| l.ends_with(" 0")));
    }

    #[test]
    fn coi_drops_a_dangling_gate_and_keeps_the_verdict() {
        // Counter plus a gate 16 = l0 ∧ i (input 2 renumbered) that nothing
        // reads: M=9 with one input. Encode: input 2; latches 4 (l0), 6 (l1);
        // gates as in the counter shifted by one variable; dangling 18.
        let text =
            "aag 9 1 2 0 6 1\n2\n4 5\n6 17\n8\n8 6 4\n10 7 5\n12 7 4\n14 6 5\n16 15 13\n18 4 2\n";
        let aig = parse(text.as_bytes()).unwrap();
        let keep = cone_of_influence(&aig).unwrap();
        assert!(!keep[9], "gate 18 is outside the cone");
        assert!(!keep[1], "the input feeds only the dangling gate");
        let (full, lwf) = lower(&aig).unwrap();
        let (coi, lwc) = lower_coi(&aig).unwrap();
        assert_eq!(lwf.kept(), 9);
        // Kept: l0, l1, bad gate 8, the xor's three gates 12/14/16; dropped:
        // gate 10 (¬l0 ∧ ¬l1, read by nothing even in the plain counter),
        // gate 18 and the input.
        assert!(!keep[5], "gate 10 is read by nothing");
        assert_eq!(lwc.kept(), 6);
        assert!(coi.num_state_vars() < full.num_state_vars());
        let d = |sys: &TransitionSystem| match check(sys, 10, 0) {
            BmcResult::CounterexampleFound { depth, .. } => depth,
            r => panic!("{r:?}"),
        };
        assert_eq!(d(&full), 3);
        assert_eq!(d(&coi), 3);
        // Literal map round-trips on the reduced lowering too.
        for &lit in &[0u32, 1, 8, 9, 16, 17] {
            assert_eq!(lwc.aiger_lit(lwc.lit(lit)), lit);
        }
        assert!(lwc.try_lit(18).is_none());
    }

    #[test]
    fn coi_keeps_constraint_cones() {
        // Constraint on an otherwise-dangling input keeps it.
        let text = "aag 9 1 2 0 6 1 1\n2\n4 5\n6 17\n8\n3\n8 6 4\n10 7 5\n12 7 4\n14 6 5\n16 15 13\n18 4 2\n";
        let aig = parse(text.as_bytes()).unwrap();
        let keep = cone_of_influence(&aig).unwrap();
        assert!(keep[1], "constraint literal 3 keeps input 2");
        assert!(!keep[9]);
    }

    #[test]
    fn rejects_multiple_bad() {
        let text = "aag 1 1 0 0 0 2\n2\n2\n3\n";
        let aig = parse(text.as_bytes()).unwrap();
        assert!(lower(&aig).is_err());
    }
}
