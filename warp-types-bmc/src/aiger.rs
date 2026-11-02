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

/// How AIGER literals map onto the lowered system's state variables.
#[derive(Debug, Clone, Copy)]
pub struct Lowering {
    /// Number of state variables, `M + 1`.
    pub num_state_vars: u32,
    /// The state variable that is constrained to `true` (index `M`).
    pub const_var: u32,
}

impl Lowering {
    /// AIGER literal → current-state [`Lit`].
    pub fn lit(&self, aiger_lit: u32) -> Lit {
        let index = aiger_lit >> 1;
        let neg = aiger_lit & 1 == 1;
        if index == 0 {
            // AIGER literal 0 is FALSE and literal 1 is TRUE: the polarity is
            // the reverse of an ordinary variable's, because the constant
            // variable is constrained to true.
            return if neg {
                Lit::pos(self.const_var)
            } else {
                Lit::neg(self.const_var)
            };
        }
        let var = index - 1;
        if neg {
            Lit::neg(var)
        } else {
            Lit::pos(var)
        }
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
    /// State variable → AIGER literal (positive polarity); the constant
    /// variable maps to literal `1`.
    pub fn aiger_lit(&self, lit: Lit) -> u32 {
        let base = if lit.var() == self.const_var {
            1
        } else {
            2 * (lit.var() + 1)
        };
        if lit.is_negated() {
            base ^ 1
        } else {
            base
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

/// Lower an AIG into a [`TransitionSystem`] (see the module docs for the
/// encoding). Errors if the AIG does not have exactly one bad literal (or,
/// for pre-1.9 files, exactly one output).
pub fn lower(aig: &Aig) -> Result<(TransitionSystem, Lowering), AigerError> {
    let bad = bad_literal(aig)?;
    let n = aig.max_var + 1;
    let lw = Lowering {
        num_state_vars: n,
        const_var: aig.max_var,
    };
    let mut sys = TransitionSystem::new(n);

    // Constant true: initial and next.
    sys.add_initial(vec![Lit::pos(lw.const_var)]);
    sys.add_transition(vec![Lit::pos(lw.const_var + n)]);

    // Latches.
    for la in &aig.latches {
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
        let lw = Lowering {
            num_state_vars: 8,
            const_var: 7,
        };
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
    fn rejects_multiple_bad() {
        let text = "aag 1 1 0 0 0 2\n2\n2\n3\n";
        let aig = parse(text.as_bytes()).unwrap();
        assert!(lower(&aig).is_err());
    }
}
