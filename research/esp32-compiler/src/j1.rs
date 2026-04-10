//! J1-32 instruction encoding.
//!
//! The J1 is a dual-stack machine (data stack T/N, return stack R).
//! Instructions are 32-bit words. We use a simple tagged encoding:
//!   bits [31:24] = opcode tag
//!   bits [23:0]  = operand (address or immediate)

extern crate alloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Op {
    Lit   = 0x01, // Push immediate to TOS
    Add   = 0x10, // NOS + TOS → TOS
    Sub   = 0x11, // NOS - TOS → TOS
    Mul   = 0x12, // NOS * TOS → TOS
    Eq    = 0x13, // NOS == TOS → TOS (1 or 0)
    Lt    = 0x14, // NOS < TOS → TOS
    Gt    = 0x15, // NOS > TOS → TOS
    Dup   = 0x20,
    Drop  = 0x21,
    Swap  = 0x22,
    ToR   = 0x23, // >R: TOS → return stack
    FromR = 0x24, // R>: return stack → TOS
    Bz    = 0x30, // Branch if TOS == 0
    Jmp   = 0x31, // Unconditional jump
    Call  = 0x32, // Call subroutine
    Ret   = 0x33, // Return from subroutine
    ChSend = 0x40, // CSP send: TOS=channel, NOS=value
    ChRecv = 0x41, // CSP recv: TOS=channel, result→TOS
}

/// Encode an instruction word.
pub fn encode(op: Op, operand: u32) -> u32 {
    ((op as u32) << 24) | (operand & 0x00FF_FFFF)
}

/// Encode an instruction with no operand.
pub fn encode_simple(op: Op) -> u32 {
    encode(op, 0)
}
