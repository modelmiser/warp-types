//! AST → J1-32 instruction words.
//!
//! Tree-walking code generator. Emits a flat Vec<u32> of instruction words.
//! Variables are tracked by stack position (pure stack machine — no registers).

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ast::*;
use crate::j1::{self, Op};

/// Channel name → numeric channel ID for J1 CSP instructions.
fn channel_id(name: &str) -> u32 {
    match name {
        "ch01" => 1,
        "ch10" => 0,
        _ => 0xFF, // unknown channel
    }
}

pub struct Codegen {
    pub code: Vec<u32>,
}

impl Codegen {
    pub fn new() -> Self {
        Codegen { code: Vec::new() }
    }

    fn emit(&mut self, word: u32) {
        self.code.push(word);
    }

    fn here(&self) -> usize {
        self.code.len()
    }

    /// Patch a branch/jump instruction at `addr` to point to `target`.
    fn patch(&mut self, addr: usize, target: usize) {
        let opcode = self.code[addr] & 0xFF00_0000;
        self.code[addr] = opcode | (target as u32 & 0x00FF_FFFF);
    }

    // ── Public API ─────────────────────────────────────────────

    pub fn gen_function(&mut self, f: &FnDef) -> Result<(), String> {
        // Function label is just the current position.
        // Parameters are already on the data stack (caller pushes them).
        self.gen_block(&f.body)?;
        self.emit(j1::encode_simple(Op::Ret));
        Ok(())
    }

    fn gen_block(&mut self, block: &Block) -> Result<(), String> {
        for stmt in &block.stmts {
            self.gen_stmt(stmt)?;
        }
        if let Some(tail) = &block.tail {
            self.gen_expr(tail)?;
        }
        Ok(())
    }

    fn gen_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let { value, .. } => {
                // Evaluate value — result stays on stack as the "variable"
                self.gen_expr(value)?;
            }
            Stmt::Send { value, channel } => {
                self.gen_expr(value)?;
                self.emit(j1::encode(Op::Lit, channel_id(channel)));
                self.emit(j1::encode_simple(Op::ChSend));
            }
            Stmt::Recv { channel, .. } => {
                self.emit(j1::encode(Op::Lit, channel_id(channel)));
                self.emit(j1::encode_simple(Op::ChRecv));
                // Result is now on TOS as the bound variable
            }
            Stmt::Expr(expr) => {
                self.gen_expr(expr)?;
                self.emit(j1::encode_simple(Op::Drop)); // discard result
            }
        }
        Ok(())
    }

    fn gen_expr(&mut self, expr: &Expr) -> Result<(), String> {
        match expr {
            Expr::Number(n) => {
                self.emit(j1::encode(Op::Lit, *n));
            }
            Expr::Ident(_name) => {
                // In a full compiler, we'd look up the stack offset.
                // For this sketch, identifiers reference values already on stack.
                // A real implementation would emit DUP + stack-relative pick.
                self.emit(j1::encode(Op::Dup, 0));
            }
            Expr::Group => {
                // Group literal — emit the bitmask (hardcoded 0x3 = two cores)
                self.emit(j1::encode(Op::Lit, 0x3));
            }
            Expr::Binary { left, op, right } => {
                self.gen_expr(left)?;
                self.gen_expr(right)?;
                let alu = match op {
                    BinOp::Add => Op::Add,
                    BinOp::Sub => Op::Sub,
                    BinOp::Mul => Op::Mul,
                    BinOp::Eq  => Op::Eq,
                    BinOp::Lt  => Op::Lt,
                    BinOp::Gt  => Op::Gt,
                };
                self.emit(j1::encode_simple(alu));
            }
            Expr::If { cond, then_block, else_block } => {
                self.gen_expr(cond)?;
                let bz_addr = self.here();
                self.emit(j1::encode(Op::Bz, 0)); // placeholder
                self.gen_block(then_block)?;
                let jmp_addr = self.here();
                self.emit(j1::encode(Op::Jmp, 0)); // placeholder
                let else_start = self.here();
                self.patch(bz_addr, else_start);
                self.gen_block(else_block)?;
                let end = self.here();
                self.patch(jmp_addr, end);
            }
            Expr::Loop { count, body } => {
                self.emit(j1::encode(Op::Lit, *count));
                self.emit(j1::encode_simple(Op::ToR));
                let loop_top = self.here();
                self.gen_block(body)?;
                self.emit(j1::encode_simple(Op::FromR));
                self.emit(j1::encode(Op::Lit, 1));
                self.emit(j1::encode_simple(Op::Sub));
                self.emit(j1::encode_simple(Op::Dup));
                self.emit(j1::encode_simple(Op::ToR));
                // BNZ = branch if NOT zero. We encode as: DUP, BZ(skip), JMP(loop_top)
                // But simpler: just use BZ to exit, JMP to loop.
                // After SUB+DUP+>R, TOS has the counter copy.
                let bz_addr = self.here();
                self.emit(j1::encode(Op::Bz, 0)); // exit if zero
                self.emit(j1::encode(Op::Jmp, loop_top as u32));
                let exit = self.here();
                self.patch(bz_addr, exit);
                self.emit(j1::encode_simple(Op::FromR));
                self.emit(j1::encode_simple(Op::Drop));
            }
            Expr::Merge { .. } => {
                // merge generates ZERO runtime code.
                // Protocol checking happens in protocol.rs, not here.
            }
            Expr::Call { name: _, args } => {
                // Push args, then CALL. We don't resolve addresses here —
                // a linker pass would fill in the CALL target.
                for arg in args {
                    self.gen_expr(arg)?;
                }
                self.emit(j1::encode(Op::Call, 0)); // placeholder address
            }
            Expr::Return(value) => {
                self.gen_expr(value)?;
                self.emit(j1::encode_simple(Op::Ret));
            }
        }
        Ok(())
    }
}
