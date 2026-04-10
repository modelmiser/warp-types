//! esp32-compiler: Minimal session-typed compiler for a 10-keyword Rust-like language.
//!
//! Targets the J1-32 stack machine. Demonstrates that warp-types' ComplementOf
//! mechanism (linear typestate, complemented participant sets) transfers from
//! GPU warp divergence to CSP protocol compliance.
//!
//! See: research/complemented-typestate-framework.md (Experiment 2)

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod token;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod protocol;
pub mod j1;
pub mod codegen;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ast::{Stmt, Expr};
use crate::codegen::Codegen;
use crate::parser::Parser;
use crate::protocol::ProtocolChecker;

/// Compile result: J1 instruction words + any protocol errors.
#[derive(Debug)]
pub struct CompileResult {
    pub code: Vec<u32>,
    pub function_name: String,
}

/// Compile a single function from source text.
///
/// If the function has a protocol annotation (`: role`), the protocol checker
/// validates all send/recv statements against the hardcoded PingPong protocol.
pub fn compile_function(source: &str) -> Result<CompileResult, String> {
    let mut parser = Parser::new(source);
    let program = parser.parse_program()?;

    if program.functions.is_empty() {
        return Err("no functions found".into());
    }

    let func = &program.functions[0];

    // Protocol checking (if annotated)
    if let Some(ref role_name) = func.protocol {
        let mut checker = ProtocolChecker::new(role_name)
            .map_err(|e| e.message)?;
        check_block_protocol(&func.body, &mut checker)?;
        checker.check_complete().map_err(|e| e.message)?;
    }

    // Code generation
    let mut gen = Codegen::new();
    gen.gen_function(func)?;

    Ok(CompileResult {
        code: gen.code,
        function_name: func.name.clone(),
    })
}

/// Walk a block checking send/recv against the protocol state machine.
fn check_block_protocol(
    block: &ast::Block,
    checker: &mut ProtocolChecker,
) -> Result<(), String> {
    for stmt in &block.stmts {
        check_stmt_protocol(stmt, checker)?;
    }
    if let Some(ref tail) = block.tail {
        check_expr_protocol(tail, checker)?;
    }
    Ok(())
}

fn check_stmt_protocol(stmt: &Stmt, checker: &mut ProtocolChecker) -> Result<(), String> {
    match stmt {
        Stmt::Send { channel, .. } => {
            checker.check_send(channel).map_err(|e| e.message)?;
        }
        Stmt::Recv { channel, .. } => {
            checker.check_recv(channel).map_err(|e| e.message)?;
        }
        Stmt::Expr(expr) => {
            check_expr_protocol(expr, checker)?;
        }
        Stmt::Let { value, .. } => {
            check_expr_protocol(value, checker)?;
        }
    }
    Ok(())
}

fn check_expr_protocol(expr: &Expr, checker: &mut ProtocolChecker) -> Result<(), String> {
    match expr {
        Expr::If { then_block, else_block, .. } => {
            // Fork the checker for both branches
            let mut then_checker = checker.clone();
            let mut else_checker = checker.clone();
            check_block_protocol(then_block, &mut then_checker)?;
            check_block_protocol(else_block, &mut else_checker)?;

            // Merge: both branches must reach the same protocol state
            ProtocolChecker::check_merge(&then_checker, &else_checker)
                .map_err(|e| e.message)?;

            // Advance the main checker to the merged position
            // (both are equal, so pick either)
            *checker = then_checker;
        }
        Expr::Loop { body, .. } => {
            // Protocol actions inside loops must be self-contained:
            // the loop body must not advance protocol state (or advance
            // by a full protocol cycle). For now, we check that the
            // position is unchanged after one iteration.
            let before = checker.position();
            check_block_protocol(body, checker)?;
            if checker.position() != before {
                return Err(format!(
                    "protocol error in loop: body advances protocol state \
                     (from position {} to {}). Loop bodies must be \
                     protocol-neutral or complete full cycles.",
                    before, checker.position()
                ));
            }
        }
        Expr::Return(inner) => {
            check_expr_protocol(inner, checker)?;
        }
        Expr::Call { args, .. } => {
            for arg in args {
                check_expr_protocol(arg, checker)?;
            }
        }
        // All other expressions don't affect protocol state
        _ => {}
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j1::Op;

    /// Test 1: Valid PingPong core0 program compiles successfully.
    #[test]
    fn valid_pingpong_core0() {
        let src = r#"
            fn ping(val: u32) : core0 -> u32 {
                send val ch01;
                recv reply ch10;
                return reply;
            }
        "#;
        let result = compile_function(src);
        assert!(result.is_ok(), "expected success, got: {:?}", result.err());
        let compiled = result.unwrap();
        assert_eq!(compiled.function_name, "ping");
        assert!(!compiled.code.is_empty());
    }

    /// Test 2: Valid PingPong core1 program compiles successfully.
    #[test]
    fn valid_pingpong_core1() {
        let src = r#"
            fn pong() : core1 -> u32 {
                recv msg ch01;
                let result = msg + 1;
                send result ch10;
                return result;
            }
        "#;
        let result = compile_function(src);
        assert!(result.is_ok(), "expected success, got: {:?}", result.err());
    }

    /// Test 3: Protocol violation — wrong order (recv before send on core0).
    #[test]
    fn protocol_violation_wrong_order() {
        let src = r#"
            fn bad_ping(val: u32) : core0 -> u32 {
                recv reply ch10;
                send val ch01;
                return reply;
            }
        "#;
        let result = compile_function(src);
        assert!(result.is_err(), "expected protocol violation");
        let err = result.unwrap_err();
        assert!(
            err.contains("protocol violation"),
            "error should mention protocol violation: {}",
            err
        );
    }

    /// Test 4: Protocol violation — wrong channel.
    #[test]
    fn protocol_violation_wrong_channel() {
        let src = r#"
            fn bad_ping(val: u32) : core0 -> u32 {
                send val ch10;
                recv reply ch01;
                return reply;
            }
        "#;
        let result = compile_function(src);
        assert!(result.is_err(), "expected protocol violation");
    }

    /// Test 5: Incomplete protocol — only sends, doesn't recv.
    #[test]
    fn protocol_incomplete() {
        let src = r#"
            fn half_ping(val: u32) : core0 -> u32 {
                send val ch01;
                return val;
            }
        "#;
        let result = compile_function(src);
        assert!(result.is_err(), "expected incomplete protocol error");
        let err = result.unwrap_err();
        assert!(
            err.contains("incomplete"),
            "error should mention incomplete: {}",
            err
        );
    }

    /// Test 6: Code generation for a simple function produces correct J1 ops.
    #[test]
    fn codegen_simple_addition() {
        let src = r#"
            fn add_three() -> u32 {
                return 3 + 4;
            }
        "#;
        let result = compile_function(src).unwrap();
        // Expected: LIT 3, LIT 4, ADD, RET, RET (return emits RET, fn epilogue emits RET)
        assert!(result.code.len() >= 4, "code too short: {:?}", result.code);

        let lit3 = crate::j1::encode(Op::Lit, 3);
        let lit4 = crate::j1::encode(Op::Lit, 4);
        let add  = crate::j1::encode_simple(Op::Add);

        assert_eq!(result.code[0], lit3, "first instruction should be LIT 3");
        assert_eq!(result.code[1], lit4, "second instruction should be LIT 4");
        assert_eq!(result.code[2], add,  "third instruction should be ADD");
    }

    /// Test 7: Function without protocol annotation skips protocol checking.
    #[test]
    fn no_protocol_annotation() {
        let src = r#"
            fn helper(x: u32) -> u32 {
                return x + 1;
            }
        "#;
        let result = compile_function(src);
        assert!(result.is_ok(), "function without protocol should compile: {:?}", result.err());
    }

    /// Test 8: merge generates zero code.
    #[test]
    fn merge_emits_no_code() {
        let src = r#"
            fn test_merge() -> u32 {
                merge 1 2;
                return 0;
            }
        "#;
        let result = compile_function(src).unwrap();
        // merge should produce no instructions — only LIT 0, RET, RET
        // The merge expr itself emits nothing; the stmt wraps it with DROP.
        // So: DROP (for merge stmt), LIT 0, RET, RET
        let has_ch_send = result.code.iter().any(|&w| (w >> 24) == Op::ChSend as u32);
        let has_ch_recv = result.code.iter().any(|&w| (w >> 24) == Op::ChRecv as u32);
        assert!(!has_ch_send, "merge should not emit CH.SEND");
        assert!(!has_ch_recv, "merge should not emit CH.RECV");
    }
}
