//! AST node types for the 10-keyword language.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

/// Top-level: a program is a sequence of function definitions.
#[derive(Debug, Clone)]
pub struct Program {
    pub functions: Vec<FnDef>,
}

/// Function definition with optional protocol annotation.
#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub protocol: Option<String>, // e.g. "PingPong.core0"
    pub ret_type: Type,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    U32,
    Bool,
    Group,
}

/// A block is a sequence of statements with an optional trailing expression.
#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let { name: String, value: Expr },
    Send { value: Expr, channel: String },
    Recv { binding: String, channel: String },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Number(u32),
    Ident(String),
    Group,
    Binary { left: Box<Expr>, op: BinOp, right: Box<Expr> },
    If { cond: Box<Expr>, then_block: Block, else_block: Block },
    Loop { count: u32, body: Block },
    Merge { left: Box<Expr>, right: Box<Expr> },
    Call { name: String, args: Vec<Expr> },
    Return(Box<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Eq,
    Lt,
    Gt,
}
