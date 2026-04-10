//! Recursive descent parser for the 10-keyword language.
//!
//! Produces an AST from a token stream. Error handling is simple:
//! Result<T, String> with descriptive messages.

extern crate alloc;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ast::*;
use crate::lexer::Lexer;
use crate::token::Token;

pub struct Parser<'src> {
    lexer: Lexer<'src>,
    current: Token,
}

type R<T> = Result<T, String>;

impl<'src> Parser<'src> {
    pub fn new(src: &'src str) -> Self {
        let mut lexer = Lexer::new(src);
        let current = lexer.next_token();
        Parser { lexer, current }
    }

    fn bump(&mut self) -> Token {
        let old = core::mem::replace(&mut self.current, self.lexer.next_token());
        old
    }

    fn expect(&mut self, expected: &Token) -> R<()> {
        if &self.current == expected {
            self.bump();
            Ok(())
        } else {
            Err(format!("expected {:?}, got {:?}", expected, self.current))
        }
    }

    fn expect_ident(&mut self) -> R<String> {
        match self.bump() {
            Token::Ident(s) => Ok(s),
            other => Err(format!("expected identifier, got {:?}", other)),
        }
    }

    // ── Top-level ──────────────────────────────────────────────

    pub fn parse_program(&mut self) -> R<Program> {
        let mut functions = Vec::new();
        while self.current != Token::Eof {
            functions.push(self.parse_fn_def()?);
        }
        Ok(Program { functions })
    }

    fn parse_fn_def(&mut self) -> R<FnDef> {
        self.expect(&Token::Fn)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;

        // Optional protocol annotation: ':' IDENT
        let protocol = if self.current == Token::Colon {
            self.bump();
            Some(self.expect_ident()?)
        } else {
            None
        };

        self.expect(&Token::Arrow)?;
        let ret_type = self.parse_type()?;
        let body = self.parse_block()?;
        Ok(FnDef { name, params, protocol, ret_type, body })
    }

    fn parse_params(&mut self) -> R<Vec<Param>> {
        let mut params = Vec::new();
        if self.current == Token::RParen { return Ok(params); }
        params.push(self.parse_param()?);
        while self.current == Token::Comma {
            self.bump();
            params.push(self.parse_param()?);
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> R<Param> {
        let name = self.expect_ident()?;
        self.expect(&Token::Colon)?;
        let ty = self.parse_type()?;
        Ok(Param { name, ty })
    }

    fn parse_type(&mut self) -> R<Type> {
        match &self.current {
            Token::U32   => { self.bump(); Ok(Type::U32) }
            Token::Bool  => { self.bump(); Ok(Type::Bool) }
            Token::Group => { self.bump(); Ok(Type::Group) }
            other => Err(format!("expected type, got {:?}", other)),
        }
    }

    // ── Block ──────────────────────────────────────────────────

    fn parse_block(&mut self) -> R<Block> {
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        let mut tail = None;

        while self.current != Token::RBrace && self.current != Token::Eof {
            // Try to parse a statement; if the token starts an expression
            // and there's no semicolon, it's the tail expression.
            match &self.current {
                Token::Let => stmts.push(self.parse_let_stmt()?),
                Token::Send => stmts.push(self.parse_send_stmt()?),
                Token::Recv => stmts.push(self.parse_recv_stmt()?),
                _ => {
                    let expr = self.parse_expr()?;
                    if self.current == Token::Semi {
                        self.bump();
                        stmts.push(Stmt::Expr(expr));
                    } else {
                        // Tail expression (no semicolon before '}')
                        tail = Some(Box::new(expr));
                        break;
                    }
                }
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(Block { stmts, tail })
    }

    // ── Statements ─────────────────────────────────────────────

    fn parse_let_stmt(&mut self) -> R<Stmt> {
        self.expect(&Token::Let)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        self.expect(&Token::Semi)?;
        Ok(Stmt::Let { name, value })
    }

    fn parse_send_stmt(&mut self) -> R<Stmt> {
        self.expect(&Token::Send)?;
        let value = self.parse_atom()?;
        let channel = self.expect_ident()?;
        self.expect(&Token::Semi)?;
        Ok(Stmt::Send { value, channel })
    }

    fn parse_recv_stmt(&mut self) -> R<Stmt> {
        self.expect(&Token::Recv)?;
        let binding = self.expect_ident()?;
        let channel = self.expect_ident()?;
        self.expect(&Token::Semi)?;
        Ok(Stmt::Recv { binding, channel })
    }

    // ── Expressions ────────────────────────────────────────────

    fn parse_expr(&mut self) -> R<Expr> {
        match &self.current {
            Token::If     => self.parse_if_expr(),
            Token::Loop   => self.parse_loop_expr(),
            Token::Merge  => self.parse_merge_expr(),
            Token::Return => self.parse_return_expr(),
            _ => self.parse_binary_or_call(),
        }
    }

    fn parse_if_expr(&mut self) -> R<Expr> {
        self.expect(&Token::If)?;
        let cond = Box::new(self.parse_atom()?);
        let then_block = self.parse_block()?;
        self.expect(&Token::Else)?;
        let else_block = self.parse_block()?;
        Ok(Expr::If { cond, then_block, else_block })
    }

    fn parse_loop_expr(&mut self) -> R<Expr> {
        self.expect(&Token::Loop)?;
        let count = match self.bump() {
            Token::Number(n) => n,
            other => return Err(format!("expected loop count, got {:?}", other)),
        };
        let body = self.parse_block()?;
        Ok(Expr::Loop { count, body })
    }

    fn parse_merge_expr(&mut self) -> R<Expr> {
        self.expect(&Token::Merge)?;
        let left = Box::new(self.parse_atom()?);
        let right = Box::new(self.parse_atom()?);
        Ok(Expr::Merge { left, right })
    }

    fn parse_return_expr(&mut self) -> R<Expr> {
        self.expect(&Token::Return)?;
        let value = Box::new(self.parse_expr()?);
        Ok(Expr::Return(value))
    }

    fn parse_binary_or_call(&mut self) -> R<Expr> {
        let left = self.parse_atom()?;

        // Check for binary operator
        let op = match &self.current {
            Token::Plus  => Some(BinOp::Add),
            Token::Minus => Some(BinOp::Sub),
            Token::Star  => Some(BinOp::Mul),
            Token::EqEq  => Some(BinOp::Eq),
            Token::Lt    => Some(BinOp::Lt),
            Token::Gt    => Some(BinOp::Gt),
            _ => None,
        };

        if let Some(op) = op {
            self.bump();
            let right = self.parse_atom()?;
            return Ok(Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });
        }

        Ok(left)
    }

    fn parse_atom(&mut self) -> R<Expr> {
        match &self.current {
            Token::Number(_) => {
                if let Token::Number(n) = self.bump() {
                    Ok(Expr::Number(n))
                } else {
                    unreachable!()
                }
            }
            Token::Group => {
                self.bump();
                Ok(Expr::Group)
            }
            Token::LParen => {
                self.bump();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::Ident(_) => {
                let name = self.expect_ident()?;
                // Check for function call
                if self.current == Token::LParen {
                    self.bump();
                    let args = self.parse_args()?;
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call { name, args })
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            other => Err(format!("expected expression, got {:?}", other)),
        }
    }

    fn parse_args(&mut self) -> R<Vec<Expr>> {
        let mut args = Vec::new();
        if self.current == Token::RParen { return Ok(args); }
        args.push(self.parse_expr()?);
        while self.current == Token::Comma {
            self.bump();
            args.push(self.parse_expr()?);
        }
        Ok(args)
    }
}
