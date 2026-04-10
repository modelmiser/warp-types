//! Zero-allocation lexer for the 10-keyword language.
//!
//! Operates on `&str` slices — no heap allocation during tokenization.
//! Tokens that carry data (Number, Ident) allocate via alloc::string::String.

extern crate alloc;
use alloc::string::String;
use crate::token::{self, Token};

pub struct Lexer<'src> {
    src: &'src str,
    pos: usize,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Lexer { src, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_whitespace() {
                self.advance();
            } else if ch == '/' && self.src[self.pos..].starts_with("//") {
                // Line comment
                while let Some(c) = self.advance() {
                    if c == '\n' { break; }
                }
            } else {
                break;
            }
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let ch = match self.peek_char() {
            None => return Token::Eof,
            Some(c) => c,
        };

        // Single/double character tokens
        match ch {
            '(' => { self.advance(); return Token::LParen; }
            ')' => { self.advance(); return Token::RParen; }
            '{' => { self.advance(); return Token::LBrace; }
            '}' => { self.advance(); return Token::RBrace; }
            ',' => { self.advance(); return Token::Comma; }
            ':' => { self.advance(); return Token::Colon; }
            ';' => { self.advance(); return Token::Semi; }
            '+' => { self.advance(); return Token::Plus; }
            '*' => { self.advance(); return Token::Star; }
            '-' => {
                self.advance();
                if self.peek_char() == Some('>') {
                    self.advance();
                    return Token::Arrow;
                }
                return Token::Minus;
            }
            '=' => {
                self.advance();
                if self.peek_char() == Some('=') {
                    self.advance();
                    return Token::EqEq;
                }
                return Token::Eq;
            }
            '<' => { self.advance(); return Token::Lt; }
            '>' => { self.advance(); return Token::Gt; }
            _ => {}
        }

        // Number literal
        if ch.is_ascii_digit() {
            let start = self.pos;
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() { self.advance(); } else { break; }
            }
            let word = &self.src[start..self.pos];
            let n = word.parse::<u32>().unwrap_or(0);
            return Token::Number(n);
        }

        // Identifier or keyword
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = self.pos;
            while let Some(c) = self.peek_char() {
                if c.is_ascii_alphanumeric() || c == '_' { self.advance(); } else { break; }
            }
            let word = &self.src[start..self.pos];
            if let Some(kw) = token::keyword(word) {
                return kw;
            }
            return Token::Ident(String::from(word));
        }

        // Unknown character — skip and return Eof (simple error recovery)
        self.advance();
        self.next_token()
    }
}
