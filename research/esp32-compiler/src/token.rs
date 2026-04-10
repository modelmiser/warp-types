//! Token types and keyword recognition for the 10-keyword language.

extern crate alloc;
use alloc::string::String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    // The 10 keywords
    Fn,
    Let,
    Send,
    Recv,
    If,
    Else,
    Loop,
    Merge,
    Group,
    Return,

    // Types
    U32,
    Bool,

    // Literals and identifiers
    Number(u32),
    Ident(String),

    // Operators
    Plus,
    Minus,
    Star,
    EqEq,   // ==
    Lt,     // <
    Gt,     // >
    Eq,     // =

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semi,
    Arrow, // ->

    Eof,
}

/// Match a word to a keyword token, or return None for identifiers.
pub fn keyword(word: &str) -> Option<Token> {
    match word {
        "fn"     => Some(Token::Fn),
        "let"    => Some(Token::Let),
        "send"   => Some(Token::Send),
        "recv"   => Some(Token::Recv),
        "if"     => Some(Token::If),
        "else"   => Some(Token::Else),
        "loop"   => Some(Token::Loop),
        "merge"  => Some(Token::Merge),
        "group"  => Some(Token::Group),
        "return" => Some(Token::Return),
        "u32"    => Some(Token::U32),
        "bool"   => Some(Token::Bool),
        _        => None,
    }
}
