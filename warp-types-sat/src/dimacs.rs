//! DIMACS CNF parser.
//!
//! Parses the standard DIMACS CNF format used by SAT competitions and SATLIB.
//!
//! # Format
//!
//! ```text
//! c comment line
//! p cnf <num_vars> <num_clauses>
//! 1 -2 3 0        ← clause: (x1 ∨ ¬x2 ∨ x3), terminated by 0
//! -1 2 0           ← clause: (¬x1 ∨ x2)
//! ```
//!
//! Variables are 1-indexed in DIMACS, 0-indexed internally.
//! Negative literals are negated variables.

use crate::bcp::ClauseDb;
use crate::literal::Lit;
use std::io::BufRead;

/// Error during DIMACS parsing.
#[derive(Debug)]
pub enum DimacsError {
    /// Missing or malformed "p cnf" header line.
    MissingHeader,
    /// I/O error while reading.
    Io(std::io::Error),
    /// Malformed literal (not an integer).
    BadLiteral(String),
    /// Variable index 0 in a clause (not valid in DIMACS).
    ZeroVariable,
    /// Variable index exceeds declared num_vars.
    VariableOutOfRange { var: u32, declared: u32 },
}

impl From<std::io::Error> for DimacsError {
    fn from(e: std::io::Error) -> Self {
        DimacsError::Io(e)
    }
}

impl std::fmt::Display for DimacsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimacsError::MissingHeader => write!(f, "missing 'p cnf' header"),
            DimacsError::Io(e) => write!(f, "I/O error: {e}"),
            DimacsError::BadLiteral(s) => write!(f, "bad literal: {s:?}"),
            DimacsError::ZeroVariable => write!(f, "variable 0 is not valid in DIMACS"),
            DimacsError::VariableOutOfRange { var, declared } => {
                write!(f, "variable {var} exceeds declared {declared}")
            }
        }
    }
}

/// Parsed DIMACS instance.
pub struct DimacsInstance {
    /// Number of variables declared in header.
    pub num_vars: u32,
    /// Number of clauses declared in header.
    pub num_clauses: u32,
    /// The clause database.
    pub db: ClauseDb,
}

/// Parse a DIMACS CNF instance from a reader.
///
/// Tolerant: accepts clauses beyond `num_clauses` (some generators emit extra).
/// Strict: rejects variable 0 and variables beyond `num_vars`.
pub fn parse_dimacs(reader: impl BufRead) -> Result<DimacsInstance, DimacsError> {
    let mut num_vars = 0u32;
    let mut num_clauses = 0u32;
    let mut header_seen = false;
    let mut db = ClauseDb::new();
    let mut current_clause: Vec<Lit> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('c') || trimmed.starts_with('%') {
            continue;
        }

        // Header line
        if trimmed.starts_with("p ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 && parts[1] == "cnf" {
                num_vars = parts[2]
                    .parse()
                    .map_err(|_| DimacsError::BadLiteral(parts[2].to_string()))?;
                num_clauses = parts[3]
                    .parse()
                    .map_err(|_| DimacsError::BadLiteral(parts[3].to_string()))?;
                header_seen = true;
                continue;
            }
            return Err(DimacsError::MissingHeader);
        }

        if !header_seen {
            return Err(DimacsError::MissingHeader);
        }

        // Clause data: space-separated integers, 0 terminates clause
        for token in trimmed.split_whitespace() {
            let val: i64 = token
                .parse()
                .map_err(|_| DimacsError::BadLiteral(token.to_string()))?;

            if val == 0 {
                // End of clause
                if !current_clause.is_empty() {
                    db.add_clause(std::mem::take(&mut current_clause));
                }
            } else {
                let abs_var = val.unsigned_abs() as u32;
                if abs_var == 0 {
                    return Err(DimacsError::ZeroVariable);
                }
                if abs_var > num_vars {
                    return Err(DimacsError::VariableOutOfRange {
                        var: abs_var,
                        declared: num_vars,
                    });
                }
                // DIMACS is 1-indexed, we're 0-indexed
                let var = abs_var - 1;
                let lit = if val > 0 {
                    Lit::pos(var)
                } else {
                    Lit::neg(var)
                };
                current_clause.push(lit);
            }
        }
    }

    // Handle unterminated final clause (some files omit trailing 0)
    if !current_clause.is_empty() {
        db.add_clause(current_clause);
    }

    if !header_seen {
        return Err(DimacsError::MissingHeader);
    }

    Ok(DimacsInstance {
        num_vars,
        num_clauses,
        db,
    })
}

/// Convenience: parse from a string.
pub fn parse_dimacs_str(s: &str) -> Result<DimacsInstance, DimacsError> {
    parse_dimacs(std::io::BufReader::new(s.as_bytes()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bcp::{self, BcpResult};
    use crate::session;
    use crate::trail::Trail;

    #[test]
    fn parse_simple() {
        let cnf = "\
c example
p cnf 3 2
1 -2 3 0
-1 2 0
";
        let inst = parse_dimacs_str(cnf).unwrap();
        assert_eq!(inst.num_vars, 3);
        assert_eq!(inst.num_clauses, 2);
        assert_eq!(inst.db.len(), 2);
    }

    #[test]
    fn parse_multiline_clause() {
        // Some DIMACS files split clauses across lines
        let cnf = "\
p cnf 3 1
1 -2
3 0
";
        let inst = parse_dimacs_str(cnf).unwrap();
        assert_eq!(inst.db.len(), 1);
        assert_eq!(inst.db.clause(0).literals.len(), 3);
    }

    #[test]
    fn parse_unterminated() {
        // Missing trailing 0 — should still parse
        let cnf = "p cnf 2 1\n1 -2";
        let inst = parse_dimacs_str(cnf).unwrap();
        assert_eq!(inst.db.len(), 1);
    }

    #[test]
    fn parse_comments_and_blanks() {
        let cnf = "\
c lots of comments
c and more
p cnf 2 1

1 2 0
c trailing comment
";
        let inst = parse_dimacs_str(cnf).unwrap();
        assert_eq!(inst.db.len(), 1);
    }

    #[test]
    fn bcp_on_parsed_instance() {
        let cnf = "\
p cnf 3 2
-1 2 0
-2 3 0
";
        let inst = parse_dimacs_str(cnf).unwrap();
        let mut trail = Trail::new(3);
        trail.new_decision(crate::literal::Lit::pos(0)); // x0=true

        let result = session::with_session(|s| {
            let p = s.decide().propagate();
            bcp::run_bcp(&inst.db, &mut trail, &p)
        });

        assert_eq!(result, BcpResult::Ok);
        assert_eq!(trail.value(1), Some(true));
        assert_eq!(trail.value(2), Some(true));
    }

    #[test]
    fn bcp_conflict_from_dimacs() {
        let cnf = "\
p cnf 2 2
-1 2 0
-1 -2 0
";
        let inst = parse_dimacs_str(cnf).unwrap();
        let mut trail = Trail::new(2);
        trail.new_decision(crate::literal::Lit::pos(0));

        let result = session::with_session(|s| {
            let p = s.decide().propagate();
            bcp::run_bcp(&inst.db, &mut trail, &p)
        });

        match result {
            BcpResult::Conflict { .. } => {}
            other => panic!("expected conflict, got {:?}", other),
        }
    }

    #[test]
    fn parse_error_missing_header() {
        let cnf = "1 2 0";
        assert!(parse_dimacs_str(cnf).is_err());
    }

    #[test]
    fn pigeonhole_2_1() {
        // Pigeonhole: 2 pigeons, 1 hole. UNSAT.
        // Pigeon 1 must be in hole 1: (x1)
        // Pigeon 2 must be in hole 1: (x2)
        // At most one pigeon per hole: (¬x1 ∨ ¬x2)
        let cnf = "\
p cnf 2 3
1 0
2 0
-1 -2 0
";
        let inst = parse_dimacs_str(cnf).unwrap();
        let mut trail = Trail::new(2);

        let result = session::with_session(|s| {
            let p = s.propagate();
            bcp::run_bcp(&inst.db, &mut trail, &p)
        });

        match result {
            BcpResult::Conflict { .. } => {}
            other => panic!("expected conflict on pigeonhole, got {:?}", other),
        }
    }
}
