//! Hardcoded PingPong protocol and session-type checker.
//!
//! The protocol is a state machine: each role (core0, core1) has a sequence
//! of send/recv actions that must be followed exactly. The checker advances
//! state on each send/recv statement and errors on protocol violation.
//!
//! `merge` is checked here: both branches of an `if` must reach the same
//! protocol state for reconvergence to be valid.

extern crate alloc;
use alloc::format;
use alloc::string::String;

/// A single protocol action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Send { channel: String },
    Recv { channel: String },
}

/// Protocol role — the expected sequence of actions for one participant.
#[derive(Debug, Clone)]
pub struct Role {
    pub name: String,
    pub actions: alloc::vec::Vec<Action>,
}

/// Protocol checker state machine.
#[derive(Debug, Clone)]
pub struct ProtocolChecker {
    role: Role,
    position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolError {
    pub message: String,
}

impl ProtocolChecker {
    /// Create a checker for a named role in the hardcoded PingPong protocol.
    pub fn new(role_name: &str) -> Result<Self, ProtocolError> {
        let role = lookup_role(role_name)?;
        Ok(ProtocolChecker { role, position: 0 })
    }

    /// Current position in the protocol (for branching/merging).
    pub fn position(&self) -> usize {
        self.position
    }

    /// Check and advance on a send action.
    pub fn check_send(&mut self, channel: &str) -> Result<(), ProtocolError> {
        let expected = self.next_action()?;
        match &expected {
            Action::Send { channel: ch } if ch == channel => {
                self.position += 1;
                Ok(())
            }
            _ => Err(ProtocolError {
                message: format!(
                    "protocol violation: expected {:?}, got send on '{}'",
                    expected, channel
                ),
            }),
        }
    }

    /// Check and advance on a recv action.
    pub fn check_recv(&mut self, channel: &str) -> Result<(), ProtocolError> {
        let expected = self.next_action()?;
        match &expected {
            Action::Recv { channel: ch } if ch == channel => {
                self.position += 1;
                Ok(())
            }
            _ => Err(ProtocolError {
                message: format!(
                    "protocol violation: expected {:?}, got recv on '{}'",
                    expected, channel
                ),
            }),
        }
    }

    /// Verify both branches of an `if` reached the same protocol state.
    /// This is what `merge` checks at compile time — zero runtime code.
    pub fn check_merge(
        then_state: &ProtocolChecker,
        else_state: &ProtocolChecker,
    ) -> Result<(), ProtocolError> {
        if then_state.position != else_state.position {
            Err(ProtocolError {
                message: format!(
                    "merge error: then-branch at protocol position {}, \
                     else-branch at position {} — branches must reach \
                     the same protocol state to reconverge",
                    then_state.position, else_state.position
                ),
            })
        } else {
            Ok(())
        }
    }

    /// Verify the protocol is fully consumed at function end.
    pub fn check_complete(&self) -> Result<(), ProtocolError> {
        if self.position < self.role.actions.len() {
            Err(ProtocolError {
                message: format!(
                    "protocol incomplete: {} actions remaining (at position {}/{})",
                    self.role.actions.len() - self.position,
                    self.position,
                    self.role.actions.len()
                ),
            })
        } else {
            Ok(())
        }
    }

    fn next_action(&self) -> Result<Action, ProtocolError> {
        if self.position >= self.role.actions.len() {
            Err(ProtocolError {
                message: format!(
                    "protocol exhausted: no more actions expected (at position {})",
                    self.position
                ),
            })
        } else {
            Ok(self.role.actions[self.position].clone())
        }
    }
}

// ── Hardcoded PingPong Protocol ────────────────────────────────

fn lookup_role(name: &str) -> Result<Role, ProtocolError> {
    use alloc::string::ToString;
    use alloc::vec;

    match name {
        "core0" => Ok(Role {
            name: "core0".into(),
            actions: vec![
                Action::Send { channel: "ch01".to_string() },
                Action::Recv { channel: "ch10".to_string() },
            ],
        }),
        "core1" => Ok(Role {
            name: "core1".into(),
            actions: vec![
                Action::Recv { channel: "ch01".to_string() },
                Action::Send { channel: "ch10".to_string() },
            ],
        }),
        _ => Err(ProtocolError {
            message: format!("unknown protocol role: '{}'", name),
        }),
    }
}
