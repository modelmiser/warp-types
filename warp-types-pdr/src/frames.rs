//! Frame sequence for PDR.
//!
//! PDR maintains F₀, F₁, ..., Fₖ where each Fᵢ is a set of clauses
//! overapproximating states reachable at depth ≤ i. Clauses are the
//! negations of blocked cubes. The sequence is monotone: Fᵢ ⊆ Fᵢ₊₁.
//!
//! Convergence: if Fᵢ = Fᵢ₊₁ for any i ≥ 1, the property is proved
//! (Fᵢ is an inductive invariant).

use warp_types_sat::literal::Lit;

// ============================================================================
// Frame
// ============================================================================

/// A single frame in the PDR sequence.
///
/// Contains blocking clauses (negations of blocked cubes) that
/// overapproximate the set of reachable states at this depth.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Blocking clauses. Each clause is ¬cube for some blocked cube.
    clauses: Vec<Vec<Lit>>,
}

impl Frame {
    /// Create an empty frame.
    pub fn new() -> Self {
        Frame {
            clauses: Vec::new(),
        }
    }

    /// Create a frame from initial-state clauses.
    pub fn from_clauses(clauses: Vec<Vec<Lit>>) -> Self {
        Frame { clauses }
    }

    /// Add a blocking clause (deduplicating).
    pub fn add_clause(&mut self, clause: Vec<Lit>) {
        // Normalize for comparison: sort literals
        let mut sorted: Vec<u32> = clause.iter().map(|l| l.code()).collect();
        sorted.sort();

        // Check for duplicate
        let is_dup = self.clauses.iter().any(|existing| {
            let mut ex_sorted: Vec<u32> = existing.iter().map(|l| l.code()).collect();
            ex_sorted.sort();
            ex_sorted == sorted
        });

        if !is_dup {
            self.clauses.push(clause);
        }
    }

    /// Read access to clauses.
    pub fn clauses(&self) -> &[Vec<Lit>] {
        &self.clauses
    }

    /// Number of blocking clauses.
    pub fn len(&self) -> usize {
        self.clauses.len()
    }

    /// Whether the frame has no blocking clauses.
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Frame sequence
// ============================================================================

/// The sequence of frames maintained by PDR.
///
/// F₀ contains the initial-state clauses. F₁..Fₖ accumulate blocking
/// clauses as cubes are blocked during the strengthen phase.
pub struct FrameSequence {
    frames: Vec<Frame>,
}

impl FrameSequence {
    /// Create an empty frame sequence.
    pub fn new() -> Self {
        FrameSequence { frames: Vec::new() }
    }

    /// Add a frame to the sequence.
    pub fn push(&mut self, frame: Frame) {
        self.frames.push(frame);
    }

    /// Current frontier level (index of the last frame).
    pub fn frontier(&self) -> usize {
        self.frames.len().saturating_sub(1)
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Access a frame by index.
    pub fn frame(&self, level: usize) -> &Frame {
        &self.frames[level]
    }

    /// Mutable access to a frame by index.
    pub fn frame_mut(&mut self, level: usize) -> &mut Frame {
        &mut self.frames[level]
    }

    /// Add a blocking clause to a specific frame level.
    /// For monotonicity, also add to all frames at lower levels (1..=level).
    pub fn add_blocked_clause(&mut self, level: usize, clause: Vec<Lit>) {
        for i in 1..=level {
            if i < self.frames.len() {
                self.frames[i].add_clause(clause.clone());
            }
        }
    }

    /// Check convergence: if Fᵢ = Fᵢ₊₁ for any i ≥ 1, return Some(i).
    ///
    /// Compares frames by their sorted clause sets (order-independent).
    pub fn check_convergence(&self) -> Option<usize> {
        if self.frames.len() < 3 {
            return None; // Need at least F₀, F₁, F₂
        }
        (1..self.frames.len() - 1).find(|&i| frames_equal(&self.frames[i], &self.frames[i + 1]))
    }
}

impl Default for FrameSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Compare two frames for equality (order-independent clause comparison).
fn frames_equal(a: &Frame, b: &Frame) -> bool {
    if a.len() != b.len() {
        return false;
    }
    // Normalize: sort literals within each clause, then sort clauses
    let normalize = |f: &Frame| -> Vec<Vec<u32>> {
        let mut clauses: Vec<Vec<u32>> = f
            .clauses()
            .iter()
            .map(|c| {
                let mut sorted: Vec<u32> = c.iter().map(|l| l.code()).collect();
                sorted.sort();
                sorted
            })
            .collect();
        clauses.sort();
        clauses
    };
    normalize(a) == normalize(b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_add_and_read() {
        let mut frame = Frame::new();
        frame.add_clause(vec![Lit::pos(0), Lit::neg(1)]);
        assert_eq!(frame.len(), 1);
        assert_eq!(frame.clauses()[0].len(), 2);
    }

    #[test]
    fn frames_equal_order_independent() {
        let mut a = Frame::new();
        a.add_clause(vec![Lit::pos(0)]);
        a.add_clause(vec![Lit::neg(1)]);

        let mut b = Frame::new();
        b.add_clause(vec![Lit::neg(1)]);
        b.add_clause(vec![Lit::pos(0)]);

        assert!(frames_equal(&a, &b));
    }

    #[test]
    fn convergence_detected() {
        let mut seq = FrameSequence::new();
        seq.push(Frame::new()); // F₀

        let mut f1 = Frame::new();
        f1.add_clause(vec![Lit::neg(0)]);
        seq.push(f1); // F₁

        let mut f2 = Frame::new();
        f2.add_clause(vec![Lit::neg(0)]);
        seq.push(f2); // F₂ = F₁

        assert_eq!(seq.check_convergence(), Some(1));
    }

    #[test]
    fn no_convergence_when_different() {
        let mut seq = FrameSequence::new();
        seq.push(Frame::new()); // F₀

        let mut f1 = Frame::new();
        f1.add_clause(vec![Lit::neg(0)]);
        seq.push(f1);

        let f2 = Frame::new(); // F₂ empty, F₁ has a clause
        seq.push(f2);

        assert_eq!(seq.check_convergence(), None);
    }
}
