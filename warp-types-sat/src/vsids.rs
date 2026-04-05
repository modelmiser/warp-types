//! VSIDS (Variable State Independent Decaying Sum) branching heuristic.
//!
//! Variables appearing in recent conflicts get higher activity scores and are
//! decided first. Combined with phase saving (remember last polarity) to avoid
//! re-exploring refuted subtrees.
//!
//! MiniSat optimization: instead of decaying all activities by 0.95 each conflict
//! (O(n)), increase the bump increment by 1/0.95 (O(1)). Mathematically equivalent.

use crate::bcp::ClauseDb;

/// VSIDS branching heuristic with phase saving.
pub struct Vsids {
    /// Per-variable activity score.
    activity: Vec<f64>,
    /// Current bump increment (grows to implement implicit decay).
    increment: f64,
    /// Decay factor. Standard value: 0.95 (MiniSat default).
    decay_factor: f64,
    /// Saved phase (polarity) for each variable.
    phase: Vec<bool>,
}

impl Vsids {
    /// Create with all activities at zero, all phases defaulting positive.
    pub fn new(num_vars: u32) -> Self {
        Vsids {
            activity: vec![0.0; num_vars as usize],
            increment: 1.0,
            decay_factor: 0.95,
            phase: vec![true; num_vars as usize],
        }
    }

    /// Warm-start activities from clause occurrence counts.
    /// Variables appearing more often are more constrained — decide them first.
    pub fn initialize_from_clauses(&mut self, db: &ClauseDb) {
        for ci in 0..db.len() {
            for &lit in &db.clause(ci).literals {
                self.activity[lit.var() as usize] += 1.0;
            }
        }
    }

    /// Bump activity of a variable (call for each var in the learned clause).
    pub fn bump(&mut self, var: u32) {
        self.activity[var as usize] += self.increment;
        // Rescale to prevent floating-point overflow.
        if self.activity[var as usize] > 1e100 {
            for a in &mut self.activity {
                *a *= 1e-100;
            }
            self.increment *= 1e-100;
        }
    }

    /// Apply decay after each conflict (O(1) via increment scaling).
    pub fn decay(&mut self) {
        self.increment /= self.decay_factor;
    }

    /// Pick the highest-activity unassigned variable and its saved phase.
    pub fn pick(&self, assignments: &[Option<bool>]) -> (u32, bool) {
        let (var, _) = assignments
            .iter()
            .enumerate()
            .filter(|(_, a)| a.is_none())
            .max_by(|(i, _), (j, _)| {
                self.activity[*i]
                    .partial_cmp(&self.activity[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("pick called with all variables assigned");
        (var as u32, self.phase[var])
    }

    /// Save the phase (polarity) of a variable.
    pub fn save_phase(&mut self, var: u32, polarity: bool) {
        self.phase[var as usize] = polarity;
    }

    /// Set initial activity for a variable (e.g., from gradient confidence).
    pub fn set_initial_activity(&mut self, var: u32, activity: f64) {
        self.activity[var as usize] = activity;
    }

    /// Set phase hint for a variable (e.g., from gradient polarity).
    pub fn set_phase(&mut self, var: u32, polarity: bool) {
        self.phase[var as usize] = polarity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Lit;

    #[test]
    fn picks_highest_activity() {
        let mut vsids = Vsids::new(3);
        vsids.bump(2);
        vsids.bump(2);
        vsids.bump(0);

        let assignments = vec![None, None, None];
        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 2);
    }

    #[test]
    fn skips_assigned() {
        let mut vsids = Vsids::new(3);
        vsids.bump(0);
        vsids.bump(0);
        vsids.bump(1);

        let assignments = vec![Some(true), None, None];
        let (var, _) = vsids.pick(&assignments);
        assert_eq!(var, 1);
    }

    #[test]
    fn phase_saving() {
        let mut vsids = Vsids::new(3);
        vsids.save_phase(1, false);

        let assignments = vec![Some(true), None, Some(true)];
        let (var, pol) = vsids.pick(&assignments);
        assert_eq!(var, 1);
        assert!(!pol);
    }

    #[test]
    fn decay_increases_increment() {
        let mut vsids = Vsids::new(1);
        let before = vsids.increment;
        vsids.decay();
        assert!(vsids.increment > before);
    }

    #[test]
    fn overflow_protection() {
        let mut vsids = Vsids::new(2);
        for _ in 0..10000 {
            vsids.bump(0);
            vsids.decay();
        }
        assert!(vsids.activity[0].is_finite());
        assert!(vsids.activity[0] > 0.0);
    }

    #[test]
    fn clause_initialization() {
        let mut db = ClauseDb::new();
        db.add_clause(vec![Lit::pos(0), Lit::neg(1)]);
        db.add_clause(vec![Lit::pos(0), Lit::pos(2)]);
        db.add_clause(vec![Lit::neg(1), Lit::pos(2)]);

        let mut vsids = Vsids::new(3);
        vsids.initialize_from_clauses(&db);

        // var 0 appears 2x, var 1 appears 2x, var 2 appears 2x
        let assignments = vec![None, None, None];
        let (_, _) = vsids.pick(&assignments); // should not panic
    }
}
