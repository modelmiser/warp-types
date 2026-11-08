//! Batched compositional checking: many `(formula_j, proof_j)` pairs that
//! share one CNF, laid out as ONE [`FlatProof`] so the per-lemma check (SimWarp
//! or the device kernel) runs once over every lemma of every step.
//!
//! The shape this serves is cake_lpr's compositional proof (`research/edp-record.md`):
//! step `j`'s formula is the shared CNF plus a few private clauses (the negated
//! cube as units, or every earlier lemma for the final step), and its proof
//! derives the empty clause from that formula. Checked one at a time, 5226
//! steps mean 5226 CNF parses, uploads and launches; batched, the CNF is in
//! the clause table once and the fixed costs are paid once.
//!
//! **Soundness of sharing.** The per-lemma check reads, for lemma `i`, only
//! its hints' clauses (`lits`, `clause_start`), their `deleted_at` against the
//! lemma's own `lemma_step`, and the dense order (`hint >= lemma_clause` is a
//! forward hint). So two steps may share a clause slot exactly when the slot
//! reads the same in both: same literals, and `deleted_at = never`. A CNF
//! clause that some step deletes is therefore copied into that step's private
//! region (with its deletion step) and the shared slot stays undeleted; every
//! other clause of a step — its extras and its lemmas — is private. Step
//! indices need no offset: a hint never crosses steps, and the shared slots
//! have no deletion to compare against. Dense order is preserved within a
//! step (one monotone shift per region), so the forward-hint test is
//! unchanged.
//!
//! **Verdicts** are per step and come from the per-step [`FlatProof`]s, not
//! from the batch: the codes the check returns are split by
//! [`Batch::lemma_range`] and fed to [`combine`] with each step's own flat
//! proof, so ids, steps and reasons are the ones the sequential reference
//! reports. The contract, tested below and on every step of the EDP run:
//! `codes(batch)[range_j] == codes(step_j)` for every step.

use crate::lrat::Verdict;
use crate::lrat_warp::{combine, FlatProof, LemmaCode};

/// Many steps as one flat proof.
#[derive(Debug, Clone)]
pub struct Batch {
    /// The batched layout: shared CNF, then every step's private originals,
    /// then every step's lemmas in step order. Its `first_empty` and `early`
    /// are not meaningful (verdicts are per step — see [`combine_steps`]).
    pub flat: FlatProof,
    /// Per step: `lemma_range[j] = (lo, hi)`, the lemma indices of step `j`
    /// in `flat`, in the step's own lemma order.
    pub lemma_range: Vec<(usize, usize)>,
}

/// Lay out `steps` — each flattened against `cnf ++ extras_j`, so each has
/// `n_original >= n_cnf` and its dense indices `0..n_cnf` are the shared CNF
/// — as one flat proof. The shared region is always taken from `cnf`; a step
/// may carry **empty placeholders** for dense `0..n_cnf` (decoded against a
/// CNF of `n_cnf` empty clauses, so 5226 steps do not each copy the CNF's
/// literals), or the real clauses. Panics if a step's dense `0..n_cnf` are
/// neither (the caller built it against something else).
pub fn batch(cnf: &[Vec<i32>], steps: &[FlatProof]) -> Batch {
    let n_cnf = cnf.len();
    let mut flat = FlatProof {
        lits: Vec::new(),
        clause_start: vec![0],
        deleted_at: Vec::new(),
        lemma_clause: Vec::new(),
        lemma_step: Vec::new(),
        lemma_id: Vec::new(),
        hint_start: vec![0],
        hints: Vec::new(),
        first_empty: None,
        n_original: 0,
        early: None,
    };
    for c in cnf {
        flat.lits.extend_from_slice(c);
        flat.clause_start.push(flat.lits.len() as u32);
        flat.deleted_at.push(u32::MAX);
    }
    // Pass 1: private originals of every step (extras, then copies of the
    // CNF clauses the step deletes). `remap[j]` maps the step's dense index
    // of a private original to the batch index; shared CNF indices map to
    // themselves and are absent from it.
    let mut remap: Vec<Vec<(u32, u32)>> = Vec::with_capacity(steps.len());
    for (j, fp) in steps.iter().enumerate() {
        assert!(
            fp.n_original as usize >= n_cnf,
            "step {j}: n_original {} < shared CNF {n_cnf}",
            fp.n_original
        );
        for (d, c) in cnf.iter().enumerate() {
            let (s, e) = (fp.clause_start[d] as usize, fp.clause_start[d + 1] as usize);
            assert!(
                s == e || &fp.lits[s..e] == c.as_slice(),
                "step {j}: dense {d} is neither the shared CNF clause nor a placeholder"
            );
        }
        assert_eq!(
            fp.hint_start.len(),
            fp.lemma_clause.len() + 1,
            "step {j}: hint_start is not one per lemma plus one"
        );
        assert_eq!(
            fp.hints.len(),
            *fp.hint_start.last().unwrap() as usize,
            "step {j}: hints are not on the host (device-resident step?)"
        );
        let mut m = Vec::new();
        let mut copy = |flat: &mut FlatProof, d: usize, lits: &[i32]| {
            let dense = flat.clause_start.len() as u32 - 1;
            flat.lits.extend_from_slice(lits);
            flat.clause_start.push(flat.lits.len() as u32);
            flat.deleted_at.push(fp.deleted_at[d]);
            m.push((d as u32, dense));
        };
        for d in n_cnf..fp.n_original as usize {
            let (s, e) = (fp.clause_start[d] as usize, fp.clause_start[d + 1] as usize);
            copy(&mut flat, d, &fp.lits[s..e]);
        }
        for (d, c) in cnf.iter().enumerate() {
            if fp.deleted_at[d] != u32::MAX {
                // The deleted shared clause, from `cnf` (the step may hold a placeholder).
                copy(&mut flat, d, c);
            }
        }
        remap.push(m);
    }
    flat.n_original = flat.clause_start.len() as u32 - 1;
    // Pass 2: lemmas, step by step, hints remapped.
    let mut lemma_range = Vec::with_capacity(steps.len());
    for (fp, m) in steps.iter().zip(&remap) {
        let lo = flat.lemma_clause.len();
        let lemma_base = flat.clause_start.len() as u32 - 1; // batch dense of this step's lemma 0
        let map = |d: u32| -> u32 {
            if d >= fp.n_original {
                lemma_base + (d - fp.n_original)
            } else if (d as usize) < n_cnf && fp.deleted_at[d as usize] == u32::MAX {
                d
            } else {
                // Private originals are few per step; a linear scan beats a map.
                m.iter()
                    .find(|(from, _)| *from == d)
                    .map(|&(_, to)| to)
                    .unwrap()
            }
        };
        for i in 0..fp.lemma_clause.len() {
            let d = fp.lemma_clause[i] as usize;
            let (s, e) = (fp.clause_start[d] as usize, fp.clause_start[d + 1] as usize);
            let dense = flat.clause_start.len() as u32 - 1;
            debug_assert_eq!(dense, map(fp.lemma_clause[i]));
            flat.lits.extend_from_slice(&fp.lits[s..e]);
            flat.clause_start.push(flat.lits.len() as u32);
            flat.deleted_at.push(fp.deleted_at[d]);
            flat.lemma_clause.push(dense);
            flat.lemma_step.push(fp.lemma_step[i]);
            flat.lemma_id.push(fp.lemma_id[i]);
            let (hs, he) = (fp.hint_start[i] as usize, fp.hint_start[i + 1] as usize);
            flat.hints.extend(fp.hints[hs..he].iter().map(|&h| map(h)));
            flat.hint_start.push(flat.hints.len() as u32);
        }
        lemma_range.push((lo, flat.lemma_clause.len()));
    }
    Batch { flat, lemma_range }
}

/// Per-step verdicts from the batch's per-lemma codes: step `j`'s slice of
/// `codes` combined with step `j`'s own flat proof, exactly as the
/// one-at-a-time path does. `codes.len()` must equal the batch's lemma count.
pub fn combine_steps(steps: &[FlatProof], batch: &Batch, codes: &[LemmaCode]) -> Vec<Verdict> {
    assert_eq!(
        codes.len(),
        batch.flat.lemma_clause.len(),
        "one code per batched lemma"
    );
    steps
        .iter()
        .zip(&batch.lemma_range)
        .map(|(fp, &(lo, hi))| combine(fp, &codes[lo..hi]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lrat::{check_sequential, parse_text, Reason};
    use crate::lrat_warp::{check_warp, check_warp_codes, flatten};

    /// Same variant, step and id. The reason's hint payload is not compared:
    /// `combine` reports `NotUnit(0)` where the sequential reference names the
    /// hint — a pre-existing difference of the warp path, not of batching.
    fn same_outcome(a: &Verdict, b: &Verdict) -> bool {
        use Verdict::*;
        match (a, b) {
            (Verified { .. }, Verified { .. }) | (NoEmptyClause { .. }, NoEmptyClause { .. }) => {
                true
            }
            (
                Rejected {
                    step: s1, id: i1, ..
                },
                Rejected {
                    step: s2, id: i2, ..
                },
            )
            | (Unsupported { step: s1, id: i1 }, Unsupported { step: s2, id: i2 }) => {
                s1 == s2 && i1 == i2
            }
            _ => false,
        }
    }

    // Shared CNF: 1:(1 2) 2:(-1 2) 3:(1 -2) — satisfiable (both true).
    fn cnf() -> Vec<Vec<i32>> {
        vec![vec![1, 2], vec![-1, 2], vec![1, -2]]
    }

    /// A step: the CNF plus `extras`, and its proof text.
    fn step(extras: &[Vec<i32>], proof: &str) -> (Vec<Vec<i32>>, FlatProof) {
        let mut f = cnf();
        f.extend(extras.iter().cloned());
        let p = parse_text(proof).unwrap();
        let fp = flatten(&f, &p);
        (f, fp)
    }

    fn verdicts_match(steps: &[(Vec<Vec<i32>>, FlatProof)], proofs: &[&str]) -> Vec<Verdict> {
        let fps: Vec<FlatProof> = steps.iter().map(|(_, fp)| fp.clone()).collect();
        let b = batch(&cnf(), &fps);
        // Contract 1: the batch's codes are the per-step codes, concatenated.
        let batched = check_warp_codes(&b.flat);
        let mut expect = Vec::new();
        for fp in &fps {
            expect.extend(check_warp_codes(fp));
        }
        assert_eq!(batched, expect, "batched codes != per-step codes");
        // Contract 2: per-step verdicts equal the one-at-a-time warp path's,
        // and have the sequential reference's outcome.
        let vs = combine_steps(&fps, &b, &batched);
        for ((f, _), (v, p)) in steps.iter().zip(vs.iter().zip(proofs)) {
            let proof = parse_text(p).unwrap();
            assert_eq!(
                *v,
                check_warp(f, &proof),
                "batched verdict != per-step warp"
            );
            let seq = check_sequential(f, &proof);
            assert!(same_outcome(v, &seq), "batched {v:?} vs sequential {seq:?}");
        }
        // Contract 3: placeholder steps (empty dense 0..n_cnf) give the same codes.
        let n = cnf().len();
        let holders: Vec<FlatProof> = steps
            .iter()
            .map(|(f, _)| {
                let mut v: Vec<Vec<i32>> = vec![Vec::new(); n];
                v.extend(f[n..].iter().cloned());
                v
            })
            .zip(proofs)
            .map(|(vcnf, p)| flatten(&vcnf, &parse_text(p).unwrap()))
            .collect();
        let hb = batch(&cnf(), &holders);
        assert_eq!(
            check_warp_codes(&hb.flat),
            batched,
            "placeholder batch != real batch"
        );
        assert_eq!(hb.flat.lits, b.flat.lits);
        vs
    }

    #[test]
    fn two_cube_steps_share_the_cnf() {
        // Step A: unit ¬2 (id 4): ¬2, clause 1 gives 1, clause 2 conflicts.
        let pa = "5 0 4 1 2 0\n";
        // Step B: unit ¬1 (id 4): ¬1, clause 1 gives 2, clause 3 conflicts.
        let pb = "5 0 4 1 3 0\n";
        let steps = vec![step(&[vec![-2]], pa), step(&[vec![-1]], pb)];
        let vs = verdicts_match(&steps, &[pa, pb]);
        assert!(vs.iter().all(|v| v.is_verified()), "{vs:?}");
        let b = batch(&cnf(), &[steps[0].1.clone(), steps[1].1.clone()]);
        // Layout: 3 shared + 1 private per step, then 2 lemmas.
        assert_eq!(b.flat.n_original, 5);
        assert_eq!(b.lemma_range, vec![(0, 1), (1, 2)]);
        assert_eq!(b.flat.clause_start.len(), 8);
    }

    #[test]
    fn a_step_deleting_a_shared_clause_does_not_leak() {
        // Step C deletes CNF clause 1 then uses it: DeletedHint in C only.
        let pc = "5 d 1 0\n6 0 4 1 2 0\n";
        // Step A uses clause 1 and must still verify.
        let pa = "5 0 4 1 2 0\n";
        // Step D deletes clause 3, never uses it: verified.
        let pd = "5 d 3 0\n6 0 4 1 2 0\n";
        let steps = vec![
            step(&[vec![-2]], pc),
            step(&[vec![-2]], pa),
            step(&[vec![-2]], pd),
        ];
        let vs = verdicts_match(&steps, &[pc, pa, pd]);
        assert!(
            matches!(
                vs[0],
                Verdict::Rejected {
                    id: 6,
                    reason: Reason::DeletedHint(1),
                    ..
                }
            ),
            "{:?}",
            vs[0]
        );
        assert!(vs[1].is_verified(), "{:?}", vs[1]);
        assert!(vs[2].is_verified(), "{:?}", vs[2]);
        // The shared slot of clause 1 stays undeleted; C's copy carries the deletion.
        let b = batch(
            &cnf(),
            &steps.iter().map(|(_, fp)| fp.clone()).collect::<Vec<_>>(),
        );
        assert_eq!(b.flat.deleted_at[0], u32::MAX);
        assert_eq!(
            b.flat.deleted_at.iter().filter(|&&d| d != u32::MAX).count(),
            2
        );
    }

    #[test]
    fn mutants_fail_in_their_own_step_only() {
        let good = "5 0 4 1 2 0\n";
        let not_unit = "5 0 1 2 0\n"; // no unit: clause 1 is not unit
        let no_conflict = "5 0 4 1 0\n"; // chain ends without a conflict
        let forward = "5 0 4 6 2 0\n6 0 4 1 2 0\n"; // hint 6 names a later lemma
        let no_empty = "5 1 0 4 1 0\n"; // derives the unit 1 only (¬1, ¬2 from 4, clause 1 conflicts)
        let steps = vec![
            step(&[vec![-2]], good),
            step(&[vec![-2]], not_unit),
            step(&[vec![-2]], no_conflict),
            step(&[vec![-2]], forward),
            step(&[vec![-2]], no_empty),
        ];
        let vs = verdicts_match(&steps, &[good, not_unit, no_conflict, forward, no_empty]);
        assert!(vs[0].is_verified());
        assert!(
            matches!(
                vs[1],
                Verdict::Rejected {
                    reason: Reason::NotUnit(_),
                    ..
                }
            ),
            "{:?}",
            vs[1]
        );
        assert!(
            matches!(
                vs[2],
                Verdict::Rejected {
                    reason: Reason::NoConflict,
                    ..
                }
            ),
            "{:?}",
            vs[2]
        );
        // The forward hint names lemma 6, an empty clause; which reason the
        // checkers give for that chain is theirs to decide (they agree, per
        // contract 2) — the batch must reject the same lemma.
        assert!(
            matches!(vs[3], Verdict::Rejected { step: 0, id: 5, .. }),
            "{:?}",
            vs[3]
        );
        assert!(
            matches!(vs[4], Verdict::NoEmptyClause { .. }),
            "{:?}",
            vs[4]
        );
    }

    #[test]
    fn a_step_with_many_extras_remaps_its_hints() {
        // The final compositional step: CNF + lemmas (4..=6 here) → empty clause.
        let extras = vec![vec![-2], vec![2, -1], vec![1]];
        // ¬2 (4), then (2 ∨ ¬1) gives ¬1, then (1) conflicts.
        let p = "7 0 4 5 6 0\n";
        let steps = vec![step(&extras, p), step(&[vec![-1]], "5 0 4 1 3 0\n")];
        let vs = verdicts_match(&steps, &[p, "5 0 4 1 3 0\n"]);
        assert!(vs.iter().all(|v| v.is_verified()), "{vs:?}");
    }
}
