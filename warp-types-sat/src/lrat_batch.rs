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

/// Carve `lens` consecutive sub-slices off the front of `s`.
fn carve<'a, T>(mut s: &'a mut [T], lens: impl Iterator<Item = usize>) -> Vec<&'a mut [T]> {
    let mut out = Vec::new();
    for l in lens {
        let (a, b) = s.split_at_mut(l);
        out.push(a);
        s = b;
    }
    out
}

/// One step's destination regions in the batch arrays.
struct StepSlices<'a> {
    lits_priv: &'a mut [i32],
    lits_lemma: &'a mut [i32],
    cs_priv: &'a mut [u32],
    cs_lemma: &'a mut [u32],
    del_priv: &'a mut [u32],
    del_lemma: &'a mut [u32],
    lemma_clause: &'a mut [u32],
    lemma_step: &'a mut [u32],
    lemma_id: &'a mut [u64],
    hint_start: &'a mut [u32],
    hints: &'a mut [u32],
}

/// Per-step offsets in the batch, computed from sizes alone.
#[derive(Clone, Copy)]
struct Off {
    n_priv: usize,
    priv_lits_len: usize,
    n_lemmas: usize,
    lemma_lits_len: usize,
    hints_len: usize,
    priv_dense: u32,
    priv_lits: usize,
    lemma_dense: u32,
    lemma_lits: usize,
    lemma_ix: usize,
    hints: usize,
}

/// [`batch`] with the copy done on `threads` workers, each filling whole
/// steps into pre-sized arrays (step regions are disjoint, so the fill needs
/// no synchronisation). Produces exactly the arrays [`batch`] does (tested);
/// the single-threaded one stays as the reference. Pass 1 (sizes and
/// offsets) is sequential and reads no payload; the zero-fill of the arrays
/// is the price of staying in safe code.
pub fn batch_par(cnf: &[Vec<i32>], steps: &[FlatProof], threads: usize) -> Batch {
    let n_cnf = cnf.len();
    let cnf_lits: usize = cnf.iter().map(Vec::len).sum();
    // Pass 1: sizes, then offsets.
    let mut offs: Vec<Off> = Vec::with_capacity(steps.len());
    let (mut priv_dense, mut priv_lits) = (n_cnf as u32, cnf_lits);
    for (j, fp) in steps.iter().enumerate() {
        assert!(
            fp.n_original as usize >= n_cnf,
            "step {j}: n_original < shared CNF"
        );
        assert_eq!(
            fp.hint_start.len(),
            fp.lemma_clause.len() + 1,
            "step {j}: hint_start"
        );
        assert_eq!(
            fp.hints.len(),
            *fp.hint_start.last().unwrap() as usize,
            "step {j}: hints on host"
        );
        for (d, c) in cnf.iter().enumerate() {
            let (s, e) = (fp.clause_start[d] as usize, fp.clause_start[d + 1] as usize);
            assert!(
                s == e || &fp.lits[s..e] == c.as_slice(),
                "step {j}: dense {d} is neither the CNF clause nor a placeholder"
            );
        }
        let n_orig = fp.n_original as usize;
        let extras_lits = fp.clause_start[n_orig] as usize - fp.clause_start[n_cnf] as usize;
        let (mut n_del, mut del_lits) = (0usize, 0usize);
        for (d, c) in cnf.iter().enumerate() {
            if fp.deleted_at[d] != u32::MAX {
                n_del += 1;
                del_lits += c.len();
            }
        }
        let o = Off {
            n_priv: n_orig - n_cnf + n_del,
            priv_lits_len: extras_lits + del_lits,
            n_lemmas: fp.lemma_clause.len(),
            lemma_lits_len: fp.lits.len() - fp.clause_start[n_orig] as usize,
            hints_len: fp.hints.len(),
            priv_dense,
            priv_lits,
            lemma_dense: 0,
            lemma_lits: 0,
            lemma_ix: 0,
            hints: 0,
        };
        priv_dense += o.n_priv as u32;
        priv_lits += o.priv_lits_len;
        offs.push(o);
    }
    let n_original = priv_dense;
    let (mut lemma_dense, mut lemma_lits, mut lemma_ix, mut hints) =
        (n_original, priv_lits, 0usize, 0usize);
    for o in offs.iter_mut() {
        o.lemma_dense = lemma_dense;
        o.lemma_lits = lemma_lits;
        o.lemma_ix = lemma_ix;
        o.hints = hints;
        lemma_dense += o.n_lemmas as u32;
        lemma_lits += o.lemma_lits_len;
        lemma_ix += o.n_lemmas;
        hints += o.hints_len;
    }
    let n_clauses = lemma_dense as usize;
    let mut flat = FlatProof {
        lits: vec![0; lemma_lits],
        clause_start: vec![0; n_clauses + 1],
        deleted_at: vec![u32::MAX; n_clauses],
        lemma_clause: vec![0; lemma_ix],
        lemma_step: vec![0; lemma_ix],
        lemma_id: vec![0; lemma_ix],
        hint_start: vec![0; lemma_ix + 1],
        hints: vec![0; hints],
        first_empty: None,
        n_original,
        early: None,
    };
    // Shared CNF.
    {
        let mut at = 0usize;
        for (d, c) in cnf.iter().enumerate() {
            flat.lits[at..at + c.len()].copy_from_slice(c);
            at += c.len();
            flat.clause_start[d + 1] = at as u32;
        }
    }
    // Carve every step's regions (disjoint by construction).
    let (lits_priv_all, lits_lemma_all) = flat.lits[cnf_lits..].split_at_mut(priv_lits - cnf_lits);
    let (cs_priv_all, cs_lemma_all) =
        flat.clause_start[n_cnf + 1..].split_at_mut(n_original as usize - n_cnf);
    let (del_priv_all, del_lemma_all) =
        flat.deleted_at[n_cnf..].split_at_mut(n_original as usize - n_cnf);
    let mut lits_priv = carve(lits_priv_all, offs.iter().map(|o| o.priv_lits_len)).into_iter();
    let mut lits_lemma = carve(lits_lemma_all, offs.iter().map(|o| o.lemma_lits_len)).into_iter();
    let mut cs_priv = carve(cs_priv_all, offs.iter().map(|o| o.n_priv)).into_iter();
    let mut cs_lemma = carve(cs_lemma_all, offs.iter().map(|o| o.n_lemmas)).into_iter();
    let mut del_priv = carve(del_priv_all, offs.iter().map(|o| o.n_priv)).into_iter();
    let mut del_lemma = carve(del_lemma_all, offs.iter().map(|o| o.n_lemmas)).into_iter();
    let mut lemma_clause =
        carve(&mut flat.lemma_clause[..], offs.iter().map(|o| o.n_lemmas)).into_iter();
    let mut lemma_step =
        carve(&mut flat.lemma_step[..], offs.iter().map(|o| o.n_lemmas)).into_iter();
    let mut lemma_id = carve(&mut flat.lemma_id[..], offs.iter().map(|o| o.n_lemmas)).into_iter();
    let mut hint_start =
        carve(&mut flat.hint_start[1..], offs.iter().map(|o| o.n_lemmas)).into_iter();
    let mut hints_sl = carve(&mut flat.hints[..], offs.iter().map(|o| o.hints_len)).into_iter();
    let work: Vec<(usize, StepSlices)> = (0..steps.len())
        .map(|j| {
            (
                j,
                StepSlices {
                    lits_priv: lits_priv.next().unwrap(),
                    lits_lemma: lits_lemma.next().unwrap(),
                    cs_priv: cs_priv.next().unwrap(),
                    cs_lemma: cs_lemma.next().unwrap(),
                    del_priv: del_priv.next().unwrap(),
                    del_lemma: del_lemma.next().unwrap(),
                    lemma_clause: lemma_clause.next().unwrap(),
                    lemma_step: lemma_step.next().unwrap(),
                    lemma_id: lemma_id.next().unwrap(),
                    hint_start: hint_start.next().unwrap(),
                    hints: hints_sl.next().unwrap(),
                },
            )
        })
        .collect();
    // Fill one step: private originals (extras, then copies of deleted shared
    // clauses), then its lemmas with hints remapped.
    let fill = |j: usize, mut s: StepSlices| {
        let fp = &steps[j];
        let o = offs[j];
        let n_orig = fp.n_original as usize;
        let mut remap: Vec<(u32, u32)> = Vec::new();
        let (mut at, mut k) = (0usize, 0usize);
        let mut put = |s: &mut StepSlices, d: usize, lits: &[i32]| {
            s.lits_priv[at..at + lits.len()].copy_from_slice(lits);
            at += lits.len();
            s.cs_priv[k] = (o.priv_lits + at) as u32;
            s.del_priv[k] = fp.deleted_at[d];
            remap.push((d as u32, o.priv_dense + k as u32));
            k += 1;
        };
        for d in n_cnf..n_orig {
            let (a, b) = (fp.clause_start[d] as usize, fp.clause_start[d + 1] as usize);
            put(&mut s, d, &fp.lits[a..b]);
        }
        for (d, c) in cnf.iter().enumerate() {
            if fp.deleted_at[d] != u32::MAX {
                put(&mut s, d, c);
            }
        }
        let map = |d: u32| -> u32 {
            if d as usize >= n_orig {
                o.lemma_dense + (d - fp.n_original)
            } else if (d as usize) < n_cnf && fp.deleted_at[d as usize] == u32::MAX {
                d
            } else {
                remap
                    .iter()
                    .find(|(from, _)| *from == d)
                    .map(|&(_, to)| to)
                    .unwrap()
            }
        };
        let (mut at, mut ha) = (0usize, 0usize);
        for i in 0..o.n_lemmas {
            let d = fp.lemma_clause[i] as usize;
            let (a, b) = (fp.clause_start[d] as usize, fp.clause_start[d + 1] as usize);
            s.lits_lemma[at..at + (b - a)].copy_from_slice(&fp.lits[a..b]);
            at += b - a;
            s.cs_lemma[i] = (o.lemma_lits + at) as u32;
            s.del_lemma[i] = fp.deleted_at[d];
            s.lemma_clause[i] = map(fp.lemma_clause[i]);
            s.lemma_step[i] = fp.lemma_step[i];
            s.lemma_id[i] = fp.lemma_id[i];
            let (hs, he) = (fp.hint_start[i] as usize, fp.hint_start[i + 1] as usize);
            for (dst, &h) in s.hints[ha..ha + (he - hs)]
                .iter_mut()
                .zip(&fp.hints[hs..he])
            {
                *dst = map(h);
            }
            ha += he - hs;
            s.hint_start[i] = (o.hints + ha) as u32;
        }
    };
    // Runs of roughly equal bytes, in step order, one worker each.
    let weight = |o: &Off| 4 * (o.lemma_lits_len + o.hints_len) + 24 * o.n_lemmas;
    let total: usize = offs.iter().map(weight).sum();
    let threads = threads.clamp(1, steps.len().max(1));
    let per = total / threads + 1;
    std::thread::scope(|sc| {
        let fill = &fill;
        let mut acc = 0usize;
        let mut group: Vec<(usize, StepSlices)> = Vec::new();
        for (j, s) in work {
            acc += weight(&offs[j]);
            group.push((j, s));
            if acc >= per {
                let g = std::mem::take(&mut group);
                sc.spawn(move || g.into_iter().for_each(|(j, s)| fill(j, s)));
                acc = 0;
            }
        }
        if !group.is_empty() {
            sc.spawn(move || group.into_iter().for_each(|(j, s)| fill(j, s)));
        }
    });
    let lemma_range = offs
        .iter()
        .map(|o| (o.lemma_ix, o.lemma_ix + o.n_lemmas))
        .collect();
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
        // Contract 4: the parallel layout is the sequential one, array for array.
        for t in [1usize, 2, 3, 7] {
            let pb = batch_par(&cnf(), &fps, t);
            assert_eq!(pb.lemma_range, b.lemma_range, "threads {t}");
            assert_eq!(pb.flat.n_original, b.flat.n_original, "threads {t}");
            assert_eq!(pb.flat.lits, b.flat.lits, "threads {t}");
            assert_eq!(pb.flat.clause_start, b.flat.clause_start, "threads {t}");
            assert_eq!(pb.flat.deleted_at, b.flat.deleted_at, "threads {t}");
            assert_eq!(pb.flat.lemma_clause, b.flat.lemma_clause, "threads {t}");
            assert_eq!(pb.flat.lemma_step, b.flat.lemma_step, "threads {t}");
            assert_eq!(pb.flat.lemma_id, b.flat.lemma_id, "threads {t}");
            assert_eq!(pb.flat.hint_start, b.flat.hint_start, "threads {t}");
            assert_eq!(pb.flat.hints, b.flat.hints, "threads {t}");
            let hp = batch_par(&cnf(), &holders, t);
            assert_eq!(hp.flat.lits, b.flat.lits, "placeholders, threads {t}");
            assert_eq!(hp.flat.hints, b.flat.hints, "placeholders, threads {t}");
        }
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
