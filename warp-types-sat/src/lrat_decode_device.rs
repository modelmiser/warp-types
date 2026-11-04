//! The device decode path (feature `gpu`): phase-1 records from
//! `GpuContext::decode_lrat_slices` become the host's `RawChunk`s — dirty or
//! candidate-less slices through the host's own `parse_run` — and go through
//! the same `stitch_chunks` as the host workers' records. Stage 2 of
//! `research/gpu-lrat-decode.md`: the flat proof still lands on the host.

use super::decode_model::host_record;
use super::{slice_bounds, stitch_chunks, stitch_plan, FlatProof, RawChunk, RawStep, StitchPlan};
use crate::gpu_launcher::{DecodeTiming, DeviceFlatProof, DeviceSlices, GpuContext, GpuError};

/// What the device path reports besides the proof.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceDecodeStats {
    pub n_slices: usize,
    pub dirty: usize,
    pub none: usize,
    pub timing: DecodeTiming,
    /// Host time to turn the records into chunks and stitch them.
    pub host: std::time::Duration,
}

/// Per-slice records, each slice's device payload bases (`None` for a slice
/// the host parsed), and how many slices were dirty / had no candidate.
type DeviceChunks = (Vec<RawChunk>, Vec<Option<(u32, u32)>>, usize, usize);

/// Build the per-slice records from the device arrays.
fn chunks_from_device(bytes: &[u8], d: &DeviceSlices) -> DeviceChunks {
    let resident = d.d_lits.is_some();
    let mut bases: Vec<Option<(u32, u32)>> = Vec::with_capacity(d.n_slices);
    let (mut dirty, mut none) = (0usize, 0usize);
    let n = d.n_slices;
    let count = |i: usize, f: usize| -> usize {
        let lo = d.offs[4 * i + f] as usize;
        let hi = if i + 1 < n {
            d.offs[4 * (i + 1) + f] as usize
        } else {
            d.totals[f] as usize
        };
        hi - lo
    };
    let mut chunks = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i * d.slice;
        let hi = ((i + 1) * d.slice).min(bytes.len());
        match d.flags[i] {
            crate::gpu_launcher::DEC_CLEAN => {
                let (s0, l0, h0, e0) = (
                    d.offs[4 * i] as usize,
                    d.offs[4 * i + 1] as usize,
                    d.offs[4 * i + 2] as usize,
                    d.offs[4 * i + 3] as usize,
                );
                let (ns, nl, nh, ne) = (count(i, 0), count(i, 1), count(i, 2), count(i, 3));
                let steps = (0..ns)
                    .map(|k| RawStep {
                        pos: d.st_pos[s0 + k],
                        id: d.st_id[s0 + k],
                        pos_after_id: d.st_after[s0 + k],
                        lit_end: d.st_lit_end[s0 + k],
                        hint_end: d.st_hint_end[s0 + k],
                        del_end: d.st_del_end[s0 + k],
                        tag: d.st_tag[s0 + k],
                        rat: d.st_rat[s0 + k] != 0,
                    })
                    .collect();
                bases.push(if resident {
                    Some((l0 as u32, h0 as u32))
                } else {
                    None
                });
                chunks.push(RawChunk {
                    steps,
                    lits: if resident {
                        Vec::new()
                    } else {
                        d.lits[l0..l0 + nl].to_vec()
                    },
                    hints: if resident {
                        Vec::new()
                    } else {
                        d.hints[h0..h0 + nh].to_vec()
                    },
                    dels: d.del_id[e0..e0 + ne]
                        .iter()
                        .zip(&d.del_pos[e0..e0 + ne])
                        .map(|(&id, &p)| (id, p))
                        .collect(),
                    error: None,
                    end: d.ends[i],
                    partial_add: None,
                    partial_del_lo: 0,
                });
            }
            crate::gpu_launcher::DEC_NONE => {
                none += 1;
                bases.push(None);
                chunks.push(RawChunk {
                    end: lo as u64,
                    ..Default::default()
                });
            }
            _ => {
                dirty += 1;
                bases.push(None);
                chunks.push(host_record(bytes, lo, hi));
            }
        }
    }
    (chunks, bases, dirty, none)
}

impl FlatProof {
    /// [`FlatProof::from_binary`] with phase 1 on the device: slices of
    /// `slice` bytes, the warp kernel when `cooperative` (else the
    /// thread-per-slice control), `blocks` warps striding over the slices.
    pub fn from_binary_device(
        ctx: &GpuContext,
        cnf: &[Vec<i32>],
        bytes: &[u8],
        slice: u32,
        cooperative: bool,
        blocks: u32,
    ) -> Result<(FlatProof, DeviceDecodeStats), DeviceDecodeError> {
        if bytes.is_empty() {
            let fp = FlatProof::from_binary(cnf, bytes).map_err(DeviceDecodeError::Parse)?;
            return Ok((fp, DeviceDecodeStats::default()));
        }
        let d = ctx
            .decode_lrat_slices(bytes, slice, cooperative, blocks, false)
            .map_err(DeviceDecodeError::Gpu)?;
        let t = std::time::Instant::now();
        let bounds = slice_bounds(bytes.len(), slice as usize);
        debug_assert_eq!(bounds.len(), d.n_slices);
        let (chunks, _bases, dirty, none) = chunks_from_device(bytes, &d);
        let fp = stitch_chunks(cnf, bytes, &bounds, chunks, 4).map_err(DeviceDecodeError::Parse)?;
        Ok((
            fp,
            DeviceDecodeStats {
                n_slices: d.n_slices,
                dirty,
                none,
                timing: d.timing,
                host: t.elapsed(),
            },
        ))
    }
}

/// Stage times of the resident path beyond the decode kernels.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentStats {
    pub decode: DeviceDecodeStats,
    /// Host phases 2 + 3a (steps only) and segment-list building.
    pub plan: std::time::Duration,
    /// The two gather kernels, extra-buffer upload included.
    pub gather: std::time::Duration,
}

impl FlatProof {
    /// The device-resident route: phase 1 on the device, phases 2 + 3a on the
    /// host over step records only, phase 3b on the device. Literals and
    /// hints never reach the host; the returned proof's `meta` has empty
    /// `lits` / `hints` and the device arrays beside it.
    pub fn from_binary_device_resident(
        ctx: &GpuContext,
        cnf: &[Vec<i32>],
        bytes: &[u8],
        slice: u32,
        cooperative: bool,
        blocks: u32,
    ) -> Result<(DeviceFlatProof, ResidentStats), DeviceDecodeError> {
        use crate::gpu_launcher::GatherSeg;
        if bytes.is_empty() {
            return Err(DeviceDecodeError::Parse(crate::lrat::ParseError {
                line: 0,
                msg: "empty proof".into(),
            }));
        }
        let d = ctx
            .decode_lrat_slices(bytes, slice, cooperative, blocks, true)
            .map_err(DeviceDecodeError::Gpu)?;
        let t = std::time::Instant::now();
        let bounds = slice_bounds(bytes.len(), slice as usize);
        let (chunks, bases, dirty, none) = chunks_from_device(bytes, &d);
        let plan = stitch_plan(cnf, bytes, &bounds, chunks).map_err(DeviceDecodeError::Parse)?;
        let StitchPlan {
            mut fp,
            copies,
            dense_of,
            lit_total,
            hint_total,
            orig_lits,
            chunks,
            bridges,
        } = plan;
        let n_slices = chunks.len();
        // Segment lists: the originals first (from the extra buffer), then
        // each true-chain segment from the device arrays or, for bridges and
        // host-parsed slices, from the extra buffer.
        let mut extra_lits: Vec<i32> = Vec::with_capacity(orig_lits);
        let mut extra_hints: Vec<u64> = Vec::new();
        for c in cnf {
            extra_lits.extend_from_slice(c);
        }
        let mut lit_segs: Vec<GatherSeg> = Vec::with_capacity(copies.len() + 1);
        let mut hint_segs: Vec<GatherSeg> = Vec::with_capacity(copies.len());
        lit_segs.push(GatherSeg {
            sel: 1,
            src: 0,
            len: orig_lits as u32,
        });
        // For reporting a bad hint: where each hint segment's source is.
        for cp in &copies {
            let nl = (cp.lit_hi - cp.lit_lo) as u32;
            let nh = (cp.hint_hi - cp.hint_lo) as u32;
            let dev = if cp.chunk < n_slices {
                bases[cp.chunk]
            } else {
                None
            };
            match dev {
                Some((lb, hb)) => {
                    lit_segs.push(GatherSeg {
                        sel: 0,
                        src: lb + cp.lit_lo as u32,
                        len: nl,
                    });
                    hint_segs.push(GatherSeg {
                        sel: 0,
                        src: hb + cp.hint_lo as u32,
                        len: nh,
                    });
                }
                None => {
                    let c = if cp.chunk < n_slices {
                        &chunks[cp.chunk]
                    } else {
                        &bridges[cp.chunk - n_slices]
                    };
                    lit_segs.push(GatherSeg {
                        sel: 1,
                        src: extra_lits.len() as u32,
                        len: nl,
                    });
                    extra_lits.extend_from_slice(&c.lits[cp.lit_lo..cp.lit_hi]);
                    hint_segs.push(GatherSeg {
                        sel: 1,
                        src: extra_hints.len() as u32,
                        len: nh,
                    });
                    extra_hints.extend_from_slice(&c.hints[cp.hint_lo..cp.hint_hi]);
                }
            }
        }
        let t_plan = t.elapsed();
        let t2 = std::time::Instant::now();
        let raw_lits = d.d_lits.as_ref().expect("resident decode keeps lits");
        let raw_hints = d.d_hints.as_ref().expect("resident decode keeps hints");
        let (lits, hints, first_bad) = ctx
            .gather_lrat_payloads(
                &lit_segs,
                &hint_segs,
                raw_lits,
                raw_hints,
                &extra_lits,
                &extra_hints,
                &dense_of,
                lit_total,
                hint_total,
                blocks.max(1024),
            )
            .map_err(DeviceDecodeError::Gpu)?;
        let t_gather = t2.elapsed();
        if let Some(first_bad) = first_bad {
            // The raw id behind it, from wherever that hint came from.
            let mut g = 0usize;
            let mut h = 0u64;
            for seg in &hint_segs {
                let n = seg.len as usize;
                if first_bad < g + n {
                    let j = seg.src as usize + (first_bad - g);
                    h = if seg.sel == 0 {
                        ctx.read_u64_at(raw_hints, j)
                            .map_err(DeviceDecodeError::Gpu)?
                    } else {
                        extra_hints[j]
                    };
                    break;
                }
                g += n;
            }
            let l = fp.hint_start.partition_point(|&s| s as usize <= first_bad) - 1;
            fp.early = Some(super::Verdict::Rejected {
                step: fp.lemma_step[l] as usize,
                id: fp.lemma_id[l],
                reason: super::missing_hint_reason(h, fp.lemma_id[l]),
            });
            super::truncate_lemmas(&mut fp, l);
        }
        let keep = fp.lemma_clause.len();
        super::truncate_lemmas(&mut fp, keep);
        Ok((
            DeviceFlatProof {
                meta: fp,
                lits,
                hints,
            },
            ResidentStats {
                decode: DeviceDecodeStats {
                    n_slices: d.n_slices,
                    dirty,
                    none,
                    timing: d.timing,
                    host: t_plan + t_gather,
                },
                plan: t_plan,
                gather: t_gather,
            },
        ))
    }
}

/// [`combine`](super::combine) for a resident proof: the one lemma whose
/// reason names a hint has its hint range fetched from the device, so the
/// verdict is the same as the host route's, hint id included.
pub fn combine_resident(
    ctx: &GpuContext,
    dfp: &DeviceFlatProof,
    codes: &[super::LemmaCode],
) -> Result<super::Verdict, GpuError> {
    use super::LemmaCode;
    let first = codes.iter().position(|&c| c != LemmaCode::Ok);
    match first {
        Some(i)
            if codes.len() == dfp.meta.lemma_clause.len()
                && matches!(codes[i], LemmaCode::DeletedHint | LemmaCode::ForwardHint) =>
        {
            let (hs, he) = (
                dfp.meta.hint_start[i] as usize,
                dfp.meta.hint_start[i + 1] as usize,
            );
            let mut meta = dfp.meta.clone();
            meta.hints = vec![0u32; he];
            if he > hs {
                let part: Vec<u32> = ctx.stream().clone_dtoh(&dfp.hints.slice(hs..he))?;
                meta.hints[hs..he].copy_from_slice(&part);
            }
            Ok(super::combine(&meta, codes))
        }
        _ => Ok(super::combine(&dfp.meta, codes)),
    }
}

/// Either the proof is malformed (the one-thread decoder's error, reproduced
/// by the stitch) or the device failed.
#[derive(Debug)]
pub enum DeviceDecodeError {
    Parse(crate::lrat::ParseError),
    Gpu(GpuError),
}

impl std::fmt::Display for DeviceDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceDecodeError::Parse(e) => write!(f, "{e}"),
            DeviceDecodeError::Gpu(e) => write!(f, "{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    //! The device battery: the same per-slice and end-to-end comparisons as
    //! the model's, through both kernels. Needs a GPU; skipped (with a
    //! message) when no context can be created.
    use super::super::tests::{assert_flat_eq, synthetic_proof};
    use super::super::FlatProof;
    use crate::gpu_launcher::GpuContext;

    fn ctx() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("device decode tests: SKIPPED (no GPU context: {e})");
                None
            }
        }
    }

    fn same(ctx: &GpuContext, cnf: &[Vec<i32>], bytes: &[u8], slice: u32, coop: bool, what: &str) {
        let a = FlatProof::from_binary(cnf, bytes);
        let b = FlatProof::from_binary_device(ctx, cnf, bytes, slice, coop, 64);
        match (a, b) {
            (Ok(a), Ok((b, _))) => assert_flat_eq(&a, &b, what),
            (Err(a), Err(super::DeviceDecodeError::Parse(b))) => assert_eq!(a, b, "{what}: error"),
            (a, b) => panic!(
                "{what} coop={coop}: sequential {:?} vs device {:?}",
                a.map(|_| ()),
                b.map(|_| ()).map_err(|e| e.to_string())
            ),
        }
    }

    #[test]
    fn device_equals_sequential_on_valid_proofs_and_truncations() {
        let Some(ctx) = ctx() else { return };
        for coop in [true, false] {
            for with_empty in [false, true] {
                let (cnf, bytes) = synthetic_proof(with_empty);
                for &slice in &[1u32, 5, 7, 13, 64, 4096] {
                    let what = format!("valid s={slice} e={with_empty}");
                    same(&ctx, &cnf, &bytes, slice, coop, &what);
                    // Clean slices must exist: a kernel that flags everything
                    // dirty would pass through the host fallback alone.
                    let (_, st) =
                        FlatProof::from_binary_device(&ctx, &cnf, &bytes, slice, coop, 64).unwrap();
                    assert!(
                        st.dirty + st.none < st.n_slices,
                        "{what} coop={coop}: every slice fell back ({st:?})"
                    );
                }
            }
            let (cnf, bytes) = synthetic_proof(true);
            for cut in (0..bytes.len()).step_by(3) {
                for &slice in &[11u32, 64] {
                    same(
                        &ctx,
                        &cnf,
                        &bytes[..cut],
                        slice,
                        coop,
                        &format!("truncate {cut} s={slice}"),
                    );
                }
            }
        }
    }

    #[test]
    fn device_equals_sequential_on_bit_flips() {
        let Some(ctx) = ctx() else { return };
        let (cnf, bytes) = synthetic_proof(false);
        let mut m = bytes.clone();
        for coop in [true, false] {
            for i in 0..bytes.len() {
                for &bit in &[7u32, 0] {
                    m[i] ^= 1 << bit;
                    same(&ctx, &cnf, &m, 13, coop, &format!("flip {i}.{bit}"));
                    m[i] ^= 1 << bit;
                }
            }
        }
    }

    /// Long steps and RAT hints across windows, as in the model's tests.
    #[test]
    fn device_handles_straddling_steps_and_rat_hints() {
        let Some(ctx) = ctx() else { return };
        let cnf: Vec<Vec<i32>> = (1..=300).map(|v| vec![v]).collect();
        let mut p = crate::lrat::Proof { steps: Vec::new() };
        for i in 0..40u64 {
            p.steps.push(crate::lrat::Step::Add {
                id: 301 + i,
                lits: (0..7).map(|k| 100 + (i as i32 * 7 + k) % 150).collect(),
                hints: vec![1, 2],
            });
        }
        let bytes = crate::lrat::write_binary(&p);
        for coop in [true, false] {
            for &slice in &[33u32, 64, 100, 4096] {
                same(
                    &ctx,
                    &cnf,
                    &bytes,
                    slice,
                    coop,
                    &format!("straddle s={slice}"),
                );
            }
        }
        let cnf: Vec<Vec<i32>> = (1..=200).map(|v| vec![v]).collect();
        for k in [0usize, 20, 60, 79] {
            let hints: Vec<i64> = (0..80)
                .map(|h| {
                    if h == k {
                        -((h + 1) as i64)
                    } else {
                        (h + 1) as i64
                    }
                })
                .collect();
            let p = crate::lrat::Proof {
                steps: vec![crate::lrat::Step::Add {
                    id: 201,
                    lits: vec![7, 8, 9],
                    hints,
                }],
            };
            let bytes = crate::lrat::write_binary(&p);
            for coop in [true, false] {
                for &slice in &[9u32, 40, 4096] {
                    same(
                        &ctx,
                        &cnf,
                        &bytes,
                        slice,
                        coop,
                        &format!("rat {k} s={slice}"),
                    );
                }
            }
        }
    }

    /// The resident route against the sequential decoder: meta plus the
    /// downloaded arrays equal, errors equal, verdicts equal through
    /// `combine_resident` with the resident check.
    #[test]
    fn device_resident_equals_sequential() {
        let Some(ctx) = ctx() else { return };
        let mut cases: Vec<(Vec<Vec<i32>>, Vec<u8>, String)> = Vec::new();
        for with_empty in [false, true] {
            let (cnf, bytes) = synthetic_proof(with_empty);
            cases.push((cnf.clone(), bytes.clone(), format!("valid e={with_empty}")));
            for cut in (1..bytes.len()).step_by(7) {
                cases.push((cnf.clone(), bytes[..cut].to_vec(), format!("cut {cut}")));
            }
            let mut m = bytes.clone();
            for i in (0..bytes.len()).step_by(5) {
                m[i] ^= 0x80;
                cases.push((cnf.clone(), m.clone(), format!("flip {i}")));
                m[i] ^= 0x80;
            }
        }
        for (cnf, bytes, what) in &cases {
            for coop in [true, false] {
                for &slice in &[13u32, 4096] {
                    let a = FlatProof::from_binary(cnf, bytes);
                    let b =
                        FlatProof::from_binary_device_resident(&ctx, cnf, bytes, slice, coop, 64);
                    match (a, b) {
                        (Ok(a), Ok((dfp, _))) => {
                            let (l, h) = ctx.download_lrat_payloads(&dfp).unwrap();
                            let mut m = dfp.meta.clone();
                            m.lits = l;
                            m.hints = h;
                            assert_flat_eq(&a, &m, &format!("{what} s={slice} coop={coop}"));
                            let codes_host = super::super::check_warp_codes(&a);
                            let want = super::super::combine(&a, &codes_host);
                            let raw = ctx.check_lrat_codes_resident(&dfp, 64).unwrap();
                            let codes: Vec<super::super::LemmaCode> = raw
                                .iter()
                                .map(|&c| match c {
                                    0 => super::super::LemmaCode::Ok,
                                    1 => super::super::LemmaCode::NotUnit,
                                    2 => super::super::LemmaCode::NoConflict,
                                    3 => super::super::LemmaCode::DeletedHint,
                                    4 => super::super::LemmaCode::ForwardHint,
                                    5 => super::super::LemmaCode::TableOverflow,
                                    _ => super::super::LemmaCode::Unwritten,
                                })
                                .collect();
                            let got = super::combine_resident(&ctx, &dfp, &codes).unwrap();
                            assert_eq!(got, want, "{what} s={slice} coop={coop}: verdict");
                        }
                        (Err(a), Err(super::DeviceDecodeError::Parse(b))) => {
                            assert_eq!(a, b, "{what}: error")
                        }
                        (a, b) => panic!(
                            "{what} s={slice} coop={coop}: sequential {:?} vs resident {:?}",
                            a.map(|_| ()),
                            b.map(|_| ()).map_err(|e| e.to_string())
                        ),
                    }
                }
            }
        }
    }

    /// Archive agreement through the device path (skips without
    /// `LRAT_ARCHIVE_DIRS`; same layout as the host archive test: `proof/*.lrat`
    /// beside `cnf/<base>.cnf`), both kernels.
    #[test]
    fn archive_agreement_device_decode() {
        let Ok(dirs) = std::env::var("LRAT_ARCHIVE_DIRS") else {
            eprintln!("archive_agreement_device_decode: SKIPPED (set LRAT_ARCHIVE_DIRS)");
            return;
        };
        let Some(ctx) = ctx() else { return };
        let mut n = 0usize;
        for dir in dirs.split(':') {
            let dir = std::path::Path::new(dir);
            let Ok(rd) = std::fs::read_dir(dir.join("proof")) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("lrat") {
                    continue;
                }
                let stem = p.file_name().unwrap().to_str().unwrap().to_string();
                let base = stem.split('.').next().unwrap();
                let Ok(cnf_text) =
                    std::fs::read_to_string(dir.join("cnf").join(format!("{base}.cnf")))
                else {
                    continue;
                };
                let Ok(proof_text) = std::fs::read_to_string(&p) else {
                    continue;
                };
                let Ok(proof) = crate::lrat::parse_text(&proof_text) else {
                    continue;
                };
                let cnf = crate::lrat::parse_cnf_ints(&cnf_text);
                let bytes = crate::lrat::write_binary(&proof);
                for coop in [true, false] {
                    for &slice in &[4096u32, 64] {
                        same(
                            &ctx,
                            &cnf,
                            &bytes,
                            slice,
                            coop,
                            &format!("{stem} s={slice}"),
                        );
                    }
                }
                n += 1;
            }
        }
        eprintln!("archive_agreement_device_decode: {n} proofs, both kernels");
        assert!(n > 0, "no proofs found under LRAT_ARCHIVE_DIRS");
    }
}
