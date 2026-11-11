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

/// Host threads for the device routes' host stages: what this process may
/// run on (`available_parallelism` honours the CPU affinity mask).
fn host_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Build the per-slice records from the device arrays, slices split over
/// `threads` host threads.
fn chunks_from_device(bytes: &[u8], d: &DeviceSlices, threads: usize) -> DeviceChunks {
    let n = d.n_slices;
    let per = n.div_ceil(threads.max(1)).max(1);
    let parts: Vec<DeviceChunks> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n)
            .step_by(per)
            .map(|lo| sc.spawn(move || chunks_from_device_range(bytes, d, lo, (lo + per).min(n))))
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("device records"))
            .collect()
    });
    let mut out: DeviceChunks = (Vec::with_capacity(n), Vec::with_capacity(n), 0, 0);
    for (c, b, dirty, none) in parts {
        out.0.extend(c);
        out.1.extend(b);
        out.2 += dirty;
        out.3 += none;
    }
    out
}

/// [`chunks_from_device`] for slices `lo..hi`.
fn chunks_from_device_range(
    bytes: &[u8],
    d: &DeviceSlices,
    lo_slice: usize,
    hi_slice: usize,
) -> DeviceChunks {
    let resident = d.d_lits.is_some();
    let mut bases: Vec<Option<(u32, u32)>> = Vec::with_capacity(hi_slice - lo_slice);
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
    let mut chunks = Vec::with_capacity(hi_slice - lo_slice);
    for i in lo_slice..hi_slice {
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
        let threads = host_threads();
        let (chunks, _bases, dirty, none) = chunks_from_device(bytes, &d, threads);
        let fp = stitch_chunks(cnf, bytes, &bounds, chunks, threads)
            .map_err(DeviceDecodeError::Parse)?;
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
    /// Phase 3a ran on the optimistic parallel walk (`false`: the proof had
    /// something to reject, or one host thread, and the sequential walk ran).
    pub fast_plan: bool,
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
        let threads = host_threads();
        let (chunks, bases, dirty, none) = chunks_from_device(bytes, &d, threads);
        let plan =
            stitch_plan(cnf, bytes, &bounds, chunks, threads).map_err(DeviceDecodeError::Parse)?;
        let StitchPlan {
            mut fp,
            copies,
            dense_of,
            lit_total,
            hint_total,
            orig_lits,
            chunks,
            bridges,
            fast,
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
                fast_plan: fast,
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

    /// Literal 0 (encoded 1) is refused on the device as on the host — the
    /// kernel must call such a slice dirty, not clean, or the device returns
    /// a proof the host refuses. Planted in each lemma of the synthetic proof
    /// in turn by turning a literal terminator into encoded 1 plus terminator.
    #[test]
    fn device_refuses_literal_zero() {
        let Some(ctx) = ctx() else { return };
        let cnf = vec![vec![1, 2], vec![-1, 2], vec![-2]];
        let bin = vec![
            b'a', 0x08, 0x04, 0x01, 0x00, 0x02, 0x04, 0x00, b'a', 0x0a, 0x00, 0x08, 0x06, 0x00,
        ];
        assert!(FlatProof::from_binary(&cnf, &bin).is_err());
        for coop in [true, false] {
            for &slice in &[1u32, 3, 5, 7, 13, 64, 4096] {
                same(
                    &ctx,
                    &cnf,
                    &bin,
                    slice,
                    coop,
                    &format!("literal 0 s={slice}"),
                );
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

    // ---- the check kernel, lemma by lemma (research/gpu-lrat-decode.md S5) ----

    /// The device's per-lemma codes against `check_warp_codes` (the SimWarp
    /// model) at several block counts — every lemma, not only the verdict.
    fn codes_equal(ctx: &GpuContext, fp: &FlatProof, what: &str) {
        let want: Vec<u8> = super::super::check_warp_codes(fp)
            .iter()
            .map(|&c| c as u8)
            .collect();
        let lits = ctx.stream().clone_htod(&fp.lits).unwrap();
        let hints = ctx.stream().clone_htod(&fp.hints).unwrap();
        // Every proof admits both layouts since the direct region is sized to
        // the proof's largest code (edp-record §9); the launcher's own choice
        // is the third mode.
        let modes: &[Option<bool>] = &[None, Some(false), Some(true)];
        for (&mode, blocks) in modes
            .iter()
            .flat_map(|m| [1u32, 7, 64, 4096].map(|b| (m, b)))
        {
            let got = ctx
                .check_lrat_codes_mode(fp, &lits, &hints, blocks, mode)
                .unwrap();
            let what = format!("{what} mode={mode:?}");
            if got != want {
                let i = got.iter().zip(&want).position(|(a, b)| a != b);
                panic!(
                    "{what} blocks={blocks}: codes differ (len {} vs {}), first at {i:?}: {:?} vs {:?}",
                    got.len(),
                    want.len(),
                    i.map(|i| got[i]),
                    i.map(|i| want[i])
                );
            }
        }
    }

    /// Proofs that reach every code: the synthetic proofs, steps after ⊥,
    /// hint-level corruptions of the synthetic proof, and both overflow
    /// shapes (a lemma past the table cap; units pushing a lemma past it).
    fn code_battery() -> Vec<(String, Vec<Vec<i32>>, crate::lrat::Proof)> {
        use crate::lrat::{parse_binary, parse_cnf_ints, parse_text, Proof, Step};
        let mut out = Vec::new();
        for e in [false, true] {
            let (cnf, bytes) = synthetic_proof(e);
            let p = parse_binary(&bytes).unwrap();
            out.push((format!("synthetic e={e}"), cnf.clone(), p.clone()));
            // Hint corruptions, one lemma at a time across the proof: a hint
            // dropped, a hint swapped for a later id, for a deleted id.
            let mut x = 7u32;
            for k in 0..24 {
                let mut q = p.clone();
                let adds: Vec<usize> = q
                    .steps
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| matches!(s, Step::Add { hints, .. } if hints.len() > 1))
                    .map(|(i, _)| i)
                    .collect();
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let at = adds[(x as usize >> 4) % adds.len()];
                let deleted = q.steps[..at].iter().find_map(|s| match s {
                    Step::Delete { ids } => Some(ids[0]),
                    _ => None,
                });
                if let Step::Add { id, hints, .. } = &mut q.steps[at] {
                    match k % 3 {
                        0 => {
                            hints.remove(0);
                        }
                        1 => hints[0] = (*id + 1) as i64,
                        _ => {
                            if let Some(d) = deleted {
                                hints[0] = d as i64;
                            }
                        }
                    }
                }
                out.push((format!("synthetic e={e} corruption {k}"), cnf.clone(), q));
            }
        }
        let c = parse_cnf_ints("p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n");
        for (i, t) in [
            "5 2 0 1 2 0\n6 0 5 3 4 0\n7 1 0 99 0\n",
            "5 2 0 1 2 0\n6 0 5 3 4 0\n7 1 0 1 0\n",
            "5 2 0 1 2 0\n6 0 5 3 4 0\n",
            "5 2 0 1 0\n6 0 5 3 4 0\n",
        ]
        .iter()
        .enumerate()
        {
            out.push((format!("small {i}"), c.clone(), parse_text(t).unwrap()));
        }
        // Overflow: a 5,000-literal lemma; and a 4,090-literal lemma whose
        // ten unit hints push the table past its 4,095 cap.
        let mut cnf: Vec<Vec<i32>> = (0..10).map(|k| vec![1, 5000 + k]).collect();
        cnf.push(vec![-1]);
        let steps = vec![
            Step::Add {
                id: 12,
                lits: (1..=5000).collect(),
                hints: vec![11],
            },
            Step::Add {
                id: 13,
                lits: (1..=4090).collect(),
                hints: (1..=10).collect(),
            },
            Step::Add {
                id: 14,
                lits: vec![2],
                hints: vec![11, 1],
            },
        ];
        out.push(("overflow".into(), cnf, Proof { steps }));
        // At the cap with a key already present (cross-vendor panel
        // 2026-11-04): 4,095 distinct lemma literals fill the table, then the
        // 4,096th insert repeats one — once as a repeated lemma literal, once
        // as the unit of a hint clause whose free literal is the negation of
        // a lemma literal. SimWarp's table refuses at the cap before looking.
        // Clause 1 = (-1 ∨ 2): its free literal under the lemma is -1, so its
        // unit re-inserts literal 1. Clause 2 = (2 ∨ 3): falsified outright.
        let cnf: Vec<Vec<i32>> = vec![vec![-1, 2], vec![2, 3]];
        let mut rep: Vec<i32> = (1..=4095).collect();
        rep.push(1);
        let steps = vec![
            Step::Add {
                id: 3,
                lits: rep,
                hints: vec![2],
            },
            Step::Add {
                id: 4,
                lits: (1..=4095).collect(),
                hints: vec![1, 2],
            },
        ];
        out.push(("duplicate at the cap".into(), cnf, Proof { steps }));
        // Wide variables (edp-record §9): the two shapes above with every
        // variable past 4,096 codes — "small 0" on variables 5,000–5,001 and
        // "duplicate at the cap" on 1,906..=6,000 (4,095 literals, codes up
        // to 12,001) — so the direct region is wider than LRAT_TABLE while
        // the occupancy cap is seen to stay at 4,095.
        let c = parse_cnf_ints(
            "p cnf 5001 4\n5000 5001 0\n-5000 5001 0\n5000 -5001 0\n-5000 -5001 0\n",
        );
        out.push((
            "wide small".into(),
            c,
            parse_text("5 5001 0 1 2 0\n6 0 5 3 4 0\n").unwrap(),
        ));
        let cnf: Vec<Vec<i32>> = vec![vec![-5999, 6000], vec![6000, 5998]];
        let mut rep: Vec<i32> = (1906..=6000).collect();
        rep.push(5999);
        let steps = vec![
            Step::Add {
                id: 3,
                lits: rep,
                hints: vec![2],
            },
            Step::Add {
                id: 4,
                lits: (1906..=6000).collect(),
                hints: vec![1, 2],
            },
        ];
        out.push(("wide duplicate at the cap".into(), cnf, Proof { steps }));
        // ---- edp-record §10: the batched hint loop's failure modes ----
        // (a) A unit chain inside one batch and across batches: hint k's unit
        // is what makes hint k+1 unit (M-H1: in-batch falsification dropped).
        // Chains of 7 (one batch) and 46 (two 32-hint windows, several
        // batches); each also in a wrong hint order (NotUnit) and without
        // the closing conflict (NoConflict).
        for n in [7u32, 46] {
            let mut cnf: Vec<Vec<i32>> = vec![vec![1, 2]];
            for k in 2..n {
                cnf.push(vec![-(k as i32), k as i32 + 1]);
            }
            cnf.push(vec![-(n as i32)]);
            let m = cnf.len() as u64; // clause ids 1..=m; the last is the conflict
            let chain: Vec<i64> = (1..=m as i64).collect();
            let mut swapped = chain.clone();
            swapped.swap(2, 3);
            let open: Vec<i64> = (1..m as i64).collect();
            for (what, hints) in [
                ("chain", chain),
                ("chain wrong order", swapped),
                ("chain open", open),
            ] {
                let steps = vec![Step::Add {
                    id: m + 1,
                    lits: vec![1],
                    hints,
                }];
                out.push((format!("{what} {n}"), cnf.clone(), Proof { steps }));
            }
        }
        // (b) Two hints unit on the same literal with the count near the cap.
        // Clause 1 = (1 ∨ 5000), 2 = (2 ∨ 5000), 3 = (3 ∨ 5001), 4 = (-5000 ∨ -5001).
        // Lemma 1..=4094: hint 1 inserts -5000 (count 4,095), hint 2 repeats it
        // — the cap is tested before the lookup, so the reference says
        // TableOverflow. Lemma 1..=4093: the repeat passes without counting
        // (M-H2 counts it: then hint 3's insert overflows instead of
        // reaching hint 4's conflict). Lemma 1..=4095: the table is at the cap
        // before any hint, so the FIRST unit insert overflows — the §10 (b)
        // "at 4,095 (it must, at the first)" case, missing until the
        // 2026-11-11 panel's third round noticed.
        let cnf: Vec<Vec<i32>> = vec![
            vec![1, 5000],
            vec![2, 5000],
            vec![3, 5001],
            vec![-5000, -5001],
        ];
        for top in [4094i32, 4093, 4095] {
            let steps = vec![Step::Add {
                id: 5,
                lits: (1..=top).collect(),
                hints: vec![1, 2, 3, 4],
            }];
            out.push((
                format!("duplicate units near the cap {top}"),
                cnf.clone(),
                Proof { steps },
            ));
        }
        // (c) Order of failures: a conflict at hint j then a forward hint at
        // j+1 — the NEXT lemma's id, since flatten stops at an unknown id —
        // (code 0, not 4: a batch-cut control, since both admissibility
        // orders answer 0 here); a deleted hint whose clause would conflict
        // (code 3, not 0); a forward hint whose empty bounds would read as a
        // conflict (code 4, not 0) — the last two kill M-H3 (admissibility
        // tested after the count); the first does not (panel 2026-11-11).
        let cnf: Vec<Vec<i32>> = vec![vec![-1], vec![2]];
        out.push((
            "conflict then forward".into(),
            cnf.clone(),
            Proof {
                steps: vec![
                    Step::Add {
                        id: 3,
                        lits: vec![-1],
                        hints: vec![1, 4],
                    },
                    Step::Add {
                        id: 4,
                        lits: vec![2],
                        hints: vec![2],
                    },
                ],
            },
        ));
        out.push((
            "deleted conflict".into(),
            cnf.clone(),
            Proof {
                steps: vec![
                    Step::Delete { ids: vec![2] },
                    Step::Add {
                        id: 3,
                        lits: vec![2],
                        hints: vec![2, 1],
                    },
                ],
            },
        ));
        out.push((
            "forward first".into(),
            cnf,
            Proof {
                steps: vec![
                    Step::Add {
                        id: 3,
                        lits: vec![2],
                        hints: vec![4, 1],
                    },
                    Step::Add {
                        id: 4,
                        lits: vec![2],
                        hints: vec![2],
                    },
                ],
            },
        ));
        // (d) Hint clauses of 31, 32, 33 and 70 literals; the 33- and 70-wide
        // ones are unit on their last literal, and the hint after each is
        // NotUnit only if that unit was seen (M-H4 truncates to 32 lanes: a
        // conflict instead). Clause 5 = (33 ∨ 35), 6 = (70 ∨ 71). Clauses 7
        // = (¬33 ∨ 35) and 8 = (¬70 ∨ 71) test the other direction — the
        // unit found by the STRIDED fallback (the 33- and 70-wide hints run
        // outside the batch loop) must be visible to the batch that follows:
        // ¬33 falsified → unit on 35 → the lemma ends NoConflict; a store the
        // next batch cannot see reads as NotUnit instead (cross-vendor panel
        // 2026-11-11, Opus seat: (33 ∨ 35) is NotUnit either way).
        let cnf: Vec<Vec<i32>> = vec![
            (1..=31).collect(),
            (1..=32).collect(),
            (1..=33).collect(),
            (1..=70).collect(),
            vec![33, 35],
            vec![70, 71],
            vec![-33, 35],
            vec![-70, 71],
        ];
        for (what, lits, hints) in [
            ("wide 31", (1..=32).collect::<Vec<i32>>(), vec![1i64]),
            ("wide 32", (1..=32).collect(), vec![2]),
            ("wide 33 then", (1..=32).collect(), vec![3, 5]),
            ("wide 70 then", (1..=69).collect(), vec![4, 6]),
            ("wide 33 propagates", (1..=32).collect(), vec![3, 7]),
            ("wide 70 propagates", (1..=69).collect(), vec![4, 8]),
        ] {
            out.push((
                what.into(),
                cnf.clone(),
                Proof {
                    // The CNF above has 8 clauses: the lemma is id 9. (Round 1 of
                    // the 2026-11-11 panel grew the CNF from 6 to 8 clauses and
                    // left this at 7, so `flatten` stopped with IdNotIncreasing
                    // and every (d) shape compared empty against empty — a check
                    // that could not fail. The per-shape assertion in the test
                    // below is the guard.)
                    steps: vec![Step::Add { id: 9, lits, hints }],
                },
            ));
        }
        // (e) The lemma's own literals falsify the hint outright; (f) a hint
        // with a repeated literal (x x y, y falsified) — the lanes count
        // occurrences, so the reference says NotUnit, whatever a reader
        // expects.
        let cnf: Vec<Vec<i32>> = vec![vec![1, 2], vec![3, 3, 1]];
        out.push((
            "own literals".into(),
            cnf.clone(),
            Proof {
                steps: vec![Step::Add {
                    id: 3,
                    lits: vec![1, 2],
                    hints: vec![1],
                }],
            },
        ));
        out.push((
            "repeated literal".into(),
            cnf,
            Proof {
                steps: vec![Step::Add {
                    id: 3,
                    lits: vec![1],
                    hints: vec![2],
                }],
            },
        ));
        out
    }

    #[test]
    fn check_kernel_codes_equal_simwarp_on_the_battery() {
        let Some(ctx) = ctx() else { return };
        // `LRAT_BATTERY_ONLY=<substring>` restricts the run to the shapes whose
        // name contains it — so one mutant can be shown to die on ITS shape,
        // not on whichever earlier shape happens to differ first.
        // An empty value is "unset": `contains("")` would match every shape
        // while still skipping the unfiltered run's assertions (panel round 4).
        let only = std::env::var("LRAT_BATTERY_ONLY")
            .ok()
            .filter(|f| !f.is_empty());
        let mut seen = [0usize; 7];
        let mut vacuous = 0usize;
        let mut shapes = 0usize;
        for (what, cnf, proof) in code_battery() {
            if let Some(f) = &only {
                if !what.contains(f.as_str()) {
                    continue;
                }
            }
            shapes += 1;
            let fp = super::super::flatten(&cnf, &proof);
            // A shape that flattens to nothing compares empty against empty
            // and can never fail. Only a hint corruption planted in the FIRST
            // lemma may do that (the host rejects at step 0 before any lemma);
            // it is counted and printed. Every other shape must produce lemmas
            // — the (d) shapes did not for a day after their CNF grew (panel
            // 2026-11-11 round 2).
            if fp.lemma_clause.is_empty() {
                let rejected_at_step_0 = matches!(
                    fp.early,
                    Some(crate::lrat::Verdict::Rejected { step: 0, .. })
                );
                assert!(
                    what.contains("corruption") && rejected_at_step_0,
                    "battery shape {what:?} produced no lemmas (early = {:?})",
                    fp.early
                );
                vacuous += 1;
                continue;
            }
            // The §10 shapes must run to their end: an early host verdict would
            // leave later steps unchecked while the comparison still passed.
            // Older shapes may stop early by design (e.g. "small 0" ends on an
            // unknown hint id); those are printed so the list stays visible.
            const SECTION_10: [&str; 11] = [
                "chain",
                "duplicate units",
                "conflict then forward",
                "deleted conflict",
                "forward first",
                "wide 31",
                "wide 32",
                "wide 33",
                "wide 70",
                "own literals",
                "repeated literal",
            ];
            if SECTION_10.iter().any(|p| what.starts_with(p)) {
                assert!(
                    fp.early.is_none(),
                    "battery shape {what:?} cut short by the host: {:?}",
                    fp.early
                );
            } else if fp.early.is_some() && !what.contains("corruption") {
                eprintln!(
                    "battery shape {what:?} ends in an early host verdict after {} lemmas: {:?}",
                    fp.lemma_clause.len(),
                    fp.early
                );
            }
            for c in super::super::check_warp_codes(&fp) {
                seen[c as usize] += 1;
            }
            codes_equal(&ctx, &fp, &what);
        }
        eprintln!(
            "check kernel battery: {shapes} shapes ({vacuous} first-lemma corruption shapes had no lemmas for the kernel); codes seen Ok/NotUnit/NoConflict/Deleted/Forward/Overflow = {:?}",
            &seen[..6]
        );
        match &only {
            None => {
                assert_eq!(vacuous, 1, "exactly one battery shape (the corruption of the first lemma) may be lemma-less");
                // Every code the kernel can write is exercised.
                for (c, n) in seen[..6].iter().enumerate() {
                    assert!(*n > 0, "code {c} never produced by the battery");
                }
            }
            Some(f) => {
                // A filter that matches nothing must not read as a pass.
                assert!(
                    shapes > 0,
                    "LRAT_BATTERY_ONLY={f:?} matched no battery shape"
                );
                eprintln!("check kernel battery: FILTERED to {f:?} — the vacuous-count and code-coverage assertions were skipped");
            }
        }
    }

    #[test]
    fn check_kernel_codes_equal_simwarp_on_the_archive() {
        let Ok(dirs) = std::env::var("LRAT_ARCHIVE_DIRS") else {
            eprintln!(
                "check_kernel_codes_equal_simwarp_on_the_archive: SKIPPED (set LRAT_ARCHIVE_DIRS)"
            );
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
                let (Ok(cnf_text), Ok(proof_text)) = (
                    std::fs::read_to_string(dir.join("cnf").join(format!("{base}.cnf"))),
                    std::fs::read_to_string(&p),
                ) else {
                    continue;
                };
                let Ok(proof) = crate::lrat::parse_text(&proof_text) else {
                    continue;
                };
                let cnf = crate::lrat::parse_cnf_ints(&cnf_text);
                let fp = super::super::flatten(&cnf, &proof);
                codes_equal(&ctx, &fp, &stem);
                n += 1;
            }
        }
        eprintln!("check_kernel_codes_equal_simwarp_on_the_archive: {n} proofs, 4 block counts");
        // A path that resolves to nothing must FAIL, not pass vacuously: on
        // 2026-11-10 relative LRAT_ARCHIVE_DIRS found 0 proofs and this test
        // was green while the three archive_agreement tests beside it failed.
        assert!(n > 0, "no proofs found under LRAT_ARCHIVE_DIRS");
    }

    /// The direct layout's own guard: launched direct with a region (4,096
    /// slots) the proof's literal codes do NOT fit — bypassing the launcher,
    /// which sizes the region to the largest code (edp-record §9) — the
    /// kernel must fail every lemma that meets a code past the region closed
    /// (TableOverflow), never report a lemma Ok that SimWarp does not, and
    /// agree with SimWarp on the lemmas that never leave the region. Through
    /// the launcher the same proof, forced direct, is equal to SimWarp.
    #[test]
    fn check_kernel_direct_layout_fails_closed_past_the_table() {
        let Some(ctx) = ctx() else { return };
        let (what, cnf, proof) = code_battery()
            .into_iter()
            .find(|(w, _, _)| w == "overflow")
            .unwrap();
        let fp = super::super::flatten(&cnf, &proof);
        let lits = ctx.stream().clone_htod(&fp.lits).unwrap();
        let hints = ctx.stream().clone_htod(&fp.hints).unwrap();
        assert!(
            ctx.lrat_max_code(&lits).unwrap() >= 4096,
            "{what}: codes must not fit 4,096 slots"
        );
        let want: Vec<u8> = super::super::check_warp_codes(&fp)
            .iter()
            .map(|&c| c as u8)
            .collect();
        assert_eq!(
            ctx.check_lrat_codes_mode(&fp, &lits, &hints, 7, Some(true))
                .unwrap(),
            want,
            "{what}: the launcher's direct region must fit the codes"
        );
        for blocks in [1u32, 7] {
            let got = ctx
                .launch_lrat_check(&fp, &lits, &hints, blocks, true, 4096)
                .unwrap();
            let mut closed = 0;
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g == w || g == 5,
                    "{what} lemma {i}: direct gave {g}, SimWarp {w}"
                );
                assert!(
                    g != 0 || w == 0,
                    "{what} lemma {i}: direct accepted what SimWarp did not"
                );
                closed += usize::from(g == 5 && w != 5);
            }
            assert!(closed > 0, "{what} blocks={blocks}: the guard never fired");
        }
    }

    /// A caller-supplied block count past the kernel's 32-bit table
    /// arithmetic (blocks · 4 warps · 4,096 slots ≥ 2^32 from 262,144 blocks)
    /// is clamped, not wrapped. The proof needs more lemmas than that, or the
    /// clamp to the lemma count hides the case: 300,000 one-literal lemmas,
    /// each refuted by the two original units.
    #[test]
    fn check_kernel_huge_block_count_is_clamped() {
        use crate::lrat::{Proof, Step};
        let Some(ctx) = ctx() else { return };
        let cnf = vec![vec![1], vec![-1]];
        let steps = (0..300_000u64)
            .map(|k| Step::Add {
                id: 3 + k,
                lits: vec![2 + (k % 1000) as i32],
                hints: vec![1, 2],
            })
            .collect();
        let fp = super::super::flatten(&cnf, &Proof { steps });
        let want: Vec<u8> = super::super::check_warp_codes(&fp)
            .iter()
            .map(|&c| c as u8)
            .collect();
        assert!(want.iter().all(|&c| c == 0));
        for blocks in [1u32 << 18, 300_000, u32::MAX] {
            let got = ctx.check_lrat_codes(&fp, blocks).unwrap();
            assert!(got == want, "blocks={blocks}: codes differ");
        }
    }
}
