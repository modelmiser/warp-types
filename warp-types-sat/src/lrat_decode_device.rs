//! The device decode path (feature `gpu`): phase-1 records from
//! `GpuContext::decode_lrat_slices` become the host's `RawChunk`s — dirty or
//! candidate-less slices through the host's own `parse_run` — and go through
//! the same `stitch_chunks` as the host workers' records. Stage 2 of
//! `research/gpu-lrat-decode.md`: the flat proof still lands on the host.

use super::decode_model::host_record;
use super::{slice_bounds, stitch_chunks, FlatProof, RawChunk, RawStep};
use crate::gpu_launcher::{DecodeTiming, DeviceSlices, GpuContext, GpuError};

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

/// Build the per-slice records from the device arrays.
fn chunks_from_device(bytes: &[u8], d: &DeviceSlices) -> (Vec<RawChunk>, usize, usize) {
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
                chunks.push(RawChunk {
                    steps,
                    lits: d.lits[l0..l0 + nl].to_vec(),
                    hints: d.hints[h0..h0 + nh].to_vec(),
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
                chunks.push(RawChunk {
                    end: lo as u64,
                    ..Default::default()
                });
            }
            _ => {
                dirty += 1;
                chunks.push(host_record(bytes, lo, hi));
            }
        }
    }
    (chunks, dirty, none)
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
            .decode_lrat_slices(bytes, slice, cooperative, blocks)
            .map_err(DeviceDecodeError::Gpu)?;
        let t = std::time::Instant::now();
        let bounds = slice_bounds(bytes.len(), slice as usize);
        debug_assert_eq!(bounds.len(), d.n_slices);
        let (chunks, dirty, none) = chunks_from_device(bytes, &d);
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
