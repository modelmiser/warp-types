//! The batched device-resident route (feature `gpu`): many
//! `(CNF ++ extras_j, proof_j)` steps through the single-proof resident
//! decode's kernels per chunk, laid out on the device as one flat proof for
//! one launch of the check kernel. `research/edp-record.md` §6, design C1.
//!
//! No kernel is changed. The two-pass scan runs once over a padded buffer of
//! every step's bytes; the literal gather runs once per chunk; the hint gather
//! runs once per step (its dense table is per step); the check runs once per
//! chunk. The host sees step records and codes — literals and hints never
//! come down.
//!
//! **Layout.** `lits` = the shared CNF's literals once, then per step its
//! private literals (extras, then lemmas) as the step's own decode lays them
//! out. Per step a region of `clause_start` / `deleted_at`: the CNF's clause
//! starts (so a hint naming CNF clause `i` reads the shared literals through
//! the step's own `deleted_at`), a sentinel entry so the last CNF clause ends
//! at the CNF's end and not at the step's first private literal, the
//! privates, and a terminator entry so the last private ends where it should
//! and not at the next region's first entry. Batch dense of a step-local
//! dense `d` in step `j` with region base `c_j`: `c_j + d` for `d < n_cnf`,
//! `c_j + d + 1` otherwise. Order within a region is the step's order
//! shifted, so the forward-hint test and the deletion test (both within one
//! step) read exactly as in the step alone — `lrat_batch`'s argument, with
//! per-step copies of the CNF's index entries instead of shared slots.
//!
//! **Routes.** A step whose slices all scanned clean or were re-parsed on the
//! host within the step's own bytes, whose chain never ran past its end, and
//! whose plan found nothing to reject, takes the device route. Any other step
//! is decoded on the host by [`FlatProof::from_binary`] (the proven decoder)
//! and its arrays are copied in. A hint naming an id never added — found by
//! the gather — sends that step to the host route and the chunk's layout is
//! redone. The contract, tested below: every step's codes and verdict equal
//! `from_binary` + `check_warp_codes` + `combine` on the step alone.

use super::decode_model::host_record;
use super::{combine, stitch_plan, FlatProof, LemmaCode, RawChunk, RawStep, StitchPlan};
use crate::gpu_launcher::{
    DecodeTiming, DeviceSlices, GpuContext, GpuError, DEC_CLEAN, DEC_DIRTY, DEC_NONE,
};
use crate::lrat::{ParseError, Verdict};
use cudarc::driver::CudaSlice;
use std::time::{Duration, Instant};

/// Knobs of one chunk.
#[derive(Debug, Clone, Copy)]
pub struct BatchOpts {
    /// Scan slice in bytes (every step is padded to end on a slice boundary).
    pub slice: u32,
    /// Blocks for the scan kernels.
    pub scan_blocks: u32,
    /// Blocks for the gather kernels.
    pub gather_blocks: u32,
    /// Blocks for the check kernel (0 = the launcher's choice).
    pub check_blocks: u32,
    /// Host threads for the padding, plan and layout phases.
    pub threads: usize,
    /// The warp scan kernel (`true`) or the thread-per-slice control.
    pub cooperative: bool,
}

impl Default for BatchOpts {
    fn default() -> Self {
        BatchOpts {
            slice: 4096,
            scan_blocks: 4096,
            gather_blocks: 4096,
            check_blocks: 0,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            cooperative: true,
        }
    }
}

/// Sizes and stage times of one chunk.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkStats {
    pub steps: usize,
    pub padded_bytes: usize,
    pub slices: usize,
    pub dirty: usize,
    pub lemmas: usize,
    pub hints: usize,
    pub lits: usize,
    pub clauses: usize,
    pub table: usize,
    /// Steps decoded by the host route (malformed, structurally rejected,
    /// overrun, or holding a hint to an id never added).
    pub host_route: usize,
    /// Layout rounds beyond the first (one per bad-hint re-route).
    pub relayouts: usize,
    /// The per-step copies of every proof's bytes (and pads) into the buffer.
    pub t_upload: Duration,
    /// The scan over the buffer: both passes, prefix, record download.
    pub t_scan: Duration,
    pub timing: DecodeTiming,
    pub t_plan: Duration,
    pub t_layout: Duration,
    /// Segment/table uploads, the gathers, the host-route copies, the bad download.
    pub t_gather: Duration,
    pub t_check: Duration,
    pub t_combine: Duration,
}

/// What one chunk returns.
pub struct ChunkOutcome {
    /// Per step: its verdict, or the decoder's error for a malformed proof.
    pub verdicts: Vec<Result<Verdict, ParseError>>,
    /// Per step: its per-lemma codes (empty for a malformed step).
    pub codes: Vec<Vec<LemmaCode>>,
    /// Per step: the route it took.
    pub routes: Vec<RouteKind>,
    pub stats: ChunkStats,
}

/// Which route a step took, and why when it was the host's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteKind {
    Device,
    /// An empty proof.
    HostEmpty,
    /// A record before the step's start (the padding makes this impossible).
    HostRecords,
    /// The stitch reported a format error.
    HostPlanError,
    /// The plan reached a structural verdict (id order, deletion, RAT).
    HostEarly,
    /// The gather met a hint naming an id never added.
    HostBadHint,
}

enum Route {
    Dev {
        plan: StitchPlan,
        /// Per slice of the step: device offsets of its raw literals and
        /// hints when the slice scanned clean, `None` when the host parsed it.
        bases: Vec<Option<(u32, u32)>>,
    },
    Host(FlatProof, RouteKind),
    Bad(ParseError, RouteKind),
}

/// Run `f(j)` for `j in 0..n` on `threads` workers, results in order.
fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<T>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads.max(1).min(n.max(1)) {
            s.spawn(|| loop {
                let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if j >= n {
                    break;
                }
                *out[j].lock().unwrap() = Some(f(j));
            });
        }
    });
    out.into_iter()
        .map(|m| m.into_inner().unwrap().unwrap())
        .collect()
}

/// Carve `lens` consecutive sub-slices off the front of `s`.
fn carve<T>(mut s: &mut [T], lens: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    let mut out = Vec::new();
    for l in lens {
        let (a, b) = s.split_at_mut(l);
        out.push(a);
        s = b;
    }
    out
}

fn code_of(c: u8) -> LemmaCode {
    match c {
        0 => LemmaCode::Ok,
        1 => LemmaCode::NotUnit,
        2 => LemmaCode::NoConflict,
        3 => LemmaCode::DeletedHint,
        4 => LemmaCode::ForwardHint,
        5 => LemmaCode::TableOverflow,
        _ => LemmaCode::Unwritten,
    }
}

/// Per-step offsets and sizes in the chunk arrays.
#[derive(Clone, Copy, Default)]
struct Off {
    c: usize,
    n_clauses: usize,
    l: usize,
    n_lits: usize,
    h: usize,
    n_hints: usize,
    lemma: usize,
    n_lemmas: usize,
    tab: usize,
    n_tab: usize,
    n_lseg: usize,
    hseg: usize,
    n_hseg: usize,
    xl: usize,
    n_xl: usize,
    xh: usize,
    n_xh: usize,
}

/// One step's destination regions.
struct Regions<'a> {
    cs: &'a mut [u32],
    del: &'a mut [u32],
    lemma_clause: &'a mut [u32],
    lemma_step: &'a mut [u32],
    hint_start: &'a mut [u32],
    tab: &'a mut [u32],
    lseg_dst: &'a mut [u32],
    lseg_src: &'a mut [u32],
    lseg_sel: &'a mut [u32],
    hseg_dst: &'a mut [u32],
    hseg_src: &'a mut [u32],
    hseg_sel: &'a mut [u32],
    xl: &'a mut [i32],
    xh: &'a mut [u64],
}

/// Rebuild one step's slice records from the device arrays, positions made
/// relative to the step's first byte. A clean slice whose speculative chain
/// ran past the step's end (a false start near the end keeps parsing into the
/// next step's padding and bytes — the single-proof route bounds that chain
/// by the proof's length) is re-parsed on the host within the step's bytes,
/// exactly as a dirty slice is; the stitch, which is proven for host-parsed
/// slices, decides what the true chain uses. `None` only for a record before
/// the step's start, which the padding makes impossible.
#[allow(clippy::type_complexity)]
fn step_records(
    bytes: &[u8],
    d: &DeviceSlices,
    k_lo: usize,
    k_hi: usize,
    base: u64,
    bounds: &[(usize, usize)],
) -> Option<(Vec<RawChunk>, Vec<Option<(u32, u32)>>)> {
    let len = bytes.len() as u64;
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
    let mut chunks = Vec::with_capacity(k_hi - k_lo);
    let mut bases = Vec::with_capacity(k_hi - k_lo);
    for (q, i) in (k_lo..k_hi).enumerate() {
        let (lo, hi) = bounds[q];
        let overrun = d.flags[i] == DEC_CLEAN && d.ends[i] > base + len;
        match d.flags[i] {
            DEC_CLEAN if !overrun => {
                let (s0, l0, h0, e0) = (
                    d.offs[4 * i] as usize,
                    d.offs[4 * i + 1] as usize,
                    d.offs[4 * i + 2] as usize,
                    d.offs[4 * i + 3] as usize,
                );
                let (ns, ne) = (count(i, 0), count(i, 3));
                if d.starts[i] < base {
                    return None;
                }
                let mut steps = Vec::with_capacity(ns);
                for k in 0..ns {
                    let pos = d.st_pos[s0 + k];
                    let tag = d.st_tag[s0 + k];
                    // `st_after` is set for `a` steps only (unused, 0, for `d`).
                    let after = d.st_after[s0 + k];
                    if pos < base || (tag == b'a' && after < base) {
                        return None;
                    }
                    steps.push(RawStep {
                        pos: pos - base,
                        id: d.st_id[s0 + k],
                        pos_after_id: if tag == b'a' { after - base } else { 0 },
                        lit_end: d.st_lit_end[s0 + k],
                        hint_end: d.st_hint_end[s0 + k],
                        del_end: d.st_del_end[s0 + k],
                        tag,
                        rat: d.st_rat[s0 + k] != 0,
                    });
                }
                let mut dels = Vec::with_capacity(ne);
                for k in 0..ne {
                    let p = d.del_pos[e0 + k];
                    if p < base {
                        return None;
                    }
                    dels.push((d.del_id[e0 + k], p - base));
                }
                bases.push(Some((l0 as u32, h0 as u32)));
                chunks.push(RawChunk {
                    steps,
                    lits: Vec::new(),
                    hints: Vec::new(),
                    dels,
                    error: None,
                    end: d.ends[i] - base,
                    partial_add: None,
                    partial_del_lo: 0,
                });
            }
            DEC_NONE => {
                bases.push(None);
                chunks.push(RawChunk {
                    end: lo as u64,
                    ..Default::default()
                });
            }
            _ => {
                // Dirty, or clean but overrun: the host's parse of this slice
                // within the step's own bytes, errors and all — the stitch
                // treats it exactly as it treats a host worker's record.
                bases.push(None);
                chunks.push(host_record(bytes, lo, hi));
            }
        }
    }
    Some((chunks, bases))
}

/// A chunk after [`prepare_chunk`]: its routes, the batched index arrays on
/// the host, and the gathered literal and hint arrays on the device (complete
/// — the preparing stream was synchronised by the bad-hint download). Owns no
/// borrow of the proofs, so it can cross to another thread and another stream
/// ([`check_pipelined`]). The device arrays are freed on the stream that
/// allocated them when this is dropped.
pub struct PreparedChunk {
    routes: Vec<Route>,
    offs: Vec<Off>,
    meta: FlatProof,
    d_lits: CudaSlice<i32>,
    d_hints: CudaSlice<u32>,
    n_cnf: usize,
    check_blocks: u32,
    stats: ChunkStats,
}

impl PreparedChunk {
    /// Sizes and stage times so far (the check and combine fields are zero).
    pub fn stats(&self) -> &ChunkStats {
        &self.stats
    }
}

/// Check one chunk of steps: step `j` is `cnf ++ extras[j]` with the binary
/// LRAT `proofs[j]`. Verdicts and codes per step, in order. The two halves,
/// [`prepare_chunk`] then [`check_prepared`], on the one stream.
pub fn check_chunk(
    ctx: &GpuContext,
    cnf: &[Vec<i32>],
    extras: &[Vec<Vec<i32>>],
    proofs: &[&[u8]],
    opts: &BatchOpts,
) -> Result<ChunkOutcome, GpuError> {
    let p = prepare_chunk(ctx, cnf, extras, proofs, opts)?;
    check_prepared(ctx, p)
}

/// Steps (1)–(5) of the design: upload and scan, records and plan per step,
/// layout, gathers. Everything but the check launch and the verdicts.
pub fn prepare_chunk(
    ctx: &GpuContext,
    cnf: &[Vec<i32>],
    extras: &[Vec<Vec<i32>>],
    proofs: &[&[u8]],
    opts: &BatchOpts,
) -> Result<PreparedChunk, GpuError> {
    assert_eq!(extras.len(), proofs.len(), "one extras list per proof");
    let n = proofs.len();
    let n_cnf = cnf.len();
    let threads = opts.threads.max(1);
    let slice = opts.slice.max(1) as usize;
    let mut st = ChunkStats {
        steps: n,
        ..Default::default()
    };
    let mut cs_cnf: Vec<u32> = Vec::with_capacity(n_cnf + 1);
    cs_cnf.push(0);
    for c in cnf {
        cs_cnf.push(cs_cnf.last().unwrap() + c.len() as u32);
    }
    let cnf_lits = cs_cnf[n_cnf] as usize;
    let formula = |j: usize| -> Vec<Vec<i32>> {
        let mut f: Vec<Vec<i32>> = vec![Vec::new(); n_cnf];
        f.extend(extras[j].iter().cloned());
        f
    };
    let host_route = |j: usize, why: RouteKind| -> Route {
        match FlatProof::from_binary(&formula(j), proofs[j]) {
            Ok(fp) => Route::Host(fp, why),
            Err(e) => Route::Bad(e, why),
        }
    };

    // ---- (1) + (2): each step's bytes go straight into its place in one
    // device buffer, front-padded with 0xFF…0x00 to end on a slice boundary;
    // then the scan, once over the buffer ----
    let t = Instant::now();
    let mut pad = vec![0usize; n];
    let mut base = vec![0usize; n];
    let mut total = 0usize;
    for j in 0..n {
        let len = proofs[j].len();
        let p = if len == 0 {
            0
        } else {
            (slice - len % slice) % slice
        };
        pad[j] = p;
        base[j] = total + p;
        total += p + len;
    }
    let s = ctx.stream();
    let mut d: Option<DeviceSlices> = None;
    if total > 0 {
        let mut d_buf: CudaSlice<u8> = s.alloc_zeros(total)?;
        // The pad source: `slice - 1` bytes of 0xFF then one 0; a pad of `p`
        // bytes is its last `p` bytes, so the 0 sits just before the proof.
        let mut pad_src = vec![0xFFu8; slice];
        pad_src[slice - 1] = 0;
        let d_pad: CudaSlice<u8> = s.clone_htod(&pad_src)?;
        for j in 0..n {
            let len = proofs[j].len();
            if len == 0 {
                continue;
            }
            let p = pad[j];
            if p > 0 {
                s.memcpy_dtod(
                    &d_pad.slice(slice - p..),
                    &mut d_buf.slice_mut(base[j] - p..base[j]),
                )?;
            }
            s.memcpy_htod(proofs[j], &mut d_buf.slice_mut(base[j]..base[j] + len))?;
        }
        s.synchronize()?;
        st.t_upload = t.elapsed();
        st.padded_bytes = total;
        let t = Instant::now();
        d = Some(ctx.decode_lrat_slices_device(
            &d_buf,
            total,
            slice as u32,
            opts.cooperative,
            opts.scan_blocks,
            true,
        )?);
        drop(d_buf);
        st.t_scan = t.elapsed();
    }
    if let Some(d) = &d {
        st.timing = d.timing;
        st.slices = d.n_slices;
        st.dirty = d.flags.iter().filter(|&&f| f == DEC_DIRTY).count();
    }

    // ---- (3) per step: records, plan, route ----
    let t = Instant::now();
    let mut routes: Vec<Route> = par_map(n, threads, |j| {
        let bytes = proofs[j];
        let len = bytes.len();
        let Some(d) = d.as_ref() else {
            return host_route(j, RouteKind::HostEmpty);
        };
        if len == 0 {
            return host_route(j, RouteKind::HostEmpty);
        }
        let k_lo = (base[j] - pad[j]) / slice;
        let k_hi = (base[j] + len) / slice;
        let bounds: Vec<(usize, usize)> = (k_lo..k_hi)
            .map(|k| {
                let lo = (k * slice).max(base[j]) - base[j];
                let hi = (k + 1) * slice - base[j];
                (lo, hi)
            })
            .collect();
        let Some((chunks, bases)) = step_records(bytes, d, k_lo, k_hi, base[j] as u64, &bounds)
        else {
            return host_route(j, RouteKind::HostRecords);
        };
        match stitch_plan(&formula(j), bytes, &bounds, chunks, 1) {
            Ok(plan) if plan.fp.early.is_none() => Route::Dev { plan, bases },
            Ok(_) => host_route(j, RouteKind::HostEarly),
            Err(_) => host_route(j, RouteKind::HostPlanError),
        }
    });
    st.t_plan = t.elapsed();
    let (raw_lits, raw_hints) = match d.as_mut() {
        Some(d) => (d.d_lits.take(), d.d_hints.take()),
        None => (None, None),
    };
    drop(d);
    let dummy_i: CudaSlice<i32> = s.alloc_zeros(1)?;
    let dummy_u: CudaSlice<u64> = s.alloc_zeros(1)?;
    let raw_lits = raw_lits.unwrap_or(dummy_i);
    let raw_hints = raw_hints.unwrap_or(dummy_u);

    // ---- (4) + (5): layout on the host, gathers on the device; redone when
    // the gather finds a hint to an id never added ----
    let (meta, d_lits, d_hints, offs) = loop {
        let t = Instant::now();
        let mut offs: Vec<Off> = Vec::with_capacity(n);
        let (mut c, mut l, mut h, mut lemma, mut tab, mut lseg, mut hseg, mut xl, mut xh) = (
            0usize, cnf_lits, 0usize, 0usize, 0usize, 1usize, 0usize, cnf_lits, 0usize,
        );
        for r in &routes {
            let mut o = Off {
                c,
                l,
                h,
                lemma,
                tab,
                hseg,
                xl,
                xh,
                ..Default::default()
            };
            match r {
                Route::Dev { plan, bases } => {
                    let fp = &plan.fp;
                    let n_priv = fp.n_original as usize - n_cnf;
                    o.n_lemmas = fp.lemma_clause.len();
                    o.n_clauses = n_cnf + 2 + n_priv + o.n_lemmas;
                    o.n_lits = plan.lit_total;
                    o.n_hints = plan.hint_total;
                    o.n_tab = plan.dense_of.len();
                    o.n_lseg = 1 + plan.copies.len();
                    o.n_hseg = plan.copies.len();
                    o.n_xl = plan.orig_lits;
                    for cp in &plan.copies {
                        let dev = cp.chunk < bases.len() && bases[cp.chunk].is_some();
                        if !dev {
                            o.n_xl += cp.lit_hi - cp.lit_lo;
                            o.n_xh += cp.hint_hi - cp.hint_lo;
                        }
                    }
                }
                Route::Host(fp, _) => {
                    let n_priv = fp.n_original as usize - n_cnf;
                    o.n_lemmas = fp.lemma_clause.len();
                    o.n_clauses = n_cnf + 2 + n_priv + o.n_lemmas;
                    o.n_lits = fp.lits.len();
                    o.n_hints = fp.hints.len();
                    o.n_lseg = 1;
                    o.n_xl = fp.lits.len();
                }
                Route::Bad(..) => {}
            }
            c += o.n_clauses;
            l += o.n_lits;
            h += o.n_hints;
            lemma += o.n_lemmas;
            tab += o.n_tab;
            lseg += o.n_lseg;
            hseg += o.n_hseg;
            xl += o.n_xl;
            xh += o.n_xh;
            offs.push(o);
        }
        let (n_c, n_l, n_h, n_n, n_t, n_ls, n_hs, n_xl, n_xh) =
            (c, l, h, lemma, tab, lseg, hseg, xl, xh);
        let limit = (u32::MAX - 64) as usize;
        if n_c > limit || n_l > limit || n_h > limit || n_t > limit || n_xl > limit {
            return Err(GpuError::Cuda(
                "batch_device: chunk too large for the kernels' 32-bit indices".into(),
            ));
        }
        let mut clause_start = vec![0u32; n_c + 1];
        let mut deleted_at = vec![u32::MAX; n_c];
        let mut lemma_clause = vec![0u32; n_n];
        let mut lemma_step = vec![0u32; n_n];
        let mut hint_start = vec![0u32; n_n + 1];
        let mut table = vec![0u32; n_t];
        let mut lseg_dst = vec![0u32; n_ls];
        let mut lseg_src = vec![0u32; n_ls];
        let mut lseg_sel = vec![0u32; n_ls];
        let mut hseg_dst = vec![0u32; n_hs];
        let mut hseg_src = vec![0u32; n_hs];
        let mut hseg_sel = vec![0u32; n_hs];
        let mut extra_lits = vec![0i32; n_xl];
        let mut extra_hints = vec![0u64; n_xh];
        clause_start[n_c] = n_l as u32;
        // Segment 0: the shared CNF's literals.
        for (k, c) in cnf.iter().enumerate() {
            let a = cs_cnf[k] as usize;
            extra_lits[a..a + c.len()].copy_from_slice(c);
        }
        lseg_sel[0] = 1;
        // Carve every step's regions (disjoint by construction).
        let regions: Vec<Regions> = {
            let mut cs =
                carve(&mut clause_start[..n_c], offs.iter().map(|o| o.n_clauses)).into_iter();
            let mut del = carve(&mut deleted_at[..], offs.iter().map(|o| o.n_clauses)).into_iter();
            let mut lc = carve(&mut lemma_clause[..], offs.iter().map(|o| o.n_lemmas)).into_iter();
            let mut ls = carve(&mut lemma_step[..], offs.iter().map(|o| o.n_lemmas)).into_iter();
            let mut hs = carve(&mut hint_start[1..], offs.iter().map(|o| o.n_lemmas)).into_iter();
            let mut tb = carve(&mut table[..], offs.iter().map(|o| o.n_tab)).into_iter();
            let mut ld = carve(&mut lseg_dst[1..], offs.iter().map(|o| o.n_lseg)).into_iter();
            let mut lsr = carve(&mut lseg_src[1..], offs.iter().map(|o| o.n_lseg)).into_iter();
            let mut lse = carve(&mut lseg_sel[1..], offs.iter().map(|o| o.n_lseg)).into_iter();
            let mut hd = carve(&mut hseg_dst[..], offs.iter().map(|o| o.n_hseg)).into_iter();
            let mut hsr = carve(&mut hseg_src[..], offs.iter().map(|o| o.n_hseg)).into_iter();
            let mut hse = carve(&mut hseg_sel[..], offs.iter().map(|o| o.n_hseg)).into_iter();
            let mut xls =
                carve(&mut extra_lits[cnf_lits..], offs.iter().map(|o| o.n_xl)).into_iter();
            let mut xhs = carve(&mut extra_hints[..], offs.iter().map(|o| o.n_xh)).into_iter();
            (0..n)
                .map(|_| Regions {
                    cs: cs.next().unwrap(),
                    del: del.next().unwrap(),
                    lemma_clause: lc.next().unwrap(),
                    lemma_step: ls.next().unwrap(),
                    hint_start: hs.next().unwrap(),
                    tab: tb.next().unwrap(),
                    lseg_dst: ld.next().unwrap(),
                    lseg_src: lsr.next().unwrap(),
                    lseg_sel: lse.next().unwrap(),
                    hseg_dst: hd.next().unwrap(),
                    hseg_src: hsr.next().unwrap(),
                    hseg_sel: hse.next().unwrap(),
                    xl: xls.next().unwrap(),
                    xh: xhs.next().unwrap(),
                })
                .collect()
        };
        // Fill one step. Returns the host-route step's remapped hints.
        let fill = |j: usize, r: Regions| -> Option<Vec<u32>> {
            let o = offs[j];
            let batch_dense = |d: u32| -> u32 {
                if (d as usize) < n_cnf {
                    (o.c + d as usize) as u32
                } else {
                    (o.c + d as usize + 1) as u32
                }
            };
            // The index region: CNF entries, sentinel, privates, terminator.
            let index = |r: &mut Regions, fp: &FlatProof| {
                let n_priv = fp.n_original as usize - n_cnf;
                r.cs[..=n_cnf].copy_from_slice(&cs_cnf);
                for k in 0..=(n_priv + o.n_lemmas) {
                    r.cs[n_cnf + 1 + k] = (o.l + fp.clause_start[n_cnf + k] as usize) as u32;
                }
                r.del[..n_cnf].copy_from_slice(&fp.deleted_at[..n_cnf]);
                r.del[n_cnf] = u32::MAX;
                r.del[n_cnf + 1..n_cnf + 1 + n_priv + o.n_lemmas]
                    .copy_from_slice(&fp.deleted_at[n_cnf..]);
                r.del[n_cnf + 1 + n_priv + o.n_lemmas] = u32::MAX;
                for i in 0..o.n_lemmas {
                    r.lemma_clause[i] = batch_dense(fp.lemma_clause[i]);
                    r.lemma_step[i] = fp.lemma_step[i];
                    r.hint_start[i] = (o.h + fp.hint_start[i + 1] as usize) as u32;
                }
            };
            match &routes[j] {
                Route::Dev { plan, bases } => {
                    let mut r = r;
                    index(&mut r, &plan.fp);
                    for (k, &e) in plan.dense_of.iter().enumerate() {
                        r.tab[k] = if e == 0 { 0 } else { batch_dense(e - 1) + 1 };
                    }
                    // The extras' literals, then the true chain's segments.
                    let mut at = 0usize;
                    for c in &extras[j] {
                        r.xl[at..at + c.len()].copy_from_slice(c);
                        at += c.len();
                    }
                    debug_assert_eq!(at, plan.orig_lits);
                    r.lseg_dst[0] = o.l as u32;
                    r.lseg_src[0] = o.xl as u32;
                    r.lseg_sel[0] = 1;
                    let (mut lat, mut hat, mut xl_at, mut xh_at) = (at, 0usize, at, 0usize);
                    let n_slices_j = plan.chunks.len();
                    for (q, cp) in plan.copies.iter().enumerate() {
                        let nl = cp.lit_hi - cp.lit_lo;
                        let nh = cp.hint_hi - cp.hint_lo;
                        let dev = if cp.chunk < n_slices_j {
                            bases[cp.chunk]
                        } else {
                            None
                        };
                        r.lseg_dst[1 + q] = (o.l + lat) as u32;
                        r.hseg_dst[q] = hat as u32;
                        match dev {
                            Some((lb, hb)) => {
                                r.lseg_src[1 + q] = lb + cp.lit_lo as u32;
                                r.lseg_sel[1 + q] = 0;
                                r.hseg_src[q] = hb + cp.hint_lo as u32;
                                r.hseg_sel[q] = 0;
                            }
                            None => {
                                let ch = if cp.chunk < n_slices_j {
                                    &plan.chunks[cp.chunk]
                                } else {
                                    &plan.bridges[cp.chunk - n_slices_j]
                                };
                                r.xl[xl_at..xl_at + nl]
                                    .copy_from_slice(&ch.lits[cp.lit_lo..cp.lit_hi]);
                                r.lseg_src[1 + q] = (o.xl + xl_at) as u32;
                                r.lseg_sel[1 + q] = 1;
                                xl_at += nl;
                                r.xh[xh_at..xh_at + nh]
                                    .copy_from_slice(&ch.hints[cp.hint_lo..cp.hint_hi]);
                                r.hseg_src[q] = (o.xh + xh_at) as u32;
                                r.hseg_sel[q] = 1;
                                xh_at += nh;
                            }
                        }
                        lat += nl;
                        hat += nh;
                    }
                    debug_assert_eq!(lat, o.n_lits);
                    debug_assert_eq!(hat, o.n_hints);
                    None
                }
                Route::Host(fp, _) => {
                    let mut r = r;
                    index(&mut r, fp);
                    r.xl.copy_from_slice(&fp.lits);
                    r.lseg_dst[0] = o.l as u32;
                    r.lseg_src[0] = o.xl as u32;
                    r.lseg_sel[0] = 1;
                    Some(fp.hints.iter().map(|&h| batch_dense(h)).collect())
                }
                Route::Bad(..) => None,
            }
        };
        let weight =
            |o: &Off| 4 * (o.n_lits + o.n_hints + o.n_tab) + 24 * o.n_lemmas + 8 * o.n_clauses;
        let total_w: usize = offs.iter().map(weight).sum();
        let per = total_w / threads + 1;
        let host_hints: Vec<Option<Vec<u32>>> = {
            let slots: Vec<std::sync::Mutex<Option<Vec<u32>>>> =
                (0..n).map(|_| std::sync::Mutex::new(None)).collect();
            std::thread::scope(|sc| {
                let fill = &fill;
                let slots = &slots;
                let mut acc = 0usize;
                let mut group: Vec<(usize, Regions)> = Vec::new();
                for (j, r) in regions.into_iter().enumerate() {
                    acc += weight(&offs[j]);
                    group.push((j, r));
                    if acc >= per {
                        let g = std::mem::take(&mut group);
                        sc.spawn(move || {
                            for (j, r) in g {
                                *slots[j].lock().unwrap() = fill(j, r);
                            }
                        });
                        acc = 0;
                    }
                }
                if !group.is_empty() {
                    sc.spawn(move || {
                        for (j, r) in group {
                            *slots[j].lock().unwrap() = fill(j, r);
                        }
                    });
                }
            });
            slots.into_iter().map(|m| m.into_inner().unwrap()).collect()
        };
        st.t_layout += t.elapsed();

        // ---- the gathers ----
        let t = Instant::now();
        let one_u32 = [0u32];
        let up = |v: &[u32]| -> Result<CudaSlice<u32>, GpuError> {
            Ok(s.clone_htod(if v.is_empty() { &one_u32[..] } else { v })?)
        };
        let d_ld = up(&lseg_dst)?;
        let d_ls = up(&lseg_src)?;
        let d_lse = up(&lseg_sel)?;
        let d_hd = up(&hseg_dst)?;
        let d_hs = up(&hseg_src)?;
        let d_hse = up(&hseg_sel)?;
        let d_tab = up(&table)?;
        let one_i = [0i32];
        let one_u64 = [0u64];
        let d_xl: CudaSlice<i32> = s.clone_htod(if extra_lits.is_empty() {
            &one_i[..]
        } else {
            &extra_lits[..]
        })?;
        let d_xh: CudaSlice<u64> = s.clone_htod(if extra_hints.is_empty() {
            &one_u64[..]
        } else {
            &extra_hints[..]
        })?;
        drop(extra_lits);
        drop(extra_hints);
        drop(table);
        let mut d_lits: CudaSlice<i32> = s.alloc_zeros(n_l.max(1))?;
        let mut d_hints: CudaSlice<u32> = s.alloc_zeros(n_h.max(1))?;
        let mut d_bad: CudaSlice<u32> = s.clone_htod(&vec![u32::MAX; n.max(1)])?;
        ctx.gather_lits_into(
            &d_ld.slice(..),
            &d_ls.slice(..),
            &d_lse.slice(..),
            n_ls as u32,
            &raw_lits,
            &d_xl,
            &mut d_lits.slice_mut(..),
            n_l as u32,
            opts.gather_blocks,
        )?;
        for (j, r) in routes.iter().enumerate() {
            let o = offs[j];
            match r {
                Route::Dev { .. } => {
                    if o.n_hints == 0 || o.n_hseg == 0 {
                        continue;
                    }
                    let blocks = opts
                        .gather_blocks
                        .min(((o.n_hints as u32).div_ceil(32)).max(1));
                    ctx.gather_hints_into(
                        &d_hd.slice(o.hseg..o.hseg + o.n_hseg),
                        &d_hs.slice(o.hseg..o.hseg + o.n_hseg),
                        &d_hse.slice(o.hseg..o.hseg + o.n_hseg),
                        o.n_hseg as u32,
                        &raw_hints,
                        &d_xh,
                        &d_tab.slice(o.tab..o.tab + o.n_tab),
                        o.n_tab as u32,
                        &mut d_hints.slice_mut(o.h..o.h + o.n_hints),
                        o.n_hints as u32,
                        &mut d_bad.slice_mut(j..j + 1),
                        blocks,
                    )?;
                }
                Route::Host(..) => {
                    if o.n_hints > 0 {
                        let hh = host_hints[j].as_ref().expect("host-route hints");
                        s.memcpy_htod(&hh[..], &mut d_hints.slice_mut(o.h..o.h + o.n_hints))?;
                    }
                }
                Route::Bad(..) => {}
            }
        }
        let bad: Vec<u32> = s.clone_dtoh(&d_bad)?;
        st.t_gather += t.elapsed();
        let rerouted: Vec<usize> = (0..n)
            .filter(|&j| bad[j] != u32::MAX && matches!(routes[j], Route::Dev { .. }))
            .collect();
        if rerouted.is_empty() {
            let meta = FlatProof {
                lits: Vec::new(),
                clause_start,
                deleted_at,
                lemma_clause,
                lemma_step,
                lemma_id: Vec::new(),
                hint_start,
                hints: Vec::new(),
                first_empty: None,
                n_original: 0,
                early: None,
            };
            st.lemmas = n_n;
            st.hints = n_h;
            st.lits = n_l;
            st.clauses = n_c;
            st.table = n_t;
            break (meta, d_lits, d_hints, offs);
        }
        st.relayouts += 1;
        for j in rerouted {
            routes[j] = host_route(j, RouteKind::HostBadHint);
        }
    };
    drop(raw_lits);
    drop(raw_hints);
    st.host_route = routes
        .iter()
        .filter(|r| !matches!(r, Route::Dev { .. }))
        .count();
    Ok(PreparedChunk {
        routes,
        offs,
        meta,
        d_lits,
        d_hints,
        n_cnf,
        check_blocks: opts.check_blocks,
        stats: st,
    })
}

/// Steps (6)–(7): the check launch once over the prepared chunk on `ctx`'s
/// stream, then the verdicts per step. `ctx` may be a [`GpuContext::fork`]
/// sibling of the preparing handle: the arrays are complete when
/// [`prepare_chunk`] returns.
pub fn check_prepared(ctx: &GpuContext, p: PreparedChunk) -> Result<ChunkOutcome, GpuError> {
    let PreparedChunk {
        routes,
        offs,
        meta,
        d_lits,
        d_hints,
        n_cnf,
        check_blocks: blocks,
        mut stats,
    } = p;
    let n = routes.len();

    // ---- (6) the check, once ----
    let t = Instant::now();
    let raw = ctx.check_lrat_codes_mode(&meta, &d_lits, &d_hints, blocks, None)?;
    stats.t_check = t.elapsed();

    // ---- (7) verdicts per step ----
    let t = Instant::now();
    let codes: Vec<LemmaCode> = raw.iter().map(|&c| code_of(c)).collect();
    let mut verdicts = Vec::with_capacity(n);
    let mut per_step = Vec::with_capacity(n);
    let mut kinds = Vec::with_capacity(n);
    for (j, r) in routes.iter().enumerate() {
        let o = offs[j];
        let cj = &codes[o.lemma..o.lemma + o.n_lemmas];
        let (v, k) = match r {
            Route::Dev { plan, .. } => (
                Ok(combine_dev(ctx, &plan.fp, cj, &d_hints, &o, n_cnf)?),
                RouteKind::Device,
            ),
            Route::Host(fp, k) => (Ok(combine(fp, cj)), *k),
            Route::Bad(e, k) => (Err(e.clone()), *k),
        };
        verdicts.push(v);
        per_step.push(cj.to_vec());
        kinds.push(k);
    }
    stats.t_combine = t.elapsed();
    Ok(ChunkOutcome {
        verdicts,
        codes: per_step,
        routes: kinds,
        stats,
    })
}

/// One chunk's intervals in the pipeline, seconds from the call's start.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipeTimes {
    pub read: (f64, f64),
    pub prepare: (f64, f64),
    pub check: (f64, f64),
}

/// The two-stream pipeline (design C2, `research/edp-record.md` §8): chunk
/// `k+1` is read and prepared while chunk `k` is checked. Three threads —
/// a reader (`load(j)` for each step of the next chunk, on `opts.threads`
/// workers), a preparer on a [`GpuContext::fork`] of `ctx`, and the checker
/// on `ctx` itself — joined by rendezvous channels, so at most one chunk
/// waits prepared. `chunks` are consecutive step ranges `lo..hi` covering
/// the steps in order; `extras[j]` is step `j`'s formula beyond `cnf`.
/// Outcomes per chunk, in chunk order, with each chunk's intervals. A
/// failure on any thread ends the pipeline with that error.
pub fn check_pipelined(
    ctx: &GpuContext,
    cnf: &[Vec<i32>],
    extras: &[Vec<Vec<i32>>],
    chunks: &[(usize, usize)],
    load: impl Fn(usize) -> Vec<u8> + Sync,
    opts: &BatchOpts,
) -> Result<Vec<(ChunkOutcome, PipeTimes)>, GpuError> {
    use std::sync::mpsc::sync_channel;
    let t0 = Instant::now();
    let at = |t: Instant| t.duration_since(t0).as_secs_f64();
    let prep_ctx = ctx.fork()?;
    let (tx_bytes, rx_bytes) = sync_channel::<(usize, Vec<Vec<u8>>, (f64, f64))>(1);
    let (tx_prep, rx_prep) =
        sync_channel::<Result<(usize, PreparedChunk, (f64, f64), (f64, f64)), GpuError>>(0);
    let threads = opts.threads.max(1);
    std::thread::scope(|s| -> Result<Vec<(ChunkOutcome, PipeTimes)>, GpuError> {
        // Reader.
        let load = &load;
        s.spawn(move || {
            for (k, &(lo, hi)) in chunks.iter().enumerate() {
                let a = Instant::now();
                let bytes = par_map(hi - lo, threads, |i| load(lo + i));
                let iv = (at(a), at(Instant::now()));
                if tx_bytes.send((k, bytes, iv)).is_err() {
                    return;
                }
            }
        });
        // Preparer.
        let prep_ctx = &prep_ctx;
        s.spawn(move || {
            for (k, bytes, read) in rx_bytes {
                let (lo, hi) = chunks[k];
                let proofs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
                let a = Instant::now();
                let r = prepare_chunk(prep_ctx, cnf, &extras[lo..hi], &proofs, opts);
                let iv = (at(a), at(Instant::now()));
                let msg = r.map(|p| (k, p, read, iv));
                let stop = msg.is_err();
                if tx_prep.send(msg).is_err() || stop {
                    return;
                }
            }
        });
        // Checker (this thread).
        let mut out = Vec::with_capacity(chunks.len());
        for msg in rx_prep {
            let (k, p, read, prepare) = msg?;
            debug_assert_eq!(k, out.len(), "chunks arrive in order");
            let a = Instant::now();
            let o = check_prepared(ctx, p)?;
            let check = (at(a), at(Instant::now()));
            out.push((
                o,
                PipeTimes {
                    read,
                    prepare,
                    check,
                },
            ));
        }
        Ok(out)
    })
}

/// [`combine`] for a device-route step: when the first failing lemma's reason
/// names a hint, that lemma's hint range is fetched from the device and mapped
/// back to the step's own dense indices, so the verdict carries the hint id
/// exactly as the host route's does.
fn combine_dev(
    ctx: &GpuContext,
    fp: &FlatProof,
    codes: &[LemmaCode],
    d_hints: &CudaSlice<u32>,
    o: &Off,
    n_cnf: usize,
) -> Result<Verdict, GpuError> {
    let first = codes.iter().position(|&c| c != LemmaCode::Ok);
    match first {
        Some(i)
            if codes.len() == fp.lemma_clause.len()
                && matches!(codes[i], LemmaCode::DeletedHint | LemmaCode::ForwardHint) =>
        {
            let (hs, he) = (fp.hint_start[i] as usize, fp.hint_start[i + 1] as usize);
            let mut meta = fp.clone();
            meta.hints = vec![0u32; he];
            if he > hs {
                let part: Vec<u32> = ctx
                    .stream()
                    .clone_dtoh(&d_hints.slice(o.h + hs..o.h + he))?;
                for (dst, &b) in meta.hints[hs..he].iter_mut().zip(&part) {
                    let b = b as usize;
                    *dst = if b < o.c + n_cnf {
                        (b - o.c) as u32
                    } else {
                        (b - o.c - 1) as u32
                    };
                }
            }
            Ok(combine(&meta, codes))
        }
        _ => Ok(combine(fp, codes)),
    }
}

#[cfg(test)]
mod tests {
    //! Every step through the batched route against the one-at-a-time
    //! reference (`from_binary` on the real formula, SimWarp codes, `combine`),
    //! codes and verdicts both. Needs a GPU; skipped with a message otherwise.
    use super::*;
    use crate::lrat::{parse_text, write_binary};
    use crate::lrat_warp::check_warp_codes;

    fn ctx() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("batch_device tests: SKIPPED (no GPU context: {e})");
                None
            }
        }
    }

    // Shared CNF: 1:(1 2) 2:(-1 2) 3:(1 -2) — satisfiable (both true).
    fn cnf() -> Vec<Vec<i32>> {
        vec![vec![1, 2], vec![-1, 2], vec![1, -2]]
    }

    fn bin(p: &str) -> Vec<u8> {
        write_binary(&parse_text(p).unwrap())
    }

    /// Run a chunk and hold every step to the reference. Returns the outcome
    /// for further assertions.
    fn agree(
        ctx: &GpuContext,
        cnf: &[Vec<i32>],
        extras: &[Vec<Vec<i32>>],
        proofs: &[Vec<u8>],
        opts: &BatchOpts,
        what: &str,
    ) -> ChunkOutcome {
        let ps: Vec<&[u8]> = proofs.iter().map(|p| p.as_slice()).collect();
        let out = check_chunk(ctx, cnf, extras, &ps, opts).unwrap();
        assert_eq!(out.verdicts.len(), ps.len());
        for j in 0..ps.len() {
            let mut f = cnf.to_vec();
            f.extend(extras[j].iter().cloned());
            match FlatProof::from_binary(&f, ps[j]) {
                Ok(fp) => {
                    let codes = check_warp_codes(&fp);
                    let want = combine(&fp, &codes);
                    assert_eq!(out.codes[j], codes, "{what}: step {j} codes");
                    assert_eq!(
                        out.verdicts[j].as_ref().unwrap(),
                        &want,
                        "{what}: step {j} verdict"
                    );
                }
                Err(e) => assert_eq!(
                    out.verdicts[j].as_ref().unwrap_err(),
                    &e,
                    "{what}: step {j} error"
                ),
            }
        }
        out
    }

    fn opts(slice: u32, threads: usize) -> BatchOpts {
        BatchOpts {
            slice,
            threads,
            scan_blocks: 64,
            gather_blocks: 64,
            ..Default::default()
        }
    }

    /// The `lrat_batch` shapes in one chunk: cube steps, a step deleting a
    /// shared clause, the mutants, many extras. All well formed: every step
    /// takes the device route.
    #[test]
    fn batch_device_agrees_on_the_lrat_batch_shapes() {
        let Some(ctx) = ctx() else { return };
        let u2 = vec![vec![-2]];
        let u1 = vec![vec![-1]];
        let many = vec![vec![-2], vec![2, -1], vec![1]];
        let steps: Vec<(Vec<Vec<i32>>, &str)> = vec![
            (u2.clone(), "5 0 4 1 2 0\n"),
            (u1.clone(), "5 0 4 1 3 0\n"),
            (u2.clone(), "5 d 1 0\n6 0 4 1 2 0\n"),
            (u2.clone(), "5 0 4 1 2 0\n"),
            (u2.clone(), "5 d 3 0\n6 0 4 1 2 0\n"),
            (u2.clone(), "5 0 1 2 0\n"),
            (u2.clone(), "5 0 4 1 0\n"),
            (u2.clone(), "5 0 4 6 2 0\n6 0 4 1 2 0\n"),
            (u2.clone(), "5 1 0 4 1 0\n"),
            (many.clone(), "7 0 4 5 6 0\n"),
            (u1.clone(), "5 0 4 1 3 0\n"),
        ];
        let extras: Vec<Vec<Vec<i32>>> = steps.iter().map(|(e, _)| e.clone()).collect();
        let proofs: Vec<Vec<u8>> = steps.iter().map(|(_, p)| bin(p)).collect();
        for slice in [7u32, 64, 4096] {
            for threads in [1usize, 3] {
                let out = agree(
                    &ctx,
                    &cnf(),
                    &extras,
                    &proofs,
                    &opts(slice, threads),
                    &format!("shapes s={slice} t={threads}"),
                );
                assert_eq!(
                    out.stats.host_route, 0,
                    "s={slice}: a well-formed step left the device route: {:?}",
                    out.routes
                );
                assert_eq!(out.stats.relayouts, 0);
                assert!(out.routes.iter().all(|r| *r == RouteKind::Device));
                assert!(out.verdicts[0].as_ref().unwrap().is_verified());
                assert!(
                    matches!(
                        out.verdicts[2],
                        Ok(Verdict::Rejected {
                            id: 6,
                            reason: crate::lrat::Reason::DeletedHint(1),
                            ..
                        })
                    ),
                    "{:?}",
                    out.verdicts[2]
                );
                assert!(matches!(out.verdicts[8], Ok(Verdict::NoEmptyClause { .. })));
                assert!(out.verdicts[9].as_ref().unwrap().is_verified());
            }
        }
    }

    /// Malformed and structurally rejected steps take the host route and
    /// still get the reference's answer; a hint to an id never added is found
    /// by the gather and re-routed; valid steps beside them stay on the
    /// device route.
    #[test]
    fn malformed_steps_take_the_host_route() {
        let Some(ctx) = ctx() else { return };
        let good = bin("5 0 4 1 2 0\n");
        let bad_hint = bin("5 0 4 99 0\n");
        let bad_del = bin("5 d 77 0\n6 0 4 1 2 0\n");
        let id_down = bin("5 0 4 1 2 0\n4 0 4 1 2 0\n");
        let rat = bin("5 0 4 -1 2 0\n");
        let mut flipped = good.clone();
        flipped[3] ^= 0x80;
        let u2 = vec![vec![-2]];
        let mut extras = Vec::new();
        let mut proofs = Vec::new();
        let mut expect_host = 0usize;
        let push = |extras: &mut Vec<_>, proofs: &mut Vec<Vec<u8>>, p: Vec<u8>| {
            extras.push(u2.clone());
            proofs.push(p);
        };
        push(&mut extras, &mut proofs, good.clone());
        push(&mut extras, &mut proofs, Vec::new());
        expect_host += 1;
        push(&mut extras, &mut proofs, bad_hint.clone());
        expect_host += 1;
        push(&mut extras, &mut proofs, good.clone());
        push(&mut extras, &mut proofs, bad_del.clone());
        expect_host += 1;
        push(&mut extras, &mut proofs, id_down.clone());
        expect_host += 1;
        push(&mut extras, &mut proofs, rat.clone());
        expect_host += 1;
        push(&mut extras, &mut proofs, flipped.clone());
        expect_host += 1;
        for cut in 1..good.len() {
            push(&mut extras, &mut proofs, good[..cut].to_vec());
            expect_host += 1;
        }
        push(&mut extras, &mut proofs, good.clone());
        for slice in [5u32, 64, 4096] {
            let out = agree(
                &ctx,
                &cnf(),
                &extras,
                &proofs,
                &opts(slice, 2),
                &format!("malformed s={slice}"),
            );
            assert_eq!(
                out.stats.host_route, expect_host,
                "s={slice}: host-route count; routes {:?}",
                out.routes
            );
            assert_eq!(
                out.stats.relayouts, 1,
                "s={slice}: one re-layout for the bad hints"
            );
            assert_eq!(out.routes[0], RouteKind::Device);
            assert_eq!(out.routes[1], RouteKind::HostEmpty);
            assert_eq!(out.routes[2], RouteKind::HostBadHint);
            assert_eq!(out.routes[3], RouteKind::Device);
            assert_eq!(out.routes[4], RouteKind::HostEarly, "bad deletion");
            assert_eq!(out.routes[5], RouteKind::HostEarly, "id not increasing");
            assert_eq!(out.routes[6], RouteKind::HostEarly, "RAT");
            assert_eq!(
                out.routes[7],
                RouteKind::HostBadHint,
                "flipped byte = hint 132"
            );
            assert!(out.routes[8..8 + good.len() - 1]
                .iter()
                .all(|r| *r == RouteKind::HostPlanError));
            assert_eq!(*out.routes.last().unwrap(), RouteKind::Device);
            eprintln!(
                "malformed s={slice}: bad hint → {:?}; RAT → {:?}",
                out.verdicts[2], out.verdicts[6]
            );
            assert!(matches!(out.verdicts[6], Ok(Verdict::Unsupported { .. })));
        }
    }

    /// The two-stream pipeline (§8) over the `lrat_batch` shapes with
    /// malformed and rejected steps mixed in, cut into 1-, 2- and 3-step
    /// chunks: every step's codes and verdict equal the one-at-a-time
    /// reference, chunk outcomes come back in chunk order and cover the
    /// ranges, and the well-formed steps stay on the device route.
    #[test]
    fn pipelined_route_agrees_with_the_reference() {
        let Some(ctx) = ctx() else { return };
        let u2 = vec![vec![-2]];
        let u1 = vec![vec![-1]];
        let many = vec![vec![-2], vec![2, -1], vec![1]];
        let good = bin("5 0 4 1 2 0\n");
        let mut flipped = good.clone();
        flipped[3] ^= 0x80;
        let steps: Vec<(Vec<Vec<i32>>, Vec<u8>)> = vec![
            (u2.clone(), bin("5 0 4 1 2 0\n")),
            (u1.clone(), bin("5 0 4 1 3 0\n")),
            (u2.clone(), bin("5 d 1 0\n6 0 4 1 2 0\n")),
            (u2.clone(), Vec::new()),
            (u2.clone(), bin("5 0 4 99 0\n")),
            (u2.clone(), bin("5 d 3 0\n6 0 4 1 2 0\n")),
            (u2.clone(), bin("5 0 1 2 0\n")),
            (u2.clone(), bin("5 0 4 -1 2 0\n")),
            (u2.clone(), bin("5 0 4 6 2 0\n6 0 4 1 2 0\n")),
            (u2.clone(), flipped),
            (u2.clone(), bin("5 1 0 4 1 0\n")),
            (many.clone(), bin("7 0 4 5 6 0\n")),
            (u2.clone(), good[..5].to_vec()),
            (u1.clone(), bin("5 0 4 1 3 0\n")),
        ];
        let extras: Vec<Vec<Vec<i32>>> = steps.iter().map(|(e, _)| e.clone()).collect();
        let proofs: Vec<Vec<u8>> = steps.iter().map(|(_, p)| p.clone()).collect();
        let n = steps.len();
        let device_steps = [0usize, 1, 2, 5, 6, 8, 10, 11, 13];
        for per_chunk in [1usize, 2, 3] {
            let chunks: Vec<(usize, usize)> = (0..n)
                .step_by(per_chunk)
                .map(|lo| (lo, (lo + per_chunk).min(n)))
                .collect();
            let out = check_pipelined(
                &ctx,
                &cnf(),
                &extras,
                &chunks,
                |j| proofs[j].clone(),
                &opts(64, 2),
            )
            .unwrap();
            assert_eq!(out.len(), chunks.len(), "per_chunk={per_chunk}: one outcome per chunk");
            let mut j = 0usize;
            for (k, (o, t)) in out.iter().enumerate() {
                let (lo, hi) = chunks[k];
                assert_eq!(o.verdicts.len(), hi - lo, "per_chunk={per_chunk}: chunk {k} size");
                assert!(t.read.1 <= t.prepare.1 && t.prepare.1 <= t.check.1);
                for i in 0..hi - lo {
                    let mut f = cnf();
                    f.extend(extras[j].iter().cloned());
                    match FlatProof::from_binary(&f, &proofs[j]) {
                        Ok(fp) => {
                            let codes = check_warp_codes(&fp);
                            assert_eq!(o.codes[i], codes, "per_chunk={per_chunk}: step {j} codes");
                            assert_eq!(
                                o.verdicts[i].as_ref().unwrap(),
                                &combine(&fp, &codes),
                                "per_chunk={per_chunk}: step {j} verdict"
                            );
                        }
                        Err(e) => assert_eq!(
                            o.verdicts[i].as_ref().unwrap_err(),
                            &e,
                            "per_chunk={per_chunk}: step {j} error"
                        ),
                    }
                    assert_eq!(
                        o.routes[i] == RouteKind::Device,
                        device_steps.contains(&j),
                        "per_chunk={per_chunk}: step {j} route {:?}",
                        o.routes[i]
                    );
                    j += 1;
                }
            }
            assert_eq!(j, n);
        }
    }

    /// Long proofs on the synthetic CNF: steps straddling many slices, the
    /// padding before every step, the chunk's first slice starting in padding,
    /// truncations of a middle step; both scan kernels.
    #[test]
    fn long_steps_straddle_slices_and_padding() {
        let Some(ctx) = ctx() else { return };
        let (cnf, bytes) = crate::lrat_warp::tests::synthetic_proof(true);
        let (_, bytes_no_empty) = crate::lrat_warp::tests::synthetic_proof(false);
        let mut proofs: Vec<Vec<u8>> = vec![bytes.clone(), bytes_no_empty.clone(), bytes.clone()];
        for cut in (1..bytes.len()).step_by(29) {
            proofs.push(bytes[..cut].to_vec());
        }
        proofs.push(bytes.clone());
        let extras: Vec<Vec<Vec<i32>>> = vec![Vec::new(); proofs.len()];
        for coop in [true, false] {
            for slice in [13u32, 64, 4096] {
                let mut o = opts(slice, 4);
                o.cooperative = coop;
                let out = agree(
                    &ctx,
                    &cnf,
                    &extras,
                    &proofs,
                    &o,
                    &format!("synthetic s={slice} coop={coop}"),
                );
                eprintln!(
                    "synthetic s={slice} coop={coop}: {:?} / {:?}; routes {:?}",
                    out.verdicts[0], out.verdicts[1], out.routes
                );
                // The four whole proofs stay on the device route.
                for j in [0usize, 1, 2, proofs.len() - 1] {
                    assert_eq!(
                        out.routes[j],
                        RouteKind::Device,
                        "s={slice} coop={coop}: step {j}"
                    );
                }
            }
        }
    }
}
