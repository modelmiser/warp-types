//! `edp_resident` — the compositional check through the batched
//! device-resident route (C of `research/edp-record.md` §6): per chunk, every
//! step's bytes go up once, the scan / gathers / check run on the card, and
//! the host sees step records and codes only.
//!
//! ```text
//! edp_resident CNF COMP DIR [--pattern s{}.blrat] [--threads 6] [--chunk-gb 12]
//!              [--slice 4096] [--scan-blocks 4096] [--gather-blocks 4096]
//!              [--check-blocks 0] [--steps N] [--simwarp K] [--verify-b2]
//! ```
//!
//! Step `j` is the CNF plus one unit per literal of the negated lemma `j`;
//! the last step (the line `0`) is the CNF plus every earlier lemma.
//! `--simwarp K` holds K evenly spaced steps' codes to the one-at-a-time host
//! decode against the real formula + SimWarp. `--verify-b2` re-runs every
//! chunk through B2 (host decode, `batch_par`, the non-resident device check)
//! and compares every code of every step — the O1 oracle.
//!
//! Chunks are cut by an estimated device footprint: ≈ 4.5 bytes of device
//! memory per proof byte at the gather's peak (raw u64 hints + gathered
//! hints + both literal arrays), plus the per-step index region and table.

use std::time::Instant;
use warp_types_sat::gpu_launcher::GpuContext;
use warp_types_sat::lrat::{parse_cnf_ints, Verdict};
use warp_types_sat::lrat_warp::batch_device::{check_chunk, BatchOpts};
use warp_types_sat::lrat_warp::{check_warp_codes, FlatProof, LemmaCode};

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<T>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
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

fn codes_of(raw: &[u8]) -> Vec<LemmaCode> {
    raw.iter()
        .map(|&c| match c {
            0 => LemmaCode::Ok,
            1 => LemmaCode::NotUnit,
            2 => LemmaCode::NoConflict,
            3 => LemmaCode::DeletedHint,
            4 => LemmaCode::ForwardHint,
            5 => LemmaCode::TableOverflow,
            _ => LemmaCode::Unwritten,
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: edp_resident CNF COMP DIR [--pattern s{{}}.blrat] [--threads 6] [--chunk-gb 12] [--slice 4096] [--scan-blocks 4096] [--gather-blocks 4096] [--check-blocks 0] [--steps N] [--simwarp K] [--verify-b2]");
        std::process::exit(1);
    }
    let pattern: String = arg(&args, "--pattern", "s{}.blrat".to_string());
    let threads: usize = arg(&args, "--threads", 6);
    let chunk_gb: f64 = arg(&args, "--chunk-gb", 12.0);
    let simwarp: usize = arg(&args, "--simwarp", 0);
    let verify_b2 = args.iter().any(|a| a == "--verify-b2");
    let opts = BatchOpts {
        slice: arg(&args, "--slice", 4096),
        scan_blocks: arg(&args, "--scan-blocks", 4096),
        gather_blocks: arg(&args, "--gather-blocks", 4096),
        check_blocks: arg(&args, "--check-blocks", 0),
        threads,
        cooperative: true,
    };

    let t0 = Instant::now();
    let cnf = parse_cnf_ints(&std::fs::read_to_string(&args[0]).expect("read cnf"));
    let n_cnf = cnf.len();
    let comp: Vec<Vec<i32>> = std::fs::read_to_string(&args[1])
        .expect("read comp")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<i32>().expect("comp literal"))
                .take_while(|&v| v != 0)
                .collect()
        })
        .collect();
    let n_steps_all = comp.len();
    let n_steps: usize = arg(&args, "--steps", n_steps_all).min(n_steps_all);
    let extras: Vec<Vec<Vec<i32>>> = (0..n_steps)
        .map(|j| {
            if comp[j].is_empty() {
                comp[..j].to_vec()
            } else {
                comp[j].iter().map(|&l| vec![-l]).collect()
            }
        })
        .collect();
    let t1 = Instant::now();
    println!(
        "c cnf {} vars? clauses {n_cnf}; comp {n_steps_all} lines; checking {n_steps} steps; threads {threads}; slice {} scan-blocks {} gather-blocks {} check-blocks {}",
        cnf.iter().flatten().map(|l| l.unsigned_abs()).max().unwrap_or(0),
        opts.slice,
        opts.scan_blocks,
        opts.gather_blocks,
        opts.check_blocks
    );

    // Read every proof.
    let bytes: Vec<Vec<u8>> = par_map(n_steps, threads, |j| {
        let path = format!(
            "{}/{}",
            args[2],
            pattern.replace("{}", &(j + 1).to_string())
        );
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    });
    let total_bytes: usize = bytes.iter().map(Vec::len).sum();
    let t2 = Instant::now();

    let t_init0 = Instant::now();
    let ctx = GpuContext::new().expect("GPU context");
    let t_init = t_init0.elapsed().as_secs_f64();

    // Chunk by estimated footprint.
    let est = |j: usize| -> usize {
        (4.5 * bytes[j].len() as f64) as usize + 12 * n_cnf + 32 * extras[j].len() + (1 << 10)
    };
    let budget = (chunk_gb * 1e9) as usize;
    let mut verdicts: Vec<Result<Verdict, _>> = Vec::with_capacity(n_steps);
    let mut codes_all: Vec<Vec<LemmaCode>> = Vec::with_capacity(n_steps);
    let mut chunks = 0usize;
    let (mut t_pad, mut t_scan, mut t_plan, mut t_layout, mut t_gather, mut t_check, mut t_combine) =
        (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut p1, mut px, mut p2, mut dl) = (0.0f64, 0.0, 0.0, 0.0);
    let (
        mut lemmas,
        mut hints,
        mut lits,
        mut clauses,
        mut table,
        mut host_route,
        mut relayouts,
        mut dirty,
    ) = (
        0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
    );
    let mut b2_mismatch = 0usize;
    let mut shown_routes = 0usize;
    let mut lo = 0usize;
    while lo < n_steps {
        let mut hi = lo;
        let mut used = 0usize;
        while hi < n_steps && (hi == lo || used + est(hi) <= budget) {
            used += est(hi);
            hi += 1;
        }
        let proofs: Vec<&[u8]> = bytes[lo..hi].iter().map(Vec::as_slice).collect();
        let a = Instant::now();
        let out = check_chunk(&ctx, &cnf, &extras[lo..hi], &proofs, &opts).expect("chunk");
        let wall = a.elapsed().as_secs_f64();
        let s = out.stats;
        chunks += 1;
        println!(
            "c chunk {chunks}: steps {}..={} ({} lemmas, {} hints, {:.2} GB bytes, est {:.2} GB) wall {wall:.3}s = upload {:.3} + scan {:.3} (p1 {:.3} px {:.3} p2 {:.3} dl {:.3}) + plan {:.3} + layout {:.3} + gather {:.3} + check {:.3} + combine {:.3}; host-route {} relayouts {} dirty {}/{}",
            lo + 1,
            hi,
            s.lemmas,
            s.hints,
            s.padded_bytes as f64 / 1e9,
            used as f64 / 1e9,
            s.t_upload.as_secs_f64(),
            s.t_scan.as_secs_f64(),
            s.timing.pass1.as_secs_f64(),
            s.timing.prefix.as_secs_f64(),
            s.timing.pass2.as_secs_f64(),
            s.timing.download.as_secs_f64(),
            s.t_plan.as_secs_f64(),
            s.t_layout.as_secs_f64(),
            s.t_gather.as_secs_f64(),
            s.t_check.as_secs_f64(),
            s.t_combine.as_secs_f64(),
            s.host_route,
            s.relayouts,
            s.dirty,
            s.slices
        );
        t_pad += s.t_upload.as_secs_f64();
        t_scan += s.t_scan.as_secs_f64();
        p1 += s.timing.pass1.as_secs_f64();
        px += s.timing.prefix.as_secs_f64();
        p2 += s.timing.pass2.as_secs_f64();
        dl += s.timing.download.as_secs_f64();
        t_plan += s.t_plan.as_secs_f64();
        t_layout += s.t_layout.as_secs_f64();
        t_gather += s.t_gather.as_secs_f64();
        t_check += s.t_check.as_secs_f64();
        t_combine += s.t_combine.as_secs_f64();
        lemmas += s.lemmas;
        hints += s.hints;
        lits += s.lits;
        clauses += s.clauses;
        table += s.table;
        host_route += s.host_route;
        relayouts += s.relayouts;
        dirty += s.dirty;
        // O1: the same chunk through B2, every code compared.
        if verify_b2 {
            let a = Instant::now();
            let holder: Vec<Vec<i32>> = vec![Vec::new(); n_cnf];
            let decoded: Vec<FlatProof> = par_map(hi - lo, threads, |k| {
                let mut f = holder.clone();
                f.extend(extras[lo + k].iter().cloned());
                FlatProof::from_binary(&f, proofs[k]).expect("B2 decode")
            });
            let b = warp_types_sat::lrat_batch::batch_par(&cnf, &decoded, threads);
            let b2 = codes_of(&ctx.check_lrat_codes(&b.flat, 4096).expect("B2 check"));
            let mut bad = 0usize;
            for (k, &(l, h)) in b.lemma_range.iter().enumerate() {
                if b2[l..h] != out.codes[k][..] {
                    bad += 1;
                    if bad <= 5 {
                        println!("c B2 MISMATCH at step {}", lo + k + 1);
                    }
                }
            }
            b2_mismatch += bad;
            println!(
                "c chunk {chunks} verify-b2: {} steps compared, {bad} mismatches ({:.3}s)",
                hi - lo,
                a.elapsed().as_secs_f64()
            );
        }
        for (k, r) in out.routes.iter().enumerate() {
            if *r != warp_types_sat::lrat_warp::batch_device::RouteKind::Device && shown_routes < 20
            {
                shown_routes += 1;
                println!(
                    "c step {} took the host route: {r:?} ({} bytes)",
                    lo + k + 1,
                    bytes[lo + k].len()
                );
            }
        }
        verdicts.extend(out.verdicts);
        codes_all.extend(out.codes);
        lo = hi;
    }
    let t4 = Instant::now();

    // Report.
    let ok = verdicts
        .iter()
        .filter(|v| v.as_ref().map_or(false, Verdict::is_verified))
        .count();
    let unsupported = verdicts
        .iter()
        .filter(|v| matches!(v, Ok(Verdict::Unsupported { .. })))
        .count();
    let malformed = verdicts.iter().filter(|v| v.is_err()).count();
    for (j, v) in verdicts
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.as_ref().map_or(false, Verdict::is_verified))
        .take(10)
    {
        println!("c step {} NOT verified: {v:?}", j + 1);
    }
    println!("c proofs {total_bytes} bytes; lemmas {lemmas}; hints {hints}; lits {lits}; clause entries {clauses}; table entries {table}; host-route steps {host_route}; relayouts {relayouts}; dirty slices {dirty}");
    println!(
        "c times: cnf+comp {:.3}s | read proofs {:.3}s | gpu init {t_init:.3}s | upload {t_pad:.3}s | scan {t_scan:.3}s (pass1 {p1:.3} prefix {px:.3} pass2 {p2:.3} download {dl:.3}) | plan {t_plan:.3}s | layout {t_layout:.3}s | gather {t_gather:.3}s | check {t_check:.3}s | combine {t_combine:.3}s | total {:.3}s",
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        (t4 - t0).as_secs_f64()
    );
    println!(
        "s {} ({ok}/{n_steps} steps verified, {unsupported} unsupported, {malformed} malformed, {chunks} chunks, resident mode)",
        if ok == n_steps {
            "VERIFIED"
        } else {
            "NOT VERIFIED"
        }
    );
    if verify_b2 {
        println!("c verify-b2: {b2_mismatch} mismatching steps");
    }

    // O2: K steps re-decoded against the real formula, SimWarp one at a time.
    if simwarp > 0 {
        let k = simwarp.min(n_steps);
        let mut mismatches = 0;
        for i in 0..k {
            let j = i * n_steps / k;
            let mut f = cnf.clone();
            f.extend(extras[j].iter().cloned());
            let fp = FlatProof::from_binary(&f, &bytes[j]).expect("decode");
            let one = check_warp_codes(&fp);
            if one != codes_all[j] {
                mismatches += 1;
                println!(
                    "c CONTRACT MISMATCH at step {}: one-at-a-time {:?} vs resident {:?} non-Ok",
                    j + 1,
                    one.iter().filter(|c| **c != LemmaCode::Ok).count(),
                    codes_all[j].iter().filter(|c| **c != LemmaCode::Ok).count()
                );
            }
        }
        println!("c simwarp contract: {k} steps compared, {mismatches} mismatches");
        if mismatches > 0 {
            std::process::exit(3);
        }
    }
    if b2_mismatch > 0 {
        std::process::exit(3);
    }
}
