//! `edp_batch` — the batched compositional check (B2 of `research/edp-record.md`):
//! the shared CNF parsed once, every step's binary LRAT read and decoded in
//! parallel, all steps laid out as one flat proof per chunk, the per-lemma
//! check run once per chunk (device with `--features gpu`, SimWarp without),
//! verdicts per step.
//!
//! ```text
//! edp_batch CNF COMP DIR [--pattern s{}.blrat] [--threads 6] [--blocks 4096]
//!           [--chunk-gb 6] [--steps N] [--simwarp K] [--real-cnf]
//! ```
//!
//! Step `j` (1-based, a line of COMP) is the CNF plus one unit per literal of
//! the negated lemma, in the lemma's literal order; the last step (the line
//! `0`) is the CNF plus every earlier lemma as a clause — exactly the formulas
//! the proofs in DIR were produced against. `--simwarp K` additionally decodes
//! K evenly spaced steps against the real formula and checks that the batch's
//! codes for them equal the one-at-a-time SimWarp codes (the batching
//! contract, on the real data). `--real-cnf` decodes every step against the
//! real CNF instead of placeholders (slower, more memory; a control).

use std::time::Instant;
use warp_types_sat::lrat::{parse_cnf_ints, Verdict};
use warp_types_sat::lrat_batch::{batch, combine_steps};
use warp_types_sat::lrat_warp::{check_warp_codes, FlatProof, LemmaCode};

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Run `f(j)` for `j in 0..n` on `threads` workers, results in order.
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

/// Bytes the check's device arrays take for one step's flat proof (lits,
/// hints, clause_start, deleted_at, lemma_clause, lemma_step, lemma_id,
/// hint_start), without the shared CNF.
fn device_bytes(fp: &FlatProof, n_cnf: usize) -> usize {
    let clauses = fp.clause_start.len() - 1 - n_cnf;
    let cnf_lits = fp.clause_start[n_cnf] as usize;
    4 * (fp.lits.len() - cnf_lits) + 4 * fp.hints.len() + 8 * clauses + 20 * fp.lemma_clause.len()
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
        eprintln!("usage: edp_batch CNF COMP DIR [--pattern s{{}}.blrat] [--threads 6] [--blocks 4096] [--chunk-gb 6] [--steps N] [--simwarp K] [--real-cnf]");
        std::process::exit(1);
    }
    let pattern: String = arg(&args, "--pattern", "s{}.blrat".to_string());
    let threads: usize = arg(&args, "--threads", 6);
    let blocks: u32 = arg(&args, "--blocks", 4096);
    let chunk_gb: f64 = arg(&args, "--chunk-gb", 6.0);
    let simwarp: usize = arg(&args, "--simwarp", 0);
    let real_cnf = args.iter().any(|a| a == "--real-cnf");

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
    // extras_j: the step's private original clauses, in id order.
    let extras = |j: usize| -> Vec<Vec<i32>> {
        if comp[j].is_empty() {
            comp[..j].to_vec() // the final step: every earlier lemma
        } else {
            comp[j].iter().map(|&l| vec![-l]).collect()
        }
    };
    let t1 = Instant::now();
    println!(
        "c cnf {} vars? clauses {n_cnf}; comp {n_steps_all} lines; checking {n_steps} steps; threads {threads}",
        cnf.iter().flatten().map(|l| l.unsigned_abs()).max().unwrap_or(0)
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

    // Decode each step against placeholders (or the real CNF) plus its extras.
    let holder: Vec<Vec<i32>> = vec![Vec::new(); n_cnf];
    let decoded: Vec<Result<FlatProof, String>> = par_map(n_steps, threads, |j| {
        let mut f: Vec<Vec<i32>> = if real_cnf {
            cnf.clone()
        } else {
            holder.clone()
        };
        f.extend(extras(j));
        FlatProof::from_binary(&f, &bytes[j]).map_err(|e| format!("step {}: {e:?}", j + 1))
    });
    let t3 = Instant::now();
    let mut steps: Vec<FlatProof> = Vec::with_capacity(n_steps);
    for d in decoded {
        match d {
            Ok(fp) => steps.push(fp),
            Err(e) => {
                println!("s ERROR {e}");
                std::process::exit(2);
            }
        }
    }
    let lemmas: usize = steps.iter().map(|s| s.lemma_clause.len()).sum();
    let hints: usize = steps.iter().map(|s| s.hints.len()).sum();
    let early = steps.iter().filter(|s| s.early.is_some()).count();
    println!("c proofs {total_bytes} bytes; lemmas {lemmas}; hints {hints}; steps with an early verdict {early}");

    // Chunk, lay out, check.
    let t_init0 = Instant::now();
    #[cfg(feature = "gpu")]
    let ctx = warp_types_sat::gpu_launcher::GpuContext::new().expect("GPU context");
    let t_init = t_init0.elapsed().as_secs_f64();
    let budget = (chunk_gb * 1e9) as usize;
    let mut verdicts: Vec<Verdict> = Vec::with_capacity(n_steps);
    let mut codes_all: Vec<Vec<LemmaCode>> = Vec::with_capacity(n_steps);
    let (mut t_batch, mut t_check, mut t_combine) = (0.0, 0.0, 0.0);
    let mut chunks = 0;
    let mut lo = 0;
    while lo < n_steps {
        let mut hi = lo;
        let mut used = 0;
        while hi < n_steps && (hi == lo || used + device_bytes(&steps[hi], n_cnf) <= budget) {
            used += device_bytes(&steps[hi], n_cnf);
            hi += 1;
        }
        let a = Instant::now();
        let b = batch(&cnf, &steps[lo..hi]);
        let bt = Instant::now();
        #[cfg(feature = "gpu")]
        let codes = codes_of(&ctx.check_lrat_codes(&b.flat, blocks).expect("device check"));
        #[cfg(not(feature = "gpu"))]
        let codes = {
            let _ = blocks;
            check_warp_codes(&b.flat)
        };
        let ct = Instant::now();
        let vs = combine_steps(&steps[lo..hi], &b, &codes);
        for &(l, h) in &b.lemma_range {
            codes_all.push(codes[l..h].to_vec());
        }
        verdicts.extend(vs);
        let et = Instant::now();
        t_batch += (bt - a).as_secs_f64();
        t_check += (ct - bt).as_secs_f64();
        t_combine += (et - ct).as_secs_f64();
        chunks += 1;
        println!(
            "c chunk {chunks}: steps {}..={} ({} lemmas, {:.2} GB est.) batch {:.3}s check {:.3}s combine {:.3}s",
            lo + 1,
            hi,
            b.flat.lemma_clause.len(),
            used as f64 / 1e9,
            (bt - a).as_secs_f64(),
            (ct - bt).as_secs_f64(),
            (et - ct).as_secs_f64()
        );
        lo = hi;
    }
    let t4 = Instant::now();

    // Report.
    let ok = verdicts.iter().filter(|v| v.is_verified()).count();
    let unsupported = verdicts
        .iter()
        .filter(|v| matches!(v, Verdict::Unsupported { .. }))
        .count();
    for (j, v) in verdicts
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_verified())
        .take(10)
    {
        println!("c step {} NOT verified: {v:?}", j + 1);
    }
    println!(
        "c times: cnf+comp {:.3}s | read proofs {:.3}s | decode {:.3}s | gpu init {:.3}s | batch {:.3}s | check {:.3}s | combine {:.3}s | total {:.3}s",
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        (t3 - t2).as_secs_f64(),
        t_init,
        t_batch,
        t_check,
        t_combine,
        (t4 - t0).as_secs_f64()
    );
    println!(
        "s {} ({ok}/{n_steps} steps verified, {unsupported} unsupported, {chunks} chunks, {} mode)",
        if ok == n_steps {
            "VERIFIED"
        } else {
            "NOT VERIFIED"
        },
        if cfg!(feature = "gpu") {
            "device"
        } else {
            "SimWarp"
        }
    );

    // The batching contract on real data: K steps re-decoded against the
    // real formula and checked one at a time by SimWarp.
    if simwarp > 0 {
        let k = simwarp.min(n_steps);
        let mut mismatches = 0;
        for i in 0..k {
            let j = i * n_steps / k;
            let mut f = cnf.clone();
            f.extend(extras(j));
            let fp = FlatProof::from_binary(&f, &bytes[j]).expect("decode");
            let one = check_warp_codes(&fp);
            if one != codes_all[j] {
                mismatches += 1;
                println!(
                    "c CONTRACT MISMATCH at step {}: one-at-a-time {:?} vs batch {:?}",
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
}
