//! `cargo run --release -p warp-types-sat --example lrat_check -- file.cnf file.lrat [threads]`
//! Checks an LRAT proof (text or binary) with the reference checker and the
//! lemma-independent checker, printing verdicts and wall times.
use std::time::Instant;
use warp_types_sat::lrat::{check_independent, check_sequential, parse_any, parse_cnf_ints};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: lrat_check file.cnf file.lrat [threads]");
        std::process::exit(2);
    }
    let threads: usize = args.get(2).and_then(|t| t.parse().ok()).unwrap_or(1);
    // `--to-binary OUT` as a 4th argument converts the (text) proof and exits.
    if args.get(3).map(String::as_str) == Some("--to-binary") {
        let out = args.get(4).expect("--to-binary needs a path");
        let bytes = std::fs::read(&args[1]).expect("read proof");
        let proof = warp_types_sat::lrat::parse_any(&bytes).expect("parse");
        std::fs::write(out, warp_types_sat::lrat::write_binary(&proof)).expect("write");
        println!("c wrote binary LRAT to {out}");
        return;
    }
    let t0 = Instant::now();
    let cnf = parse_cnf_ints(&std::fs::read_to_string(&args[0]).expect("read cnf"));
    let bytes = std::fs::read(&args[1]).expect("read proof");
    let t1 = Instant::now();
    let proof = match parse_any(&bytes) {
        Ok(p) => p,
        Err(e) => {
            println!("s ERROR {e}");
            std::process::exit(2);
        }
    };
    let skip_slow = std::env::var_os("LRAT_SKIP_SLOW").is_some();
    let t2 = Instant::now();
    let ind = check_independent(&cnf, &proof, threads);
    let t3 = Instant::now();
    // The sequential reference is the slowest CPU pass; LRAT_SKIP_SLOW keeps
    // only the independent checker as the comparison point (they agree on
    // every archived proof; the archive test holds them to it).
    let seq = if skip_slow {
        ind.clone()
    } else {
        check_sequential(&cnf, &proof)
    };
    let t4 = Instant::now();
    let warp = if skip_slow {
        ind.clone()
    } else {
        warp_types_sat::lrat_warp::check_warp(&cnf, &proof)
    };
    let t5 = Instant::now();
    if !skip_slow {
        println!("c warp(SimWarp) {:.3}s  {warp:?}", (t5 - t4).as_secs_f64());
    }
    // Device-path host cost: decode binary straight into the flat layout.
    if matches!(bytes.first(), Some(b'a') | Some(b'd')) {
        let decode_threads: usize = std::env::var("LRAT_DECODE_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(threads);
        if std::env::var_os("LRAT_DECODE_VERIFY").is_some() {
            // Hold the parallel decoder to the one-thread decoder on this file.
            let t0 = Instant::now();
            let seq = warp_types_sat::lrat_warp::FlatProof::from_binary(&cnf, &bytes);
            let t1 = Instant::now();
            let par =
                warp_types_sat::lrat_warp::FlatProof::from_binary_par(&cnf, &bytes, decode_threads);
            let t2 = Instant::now();
            let same = match (&seq, &par) {
                (Ok(a), Ok(b)) => {
                    a.lits == b.lits
                        && a.clause_start == b.clause_start
                        && a.deleted_at == b.deleted_at
                        && a.lemma_clause == b.lemma_clause
                        && a.lemma_step == b.lemma_step
                        && a.lemma_id == b.lemma_id
                        && a.hint_start == b.hint_start
                        && a.hints == b.hints
                        && a.first_empty == b.first_empty
                }
                (Err(a), Err(b)) => a == b,
                _ => false,
            };
            println!(
                "c decode-verify: sequential {:.3}s parallel({decode_threads}) {:.3}s  {}",
                (t1 - t0).as_secs_f64(),
                (t2 - t1).as_secs_f64(),
                if same { "EQUAL" } else { "DIFFER" }
            );
            if !same {
                println!("s DISAGREE-DECODE");
                std::process::exit(3);
            }
        }
        let tf = Instant::now();
        match warp_types_sat::lrat_warp::FlatProof::from_binary_par(&cnf, &bytes, decode_threads) {
            Ok(fp) => {
                let tg = Instant::now();
                println!(
                    "c flat-decode({decode_threads}) {:.3}s (lemmas={} hints={} lits={})",
                    (tg - tf).as_secs_f64(),
                    fp.lemma_clause.len(),
                    fp.hints.len(),
                    fp.lits.len()
                );
                if !skip_slow {
                    let codes = warp_types_sat::lrat_warp::check_warp_codes(&fp);
                    let th = Instant::now();
                    let v = warp_types_sat::lrat_warp::combine(&fp, &codes);
                    println!("c simwarp-on-flat {:.3}s  {v:?}", (th - tg).as_secs_f64());
                    if v.is_verified() != seq.is_verified() {
                        println!("s DISAGREE-FLAT");
                        std::process::exit(3);
                    }
                }
                #[cfg(feature = "gpu")]
                if std::env::var_os("LRAT_GPU").is_some() {
                    let ctx = warp_types_sat::gpu_launcher::GpuContext::new().expect("GPU context");
                    let blocks_list: Vec<u32> = std::env::var("LRAT_GPU_BLOCKS")
                        .ok()
                        .map(|b| b.split(',').filter_map(|x| x.parse().ok()).collect())
                        .unwrap_or_else(|| vec![4096]);
                    for blocks in blocks_list {
                        let ta = Instant::now();
                        let raw = ctx.check_lrat_codes(&fp, blocks).expect("GPU check");
                        let tb = Instant::now();
                        let codes: Vec<warp_types_sat::lrat_warp::LemmaCode> = raw
                            .iter()
                            .map(|&c| match c {
                                0 => warp_types_sat::lrat_warp::LemmaCode::Ok,
                                1 => warp_types_sat::lrat_warp::LemmaCode::NotUnit,
                                2 => warp_types_sat::lrat_warp::LemmaCode::NoConflict,
                                3 => warp_types_sat::lrat_warp::LemmaCode::DeletedHint,
                                4 => warp_types_sat::lrat_warp::LemmaCode::ForwardHint,
                                5 => warp_types_sat::lrat_warp::LemmaCode::TableOverflow,
                                _ => warp_types_sat::lrat_warp::LemmaCode::Unwritten,
                            })
                            .collect();
                        let g = warp_types_sat::lrat_warp::combine(&fp, &codes);
                        let overflow = codes
                            .iter()
                            .filter(|&&c| c == warp_types_sat::lrat_warp::LemmaCode::TableOverflow)
                            .count();
                        println!(
                            "c gpu-flat({}; {blocks} blocks) device {:.3}s (upload+launch+download) overflow={overflow}  {g:?}",
                            ctx.device_name(),
                            (tb - ta).as_secs_f64()
                        );
                        if g.is_verified() != seq.is_verified() {
                            println!("s DISAGREE-GPU-FLAT");
                            std::process::exit(3);
                        }
                    }
                }
                // Device decode (research/gpu-lrat-decode.md): both kernels at
                // each slice size in LRAT_DECODE_SLICES (default 4096), best
                // of LRAT_DECODE_REPS (default 1); the result must equal the
                // host decode above, field for field.
                #[cfg(feature = "gpu")]
                if std::env::var_os("LRAT_GPU_DECODE").is_some() {
                    let ctx = warp_types_sat::gpu_launcher::GpuContext::new().expect("GPU context");
                    let slices: Vec<u32> = std::env::var("LRAT_DECODE_SLICES")
                        .ok()
                        .map(|b| b.split(',').filter_map(|x| x.parse().ok()).collect())
                        .unwrap_or_else(|| vec![4096]);
                    let reps: usize = std::env::var("LRAT_DECODE_REPS")
                        .ok()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(1);
                    let blocks: u32 = std::env::var("LRAT_GPU_BLOCKS")
                        .ok()
                        .and_then(|b| b.split(',').next().and_then(|x| x.parse().ok()))
                        .unwrap_or(4096);
                    for &slice in &slices {
                        for coop in [true, false] {
                            let mut best: Option<(
                                f64,
                                warp_types_sat::lrat_warp::decode_device::DeviceDecodeStats,
                            )> = None;
                            let mut equal = true;
                            for _ in 0..reps.max(1) {
                                let ta = Instant::now();
                                match warp_types_sat::lrat_warp::FlatProof::from_binary_device(
                                    &ctx, &cnf, &bytes, slice, coop, blocks,
                                ) {
                                    Ok((dfp, st)) => {
                                        let t = ta.elapsed().as_secs_f64();
                                        equal &= dfp == fp;
                                        if best.map_or(true, |(b, _)| t < b) {
                                            best = Some((t, st));
                                        }
                                    }
                                    Err(e) => {
                                        println!(
                                            "c gpu-decode error slice={slice} coop={coop}: {e}"
                                        );
                                        equal = false;
                                        break;
                                    }
                                }
                            }
                            if let Some((t, st)) = best {
                                println!(
                                    "c gpu-decode({}; {} slice={slice} blocks={blocks}) total {t:.3}s = upload {:.3} + pass1 {:.3} + prefix {:.3} + pass2 {:.3} + download {:.3} + host {:.3}  slices={} dirty={} none={}  {}",
                                    ctx.device_name(),
                                    if coop { "warp" } else { "thread" },
                                    st.timing.upload.as_secs_f64(),
                                    st.timing.pass1.as_secs_f64(),
                                    st.timing.prefix.as_secs_f64(),
                                    st.timing.pass2.as_secs_f64(),
                                    st.timing.download.as_secs_f64(),
                                    st.host.as_secs_f64(),
                                    st.n_slices,
                                    st.dirty,
                                    st.none,
                                    if equal { "EQUAL" } else { "DIFFERENT" }
                                );
                            }
                            if !equal {
                                println!("s DISAGREE-GPU-DECODE");
                                std::process::exit(3);
                            }
                        }
                    }
                }
                // Device-resident route: decode on the device, plan on the
                // host over step records, payloads gathered on the device,
                // check on the resident arrays. LRAT_RESIDENT_VERIFY downloads
                // the arrays once and compares them with the host decode.
                #[cfg(feature = "gpu")]
                if std::env::var_os("LRAT_GPU_RESIDENT").is_some() {
                    let ctx = warp_types_sat::gpu_launcher::GpuContext::new().expect("GPU context");
                    let slice: u32 = std::env::var("LRAT_DECODE_SLICES")
                        .ok()
                        .and_then(|b| b.split(',').next().and_then(|x| x.parse().ok()))
                        .unwrap_or(4096);
                    let reps: usize = std::env::var("LRAT_DECODE_REPS")
                        .ok()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(1);
                    let blocks: u32 = std::env::var("LRAT_GPU_BLOCKS")
                        .ok()
                        .and_then(|b| b.split(',').next().and_then(|x| x.parse().ok()))
                        .unwrap_or(4096);
                    for coop in [true, false] {
                        let mut best: Option<(
                            f64,
                            f64,
                            warp_types_sat::lrat_warp::decode_device::ResidentStats,
                            warp_types_sat::lrat::Verdict,
                        )> = None;
                        for _ in 0..reps.max(1) {
                            let ta = Instant::now();
                            let (dfp, st) =
                                warp_types_sat::lrat_warp::FlatProof::from_binary_device_resident(
                                    &ctx, &cnf, &bytes, slice, coop, blocks,
                                )
                                .unwrap_or_else(|e| panic!("resident decode: {e}"));
                            let tb = Instant::now();
                            let raw = ctx
                                .check_lrat_codes_resident(&dfp, blocks)
                                .expect("resident check");
                            let tc = Instant::now();
                            let codes: Vec<warp_types_sat::lrat_warp::LemmaCode> = raw
                                .iter()
                                .map(|&c| match c {
                                    0 => warp_types_sat::lrat_warp::LemmaCode::Ok,
                                    1 => warp_types_sat::lrat_warp::LemmaCode::NotUnit,
                                    2 => warp_types_sat::lrat_warp::LemmaCode::NoConflict,
                                    3 => warp_types_sat::lrat_warp::LemmaCode::DeletedHint,
                                    4 => warp_types_sat::lrat_warp::LemmaCode::ForwardHint,
                                    5 => warp_types_sat::lrat_warp::LemmaCode::TableOverflow,
                                    _ => warp_types_sat::lrat_warp::LemmaCode::Unwritten,
                                })
                                .collect();
                            let v = warp_types_sat::lrat_warp::decode_device::combine_resident(
                                &ctx, &dfp, &codes,
                            )
                            .expect("combine");
                            let (td, tk) = ((tb - ta).as_secs_f64(), (tc - tb).as_secs_f64());
                            if std::env::var_os("LRAT_RESIDENT_VERIFY").is_some() && best.is_none()
                            {
                                let (l, h) = ctx.download_lrat_payloads(&dfp).expect("download");
                                let mut m = dfp.meta.clone();
                                m.lits = l;
                                m.hints = h;
                                println!(
                                    "c resident-verify({}) arrays {}",
                                    if coop { "warp" } else { "thread" },
                                    if m == fp { "EQUAL" } else { "DIFFERENT" }
                                );
                                if m != fp {
                                    println!("s DISAGREE-GPU-RESIDENT");
                                    std::process::exit(3);
                                }
                            }
                            if best.as_ref().map_or(true, |(d, k, _, _)| td + tk < d + k) {
                                best = Some((td, tk, st, v));
                            }
                        }
                        if let Some((td, tk, st, v)) = best {
                            println!(
                                "c gpu-resident({}; {} slice={slice} blocks={blocks}) decode {td:.3}s = upload {:.3} + pass1 {:.3} + prefix {:.3} + pass2 {:.3} + download {:.3} + plan {:.3} + gather {:.3}; check {tk:.3}s; bytes→verdict {:.3}s  slices={} dirty={}  {v:?}",
                                ctx.device_name(),
                                if coop { "warp" } else { "thread" },
                                st.decode.timing.upload.as_secs_f64(),
                                st.decode.timing.pass1.as_secs_f64(),
                                st.decode.timing.prefix.as_secs_f64(),
                                st.decode.timing.pass2.as_secs_f64(),
                                st.decode.timing.download.as_secs_f64(),
                                st.plan.as_secs_f64(),
                                st.gather.as_secs_f64(),
                                td + tk,
                                st.decode.n_slices,
                                st.decode.dirty,
                            );
                            if v.is_verified() != seq.is_verified() {
                                println!("s DISAGREE-GPU-RESIDENT-VERDICT");
                                std::process::exit(3);
                            }
                        }
                    }
                }
            }
            Err(e) => println!("c flat-decode error: {e}"),
        }
    }
    if warp.is_verified() != seq.is_verified() {
        println!("s DISAGREE-WARP");
        std::process::exit(3);
    }
    println!(
        "c read {:.3}s parse {:.3}s independent({threads}) {:.3}s sequential {:.3}s{}  bytes={} steps={}",
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        (t3 - t2).as_secs_f64(),
        (t4 - t3).as_secs_f64(),
        if skip_slow { " (skipped)" } else { "" },
        bytes.len(),
        proof.steps.len()
    );
    println!("c sequential:  {seq:?}");
    println!("c independent: {ind:?}");
    if seq != ind {
        println!("s DISAGREE");
        std::process::exit(3);
    }
    #[cfg(feature = "gpu")]
    if std::env::var_os("LRAT_GPU").is_some() {
        let ctx = warp_types_sat::gpu_launcher::GpuContext::new().expect("GPU context");
        let blocks: u32 = std::env::var("LRAT_GPU_BLOCKS")
            .ok()
            .and_then(|b| b.parse().ok())
            .unwrap_or(4096);
        let t6 = Instant::now();
        let g = warp_types_sat::lrat_warp::check_warp_gpu(&ctx, &cnf, &proof, blocks)
            .expect("GPU check");
        let t7 = Instant::now();
        println!(
            "c gpu({}; {blocks} blocks) {:.3}s  {g:?}",
            ctx.device_name(),
            (t7 - t6).as_secs_f64()
        );
        if g.is_verified() != seq.is_verified() {
            println!("s DISAGREE-GPU");
            std::process::exit(3);
        }
    }
    println!(
        "s {}",
        if seq.is_verified() {
            "VERIFIED"
        } else {
            "NOT VERIFIED"
        }
    );
    std::process::exit(if seq.is_verified() { 0 } else { 1 });
}
