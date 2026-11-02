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
    let t2 = Instant::now();
    let seq = check_sequential(&cnf, &proof);
    let t3 = Instant::now();
    let ind = check_independent(&cnf, &proof, threads);
    let t4 = Instant::now();
    let warp = warp_types_sat::lrat_warp::check_warp(&cnf, &proof);
    let t5 = Instant::now();
    println!("c warp(SimWarp) {:.3}s  {warp:?}", (t5 - t4).as_secs_f64());
    // Device-path host cost: decode binary straight into the flat layout.
    if matches!(bytes.first(), Some(b'a') | Some(b'd')) {
        let tf = Instant::now();
        match warp_types_sat::lrat_warp::FlatProof::from_binary(&cnf, &bytes) {
            Ok(fp) => {
                let tg = Instant::now();
                let codes = warp_types_sat::lrat_warp::check_warp_codes(&fp);
                let th = Instant::now();
                let v = warp_types_sat::lrat_warp::combine(&fp, &codes);
                println!(
                    "c flat-decode {:.3}s (lemmas={} hints={} lits={})  simwarp-on-flat {:.3}s  {v:?}",
                    (tg - tf).as_secs_f64(),
                    fp.lemma_clause.len(),
                    fp.hints.len(),
                    fp.lits.len(),
                    (th - tg).as_secs_f64()
                );
                if v.is_verified() != seq.is_verified() {
                    println!("s DISAGREE-FLAT");
                    std::process::exit(3);
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
        "c read {:.3}s parse {:.3}s sequential {:.3}s independent({threads}) {:.3}s  bytes={} steps={}",
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        (t3 - t2).as_secs_f64(),
        (t4 - t3).as_secs_f64(),
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
