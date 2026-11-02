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
