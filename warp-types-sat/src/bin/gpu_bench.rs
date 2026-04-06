//! GPU vs CPU loss evaluation benchmark.
//!
//! Measures total_loss across problem sizes to find the GPU crossover point.
//!
//! Usage: cargo run --release --features gpu --bin gpu_bench

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
    std::process::exit(1);
}

#[cfg(feature = "gpu")]
fn main() {
    use std::time::Instant;
    use warp_types_sat::bench::generate_3sat_phase_transition;
    use warp_types_sat::gpu_gradient::{total_loss_simwarp, ClauseDataSoA};
    use warp_types_sat::gpu_launcher::GpuContext;

    let ctx = GpuContext::new().expect("CUDA init failed");

    // Header
    println!("{:>6} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "vars", "clauses", "cpu_μs", "simwarp_μs", "gpu_μs", "speedup");
    println!("{}", "-".repeat(62));

    for &num_vars in &[20, 50, 100, 200, 500, 1000, 2000, 5000] {
        let db = generate_3sat_phase_transition(num_vars, 42);
        let weights = vec![1.0; db.len()];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);

        // Deterministic x values
        let x: Vec<f64> = (0..num_vars as usize)
            .map(|i| {
                let raw = (i as f64 * std::f64::consts::FRAC_1_SQRT_2) % 1.0;
                0.05 + raw * 0.9
            })
            .collect();

        let gpu_data = ctx.upload_clause_data(&soa).expect("upload failed");

        // Warm up GPU (first launch has JIT overhead)
        let _ = ctx.total_loss(&gpu_data, &x);

        // Number of iterations to get stable timing
        let iters = if num_vars <= 100 { 1000 } else if num_vars <= 500 { 200 } else { 50 };

        // CPU sequential (gradient.rs::loss via SoA)
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(total_loss_simwarp(&soa, &x));
        }
        let cpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // SimWarp (same function, just measuring separately for clarity)
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(total_loss_simwarp(&soa, &x));
        }
        let simwarp_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // GPU
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(ctx.total_loss(&gpu_data, &x).unwrap());
        }
        let gpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        let speedup = simwarp_us / gpu_us;

        println!("{:>6} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>8.2}x",
            num_vars, soa.num_clauses, cpu_us, simwarp_us, gpu_us, speedup);
    }
}
