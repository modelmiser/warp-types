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
    use warp_types_sat::gpu_gradient::{gradient_soa, total_loss_simwarp, ClauseDataSoA, VarIndexSoA};
    use warp_types_sat::gpu_launcher::GpuContext;

    let ctx = GpuContext::new().expect("CUDA init failed");

    // ── Loss-only benchmark ──
    println!("=== Loss only (GPU kernel vs CPU) ===");
    println!("{:>6} {:>8} {:>10} {:>10} {:>8}",
        "vars", "clauses", "cpu_μs", "gpu_μs", "speedup");
    println!("{}", "-".repeat(50));

    for &num_vars in &[20, 50, 100, 200, 500, 1000, 2000, 5000] {
        let db = generate_3sat_phase_transition(num_vars, 42);
        let weights = vec![1.0; db.len()];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);
        let x: Vec<f64> = (0..num_vars as usize)
            .map(|i| {
                let raw = (i as f64 * std::f64::consts::FRAC_1_SQRT_2) % 1.0;
                0.05 + raw * 0.9
            })
            .collect();

        let gpu_data = ctx.upload_clause_data(&soa).expect("upload failed");
        let _ = ctx.total_loss(&gpu_data, &x); // warm up

        let iters = if num_vars <= 100 { 1000 } else if num_vars <= 500 { 200 } else { 50 };

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(total_loss_simwarp(&soa, &x));
        }
        let cpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(ctx.total_loss(&gpu_data, &x).unwrap());
        }
        let gpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        println!("{:>6} {:>8} {:>10.1} {:>10.1} {:>8.2}x",
            num_vars, soa.num_clauses, cpu_us, gpu_us, cpu_us / gpu_us);
    }

    // ── Fused loss+gradient benchmark ──
    println!();
    println!("=== Loss + Gradient (fused GPU kernel vs CPU loss + CPU grad) ===");
    println!("{:>6} {:>8} {:>10} {:>10} {:>8}",
        "vars", "clauses", "cpu_μs", "fused_μs", "speedup");
    println!("{}", "-".repeat(50));

    for &num_vars in &[20, 50, 100, 200, 500, 1000, 2000, 5000] {
        let db = generate_3sat_phase_transition(num_vars, 42);
        let weights = vec![1.0; db.len()];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);
        let var_idx = VarIndexSoA::build(&soa, num_vars);
        let x: Vec<f64> = (0..num_vars as usize)
            .map(|i| {
                let raw = (i as f64 * std::f64::consts::FRAC_1_SQRT_2) % 1.0;
                0.05 + raw * 0.9
            })
            .collect();
        let mut grad = vec![0.0f64; num_vars as usize];

        let gpu_data = ctx.upload_clause_data(&soa).expect("upload failed");
        let _ = ctx.total_loss_and_grad(&gpu_data, &x, num_vars as usize); // warm up

        let iters = if num_vars <= 100 { 1000 } else if num_vars <= 500 { 200 } else { 50 };

        // CPU: loss + gradient (both sequential)
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(total_loss_simwarp(&soa, &x));
            gradient_soa(&soa, &x, &var_idx, &mut grad);
            std::hint::black_box(&grad);
        }
        let cpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // GPU: fused loss + gradient (single kernel launch)
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(
                ctx.total_loss_and_grad(&gpu_data, &x, num_vars as usize).unwrap()
            );
        }
        let fused_us = t0.elapsed().as_micros() as f64 / iters as f64;

        println!("{:>6} {:>8} {:>10.1} {:>10.1} {:>8.2}x",
            num_vars, soa.num_clauses, cpu_us, fused_us, cpu_us / fused_us);
    }
}
