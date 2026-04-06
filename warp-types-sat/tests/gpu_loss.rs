//! GPU kernel correctness tests.
//!
//! Run with: cargo test --features gpu -- gpu_loss
//! Requires CUDA-capable GPU (tested on RTX 4000 Ada).

#![cfg(feature = "gpu")]

use warp_types_sat::bench::generate_3sat_phase_transition;
use warp_types_sat::gpu_gradient::{
    gradient_search_gpu, gradient_search_simwarp, gradient_soa, hybrid_solve_gpu,
    hybrid_solve_simwarp, total_loss_simwarp, ClauseDataSoA, VarIndexSoA,
};
use warp_types_sat::gpu_launcher::GpuContext;
use warp_types_sat::gradient::GradientConfig;

/// Deterministic variable initialization: golden ratio spacing in [0.05, 0.95].
/// Same as gpu_gradient::tests::init_vars.
fn init_vars(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let raw = (i as f64 * std::f64::consts::FRAC_1_SQRT_2) % 1.0;
            0.05 + raw * 0.9
        })
        .collect()
}

#[test]
fn gpu_loss_matches_simwarp() {
    let ctx = GpuContext::new().expect("CUDA init failed — is a GPU available?");

    // Test across 10 seeds with 20-variable 3-SAT at phase transition (ratio 4.267)
    for seed in 0..10 {
        let db = generate_3sat_phase_transition(20, seed);
        let weights = vec![1.0; db.len()];
        let (soa, _skipped) = ClauseDataSoA::from_clause_db(&db, &weights);
        let x = init_vars(20);

        let simwarp_loss = total_loss_simwarp(&soa, &x);

        let gpu_data = ctx.upload_clause_data(&soa).expect("upload failed");
        let gpu_loss = ctx.total_loss(&gpu_data, &x).expect("kernel launch failed");

        assert!(
            (simwarp_loss - gpu_loss).abs() < 1e-10,
            "seed {seed}: SimWarp loss {simwarp_loss:.12} != GPU loss {gpu_loss:.12} (diff {:.2e})",
            (simwarp_loss - gpu_loss).abs()
        );
    }
}

#[test]
fn gpu_loss_with_nonuniform_weights() {
    let ctx = GpuContext::new().expect("CUDA init failed");

    let db = generate_3sat_phase_transition(20, 7);
    let weights: Vec<f64> = (0..db.len()).map(|i| 0.5 + (i as f64) * 0.01).collect();
    let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);
    let x = init_vars(20);

    let simwarp_loss = total_loss_simwarp(&soa, &x);

    let gpu_data = ctx.upload_clause_data(&soa).expect("upload failed");
    let gpu_loss = ctx.total_loss(&gpu_data, &x).expect("kernel launch failed");

    assert!(
        (simwarp_loss - gpu_loss).abs() < 1e-10,
        "Weighted: SimWarp {simwarp_loss:.12} != GPU {gpu_loss:.12}"
    );
}

#[test]
fn gpu_loss_larger_instance() {
    // 100 variables → ~427 clauses → ~14 warp batches. Tests multi-block dispatch.
    let ctx = GpuContext::new().expect("CUDA init failed");

    let db = generate_3sat_phase_transition(100, 42);
    let weights = vec![1.0; db.len()];
    let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);
    let x = init_vars(100);

    let simwarp_loss = total_loss_simwarp(&soa, &x);

    let gpu_data = ctx.upload_clause_data(&soa).expect("upload failed");
    let gpu_loss = ctx.total_loss(&gpu_data, &x).expect("kernel launch failed");

    assert!(
        (simwarp_loss - gpu_loss).abs() < 1e-10,
        "100-var: SimWarp {simwarp_loss:.12} != GPU {gpu_loss:.12} (diff {:.2e})",
        (simwarp_loss - gpu_loss).abs()
    );
}

// ─── Gradient search: GPU vs SimWarp ─────────────────────────────────

#[test]
fn gpu_gradient_search_matches_simwarp() {
    // Both paths use the same RNG seed, same algorithm — only the loss
    // evaluation differs (GPU kernel vs SimWarp). Results must match.
    let ctx = GpuContext::new().expect("CUDA init failed");

    for seed in 0..5u64 {
        let db = generate_3sat_phase_transition(20, seed);
        let config = GradientConfig {
            seed,
            num_starts: 3,
            max_iters: 200,
            ..GradientConfig::default()
        };

        let simwarp = gradient_search_simwarp(&db, 20, &config);
        let gpu = gradient_search_gpu(&db, 20, &config, &ctx);

        // Same number of starts attempted
        assert_eq!(
            simwarp.starts.len(),
            gpu.starts.len(),
            "seed {seed}: start count mismatch"
        );

        // Same SAT/UNSAT outcome
        assert_eq!(
            simwarp.assignment.is_some(),
            gpu.assignment.is_some(),
            "seed {seed}: assignment presence mismatch"
        );

        // If both found SAT, assignments must match
        if let (Some(ref sw_assign), Some(ref gpu_assign)) = (&simwarp.assignment, &gpu.assignment)
        {
            assert_eq!(sw_assign, gpu_assign, "seed {seed}: assignments differ");
        }

        // Per-start loss trajectories must match
        for (i, (sw, gp)) in simwarp.starts.iter().zip(gpu.starts.iter()).enumerate() {
            assert!(
                (sw.best_loss - gp.best_loss).abs() < 1e-10,
                "seed {seed} start {i}: SimWarp best_loss {:.12} != GPU {:.12}",
                sw.best_loss,
                gp.best_loss
            );
            assert_eq!(
                sw.iterations, gp.iterations,
                "seed {seed} start {i}: iteration count mismatch"
            );
        }
    }
}

#[test]
fn gpu_hybrid_solve_matches_simwarp() {
    // End-to-end: gradient→CDCL hybrid must produce same result via GPU or SimWarp.
    let ctx = GpuContext::new().expect("CUDA init failed");

    for seed in 0..5u64 {
        let db = generate_3sat_phase_transition(20, seed);
        let config = GradientConfig {
            seed,
            num_starts: 3,
            max_iters: 200,
            ..GradientConfig::default()
        };

        // ClauseDb doesn't impl Clone — generate twice (same seed = same db).
        let db2 = generate_3sat_phase_transition(20, seed);
        let sw_result = hybrid_solve_simwarp(db, 20, &config);
        let gpu_result = hybrid_solve_gpu(db2, 20, &config, &ctx);

        // Same SAT/UNSAT outcome
        match (&sw_result, &gpu_result) {
            (warp_types_sat::SolveResult::Sat(sw_a), warp_types_sat::SolveResult::Sat(gpu_a)) => {
                assert_eq!(sw_a, gpu_a, "seed {seed}: SAT assignments differ");
            }
            (warp_types_sat::SolveResult::Unsat, warp_types_sat::SolveResult::Unsat) => {}
            _ => panic!(
                "seed {seed}: outcome mismatch: SimWarp={:?} GPU={:?}",
                matches!(sw_result, warp_types_sat::SolveResult::Sat(_)),
                matches!(gpu_result, warp_types_sat::SolveResult::Sat(_)),
            ),
        }
    }
}

// ─── Fused loss+gradient kernel ──────────────────────────────────────

#[test]
fn gpu_fused_grad_matches_cpu() {
    // Fused GPU gradient (atomicAdd) must match CPU gradient_soa to ~1e-6.
    // Looser tolerance than loss because atomic reduction order is non-deterministic.
    let ctx = GpuContext::new().expect("CUDA init failed");

    for seed in 0..5u64 {
        let num_vars = 50u32;
        let db = generate_3sat_phase_transition(num_vars, seed);
        let weights = vec![1.0; db.len()];
        let (soa, _) = ClauseDataSoA::from_clause_db(&db, &weights);
        let x = init_vars(num_vars as usize);

        // CPU gradient
        let var_idx = VarIndexSoA::build(&soa, num_vars);
        let mut cpu_grad = vec![0.0f64; num_vars as usize];
        gradient_soa(&soa, &x, &var_idx, &mut cpu_grad);

        // GPU fused
        let gpu_data = ctx.upload_clause_data(&soa).expect("upload failed");
        let (gpu_loss, gpu_grad) = ctx
            .total_loss_and_grad(&gpu_data, &x, num_vars as usize)
            .expect("fused kernel failed");

        // Loss should still match exactly (same butterfly reduce)
        let cpu_loss = total_loss_simwarp(&soa, &x);
        assert!(
            (cpu_loss - gpu_loss).abs() < 1e-10,
            "seed {seed}: loss mismatch: {cpu_loss:.12} vs {gpu_loss:.12}"
        );

        // Gradient matches to ~1e-10 (atomicAdd order may differ)
        for v in 0..num_vars as usize {
            assert!(
                (cpu_grad[v] - gpu_grad[v]).abs() < 1e-6,
                "seed {seed} var {v}: CPU grad {:.10} != GPU grad {:.10} (diff {:.2e})",
                cpu_grad[v],
                gpu_grad[v],
                (cpu_grad[v] - gpu_grad[v]).abs()
            );
        }
    }
}
