//! GPU kernel correctness test: compare GPU loss against SimWarp loss.
//!
//! Run with: cargo test --features gpu -- gpu_loss
//! Requires CUDA-capable GPU (tested on H200 via RunPod).

#![cfg(feature = "gpu")]

use warp_types_sat::bench::generate_3sat_phase_transition;
use warp_types_sat::gpu_gradient::{total_loss_simwarp, ClauseDataSoA};
use warp_types_sat::gpu_launcher::GpuContext;

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

        let gpu_data = ctx
            .upload_clause_data(&soa)
            .expect("upload failed");
        let gpu_loss = ctx
            .total_loss(&gpu_data, &x)
            .expect("kernel launch failed");

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
