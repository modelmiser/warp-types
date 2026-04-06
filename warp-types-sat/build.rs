fn main() {
    #[cfg(feature = "gpu")]
    {
        // SM arch from env (e.g. SM_ARCH=sm_90 for H200) or default sm_70 (works on all Volta+).
        println!("cargo:rerun-if-env-changed=SM_ARCH");
        let sm = std::env::var("SM_ARCH").unwrap_or_else(|_| "sm_70".to_string());
        eprintln!("sat-kernels: compiling for {}", sm);
        warp_types_builder::WarpBuilder::new("sat-kernels")
            .sm_arch(&sm)
            .build()
            .unwrap_or_else(|e| panic!("Failed to compile SAT GPU kernels to PTX ({}): {}", sm, e));
    }
}
