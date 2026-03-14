fn main() {
    warp_types_builder::WarpBuilder::new("my-kernels")
        .build()
        .expect("Failed to compile GPU kernels to PTX");
}
