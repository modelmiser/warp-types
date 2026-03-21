# CMakeLists.txt — C++ host loading Rust-compiled PTX
#
# This builds a C++ program that loads warp-types PTX at runtime via the
# CUDA Driver API. The PTX is compiled separately by Cargo (Rust nightly).
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   make
#   make ptx          # build Rust kernels (requires nightly + rust-src)
#   ./warp_demo <path-to-ptx>
#
# For HIP: use hipcc and replace cuda calls (API shape is 1:1).

cmake_minimum_required(VERSION 3.18)
project(warp_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find CUDA Toolkit (for Driver API: libcuda) ---
find_package(CUDAToolkit REQUIRED)

# --- Build the C++ host program ---
add_executable(warp_demo main.cu)
target_link_libraries(warp_demo PRIVATE CUDA::cuda_driver)

# --- Optional: build Rust kernels to PTX ---
# Requires: rustup install nightly && rustup component add rust-src --toolchain nightly
set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../gpu-project")
set(PTX_OUTPUT "${KERNEL_DIR}/target/nvptx64-nvidia-cuda/release/my_kernels.ptx")

add_custom_target(ptx
    COMMAND cargo +nightly build --release
    WORKING_DIRECTORY "${KERNEL_DIR}"
    COMMENT "Building Rust kernels to PTX (nightly required)"
    VERBATIM
)

# --- Convenience: run with PTX ---
add_custom_target(run
    COMMAND $<TARGET_FILE:warp_demo> "${PTX_OUTPUT}"
    DEPENDS warp_demo ptx
    COMMENT "Running warp_demo with Rust-compiled PTX"
    VERBATIM
)
