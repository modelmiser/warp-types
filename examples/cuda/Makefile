# Makefile — C++ host loading Rust-compiled PTX via CUDA Driver API
#
# This builds a C++ program that loads PTX at runtime and launches kernels.
# The PTX comes from building Rust kernels with warp-types.
#
# Prerequisites:
#   - CUDA Toolkit (nvcc, libcuda)
#   - Rust nightly with rust-src (for building the Rust kernel crate)
#
# Usage:
#   make                            # Build the C++ host program
#   make ptx                        # Build Rust kernels to PTX
#   make run                        # Build everything and run
#   make clean                      # Remove build artifacts

NVCC     ?= nvcc
CXXFLAGS  = --std=c++20 -O2
LDFLAGS   = -lcuda

# Path to the Rust kernel PTX (from gpu-project example)
KERNEL_DIR = ../gpu-project
PTX_PATH   = $(KERNEL_DIR)/target/nvptx64-nvidia-cuda/release/my_kernels.ptx

.PHONY: all clean run ptx

all: warp_demo

warp_demo: main.cu
	$(NVCC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Build Rust kernels to PTX (requires nightly + rust-src)
ptx:
	cd $(KERNEL_DIR) && cargo +nightly build --release

# Build everything and run
run: warp_demo ptx
	./warp_demo $(PTX_PATH)

clean:
	rm -f warp_demo
