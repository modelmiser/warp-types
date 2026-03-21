// main.cu — Load Rust-compiled PTX and launch a typed warp kernel from C++
//
// This demonstrates the C++ interop path for warp-types:
//   1. Build kernels in Rust with warp-types (type-safe, zero overhead)
//   2. cargo build produces PTX
//   3. C++ host loads PTX via CUDA Driver API
//   4. Launch kernels — identical to any PTX loading workflow
//
// Build:
//   nvcc --std=c++20 -o warp_demo main.cu -lcuda
//
// Run:
//   # First, build the Rust kernels to PTX:
//   cd ../gpu-project && cargo +nightly build --release
//   # Then load and launch from C++:
//   ./warp_demo ../../examples/gpu-project/target/nvptx64-nvidia-cuda/release/my_kernels.ptx
//
// For HIP on NVIDIA: replace cuInit → hipInit, cuModuleLoad → hipModuleLoad, etc.
// The API shape is 1:1.

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define CHECK(call) do {                                              \
    CUresult err = (call);                                            \
    if (err != CUDA_SUCCESS) {                                        \
        const char* msg = "unknown error";                            \
        cuGetErrorString(err, &msg);                                  \
        fprintf(stderr, "%s:%d: CUDA error: %s\n",                   \
                __FILE__, __LINE__, msg);                             \
        exit(1);                                                      \
    }                                                                 \
} while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.ptx> [kernel_name]\n\n", argv[0]);
        fprintf(stderr, "  kernel.ptx    — PTX file from Rust build\n");
        fprintf(stderr, "  kernel_name   — function name (default: reduce_n)\n\n");
        fprintf(stderr, "Generate PTX from Rust:\n");
        fprintf(stderr, "  cd ../gpu-project && cargo +nightly build --release\n");
        fprintf(stderr, "  PTX: target/nvptx64-nvidia-cuda/release/my_kernels.ptx\n");
        return 1;
    }

    const char* ptx_path = argv[1];
    const char* kernel_name = argc >= 3 ? argv[2] : "reduce_n";

    // --- Initialize CUDA Driver API ---
    CHECK(cuInit(0));

    CUdevice device;
    CHECK(cuDeviceGet(&device, 0));

    char name[128];
    cuDeviceGetName(name, sizeof(name), device);
    printf("GPU: %s\n", name);

    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, device));

    // --- Load PTX module ---
    CUmodule module;
    CHECK(cuModuleLoad(&module, ptx_path));
    printf("Loaded: %s\n", ptx_path);

    // --- Get kernel function ---
    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, module, kernel_name));
    printf("Kernel: %s\n\n", kernel_name);

    // --- Allocate device memory ---
    constexpr int N = 32;
    CUdeviceptr d_input, d_output;
    CHECK(cuMemAlloc(&d_input,  N * sizeof(int)));
    CHECK(cuMemAlloc(&d_output, sizeof(int)));

    // --- Initialize: 32 ones ---
    int input[N];
    for (int i = 0; i < N; i++) input[i] = 1;
    CHECK(cuMemcpyHtoD(d_input, input, N * sizeof(int)));

    int zero = 0;
    CHECK(cuMemcpyHtoD(d_output, &zero, sizeof(int)));

    // --- Launch kernel (1 block × 32 threads = 1 warp) ---
    // reduce_n signature: (input: *const i32, output: *mut i32, n: u32)
    unsigned int n = N;
    void* args[] = { &d_input, &d_output, &n };
    CHECK(cuLaunchKernel(kernel,
        1, 1, 1,     // grid:  1 block
        32, 1, 1,    // block: 32 threads (1 warp)
        0, nullptr,  // shared mem, stream
        args, nullptr));
    CHECK(cuCtxSynchronize());

    // --- Read result ---
    int result;
    CHECK(cuMemcpyDtoH(&result, d_output, sizeof(int)));
    printf("Input:    [1, 1, 1, ..., 1]  (%d ones)\n", N);
    printf("Expected: %d\n", N);
    printf("Got:      %d\n", result);
    printf("Result:   %s\n", result == N ? "PASS" : "FAIL");

    // --- Cleanup ---
    cuMemFree(d_input);
    cuMemFree(d_output);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    return result == N ? 0 : 1;
}
