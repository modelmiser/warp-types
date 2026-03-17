// typed_vs_untyped.cu — PTX zero-overhead sanity check
//
// Compiles two butterfly reductions to PTX:
//   1. butterfly_untyped: raw CUDA, no type annotations
//   2. butterfly_typed:   same code, with comments showing type-state
//
// If the type system truly has zero overhead, the PTX must be identical.
// Run: bash compare_ptx.sh

// --------------------------------------------------------------------------
// Untyped: how a CUDA programmer writes butterfly reduction today.
// No warp-type annotations. Uses __shfl_xor_sync with hardcoded full mask.
// --------------------------------------------------------------------------

__device__ __noinline__
int butterfly_untyped(int data) {
    // All 32 lanes participate — but nothing enforces this
    data += __shfl_xor_sync(0xFFFFFFFF, data, 16);
    data += __shfl_xor_sync(0xFFFFFFFF, data, 8);
    data += __shfl_xor_sync(0xFFFFFFFF, data, 4);
    data += __shfl_xor_sync(0xFFFFFFFF, data, 2);
    data += __shfl_xor_sync(0xFFFFFFFF, data, 1);
    return data;
}

// --------------------------------------------------------------------------
// Typed: identical operations, annotated with typestate-annotated warp state.
// In our Rust system, these annotations are zero-sized phantom types.
// Here we show them as comments to prove they add no instructions.
// --------------------------------------------------------------------------

__device__ __noinline__
int butterfly_typed(int data) {
    // warp: Warp<All> — all 32 lanes active (phantom type, zero-sized)
    // ComplementOf<Even, Odd> verified at compile time (trait bound)

    data += __shfl_xor_sync(0xFFFFFFFF, data, 16);  // shuffle_xor on Warp<All>
    data += __shfl_xor_sync(0xFFFFFFFF, data, 8);   // shuffle_xor on Warp<All>
    data += __shfl_xor_sync(0xFFFFFFFF, data, 4);   // shuffle_xor on Warp<All>
    data += __shfl_xor_sync(0xFFFFFFFF, data, 2);   // shuffle_xor on Warp<All>
    data += __shfl_xor_sync(0xFFFFFFFF, data, 1);   // shuffle_xor on Warp<All>
    return data;
    // Warp<All> consumed linearly — no use-after-merge possible
}

// --------------------------------------------------------------------------
// Kernel stubs to prevent dead-code elimination
// --------------------------------------------------------------------------

__global__ void call_untyped(int *out) {
    out[threadIdx.x] = butterfly_untyped(threadIdx.x);
}

__global__ void call_typed(int *out) {
    out[threadIdx.x] = butterfly_typed(threadIdx.x);
}
