/**
 * Reproduction of cuda-samples #398: shuffle-mask bug in reduce7
 *
 * The bug: after block-level tree reduction with blockDim.x=32, only
 * thread 0 enters the final warp reduction. __ballot_sync(0xFFFFFFFF, ...)
 * returns mask=1. __shfl_down_sync(1, val, 16) reads from lane 16, which
 * is NOT in the mask. Result: undefined value folded into the sum.
 *
 * Expected sum of 32 ones = 32.
 * Buggy code may produce 1, 2, or some other wrong value.
 *
 * Source: https://github.com/NVIDIA/cuda-samples/issues/398
 */

#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// BUGGY VERSION (from cuda-samples reduce7 pattern)
// ============================================================================

__global__ void reduce7_buggy(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    // Block-level tree reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp reduction
    // BUG: when blockDim.x == 32, the loop above doesn't execute (32/2=16, 16 > 32 is false).
    // So we go straight to the warp reduction with all data still in sdata[].
    // The condition tid < blockDim.x / warpSize narrows to tid < 1 (only tid 0).
    if (tid < warpSize) {
        // This ballot gets only lane 0 when blockDim.x == 32
        unsigned mask = __ballot_sync(0xFFFFFFFF, tid < blockDim.x / warpSize);
        // When blockDim.x == 32: mask = 1 (only lane 0 voted true)
        // BUG: shfl_down reads from lane 16, which is NOT in mask
        if (tid < blockDim.x / warpSize) {
            sdata[tid] += __shfl_down_sync(mask, sdata[tid], 16);
            sdata[tid] += __shfl_down_sync(mask, sdata[tid], 8);
            sdata[tid] += __shfl_down_sync(mask, sdata[tid], 4);
            sdata[tid] += __shfl_down_sync(mask, sdata[tid], 2);
            sdata[tid] += __shfl_down_sync(mask, sdata[tid], 1);
        }
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================================
// CORRECT VERSION (the fix)
// ============================================================================

__global__ void reduce7_fixed(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    // Block-level tree reduction (same as above)
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Fixed warp reduction: use FULL mask, ALL lanes participate
    if (tid < 32) {
        // Load into register, inactive lanes get 0
        int myVal = (tid < blockDim.x) ? sdata[tid] : 0;

        // Full warp participates — all lanes have valid data
        myVal += __shfl_down_sync(0xFFFFFFFF, myVal, 16);
        myVal += __shfl_down_sync(0xFFFFFFFF, myVal, 8);
        myVal += __shfl_down_sync(0xFFFFFFFF, myVal, 4);
        myVal += __shfl_down_sync(0xFFFFFFFF, myVal, 2);
        myVal += __shfl_down_sync(0xFFFFFFFF, myVal, 1);

        if (tid == 0) g_odata[blockIdx.x] = myVal;
    }
}

// ============================================================================
// TEST HARNESS
// ============================================================================

int main() {
    const int N = 32;  // Exactly one warp — triggers the bug
    int h_idata[N];
    int h_odata_buggy = 0;
    int h_odata_fixed = 0;

    // Fill with ones — expected sum = 32
    for (int i = 0; i < N; i++) {
        h_idata[i] = 1;
    }

    int *d_idata, *d_odata;
    cudaMalloc(&d_idata, N * sizeof(int));
    cudaMalloc(&d_odata, sizeof(int));
    cudaMemcpy(d_idata, h_idata, N * sizeof(int), cudaMemcpyHostToDevice);

    printf("cuda-samples #398 Reproduction\n");
    printf("==============================\n");
    printf("Input: %d ones, expected sum = %d\n", N, N);
    printf("Grid: (1,1,1), Block: (%d,1,1)\n\n", N);

    // Run buggy version
    cudaMemset(d_odata, 0, sizeof(int));
    reduce7_buggy<<<1, N, N * sizeof(int)>>>(d_idata, d_odata, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_odata_buggy, d_odata, sizeof(int), cudaMemcpyDeviceToHost);

    // Run fixed version
    cudaMemset(d_odata, 0, sizeof(int));
    reduce7_fixed<<<1, N, N * sizeof(int)>>>(d_idata, d_odata, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_odata_fixed, d_odata, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Buggy reduce7: sum = %d  %s\n", h_odata_buggy,
           h_odata_buggy == N ? "(happened to be correct)" : "(WRONG)");
    printf("Fixed reduce7: sum = %d  %s\n", h_odata_fixed,
           h_odata_fixed == N ? "(correct)" : "(WRONG)");

    // Run buggy version multiple times to show non-determinism
    printf("\nBuggy version, 10 runs:\n");
    for (int run = 0; run < 10; run++) {
        cudaMemset(d_odata, 0, sizeof(int));
        reduce7_buggy<<<1, N, N * sizeof(int)>>>(d_idata, d_odata, N);
        cudaDeviceSynchronize();
        int result;
        cudaMemcpy(&result, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
        printf("  run %d: sum = %d%s\n", run, result,
               result == N ? "" : " (WRONG)");
    }

    // Test with larger block size — bug ALSO manifests here because
    // blockDim.x/warpSize < 32, so ballot mask is still incomplete
    const int N2 = 256;
    int h_idata2[N2];
    for (int i = 0; i < N2; i++) h_idata2[i] = 1;

    int *d_idata2;
    cudaMalloc(&d_idata2, N2 * sizeof(int));
    cudaMemcpy(d_idata2, h_idata2, N2 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_odata, 0, sizeof(int));
    reduce7_buggy<<<1, N2, N2 * sizeof(int)>>>(d_idata2, d_odata, N2);
    cudaDeviceSynchronize();
    int result_256;
    cudaMemcpy(&result_256, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nBuggy reduce7 with block_size=256: sum = %d  %s\n",
           result_256, result_256 == N2 ? "(correct)" : "(WRONG — bug affects any blockDim/warpSize < 32)");

    cudaFree(d_idata);
    cudaFree(d_idata2);
    cudaFree(d_odata);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\nCUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\nGPU: ");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%s (compute %d.%d)\n", prop.name, prop.major, prop.minor);

    return 0;
}
