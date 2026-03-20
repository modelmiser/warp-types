// GPU Semantics Verification — warp-types project
// Tests hardware behavior for:
//   1. reduce_sum overflow (wrap vs saturate vs trap)
//   2. shuffle_down clamp (OOB lanes read own value?)
//   3. shuffle_idx OOB (src_lane >= 32 behavior)
//
// Build: nvcc -arch=sm_89 -o gpu_semantics_test gpu_semantics_test.cu
// Run:   ./gpu_semantics_test

#include <cstdio>
#include <cstdint>

// Test 1: Butterfly reduce_sum with INT32_MAX — does hardware wrap?
__global__ void test_reduce_overflow(int32_t* result) {
    int val = INT32_MAX;  // 2147483647 in every lane

    // 5-step butterfly XOR reduce (same as warp-types reduce_sum)
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);

    if (threadIdx.x == 0) {
        result[0] = val;
    }
}

// Test 2: shuffle_down with delta that exceeds lane count
__global__ void test_shuffle_down_clamp(int32_t* result) {
    int lane = threadIdx.x;
    int val = lane * 100;  // lane 0=0, lane 1=100, ..., lane 31=3100

    // delta=1: lane 31 should read own value (3100) if clamp, or lane 0 (0) if wrap
    int down1 = __shfl_down_sync(0xFFFFFFFF, val, 1);

    // delta=16: lanes 16-31 should read own value if clamp
    int down16 = __shfl_down_sync(0xFFFFFFFF, val, 16);

    // delta=31: only lane 0 reads from lane 31; lanes 1-31 read own value if clamp
    int down31 = __shfl_down_sync(0xFFFFFFFF, val, 31);

    // delta=32: ALL lanes should read own value if clamp (no source exists)
    int down32 = __shfl_down_sync(0xFFFFFFFF, val, 32);

    if (lane == 31) {
        result[0] = down1;   // expect 3100 (clamp) or 0 (wrap)
        result[1] = down16;  // expect 3100 (clamp) or 1500 (wrap from lane 15)
        result[2] = down31;  // expect 3100 (clamp)
        result[3] = down32;  // expect 3100 (clamp)
    }
    if (lane == 0) {
        result[4] = down31;  // lane 0 reads from lane 31 = 3100
    }
}

// Test 3: shuffle_idx with src_lane >= 32
__global__ void test_shuffle_idx_oob(int32_t* result) {
    int lane = threadIdx.x;
    int val = lane + 1;  // lane 0=1, lane 1=2, ..., lane 31=32

    // Normal: read from lane 5
    int idx5 = __shfl_sync(0xFFFFFFFF, val, 5);

    // OOB: read from lane 32 (first OOB value)
    int idx32 = __shfl_sync(0xFFFFFFFF, val, 32);

    // OOB: read from lane 33
    int idx33 = __shfl_sync(0xFFFFFFFF, val, 33);

    // OOB: read from lane 63 (max before next power of 2)
    int idx63 = __shfl_sync(0xFFFFFFFF, val, 63);

    if (lane == 0) {
        result[0] = idx5;   // expect 6 (lane 5 value)
        result[1] = idx32;  // unknown: clamp to 31? wrap to 0? own value?
        result[2] = idx33;  // unknown
        result[3] = idx63;  // unknown
    }
}

// Test 4: shuffle_up clamp behavior
__global__ void test_shuffle_up_clamp(int32_t* result) {
    int lane = threadIdx.x;
    int val = lane * 100;

    // delta=1: lane 0 should read own value (0) if clamp
    int up1 = __shfl_up_sync(0xFFFFFFFF, val, 1);

    // delta=16: lanes 0-15 should read own value if clamp
    int up16 = __shfl_up_sync(0xFFFFFFFF, val, 16);

    if (lane == 0) {
        result[0] = up1;   // expect 0 (clamp — own value)
        result[1] = up16;  // expect 0 (clamp — own value)
    }
    if (lane == 15) {
        result[2] = up16;  // expect 1500 (clamp — own value, since 15 < 16)
    }
    if (lane == 16) {
        result[3] = up16;  // expect 0 (reads from lane 0)
    }
}

int main() {
    int32_t *d_result, h_result[8];

    cudaMalloc(&d_result, 8 * sizeof(int32_t));

    printf("=== GPU Semantics Verification (warp-types) ===\n");
    printf("GPU: RTX 4000 Ada, Compute 8.9\n\n");

    // Test 1: Overflow
    cudaMemset(d_result, 0, 8 * sizeof(int32_t));
    test_reduce_overflow<<<1, 32>>>(d_result);
    cudaMemcpy(h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost);
    printf("TEST 1: reduce_sum overflow (INT32_MAX * 32 lanes)\n");
    printf("  Result: %d (0x%08X)\n", h_result[0], (uint32_t)h_result[0]);
    int64_t expected_wrap = (int64_t)INT32_MAX * 32;
    printf("  Expected if wrap: %d (0x%08X) [low 32 bits of %lld]\n",
           (int32_t)(expected_wrap & 0xFFFFFFFF),
           (uint32_t)(expected_wrap & 0xFFFFFFFF),
           (long long)expected_wrap);
    if (h_result[0] == (int32_t)(expected_wrap & 0xFFFFFFFF))
        printf("  VERDICT: WRAPS (two's complement, no trap)\n");
    else
        printf("  VERDICT: UNKNOWN behavior — result doesn't match wrap prediction\n");

    // Test 2: shuffle_down clamp
    cudaMemset(d_result, 0, 8 * sizeof(int32_t));
    test_shuffle_down_clamp<<<1, 32>>>(d_result);
    cudaMemcpy(h_result, d_result, 5 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    printf("\nTEST 2: shuffle_down clamp behavior\n");
    printf("  Lane 31, delta=1:  %d (expect 3100 if clamp, 0 if wrap)\n", h_result[0]);
    printf("  Lane 31, delta=16: %d (expect 3100 if clamp)\n", h_result[1]);
    printf("  Lane 31, delta=31: %d (expect 3100 if clamp)\n", h_result[2]);
    printf("  Lane 31, delta=32: %d (expect 3100 if clamp)\n", h_result[3]);
    printf("  Lane 0,  delta=31: %d (expect 3100 — reads from lane 31)\n", h_result[4]);
    if (h_result[0] == 3100 && h_result[1] == 3100)
        printf("  VERDICT: CLAMPS (OOB lanes read own value)\n");
    else if (h_result[0] == 0)
        printf("  VERDICT: WRAPS (lane 31+1 → lane 0)\n");
    else
        printf("  VERDICT: UNKNOWN\n");

    // Test 3: shuffle_idx OOB
    cudaMemset(d_result, 0, 8 * sizeof(int32_t));
    test_shuffle_idx_oob<<<1, 32>>>(d_result);
    cudaMemcpy(h_result, d_result, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    printf("\nTEST 3: shuffle_idx with src_lane >= 32\n");
    printf("  src=5:  %d (expect 6)\n", h_result[0]);
    printf("  src=32: %d", h_result[1]);
    if (h_result[1] == h_result[0] - 5 + 0 + 1) printf(" (=lane 0 value → wraps mod 32)");
    else if (h_result[1] == 1) printf(" (=own lane 0 value → clamps to self)");
    printf("\n");
    printf("  src=33: %d", h_result[2]);
    if (h_result[2] == 2) printf(" (=lane 1 value → wraps mod 32)");
    printf("\n");
    printf("  src=63: %d", h_result[3]);
    if (h_result[3] == 32) printf(" (=lane 31 value → wraps mod 32)");
    printf("\n");
    if (h_result[1] == 1 && h_result[2] == 2)
        printf("  VERDICT: WRAPS mod 32 (src_lane & 0x1F)\n");
    else
        printf("  VERDICT: see values above\n");

    // Test 4: shuffle_up clamp
    cudaMemset(d_result, 0, 8 * sizeof(int32_t));
    test_shuffle_up_clamp<<<1, 32>>>(d_result);
    cudaMemcpy(h_result, d_result, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    printf("\nTEST 4: shuffle_up clamp behavior\n");
    printf("  Lane 0,  delta=1:  %d (expect 0 if clamp)\n", h_result[0]);
    printf("  Lane 0,  delta=16: %d (expect 0 if clamp)\n", h_result[1]);
    printf("  Lane 15, delta=16: %d (expect 1500 if clamp)\n", h_result[2]);
    printf("  Lane 16, delta=16: %d (expect 0 — reads from lane 0)\n", h_result[3]);
    if (h_result[0] == 0 && h_result[2] == 1500 && h_result[3] == 0)
        printf("  VERDICT: CLAMPS (underflow lanes read own value)\n");
    else
        printf("  VERDICT: UNKNOWN\n");

    printf("\n=== Done ===\n");

    cudaFree(d_result);
    return 0;
}
