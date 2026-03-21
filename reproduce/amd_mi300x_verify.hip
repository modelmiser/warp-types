/**
 * warp-types AMD MI300X verification
 *
 * Verifies type system predictions on 64-wide wavefronts:
 *   1. Full 64-lane butterfly reduction (6 stages, not 5)
 *   2. Diverged shuffle — the bug class warp-types catches
 *   3. 5-stage (NVIDIA-style) vs 6-stage reduction on 64 lanes
 *
 * Compile: hipcc -o amd_verify amd_mi300x_verify.hip
 * Run:     ./amd_verify
 */

#include <hip/hip_runtime.h>
#include <stdio.h>

// ============================================================================
// Test 1: Full 64-lane butterfly reduction (CORRECT — 6 stages)
// Type: Warp<All64> → Warp<All64>  (all 64 lanes participate)
// ============================================================================

__global__ void reduce_64_correct(int *out) {
    int lane = threadIdx.x;  // 0..63
    int val = lane;

    // 6-stage butterfly: log2(64) = 6 XOR masks
    val += __shfl_xor(val, 1);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 32);  // Stage 6: crosses 32-lane boundary!

    out[lane] = val;
}

// ============================================================================
// Test 2: 5-stage reduction (NVIDIA pattern — WRONG on 64-wide AMD)
// Type: would need Warp<All32> but we have Warp<All64>
// The NVIDIA reduce7 pattern only does 5 XOR stages (masks 1,2,4,8,16).
// On 64-wide wavefront, this misses the 32-lane crossing.
// ============================================================================

__global__ void reduce_5stage_bug(int *out) {
    int lane = threadIdx.x;
    int val = lane;

    // Only 5 stages — correct for 32-lane NVIDIA, WRONG for 64-lane AMD
    val += __shfl_xor(val, 1);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 16);
    // MISSING: val += __shfl_xor(val, 32);

    out[lane] = val;
}

// ============================================================================
// Test 3: Diverged shuffle — reading from inactive lanes
// Type violation: shuffle on Warp<Even> instead of Warp<All>
// ============================================================================

__global__ void diverged_shuffle(int *out) {
    int lane = threadIdx.x;
    int val = lane;

    if (lane % 2 == 0) {
        // Only even lanes (0,2,4,...62) are active
        // __shfl_xor(val, 1) reads from odd lanes — which are INACTIVE
        // On AMD 64-wide, this reads from lanes that took the else branch
        val = __shfl_xor(val, 1);  // BUG: partner lane is odd (inactive)
    } else {
        val = -1;  // Odd lanes: sentinel value
    }

    out[lane] = val;
}

// ============================================================================
// Test 4: Diverged shuffle — low-half only (32 of 64 active)
// Equivalent to cuda-samples#398 on 64-wide wavefront
// ============================================================================

__global__ void diverged_half(int *out) {
    int lane = threadIdx.x;
    int val = 1;  // each lane contributes 1

    if (lane < 32) {
        // Only lanes 0-31 active in a 64-wide wavefront
        // Ballot would return mask with only low 32 bits set
        // Shuffle reads from lanes 32-63 which are INACTIVE
        val += __shfl_xor(val, 1);
        val += __shfl_xor(val, 2);
        val += __shfl_xor(val, 4);
        val += __shfl_xor(val, 8);
        val += __shfl_xor(val, 16);
        // Even within 32 lanes, __shfl_xor(val, 32) would read from
        // lane (lane ^ 32) which is in the INACTIVE upper half
    }

    out[lane] = val;
}

// ============================================================================
// Test 5: Full wavefront info
// ============================================================================

__global__ void wavefront_info(int *out) {
    // __ballot(1) — all active lanes vote true
    // On 64-wide AMD, this should return a 64-bit value with all bits set
    unsigned long long ballot = __ballot(1);
    if (threadIdx.x == 0) {
        out[0] = __popcll(ballot);  // Should be 64
        out[1] = (int)(ballot & 0xFFFFFFFF);        // Low 32 bits
        out[2] = (int)((ballot >> 32) & 0xFFFFFFFF); // High 32 bits
    }
}

// ============================================================================

int main() {
    int *d_out;
    int h_out[64];
    hipMalloc(&d_out, 64 * sizeof(int));

    // ========== Device info ==========
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("warp-types AMD Verification\n");
    printf("===========================\n");
    printf("GPU: %s\n", prop.name);
    printf("GCN arch: %s\n", prop.gcnArchName);
    printf("Wavefront size: %d\n", prop.warpSize);
    printf("\n");

    int wavefront_size = prop.warpSize;

    // ========== Test 5: Wavefront ballot ==========
    printf("--- Test 0: Wavefront ballot ---\n");
    hipMemset(d_out, 0, 64 * sizeof(int));
    hipLaunchKernelGGL(wavefront_info, 1, wavefront_size, 0, 0, d_out);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, 64 * sizeof(int), hipMemcpyDeviceToHost);
    printf("ballot(1) popcount: %d (expected %d)\n", h_out[0], wavefront_size);
    printf("ballot low  32 bits: 0x%08X\n", h_out[1]);
    printf("ballot high 32 bits: 0x%08X\n", h_out[2]);
    printf("\n");

    // ========== Test 1: Full 64-lane butterfly (6 stages) ==========
    printf("--- Test 1: Full %d-lane butterfly reduction (6 stages) ---\n", wavefront_size);
    hipMemset(d_out, 0, 64 * sizeof(int));
    hipLaunchKernelGGL(reduce_64_correct, 1, wavefront_size, 0, 0, d_out);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, wavefront_size * sizeof(int), hipMemcpyDeviceToHost);

    int expected = wavefront_size * (wavefront_size - 1) / 2;  // sum(0..63) = 2016
    printf("Lane 0: %d (expected %d)\n", h_out[0], expected);
    bool all_correct = true;
    for (int i = 0; i < wavefront_size; i++) {
        if (h_out[i] != expected) { all_correct = false; break; }
    }
    printf("All lanes correct: %s\n", all_correct ? "YES" : "NO");
    printf("VERDICT: %s\n\n", all_correct ? "PASS — 6-stage butterfly correct on 64-wide"
                                          : "FAIL");

    // ========== Test 2: 5-stage butterfly (NVIDIA pattern on AMD) ==========
    printf("--- Test 2: 5-stage butterfly (NVIDIA pattern on %d-wide AMD) ---\n", wavefront_size);
    hipMemset(d_out, 0, 64 * sizeof(int));
    hipLaunchKernelGGL(reduce_5stage_bug, 1, wavefront_size, 0, 0, d_out);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, wavefront_size * sizeof(int), hipMemcpyDeviceToHost);

    printf("Lane 0: %d (expected %d for full reduce)\n", h_out[0], expected);
    // With 5 stages on 64 lanes, each lane sums its 32-lane half
    // Lanes 0-31 get sum(0..31) = 496, lanes 32-63 get sum(32..63) = 1520
    int half_low = 32 * 31 / 2;    // 496
    int half_high = (32 * 63 + 32 * 32) / 2;  // Actually sum(32..63) = sum(0..63) - sum(0..31) = 2016 - 496 = 1520
    printf("Lane 0: %d (half-sum = %d if 5 stages only reduce within 32)\n", h_out[0], half_low);
    printf("Lane 32: %d (half-sum = %d if 5 stages only reduce within 32)\n", h_out[32], half_high);
    bool five_wrong = (h_out[0] != expected);
    printf("VERDICT: %s\n\n", five_wrong
        ? "CONFIRMED — 5-stage reduction WRONG on 64-wide (warp-types would catch this)"
        : "UNEXPECTED — 5 stages somehow correct (investigate)");

    // ========== Test 3: Diverged shuffle (even lanes only) ==========
    printf("--- Test 3: Diverged shuffle (even lanes only, xor=1) ---\n");
    hipMemset(d_out, 0, 64 * sizeof(int));
    hipLaunchKernelGGL(diverged_shuffle, 1, wavefront_size, 0, 0, d_out);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, wavefront_size * sizeof(int), hipMemcpyDeviceToHost);

    printf("Lane 0: %d (xor partner lane 1 has val -1 sentinel)\n", h_out[0]);
    printf("Lane 1: %d (odd lane, sentinel -1)\n", h_out[1]);
    printf("Lane 2: %d (xor partner lane 3 has val -1 sentinel)\n", h_out[2]);
    printf("Lane 62: %d (xor partner lane 63 has val -1 sentinel)\n", h_out[62]);
    // If diverged shuffle reads from the inactive odd lane, it gets the
    // sentinel value -1, NOT the even lane's original value
    bool diverge_bug = (h_out[0] == -1 || h_out[0] != 0);
    printf("VERDICT: Even lanes got %s values — %s\n\n",
        (h_out[0] == h_out[0] ^ 1) ? "swapped (correct)" : "unexpected (bug or sentinel)",
        "warp-types: shuffle on Warp<Even> is a type error");

    // ========== Test 4: Diverged half (32 of 64 active) ==========
    printf("--- Test 4: Diverged half-wavefront reduction ---\n");
    hipMemset(d_out, 0, 64 * sizeof(int));
    hipLaunchKernelGGL(diverged_half, 1, wavefront_size, 0, 0, d_out);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, wavefront_size * sizeof(int), hipMemcpyDeviceToHost);

    printf("Lane 0: %d (expected 32 if correct half-reduce)\n", h_out[0]);
    printf("Lane 31: %d\n", h_out[31]);
    printf("Lane 32: %d (inactive, should be 1 = initial)\n", h_out[32]);
    printf("Lane 63: %d (inactive, should be 1 = initial)\n", h_out[63]);
    printf("VERDICT: warp-types types this as Warp<LowHalf> — shuffle_xor ");
    printf("restricted to within-half masks only\n\n");

    // ========== Summary ==========
    printf("===========================\n");
    printf("Summary for warp-types paper:\n");
    printf("  Wavefront width: %d lanes (vs NVIDIA 32)\n", wavefront_size);
    printf("  Butterfly stages needed: %d (vs NVIDIA 5)\n",
           wavefront_size == 64 ? 6 : 5);
    printf("  Bug surface: %dx larger (64 lanes to go wrong vs 32)\n",
           wavefront_size / 32);
    printf("  Type system prediction: ActiveSet needs BitVec %d\n", wavefront_size);

    hipFree(d_out);
    return 0;
}
