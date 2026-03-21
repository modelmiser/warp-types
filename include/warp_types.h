// warp_types.h — C++20 type-safe warp programming for CUDA and HIP
//
// Mirrors the Rust warp-types library's compile-time safety guarantees
// using C++20 concepts and requires clauses.
//
// Core rule: shuffle/reduce/ballot only compile on Warp<All>.
// After diverge_even_odd(), calling shuffle_xor() is a compile error.
// Merge complementary sub-warps to get Warp<All> back.
//
// Three modes:
//   1. CUDA device code  — real __shfl_xor_sync intrinsics  (nvcc --std=c++20)
//   2. HIP  device code  — real __shfl_xor intrinsics       (hipcc --std=c++20)
//   3. Host-only          — modeling mode, identity shuffles  (g++ --std=c++20)
//
// Requires: C++20
// License:  MIT (same as warp-types)
//
// See also: https://github.com/modelmiser/warp-types

#pragma once

#include <cstdint>
#include <concepts>
#include <utility>

// ============================================================================
// Platform detection
// ============================================================================

#if defined(__CUDACC__)
  #define WT_DEVICE   __device__ __forceinline__
  #define WT_HOST_DEVICE __host__ __device__ __forceinline__
  #define WT_SHFL_XOR(val, mask)    __shfl_xor_sync(0xFFFFFFFFu, (val), (mask))
  #define WT_SHFL_DOWN(val, delta)  __shfl_down_sync(0xFFFFFFFFu, (val), (delta))
  #define WT_BALLOT(pred)           __ballot_sync(0xFFFFFFFFu, (pred))
  #define WT_WARP_SIZE 32
#elif defined(__HIPCC__)
  #define WT_DEVICE   __device__ __attribute__((always_inline))
  #define WT_HOST_DEVICE __host__ __device__ __attribute__((always_inline))
  #define WT_SHFL_XOR(val, mask)    __shfl_xor((val), (mask))
  #define WT_SHFL_DOWN(val, delta)  __shfl_down((val), (delta))
  #define WT_BALLOT(pred)           __ballot((pred))
  #if defined(__gfx9__) || defined(__gfx10__) || defined(__gfx11__)
    #define WT_WARP_SIZE 64
  #else
    #define WT_WARP_SIZE 32
  #endif
#else
  // Host-only: modeling mode (no real shuffles — identity, like Rust CPU path)
  #define WT_DEVICE   inline
  #define WT_HOST_DEVICE inline
  #define WT_SHFL_XOR(val, mask)    (val)
  #define WT_SHFL_DOWN(val, delta)  (val)
  #define WT_BALLOT(pred)           static_cast<uint32_t>((pred) ? 1u : 0u)
  #define WT_WARP_SIZE 32
#endif

namespace warp_types {

// ============================================================================
// Active set concept and concrete types
// ============================================================================

/// Every active set has a compile-time MASK and NAME.
template<typename S>
concept ActiveSet = requires {
    { S::MASK } -> std::convertible_to<uint64_t>;
    { S::NAME } -> std::convertible_to<const char*>;
};

// --- 32-lane active sets (NVIDIA warp) ---

struct All      { static constexpr uint64_t MASK = 0xFFFF'FFFF;
                  static constexpr const char* NAME = "All"; };
struct Even     { static constexpr uint64_t MASK = 0x5555'5555;
                  static constexpr const char* NAME = "Even"; };
struct Odd      { static constexpr uint64_t MASK = 0xAAAA'AAAA;
                  static constexpr const char* NAME = "Odd"; };
struct LowHalf  { static constexpr uint64_t MASK = 0x0000'FFFF;
                  static constexpr const char* NAME = "LowHalf"; };
struct HighHalf { static constexpr uint64_t MASK = 0xFFFF'0000;
                  static constexpr const char* NAME = "HighHalf"; };
struct Lane0    { static constexpr uint64_t MASK = 0x0000'0001;
                  static constexpr const char* NAME = "Lane0"; };
struct NotLane0 { static constexpr uint64_t MASK = 0xFFFF'FFFE;
                  static constexpr const char* NAME = "NotLane0"; };
struct EvenLow  { static constexpr uint64_t MASK = 0x0000'5555;
                  static constexpr const char* NAME = "EvenLow"; };
struct EvenHigh { static constexpr uint64_t MASK = 0x5555'0000;
                  static constexpr const char* NAME = "EvenHigh"; };
struct OddLow   { static constexpr uint64_t MASK = 0x0000'AAAA;
                  static constexpr const char* NAME = "OddLow"; };
struct OddHigh  { static constexpr uint64_t MASK = 0xAAAA'0000;
                  static constexpr const char* NAME = "OddHigh"; };
struct Empty    { static constexpr uint64_t MASK = 0x0000'0000;
                  static constexpr const char* NAME = "Empty"; };

// --- 64-lane active sets (AMD wavefront — enable with WT_WARP_SIZE=64) ---

struct All64    { static constexpr uint64_t MASK = 0xFFFF'FFFF'FFFF'FFFF;
                  static constexpr const char* NAME = "All64"; };
struct Even64   { static constexpr uint64_t MASK = 0x5555'5555'5555'5555;
                  static constexpr const char* NAME = "Even64"; };
struct Odd64    { static constexpr uint64_t MASK = 0xAAAA'AAAA'AAAA'AAAA;
                  static constexpr const char* NAME = "Odd64"; };

// ============================================================================
// Complement relationship (compile-time)
// ============================================================================

/// Two sets are complements if they cover All and don't overlap.
template<typename S1, typename S2>
concept ComplementOf = ActiveSet<S1> && ActiveSet<S2>
    && (S1::MASK | S2::MASK) == All::MASK
    && (S1::MASK & S2::MASK) == 0;

// ============================================================================
// Data value types — thin wrappers matching Rust's repr(transparent) types
// ============================================================================

/// A value guaranteed uniform across all lanes.
/// Same ABI as T (repr(transparent) in Rust).
template<typename T>
struct Uniform {
    T value;

    WT_HOST_DEVICE constexpr T get() const { return value; }
    WT_HOST_DEVICE static constexpr Uniform from_const(T v) { return {v}; }
};

/// A value that may differ per lane.
/// Same ABI as T (repr(transparent) in Rust).
template<typename T>
struct PerLane {
    T value;

    WT_HOST_DEVICE constexpr T get() const { return value; }
    WT_HOST_DEVICE static constexpr PerLane from(T v) { return {v}; }
};

// ============================================================================
// Warp<S> — the core type
// ============================================================================

/// A warp handle parameterized by active lane set.
///
/// Shuffle, reduce, and ballot methods only exist when S is All.
/// After diverge_even_odd(), those methods vanish from the type —
/// calling them is a compile error, not a runtime crash.
///
/// Zero-sized: Warp<S> carries no runtime data.
///
/// C++ limitation: unlike Rust's linear types, C++ can't enforce
/// move-only semantics on empty types. The protocol (don't use a
/// warp handle after diverge) is enforced by API design, not the
/// language. Follow the same discipline as Rust: treat Warp<S> as
/// consumed after diverge, restored after merge.
template<ActiveSet S>
class Warp {
public:
    static constexpr uint64_t MASK = S::MASK;

    /// Kernel entry point — all lanes active.
    WT_HOST_DEVICE static constexpr Warp kernel_entry()
        requires std::same_as<S, All>
    {
        return {};
    }

    // ========================================================================
    // Shuffle operations — ONLY on Warp<All>
    // ========================================================================

    /// Butterfly shuffle: lane[i] exchanges with lane[i ^ mask].
    /// CUDA: __shfl_xor_sync(0xFFFFFFFF, val, mask)
    /// HIP:  __shfl_xor(val, mask)
    template<typename T>
    WT_DEVICE PerLane<T> shuffle_xor(PerLane<T> data, uint32_t mask) const
        requires std::same_as<S, All>
    {
        return {WT_SHFL_XOR(data.value, mask)};
    }

    /// Shuffle down: lane[i] reads from lane[i + delta].
    /// CUDA: __shfl_down_sync(0xFFFFFFFF, val, delta)
    template<typename T>
    WT_DEVICE PerLane<T> shuffle_down(PerLane<T> data, uint32_t delta) const
        requires std::same_as<S, All>
    {
        return {WT_SHFL_DOWN(data.value, delta)};
    }

    // ========================================================================
    // Reductions — ONLY on Warp<All>
    // ========================================================================

    /// Butterfly reduce-sum across all lanes. Result is uniform.
    template<typename T>
    WT_DEVICE Uniform<T> reduce_sum(PerLane<T> data) const
        requires std::same_as<S, All>
    {
        T val = data.value;
#if WT_WARP_SIZE == 64
        val += WT_SHFL_XOR(val, 32);
#endif
        val += WT_SHFL_XOR(val, 16);
        val += WT_SHFL_XOR(val, 8);
        val += WT_SHFL_XOR(val, 4);
        val += WT_SHFL_XOR(val, 2);
        val += WT_SHFL_XOR(val, 1);
        return {val};
    }

    /// Ballot: collect per-lane predicate into a bitmask.
    WT_DEVICE Uniform<uint32_t> ballot(bool predicate) const
        requires std::same_as<S, All>
    {
        return {WT_BALLOT(predicate)};
    }

    /// Broadcast: all lanes get the same value (identity in SIMT).
    template<typename T>
    WT_HOST_DEVICE constexpr PerLane<T> broadcast(T value) const
        requires std::same_as<S, All>
    {
        return {value};
    }

    // ========================================================================
    // Diverge — split into complementary sub-warps
    // ========================================================================

    /// Split into Even and Odd sub-warps.
    WT_HOST_DEVICE constexpr std::pair<Warp<Even>, Warp<Odd>>
    diverge_even_odd() const
        requires std::same_as<S, All>
    {
        return {{}, {}};
    }

    /// Split into LowHalf and HighHalf sub-warps.
    WT_HOST_DEVICE constexpr std::pair<Warp<LowHalf>, Warp<HighHalf>>
    diverge_low_high() const
        requires std::same_as<S, All>
    {
        return {{}, {}};
    }
};

// ============================================================================
// Merge — rejoin complementary sub-warps
// ============================================================================

/// Merge two complementary sub-warps back into Warp<All>.
/// Compile error if the sets don't complement (e.g., merging Even + LowHalf).
template<ActiveSet S1, ActiveSet S2>
    requires ComplementOf<S1, S2>
WT_HOST_DEVICE constexpr Warp<All> merge(Warp<S1>, Warp<S2>) {
    return {};
}

} // namespace warp_types

// ============================================================================
// Cleanup platform macros (don't leak into user code)
// ============================================================================

// Users who need these can #define WT_KEEP_MACROS before including this header.
#ifndef WT_KEEP_MACROS
  #undef WT_DEVICE
  #undef WT_HOST_DEVICE
  #undef WT_SHFL_XOR
  #undef WT_SHFL_DOWN
  #undef WT_BALLOT
  // WT_WARP_SIZE is kept — users may need it.
#endif
