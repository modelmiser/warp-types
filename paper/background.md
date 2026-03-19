# 2. Background

This section provides background on GPU execution, warp-level primitives, and session types. Readers familiar with these topics may skip to §3.

## 2.1 GPU Execution Model

Modern GPUs achieve high throughput through massive parallelism. A GPU program (*kernel*) is executed by thousands of threads organized hierarchically: **grids** contain **blocks** (128-1024 threads sharing memory), which contain **warps** (NVIDIA, 32 threads) or **wavefronts** (AMD, 64 threads).

The warp is the fundamental execution unit. All threads in a warp execute the same instruction simultaneously on different data—the SIMT (Single Instruction, Multiple Threads) model. Unlike CPU SIMD, SIMT supports *divergence*.

### Divergence

When threads in a warp encounter a conditional branch, they may take different paths:

```cuda
if (threadIdx.x % 2 == 0) {
    do_even_work();
} else {
    do_odd_work();
}
// All threads reconverge here
```

The hardware handles this by *predicated execution*: both paths execute, but threads not taking a path are masked out. During divergence, some threads are *active* and others *inactive*, tracked by an *active mask*—a bitmask where bit *i* indicates whether lane *i* is active.

Divergence has performance implications (serialized branches), but our focus is on *correctness*: operations that read from inactive lanes produce undefined behavior.

### Reconvergence

After a divergent branch, threads *reconverge* at a *reconvergence point*—typically the point after the if-else or loop. NVIDIA's Volta and later architectures use *Independent Thread Scheduling* for more flexible reconvergence, but the fundamental invariant remains: some operations require all threads to be active.

## 2.2 Warp-Level Primitives

GPUs provide special instructions for communication *within* a warp. These are much faster than shared memory because they operate directly on registers.

### Shuffle Operations

*Shuffle* instructions allow threads to read values from other threads' registers:

```cuda
// Each thread reads from its XOR partner
int partner_val = __shfl_xor_sync(mask, my_val, 1);

// Each thread reads from the thread delta positions below
int lower_val = __shfl_down_sync(mask, my_val, delta);

// Each thread reads from an arbitrary source lane
int other_val = __shfl_sync(mask, my_val, src_lane);
```

The `mask` parameter specifies which threads participate. The CUDA documentation states:

> "Threads may only read data from another thread which is actively participating in the shuffle command. If the target thread is inactive, the retrieved value is undefined."

This is the source of our bug class: if the mask claims all threads participate (`0xFFFFFFFF`) but some threads are actually inactive due to divergence, the shuffle reads undefined values.

### Ballot and Vote

*Ballot* collects a predicate from each thread into a bitmask; *vote* operations (`__all_sync`, `__any_sync`) compute collective predicates. Both require correct active masks—reading from an inactive thread is undefined.

```cuda
unsigned int mask = __ballot_sync(0xFFFFFFFF, predicate);
bool all_true = __all_sync(0xFFFFFFFF, predicate);
```

### The Active Mask Parameter

All modern warp primitives take a `mask` parameter specifying participating threads, added in CUDA 9.0 to make synchronization explicit. However, the programmer must ensure the mask matches reality:

```cuda
if (condition) {
    // WRONG: mask says all threads, but only some are here
    __shfl_xor_sync(0xFFFFFFFF, data, 1);

    // LESS WRONG: use __activemask() to get actual active threads
    __shfl_xor_sync(__activemask(), data, 1);
}
```

But even `__activemask()` is not a complete solution—it tells you which threads are active *right now*, but doesn't prevent you from expecting a value from a thread that couldn't have computed a meaningful result.

## 2.3 The Divergence Bug

The bug class arises whenever warp primitives are used inside divergent code: the mask may claim threads participate that are actually inactive, yielding undefined values. The bug compiles without warning, may appear to work when inactive lanes happen to contain plausible data, fails non-deterministically across runs and architectures, and produces silently wrong results. We present a detailed catalog of bug patterns and their type-level prevention in §7.

## 2.4 Session Types

Session types [Honda 1993] are a type discipline for communication protocols, extended to multiparty sessions (MPST) by Honda, Yoshida, and Carbone [2008]. A session type describes a protocol's structure—sends, receives, branches, and recursion—as a type, ensuring *communication safety*: well-typed programs follow their protocols and never deadlock. In MPST, each of *n* participants holds a *local type* projected from a global protocol, guaranteeing safety (no stuck states), progress (eventual completion), and fidelity (adherence to the prescribed protocol).

### Our Extension: Quiescence

Traditional session types assume all participants remain active throughout the session. GPU divergence introduces a different pattern: some participants *go quiescent*. They don't leave the session or fail—they temporarily stop participating, then rejoin at reconvergence. This is not a failure mode; it's the normal execution model.

We extend session types with:

- **Active sets**: Which participants are currently active
- **Quiescence**: A participant may become inactive (not failed, not departed—just paused)
- **Reconvergence**: Quiescent participants rejoin

This models GPU divergence naturally: diverging is branching where one branch is "go quiescent," and reconverging is joining where quiescent participants resume. Our contribution is recognizing this correspondence and building a type system that exploits it to prevent bugs.
