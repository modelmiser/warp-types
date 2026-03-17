# 2. Background

This section provides background on GPU execution, warp-level primitives, and session types. Readers familiar with these topics may skip to §3.

## 2.1 GPU Execution Model

Modern GPUs achieve high throughput through massive parallelism. A GPU program, called a *kernel*, is executed by thousands of threads organized in a hierarchy:

- **Grid**: The entire kernel invocation
- **Block** (or *thread block*): A group of threads that can synchronize and share memory (typically 128-1024 threads)
- **Warp** (NVIDIA) or *Wavefront* (AMD): A group of threads that execute in lockstep (32 for NVIDIA, 64 for AMD)

The warp is the fundamental unit of execution. All threads in a warp execute the *same instruction* at the *same time*, but operate on different data—a model called SIMT (Single Instruction, Multiple Threads). This is similar to SIMD (Single Instruction, Multiple Data) on CPUs, but with a crucial difference: SIMT supports *divergence*.

### Divergence

When threads in a warp encounter a conditional branch, they may take different paths:

```cuda
if (threadIdx.x % 2 == 0) {
    // Even threads execute this
    do_even_work();
} else {
    // Odd threads execute this
    do_odd_work();
}
// All threads reconverge here
```

The hardware handles this by *predicated execution*: both paths are executed, but threads not taking a path are masked out. Their instructions are effectively no-ops, and their results are discarded.

During divergence, we say some threads are *active* and others are *inactive*. The set of active threads is tracked by an *active mask*—a 32-bit (or 64-bit) bitmask where bit *i* indicates whether lane *i* is active.

Divergence has performance implications (serialized execution of branches), but our focus is on *correctness*: operations that read from inactive lanes produce undefined behavior.

### Reconvergence

After a divergent branch, threads *reconverge*—all threads become active again at a *reconvergence point*. In structured control flow, this is the point immediately after the if-else or loop. NVIDIA's recent architectures (Volta and later) use *Independent Thread Scheduling*, which allows more flexible reconvergence, but the fundamental model remains: some operations require all threads to be active.

## 2.2 Warp-Level Primitives

GPUs provide special instructions for communication *within* a warp. These are much faster than going through shared memory because they operate directly on registers.

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

*Ballot* collects a predicate from each thread into a bitmask:

```cuda
// Returns a 32-bit mask: bit i is set if thread i's predicate is true
unsigned int mask = __ballot_sync(0xFFFFFFFF, predicate);
```

*Vote* operations compute collective predicates:

```cuda
bool all_true = __all_sync(0xFFFFFFFF, predicate);  // AND of all predicates
bool any_true = __any_sync(0xFFFFFFFF, predicate);  // OR of all predicates
```

These operations also require correct active masks. Reading a predicate from an inactive thread is undefined.

### The Active Mask Parameter

All modern warp primitives take a `mask` parameter specifying participating threads. This was added in CUDA 9.0 to make synchronization explicit. However, the programmer must ensure the mask matches reality:

```cuda
if (condition) {
    // WRONG: mask says all threads, but only some are here
    __shfl_xor_sync(0xFFFFFFFF, data, 1);

    // LESS WRONG: use __activemask() to get actual active threads
    __shfl_xor_sync(__activemask(), data, 1);
}
```

But even `__activemask()` is not a complete solution—it tells you which threads are active *right now*, but doesn't prevent you from expecting a value from a thread that couldn't have computed a meaningful result.

## 2.3 The Divergence Bug in Detail

Consider a filtered reduction where some lanes opt out:

```cuda
__device__ int filtered_sum(int* data, bool* keep, int lane) {
    if (keep[lane]) {
        int val = data[lane];

        // Butterfly reduction
        val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 1);

        return val;
    }
    return 0;
}
```

This code has undefined behavior. The shuffles read from all 32 lanes, but only some lanes have valid `val` values. Lanes where `keep[lane]` is false:

1. Did not execute `val = data[lane]`
2. Have undefined values in their `val` register
3. Will contribute garbage to the reduction

The correct code must either:
- Use `__activemask()` and handle partial participation, or
- Ensure non-participating lanes have a neutral value (e.g., 0 for sum), or
- Not use warp primitives in divergent code

Each approach requires careful reasoning about which lanes are active. Our type system automates this reasoning.

### Why This Bug Is Hard to Find

1. **It compiles successfully.** The CUDA compiler cannot track active masks through control flow.

2. **It may appear to work.** If inactive lanes happen to contain zeros or valid-looking data, the result may be plausible.

3. **It fails non-deterministically.** Different runs, different inputs, or different GPU architectures may expose or hide the bug.

4. **It's silent.** There's no exception, no error code—just wrong results.

5. **It's common.** Any time warp primitives are used after a conditional, this bug is possible.

## 2.4 Session Types

Session types are a type discipline for communication protocols, originally developed for the π-calculus [Honda 1993] and later extended to multiparty sessions [Honda, Yoshida, Carbone 2008].

### Basic Idea

A session type describes a protocol as a type. For example:

```
S = !int.?bool.end
```

This type describes a session that sends an integer, receives a boolean, then ends. The dual type—what the other participant sees—is:

```
S̄ = ?int.!bool.end
```

Session types ensure *communication safety*: well-typed programs follow their protocols and never get stuck waiting for a message that won't arrive.

### Branching

Protocols can branch:

```
S = !int.(?ok.end ⊕ ?error.!retry.S)
```

This sends an integer, then either receives "ok" and ends, or receives "error", sends "retry", and loops. The `⊕` (internal choice) means the sender decides which branch; `&` (external choice) means the receiver decides.

### Multiparty Session Types

In multiparty session types (MPST), more than two participants interact according to a global protocol. Each participant has a *local type* projected from the global protocol. The type system ensures:

1. **Safety**: No stuck states
2. **Progress**: The protocol eventually completes
3. **Fidelity**: Participants follow the prescribed protocol

### Our Extension: Quiescence

Traditional session types assume all participants remain active throughout the session. If a participant stops responding, the session is stuck.

GPU divergence introduces a different pattern: some participants *go quiescent*. They don't leave the session or fail—they temporarily stop participating, then rejoin at reconvergence. This is not a failure mode; it's the normal execution model.

We extend session types with:

- **Active sets**: Which participants are currently active
- **Quiescence**: A participant may become inactive (not failed, not departed—just paused)
- **Reconvergence**: Quiescent participants rejoin

This models GPU divergence naturally: diverging is branching where one branch is "go quiescent," and reconverging is joining where quiescent participants resume.

## 2.5 Summary

| Concept | GPU Term | Session Type Term |
|---------|----------|-------------------|
| Warp | 32 threads in lockstep | 32-party session |
| Lane | One thread in the warp | One participant |
| Divergence | Conditional branch | Protocol branch |
| Inactive lane | Masked-out thread | Quiescent participant |
| Reconvergence | Threads rejoin | Participants resume |
| Shuffle | Register exchange | All-to-all communication |
| Active mask | Which threads participate | Active set type |

Our contribution is recognizing this correspondence and building a type system that exploits it to prevent bugs.
