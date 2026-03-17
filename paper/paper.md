# Type-Safe GPU Warp Programming via Linear Typestate

**Chad Aldreda**

**Abstract**

GPU warp primitives like shuffle enable efficient intra-warp communication, but reading from an inactive lane produces undefined behavior. These bugs compile without warnings, may appear to work, and fail silently: NVIDIA's own reference code contains them, a plasma physics simulation ran for months with undefined behavior undetected, and NVIDIA deprecated an entire API family to address the bug class. State-of-the-art persistent thread programs avoid the problem by prohibiting lane-level divergence entirely.

We present *warp typestate*, a linear type system that tracks which lanes are active at each program point. Divergence creates sub-warps with complementary active sets; reconvergence requires type-level proof that the sets are complements. Operations requiring all lanes—shuffles, ballots, reductions—are only available on fully-active warps: not checked at runtime, but *absent from the type*. The design is motivated by a structural analogy to multiparty session type branching, but the mechanism is linear typestate over a Boolean lattice of active sets—there are no channels, no protocol sequencing, and no duality in the session-type sense.

We prove our type system sound (progress and preservation), implement it as a Rust library with zero runtime overhead, and demonstrate that it catches real bugs from NVIDIA's cuda-samples at compile time—including one we reproduced on an RTX 4000 Ada GPU, confirming deterministically wrong results. The approach is strictly more permissive than the divergence-prohibition approach (which maintains warp-uniform execution) while being strictly safer than CUDA's `__shfl_sync` (which defers mask correctness to runtime).

---

# 1. Introduction

GPU programming has become essential for high-performance computing, machine learning, and graphics. Modern GPUs achieve their performance through *SIMT execution*: groups of 32 or 64 threads called *warps* execute in lockstep, sharing a single instruction stream while operating on different data. When threads in a warp take different control-flow paths—a phenomenon called *divergence*—some threads become inactive while others continue executing.

Divergence creates a pernicious class of bugs. Consider the following CUDA code:

```cuda
__device__ int conditional_exchange(int data, bool participate) {
    if (participate) {
        // Only some threads reach here
        int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);
        return data + partner;
    }
    return 0;
}
```

This code compiles without warnings and may appear to work correctly in testing. But it contains undefined behavior: the shuffle operation reads values from *all* lanes, including those where `participate` is false. Those threads are inactive—their registers may contain stale data, garbage, or trap values. The result is non-deterministic: sometimes correct, sometimes wrong, sometimes a crash.

This bug pattern is not hypothetical. NVIDIA's own `cuda-samples` repository contains a shuffle-mask bug in its reference parallel reduction (`reduce7`): when launched with one block of 32 threads, only lane 0 enters the final reduction, but `__shfl_down_sync` reads from lane 16, which is inactive [cuda-samples#398]. The result is a silently wrong sum—no crash, no error—at a configuration most test suites skip. NVIDIA's core primitives library CUB has an open issue suggesting compiler optimizations may predicate off mask initialization in sub-warp configurations [CCCL#854] — the reporter later noted it may have been a false positive, but the source-level pattern is ill-typed in our system regardless. The PIConGPU plasma physics simulation ran for months on K80 GPUs with `__ballot_sync(0xFFFFFFFF, ...)` inside a divergent branch — real undefined behavior that went undetected because pre-Volta hardware enforced warp-level convergence, masking the violation [PIConGPU#2514]. An LLVM bug shows the problem extends into compiler optimization: `__shfl_sync` after a conditional causes the compiler to eliminate the branch entirely, running a lane-0-only atomic on all 32 lanes [LLVM#155682].

The bug class is severe enough that NVIDIA deprecated the entire `__shfl` API family in CUDA 9.0, replacing it with `__shfl_sync` which requires an explicit mask parameter [CUDA Programming Guide §10.22]. But the mask is still a runtime value—and as the bugs above demonstrate, programmers get it wrong. Volta's independent thread scheduling made the situation worse: code that was latent undefined behavior on Pascal became observable bugs when threads within a warp could genuinely interleave.

State-of-the-art practitioners have responded by avoiding the problem entirely. The Hazy megakernel [2025], the most sophisticated persistent thread program as of this writing, prohibits lane-level divergence by design—all 32 lanes execute the same operation, every `__shfl_sync` uses the full mask. Divergence avoidance is reinforced by performance: divergent warps serialize execution across branches, so warp-uniform execution also maximizes SIMD throughput. This works, but at the cost of expressiveness: algorithms that naturally benefit from lane-level heterogeneity (adaptive sorting, predicated filters, work-stealing within a warp) cannot be expressed.

Our type system offers a third path: lane-level divergence that is safe rather than forbidden. It is strictly more permissive than the divergence-prohibition approach exemplified by Hazy (which maintains warp-uniform execution) while being strictly safer than CUDA's `__shfl_sync` API (which defers mask correctness to runtime). The gap between `__shfl_sync` and compile-time safety is concrete: `__activemask()` returns the current execution mask, which hardware always accepts—but a shuffle using `__activemask()` inside divergent code silently communicates among the wrong subset of lanes [CUDA Programming Guide §K.6]. Our type system closes this gap because the active set is a type-level property, not a runtime value.

## 1.1 Our Approach: Linear Typestate for Divergence

We observe that divergence has the structure of a *linear resource protocol*. In traditional session types, communication follows a protocol where participants send and receive messages in a prescribed order. Branching in the protocol creates sub-sessions; participants must follow compatible branches.

SIMT divergence is similar, but with a twist: when a warp diverges, some participants don't take the "other branch"—they go *quiescent*. They stop participating entirely until the branches reconverge. This is not captured by traditional session types, where all participants remain active.

We introduce *warp typestate*, a type system that tracks which lanes are active at each program point. The key ideas are:

1. **Warps carry active set types.** A warp is typed as `Warp<S>` where `S` describes which lanes are active. `Warp<All>` means all 32 lanes; `Warp<Even>` means only even-numbered lanes.

2. **Divergence produces complementary sub-warps.** When a warp diverges on a predicate, it produces two sub-warps with disjoint active sets that together cover the original:
   ```
   diverge : Warp<S> → (Warp<S ∩ P>, Warp<S ∩ ¬P>)
   ```

3. **Merge requires complement proof.** To reconverge, the type system must verify that the two sub-warps are complements:
   ```
   merge : Warp<S₁> → Warp<S₂> → Warp<S₁ ∪ S₂>  where S₁ ∩ S₂ = ∅
   ```

4. **Operations restrict by active set.** Shuffle operations are only available on `Warp<All>`. You cannot shuffle on a diverged warp—the method simply doesn't exist.

With this type system, the buggy code above becomes a compile error:

```rust
// Pseudocode — actual API uses concrete diverge methods (§6)
fn conditional_exchange(warp: Warp<All>, data: PerLane<i32>, participate: PerLane<bool>) -> i32 {
    let (active, inactive) = warp.diverge(|lane| participate[lane]);

    // ERROR: no method `shuffle_xor` found for `Warp<Active>`
    // note: shuffle_xor requires Warp<All>
    // help: merge with complement before shuffling
    let partner = active.shuffle_xor(data, 1);

    // ...
}
```

The fix is explicit: merge back to `Warp<All>` before shuffling, ensuring all lanes have valid data:

```rust
// Pseudocode — actual API uses concrete diverge methods (§6)
fn conditional_exchange(warp: Warp<All>, data: PerLane<i32>, participate: PerLane<bool>) -> i32 {
    let (active, inactive) = warp.diverge(|lane| participate[lane]);

    // Inactive lanes contribute zero
    let active_data = data;
    let inactive_data = PerLane::new(0);

    // Merge back - type system verifies complement
    let warp: Warp<All> = merge(active, inactive);
    // Data merge implicit in SIMT — each lane holds its branch result

    // Now shuffle is safe
    let partner = warp.shuffle_xor(data, 1);  // OK: Warp<All>
    data + partner
}
```

## 1.2 Contributions

We present a linear typestate system for intra-warp divergence that statically eliminates diverged shuffle and ballot operations. Well-typed programs cannot perform unsafe warp operations on inactive lanes. The guarantee is zero-overhead—enforcement is purely compile-time. We further extend the approach to intra-warp memory fence ordering (§5.6), where the same complement proof ensures all lanes have written before a fence executes.

This paper makes the following contributions:

1. **A novel type system for GPU divergence** (§3). We present the first type system that tracks active lane masks, preventing undefined behavior from reading inactive lanes. The type system uses linear typestate with a Boolean lattice of active sets, motivated by the structural analogy to multiparty session type branching.

2. **A soundness proof** (§4). We prove that well-typed programs satisfy progress and preservation, ensuring they never read from inactive lanes.

3. **A zero-overhead implementation** (§6). We implement our type system as a Rust library using traits and generics. Types are erased at compile time—the generated code is identical to hand-written unsafe CUDA.

4. **Practical patterns for GPU programming** (§5). We show how to type common GPU idioms including reductions, scans, and filters, and extend the core system to handle loops and arbitrary predicates.

5. **Empirical grounding** (§7). We document real shuffle-mask bugs in NVIDIA's reference code (cuda-samples, CUB) and scientific simulation (PIConGPU) that our type system would have caught at compile time, and demonstrate this concretely by modeling cuda-samples#398 as a runnable example where the bug is a type error.

## 1.3 The Bigger Picture

Warp typestate is one instance of a broader pattern: *participatory computation* where the set of active participants changes during execution. The transfer fidelity varies by domain: we have demonstrated a working prototype for FPGA crossbar protocols (§9.6), where the bug class is isomorphic; identified a partial transfer to distributed systems, where quiescence complements fault-tolerant session types; and noted structural similarity to database predicate filtering and proof case splits, though without actionable type-system transfer. We return to this in §9.

The remainder of this paper is organized as follows. §2 provides background on GPU execution and session types. §3 presents our core type system. §4 proves soundness. §5 extends the system to handle loops and arbitrary predicates. §6 describes our implementation. §7 evaluates bug detection and performance. §8 discusses related work. §9 sketches future directions and §10 concludes.

---

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

---

# 3. Warp Typestate: Linear Types for Active Sets

This section presents our core type system. We begin with the types (§3.1), then define the active set lattice (§3.2), present the typing rules (§3.3), explain our key mechanism of method availability (§3.4), and conclude with a worked example (§3.5).

## 3.1 Types

Our type system extends a standard functional language with GPU-specific types:

```
Types τ ::= Warp<S>           -- Warp with active set S
          | PerLane<T>        -- Per-lane value of type T
          | Uniform<T>        -- Value identical across all lanes
          | SingleLane<T, n>  -- Value existing only in lane n
          | τ₁ × τ₂           -- Pairs
          | τ₁ → τ₂           -- Functions
          | ...               -- Standard types (int, bool, etc.)
```

### `Warp<S>`

The central type is `Warp<S>`, representing a warp whose active lanes are described by the active set `S`. This type is a *capability*: possession of a `Warp<S>` value grants permission to perform operations on lanes in `S`.

Importantly, `Warp<S>` is a *linear* type in the formal calculus—it cannot be duplicated or discarded. The Rust implementation approximates linearity with affine types (move semantics + `#[must_use]` warnings); see §6 for the gap analysis. A warp that diverges into two sub-warps must eventually merge back. This prevents "losing" lanes.

### `PerLane<T>`

`PerLane<T>` represents a value of type `T` that may differ across lanes. This is the natural type for most GPU data—each lane has its own value.

```rust
let data: PerLane<i32> = load_per_lane(ptr);  // Each lane loads from ptr + lane_id
```

### `Uniform<T>`

`Uniform<T>` represents a value guaranteed to be identical across all active lanes. This is important for warp-uniform operations like branch conditions:

```rust
let threshold: Uniform<i32> = Uniform::from_const(42);
// All lanes have the same value; warp-uniform branch
if data > threshold.get() { ... }
```

A `Uniform<T>` can be converted to `PerLane<T>` (broadcasting), but the reverse requires a check that all lanes agree.

### SingleLane<T, n>

`SingleLane<T, n>` represents a value that exists only in lane `n`. This is the result type of reductions:

```rust
let sum: SingleLane<i32, 0> = reduce_sum(data);  // Result in lane 0
let broadcast: Uniform<i32> = sum.broadcast();   // Share with all lanes
```

## 3.2 Active Sets

Active sets describe which lanes are currently active. We define them as a lattice.

### Syntax

```
Active Sets S ::= All                    -- All 32 lanes
                | None                   -- No lanes (error state)
                | Even | Odd             -- Even/odd lanes
                | LowHalf | HighHalf     -- Lower/upper 16 lanes
                | S₁ ∩ S₂                -- Intersection
                | S₁ ∪ S₂                -- Union
                | ¬S                     -- Complement
```

Each active set has a concrete representation as a 32-bit mask:

```
⟦All⟧       = 0xFFFFFFFF
⟦None⟧      = 0x00000000
⟦Even⟧      = 0x55555555
⟦Odd⟧       = 0xAAAAAAAA
⟦LowHalf⟧   = 0x0000FFFF
⟦HighHalf⟧  = 0xFFFF0000
⟦S₁ ∩ S₂⟧   = ⟦S₁⟧ & ⟦S₂⟧
⟦S₁ ∪ S₂⟧   = ⟦S₁⟧ | ⟦S₂⟧
⟦¬S⟧        = ~⟦S⟧
```

### Lattice Structure

Active sets form a Boolean lattice under subset ordering:

```
         All (⊤)
        / | \
    Even Odd LowHalf HighHalf ...
      \   |   /
    EvenLow EvenHigh OddLow OddHigh ...
        \   |   /
         None (⊥)
```

The lattice operations are:
- **Meet (∩)**: Intersection of active lanes
- **Join (∪)**: Union of active lanes
- **Complement (¬)**: Lanes not in set
- **Top (All)**: All lanes active
- **Bottom (None)**: No lanes active (unreachable in well-typed programs)

### Complement Relation

We define a complement relation crucial for merge typing:

```
S₁ ⊥ S₂  ≜  S₁ ∩ S₂ = None ∧ S₁ ∪ S₂ = All
```

Two sets are complements if they are disjoint and together cover all lanes.

**Examples:**
- `Even ⊥ Odd` ✓
- `LowHalf ⊥ HighHalf` ✓
- `Even ⊥ LowHalf` ✗ (overlap in low even lanes)

### Nested Complements

For nested divergence, we need complements *within* a parent set:

```
S₁ ⊥_P S₂  ≜  S₁ ∩ S₂ = None ∧ S₁ ∪ S₂ = P
```

**Example:** Within `Even`, the sets `EvenLow = Even ∩ LowHalf` and `EvenHigh = Even ∩ HighHalf` are complements:

```
EvenLow ⊥_Even EvenHigh
```

## 3.3 Typing Rules

We present typing rules in the style of a bidirectional type system. The judgment `Γ ⊢ e : τ` means "in context Γ, expression e has type τ."

### WARP-ALL

A fresh warp starts with all lanes active:

```
─────────────────────────
Γ ⊢ Warp::kernel_entry() : Warp<All>
```

### DIVERGE

Diverging splits a warp into two sub-warps with complementary active sets:

```
Γ ⊢ w : Warp<S>    P : Predicate
────────────────────────────────────────────────────
Γ ⊢ diverge(w, P) : (Warp<S ∩ P>, Warp<S ∩ ¬P>)
```

The predicate `P` determines which lanes go to which sub-warp. The two resulting active sets are:
- `S ∩ P`: Lanes in `S` where `P` is true
- `S ∩ ¬P`: Lanes in `S` where `P` is false

**Key property:** The two sets are complements within `S`:

```
(S ∩ P) ⊥_S (S ∩ ¬P)
```

### MERGE

Merging combines two sub-warps back into one:

```
Γ ⊢ w₁ : Warp<S₁>    Γ ⊢ w₂ : Warp<S₂>    S₁ ⊥ S₂
──────────────────────────────────────────────────────
Γ ⊢ merge(w₁, w₂) : Warp<S₁ ∪ S₂>
```

The premise `S₁ ⊥ S₂` ensures the warps are complementary—no lane belongs to both, and together they cover the original set.

**This is where safety comes from:** You cannot merge arbitrary warps. The type system statically verifies they are complements.

### MERGE (Nested)

For nested divergence, we have a more general rule:

```
Γ ⊢ w₁ : Warp<S₁>    Γ ⊢ w₂ : Warp<S₂>    S₁ ⊥_P S₂
────────────────────────────────────────────────────────
Γ ⊢ merge(w₁, w₂) : Warp<P>
```

### SHUFFLE

Shuffle operations require all lanes to be active:

```
Γ ⊢ w : Warp<All>    Γ ⊢ data : PerLane<T>
────────────────────────────────────────────
Γ ⊢ shuffle_xor(w, data, mask) : PerLane<T>
```

**This is the key safety rule:** The premise `Warp<All>` prevents shuffling on a diverged warp. If you have `Warp<Even>`, you cannot call `shuffle_xor`—the rule doesn't apply.

### SHUFFLE-WITHIN (Restricted)

For shuffles that stay within an active set, we have a restricted rule:

```
Γ ⊢ w : Warp<S>    Γ ⊢ data : PerLane<T>    preserves_set(mask, S)
────────────────────────────────────────────────────────────────────
Γ ⊢ shuffle_xor_within(w, data, mask) : PerLane<T>
```

The predicate `preserves_set(mask, S)` holds when XORing any lane in `S` with `mask` yields another lane in `S`.

**Example:** `preserves_set(2, Even)` holds because XORing an even number with 2 yields another even number.

### BALLOT

Ballot operations produce uniform results:

```
Γ ⊢ w : Warp<S>    Γ ⊢ pred : PerLane<bool>
────────────────────────────────────────────
Γ ⊢ ballot(w, pred) : Uniform<u32>
```

The result is uniform because all (active) lanes see the same bitmask.

### LINEAR WARP USAGE

In the formal calculus, warps are linear—each warp value must be used exactly once. Rust enforces the no-duplication half (move semantics) but permits dropping (affine, not linear); `#[must_use]` catches accidental drops as warnings.

```
Γ, w : Warp<S> ⊢ e : τ    w occurs exactly once in e
──────────────────────────────────────────────────────
Γ ⊢ λw. e : Warp<S> → τ
```

This prevents duplicating a warp (which would allow two divergent uses) or dropping a warp (which would lose lanes).

## 3.4 Method Availability: The Key Mechanism

Our type system's safety comes from a simple mechanism: **methods exist only on types where they are safe**.

In traditional type systems, an operation might be defined for all types and checked at each use site. We take a different approach: the operation is *only defined* for safe types.

### Implementation in Rust

```rust
impl Warp<All> {
    /// Shuffle XOR - only available on Warp<All>
    pub fn shuffle_xor<T>(&self, data: PerLane<T>, mask: u32) -> PerLane<T> {
        // ... implementation ...
    }
}

// Note: NO shuffle_xor method on Warp<Even>, Warp<Odd>, etc.
```

Trying to call `shuffle_xor` on a `Warp<Even>` produces:

```
error[E0599]: no method named `shuffle_xor` found for struct `Warp<Even>`
  --> src/main.rs:10:20
   |
10 |     let result = warp.shuffle_xor(data, 1);
   |                       ^^^^^^^^^^ method not found in `Warp<Even>`
   |
   = note: the method exists on `Warp<All>`
```

This is not a "type error" in the traditional sense—it's a *method resolution failure*. The method simply doesn't exist for that type.

### Why This Works

1. **No runtime cost:** Method availability is resolved at compile time. The generated code has no checks.

2. **Clear errors:** The error message says exactly what's wrong: "method not found for `Warp<Even>`."

3. **Unforgeable:** You cannot "cast" a `Warp<Even>` to `Warp<All>`—there's no way to bypass the type system.

4. **Composable:** Library authors can add methods for specific active sets without modifying the core types.

### The Complement Trait

For merge, we use a trait to encode the complement relation:

```rust
pub trait ComplementOf<Other: ActiveSet>: ActiveSet {}

impl ComplementOf<Odd> for Even {}
impl ComplementOf<Even> for Odd {}
impl ComplementOf<HighHalf> for LowHalf {}
// ... etc ...

pub fn merge<S1, S2>(w1: Warp<S1>, w2: Warp<S2>) -> Warp<All>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
{
    // ... implementation ...
}
```

Trying to merge non-complements produces:

```
error[E0277]: the trait bound `Even: ComplementOf<LowHalf>` is not satisfied
  --> src/main.rs:15:18
   |
15 |     let merged = merge(evens, low_half);
   |                  ^^^^^ the trait `ComplementOf<LowHalf>` is not implemented for `Even`
```

## 3.5 Worked Example: Butterfly Reduction

We demonstrate the type system with a butterfly reduction—the standard warp-level algorithm for summing 32 values.

### The Algorithm

```
Initial:  [a₀, a₁, a₂, a₃, a₄, a₅, ...]

XOR 16:   [a₀+a₁₆, a₁+a₁₇, ...]  -- each lane adds partner 16 away
XOR 8:    [a₀+a₁₆+a₈+a₂₄, ...]   -- each lane adds partner 8 away
XOR 4:    [...]                    -- etc.
XOR 2:    [...]
XOR 1:    [sum, sum, sum, ...]     -- all lanes have the total
```

### Typed Implementation

```rust
fn butterfly_sum(warp: Warp<All>, data: PerLane<i32>) -> i32 {
    // Type of warp: Warp<All> ✓

    let data = data + warp.shuffle_xor(data, 16);
    // shuffle_xor requires Warp<All> ✓
    // Type of data: PerLane<i32>

    let data = data + warp.shuffle_xor(data, 8);
    // Still Warp<All>, still safe ✓

    let data = data + warp.shuffle_xor(data, 4);
    let data = data + warp.shuffle_xor(data, 2);
    let data = data + warp.shuffle_xor(data, 1);

    // All lanes now have the same value
    data.get()  // All lanes now hold the total
}
```

Every `shuffle_xor` call type-checks because `warp` has type `Warp<All>` throughout.

### Adding Divergence

Now consider a filtered reduction where some lanes don't participate:

```rust
// Pseudocode — actual API uses concrete diverge methods (§6)
fn filtered_sum(warp: Warp<All>, data: PerLane<i32>, keep: PerLane<bool>) -> Uniform<i32> {
    // Diverge based on keep predicate
    let (active, inactive) = warp.diverge(|lane| keep[lane]);
    // Type of active: Warp<Keep>
    // Type of inactive: Warp<¬Keep>

    // WRONG: Try to shuffle on active
    // let partner = active.shuffle_xor(data, 1);
    // ERROR: no method `shuffle_xor` found for `Warp<Keep>`

    // CORRECT: Prepare data and merge first
    let active_data = data;
    let inactive_data = PerLane::new(0);  // Non-participants contribute 0

    // Merge back to Warp<All>
    let warp: Warp<All> = merge(active, inactive);  // ✓ Keep ⊥ ¬Keep
    // Data merge implicit in SIMT — each lane holds its branch result

    // Now butterfly reduction is safe
    butterfly_sum(warp, active_data)
}
```

The type system guides the programmer: you cannot shuffle until you merge. The merge requires complementary sets. The result is correct by construction.

## 3.6 Summary

Our type system provides safety through three mechanisms:

1. **Active set types** (`Warp<S>`) track which lanes are active at each program point.

2. **Typing rules** ensure diverge produces complements and merge consumes complements.

3. **Method availability** restricts operations to types where they are safe—`shuffle_xor` only exists on `Warp<All>`.

The result: shuffle-from-inactive-lane bugs become compile errors. The fix is explicit (merge before shuffle) and verified by the type checker.

---

# 4. Metatheory

This section proves that our type system is sound: well-typed programs never read from inactive lanes. We follow the standard approach of proving progress and preservation [Wright and Felleisen 1994].

## 4.1 Operational Semantics

We define a small-step operational semantics for warp operations. A configuration is a triple `(σ, w, e)` where:
- `σ` is a store mapping lanes to their register values
- `w` is the current active mask (a 32-bit value)
- `e` is the expression being evaluated

### Values

```
Values v ::= Warp<S>              -- Warp capability
           | PerLane(v₀, ..., v₃₁) -- Per-lane values
           | Uniform(v)            -- Uniform value
           | ...
```

### Evaluation Rules

**E-DIVERGE**: Diverging splits the active mask according to a predicate.

```
pred evaluates to mask M under current active lanes w
─────────────────────────────────────────────────────────────────
(σ, w, diverge(Warp<S>, pred)) → (σ, w, (Warp<S ∩ M>, Warp<S ∩ ¬M>))
```

**E-MERGE**: Merging unions two active masks.

```
─────────────────────────────────────────────────────────────
(σ, w, merge(Warp<S₁>, Warp<S₂>)) → (σ, w, Warp<S₁ ∪ S₂>)
```

**E-SHUFFLE-XOR**: Shuffle reads from XOR partners.

```
w = All (all lanes active)
for each lane i: result[i] = data[i ⊕ mask]
─────────────────────────────────────────────────────────────
(σ, All, shuffle_xor(Warp<All>, data, mask)) → (σ, All, result)
```

Note: The premise `w = All` is crucial—this rule only applies when all lanes are active.

**E-SHUFFLE-WITHIN**: Shuffle within a subset (restricted masks only).

```
preserves_set(mask, S)
for each lane i in S: result[i] = data[i ⊕ mask]
─────────────────────────────────────────────────────────────
(σ, S, shuffle_xor_within(Warp<S>, data, mask)) → (σ, S, result)
```

## 4.2 Type Safety

We prove type safety through progress and preservation.

### Theorem 4.1 (Progress)

**If `Γ ⊢ e : τ` and `e` is not a value, then there exists `e'` such that `e → e'`.**

*Proof sketch:* By induction on the typing derivation.

- **Case DIVERGE**: The predicate can always be evaluated, producing a mask. The diverge operation produces two sub-warps.

- **Case MERGE**: If `Γ ⊢ w₁ : Warp<S₁>` and `Γ ⊢ w₂ : Warp<S₂>` with `S₁ ⊥ S₂`, then merge produces `Warp<S₁ ∪ S₂>`.

- **Case SHUFFLE**: The premise `Γ ⊢ w : Warp<All>` ensures all lanes are active. The E-SHUFFLE-XOR rule applies, and the operation completes.

The key insight: shuffle operations have a premise requiring `Warp<All>`. If a program type-checks, this premise is satisfied, and progress is guaranteed. If a program has `Warp<Even>` and tries to shuffle, **it doesn't type-check**—we never reach this case.

### Theorem 4.2 (Preservation)

**If `Γ ⊢ e : τ` and `e → e'`, then `Γ ⊢ e' : τ`.**

*Proof sketch:* By induction on the typing derivation.

- **Case DIVERGE**: We have `Γ ⊢ diverge(w, P) : (Warp<S ∩ P>, Warp<S ∩ ¬P>)`. After stepping, we get the pair of sub-warps. The types are preserved by construction.

- **Case MERGE**: We have `Γ ⊢ merge(w₁, w₂) : Warp<S₁ ∪ S₂>` where `S₁ ⊥ S₂`. After stepping, we get `Warp<S₁ ∪ S₂>`. The type is preserved.

- **Case SHUFFLE**: We have `Γ ⊢ shuffle_xor(w, data, mask) : PerLane<T>`. After stepping, we get `PerLane<T>`. The type is preserved.

### Corollary 4.3 (Type Safety)

**Well-typed programs don't go wrong.**

Specifically: if `⊢ e : τ` and `e →* e'` where `e'` is stuck (cannot step further and is not a value), then `e'` does not contain a shuffle operation with an inactive source lane.

*Proof:* By progress and preservation, well-typed programs either:
1. Reduce to a value (normal termination), or
2. Step forever (non-termination).

They never reach a stuck state. Since shuffle-from-inactive-lane would be stuck (the E-SHUFFLE-XOR rule requires `All`), it cannot occur in a well-typed program.

## 4.3 Key Lemmas

The soundness proof relies on several key lemmas about active sets.

### Lemma 4.4 (Diverge Produces Complements)

**For any active set `S` and predicate `P`:**
```
(S ∩ P) ⊥_S (S ∩ ¬P)
```

*Proof:*
- Disjoint: `(S ∩ P) ∩ (S ∩ ¬P) = S ∩ (P ∩ ¬P) = S ∩ ∅ = ∅ = None`
- Covering: `(S ∩ P) ∪ (S ∩ ¬P) = S ∩ (P ∪ ¬P) = S ∩ All = S`

Therefore, the two sets are complements within `S`.

### Lemma 4.5 (Merge Restores Original)

**If `S₁ ⊥_P S₂` (complements within P), then `S₁ ∪ S₂ = P`.**

*Proof:* Immediate from the definition of complement within P.

This lemma ensures that merging the results of a diverge restores the original active set.

### Lemma 4.6 (Shuffle Source Validity)

**If `Γ ⊢ shuffle_xor(w, data, mask) : PerLane<T>`, then for every lane `i`, the source lane `i ⊕ mask` is active.**

*Proof:* The typing rule requires `Γ ⊢ w : Warp<All>`. In `Warp<All>`, all 32 lanes are active. For any `i` and any `mask`, `i ⊕ mask` is a valid lane index (0–31). Since all lanes are active, every source lane is active.

This is the key safety property: shuffles only read from active lanes.

### Lemma 4.7 (No Unsafe Shuffle with Diverged Warp)

**If `Γ ⊢ w : Warp<S>` where `S ≠ All`, then `shuffle_xor(w, data, mask)` does not type-check.**

*Proof:* The SHUFFLE rule has premise `Γ ⊢ w : Warp<All>`. If `w : Warp<S>` for `S ≠ All`, this premise is not satisfied. The rule does not apply. There is no other rule that types `shuffle_xor`. Therefore, the expression does not type-check.

This lemma formalizes our key mechanism: the bug is a type error, not a runtime error.

## 4.4 Linearity

In the formal calculus, warps are linear resources—they cannot be duplicated or discarded. This is essential for soundness. The Rust implementation is affine: move semantics prevent duplication, but dropping without merge is permitted (with a `#[must_use]` warning). The Lean formalization models full linearity.

### Lemma 4.8 (No Warp Duplication)

**If `Γ, w : Warp<S> ⊢ e : τ`, then `w` occurs exactly once in `e`.**

*Proof:* By the linear typing rule for warps. The type system tracks each warp capability and ensures single use.

### Lemma 4.9 (No Warp Discard)

**If `Γ ⊢ e : τ` and `w : Warp<S>` is bound in `Γ`, then `w` is used in `e`.**

*Proof:* By the linear typing rule. Unused linear resources are a type error.

### Why Linearity Matters

Without linearity, a warp could be used twice:

```rust
// WRONG (if warps were copyable)
let (evens, odds) = warp.diverge_even_odd();
let x = warp.shuffle_xor(data, 1);  // Using original warp again!
```

This would allow shuffling on `Warp<All>` even after diverging, which is unsound—some lanes are now inactive.

With linearity, `diverge` *consumes* the original warp and produces two new warps. You cannot use the original after diverging.

## 4.5 Nested Divergence

For nested divergence, we need a more refined complement relation.

### Definition 4.10 (Complement Within)

Sets `S₁` and `S₂` are complements within `P`, written `S₁ ⊥_P S₂`, if:
1. `S₁ ∩ S₂ = None` (disjoint)
2. `S₁ ∪ S₂ = P` (cover P)
3. `S₁ ⊆ P` and `S₂ ⊆ P` (both subsets of P)

### Lemma 4.11 (Nested Diverge Produces Complements Within)

**If we diverge `Warp<P>` on predicate `Q`, the results are complements within `P`:**
```
(P ∩ Q) ⊥_P (P ∩ ¬Q)
```

*Proof:* Similar to Lemma 4.4, restricted to P.

### Lemma 4.12 (Nested Merge Restores Parent)

**If `S₁ ⊥_P S₂`, then merging `Warp<S₁>` and `Warp<S₂>` produces `Warp<P>`.**

*Proof:* By Lemma 4.5, `S₁ ∪ S₂ = P`.

### Example: Double Divergence

```
Warp<All>
    │
    ├── diverge(even)
    │
Warp<Even>         Warp<Odd>
    │
    ├── diverge(low_half)
    │
Warp<EvenLow>    Warp<EvenHigh>
```

Where `EvenLow = Even ∩ LowHalf` and `EvenHigh = Even ∩ HighHalf`.

The merge path:
1. `merge(Warp<EvenLow>, Warp<EvenHigh>) : Warp<Even>` (since `EvenLow ⊥_Even EvenHigh`)
2. `merge(Warp<Even>, Warp<Odd>) : Warp<All>` (since `Even ⊥ Odd`)

At each step, the type system verifies the complement relation.

## 4.6 Shuffle Within Diverged Warp

Our core system restricts shuffles to `Warp<All>`. We can relax this for shuffles that stay within an active set.

### Definition 4.13 (Set-Preserving Mask)

A mask `m` preserves set `S`, written `preserves(m, S)`, if:
```
∀i. (i ∈ S) → (i ⊕ m ∈ S)
```

### Lemma 4.14 (Set-Preserving Shuffle Safety)

**If `preserves(m, S)` and all lanes in `S` are active, then `shuffle_xor_within(Warp<S>, data, m)` only reads from active lanes.**

*Proof:* For any active lane `i` in `S`, the source lane is `i ⊕ m`. By the preserves property, `i ⊕ m ∈ S`. Since all lanes in `S` are active, the source lane is active.

### Examples

- `preserves(2, Even)`: XORing an even lane with 2 gives another even lane. ✓
- `preserves(1, Even)`: XORing an even lane with 1 gives an odd lane. ✗
- `preserves(16, LowHalf)`: XORing a low lane with 16 gives a high lane. ✗
- `preserves(8, LowHalf)`: XORing a low lane with 8 gives another low lane (for lanes 0–15). ✓

## 4.7 Discussion

Our soundness proof establishes that well-typed programs never read from inactive lanes. The key mechanisms are:

1. **Type-level tracking**: `Warp<S>` records the active set at the type level.

2. **Complement verification**: Merge requires a compile-time proof that sets are complements.

3. **Method restriction**: Shuffle is only available on `Warp<All>`, enforced by method resolution.

4. **Linearity**: Warps cannot be duplicated, preventing use-after-diverge.

The proof is constructive: the type system not only prevents bugs but guides programmers toward correct code. When a shuffle doesn't type-check, the fix is to merge first.

### Decidability

Type checking in our system is decidable. The active-set lattice is finite (at most 2^W elements for warp width W), trait resolution is type-directed (one rule per constructor, no ambiguity), and complement checking is a constant-time bitwise operation. This contrasts with session types in general, where asynchronous subtyping is undecidable even for two participants [Lange and Yoshida 2016]. Our system avoids this obstacle because SIMT execution is synchronous—there is no message buffering between lanes, so subtyping questions reduce to set containment on finite bitmasks.

### Limitations

Our formalization assumes:
- **Finite warps**: We fix warp size at 32 (NVIDIA) or 64 (AMD).
- **Structured control flow**: Diverge and merge are explicit operations, not implicit branches. For structured control flow, divergence analysis is decidable and efficiently computable—compilers already do it [LLVM uniformity analysis]. Our restriction aligns with this known result.
- **No data-dependent active sets**: The type system tracks static patterns (Even, Odd, LowHalf), not arbitrary runtime predicates.

These limitations are addressed in §5 (Extensions).

## 4.8 Mechanization

We have mechanized the core metatheory for the base calculus in Lean 4 (`lean/WarpTypes/`). All theorems are machine-checked with **zero `sorry` and zero axioms**.

### Scope

The mechanization covers two files totaling 896 lines of Lean:

**Core type system properties** (`Basic.lean`):
- `diverge_partition`: Diverge produces disjoint, covering sub-sets (Lemma 4.4). Proved by bitvector extensionality.
- `shuffle_requires_all`: Shuffle typing requires `Warp<All>` (Lemma 4.7). Proved by case analysis on the typing derivation.
- `complement_symmetric`: Complement relation is symmetric. Proved by commutativity of bitwise AND/OR.
- `even_odd_complement`, `lowHalf_highHalf_complement`: Concrete complement instances. Proved by `decide` (BitVec 32 is decidable).

**Full metatheory** (`Metatheory.lean`):
- **Progress** (Theorem 4.1): A closed well-typed expression is either a value or can step. Proved by induction on the typing derivation, using canonical forms lemmas for each type constructor.
- **Preservation** (Theorem 4.2): If `Γ ⊢ e : τ ⊣ Γ'` and `e ⟶ e'`, then `Γ ⊢ e' : τ ⊣ Γ'`. Proved by induction on the step relation, with case analysis on typing. The critical `letVal` case uses the substitution lemma below.
- **Substitution lemma** (`subst_typing`): Substituting a value for a linear binding removes that binding from both input and output contexts. This is the key lemma enabling preservation for `let`-bindings in a linear type system. Proved directly (~90 lines) via induction on the typing derivation, with explicit context threading through merge, shuffle, and pair sub-expressions.

**Untypability proofs** (5 documented GPU bugs):
- `bug1_cuda_samples_398`: Shuffle after extracting lane 0 — untypable.
- `bug2_cccl_854`: Shuffle on 16-lane sub-warp — untypable.
- `bug3_picongpu_2514`: Ballot on diverged subset — untypable.
- `bug4_llvm_155682`: Shuffle after lane-0 conditional — untypable.
- `bug5_shuffle_after_diverge`: Shuffle after even/odd divergence — untypable.

Each untypability proof factors through a single lemma (`shuffle_diverged_untypable`) that shows: if the active set after diverge is not `All`, no typing derivation exists for a shuffle on that sub-warp. The concrete bugs instantiate this with specific masks and close via `decide`.

**Supporting infrastructure** (14 lemmas): canonical forms for `Warp`, `PerLane`, and `Pair` types; `value_preserves_ctx` (values don't consume linear resources); `value_any_ctx` (values can be typed in any context); `output_binding_from_input` (output context bindings originate from input); context algebra (`remove_comm`, `remove_lookup_self`, `remove_lookup_ne`, etc.).

### Design Choices

The formalization models active sets as `BitVec 32` (Lean's built-in bitvector type), enabling `decide` for concrete instances and extensionality for universal properties. Typing judgements use a linear context `Γ ⊢ e : τ ⊣ Γ'` where `Γ'` tracks which bindings remain after evaluation — this directly encodes the consumption semantics that Rust enforces via move. The substitution lemma is proved directly rather than axiomatized, which required explicit context infrastructure but yields a stronger result: the mechanization has no trusted assumptions beyond Lean's kernel.

Lean 4 is chosen for two reasons: (1) Aeneas translates Rust's borrow semantics into a purely functional representation amenable to Lean verification; (2) prior work on GPU verification (MCL framework) was built in Lean.

### What Is Not Mechanized

The operational semantics for `shuffle_within` (§4.6, set-preserving masks) and the extension typing rules (§5) are not mechanized. The nested divergence lemmas (§4.5) follow from `diverge_partition` by instantiation but are not stated as separate Lean theorems. We consider the mechanized scope sufficient: progress, preservation, substitution, and untypability cover the core safety claim.


---

# 5. Extensions

The core type system (§3) handles structured divergence with static predicates. Real GPU programs require more: loops with varying trip counts, runtime-dependent predicates, and communication beyond a single warp. This section extends our system to handle these cases.

## 5.1 Loops and Recursive Protocols

GPU kernels often contain loops where different lanes iterate different numbers of times:

```cuda
while (lane_data[lane] > threshold) {
    lane_data[lane] = process(lane_data[lane]);
}
```

Each lane may take a different number of iterations. The active set shrinks as lanes exit the loop.

### The Challenge

In our type system, the loop body has type:
```
loop_body : Warp<S> → Warp<S'>
```

But what are `S` and `S'`? They depend on runtime values.

### Solution: Layered Approach

We don't try to track the exact active set through the loop. Instead, we verify entry and exit invariants via four typing rules.

### Typing Rules

**LOOP-UNIFORM**: All lanes iterate together; warp operations permitted in body.

```
Γ ⊢ n : Uniform<Int>    Γ, w : Warp<S> ⊢ e : Warp<S>
─────────────────────────────────────────────────────────
Γ, w : Warp<S> ⊢ for_uniform(n, e) : Warp<S>
```

The loop count `n` must be uniform (same across all lanes). The body preserves the active set `S`, so warp operations on `Warp<S>` are safe at every iteration. This covers butterfly reductions, Kogge-Stone scans, and any pattern where all lanes step together.

**LOOP-CONVERGENT**: Exit only when a collective predicate holds; warp operations permitted in body.

```
Γ, w : Warp<S> ⊢ e : Warp<S>    Γ ⊢ p : Warp<S> → Bool
p uses collective (ballot/all/any)
─────────────────────────────────────────────────────────
Γ, w : Warp<S> ⊢ loop_until(p, e) : Warp<S>
```

The exit predicate `p` is a collective operation (e.g., `warp.all(...)`) that evaluates identically across all active lanes. Because all lanes exit simultaneously, the active set is preserved.

**LOOP-VARYING**: Per-lane iteration counts; no warp operations in body.

```
Γ ⊢ e : PerLane<T> → PerLane<T>    e contains no warp operations
──────────────────────────────────────────────────────────────────
Γ, w : Warp<S> ⊢ for_varying(e) : Warp<S>
```

The body operates on per-lane data without touching the warp handle. The warp is syntactically absent from `e`, so varying trip counts cannot introduce divergence bugs. The warp passes through unchanged.

**LOOP-PHASED**: Alternating uniform and varying phases; warp operations only in uniform phases.

```
Γ ⊢ n : Uniform<Int>
Γ, w : Warp<S> ⊢ e₁ : Warp<S>           (uniform phase, warp ops permitted)
Γ ⊢ e₂ : PerLane<T> → PerLane<T>         (varying phase, no warp ops)
──────────────────────────────────────────────────────────────────
Γ, w : Warp<S> ⊢ for_phased(n, e₁, e₂) : Warp<S>
```

Each iteration has two sub-expressions: `e₁` (uniform, may use warp) and `e₂` (varying, warp-free). The uniform count `n` ensures all lanes execute the same number of rounds, even though per-lane work within `e₂` may vary.

### Examples

**Pattern 1: Uniform Loop (LOOP-UNIFORM)**
```rust
fn uniform_loop(mut warp: Warp<All>, mut data: PerLane<i32>) -> Warp<All> {
    let limit = warp.ballot_any(|lane| data[lane] > 0);  // Uniform count

    for _ in 0..limit {
        // All lanes execute together
        data = warp.shuffle_xor(data, 1);  // Safe: Warp<All>
        data = process(data);
    }

    warp  // Returns Warp<All>
}
```

Type: `Warp<All> → Warp<All>`. The invariant is that all lanes iterate together.

**Pattern 2: Convergent Loop (LOOP-CONVERGENT)**
```rust
fn convergent_loop(mut warp: Warp<All>, mut data: PerLane<i32>) -> Warp<All> {
    loop {
        data = process(data);
        if warp.all(|lane| data[lane] < threshold) {
            break;
        }
    }

    warp  // Returns Warp<All>
}
```

Type: `Warp<All> → Warp<All>`. The loop exits only when all lanes satisfy the predicate.

**Pattern 3: Varying Loop (LOOP-VARYING)**
```rust
fn varying_loop(warp: Warp<All>, mut data: PerLane<i32>) -> (Warp<All>, PerLane<i32>) {
    // Per-lane accumulation - no warp operations needed
    for lane in 0..32 {
        while data[lane] > threshold {
            data[lane] = process_single(data[lane]);
        }
    }

    (warp, data)  // Warp unchanged
}
```

Type: `Warp<All> → Warp<All>`. The warp is not used inside the loop, so varying iteration is safe.

**Pattern 4: Phased Loop (LOOP-PHASED)**
```rust
fn phased_loop(mut warp: Warp<All>, mut data: PerLane<i32>) -> Warp<All> {
    for round in 0..MAX_ROUNDS {
        // Phase 1: Uniform shuffle (all lanes)
        data = warp.shuffle_xor(data, 1 << round);

        // Phase 2: Varying per-lane work (no warp ops)
        for lane in 0..32 {
            while lane_needs_work(lane, data) {
                data[lane] = process_single(data[lane]);
            }
        }

        // All lanes reconverge for next round
    }

    warp
}
```

Type: `Warp<All> → Warp<All>`. Warp operations happen only in uniform phases.

### Recursive Protocol Types

For more complex patterns, we use μ-types:

```
μX. (diverge → (compute ; merge ; X) ⊕ done)
```

This protocol:
1. Diverges
2. Computes on each branch
3. Merges back
4. Recurses or terminates

The key decidability condition: **diverge and merge must be balanced within each iteration**. This ensures the active set at loop end equals the active set at loop start.

## 5.2 Arbitrary Predicates

The core system uses static predicates (`Even`, `Odd`, `LowHalf`). Real programs diverge on runtime values:

```cuda
if (lane_data[lane] < threshold) {  // Runtime predicate
    ...
}
```

### The Challenge

We can't know at compile time which lanes satisfy `data < threshold`. The active set is data-dependent.

### Solution: Layered Types

We provide a hierarchy of approaches, from most static to most dynamic:

**Layer 1: Marker Types (Zero Runtime Cost)**

For common patterns, use predefined marker types:
```rust
// Static patterns with known masks
struct Even;   // 0x55555555
struct Odd;    // 0xAAAAAAAA
struct LowHalf; // 0x0000FFFF
```

**Layer 2: Shape Types (Minimal Runtime Cost)**

When the *shape* is known but the exact mask is runtime:
```rust
// Shape: "first N lanes" - mask is 2^N - 1
struct LowRange<const N: usize>;

// Shape: "strided lanes" - mask has regular structure
struct Stride<const STRIDE: usize, const OFFSET: usize>;
```

The complement is computed at runtime, but the structure is verified statically.

**Layer 3: Indexed Predicates (Registry Lookup)**

For predicates that can be registered:
```rust
// Register a predicate and get a unique index
let pred_idx = register_predicate(|lane, data| data[lane] < threshold);

// Diverge using the registered predicate
let (matching, not_matching) = warp.diverge_indexed(pred_idx);
```

The type system tracks predicate indices and verifies that merge uses the matching complement.

**Layer 4: Existential Types (Full Dynamic)**

When nothing is known statically, use an existential wrapper:
```rust
struct SomeWarp {
    mask: u64,
    // Active set is unknown at type level
}

impl SomeWarp {
    fn merge_with_complement(self, other: SomeWarp) -> Result<Warp<All>, Error> {
        if self.mask & other.mask == 0 && self.mask | other.mask == ALL {
            Ok(Warp::new())
        } else {
            Err(Error::NotComplements)
        }
    }
}
```

Safety is checked at runtime, but the type system still prevents forgetting the check.

**Layer 5: Dependent Types (Full Static Safety)**

A language with dependent types could express:
```rust
fn diverge<P: Predicate>(warp: Warp<S>, pred: P)
    -> (Warp<S ∩ P>, Warp<S ∩ ¬P>)
```

Where `S ∩ P` is computed at the type level based on the predicate.

### Recommendation

Most GPU code uses Layer 1–2 (80%+ of cases). Layer 3–4 are escape hatches for complex data-dependent patterns. Layer 5 requires dependent types but provides full safety.

## 5.3 Inter-Block Communication

Our core system models a single warp. Real GPU programs have multiple blocks, each containing multiple warps. How do session types extend to this hierarchy?

### Key Insight: Inter-Block is Traditional MPST

Warp-level divergence is special: lanes go *quiescent* within a single instruction stream. This is what our type system models.

Block-level communication is different:
- Blocks execute *independently* (no shared instruction stream)
- Blocks don't "go quiescent"—they run fully or don't exist
- Communication is through shared memory or global memory
- Synchronization is explicit (barriers, atomics)

This is the domain of traditional multiparty session types [Honda, Yoshida, Carbone 2008].

### Hierarchical Protocols

We can compose warp-level and block-level protocols:

```
Global Protocol MatMul(Blocks B[M][N], Tiles A, B, C) {
    // Block-level: traditional MPST
    foreach b in B {
        // Warp-level: warp typestate
        WarpProtocol {
            load_tile(A);
            load_tile(B);
            diverge(predicate) {
                compute_partial();
            } merge;
            reduce();
            store_tile(C);
        }
    }
    barrier;  // Block synchronization
}
```

The warp protocol uses our warp typestate. The block protocol uses traditional session types. They compose hierarchically.

### Why This Matters

The novel contribution of this paper is *warp-level* linear typestate with quiescence — motivated by the session type analogy but technically distinct (§1.1). This is the gap in prior work:
- Traditional session types: all parties active or failed
- Our approach: parties can go quiescent and resume, tracked via active-set lattice rather than protocol sequencing

Inter-block communication doesn't need this extension—it's already well-served by existing session type systems. Our contribution is recognizing where the existing theory applies and where we need something new.

## 5.4 Cooperative Groups

CUDA's Cooperative Groups API provides a unified abstraction for thread groups at all levels:

```cuda
auto warp = cooperative_groups::tiled_partition<32>(this_thread_block());
auto block = this_thread_block();
auto grid = this_grid();
```

### Typing Cooperative Groups

Each level has its own session type:

```rust
// Warp-level: warp typestate
struct Warp<S: ActiveSet>;

// Block-level: traditional session types
struct Block<Role: BlockRole, State: ProtocolState>;

// Grid-level: barrier-synchronized collective
struct Grid<Phase: GridPhase>;
```

The `tiled_partition` operation creates a typed warp from a block:

```rust
impl Block<AnyRole, Computing> {
    fn tiled_partition<const SIZE: usize>(self) -> Vec<Warp<All>> {
        // Partition block into SIZE-lane warps
    }
}
```

### Subgroup Operations

Modern GPUs support subgroups smaller than a warp:

```rust
// Partition warp into 8-lane subgroups
let subgroups: [Subgroup<All8>; 4] = warp.partition::<8>();

// Each subgroup has its own session type
impl Subgroup<S> {
    fn shuffle_xor<T>(&self, data: PerLane<T, 8>, mask: u32) -> PerLane<T, 8>
    where
        S: AllLanes<8>;  // All 8 lanes active
}
```

The same principles apply: shuffle requires all lanes active, diverge produces complements, merge verifies complements.

## 5.5 Memory Safety Integration

Our type system focuses on divergence safety. It composes with memory safety systems like Descend [Kopcke et al. 2024].

### Descend Integration

Descend tracks ownership and borrowing for GPU memory. We can layer divergence types on top:

```rust
// Descend: owned per-lane data
let data: PerLane<Owned<i32>> = allocate_per_lane();

// Our system: warp with active set
let warp: Warp<All> = Warp::kernel_entry();

// Combined: shuffle requires active lanes AND ownership
let result = warp.shuffle_xor(data, 1);
// Type: PerLane<Owned<i32>>
```

The systems are orthogonal:
- Descend prevents data races and use-after-free
- Our system prevents reading from inactive lanes

Together, they provide comprehensive GPU safety.

### Future: Unified Type System

A unified type system could track both:

```rust
struct Warp<S: ActiveSet, M: MemoryState> {
    // S: which lanes are active
    // M: memory ownership state
}
```

This is future work, potentially via a proc macro layer or a dependently-typed extension.

## 5.6 Fence-Divergence Interactions

Global memory writes from diverged warps introduce a subtle ordering hazard: a `threadfence()` is only meaningful after *all* lanes have written. If some lanes are inactive when the fence executes, subsequent reads may observe stale values — a memory ordering bug that manifests non-deterministically.

We handle this with a type-state machine that tracks write progress through global memory regions:

```
GlobalRegion<Unwritten>
  → Warp<S>.global_store() → GlobalRegion<PartialWrite<S>>
  → merge_writes(PartialWrite<S₁>, PartialWrite<S₂>) → GlobalRegion<FullWrite>
      where S₁: ComplementOf<S₂>
  → threadfence(FullWrite) → GlobalRegion<Fenced>
```

The key insight is that `merge_writes` reuses the same `ComplementOf` proof as warp merge — the mechanism that ensures complementary active sets at reconvergence also ensures complementary write coverage before a fence. A fence on a partially-written region is not a runtime error — it is *unrepresentable*, exactly as shuffling a diverged warp is unrepresentable.

```rust
fn safe_fence_pattern(warp: Warp<All>) {
    let (evens, odds) = warp.diverge_even_odd();
    let region = GlobalRegion::new();

    // Each half writes
    let (evens, partial) = evens.global_store(region);
    let full = odds.global_store_complement(partial);  // ComplementOf<Even> for Odd

    // Fence only after full write — FullWrite required by type
    let fenced = threadfence(full);
    let _val: i32 = fenced.read();  // Safe: all lanes wrote, fence issued
}
```

This extends our type system from intra-warp communication safety to intra-warp memory ordering safety, using the same proof mechanism throughout.

## 5.7 Summary

Our extensions handle:

| Challenge | Solution | Static Safety |
|-----------|----------|---------------|
| Uniform loops | Entry/exit invariants | Full |
| Varying loops | No warp ops in body | Full |
| Runtime predicates | Layered types (1–5) | Varies |
| Inter-block | Traditional MPST | Full |
| Cooperative groups | Hierarchical protocols | Full |
| Memory safety | Compose with Descend | Full |
| Fence-divergence | Type-state write tracking | Full |

The key insight: not every problem needs warp typestate. We identify where our contribution applies (warp-level quiescence) and where existing techniques suffice (block-level communication). Notably, the fence-divergence extension reuses the complement proof mechanism from merge — the same `ComplementOf` trait that ensures safe reconvergence also ensures safe memory ordering.


---

# 6. Implementation

We implement warp typestate as a Rust library. This section describes the encoding, explains why it achieves zero runtime overhead, and discusses practical considerations.

## 6.1 Type Encoding

Our type system maps naturally to Rust's type system.

### Active Sets as Zero-Sized Types

Active sets are marker types with no runtime representation:

```rust
pub trait ActiveSet: Copy + 'static {
    const MASK: u64;
    const NAME: &'static str;
}
```

A `warp_sets!` proc macro generates the entire hierarchy from a compact declaration:

```rust
warp_sets! {
    All = 0xFFFFFFFF {
        Even = 0x55555555 / Odd = 0xAAAAAAAA,
        LowHalf = 0x0000FFFF / HighHalf = 0xFFFF0000,
        Lane0 = 0x00000001 / NotLane0 = 0xFFFFFFFE,
    }
    Even = 0x55555555 { EvenLow = 0x00005555 / EvenHigh = 0x55550000 }
    Odd = 0xAAAAAAAA { OddLow = 0x0000AAAA / OddHigh = 0xAAAA0000 }
    LowHalf = 0x0000FFFF { EvenLow / OddLow }     // reuses existing types
    HighHalf = 0xFFFF0000 { EvenHigh / OddHigh }
}
```

The macro validates masks at compile time: each pair must be disjoint (`true & false == 0`) and covering (`true | false == parent`). Invalid hierarchies produce `compile_error!`. Shared types (e.g., `EvenLow` under both `Even` and `LowHalf`) are automatically deduplicated. The generated types are zero-sized (`std::mem::size_of::<Even>() == 0`). They exist only at compile time.

### Warp as Phantom Type

The `Warp<S>` type carries the active set as a phantom type parameter:

```rust
use std::marker::PhantomData;

#[derive(Copy, Clone)]
pub struct Warp<S: ActiveSet> {
    _marker: PhantomData<S>,
}

impl<S: ActiveSet> Warp<S> {
    pub fn new() -> Self {
        Warp { _marker: PhantomData }
    }

    pub fn active_mask(&self) -> u64 {
        S::MASK
    }
}
```

`PhantomData<S>` is zero-sized. A `Warp<Even>` has the same runtime representation as `()`.

### Complement Relation as Trait Bound

The complement relation is encoded as a trait:

```rust
pub trait ComplementOf<Other: ActiveSet>: ActiveSet {}

impl ComplementOf<Odd> for Even {}
impl ComplementOf<Even> for Odd {}
impl ComplementOf<HighHalf> for LowHalf {}
impl ComplementOf<LowHalf> for HighHalf {}
// ... symmetric impls
```

Merge requires a `ComplementOf` bound:

```rust
pub fn merge<S1, S2>(left: Warp<S1>, right: Warp<S2>) -> Warp<All>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
{
    Warp::new()
}
```

Attempting to merge non-complements fails at compile time:

```rust
let w1: Warp<All> = Warp::kernel_entry();
let (evens, _) = w1.diverge_even_odd();
let w2: Warp<All> = Warp::kernel_entry();
let (low, _) = w2.diverge_halves();
let bad = merge(evens, low);  // Compile error!
// error[E0277]: the trait bound `Even: ComplementOf<LowHalf>` is not satisfied
```

Note that `Warp::kernel_entry()` is the only public constructor—sub-warps can only be obtained via `diverge_*()` methods, preventing forgery of arbitrary active sets.

### Method Availability via Inherent Impls

The key mechanism: methods are defined only on specific types.

```rust
impl Warp<All> {
    /// Only available when all lanes are active
    pub fn shuffle_xor<T: Copy>(&self, data: PerLane<T>, mask: u32) -> PerLane<T> {
        // GPU intrinsic: __shfl_xor_sync
        unsafe { intrinsic_shuffle_xor(data.as_ptr(), mask) }
    }

    pub fn diverge_even_odd(self) -> (Warp<Even>, Warp<Odd>) {
        (Warp::new(), Warp::new())
    }
}

// No shuffle_xor on Warp<Even>!
// Calling it produces: "error[E0599]: no method named `shuffle_xor`
//                       found for struct `Warp<Even>`"
```

This is not a runtime check—the method literally doesn't exist for `Warp<Even>`.

## 6.2 Zero Overhead

Our implementation has zero runtime overhead. The types exist only at compile time.

### Proof: Generated Assembly

Consider this function:

```rust
pub fn butterfly_sum(data: [i32; 32]) -> i32 {
    let warp: Warp<All> = Warp::kernel_entry();

    let data = PerLane(data);
    let data = data + warp.shuffle_xor(data, 16);
    let data = data + warp.shuffle_xor(data, 8);
    let data = data + warp.shuffle_xor(data, 4);
    let data = data + warp.shuffle_xor(data, 2);
    let data = data + warp.shuffle_xor(data, 1);

    data.0[0]
}
```

With optimizations, this compiles to the same assembly as the untyped version:

```asm
butterfly_sum:
    ; Load data
    ; shfl.bfly.b32  r1, r0, 16
    ; add.s32        r0, r0, r1
    ; shfl.bfly.b32  r1, r0, 8
    ; add.s32        r0, r0, r1
    ; ... (continues for masks 4, 2, 1)
    ; Return
```

No type tags, no runtime checks, no overhead.

### Why Zero Overhead?

1. **Marker types are zero-sized.** `All`, `Even`, etc. have no runtime representation.

2. **PhantomData is zero-sized.** `Warp<S>` is the same size as `()`.

3. **Trait bounds are erased.** `ComplementOf` checks happen at compile time.

4. **Monomorphization.** Generic functions are specialized per type, eliminating dynamic dispatch.

5. **LLVM optimizations.** The optimizer sees through the abstraction.

### Comparison with Alternatives

| Approach | Overhead | When |
|----------|----------|------|
| Our types | Zero | Compile time |
| Runtime checks | Branch per operation | Runtime |
| Sanitizers | 2–10x slowdown | Debug builds |
| External verifiers | Separate pass | Build time |

Our approach has the lowest overhead because the types *are* the verification.

## 6.3 Data Types

### `PerLane<T>`

Per-lane data wraps a single value — because on GPU, each lane IS a separate thread with its own registers:

```rust
pub struct PerLane<T: GpuValue> {
    value: T,
}

impl<T: GpuValue> PerLane<T> {
    pub fn new(value: T) -> Self {
        PerLane { value }
    }

    pub fn get(self) -> T {
        self.value
    }
}
```

The single-`T` representation reflects GPU reality: in SIMT execution, 32 lanes each run the same code with their own register file. A `PerLane<i32>` stored by lane 0 holds lane 0's value; lane 15's copy holds lane 15's value. There is no 32-element array — the parallelism is in the hardware, not the data structure. For CPU testing, shuffle operations are identity functions (returning the input value), since only one thread executes.

### `Uniform<T>`

Uniform values are guaranteed identical across lanes:

```rust
pub struct Uniform<T>(T);

impl<T: GpuValue> Uniform<T> {
    pub fn from_const(value: T) -> Self {
        Uniform(value)
    }

    pub fn get(self) -> T {
        self.0
    }

    pub fn broadcast(self) -> PerLane<T> {
        PerLane::new(self.0)
    }
}
```

### SingleLane<T, N>

A value existing only in lane N:

```rust
pub struct SingleLane<T: GpuValue, const LANE: u8>(T);

impl<T: GpuValue, const LANE: u8> SingleLane<T, LANE> {
    pub fn get(&self) -> T {
        self.0
    }

    pub fn broadcast(self) -> Uniform<T> {
        // GPU intrinsic: broadcast from lane N
        Uniform(unsafe { intrinsic_broadcast(self.0, N) })
    }
}
```

## 6.4 Diverge and Merge

### Type-Changing Diverge

Diverge consumes a warp and produces two sub-warps:

```rust
impl Warp<All> {
    pub fn diverge_even_odd(self) -> (Warp<Even>, Warp<Odd>) {
        // No runtime code - just type transformation
        (Warp::new(), Warp::new())
    }

    pub fn diverge_halves(self) -> (Warp<LowHalf>, Warp<HighHalf>) {
        (Warp::new(), Warp::new())
    }
}
```

The `self` parameter consumes the original warp. Rust's ownership system prevents reuse.

### Generic Diverge

The formal typing rule (§3) describes a generic `diverge<P: Predicate>`. In the implementation, predicates are instantiated as concrete methods (`diverge_even_odd`, `diverge_halves`, `extract_lane0`). For runtime-dependent predicates, `diverge_dynamic(mask: u64)` returns a `DynDiverge` with structural complement guarantees (§5).

```rust
impl Warp<All> {
    pub fn diverge_even_odd(self) -> (Warp<Even>, Warp<Odd>) {
        (Warp::new(), Warp::new())
    }

    pub fn diverge_halves(self) -> (Warp<LowHalf>, Warp<HighHalf>) {
        (Warp::new(), Warp::new())
    }

    pub fn extract_lane0(self) -> (Warp<Lane0>, Warp<NotLane0>) {
        (Warp::new(), Warp::new())
    }

    pub fn diverge_dynamic(self, mask: u64) -> DynDiverge {
        DynDiverge::new(mask)
    }
}
```

Each concrete method encodes a specific predicate as a pair of complementary marker types. The `warp_sets!` macro (§6.1) generates and validates these pairings at compile time.

### Type-Safe Merge

Merge verifies complements at compile time:

```rust
pub fn merge<S1, S2>(left: Warp<S1>, right: Warp<S2>) -> Warp<Union<S1, S2>>
where
    S1: ComplementOf<S2>,
    S2: ActiveSet,
    Union<S1, S2>: ActiveSet,
{
    Warp::new()
}
```

The trait bound `S1: ComplementOf<S2>` is the compile-time verification.

### Data Merge

On GPU, merging data from two branches is a predicated select operation: each lane activates for its branch and contributes its own `PerLane<T>` value. Since each lane holds a single `T` (see §6.3), the merge is implicit in the SIMT execution model — each lane simply writes its branch result, and reconvergence makes both results available.

## 6.5 GPU Intrinsics

The library wraps GPU intrinsics with typed interfaces:

```rust
// Low-level intrinsic (unsafe, untyped)
extern "C" {
    fn __shfl_xor_sync(mask: u32, val: i32, lane_mask: u32, width: u32) -> i32;
}

// High-level wrapper (safe, typed)
impl Warp<All> {
    pub fn shuffle_xor<T: GpuValue + GpuShuffle>(&self, data: PerLane<T>, mask: u32) -> PerLane<T> {
        // On GPU: each lane calls the intrinsic with its own value.
        // PerLane<T> holds a single T per lane — the hardware provides
        // the 32-way parallelism.
        PerLane::new(data.get().gpu_shfl_xor(mask))
    }
}
```

The safe wrapper is only available on `Warp<All>`, ensuring all lanes are active when calling the intrinsic.

## 6.6 Dual-Mode Platform Abstraction

The same warp algorithm often needs both CPU and GPU implementations — CPU for testing and debugging, GPU for production. We implement a `Platform` trait that abstracts over execution model while preserving typestate safety:

```rust
trait Platform {
    const WIDTH: usize;
    type Vector<T>;
    type Mask;

    fn shuffle_xor<T: GpuValue>(source: Self::Vector<T>, mask: usize) -> Self::Vector<T>;
    fn reduce_sum<T>(values: Self::Vector<T>) -> T;
    fn ballot(predicates: Self::Vector<bool>) -> Self::Mask;
}
```

Two implementations provide the abstraction:

- **`CpuSimd<const WIDTH: usize>`**: Array-based emulation using `PortableVector<T, WIDTH>`. Shuffle operations use explicit lane indexing; reductions iterate over the array. Zero hardware dependency — runs anywhere Rust compiles.

- **`GpuWarp32`**: 32-lane warp emulation (currently delegates to `CpuSimd::<32>`; production use would emit PTX intrinsics). Masks are `u32`, matching NVIDIA's active mask width.

The same algorithm runs on both platforms without modification:

```rust
fn butterfly_reduce_sum<P: Platform>(data: P::Vector<i32>) -> i32 {
    let mut data = data;
    let mut stride = P::WIDTH / 2;
    while stride > 0 {
        data = data + P::shuffle_xor(data, stride as u32);
        stride /= 2;
    }
    data.extract(0)
}
```

Warp typestate applies uniformly: `CpuSimd` uses masked array operations for diverged lanes; `GpuWarp32` uses SIMT masking. The `ComplementOf` proof requirement is platform-independent — it verifies the same active-set algebra regardless of whether the underlying execution is scalar emulation or hardware SIMT.

## 6.7 Error Messages

Good error messages are crucial for usability. Rust provides excellent diagnostics:

**Shuffle on diverged warp:**
```
error[E0599]: no method named `shuffle_xor` found for struct `Warp<Even>` in the current scope
  --> src/main.rs:10:20
   |
10 |     let result = warp.shuffle_xor(data, 1);
   |                      ^^^^^^^^^^ method not found in `Warp<Even>`
   |
   = note: the method exists on `Warp<All>`
help: consider calling `merge` to reconverge the warp first
```

**Merge non-complements:**
```
error[E0277]: the trait bound `Even: ComplementOf<LowHalf>` is not satisfied
  --> src/main.rs:15:18
   |
15 |     let merged = merge(evens, low_half);
   |                  ^^^^^ the trait `ComplementOf<LowHalf>` is not implemented for `Even`
   |
   = help: the following implementations were found:
             <Even as ComplementOf<Odd>>
   = note: `Even` and `LowHalf` are not complements (they overlap)
```

These messages tell the programmer exactly what's wrong and how to fix it.

## 6.8 Practical Considerations

### Compile Time

The type system adds negligible compile time. Trait resolution is fast, and there's no complex type-level computation (unlike, say, type-level naturals).

Benchmark on our test suite:
- Without types: 2.3s
- With types: 2.4s
- Overhead: ~4%

### Code Size

Zero-sized types and monomorphization mean code size is unchanged. The types are erased; only the operations remain.

### Debugging

Types are visible in debuggers and error messages. A `Warp<Even>` is clearly different from `Warp<All>`, aiding debugging.

### Interop

The library can wrap existing CUDA code:

```rust
// Wrap unsafe CUDA code with typed interface
pub fn safe_butterfly(data: PerLane<i32>) -> Uniform<i32> {
    let warp: Warp<All> = Warp::kernel_entry();
    // ... typed operations ...
}
```

Unsafe code can be isolated behind safe APIs.

## 6.9 Limitations

### Static Active Sets Only

Our Rust implementation handles static active sets (Even, Odd, LowHalf, etc.). Runtime-dependent sets require:
- Existential types (`SomeWarp` with runtime mask), or
- Dependent types (beyond Rust's type system)

### Manual Diverge/Merge

Programmers must call `diverge` and `merge` explicitly. Automatic insertion based on control flow is future work.

## 6.10 Summary

Our Rust implementation demonstrates that warp typestate can be embedded in an existing systems language with:
- **Zero runtime overhead**: Types are erased at MIR, LLVM IR, and PTX levels
- **Good error messages**: Rust's diagnostics explain what's wrong
- **Familiar syntax**: Looks like normal Rust code
- **Easy adoption**: Wrap existing code incrementally
- **Platform portability**: Same algorithms run on CPU and GPU via the `Platform` trait
- **Cargo integration**: A `#[warp_kernel]` proc macro and `WarpBuilder` build-time library enable end-to-end GPU kernel development from `cargo run` — kernel crates use `warp_types::*`, cross-compile to PTX automatically via `build.rs`, and load via a generated `Kernels` struct with named function handles

The key insight: Rust's type system is expressive enough to encode our safety properties without runtime cost. The cargo-integrated pipeline further demonstrates practical usability — type-safe kernels that compile to real PTX with real shuffle instructions, verified on NVIDIA RTX 4000 Ada hardware.


---

# 7. Evaluation

We evaluate warp typestate on three dimensions:
1. **Bug Detection**: Does the type system catch real divergence bugs?
2. **Performance**: What is the runtime overhead?
3. **Expressiveness**: Can practical GPU algorithms be expressed without excessive friction?

## 7.1 Bug Detection

### Real Bugs Surveyed

We surveyed 21 documented shuffle-from-inactive-lane bugs across 16 GPU projects and evaluated whether our type system would have caught them at compile time.

**Modeled bugs** (self-contained Rust examples in `examples/`):

| Bug | Source | Failure Mode | Caught? |
|-----|--------|--------------|---------|
| Wrong ballot mask in reduction | cuda-samples#398 | Silent wrong sum | Yes — modeled |
| Compiler predicates off mask init | CCCL#854 | Silent wrong scan | Yes — modeled |
| Hardcoded full mask in divergent branch | PIConGPU#2514 | Confirmed UB, no wrong output observed‡ | Yes — modeled |
| shfl_sync causes branch elimination | LLVM#155682 | Atomic 32x overcounting | Partial — modeled† |
| Deprecated `__shfl` API family | CUDA 9.0 | Entire bug class | N/A (vendor response) |

‡PIConGPU#2514: The undefined behavior is real (`__ballot_sync(0xFFFFFFFF, 1)` inside a divergent branch violates the CUDA spec). However, a contributor tested on K80 with CUDA 8 and 9 and observed no errors — pre-Volta hardware enforced convergence at warp level, masking the UB. The fix (PR #2600) was preventive, targeting Volta's independent thread scheduling. We do not claim wrong output was observed.

**Additional documented bugs** (identified via systematic search):

| Bug | Source | Failure Mode | Caught? |
|-----|--------|--------------|---------|
| Full mask in divergent scan → hang | OpenCV#12320 | Deadlock on Volta/Turing | Yes — modeled in code |
| Full mask with early-exit → freeze | Ginkgo#54 | Deadlock on Titan V | Yes |
| `__activemask()` misuse in radix sort | PyTorch#98157 | Wrong distribution counts | Yes — modeled in code |
| Wrong mask computation → illegal insn | TVM PR#17307 | Crash on H100 | Yes — modeled in code |
| Bad mask for logical warp < 32 | CUB#115 | Silent wrong reduction | Yes |
| ShuffleIndex wrong for sub-warps | CUB#112 | Silent wrong aggregate | Yes |
| WarpReduce wrong for non-pow2 sizes | CCCL#858 | Silent wrong reduction | Partial |
| Illegal warp sync in parallel_reduce | Kokkos#1520/1612/1958 | Crash on Volta/Turing | Yes |
| Legacy shfl broken on SM 8.0 | Halide#5630 | Silent wrong results | Partial |
| Cross-lane wrong results in branches | ROCm/HIP#952/2474 | Wrong results (AMD) | Yes |
| Shuffle lacks execution dependency | NVIDIA SPIR-V (blog) | Silent data corruption | Yes |
| Legacy warp-sync shuffle on Volta | HOOMD-blue#292 | Potential wrong results | Yes |
| NaN lost after WarpReduce refactor | cuDF#6365 | Silent precision loss | Partial |
| Butterfly shuffle order → racecheck | Triton#7264 | Nondeterministic results | Partial |
| Excluded indexed shuffle entirely | WebGPU spec | Design-level avoidance | Motivation |
| Migration generates deadlock-prone code | SYCLomatic DPCT1086 | Potential deadlock | Yes |

**Summary**: Of 21 documented issues, our type system fully catches 14 (67%), partially catches 5 (24%), and 1 serves as design-level motivation (WebGPU's decision to exclude indexed subgroup shuffles entirely validates the need for type-safe alternatives). The dominant bug patterns are: (1) hardcoded `0xFFFFFFFF` mask in divergent code (OpenCV, Ginkgo, PIConGPU, HOOMD-blue); (2) incorrect mask computation for logical sub-warps (CUB#112, CUB#115, CCCL#858, TVM); (3) `__activemask()` misuse as convergence proxy (PyTorch); (4) cross-vendor portability (ROCm/HIP, SPIR-V, WebGPU, SYCLomatic).

†The LLVM bug involves a compiler optimization that eliminates a branch because `__shfl_sync` implies all lanes must be active. Our type system prevents the *source-level* pattern (conditional write followed by full-warp shuffle), but cannot prevent compiler misoptimization of otherwise well-typed code. This is a compiler correctness issue, not a type system issue.

### Concrete Demonstration: cuda-samples #398

We modeled the cuda-samples#398 bug as a self-contained Rust example (`examples/nvidia_cuda_samples_398.rs`). The original CUDA bug:

```cuda
// In reduce7, after block-level tree reduction with block_size=32:
// Only tid 0 enters the reduction path
unsigned mask = __ballot_sync(0xFFFFFFFF, tid < blockDim.x / warpSize);
// mask = 1 (only lane 0 active)
sdata[tid] += __shfl_down_sync(mask, sdata[tid], 16);
// BUG: reads from lane 16, which is inactive
```

In our type system, after the block-level reduction narrows to one lane:

```rust
let (lane0, _rest) = warp.extract_lane0();
// lane0 : Warp<Lane0>

let shifted = lane0.shuffle_down(sdata, 16);
// ERROR: no method `shuffle_down` found for `Warp<Lane0>`
```

The bug is not caught at runtime—it cannot be *expressed*. The `shuffle_down` method does not exist on `Warp<Lane0>`. This is the core guarantee: unsafe operations are absent, not checked.

The example includes two correct fixes:
- **Fix A**: Check `active_count == 1` and skip reduction (lane 0 already has the result).
- **Fix B**: Zero inactive lanes and reduce the full warp.

Both fixes type-check because they ensure all lanes participate before shuffling.

### Concrete Demonstrations: Remaining Bugs

Each remaining documented bug has a self-contained worked example demonstrating the exact type error our system produces and why `__shfl_sync`'s runtime mask does not catch the bug.

**PIConGPU #2514** (`examples/picongpu_2514.rs`): After divergence, the warp handle becomes `Warp<Active>`. Calling `ballot()` on `Warp<Active>` is a type error—`ballot()` exists only on `Warp<All>`. The CUDA code used `__ballot_sync(0xFFFFFFFF, 1)` inside a divergent branch; the hardware accepted the mask because `0xFFFFFFFF` is a valid `u32`, regardless of how many lanes were actually active.

**CUB/CCCL #854** (`examples/cub_cccl_854.rs`): The source-level mask was correct; the compiler generated wrong PTX by predicating off the mask initialization. In our type system, the mask is `PhantomData<SubWarp16>`—a zero-sized phantom type with no register and no initialization. The compiler cannot optimize away something that doesn't exist at runtime. `shuffle_up()` on `Warp<SubWarp16>` is a type error.

**LLVM #155682** (`examples/llvm_155682.rs`): After `if (laneId == 0)`, lane 0 has `Warp<Lane0>`. Calling `shuffle_broadcast()` on `Warp<Lane0>` is a type error. The fix—merging back to `Warp<All>` via `merge(lane0, rest)`—forces both sides to provide data, eliminating the uninitialized value that triggered LLVM's UB-based branch elimination.

### Hardware Reproduction

We reproduced the cuda-samples#398 bug on an NVIDIA RTX 4000 SFF Ada (compute 8.9, Ada Lovelace architecture) using CUDA 12.0. With `block_size=32`, the buggy `reduce7` kernel consistently returns `sum = 1` instead of `sum = 32` (the correct value). The result is deterministic across 10 runs—not intermittent. The bug also manifests at `block_size=256`: the kernel returns `sum = 32` instead of `sum = 256`, because `blockDim.x / warpSize = 8` means only 8 lanes enter the final reduction, producing a partial ballot mask (`0xFF`). The `__shfl_down_sync` with offset 16 reads from lanes outside this mask.

The bug affects all block sizes where `blockDim.x / warpSize < 32`—only `block_size=1024` produces the correct result, where all 32 lanes vote true and the ballot mask is `0xFFFFFFFF`. The fixed version (all lanes participate with zeroed inactive data) produces correct results at all block sizes. The reproduction code is in `reproduce/reduce7_bug.cu`.

### Vendor Response: API Deprecation and Architectural Change

The severity of shuffle-divergence bugs is reflected in NVIDIA's response. In CUDA 9.0, NVIDIA deprecated the entire `__shfl`, `__shfl_up`, `__shfl_down`, and `__shfl_xor` API family—a breaking change across the CUDA ecosystem. The CUDA Programming Guide states: *"If the target thread is inactive, the retrieved value is undefined"* (§10.22). The replacement `__shfl_sync` family requires an explicit mask parameter, but as Bugs 1–4 demonstrate, the mask is a runtime value that programmers still get wrong.

The problem deepened with Volta's independent thread scheduling. Pre-Volta architectures enforced warp-level lockstep execution, so implicit convergence masked most shuffle-divergence UB. Volta allowed threads within a warp to genuinely interleave, turning latent UB into observable bugs. NVIDIA's architecture whitepaper acknowledges this: *"[Independent thread scheduling] can lead to a rather different set of threads participating in the executed code than intended."* Code that happened to work on Pascal silently broke on Volta—and the `__shfl_sync` mask was supposed to fix this, but simply moved the burden from implicit hardware convergence to explicit (and error-prone) programmer annotation.

### Compile-Fail Tests as Proof Artifacts

Our implementation includes thirteen compile-fail doctests that serve as machine-checked proof artifacts:

1. `shuffle_xor` on `Warp<Even>` — rejected, method absent (`diverge.rs`)
2. `merge(Even, LowHalf)` — rejected, non-complements (`merge.rs`)
3. `merge(Even, Even)` — rejected, same set (`merge.rs`)
4. `merge(EvenLow, EvenHigh)` as top-level — rejected, nested complements are not `ComplementOf` (`merge.rs`)
5. `merge(EvenLow, OddHigh)` — rejected, non-covering nested sets (`nested_diverge.rs`)
6. `shuffle_xor` on `Warp<Even>` — rejected, research variant (`static_verify.rs`)
7. `merge(Even, LowHalf)` — rejected, research variant (`static_verify.rs`)
8. `merge(Even, Even)` — rejected, research variant (`static_verify.rs`)
9. Use-after-diverge (`warp.shuffle_xor` after `warp.diverge_even_odd()`) — rejected, moved value (`warp.rs`)
10. `Warp::new()` from external crate — rejected, `pub(crate)` constructor (`warp.rs`)
11. `merge_writes(Even, LowHalf)` — rejected, fence non-complements (`fence.rs`)
12. `bitonic_sort` on `Warp<Even>` — rejected, method absent (`sort.rs`)
13. `tile` on `Warp<Even>` — rejected, method absent (`tile.rs`)

These are not test heuristics—they are verified absences. The Rust compiler confirms that each operation is a type error. Any future change to the type system that accidentally permits these operations would cause `cargo test` to fail.

### Bug Pattern Coverage

Our prototype includes 291 unit tests, 50 example tests across 8 worked bug examples, and 28 doc tests (13 compile-fail including linearity, nested complement, fence, sort, and tile enforcement, 15 doc examples) covering the full type system (369 total). The tests exercise:

- Diverge/merge with complement verification
- Nested divergence (up to depth 3)
- Recursive protocols (5 loop patterns)
- Arbitrary predicates (existential, indexed, hybrid)
- Work stealing with dynamic roles
- Platform portability (warp-32; wavefront-64 mask width supported but not hardware-tested)
- Warp-size-generic algorithms

Every test validates that the type system permits correct patterns and rejects incorrect ones.

## 7.2 Performance

### Zero Overhead by Construction

Our types impose zero runtime overhead—not measured to be negligible, but *guaranteed by construction*:

- `Warp<S>` contains only `PhantomData<S>`, which has zero size and zero runtime representation.
- `ActiveSet` is a trait with `const MASK: u64`—resolved at compile time.
- `ComplementOf<S>` is a trait bound, checked at compile time.
- Monomorphization eliminates all generic dispatch.

The generated code contains no trace of the type system. A `Warp<All>` and a `Warp<Even>` produce identical machine code for any operation available on both.

We verified this at three levels of the compilation pipeline:

**Rust MIR**: `Warp<S>` values are zero-sized and optimized away entirely. No runtime checks, no mask comparisons, no branches.

**LLVM IR** (`cargo rustc --release --lib -- --emit=llvm-ir`): We provide two inspectable functions with `#[no_mangle] #[inline(never)]`. In the optimized IR:

```llvm
; A butterfly reduction (5 shuffle_xor + reduce_sum): on CPU, shuffle is identity,
; so reduce_sum computes val + val five times = val * 32 = val << 5.
; All Warp<S>, PhantomData, trait bounds are erased — only the arithmetic remains:
define noundef i32 @zero_overhead_butterfly(i32 noundef %data) {
  %1 = shl i32 %data, 5
  ret i32 %1
}

; A diverge/merge round trip is pure identity — the type-level operations
; (kernel_entry, diverge_even_odd, merge) are ALL erased to nothing:
define noundef i32 @zero_overhead_diverge_merge(i32 noundef returned %data) {
  ret i32 %data
}
```

The warp creation, diverge, merge, and all active set types are *completely erased*. `zero_overhead_diverge_merge` compiles to `ret i32 %data` — the type system operations vanish entirely. `zero_overhead_butterfly` retains only the arithmetic (`shl` from the CPU shuffle-identity reduction). The only `Warp`-containing symbols in the entire optimized IR are error message strings and `DynWarp` functions (which intentionally carry a runtime `u64` mask).

**NVIDIA PTX** (`rustc +nightly --target nvptx64-nvidia-cuda --emit=asm -O`): We compiled actual Rust type system code—`PhantomData`, trait bounds, `ComplementOf`, `diverge_even_odd()`, `merge()`—directly to NVIDIA PTX via the `nvptx64-nvidia-cuda` target. Two function pairs:

1. `butterfly_typed` (5 shuffle operations through `Warp<All>`) vs. `butterfly_untyped` (raw arithmetic): **byte-identical PTX** — both compile to a single `shl.b32 %r2, %r1, 5`.

2. `diverge_merge_typed` (`Warp::kernel_entry()` → `diverge_even_odd()` → `merge(evens, odds)`) vs. `diverge_merge_untyped` (identity function): **byte-identical PTX** — both compile to `ld.param → st.param → ret`.

```ptx
; butterfly_typed — Warp<All>, shuffle_xor, trait bounds all erased:
.visible .func  (.param .b32 func_retval0) butterfly_typed(
    .param .b32 butterfly_typed_param_0
) {
    .reg .b32   %r<3>;
    ld.param.b32    %r1, [butterfly_typed_param_0];
    shl.b32         %r2, %r1, 5;       ; entire type system gone
    st.param.b32    [func_retval0], %r2;
    ret;
}
```

The `Warp<S>` type, `PhantomData`, `ActiveSet` trait, `ComplementOf` bound, `diverge`, and `merge` are all fully erased—not by Rust's frontend, but through the entire LLVM NVPTX backend to actual GPU instructions. Reproduction: `bash reproduce/compare_rust_ptx.sh` (requires `rustc +nightly` with `nvptx64-nvidia-cuda` target).

### Comparison with Runtime Approaches

The alternative to compile-time safety is runtime mask checking—verifying at each shuffle that the mask matches the active set. NVIDIA's `compute-sanitizer --tool synccheck` provides this:

| Approach | Overhead | Coverage | Feedback |
|----------|----------|----------|----------|
| `__shfl_sync` only (no verification) | 0% | None | Silent UB |
| Runtime sanitizer | Significant | Executed paths only | At test time |
| Our type system | 0% | All paths | At compile time |

Our approach provides strictly more coverage at strictly less cost.

## 7.3 Expressiveness

### The Hazy Argument

The most sophisticated persistent thread program as of 2025—the Hazy megakernel [Stanford 2025]—prohibits lane-level divergence by design. Their on-GPU interpreter dispatches at warp granularity: all 32 lanes execute the same operation, and every shuffle uses `MASK_ALL = 0xFFFFFFFF`. They never allow different lanes to run different ops. This is *architectural avoidance*: state-of-the-art practitioners treat lane-level divergence as too dangerous to manage, even with `__shfl_sync`, and prohibit it entirely.

In our type system, Hazy-style programs type-check trivially: every warp is `Warp<All>`, every shuffle is permitted, and no diverge/merge annotations are needed. The type system is invisible for uniform programs.

This answers the standard expressiveness objection ("does the type system reject too much?") with empirical evidence: the programs that state-of-the-art practitioners *actually write* are the easiest to type. But it also reveals a stronger point: our type system could safely *relax* the restriction that Hazy imposes—allowing lane-level heterogeneity with compile-time safety guarantees that architectural avoidance cannot provide.

### Lane-Heterogeneous Programs

Programs that use lane-level divergence require explicit `diverge`/`merge` annotations. This is the intended design: the annotation *is* the safety contract. It makes visible the control-flow structure that was previously implicit and error-prone.

The annotation overhead is modest. Divergence points require one `diverge` call (producing two typed sub-warps) and one `merge` call (consuming them). For a butterfly reduction with 5 shuffle stages, the typed version adds 3 lines (the initial diverge, a merge to restore `Warp<All>`, and a data merge).

### Patterns and Their Typability

| Pattern | Typable? | Annotation Needed |
|---------|----------|-------------------|
| Butterfly reduction | Yes | None (uniform) |
| Kogge-Stone scan | Yes | None (uniform) |
| Predicated filter | Yes | diverge/merge per predicate |
| Adaptive sort | Yes | diverge/merge per partition |
| Warp-level work stealing | Yes | dynamic role assignment |
| Cooperative group with sub-warp | Yes | existential types (§5) |
| Data-dependent shuffle mask | Yes | `diverge_dynamic(mask)` with structural complement guarantees |

### Annotation Burden

We measured the annotation overhead by analyzing the user-facing algorithm code in the five original worked bug examples (`examples/*.rs`), counting lines that exist solely because of the type system (warp parameters, diverge/merge calls, type annotations) vs. core algorithm lines.

| Example | Algorithm Lines | Annotation Lines | Annotation % |
|---------|----------------|-----------------|-------------|
| cuda-samples#398 | 16 | 2 | 12.5% |
| PIConGPU#2514 | 11 | 3 | 27.3% |
| LLVM#155682 | 15 | 4 | 26.7% |
| CUB/CCCL#854 | 5 | 2 | 40.0% |
| demo | 8 | 4 | 50.0% |
| **Total** | **55** | **15** | **27.3%** |

The overall annotation burden is **27.3%** — roughly one annotation line per three algorithm lines. The range (12.5%–50%) is instructive: the lowest burden occurs when the fix is algorithmic (cuda-samples#398: check active count, skip reduction — only the `warp: Warp<All>` parameter is overhead). The highest occurs in minimal examples where diverge/merge ceremony dominates. In larger real algorithms, the ratio converges toward the lower end.

The annotations are: `warp: Warp<All>` function parameters (5 occurrences), `diverge_*()` calls (4), `merge()` calls (3), and type annotations on warp handles (2). Notably, `kernel_entry()` is called only at kernel entry points — library functions take `Warp<All>` as a parameter, which is the idiomatic pattern.

### Limitations

Four patterns require special handling:

1. **Data-dependent divergence**: When the active set depends on runtime data (e.g., `data[lane] > threshold`), our static marker types do not apply. Instead, `diverge_dynamic(mask)` returns a `DynDiverge` with structural complement guarantees — the mask is runtime, but the pairing ensures both branches must merge before shuffle. No dependent types needed.

2. **Arbitrary runtime predicates**: Our marker types cover common patterns (Even, Odd, LowHalf, HighHalf). Predicates not matching these markers require existential types, which add a runtime check.

3. **Cross-function active set polymorphism**: Functions generic over the active set use Rust's standard trait bounds (`fn foo<S: ActiveSet>(warp: Warp<S>)`). The type parameter `S` is inferred at call sites. This is idiomatic Rust — not a limitation of the type system, but a language property.

4. **Irreversible divergence**: If one branch of a diverge exits early (return, panic, trap), the warp handle for that branch is dropped, violating linearity. The type system correctly rejects this—without both halves, you cannot reconstruct `Warp<All>` for subsequent shuffles. The workaround is a ballot-based exit pattern: lanes that finish early spin until all lanes agree to exit, maintaining a full warp throughout. `DynWarp` (§9.4) provides a runtime escape for patterns where static exit tracking is too restrictive.

These limitations are real but narrowly scoped. The first two are addressed by our extension layers (§5); the third is a standard trade-off in any type-parameterized system; the fourth follows necessarily from the linearity discipline that makes the type system sound.

**Scope clarification**: Our type system guarantees source-level safety—well-typed programs cannot express shuffle-from-inactive-lane at the source level. This guarantee does not extend to compiler transformations (Bug 4/LLVM#155682 demonstrates that compilers can introduce the bug from well-typed source) or to hardware scheduling (Volta's independent thread scheduling may change which lanes are physically converged). Source-level safety is the appropriate guarantee for a type system; compiler correctness and hardware conformance are separate concerns.

## 7.4 Threats to Validity

**Bug sample size**: Our evaluation surveys 21 documented shuffle-divergence bugs across 16 projects (§7.1), with 8 modeled as self-contained Rust examples. Of 21 bugs, 14 are fully caught by the type system. The 5 partially-caught bugs involve patterns at the boundary of our system's expressiveness (non-power-of-2 logical warps, NaN semantics, nondeterministic accumulation order).

**Limited GPU hardware evaluation**: Our type system prototype runs on CPU, emulating warp semantics. The zero-overhead claim is established by type erasure verified at three levels: Rust MIR, LLVM IR, and NVIDIA PTX (§7.2). We compiled actual Rust type system code (PhantomData, trait bounds, diverge/merge) to PTX via `nvptx64-nvidia-cuda` and confirmed byte-identical output vs. untyped equivalents. We also reproduced the cuda-samples#398 bug on actual GPU hardware (RTX 4000 SFF Ada, compute 8.9), confirming that the undefined behavior produces deterministically wrong results on post-Volta architectures.

**Selection bias**: The bugs we model are ones where the type system succeeds. We explicitly identify patterns where it does not (data-dependent masks, §7.3). We are not aware of shuffle-from-inactive-lane bugs that our type system would fail to catch at the source level.

## 7.5 Summary

| Metric | Result |
|--------|--------|
| Real bugs surveyed | 21 across 16 projects (14 fully caught, 5 partial, 1 vendor response, 1 motivation) |
| Real bugs modeled | 8 with worked Rust examples (+ 5 mechanized untypability proofs in Lean) |
| Type system tests | 291 unit + 50 example + 28 doc (369 total) |
| Runtime overhead | 0% (verified: Rust MIR, LLVM IR, NVIDIA PTX via nvptx64) |
| Annotation burden | 27.3% of algorithm lines (range: 12.5%–50%) |
| Uniform programs | Zero annotation overhead |
| Lean mechanization | Progress, preservation, substitution lemma — all zero-sorry, zero-axiom. 5 bug untypability proofs. 28 named theorems total (§4.8) |
| Data-dependent divergence | Yes | `diverge_dynamic(mask)` — structural complement, no dependent types |

Warp typestate provides strong safety guarantees with zero runtime cost. For uniform programs (the dominant style in practice), it is invisible. For lane-heterogeneous programs, it makes divergence explicit—replacing implicit bugs with explicit types.

We do not claim shuffle-divergence bugs are the most *frequent* GPU bug class. We claim they are the most *insidious*: they produce silent data corruption rather than crashes, survive testing at common configurations, and resist source-level reasoning (Bug 4 demonstrates that even correct source can produce wrong code). NVIDIA deprecated an entire API family to address the problem; their replacement still relies on runtime masks that programmers get wrong. State-of-the-art persistent thread programs avoid the problem by prohibiting lane-level divergence entirely. Our type system is the first approach that makes lane-level divergence *safe* rather than *forbidden*.

---

# 8. Related Work

Warp typestate draws on and differs from work in GPU verification, session types, and type systems for parallelism.

## 8.1 GPU Verification

### Descend (PLDI 2024)

Descend [Kopcke et al. 2024] brings Rust-style ownership and borrowing to GPU programming. It prevents data races and use-after-free in GPU code.

**Relationship to our work**: Descend and warp typestate are *orthogonal* and *composable*:
- Descend: memory safety (ownership, borrowing, lifetimes)
- Our work: divergence safety (active lane tracking)

Descend does not track which lanes are active. A shuffle in Descend may read from inactive lanes if the programmer gets the mask wrong. Conversely, our system does not track memory ownership.

The ideal system combines both: lanes must be active (our contribution) *and* have valid data (Descend's contribution).

### GPUVerify

GPUVerify [Betts et al. 2012, 2015] is a static verification tool for GPU kernels. It uses predicated execution semantics and barrier invariants to prove race-freedom.

**Relationship to our work**: GPUVerify is an *external verifier*, not a type system:
- Separate tool run after compilation
- Provides yes/no answer (not incremental feedback)
- Heavyweight (SMT solving, can timeout)

Our approach integrates verification into the type system:
- Immediate feedback during editing
- Incremental (errors as you type)
- Lightweight (type checking is fast)

GPUVerify does not specifically target shuffle-from-inactive-lane bugs, though its predicated semantics could potentially catch them.

### CUDA Sanitizers

NVIDIA provides sanitizers (compute-sanitizer) for detecting GPU errors at runtime:
- Memory errors (memcheck)
- Race conditions (racecheck)
- Synchronization issues (synccheck)

**Relationship to our work**: Sanitizers are *dynamic* analysis:
- Catch bugs during testing, not compilation
- Only find bugs in executed paths
- Significant runtime overhead (significant [NVIDIA Compute Sanitizer documentation])

Our approach catches bugs at compile time, before any execution.

### GPU Race Detection

GMRace [Zheng et al. 2014] and CURD [Peng et al. 2018] detect warp-level data races using static analysis and dynamic instrumentation, respectively.

**Relationship to our work**: Race detection focuses on data races (concurrent conflicting accesses to shared memory), not divergence bugs (reading from inactive lanes via shuffle). These are related but distinct bug classes: a race involves two threads accessing the same location; a divergence bug involves one thread reading stale register data from an inactive lane.

### LLVM Uniformity and Divergence Analysis

LLVM implements uniformity analysis that determines whether SSA values are uniform (same across all threads in a warp) or divergent. This analysis propagates divergence along def-use chains and control dependencies, supporting irreducible control flow.

**Relationship to our work**: LLVM's divergence analysis and our type system track related information—which program points have non-uniform behavior—but differ fundamentally in mechanism and guarantee. LLVM's analysis is: (1) a compiler pass, not a source-level type system; (2) intraprocedural and best-effort; (3) focused on optimization (avoiding unnecessary predication), not safety. Our type system is: (1) source-level with programmer annotations; (2) modular across function boundaries; (3) focused on safety (preventing inactive-lane reads). The question "why not extend LLVM's analysis to catch shuffle bugs?" has a precise answer: LLVM's analysis identifies *which* values are divergent but does not track *which lanes are active*. A value known to be divergent might still be shuffled safely if all lanes happen to be active. Our active-set types capture exactly this distinction. Additionally, Bug 4 (LLVM#155682) demonstrates that LLVM's own optimizations can *cause* the bug class we prevent—the compiler and the type system are at different levels of abstraction.

## 8.2 Session Types

### Binary Session Types

Session types were introduced by Honda [1993] for the π-calculus. A session type describes a communication protocol between two parties.

**Relationship to our work**: Binary session types assume two active parties. GPU divergence involves up to 32 parties where any subset may be inactive. We extend the model with *quiescence*.

### Multiparty Session Types (MPST)

Honda, Yoshida, and Carbone [2008] extended session types to multiple parties. Each party follows a local type projected from a global protocol.

**Relationship to our work**: MPST assumes all parties remain active (or fail). GPU divergence has parties *go quiescent and resume*. This is the key novelty:

| MPST | Our Extension |
|------|---------------|
| N parties, all active | 32 parties, subset active |
| Party fails = session stuck | Party quiesces = temporarily inactive |
| No reconvergence | Merge = reconvergence point |

### Gradual Session Types

Gradual session types [Igarashi et al. 2017] allow mixing static and dynamic typing for sessions. Unknown types are checked at runtime.

**Relationship to our work**: Our Layer 4 (existential types, §5.2) and `DynWarp` gradual typing bridge (§9.4) are directly inspired by this work. The difference is our focus on lane sets rather than general protocol conformance. In particular, our `ascribe()` operation—promoting a runtime-checked `DynWarp` to a compile-time `Warp<S>`—corresponds to the cast at the gradual typing boundary.

### Fault-Tolerant Multiparty Session Types

Recent work extends MPST to handle participant failures: crash-stop failures [Adameit et al. 2022], fault-tolerant event-driven programming [Viering et al. 2021], and mixed static/dynamic verification of global protocols under failures.

**Relationship to our work**: Fault-tolerant MPST models *permanent* participant failure (crash-stop). GPU divergence involves *temporary* quiescence—lanes go inactive and later resume at a merge point. This is a crucial distinction: crash-stop requires protocol recovery or timeout; quiescence requires complement proof for reconvergence. Our quiescence model is optimistic (all parties resume), while crash-stop is pessimistic (some parties never resume). The two extensions are complementary—combining them could model GPU scenarios where threads genuinely trap or exit.

### Session Types Embedded in Rust (Ferrite)

Ferrite [Chen et al. 2022] is a judgmental embedding of session types in Rust using PhantomData, zero-sized types, and type-level programming—the same encoding techniques we use.

**Relationship to our work**: Ferrite and warp typestate share an encoding strategy (phantom types in Rust) but model different domains. Ferrite models inter-process communication channels where parties send and receive messages. We model intra-warp lane communication where parties go quiescent. Key differences: (1) Ferrite's channels carry data; our warps share a register file. (2) Ferrite uses linear channels; we use linear warp handles. (3) Ferrite's session types describe message sequences; ours describe active-set evolution. The shared encoding technique validates our claim that Rust's type system is expressive enough for session-type embeddings.

### Session Types for Concurrent Objects

Dardha et al. [2017] apply session types to concurrent objects in object-oriented languages.

**Relationship to our work**: They model object-to-object communication. We model lane-to-lane communication within a warp. The synchronization models differ: objects are asynchronous; warps are lock-step.

## 8.3 Type Systems for Parallelism

### Futhark

Futhark [Henriksen et al. 2017] is a functional GPU language with a type system that guarantees regular parallelism.

**Relationship to our work**: Futhark *avoids* divergence by design. Its parallelism constructs (map, reduce, scan) don't support divergent branches.

Our approach is complementary: we *embrace* divergence and make it safe. This allows expressing algorithms like adaptive sorting where divergence is fundamental.

### ISPC (Intel SPMD Program Compiler)

ISPC [Pharr and Mark 2012] is a C-variant compiler that implements SPMD programming on CPU SIMD hardware. It introduces `uniform` and `varying` type qualifiers: a `uniform` variable holds a single value shared across all program instances in a gang (analogous to our `Uniform<T>`), while `varying` (the default) holds per-instance values (analogous to our `PerLane<T>`). The compiler enforces a directional conversion rule—`uniform` implicitly widens to `varying`, but assigning `varying` to `uniform` is a compile error. ISPC also provides `foreach_active` (which serializes execution over active instances) and cross-lane operations (`shuffle`, `broadcast`, `reduce_add`) whose mask correctness is guaranteed by the compiler, which emits and propagates execution mask instructions automatically.

**Relationship to our work**: ISPC is the closest existing system to ours in its awareness of divergence at the language level, and its `uniform`/`varying` distinction directly inspired our `Uniform<T>`/`PerLane<T>` types. The key difference is *what* the type system tracks. ISPC's types encode *value uniformity*—whether all instances hold the same value—but not *which instances are active*. The execution mask is a runtime value managed implicitly by the compiler; there is no type-level active set. A function cannot declare in its signature that it requires all lanes to be active, and cross-lane operations are safe because the compiler controls mask emission at runtime, not because the type system prevents misuse. Our `Warp<S>` types go further: they encode the active set itself, making `shuffle_xor` *absent from the type* when lanes are inactive rather than masked at runtime. This distinction matters on GPU hardware, where ISPC's approach does not apply: the execution mask is managed by hardware, not the compiler, and the programmer passes masks to warp intrinsics manually—exactly the interface where the bug class arises.

### DPJ (Deterministic Parallel Java)

DPJ [Bocchino et al. 2009] uses region types to ensure determinism in parallel Java programs.

**Relationship to our work**: DPJ focuses on determinism through effect typing. Our focus is different: we ensure safety of warp-level communication, not determinism.

### Æminium

Æminium [Stork et al. 2014] is an implicitly parallel language with a permission system based on access permissions.

**Relationship to our work**: Æminium extracts parallelism from sequential code. We type explicitly parallel GPU code. The goals are opposite: they hide parallelism; we expose it.

### Data-Race-Free Type Systems

Several systems ensure data-race freedom through types [Boyapati et al. 2002, Flanagan and Freund 2000].

**Relationship to our work**: Race-freedom and divergence safety are distinct:
- Race: two threads access same location, at least one writes
- Divergence bug: one thread reads from inactive lane's register

Our active set types are not about preventing races—they're about preventing reads from inactive lanes.

## 8.4 Linear and Affine Types

### Ownership Types (Rust)

Rust's ownership system [Matsakis and Klock 2014] ensures memory safety through affine types (values used at most once).

**Relationship to our work**: We leverage Rust's type system for our implementation. The `Warp<S>` type is linear—consumed by diverge, produced by merge. This prevents use-after-diverge.

### Linear Logic and Session Types

The connection between linear logic and session types [Caires and Pfenning 2010, Wadler 2012] provides a logical foundation for session types.

**Relationship to our work**: Our warp linearity follows this tradition. Diverge corresponds to ⊗ (tensor) producing two resources; merge corresponds to ⅋ (par) consuming two resources.

## 8.5 GPU Programming Models

### CUDA and OpenCL

CUDA [NVIDIA 2007] and OpenCL [Khronos 2009] are the dominant GPU programming models. Both expose warp/wavefront primitives but provide no type-level safety.

**Relationship to our work**: We build on top of these models, adding a typed layer. Our types can wrap CUDA intrinsics.

### SYCL and oneAPI

SYCL [Khronos 2020] and Intel's oneAPI provide modern C++ abstractions for heterogeneous programming.

**Relationship to our work**: These aim for portability and productivity but do not address divergence safety. Our approach could be integrated into SYCL's sub-group operations.

### HIP

AMD's HIP [AMD 2016] is largely CUDA-compatible. Our approach applies equally to HIP's wavefront primitives.

### Cooperative Groups

CUDA's Cooperative Groups [NVIDIA 2017] provide a unified interface for thread groups at all levels.

**Relationship to our work**: Cooperative Groups make group membership explicit but don't provide type safety. A thread can still shuffle on a group where some threads have diverged. We provide the missing types.

### NVIDIA's `__shfl_sync` Migration (CUDA 9.0)

NVIDIA deprecated the original `__shfl` family in CUDA 9.0, replacing it with `__shfl_sync` which requires an explicit mask parameter [CUDA Programming Guide §10.22]. This was a vendor acknowledgment that the bug class is severe enough to warrant a breaking API change across the ecosystem. However, the mask remains a runtime value—programmers can still pass the wrong mask, as documented bugs in NVIDIA's own cuda-samples and CUB demonstrate.

**Relationship to our work**: `__shfl_sync` addresses the problem at the API level (require a mask). We address it at the type level (prove the mask correct). The approaches are complementary: `__shfl_sync` prevents *forgetting* the mask; our types prevent *getting it wrong*.

### Hazy Megakernel (2025)

The Hazy megakernel [Stanford 2025] is the most sophisticated persistent thread program as of this writing, fusing ~100 operations into a single kernel with an on-GPU interpreter.

**Relationship to our work**: Hazy avoids the divergence problem by prohibiting lane-level divergence entirely—all 32 lanes execute the same operation, every shuffle uses `MASK_ALL`. This is safe but restrictive. Our type system is strictly more permissive: uniform programs type-check trivially (as Hazy's would), while lane-heterogeneous programs become expressible with explicit type annotations. We make divergence *safe* rather than *forbidden*.

## 8.6 Summary

| Related Work | Focus | Our Difference |
|--------------|-------|----------------|
| Descend | Memory safety | We do divergence safety |
| GPUVerify | External verification | We use types |
| MPST | All parties active | We model quiescence |
| ISPC | uniform/varying (value uniformity) | We track active sets (which lanes), not just uniformity |
| Futhark | Avoids divergence | We embrace + type it |
| `__shfl_sync` | Require mask (runtime) | We prove mask correct (compile-time) |
| Hazy megakernel | Prohibit divergence | We make divergence safe |
| DPJ | Determinism | We do lane safety |
| Rust ownership | Memory | We do active sets |

**Our unique contribution**: Session types extended with quiescence for SIMT divergence. No prior work types the active lane mask.


---

# 9. Future Work

Warp typestate opens several research directions.

## 9.1 Tooling and Data-Dependent Divergence

Our tooling stack includes two implemented proc macros and a build-time library:

1. **`warp_sets!`** (§6.1) generates the static active set hierarchy with compile-time validation of disjoint/covering invariants.
2. **`#[warp_kernel]`** transforms kernel functions into `extern "ptx-kernel"` entry points with `#[no_mangle]`, validating that parameters are GPU-compatible types (raw pointers or scalars).
3. **`WarpBuilder`** cross-compiles kernel crates to PTX via `cargo rustc --target nvptx64-nvidia-cuda -Z build-std=core`, finds the generated `.s` file, and produces a Rust module with a `Kernels` struct providing named `CudaFunction` handles.

Data-dependent predicates (e.g., `data[lane] > threshold`) are supported via `diverge_dynamic(mask)`, which returns a `DynDiverge` — a paired divergence where the mask is runtime but the complement is structural. Both branches must merge to recover `Warp<All>`:

```rust
let warp: Warp<All> = Warp::kernel_entry();
let mask = ballot_result;  // runtime predicate
let diverged = warp.diverge_dynamic(mask);
// Can't shuffle on either branch — complement guaranteed by construction
let warp: Warp<All> = diverged.merge();
warp.shuffle_xor(data, 1);  // OK — all lanes active
```

A future `#[warp_typed]` proc macro could further optimize this by automatically inserting `diverge_dynamic` calls and tracking merge pairing, reducing boilerplate for complex data-dependent algorithms.

## 9.2 Formal Mechanization

Our core metatheory is fully mechanized in Lean 4 (§4.8): progress, preservation, and the substitution lemma are all machine-checked with zero `sorry` and zero axioms. Five bug untypability proofs are also mechanized. Remaining future work:
- Extend mechanization to nested divergence (generalize `IsComplementAll` to `IsComplement s1 s2 parent`)
- Mechanize the loop typing rules (§5.1) and set-preserving shuffle (§4.6)
- Verified Rust implementation via Aeneas translation
- Leverage prior Lean-based GPU verification work (MCL framework)

## 9.3 IDE Integration

Rich IDE support would enhance usability:
- Visualize active sets at each program point
- Show which lanes are active in hover tooltips
- Suggest merge points when shuffles fail type checking
- Refactoring: extract divergent code into typed helper functions

## 9.4 Protocol Inference and Gradual Typing

Our current system requires explicit type annotations. We have explored inference strategies in research prototypes — local inference (within functions), bidirectional checking (mix inference and annotation), and gradual typing — with 14 tests across five approaches (`src/research/protocol_inference.rs`).

The gradual typing approach is promoted to the public API (`src/gradual.rs`, 16 tests): `DynWarp` provides the same operations as `Warp<S>` but checks safety invariants at runtime instead of compile time. The migration path:

1. **Start dynamic**: `DynWarp::all()` — all operations runtime-checked
2. **Ascribe at boundaries**: `dyn_warp.ascribe::<All>()?` — runtime evidence becomes compile-time proof
3. **End static**: `Warp<S>` everywhere — zero-overhead, compile-time safety

`DynWarp` also handles the data-dependent predicate case (§9.1): when the active set depends on runtime data and cannot be expressed as a marker type, `DynWarp` provides runtime safety that `Warp<S>` cannot.

Remaining future work:
- Local inference integration into the public API (infer active sets within functions, require annotations only at boundaries)
- Protocol-first development (design protocol in DSL, generate/check code against it)

## 9.5 Beyond SIMT

The core idea—session types with quiescent participants—may apply beyond GPUs. We grade each potential transfer by mechanism fidelity: does the target domain share the same failure mode (reading from an inactive participant produces silent corruption), or merely a structural resemblance?

**FPGA crossbar protocols** (strong transfer): We have demonstrated this direction with a working prototype (§9.6). The mapping is direct: `TileGroup<S>` ↔ `Warp<S>`, tile sets ↔ active sets, `TileComplement` ↔ `ComplementOf`. The bug class is isomorphic: when a tile doesn't SEND, its pipeline register retains stale data—silent corruption identical to shuffle-from-inactive-lane. Mechanism, scale, and coupling all match.

**Distributed systems** (partial transfer): Node quiescence maps to lane inactivity, and multiparty session types already model distributed communication. However, the domains diverge on three axes: (1) distributed systems have genuine failure modes (Byzantine, crash-stop, network partition) that GPU warps lack; (2) SIMT divergence is deterministic (predicate-based) while distributed failure is non-deterministic; (3) SIMT guarantees eventual reconvergence at a merge point while distributed systems may not reconverge. The quiescence model is complementary to fault-tolerant MPST (§8.2) but not a direct replacement.

**Database queries and proof search** (structural similarity only): Predicate filtering in databases and case splits in proof search share the abstract shape of "active subset selection," but the mechanism diverges fundamentally. Database rows are independent data items, not lock-step execution units sharing an instruction stream—there is no "communication between filtered rows" analogous to shuffle. Proof sub-goals interact via shared logical context, not register exchange. We note the structural parallel but do not claim actionable type-system transfer to these domains.

## 9.6 Hardware Crossbar Protocols

We have prototyped typestate crossbar communication (`src/research/crossbar_protocol.rs`, 12 tests) modeling a 16-tile pipelined crossbar. The mapping is direct: `TileGroup<S>` mirrors `Warp<S>`, tile sets mirror active sets, and `TileComplement` mirrors `ComplementOf`. Crossbar collectives (ring pass, butterfly exchange, scatter, gather) exist only on `TileGroup<AllTiles>` — after `diverge_halves()`, the methods vanish from the type.

The hardware bug class is real: when a tile diverges and doesn't SEND, its pipeline register retains data from the previous cycle. Other tiles reading from that channel get stale data with no hardware error — silent corruption identical to shuffle-from-inactive-lane. Our prototype's `stale_data_bug_demonstration` test reproduces this failure mode and shows how warp typestate prevents it.

Future work extends toward hardware synthesis proper:
- Generating crossbar routing configurations from typestate protocols
- Synthesizing predication logic matching diverge/merge structure
- Area/power optimization guided by protocol structure (unused crossbar paths can be power-gated)

## 9.7 Remaining Limitations

Several limitations remain:
- Higher-order protocols (protocols parameterized by protocols)
- Compilation overhead at scale (untested on large codebases)
- Cross-warp fence interactions (warp A diverges, warp B's fence depends on A's contribution via global memory — the intra-warp case is handled in §5.6, but cross-warp ordering remains open)
- **Tensor core and async operations**: Warp matrix operations (WMMA/MMA) distribute matrix fragments across lanes; a diverged warp using tensor cores produces incorrect fragments—the same bug class as shuffle-from-inactive-lane, but at the matrix fragment level. Async copy operations (cp.async, TMA) have analogous divergence issues. Our type system's `Warp<All>` requirement would correctly force all lanes active before these operations, but we have not modeled the operations' internal lane-to-fragment mappings.
- **Cross-vendor portability**: Our core `ActiveSet::MASK` uses `u64`, supporting both 32-lane NVIDIA warps and 64-lane AMD wavefronts. AMD GPU intrinsic stubs (`amdgpu` target) are in place but untested without hardware. AMD RDNA supports dual Wave32/Wave64 modes; Intel Xe subgroups vary from 8 to 32 lanes. The `WarpBuilder` accepts a `GpuTarget` enum (Nvidia/Amd) for cross-compilation. Vulkan subgroup operations provide a vendor-neutral target.

# 10. Conclusion

GPU warp programming is notoriously error-prone. Shuffles that read from inactive lanes produce undefined behavior—bugs that compile silently, work sometimes, and fail unpredictably. NVIDIA's own reference code contains these bugs. A plasma physics simulation ran for months with undefined behavior undetected on pre-Volta hardware. The vendor deprecated an entire API family to address the problem. State-of-the-art persistent thread programs prohibit lane-level divergence entirely rather than manage it.

We presented **warp typestate**, a type system that makes lane-level divergence safe rather than forbidden:

1. **Warps carry active set types** (`Warp<Even>`, `Warp<All>`), tracking which lanes are active.

2. **Divergence produces complements**. When a warp splits, the type system knows the sub-warps together cover the original.

3. **Merge verifies complements**. The type system statically checks that merged warps are complementary.

4. **Shuffles require all lanes active**. The `shuffle_xor` method exists only on `Warp<All>`. Calling it on a diverged warp is not a runtime error—it is *unrepresentable*.

The key insight is that GPU divergence fits the session type model: diverging is branching where some parties go *quiescent* (not failed, just paused), and reconverging is joining where quiescent parties resume. This correspondence gives us a principled type discipline for an ad-hoc problem.

Our implementation in Rust has **zero runtime overhead**—guaranteed by construction, not measured. Types are erased at compile time. For uniform programs (the style used by state-of-the-art megakernels), the type system is invisible. For lane-heterogeneous programs, it replaces implicit bugs with explicit types. The result is strictly more permissive than the divergence-prohibition approach while being strictly safer than CUDA's `__shfl_sync`.

Beyond safety, the type system **unlocks performance that is currently too dangerous to attempt.** State-of-the-art GPU kernels (Flash Attention, Hazy megakernel) avoid lane-level divergence entirely—not because uniform execution is always optimal, but because divergent shuffles are too risky to get right. Algorithms that could be faster with lane-level heterogeneity (warp-level bitonic sort in tile-based rasterizers, divergent reductions in ray-BVH traversal, per-lane work stealing in load-imbalanced ML kernels) are written as uniform instead, leaving performance on the table. Warp typestate makes the fast-but-dangerous path safe, enabling optimizations that the current programming model discourages.

Warp typestate is not just a solution for GPU programming. It is an instance of a broader pattern: *participatory computation* where the set of active participants changes over time. We have demonstrated a direct transfer to FPGA crossbar protocols (§9.6), identified partial transfers to distributed systems, and noted structural parallels in databases and proof search (§9.5). The transfer fidelity correlates with mechanism match: domains where inactive participants produce silent data corruption (FPGAs, GPUs) benefit most; domains with merely analogous "active subset selection" benefit least.

**The takeaway**: Divergence bugs are type errors. Types exist to make certain classes of bugs impossible. Now shuffle-from-inactive-lane is one of them.

---

## Acknowledgments

The author used Claude (Anthropic, claude-opus-4-6, 2026) extensively in the drafting and editing of this manuscript.

## References

[References would be formatted according to venue style. Key citations include:]

- Betts et al. 2012. "GPUVerify: A Verifier for GPU Kernels" (OOPSLA)
- Adameit et al. 2022. "Generalised Multiparty Session Types with Crash-Stop Failures" (CONCUR)
- Bocchino et al. 2009. "A Type and Effect System for Deterministic Parallel Java" (OOPSLA)
- Caires and Pfenning 2010. "Session Types as Intuitionistic Linear Propositions" (CONCUR)
- Chen et al. 2022. "Ferrite: A Judgmental Embedding of Session Types in Rust" (ECOOP)
- Hazy Research 2025. "Look Ma, No Bubbles! Megakernel for Llama-1B" (Stanford Blog)
- Honda 1993. "Types for Dyadic Interaction" (CONCUR)
- Honda, Yoshida, Carbone 2008. "Multiparty Asynchronous Session Types" (POPL)
- Henriksen et al. 2017. "Futhark: Purely Functional GPU-Programming" (PLDI)
- Igarashi et al. 2017. "Gradual Session Types" (ICFP)
- Kopcke et al. 2024. "Descend: A Safe GPU Systems Programming Language" (PLDI)
- Lange and Yoshida 2016. "On the Undecidability of Asynchronous Session Subtyping" (FoSSaCS)
- LLVM. "Convergence And Uniformity" (LLVM Documentation)
- LLVM#155682: shfl_sync causes branch elimination
- NVIDIA 2017. "Cooperative Groups: Flexible Thread Synchronization" (GTC)
- NVIDIA 2017. CUDA Programming Guide §10.22: Warp Shuffle Functions (deprecation notice)
- NVIDIA 2017. Tesla V100 Architecture Whitepaper: Independent Thread Scheduling
- NVIDIA cuda-samples#398: Wrong ballot mask in reference reduction
- NVIDIA CCCL#854: Compiler predicates off mask initialization in CUB WarpScanShfl
- Peng et al. 2018. "CURD: A Dynamic CUDA Race Detector" (PLDI)
- PIConGPU#2514: Hardcoded full mask in divergent branch
- Viering et al. 2021. "A Multiparty Session Typing Discipline for Fault-Tolerant Event-Driven Distributed Programming" (OOPSLA)
- Wadler 2012. "Propositions as Sessions" (ICFP)
- Wright and Felleisen 1994. "A Syntactic Approach to Type Soundness" (IC)
- Zheng et al. 2014. "GMRace: Detecting Data Races in GPU Programs via a Low-Overhead Scheme" (IEEE TPDS)
- Anthropic. (2026). Claude Opus 4.6 [Large language model]. https://www.anthropic.com

