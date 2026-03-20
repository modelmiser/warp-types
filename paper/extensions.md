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

We provide a hierarchy of four approaches, from most static to most dynamic. Each layer trades some static guarantees for increased expressiveness:

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
    mask: u32,
    // Active set is unknown at type level
}

impl SomeWarp {
    fn merge_with_complement(self, other: SomeWarp) -> Result<Warp<All>, Error>;
}
```

Safety is checked at runtime (masks must be complements), but the type system still prevents forgetting the check — you cannot use `SomeWarp` for warp operations without first proving complementarity through `merge_with_complement`.

### Recommendation

Most GPU code uses Layer 1–2 (80%+ of cases). Layer 3 handles cases where the predicate is known but the satisfying lanes are not. Layer 4 is the full escape hatch for arbitrary data-dependent patterns, trading static guarantees for runtime checks while still ensuring the check cannot be forgotten.

## 5.3 Inter-Block Communication

Our core system models a single warp. Real GPU programs have multiple blocks, each containing multiple warps. How do session types extend to this hierarchy?

### Key Insight: Inter-Block is Traditional MPST

Warp-level divergence is special: lanes go *quiescent* within a single instruction stream. This is what our type system models.

Block-level communication is different:
- Blocks execute *independently* (no shared instruction stream)
- Blocks don't "go quiescent" — they run fully or don't exist
- Communication is through shared memory or global memory
- Synchronization is explicit (barriers, atomics)

This is the domain of traditional multiparty session types [Honda, Yoshida, Carbone 2008].

### Hierarchical Protocols

We compose the two levels hierarchically: warp protocols use our typestate within each block, while inter-block protocols use traditional MPST. A matrix multiplication kernel, for example, would use traditional MPST to coordinate tile distribution across blocks, while each block internally uses warp typestate to ensure safe shuffles within its reduction phase.

### Why This Matters

The novel contribution of this paper is *linear typestate* for warp divergence — tracking active lane sets at compile time via a Boolean lattice. This is the gap in prior work:

- Traditional session types: all parties active or failed
- Our approach: lanes go temporarily inactive (quiescent) and resume, tracked in the type system via complementary active sets

Inter-block communication doesn't need this extension — it is already well-served by existing session type systems. Our contribution is recognizing where the existing theory applies and where we need something new.

## 5.4 Cooperative Groups

CUDA's Cooperative Groups API provides a unified abstraction for thread groups at all levels:

```cuda
auto warp = cooperative_groups::tiled_partition<32>(this_thread_block());
auto block = this_thread_block();
auto grid = this_grid();
```

Modern GPUs also support subgroups smaller than a warp via `tiled_partition`.

### Typing Cooperative Groups

Our type system extends naturally to these levels. Each group carries its own active-set type parameter — e.g., `Warp<S: ActiveSet>` for 32-lane warps and `Subgroup<S: ActiveSet8>` for 8-lane partitions. The `tiled_partition` operation creates typed subgroups from a warp, each inheriting the active-set discipline.

The same principles apply at every width: shuffle requires all group lanes active, diverge produces complements, and merge verifies them. The `ComplementOf` proof mechanism is parameterized by group size, not hardcoded to 32 lanes.

## 5.5 Memory Safety Integration

Our type system focuses on divergence safety and composes orthogonally with memory safety systems like Descend [Kopcke et al. 2024]. The two systems address independent concerns:

- **Descend** prevents data races and use-after-free via ownership/borrowing for GPU memory
- **Warp typestate** prevents reads from inactive lanes via active-set tracking

The systems are orthogonal: Descend ensures no data races, while our system ensures no divergence bugs. Together, they provide comprehensive GPU safety — the first preventing spatial errors (wrong memory), the second preventing temporal errors (wrong execution state). Neither subsumes the other, and they can be layered via a proc macro that checks both ownership and active-set constraints on each operation. A unified type system tracking `Warp<S: ActiveSet, M: MemoryState>` is future work (§9).

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

## 5.7 Summary

Our extensions handle:

| Challenge | Solution | Static Safety |
|-----------|----------|---------------|
| Uniform loops | Entry/exit invariants | Full |
| Varying loops | No warp ops in body | Full |
| Runtime predicates | Layered types (1–4) | Varies |
| Inter-block | Traditional MPST | Full |
| Cooperative groups | Hierarchical protocols | Full |
| Memory safety | Compose with Descend | Full |
| Fence-divergence | Type-state write tracking | Full |

The key insight: not every problem needs warp typestate. We identify where our contribution applies (warp-level quiescence) and where existing techniques suffice (block-level communication, memory safety).
