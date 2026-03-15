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
    mask: u32,
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
        // Warp-level: session-typed divergence
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

The warp protocol uses our session-typed divergence. The block protocol uses traditional session types. They compose hierarchically.

### Why This Matters

The novel contribution of this paper is *warp-level* session types with quiescence. This is the gap in prior work:
- Traditional session types: all parties active or failed
- Our extension: parties can go quiescent and resume

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
// Warp-level: session-typed divergence
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

The key insight: not every problem needs session-typed divergence. We identify where our contribution applies (warp-level quiescence) and where existing techniques suffice (block-level communication). Notably, the fence-divergence extension reuses the complement proof mechanism from merge — the same `ComplementOf` trait that ensures safe reconvergence also ensures safe memory ordering.

