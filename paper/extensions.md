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

We don't try to track the exact active set through the loop. Instead, we verify entry and exit invariants via four typing rules. The listings in this section are written in the calculus' surface syntax (`for_uniform`, `loop_until`, `for_varying`, `for_phased`, `ballot_any`, `data[lane]`); they are not crate APIs. The crate exposes no loop combinators — the loop rules are mechanized in Lean (§4.8) and Rust code uses ordinary loops over a borrowed `&Warp<S>`.

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

Layers 1 and 4 are implemented: Layer 1 is the static hierarchy, and Layer 4 ships as `diverge_dynamic`/`DynDiverge` (paired, complement by construction) and the gradual `DynWarp` (§9.1, §9.3), whose `merge` checks disjointness at run time. Layers 2–3 (`LowRange`, `Stride`, `register_predicate`, `diverge_indexed`) and the `SomeWarp::merge_with_complement` listing are design sketches; those names exist nowhere in the crate. Layer 3 handles cases where the predicate is known but the satisfying lanes are not. Layer 4 is the full escape hatch for arbitrary data-dependent patterns, trading static guarantees for runtime checks while still ensuring the check cannot be forgotten.

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

The contribution of this section is *linear typestate with quiescent participants* (linear in the formal calculus; the Rust crate is affine, §6). Relative to session types:

- Traditional session types: all parties active or failed
- Our extension: parties can go quiescent and resume

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

Our type system extends to these levels in design: each group would carry its own active-set type parameter — e.g., `Warp<S: ActiveSet>` for 32-lane warps and a `Subgroup<S: ActiveSet8>` for 8-lane partitions — and a `tiled_partition` operation would create typed subgroups from a warp, each inheriting the active-set discipline. The crate implements `Warp<S>` and const-sized `Tile`s (without per-tile active sets); `Subgroup` is not implemented.

The same principles apply at every width: full-group shuffle requires all group lanes active, diverge produces complements, and merge verifies them. The complement mechanism is the same at every width; the crate's `ComplementOf` has no width parameter and is implemented for the configured warp width (32, or 64 with `warp64`), with const-sized tiles alongside.

## 5.5 Memory Safety Integration

Our type system focuses on divergence safety and could be layered with memory safety systems like Descend [Köpcke et al. 2024]. The two systems' main concerns differ:

- **Descend** prevents data races and use-after-free via ownership/borrowing for GPU memory
- **Warp typestate** prevents reads from inactive lanes via active-set tracking

Descend ensures no data races and also rejects synchronization on conditional or split thread groups; warp typestate types the complementary lane sets of a divergence and their merge. Descend's ownership and borrowing address memory; warp typestate addresses which lanes are active. Combining them is unimplemented: no checker in this repository enforces both, and a unified type system tracking `Warp<S: ActiveSet, M: MemoryState>` is future work (§9).

## 5.6 Fence-Divergence Interactions

Global memory writes from diverged warps raise an ordering question: when may a value stored by one group of lanes be read back by another, or published to observers outside the warp? Two of the instructions this crate emits bear on it, and they do different things. `bar.warp.sync` is a rendezvous: the lanes it names wait for one another, and it orders memory among them, so a lane may store, synchronise, and then read what the other lanes it synchronised with stored. (`Warp::sync` issues it on `nvptx64` in the code after 0.4.3, inside a lane guard and over the live lanes of the handle's set; through 0.4.3 `sync` emitted nothing, and a barrier over part of a warp is defined only on sm_70 and later.) `membar` (`__threadfence()`) is not a rendezvous: it makes the *executing* lane's earlier memory accesses performed — its stores visible — at the chosen level, before that lane's later memory operations, and the PTX ISA describes no waiting for any other lane. A fence therefore says nothing about whether the other lanes have stored yet.

We model a publication discipline on top of this with a typestate machine over a *phantom* region: a fence can be typed only once the write-step calls recorded for the region cover the whole warp. The discipline is a design choice of the model, not something the hardware requires of `membar`. The region is bound to no address, and its operations store and load nothing: a partial state carries a lane set in its type — the set of the handle on which a write step was called, or, after `merge_writes_within`, the parent set into which two complementary partial states were merged (`diverge` emits no branch, so which lanes *execute* a call is not what is recorded — §6.9). What the types enforce is the order of those recorded steps — not that a store took place, which remains the program's own obligation, through its own pointer. A lane that reads another lane's data inside the warp needs `Warp::sync` on a handle whose set covers both lanes, not the fence:

```
GlobalRegion<Unwritten>
  → Warp<S>.global_store() → GlobalRegion<PartialWrite<S>>
then EITHER (two regions obtained by `split`)
  → merge_writes(PartialWrite<S₁>, PartialWrite<S₂>) → GlobalRegion<FullWrite>
      where S₁: ComplementOf<S₂>
OR (one region)
  → Warp<S₂>.global_store_complement(PartialWrite<S₁>) → GlobalRegion<FullWrite>
      where S₂: ComplementOf<S₁>
then
  → threadfence(FullWrite) → GlobalRegion<Fenced>
```

The key insight is that `merge_writes` reuses the same `ComplementOf` proof as warp merge — the mechanism that ensures complementary active sets at reconvergence also ensures that the recorded write steps cover complementary lane sets before a fence. A fence on a region whose recorded steps cover only part of the warp is not a runtime error — it is a compile error within the typed API, exactly as a full-warp shuffle on a diverged warp is. The same qualifications apply (§6.9): the complement proof checks set shape, not provenance, and on `nvptx64` the fence instruction is also reachable without any proof through the public `gpu::threadfence()`.

```rust
fn safe_fence_pattern(warp: Warp<All>) {
    let (evens, odds) = warp.diverge_even_odd();
    let region = GlobalRegion::new();

    // Each half records its write step (the store itself goes through the program's own pointer)
    let (_evens, partial) = evens.global_store(region);
    let (_odds, full) = odds.global_store_complement(partial);  // ComplementOf<Even> for Odd

    // Fence only once both halves are recorded — FullWrite required by type
    let fenced = threadfence(full);
    let _val: i32 = fenced.read();  // Typed only in the Fenced state; in the model it loads nothing
}
```

This listing is schematic. The shipped API brands the region with a rank-2 closure (`GlobalRegion::with_region(|region| …)`) so that a region cannot escape its scope and partial writes from two different regions cannot be merged, splits it with `split`, and merges partial writes with `merge_writes`/`merge_writes_within`; `threadfence(FullWrite)` is the typed gate; in the code after 0.4.3 (unreleased at the time of writing) it also emits `membar.gl` on `nvptx64` (through 0.4.3 it emitted nothing), while the region and its `read` remain a model with no memory behind them (§6.9): `read` returns a default value. The Lean fence rule (`lean/WarpTypes/Fence.lean`, an instance of the family-parametric core of §4.8: `finalizeFamily` at tag `.group`) gates on write-*permission* coverage (the operand's type must be `.group (PSet.all n)`), not on writes performed.

## 5.7 Summary

Our extensions handle:

| Challenge | Solution | Static Safety |
|-----------|----------|---------------|
| Uniform loops | Entry/exit invariants | Full |
| Varying loops | No warp ops in body | Body warp-free (Lean types the loop form, not the body) |
| Runtime predicates | Layered types (1–4) | Varies |
| Inter-block | Traditional MPST | Design only (no checker shipped) |
| Cooperative groups | Hierarchical protocols | Design only (`Subgroup` not implemented) |
| Memory safety | Compose with Descend | Future work |
| Fence-divergence | Typestate over recorded write steps (phantom region) | Recorded-step coverage (Rust); permission coverage (Lean) — a model, see §5.6 |

The key insight: not every problem needs warp typestate. We identify where our contribution applies (warp-level quiescence) and where existing techniques suffice (block-level communication, memory safety).

