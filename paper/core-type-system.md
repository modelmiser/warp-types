# 3. Session-Typed Divergence

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

Importantly, `Warp<S>` is a *linear* type—it cannot be duplicated or discarded. A warp that diverges into two sub-warps must eventually merge back. This prevents "losing" lanes.

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

Warps are linear—each warp value must be used exactly once:

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
fn butterfly_sum(warp: Warp<All>, data: PerLane<i32>) -> Uniform<i32> {
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
    data.assume_uniform()  // Unsafe: programmer asserts uniformity
}
```

Every `shuffle_xor` call type-checks because `warp` has type `Warp<All>` throughout.

### Adding Divergence

Now consider a filtered reduction where some lanes don't participate:

```rust
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
    let inactive_data = PerLane::splat(0);  // Non-participants contribute 0

    // Merge back to Warp<All>
    let warp: Warp<All> = merge(active, inactive);  // ✓ Keep ⊥ ¬Keep
    let combined = merge_data(active_data, inactive_data);

    // Now butterfly reduction is safe
    butterfly_sum(warp, combined)
}
```

The type system guides the programmer: you cannot shuffle until you merge. The merge requires complementary sets. The result is correct by construction.

## 3.6 Summary

Our type system provides safety through three mechanisms:

1. **Active set types** (`Warp<S>`) track which lanes are active at each program point.

2. **Typing rules** ensure diverge produces complements and merge consumes complements.

3. **Method availability** restricts operations to types where they are safe—`shuffle_xor` only exists on `Warp<All>`.

The result: shuffle-from-inactive-lane bugs become compile errors. The fix is explicit (merge before shuffle) and verified by the type checker.
