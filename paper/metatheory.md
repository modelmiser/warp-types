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

Warps are linear resources—they cannot be duplicated or discarded. This is essential for soundness.

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
- **Structured control flow**: Diverge and merge are explicit operations, not implicit branches. For structured control flow, divergence analysis is decidable and efficiently computable—compilers already do it [LLVM uniformity analysis].
- **No data-dependent active sets**: The type system tracks static patterns (Even, Odd, LowHalf), not arbitrary runtime predicates.

These limitations are addressed in §5 (Extensions).

## 4.8 Mechanization

We have mechanized the full core metatheory in Lean 4 (`lean/WarpTypes/`). All theorems are machine-checked with **zero `sorry` and zero axioms**.

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

