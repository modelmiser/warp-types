# Type-Safe GPU Warp Programming via Linear Typestate

**Chad Aldreda**

## Abstract

GPU warp primitives like shuffle enable efficient communication between threads, but reading from an inactive lane produces undefined behavior. These bugs are notoriously difficult to detect—they compile successfully, may appear to work, and fail silently: NVIDIA's own reference code contains them, and a plasma physics simulation ran for months with undefined behavior that went undetected on pre-Volta hardware. State-of-the-art persistent thread programs avoid the problem by maintaining warp-uniform execution.

Existing approaches track which lanes are active at runtime (ISPC's compiler-emitted masks, NVIDIA's Cooperative Groups) or detect violations after execution (compute-sanitizer). We present *warp typestate*, which embeds the active lane mask in the type system. Divergence creates sub-warps with complementary active sets; reconvergence requires type-level proof that the sets are complements. Operations requiring all lanes (shuffles) are not checked at runtime but *absent from the type* — a diverged warp has no shuffle method to call.

We prove our type system sound (progress and preservation via Lean 4 mechanization), implement it as a Rust library with zero runtime overhead, and demonstrate that it catches real bugs from NVIDIA's cuda-samples and CUB library at compile time. The result is strictly more permissive than the divergence-prohibition approach (which maintains warp-uniform execution) while being strictly safer than CUDA's `__shfl_sync` (which defers mask correctness to runtime).

## Paper Sections

| Section | File | Pages | Status |
|---------|------|-------|--------|
| 1. Introduction | [introduction.md](introduction.md) | ~3 | ✅ Revised (empirical evidence, refined contributions) |
| 2. Background | [background.md](background.md) | ~3 | ✅ Complete |
| 3. Core Type System | [core-type-system.md](core-type-system.md) | ~4 | ✅ Complete |
| 4. Metatheory | [metatheory.md](metatheory.md) | ~3 | ✅ Complete |
| 5. Extensions | [extensions.md](extensions.md) | ~3 | ✅ Complete |
| 6. Implementation | [implementation.md](implementation.md) | ~2 | ✅ Complete |
| 7. Evaluation | [evaluation.md](evaluation.md) | ~3 | ✅ Revised (real bugs, honest data, Hazy argument) |
| 8. Related Work | [related-work.md](related-work.md) | ~2.5 | ✅ Revised (Hazy, NVIDIA deprecation) |
| 9-10. Future & Conclusion | [future-and-conclusion.md](future-and-conclusion.md) | ~1.5 | ✅ Revised (fence scope boundary, updated framing) |
| Appendix. Empirical Evidence | [empirical-evidence.md](empirical-evidence.md) | — | Supporting material for §1 and §7 |

**Total: ~19 pages** (condensed from ~25; see CHANGELOG)

## Key Contributions

1. **Novel type system for GPU divergence** — First to type active lane masks, preventing undefined behavior from reading inactive lanes.

2. **Linear typestate over active-set lattice** — Tracks which lanes are active via a Boolean lattice of bitmasks, enforced by Rust's move semantics and sealed traits. (Not session types proper — see §3 and §6.1 for the distinction.)

3. **Soundness proof** — Progress and preservation theorems ensure well-typed programs never read from inactive lanes.

4. **Zero-overhead implementation** — Rust embedding using traits and generics; types erased at compile time.

5. **Empirical grounding** — Real bugs from NVIDIA cuda-samples, CUB, and PIConGPU modeled and caught at compile time.

## Demo

See [../examples/demo_bug_that_types_catch.rs](../examples/demo_bug_that_types_catch.rs) for a runnable demonstration of the core type safety guarantee.

See [../examples/nvidia_cuda_samples_398.rs](../examples/nvidia_cuda_samples_398.rs) for a concrete model of a real NVIDIA bug (cuda-samples#398) that our type system catches at compile time.

```bash
cargo run --example demo_bug_that_types_catch
cargo test --example nvidia_cuda_samples_398
```

## The Bug That Types Catch

```cuda
// CUDA (undefined behavior)
if (participate) {
    int partner = __shfl_xor_sync(0xFFFFFFFF, data, 1);  // BUG!
    return data + partner;
}
```

```rust
// Our types (compile error)
let (active, _) = warp.diverge(|lane| participate[lane]);
let partner = active.shuffle_xor(data, 1);
// ERROR: no method `shuffle_xor` found for `Warp<Active>`
// note: the method exists on `Warp<All>`
// help: merge with complement before shuffling
```

## Related Files

- [../src/proof.rs](../src/proof.rs) — Executable proof sketch (9 tests)
- [../lean/](../lean/) — Lean 4 formalization (32 named theorems, zero sorry, zero axioms)

