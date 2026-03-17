# Warp Typestate: Lean 4 Formalization

A mechanized proof of type safety for a linear type system that prevents GPU warp divergence bugs at compile time.

Warp-level shuffle operations on diverged (partially active) warps are a class of undefined behavior in CUDA/GPU programming that no existing type system catches. This formalization proves that a typestate approach makes such programs untypable.

## What This Formalizes

The type system models GPU warp resources as linear values typed by their active lane sets (32-bit bitmasks). Warp handles must be consumed exactly once — diverged, merged, or shuffled — enforced by a linearity condition on let-bindings. Shuffle operations are restricted to fully-converged warps (`Warp<All>`), making it a type error to shuffle on a subset of lanes.

This catches three classes of bugs:
- **Shuffle on diverged warp**: reading from inactive lanes (undefined behavior)
- **Missing merge**: forgetting to reconverge after divergence (resource leak)
- **Mismatched merge**: merging non-complementary subsets

## Theorems Proven

All theorems are fully machine-checked. Zero sorry, zero axioms.

| Theorem | Statement | File |
|---------|-----------|------|
| **Progress** | Well-typed closed terms are values or can step | Metatheory.lean |
| **Preservation** | Reduction preserves typing | Metatheory.lean |
| **Substitution** | Substituting a value preserves typing (generalized via context removal) | Metatheory.lean |
| **Diverge partition** | `diverge` produces disjoint, covering sub-sets | Basic.lean |
| **Shuffle requires All** | Shuffle typing requires `Warp<All>` | Basic.lean |
| **Complement symmetry** | Complement relation is symmetric | Basic.lean |
| **Bug 1** (cuda-samples #398) | Shuffle after lane-0 extraction is untypable | Metatheory.lean |
| **Bug 2** (CUB/CCCL #854) | Shuffle on 16-lane sub-warp is untypable | Metatheory.lean |
| **Bug 3** (PIConGPU #2514) | Ballot on diverged subset is untypable | Metatheory.lean |
| **Bug 4** (LLVM #155682) | Shuffle after lane-0 conditional is untypable | Metatheory.lean |
| **Bug 5** (demo) | Shuffle after even/odd divergence is untypable | Metatheory.lean |

## Key Design Decisions

**Linearity on `letBind`.** The `letBind` typing rule enforces two side conditions: freshness (`ctx'.lookup name = none`, preventing shadowing) and consumption (`ctx''.lookup name = none`, requiring the binding to be used). This is domain-appropriate — an unused warp handle is a resource leak and likely a reconvergence bug.

This linearity condition was introduced to fix a soundness bug discovered during mechanization: the original affine `letBind` rule leaked bound variables into the output context. Concretely, `let x = warpVal(all) in unitVal` would type-check with `x` still in the output context, but after reduction to `unitVal`, the output context was empty — breaking preservation. The fix strengthens the type system's guarantees rather than weakening them.

**Generalized substitution via context removal.** The substitution lemma is stated as: substituting a value for a name removes that name's binding from both input and output contexts (`HasType (ctx.remove nm) (subst e nm v) t (ctx'.remove nm)`). This formulation handles the consumed case (removal is a no-op on an absent binding) and the unconsumed case (removal strips the leftover) uniformly, avoiding case-splitting in the merge, shuffle, and pair cases where the binding may have been threaded past the first sub-expression.

**Value restriction.** The substitution lemma applies only to values (`isValue v = true`). Values do not reference variables and can be typed in any context (`value_any_ctx`), which is essential for the `var` case where the substituted value must type-check in the reduced context.

## Build

Requires Lean 4.28.0. No external Lake dependencies (uses only Lean's bundled Std library for `BVDecide`).

```
lake build
```

The build completes with zero errors, zero sorry, zero axioms. Warnings are limited to unused simp arguments (cosmetic).

## Repository Structure

```
lean/
  lakefile.toml          Build configuration
  lean-toolchain         Lean version (v4.28.0)
  Main.lean              Entry point (build verification)
  WarpTypes.lean         Module root
  WarpTypes/
    Basic.lean           Type system: active sets, types, expressions,
                         typing rules, diverge/complement/shuffle theorems,
                         value predicate (194 lines)
    Metatheory.lean      Metatheory: substitution, reduction, canonical
                         forms, progress, preservation, context lemmas,
                         substitution lemma, 5 bug proofs (702 lines)
```

896 lines across the two core files.

## Provenance

Developed independently by Chad Aldreda, outside academic or institutional affiliation. Proof exploration was AI-assisted using Claude (Anthropic).

## Citation

```bibtex
@software{aldreda2025warptypes,
  author    = {Aldreda, Chad},
  title     = {Warp Typestate: Lean 4 Formalization},
  year      = {2025},
  url       = {https://github.com/modelmiser/warp-types}
}
```
