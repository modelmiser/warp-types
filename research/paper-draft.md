# One Gate, Four Domains: Complemented Typestate from GPU Warps to Tree All-Reduce

**Working draft.** Sections §1, §2, §3, §5, and §6 are drafted. §4 (domain instances), §7 (related work), §8 (conclusion), and the abstract are TK pending venue and page-budget decisions — see the TK list at the end of this file.

---

## 1. Introduction

A type system for parallel computation has to answer one question well: which operations are safe at which participant sets? A warp-wide shuffle is safe only when every lane of the warp is active. A CSP broadcast is safe only when every tile in the receiver set is ready to receive. A memory fence commits only when every byte of the buffer has been written. A tree all-reduce produces a correct result only when every leaf has contributed to the accumulator. The four operations live in four different computational models — SIMT, MIMD/CSP, relaxed shared memory, divide-and-conquer reduction — and they have historically been analysed by four different type systems. This paper argues that all four share a single structural gate, that the gate can be stated once in a generic core language, and that a single Lean 4 inductive typing judgment subsumes the four domain-specific judgments without losing theorem coverage at any domain.

The gate is complement-at-merge. Each domain carries a participant set — a bitvector over the lanes, tiles, bytes, or leaves that own an operation — and each domain has some notion of *partitioning* the set into two disjoint sub-sets. Partitioning is cheap: a group splits into its "true on predicate" and "false on predicate" halves, and linearity ensures that both halves are held by distinct sub-computations. Merging is where safety is purchased: a rule requires that the two operand sets are complementary with respect to some parent set, and only then does the rule produce a handle on the parent. The unsafe case — attempting an all-participants operation on a merged result that covers less than the full parent set — becomes untypable. The same structural gate `PSet.IsComplement s1 s2 parent` governs the merge, and a second gate `PSet.all n` governs the extract, in every one of the four domains.

The two gates are orthogonal to two further axes. The extract gate can return a barrier (the fence case — a `.unit` continuation) or an extracted value (the reduce case — a `.data` payload), and the same `PSet.all n` gate governs both. The merge gate can act on a type family indexed by participant set — a permission handle as in Fence, or an accumulator as in Reduce — and the same `PSet.IsComplement` gate governs both. Stating the gates in a language that is parametric in both axes is the technical contribution: two family-parametric typing rules, `mergeFamily` and `finalizeFamily`, factored into a generic core file `Core.lean` and instantiated per domain. The factoring uses an enumeration of type-family tags plus four `@[reducible]` dispatcher definitions, with a double equality witness on both the expression form and the conclusion type of each parametric rule.

**Contributions.** This paper makes four contributions. **(1)** A generic core language for complemented typestate (`Core.lean`) whose two family-parametric typing rules subsume the corresponding rules in four pre-existing domain instances (GPU warp divergence, CSP broadcast/collective, partial-write fencing, tree all-reduce), and whose eight domain theorems are preserved intact across the port. **(2)** A Lean 4 mechanization of the generic core and two ports (Fence and Reduce) onto it, with zero `sorry` and zero `axiom`, verified by `lake build WarpTypes`. **(3)** A structural-but-not-numerical refactor win: the four-domain witness total grows by 108 lines net after the factoring, yet adding a fifth domain costs one `TyTag` constructor, four dispatcher clauses, and a set of domain-specific examples — not a fresh copy of a full typing judgment. **(4)** An elaboration-layer finding about Lean 4's `cases` tactic: family-parametric rules whose conclusion dispatches through `@[reducible]` definitions require a *double* equality witness (one on the expression, one on the output type), not the single-witness pattern documented for monomorphic output types, and the caller-side discharge cost for these witnesses is linear in the product of nested-cases depth and parametric-rule count.

Figure 1 summarises the four witnesses.

<a id="fig-1"></a>

**Figure 1.** *The four mechanized witnesses of complemented typestate, as of the artifact for this paper.*

| Level | Source file | Lines | Type family used | Merge gate | Extract gate | Return type |
|-------|-------------|-------|------------------|-----------|-------------|-------------|
| GPU divergence | `Basic.lean` + `Metatheory.lean` | — | `.group` (`ActiveSet := PSet 32`) | `IsComplement` | `shuffle` requires `.group (PSet.all 32)` | `data` (broadcast value) |
| CSP collective | `Csp.lean` | 376 | `.group` (`TileSet := PSet 6`) | `IsComplement` | `collective` requires `.group (PSet.all 6)` | `data` (broadcast value) |
| Partial-write fence | `Fence.lean` | 238 | `.group` (`ByteBuf := PSet 8`) | `IsComplement` | `fence` requires `.group (PSet.all 8)` | `unit` (barrier) |
| Tree all-reduce | `Reduce.lean` | 222 | `.reduced` (`Col := PSet 4`) | `IsComplement` | `finalize` requires `.reduced (PSet.all 4)` | `data` (accumulator value) |

All four rows use `PSet.IsComplement s1 s2 parent` at the merge site and `PSet.all n` at the extract site. Rows differ in (i) which `PSet n`-indexed type family carries the participant set (the `.group` family for GPU, CSP, and Fence; the new `.reduced` family introduced by Reduce); and (ii) whether the extract gate returns a barrier (Fence) or a data payload (the other three). The same two gates govern every row.

The rest of the paper is organised as follows. §2 fixes notation and recalls the background on complemented participant sets, linear typestate, and the relevant concurrency models. §3 describes the generic mechanization spine — `Generic.lean` for the domain-independent layer and `Core.lean` for the family-parametric layer. §4 «TK §4» instantiates the spine for each of the four domains and states the per-domain soundness theorem. §5 gives the Core.lean factoring in detail, with the rule shapes and the type-family dispatchers. §6 evaluates the cost of the factoring — the caller-side discharge boilerplate, the line-count change, and the amortization argument for scaling to additional domains. §7 «TK §7» surveys related work. §8 «TK §8» concludes.

---

## 2. Background

The paper assumes familiarity with dependent type theory at the level of Lean 4 or Coq, and with linear type systems in the sense of Wadler's *Linear Types Can Change the World*. This section fixes notation and names the three concepts that support the generic framework: complemented participant sets, linear typestate, and the gate terminology.

### 2.1 Complemented participant sets

Fix a width `n : ℕ`. A *participant set* is a bitvector of width `n`, written `PSet n := BitVec n`. Each bit position corresponds to one participant in some parallel collective: a lane of a warp, a tile of a mesh, a byte of a buffer, a leaf of a reduction tree. Bit `i` set means "participant `i` is in the set." Participant sets form a bounded distributive lattice under `&&&` (bitwise and, meet) and `|||` (bitwise or, join), with `PSet.all n` as the top element (all participants present) and `0` as the bottom element.

Two participant sets `s1, s2 : PSet n` are *complementary with respect to a parent* `parent : PSet n` when they are disjoint at the bit level and their join equals the parent: `s1 &&& s2 = 0 ∧ s1 ||| s2 = parent`. This relation is stated in Lean as `PSet.IsComplement s1 s2 parent`. It is symmetric (`IsComplement s1 s2 p ↔ IsComplement s2 s1 p`) and has a canonical introduction form: given a parent `s` and a predicate `pred : PSet n`, the pair `(s &&& pred, s &&& ~~~pred)` is complementary with respect to `s`. This is the *diverge partition*, and it is the only way a well-typed program partitions a participant set.

The width `n` is a parameter of the framework. The generic layer `Generic.lean` proves the diverge partition's disjointness-and-cover property, complement symmetry, nested partition properties, and a `DivTree` lemma about arbitrary divergence trees at `∀ n : ℕ`. The domain instances each pin a concrete width: `n = 32` for a lane per bit of a GPU warp, `n = 6` for a tile per bit of a six-tile mesh, `n = 8` for a byte per bit of an eight-byte buffer, `n = 4` for a leaf per bit of a four-leaf reduction. The parameterization is load-bearing for the generic framework but invisible to the per-domain user.

### 2.2 Linear typestate

The typing judgment takes the form `Γ ⊢ e : τ ⊣ Γ'`, in which `Γ` and `Γ'` are linear contexts (ordered lists of name-to-type bindings) and `Γ'` records how `Γ` was consumed by evaluating `e`. The judgment is *linear*: a binding in `Γ` must be consumed exactly once to produce `Γ'`, and `let`-introduction imposes a freshness side condition (`ctx'.lookup name = none`) so that a name introduced by a binder does not shadow an existing binding. The framework does not use exchange freely — `letPair` imposes `name1 ≠ name2` explicitly — because mechanized linearity on `List` needs the ordering to be stable at cases-time.

Types are *typestate-indexed*: a participant set is a type index, not a runtime value. The type `.group s` means "a linear permission handle on the participant set `s`," and the type `.reduced s` (introduced in §5) means "a linear accumulator on the participant set `s`." The `s` is a proof-level index; it is not erased in the judgment but is erased at the hardware level, because the hardware is controlled by the participant set as a static mask, not by a runtime value. The rule that partitions a group,

`diverge : .group s ⊢ .pair (.group (s &&& pred)) (.group (s &&& ~~~pred))`

produces *two* linear handles whose indices are computed from the input index and a user-supplied partition predicate. Linearity ensures that the two sub-handles live in distinct sub-computations (they cannot both be held by the same continuation). The complement relation at the type level mirrors the partition at the bit level: `(s &&& pred)` and `(s &&& ~~~pred)` witness `PSet.IsComplement _ _ s`.

### 2.3 Gates

A *gate* in this paper is a side condition on a typing rule that requires a specific structural fact about the participant-set index of an operand or a conclusion. The paper uses two gates:

- The **merge gate**, `PSet.IsComplement s1 s2 parent`, is a premise of any rule that combines two sub-handles on disjoint sub-sets into one handle on the parent. It is the safety-purchase point: without it, a sub-handle could escape its sub-set, and an all-participants operation applied to the escaped handle would silently access participants that had never been merged in.

- The **extract gate**, `s = PSet.all n`, is a premise of any rule that produces a value from a handle at the full participant set. It is the reconvergence point: the handle must cover every participant, not just a sub-set of them. Divergence trees whose leaves do not cover the root are rejected by this gate on any attempt to extract a final result.

Both gates live at the *rule* level; they are not implemented as runtime checks. A well-typed program passes the gates at compile time by construction, and an unsafe program fails to type-check — typically with an error citing either an unsatisfied `IsComplement` premise or an unsatisfied `PSet.all n` premise on the conclusion.

### 2.4 SIMT divergence

A *warp* is a group of SIMT lanes that execute together. A *divergence* is a runtime control-flow construct under which different lanes take different branches of a conditional; lanes that do not take the current branch are masked off. A *shuffle* is a warp-wide value-passing operation that requires every lane of the warp to be active; calling it inside a divergent branch is undefined behaviour on real GPUs and is a known real-world bug class (CVE-2018-6243 and related). The typestate framework represents the warp as a `group s` with `s : PSet 32`, represents a divergence as a `diverge` rule application that partitions `s`, and gates `shuffle` on `s = PSet.all 32`. A shuffle inside a diverged sub-warp is untypable because the sub-warp's participant-set index is a strict sub-set of `PSet.all 32` and the extract gate is unsatisfied.

### 2.5 CSP protocol compliance

A *channel* in the sense of Hoare's CSP is a typed, synchronous point-to-point rendezvous between two parallel processes. A *collective* over a process set is a synchronous operation that requires every process in the set to participate. In mesh hardware with static topology (the running example is a 2×3 grid of forth cores with 7 bidirectional links and three-hop diameter), a collective is gated on a participant set that covers the entire mesh. The typestate framework represents the mesh as a `group s` with `s : PSet 6`, represents protocol branching as a `diverge` rule application (the participant set does not actually shrink — every core is always running — but protocol-level case splits are modelled as set partitions for the purposes of the proof), and gates `collective` on `s = PSet.all 6`. The framework is parameterised by a topology witness that constrains `send` and `recv` rules to physically adjacent tiles; the topology witness composes with but does not replace the complement gates.

### 2.6 Memory fences and tree reductions

The Fence instance models a single bulk write buffer at byte granularity. `group s : PSet 8` represents "linear permission to write the bytes in `s`," `write` threads the permission through a payload, and `fence` at `PSet.all 8` extracts a `.unit` barrier — every byte has been written before the barrier commits. The Reduce instance models a tree-structured all-reduce over four leaves. It introduces a new type-family constructor `.reduced s : PSet 4` that represents "an accumulator covering the participants `s`," with `leafReduce : .group s → .reduced s` as the cross-family coercion, `combineRed : .reduced s1 → .reduced s2 → .reduced parent` as the reduction-side merge (requiring `IsComplement s1 s2 parent`), and `finalize : .reduced (PSet.all 4) → .data` as the extract. The Reduce instance is the only one of the four that introduces a new type-family constructor at the Level 1 generic layer — the other three use the `.group` family, which is load-bearing for the §5 argument about family extension.

---

## 3. The Mechanization Spine

Two of the seven production Lean files in the `WarpTypes` namespace carry the domain-independent content that §1 advertises. `Generic.lean` proves bitwise-algebraic lemmas about `PSet n` at ∀ `n : ℕ`, without naming any expression language or typing judgment. `Core.lean` adds the family-parametric typing judgment `CoreHasType` over a combined expression language that is the union of the Fence-specific and Reduce-specific constructors. The remaining five files — `Basic.lean`, `Csp.lean`, `Fence.lean`, `Reduce.lean`, and `Metatheory.lean` — are either per-domain instances or instance-specific metatheory, and §4 «TK §4» describes them one at a time. This section describes the generic spine and the three asymmetries that a reader of §1 would otherwise have to reconstruct from file-name conventions alone.

The asymmetries are load-bearing and are named here so the per-domain descriptions in §4 do not have to apologise for them.

*First,* `Basic.lean` (the GPU instance) imports `Generic.lean` directly and does **not** go through `Core.lean`. Its mechanization pre-dates the factoring, and §6's caller-side cost analysis shows that a retrofit would purchase no structural saving — GPU's per-domain theorems share no rule bodies with Fence or Reduce, so no monomorphic rule in `Basic.lean` corresponds to an instance of a parametric rule in `Core.lean`. The retrofit is still possible; it is out of scope for *this paper*, not ruled out forever.

*Second,* `Csp.lean` also imports `Generic.lean` directly, for a different reason. Its typing judgment `CspHasType` is indexed by a `Topology n` witness that adds a physical-adjacency side condition to `send` and `recv`. The topology parameter is orthogonal to the complement-gate factoring, and an open-decision entry in the repository's `TODO.md` tracks whether a port onto an extended `Core.lean` is worth doing. A probe in `CspCoreExperiment.lean` has mechanically de-risked the port; the remaining question is prioritisation, not feasibility.

*Third,* the reduction metatheory — substitution, progress, preservation, and type safety — is stated and proved only over `Basic.lean`, not over any of Fence, Reduce, or Csp. §3.3 below justifies this coverage gap rather than hiding it.

<a id="fig-6"></a>

**Figure 6.** *The `WarpTypes` file graph.* Two files (`Generic.lean` and `Core.lean`) make up the generic spine; four are domain instances and one carries the reduction metatheory. The rightmost column pins the concrete width each file commits to; `—` means the file is parametric in `n`.

| File | Imports | Role | Width |
|------|---------|------|------:|
| `Generic.lean` | (none) | Width-parametric bitwise algebra | — |
| `Core.lean` | `Generic` | Family-parametric typing spine | — |
| `Basic.lean` | `Generic` | GPU domain instance (pre-Core) | n = 32 |
| `Csp.lean` | `Generic` | CSP mesh instance with topology parameter | n = 6 |
| `Fence.lean` | `Core` | Partial-write fence instance over the spine | n = 8 |
| `Reduce.lean` | `Core` | Tree all-reduce instance over the spine | n = 4 |
| `Metatheory.lean` | `Basic` | Substitution, progress, preservation, type safety, bug witnesses | n = 32 |

### 3.1 `Generic.lean`: the width-parametric layer

`Generic.lean` is sixty-two lines of Lean 4. It defines `PSet n := BitVec n` as an abbreviation, the four operators `all`, `none`, `Disjoint`, and `Covers`, the two predicates `IsComplement` and `IsComplementAll`, and two theorems at ∀ `n : ℕ`: `diverge_partition_generic`, which asserts that the pair `(s &&& pred, s &&& ~~~pred)` is disjoint and covers `s`; and `complement_symmetric_generic`, which asserts that `IsComplementAll` is symmetric. There is no expression language, no typing judgment, no context type, and no binding-name machinery in the file. Every domain file that needs a diverge-partition theorem re-exports the generic lemma under a domain-specific name: `Basic.lean`'s `diverge_partition`, `Fence.lean`'s `fence_diverge_partition`, `Reduce.lean`'s `reduce_diverge_partition`, `Csp.lean`'s `csp_diverge_partition`, and `Core.lean`'s `core_diverge_partition` are all one-line wrappers over `diverge_partition_generic`. A single generic theorem covers five call sites.

`Generic.lean`'s md5 is an explicit load-bearing invariant of the framework. `Core.lean`'s headnote records it as a postcondition: *"Generic.lean is the only dependency. Its md5 must remain unchanged."* A hash change would signal that the family-parametric layer had leaked domain-specific content into the generic layer — exactly the regression the factoring exists to prevent. The invariant is the narrowest possible statement of "domain-independent": the generic layer cannot grow to accommodate a domain without changing this hash, and no file in the current artifact changes it.

### 3.2 `Core.lean`: the family-parametric layer

`Core.lean` is 276 lines and imports only `Generic.lean`. It introduces four declarations that §5 describes in detail: `CoreTy n` (five constructors), `CoreExpr n` (sixteen constructors, including two merge-shaped and two extract-shaped), `TyTag` (two constructors, `deriving DecidableEq`), and the `CoreHasType` inductive with fourteen typing-rule constructors. Twelve of the fourteen typing rules are monomorphic — the value and variable rules, the partition rule `diverge`, the let-binding and pair suite, and the two domain-specific monomorphic rules `write` (Fence) and `leafReduce` (Reduce). The remaining two are the family-parametric `mergeFamily` and `finalizeFamily`, whose double-witness shape §5.4 and §5.5 dissect.

The expression language deserves one structural note here rather than in §5. `CoreExpr n` is the *union* of the Fence and Reduce expression constructors; it contains `write`, `merge`, and `fence` (from the Fence source) alongside `leafReduce`, `combineRed`, and `finalize` (from the Reduce source), and the parametric rules dispatch over the union via the `TyTag` enum. The alternative — each domain defining its own expression inductive and coercing into `Core` — does not elaborate: Lean 4 inductives are closed, and a family-parametric rule that mentions `(expr : CoreExpr n)` as a pattern variable needs a single inductive on which to state the pattern. The closed-inductive constraint is what forces the combined expression language into `Core.lean` rather than leaving it distributed across the two domain files. §5 describes the rule shapes and §6 quantifies the caller-side cost this shape imposes on downstream inversion proofs.

The family-parametric rules carry their safety side conditions *inline* as premises, not as stand-alone theorems. `mergeFamily` takes `PSet.IsComplement s1 s2 parent` as a typing-rule premise, and `finalizeFamily` takes `CoreHasType ctx e (tagToTy tag (PSet.all n)) ctx'` as a typing-rule premise — so any well-typed use has already discharged the safety cost at elaboration time. `Core.lean` contains no theorem named `merge_requires_complement` or `finalize_requires_all`; the per-domain files state analogous theorems after specialising the tag and the width, and §4 shows the resulting shapes.

### 3.3 Theorem placement across the layers

The generic-versus-domain split in the file graph has a counterpart at the theorem level. Figure 7 tallies which theorem classes live at which layer, which are stated parametrically in `n`, and which carry untypability proofs for a real-world bug class.

<a id="fig-7"></a>

**Figure 7.** *Theorem placement by layer.* "Width" is the `n` each theorem commits to; "∀ n?" is whether the theorem is stated parametrically; "Bug witness?" is whether the layer carries an untypability proof of a real-world bug class.

| Theorem class | Lives in | Width | ∀ n? | Bug witness? |
|---|---|---|:---:|:---:|
| Bitwise algebra (disjoint / cover / symmetry) | `Generic.lean` | abstract | ✓ | |
| Diverge partition (generic) | `Generic.lean` | abstract | ✓ | |
| Diverge partition (domain re-export) | per-domain files | abstract¹ | ✓ | |
| `*_requires_all` extract-gate inversion | per-domain files | concrete | | |
| Concrete complement instances (e.g. `nibble_complement`) | per-domain files | concrete | | |
| Per-domain untypability witnesses | per-domain files | concrete | | ✓ |
| Substitution, progress, preservation, type safety (GPU) | `Metatheory.lean` | n = 32² | ✓ | ✓ |
| Substitution, progress, preservation, type safety (Fence / Reduce) | `CoreMetatheory.lean` | abstract³ | ✓ | |
| CVE-class real-world bug witnesses (GPU) | `Metatheory.lean` | n = 32 | | ✓ |

¹ *`Fence`, `Reduce`, and `Csp` each state their own `diverge_partition` as a one-line re-export of the `Generic` theorem, and the re-export's conclusion is stated at ∀ `n` even though the file as a whole pins a concrete width at the domain type-alias level.*

² *`Metatheory.lean` imports `Basic.lean`, which pins `n = 32` at the file level via `ActiveSet := PSet 32`, but the bodies of `progress`, `preservation`, and `type_safety` are stated parametrically in `{n : Nat}`. The concrete width is only load-bearing for the bug-witness row below, which exercises `n = 32`-specific NVIDIA warp semantics.*

³ *`CoreMetatheory.lean` proves `progress`, `preservation`, and `type_safety` parametrically in the participant-set width `n`, and Fence and Reduce obtain the corresponding theorems at their concrete widths (`n = 8` and `n = 4` respectively) via one-line specialisations `fence_progress := CoreMetatheory.progress_closed`, `fence_preservation := CoreMetatheory.preservation`, `fence_type_safety := CoreMetatheory.type_safety`, and the same three for Reduce.*

Three rows pin concrete widths in Figure 7. Two of them are `Metatheory.lean` — the GPU-specific reduction-preservation chain for `Basic.lean`, 1019 lines, with `value_preserves_ctx`, canonical-forms lemmas `canonical_warp`, `canonical_perLane`, and `canonical_pair`, `progress`, the context-manipulation lemmas (`remove_lookup_self`, `remove_cons_ne`, `remove_comm`, and the rest), `subst_typing`, `subst_preserves_typing`, `preservation`, `type_safety`, and five bug-class untypability witnesses named after the real-world bugs they model: `bug1_cuda_samples_398`, `bug2_cccl_854`, `bug3_picongpu_2514`, `bug4_llvm_155682`, and `bug5_shuffle_after_diverge`. Their width fix to `n = 32` is load-bearing only for the bug witnesses, which exercise concrete NVIDIA warp semantics; the underlying progress and preservation proofs are width-parametric in their body. The third concrete row — Csp's per-domain inversion and bug witnesses — is unchanged from the original per-domain concrete-width pattern.

The remaining row is new in this work. `CoreMetatheory.lean` mechanizes the full reduction-preservation chain — `isValue`, `subst`, `Step`, the context-infrastructure lemmas, canonical forms, `value_preserves_ctx`, `subst_typing`, `subst_preserves_typing`, `progress` (with a `progress_closed` wrapper for the empty-context form), `preservation`, and `type_safety` — over `CoreHasType`, the family-parametric typing judgment from `Core.lean`. Fence and Reduce inherit all three corollaries (`_progress`, `_preservation`, `_type_safety`) via one-line specialisations at their concrete widths. No per-domain proof content: the corollaries are one-line `theorem` term aliases whose bodies are `CoreMetatheory.progress_closed`, `CoreMetatheory.preservation`, and `CoreMetatheory.type_safety` applied at the domain-fixed width (`ByteBuf = PSet 8` for Fence, `Col = PSet 4` for Reduce). The factoring's second-order benefit is now visible at the theorem level: the ~1000 lines of reduction-preservation infrastructure that `Metatheory.lean` spends on GPU alone is replaced, for Fence and Reduce, by ~1300 lines of shared proof in `CoreMetatheory.lean` plus six one-liner corollaries across the two domain files.

The uniform depth of coverage is new. Before this factoring the paper had to either reproduce the reduction metatheory for each of Fence, Reduce, and Csp (tripling the proof cost) or honestly disclose the asymmetry (only GPU has progress / preservation). The family-parametric `CoreHasType` judgment makes a third option available: prove reduction metatheory once over the generic core and inherit it at every domain instance that ports onto `Core.lean`. Fence and Reduce port; Csp does not (its topology parameter is orthogonal to the complement-gate factoring, as §3 discusses), and Csp therefore retains its per-domain inversion-only depth. GPU, pre-dating the factoring, keeps its full standalone `Metatheory.lean` and the five CVE-class bug witnesses that exercise `n = 32`-specific NVIDIA warp semantics.

A future revision of §5 and §6 will quote the preservation proofs for `mergeFamily` and `combineRedVal` specifically as instances of the gate doing metatheoretic work. `combineRedVal`'s preservation case unpacks `PSet.IsComplement`'s `Covers` clause (the conjunct saying `s1 ||| s2 = parent`) via `have ⟨_, hcov⟩ := hcomp; unfold PSet.Covers at hcov; rw [hcov]`, and uses that equality to rebuild the reduced form's type. The `Covers` conjunct is not only a typing precondition — it is the computational fact that the reduction step relies on. This is a subtle but load-bearing statement: the gate's two conjuncts each do their own kind of work — disjointness prevents double-writes at typing time, covering prevents partial-writes at reduction time. The dual-conjunct pattern was first established in `Metatheory.lean`'s GPU `mergeVal` preservation case, which uses the same three-tactic sequence; `CoreMetatheory.lean` inherits the pattern at the family-parametric layer and extends it uniformly to both `mergeFamily` and `combineRedVal`. The design choice that makes this visible is the canonical-form decision for `.reduced s`: under the option-(a) scheme that the project's mechanization adopts, the canonical value is `.leafReduce (.groupVal s)`, not a dedicated `.reducedVal s` constructor, and the resulting reduction rule for `combineRed` has to walk through its arguments' underlying `.groupVal` spines and OR their participant sets — exposing the `Covers` residual at the operational-semantics level. §5.6 describes the design decision in detail; the current subsection records that the decision earned its keep in `CoreMetatheory.lean`'s preservation proof.

The remaining asymmetry is real but narrower than the pre-factoring version. Csp does not port onto `Core.lean` because of its topology parameter (see §3.2 and the `CspCoreExperiment.lean` E-probe result), and it therefore carries no reduction metatheory of its own. GPU carries its full standalone `Metatheory.lean`, pre-dating the factoring; retrofitting `Basic.lean` onto `Core.lean` is a structurally possible but out-of-scope task that §8 marks as future work. The two asymmetries are no longer the headline disclosure of this subsection — they are edges of an otherwise uniform coverage table where three of the four domains have full-depth reduction metatheory (GPU directly via `Metatheory.lean`, Fence and Reduce via `CoreMetatheory.lean` specialisation).

---

## 4. Domain Instances

«TK §4»  Each of the four domain instances specialises the generic core with (i) a concrete participant-set width `n`, (ii) a set of domain-specific expression constructors and typing rules, and (iii) a set of domain theorems — including the untypability of at least one real-world bug class. This section is TK; each instance needs a subsection with rule bodies and theorem statements, and drafting the four subsections without first stabilising §3 would commit to a presentation order that §3 might want to change.

---

## 5. The Generic Core: `Core.lean`

Level 2c (Fence) and Level 2d (Reduce) arrive at very nearly the same typing judgment. Fence's `merge` rule combines two `.group` handles into one; Reduce's `combineRed` rule combines two `.reduced` handles into one. Apart from the type-family constructor (`.group` vs `.reduced`) and the expression-constructor name (`merge` vs `combineRed`), the two rules are byte-identical. Fence's `fence_requires_all` inversion is line-for-line identical to Reduce's `finalize_requires_all` under the same substitution. The same duplication pattern holds between the value-side and the extract-side: Fence's `fence` returns `.unit`, Reduce's `finalize` returns `.data`, and the remaining structure is the same.

This section factors both duplications into a single generic judgment, `CoreHasType`, whose parametric rules `mergeFamily` and `finalizeFamily` subsume the four domain-specific rules. The factoring lives in `lean/WarpTypes/Core.lean` and was introduced as Experiment D of the project.[^commits] Fence and Reduce are ported onto it; Csp is not (its topology parameter is orthogonal to the factoring — see §3) and GPU is not (its mechanization pre-dates the generic core). The four-row table of Figure 1 is satisfied by the *domain theorems*, not by the rule shapes — a rule that is monomorphic in one domain can be an instance of a parametric rule in another, and the four domains still share the same gates.

[^commits]: The Core.lean factoring and the two port commits are `bfbb4d272` → `1802039df` in the `warp-types` repository. No further commit hashes appear inline; the per-file lemma locations are sufficient for reviewer reproducibility.

### 5.1 Starting point: the duplication

The Level 2c and Level 2d files share nine typing rules (value, variable, diverge, letBind, pairVal, fstE, sndE, letPairE, letBind-body-scope) that differ only by a type-family rename. They share a tenth and eleventh rule — the merge and the extract — that also differ only by a type-family rename, plus a constructor rename on the expression side. The total shared structural material is roughly 170 lines of Lean 4 across the two files. The merge rule and the extract rule are the two load-bearing duplications: every other shared rule is syntactic, but these two carry the complement gate and the participant-set extraction gate respectively, and the duplication would repeat once per domain at any larger scale unless it is factored.

### 5.2 The family-parametric target

The target rule has the informal shape

```text
mergeFamily (F : PSet n → Ty n) (mkMerge : Expr n → Expr n → Expr n) :
    HasType ctx e1 (F s1) ctx'  →
    HasType ctx' e2 (F s2) ctx''  →
    IsComplement s1 s2 parent  →
    HasType ctx (mkMerge e1 e2) (F parent) ctx''
```

for any `PSet n`-indexed type constructor `F` and any binary expression constructor `mkMerge`. The rule is parametric in both arguments, and instantiating it at `(.group, .merge)` recovers the Fence merge while instantiating it at `(.reduced, .combineRed)` recovers the Reduce merge. A naive encoding that takes `F` and `mkMerge` as function-valued parameters of the inductive does not elaborate: the higher-order pattern unification the `cases` tactic attempts at the inversion site is stuck on the function-valued parameter, and the unification cannot be discharged by any elaborator hint. An encoding that replaces the functions with a first-order `TyTag` enum and `@[reducible]` dispatcher definitions does elaborate at construction time but not at inversion time. The working encoding uses the tag-plus-dispatcher pattern *with an explicit equality witness* on every dispatched field of the conclusion, which removes the stuck unification from the `cases`-site unifier and defers it to the post-cases rewriting layer where `simp only [dispatcher]` can reduce it eagerly.

### 5.3 Tag and dispatchers

The tag is a first-order enum (no function-valued fields, no indices):

```lean
inductive TyTag
  | group
  | reduced
  deriving DecidableEq
```

Four dispatcher definitions map the tag to its type-family constructor, its merge-expression constructor, its extract-expression constructor, and its extract result type. All four are `@[reducible]` so that `simp` can unfold them at any use site without an explicit lemma:

<a id="fig-4"></a>

**Figure 4.** *Tag dispatchers.* Four `@[reducible]` pattern-matching definitions are the family-selection machinery of the core. Adding a third type family means adding one constructor to `TyTag`, one clause to each of the four dispatchers, and one constructor to `CoreTy` — no new rules are needed in the typing judgment.

```lean
@[reducible]
def tagToTy {n : Nat} : TyTag → PSet n → CoreTy n
  | .group,   s => .group s
  | .reduced, s => .reduced s

@[reducible]
def tagToMergeExpr {n : Nat} : TyTag → CoreExpr n → CoreExpr n → CoreExpr n
  | .group,   e1, e2 => .merge e1 e2
  | .reduced, e1, e2 => .combineRed e1 e2

@[reducible]
def tagToFinalExpr {n : Nat} : TyTag → CoreExpr n → CoreExpr n
  | .group,   e => .fence e
  | .reduced, e => .finalize e

@[reducible]
def tagToFinalTy {n : Nat} : TyTag → CoreTy n
  | .group   => .unit
  | .reduced => .data
```

The `tagToFinalTy` dispatcher captures the one asymmetry across the four-witness table: Fence's extract is a barrier (`.unit`), Reduce's extract is an extracted value (`.data`), and the merge gate is agnostic to the distinction. A generic `finalize` rule that ignored the asymmetry would either commit to one of the two return types (and fail to subsume the other) or erase the return type entirely (and lose type-directed elaboration downstream). The dispatcher resolves the asymmetry at compile time; callers see no residual complexity.

### 5.4 The `mergeFamily` rule

The parametric merge rule is the central technical artefact of the factoring. It takes the tag, the context triple, the three expressions involved (two operands and the conclusion), the three participant sets (two operands and the parent), the conclusion type, and a *pair* of equality witnesses:

<a id="fig-2"></a>

**Figure 2.** *The `mergeFamily` rule.* The two equality witnesses `hExpr` and `hTy` abstract the stuck dispatches on the expression form and the conclusion type. At construction time, callers pass `rfl` for both. At inversion time, `cases` unifies trivially on the abstract `expr` and `ty` variables, and `cases tag` combined with `simp only [tagToMergeExpr, tagToTy]` reduces the witnesses so the live branches close and the dead branches contradict.

```lean
| mergeFamily (tag : TyTag)
    (ctx ctx' ctx'' : CoreCtx n) (e1 e2 expr : CoreExpr n)
    (s1 s2 parent : PSet n) (ty : CoreTy n)
    (hExpr : expr = tagToMergeExpr tag e1 e2)
    (hTy : ty = tagToTy tag parent) :
    CoreHasType ctx e1 (tagToTy tag s1) ctx' →
    CoreHasType ctx' e2 (tagToTy tag s2) ctx'' →
    PSet.IsComplement s1 s2 parent →
    CoreHasType ctx expr ty ctx''
```

The conclusion is `CoreHasType ctx expr ty ctx''`, with `expr` and `ty` as abstract variables. The witnesses pin them to the dispatched forms `tagToMergeExpr tag e1 e2` and `tagToTy tag parent`. Without the witnesses the conclusion would be `CoreHasType ctx (tagToMergeExpr tag e1 e2) (tagToTy tag parent) ctx''`, and `cases` on a hypothesis of shape `CoreHasType ctx (.merge _ _) (.group _) ctx''` would have to unify the concrete LHS against the dispatcher call on the RHS. The Lean 4 `cases`-site unifier is a single-shot higher-order unifier that does not unfold `@[reducible]` definitions during elimination; the dispatcher is opaque at that step, and the elimination fails. The witness variant moves the stuck term out of the conclusion and into a first-order hypothesis, which `cases` can process in the ordinary way.

### 5.5 The `finalizeFamily` rule

The parametric extract rule follows the same shape, but now with witnesses on the expression form and the *result type* — because `tagToFinalTy` is also a dispatched function, not a concrete constructor:

<a id="fig-3"></a>

**Figure 3.** *The `finalizeFamily` rule.* The `hTy` witness on the result type is where this rule diverges from the single-witness pattern documented for simpler parametric rules. Because `tagToFinalTy tag` is stuck at cases-site unification in the same way `tagToMergeExpr tag _ _` is, the result type needs its own abstraction-and-witness pair, not just the expression.

```lean
| finalizeFamily (tag : TyTag)
    (ctx ctx' : CoreCtx n) (e expr : CoreExpr n) (resultTy : CoreTy n)
    (hExpr : expr = tagToFinalExpr tag e)
    (hTy : resultTy = tagToFinalTy tag) :
    CoreHasType ctx e (tagToTy tag (PSet.all n)) ctx' →
    CoreHasType ctx expr resultTy ctx'
```

The extract gate `PSet.all n` lives on the premise, not the conclusion: the rule demands that the handle-side judgment be about a full participant set, and it returns whatever `tagToFinalTy tag` dispatches to. Fence instantiates the rule at `tag = .group`, in which case `tagToFinalExpr .group e` reduces to `.fence e` and `tagToFinalTy .group` reduces to `.unit`. Reduce instantiates it at `tag = .reduced`, in which case the same two dispatchers reduce to `.finalize e` and `.data`.

### 5.6 Double witnesses: why both dispatches need abstraction

A simpler version of the same pattern — used for rules whose conclusion type is concrete — carries a single equality witness on the expression form only. A generic `finalizeTagged` rule with a concrete output `.finalTy`, for example, needs only one abstract variable; the rule body reduces to `cases h; cases tag; simp only [tagToFinalExpr] at heq; injection heq; subst`, and the conclusion type does not participate in the `cases`-site obligation because it is already a constructor.

The real `finalizeFamily` rule returns `resultTy`, not a concrete constructor, because the fence/reduce return-type asymmetry forces the result type through `tagToFinalTy`. If the result type is left stuck in the conclusion, `cases` on a hypothesis of shape `CoreHasType ctx (.fence e) .unit ctx'` hits the dispatcher on the right-hand side and fails the same way the expression-side version fails — the stuck dispatch appears at the cases-site whether it is in an expression position or a type position. Pinning the result type to an abstract variable and pairing it with a witness `hTy : resultTy = tagToFinalTy tag` lets `cases` defer the dispatch to the post-cases rewriting layer, where `cases tag` makes the tag concrete and `simp only [tagToFinalTy] at hTy` reduces the witness. The same pattern applies to `mergeFamily`, whose conclusion type is `tagToTy tag parent`.

A rule of this shape is called *double-witnessed* in this paper: it has one witness per dispatched field of the conclusion. The double-witnessed shape is strictly more general than the single-witnessed shape — a rule with a concrete output is just a double-witnessed rule whose `hTy` witness happens to be a `rfl` on a constructor — but the single-witnessed shape is worth documenting separately because it avoids some caller-side complexity in the cases where it applies. For any future family-parametric rule whose output type is also dispatched, the double-witnessed shape is the default.

### 5.7 Conservative extension, split

The Core.lean factoring suggests a distinction between two kinds of "conservative extension" that a generic framework might admit. The first is *type-family extension*: adding a new `PSet n`-indexed constructor to the core type inductive, without modifying any existing typing rules. Reduce's `.reduced` constructor is this kind of extension — it was added without perturbing any existing Fence theorem, without touching the `Generic.lean` file at all, and without requiring re-verification of any existing rule. The second is *rule extension*: adding a new typing rule whose conclusion uses an existing type-family constructor. Rule extension is *not* safe in general — existing inversion theorems that case-analyse on the judgment have to cover the new rule, and the coverage is not automatic.

The framework admits type-family extension cheaply and rule extension expensively. A fifth or sixth domain that is *another* `.group`-based or `.reduced`-based instance costs a single `TyTag` constructor and a single clause per dispatcher. A fifth or sixth domain that introduces a structurally novel rule shape — say, a three-operand merge, or a rule whose side condition is not a complement witness — costs re-verification of every existing domain theorem that case-analyses on the generic judgment. The asymmetry is not a mechanization artefact; it is a load-bearing property of `cases`-based inversion in Lean 4, and §6 quantifies it.

### 5.8 Post-refactor domain views

Fence.lean and Reduce.lean are ported onto `CoreHasType` after the factoring lands. Each port keeps its concrete participant-set width (`ByteBuf = PSet 8` for Fence, `Col = PSet 4` for Reduce), its concrete complement witness (`nibble_complement` for Fence, `halfway_complement` for Reduce), and its per-domain theorem stack (diverge partition, inversion, negative instance, positive instance). The theorem statements are unchanged; the proofs gain the dead-branch discharge pattern documented in §6 for every nested `cases` level that walks through a concrete expression shape. No theorem in the post-port file is weaker than its pre-port counterpart; the test suite `lake build WarpTypes` is green on the four-port sequence. `Generic.lean` is untouched throughout: its md5 is invariant, and its proofs do not depend on any `Core.lean` or per-domain file.

---

## 6. Evaluation: The Caller-Side Cost of Family-Parametric Rules

The §5 refactor delivers the structural property its design aimed for — two parametric rules subsume the six monomorphic rules that Fence and Reduce carried between them — but the line-count consequence is counter-intuitive: the net line count across Fence, Reduce, and the new Core file is *larger* after the factoring than before it. This section explains where the extra lines come from, quantifies the cost, and gives an amortization argument for when the factoring pays for itself at a fifth or sixth domain.

### 6.1 Where the extra lines come from

The factoring removes the duplicated rule bodies from Fence.lean and Reduce.lean, yielding the expected line-count savings on the port side. At the same time, every nested `cases` walk in the existing Fence.lean and Reduce.lean inversion theorems has to discharge the `mergeFamily` and `finalizeFamily` branches manually, at every nesting level, even at levels whose hypothesis shape trivially constructor-clashes against the parametric rule's dispatched expression.

The failure mode is the elaboration-layer finding from §5.4 applied to the caller side. Consider Fence's `fence_fst_diverge_groupval_type` helper, which walks `.fst (.diverge (.groupVal s) pred)` through three levels of nested `cases`. Before the factoring, Lean 4 auto-eliminates every non-`fst` rule at the outer `cases` site via constructor clash on the expression constructor. After the factoring, two parametric rules remain: `mergeFamily` and `finalizeFamily`. Their conclusions have the shape `CoreHasType ctx expr ty ctx''` with `expr` as a free pattern variable. When `cases` is applied to a hypothesis of shape `CoreHasType [] (.fst (.diverge (.groupVal s) pred)) t ctx'`, the cases-site unifier trivially solves `expr := .fst (.diverge (.groupVal s) pred)` — the parametric rule's conclusion matches *any* expression, because the dispatched form is hidden behind the `hExpr` witness. Both parametric branches survive the elimination even though the dispatched expressions (`tagToMergeExpr tag _ _` and `tagToFinalExpr tag _`) would reduce to constructors that constructor-clash against `.fst`.

The discharge pattern is mechanical and identical at every site:

```lean
| mergeFamily tag _ _ _ _ _ _ _ _ _ _ hExpr _ _ _ _ =>
  cases tag <;> · simp only [tagToMergeExpr] at hExpr; cases hExpr
| finalizeFamily tag _ _ _ _ _ hExpr _ _ =>
  cases tag <;> · simp only [tagToFinalExpr] at hExpr; cases hExpr
```

Inside the `cases tag <;>` block, the dispatcher reduces to a concrete constructor (`.merge _ _` or `.combineRed _ _` in the merge case; `.fence _` or `.finalize _` in the extract case), and the equality witness `hExpr` is now of the form `.fst (.diverge _ _) = .merge _ _` (or similar). This is a constructor clash, and `cases hExpr` closes the branch. The pattern is three lines in Lean 4 syntax, and Fence.lean and Reduce.lean collectively carry fourteen copies of it: three nested levels in `fence_fst_diverge_groupval_type` times two parametric rules, four nested levels in `reduce_leaf_fst_diverge_type` times two parametric rules, plus a handful of cross-rule discharges in the inversion theorems. The three-line block is mechanical but it is not free: the pre-port Fence helper was four lines, and the post-port version is twenty-two lines.

### 6.2 Line count, honestly

Figure 5 tallies the pre-port and post-port line counts across the three files that the factoring touches.

<a id="fig-5"></a>

**Figure 5.** *Line count before and after the Core.lean factoring.* The net change is +108 lines — the factoring makes the total corpus larger, not smaller. The structural saving is real but it does not show up in a line count, because the caller-side discharge pattern consumes more lines than the removed rule duplications freed. *Note: Pre-port and Post-port reflect the state at the port commit. The Phase 6 metatheory extension (§3.3) subsequently added corollary specialisations and supporting imports to each domain file, bringing current line counts to 238 (`Fence.lean`) and 222 (`Reduce.lean`); the structural conclusion of this figure is unchanged.*

| File | Pre-port | Post-port | Δ | Δ rationale |
|------|---------:|----------:|---:|-------------|
| `Core.lean` (new) | — | 276 | +276 | 14 constructors, 4 dispatchers, 12 monomorphic rules, 2 parametric rules |
| `Fence.lean` | 271 | 209 | −62 | 6 rule bodies removed; 14 lines of discharge added across 4 helpers |
| `Reduce.lean` | 302 | 196 | −106 | 7 rule bodies removed; 26 lines of discharge added across 5 helpers |
| **Total** | **573** | **681** | **+108** | |

### 6.3 The refactor is structural, not numerical

A line count is a poor measurement of the refactor's value. The structural property the factoring delivers is that adding a fifth domain which reuses the core's complement gates costs one `TyTag` constructor, four dispatcher clauses, and a domain-specific example set — not a fresh copy of nine monomorphic rules, a fresh merge rule, and a fresh extract rule. The pre-factor cost of a fifth domain is roughly 170 lines of copied-and-renamed typing rules; the post-factor cost is roughly 15 lines of dispatchers plus whatever the domain's own examples demand. A rough amortization: the factoring breaks even at two domains (Fence and Reduce each save ~85 lines of rule copies, and the new Core file costs ~170 lines, net near zero) ignoring the discharge pattern, and costs +108 lines with the discharge pattern accounted for. The factoring pays for itself at N ≥ 5 domains on a lines-saved-per-discharge-cost basis; it pays for itself at N ≥ 3 domains if the only metric is "number of places a bug in the merge rule can hide," because after the factoring there is exactly one such place for any number of domains that reuse the core.

### 6.4 Why `@[reducible]` does not help

The natural first fix for the caller-side discharge cost is to make the dispatchers `@[reducible]` and hope that the `cases`-site unifier reduces the dispatcher to a concrete constructor automatically. It does not. Lean 4's `cases` tactic runs a single-shot higher-order-pattern unifier at the elimination site, and that unifier does not unfold `@[reducible]` definitions while solving its obligations. The `@[reducible]` attribute affects the *post-cases* rewriting layer, where `simp`, `injection`, `subst`, and friends do unfold reducible definitions eagerly — and that layer is exactly the one the discharge pattern uses (`simp only [tagToMergeExpr] at hExpr; cases hExpr`). The two elaboration layers are distinct; `cases`-site unification is the stricter one, and the pattern has to be written in the form that the stricter layer accepts.

An alternative fix would be a reflective or typeclass-based encoding of the parametric rules that eliminates the free expression variable in the conclusion. Such an encoding might, in principle, let the cases-site unifier see through the dispatcher to detect constructor clash automatically. The encoding is out of scope for this paper — it would require re-stating the typing judgment in a different shape, and the explicit-witness pattern has the advantage of being minimal and transparent to the reader. The discharge cost is what it is, and §6.5 gives its scaling behaviour.

### 6.5 Scaling to more domains

The discharge cost is linear in the product of nested-cases depth and parametric-rule count. Call the number of parametric rules `M` and the total nesting depth across all existing inversion theorems and helpers `D`. Adding the `M`-th parametric rule to a framework that already has `D` levels of nested cases costs `O(D)` three-line discharge blocks, and these are not optional — the port's `lake build` fails until every level is covered.

In the Core.lean case, `M = 2` and `D ≈ 7`, yielding 14 discharges, totalling ~42 lines of discharge after factoring. This is manageable and visible in the diff. A framework with `M = 3` parametric rules and `D ≈ 15` nested levels (roughly, one that serves five domains instead of two) would incur ~90 discharges and ~270 lines. At that scale, either (a) a custom `cases_family` tactic that handles the pattern in one place, or (b) a small-step inversion representation that avoids the nested-cases idiom entirely, becomes worth the investment. Neither is implemented in this work; both are listed as TK follow-ups in §8 «TK §8».

The practical consequence for a framework author contemplating a family-parametric refactor is: count the nested `cases` levels in the existing inversion theorems before you start, multiply by the number of parametric rules you intend to introduce, and use that product as the size of the boilerplate discharge you will have to carry. In the warp-types case the product was 14, the port landed in one session, and the framework was preserved. For larger products, a tactic or a small-step encoding becomes mandatory.

---

## 7. Related Work

«TK §7»  Related work is scoped out of the Phase C draft because it is venue-sensitive: PLDI's related-work expectations differ from POPL's, which differ from OOPSLA's, and the choice of venue has not yet been made. The three clusters that the final §7 will need to cover are: (1) session types and linear types for process calculi (Honda, Yoshida, Vasconcelos, Caires–Pfenning); (2) typestate and permission systems (DeLine & Fähndrich's typestate, Boyland's fractional permissions, Bornat et al.'s separation logic fractions); and (3) GPU correctness type systems (the CUDA-level warp-correctness work that precedes warp-types' Level 2a instance). The draft will need a comparison-table paragraph on where complemented typestate sits relative to each cluster.

---

## 8. Conclusion

«TK §8»  The conclusion is scoped out of the Phase C draft. It is not meaningful to draft before §3, §4, and §7 exist — the conclusion's work is to state what the paper's scope *was*, and the scope is still being established.

---

## TK List

The draft defers the following items to later phases. Each is flagged inline with `«TK §N»` at the point of first mention.

- `«TK §4»` — the four domain instances. Each needs a subsection with rule bodies, per-domain theorems, and at least one untypability witness for a real bug class. Out of scope for Phase C.
- `«TK §7»` — related work. Venue-sensitive; deferred until venue is chosen.
- `«TK §8»` — conclusion. Depends on §3, §4, and §7 being stable.
- **Abstract** — not drafted in Phase C per the user's brief; requires §1 to have converged first.
- **Follow-ups in §6.5** — a `cases_family` tactic that discharges the parametric-rule boilerplate in one place, or a small-step inversion representation that avoids the nested-cases idiom. Both are out of scope for this paper; a footnote or a §8 mention can reference them.
- **§1 "four different type systems" claim** — the introduction asserts that the four domains "have historically been analysed by four different type systems." This is approximately true but needs a sharper citation in §7 for GPU warp-correctness and the CSP session-type literature specifically. Mark as depending on §7.
