# Claim-Artifact Correspondence Audit

**Paper:** "One Gate, Four Domains: Complemented Typestate from GPU Warps to Tree All-Reduce"
**Artifact:** `lean/WarpTypes/` (14 Lean files) + Rust crates (5 published)
**Date:** 2026-04-13
**Method:** Every factual claim in paper.tex §1-§8 extracted and cross-referenced against Lean source, Rust code, and reproduction scripts.

---

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| PROVED | ~150 | Lean theorem, Rust test, or structural inspection directly verifies |
| PARTIAL | ~15 | Artifact exists but gap between claim and coverage |
| ASSERTED | ~17 | No artifact — reasoning, analogy, or inspection only |
| **Total** | **~182** | |

The artifact correspondence is strong. Most structural, theorem-existence, and line-count claims verify exactly. The highest-risk findings cluster in three areas: (1) factual errors in the abstract/conclusion, (2) unhedged novelty claims in §7, and (3) numerical imprecision in §6.

---

## CRITICAL — Factual Errors (fix before submission)

### C1. "Positive typability witnesses for all four domains" — GPU HAS NONE

**Location:** §8 line 611, abstract (implicit)
**Claim:** "positive typability witnesses for all four domains"
**Reality:** CSP has `j1_send_adjacent_typable`, Fence has `fence_after_full_write_typable`, Reduce has `finalize_tree_reduce_typable`. **GPU has no positive typability witness.** `Basic.lean` has complement instances and inversion theorems, but no theorem constructing a well-typed merge-then-shuffle derivation.
**Risk:** HIGH — a reviewer can trivially verify this by grepping for "typable" in Basic.lean and Metatheory.lean.
**Fix:** Either (a) add a GPU positive-typability theorem to Metatheory.lean (straightforward: merge even+odd → shuffle on the reconverged warp), or (b) change "all four" to "three of the four" in the conclusion.

### C2. Internal contradiction: "six monomorphic rules" vs "four domain-specific rules"

**Location:** §5 line 383 says "four domain-specific rules"; §6 line 509 says "six monomorphic rules"
**Claim:** The number of rules subsumed by the parametric factoring
**Reality:** The subsumed rules are: Fence.merge, Fence.fence, Reduce.combineRed, Reduce.finalize = 4 unique rule implementations. If you count Reduce.merge as a separate implementation (byte-identical to Fence.merge), that's 5. Neither 4 nor 6.
**Risk:** MEDIUM — a reviewer reading both sections will notice the inconsistency.
**Fix:** Pick one number and use it consistently. "Four domain-specific rules" (line 383) is the more defensible count — two merge-shaped rules and two extract-shaped rules, one pair per domain.

### C3. Bug witnesses: 5 theorems but only 2 distinct formal predicates

**Location:** §4.1 lines 310-316
**Claim:** Five untypability theorems modeling five distinct real-world bugs
**Reality:** bug1 and bug4 use identical predicate `0x00000001#32`; bug2 and bug3 use identical predicate `0x0000FFFF#32`. All five are one-line applications of `shuffle_diverged_untypable`. The five "different" bugs reduce to 2 distinct formal instances plus one synthetic (bug5 uses `ActiveSet.even`). A reviewer could argue: you have a generic untypability theorem for any pred ≠ all, and five instantiations — but only 3 distinct ones.
**Risk:** MEDIUM — doesn't invalidate the results, but a reviewer may question why 5 witnesses are highlighted when the generic `shuffle_diverged_untypable` already covers the entire class.
**Fix:** No code change needed. Add a sentence in §4.1: "All five are immediate corollaries of `shuffle_diverged_untypable`, which proves untypability for any predicate ≠ `PSet.all n`. The five instances are included to demonstrate correspondence with specific real-world bug reports, not to exercise different proof paths."

### C4. "Five CVE-class GPU bug witnesses" — bug5 is synthetic

**Location:** §8 line 611, abstract
**Claim:** "five CVE-class GPU bug witnesses"
**Reality:** Bugs 1-4 model real issue-tracker bugs. Bug 5 is explicitly labeled "A synthetic instance" in §4.1 (line 315). The conclusion and abstract don't qualify this.
**Risk:** LOW-MEDIUM — the body is honest, but a reader of only the abstract/conclusion is misled.
**Fix:** "four CVE-class GPU bug witnesses and one synthetic instance" or "five untypability witnesses (four modeling real-world bugs, one synthetic)."

---

## HIGH — Overstated Framing (soften before submission)

### H1. "Stated once... instantiated at four domains"

**Location:** §1 line 105, §8 line 609
**Claim:** "a single Lean 4 inductive typing judgment subsumes the four domain-specific judgments"
**Reality:** `CoreHasType` subsumes only Fence and Reduce. GPU (`Basic.lean`) has its own `HasType`; CSP (`Csp.lean`) has its own `CspHasType`. What spans all four is the `PSet.IsComplement` gate at the `Generic.lean` level — 62 lines of bitvector algebra, not the "family-parametric core."
**Risk:** MEDIUM — the paper body (§3) is transparent about all three asymmetries, but the intro/conclusion framing is stronger than the artifact supports.
**Fix:** §1/§8: "stated once in a generic core and instantiated at two domains (Fence and Reduce), with the same complement gate governing all four domains via the width-parametric `Generic.lean` layer."

### H2. "First type-directed approach to this bug class" — unhedged

**Location:** §7.3 line 595
**Claim:** "the GPU instance is the first type-directed approach to this bug class"
**Reality:** The preceding sentence has "to our knowledge" but this follow-on drops the hedge.
**Risk:** HIGH — priority claims without hedging are the highest-risk assertions in any paper. If a reviewer knows of any GPU type system with lane-mask tracking (even a workshop paper), this is falsified.
**Fix:** "To our knowledge, the GPU instance is the first type-directed approach to this bug class."

### H3. "GPU correctness work has not previously used type-directed safety"

**Location:** §7.4 line 601
**Claim:** Unhedged negative about the entire field
**Risk:** HIGH — broad claim, no survey cited.
**Fix:** "To our knowledge, GPU correctness work has not previously used type-directed safety or been generalised across domains."

### H4. "No MPST analog" for the complement gate

**Location:** §7.1 line 581
**Claim:** "requiring that two sub-groups cover a parent before merging — has no MPST analog"
**Risk:** MEDIUM — MPST has branching/choice constructs with merge operators. The bitvector-algebraic specificity is defensible, but a knowledgeable MPST reviewer could argue structural similarity.
**Fix:** Qualify: "has no direct MPST analog, though MPST's merge operator serves a structurally related but mechanically distinct role."

---

## MEDIUM — Numerical Imprecision (correct or qualify)

### M1. Discharge block count: "14" undercounts

**Location:** §6 line 526, §8 line 613
**Claim:** "14 discharge blocks across 7 nested-cases levels"
**Reality:** 14 in the helper functions (3 levels × 2 rules + 4 levels × 2 rules = 14). But the inversion theorems (`fence_requires_all`, `finalize_requires_all`) add ~2 more discharge blocks. Total is 16. The body (line 526) says "plus a handful of cross-rule discharges" but the conclusion quotes "14" without this qualifier.
**Fix:** Either count all discharges consistently or add "in the helper functions" after "14 discharge blocks."

### M2. "~42 lines of discharge" — wrong

**Location:** §6.5 line 563
**Claim:** "totalling ~42 lines of discharge after factoring"
**Reality:** 14 blocks × 2 lines each = 28 lines in helpers. Including inversion-theorem discharges: ~50. Neither number is 42.
**Fix:** "totalling ~28 lines of discharge in the helper functions (~50 including inversion-theorem branches)."

### M3. "Pre-port Fence helper was four lines" — undercounted

**Location:** §6.1 line 526
**Claim:** "the pre-port Fence helper was four lines"
**Reality:** Pre-port proof body spans 6-10 lines depending on counting method.
**Fix:** "the pre-port Fence helper was roughly ten lines" or verify exact count at the pre-port commit.

### M4. "~1300 lines" of CoreMetatheory — should be ~1360

**Location:** §3.3 line 276
**Claim:** "~1300 lines of shared proof in CoreMetatheory.lean"
**Reality:** `wc -l CoreMetatheory.lean` = 1362. The "~" qualifier makes this defensible but the 5% discrepancy is noticeable when Metatheory.lean's "1019 lines" is exact.
**Fix:** "~1360 lines" or verify and use exact count.

### M5. Table 4 rationale: "4 helpers" / "5 helpers" unclear

**Location:** §6.2 Table 4 caption, lines 541-542
**Claim:** Fence: "14 lines of discharge added across 4 helpers"; Reduce: "26 lines of discharge added across 5 helpers"
**Reality:** Only 2 functions in each domain file contain discharge blocks. The counts 4/5 don't match any obvious function-counting method.
**Fix:** Verify at the exact post-port commit. If the numbers refer to a broader definition of "helper" (including intermediate lemmas), clarify.

### M6. "six" vs "four" subsumed rules (see also C2)

**Location:** §6 line 509
**Fix:** Align with §5 line 383's "four domain-specific rules."

---

## LOW — Minor Imprecision (fix if convenient)

### L1. md5 hash location

§3.2 (line 234) says "Core.lean's headnote records it as a postcondition." Core.lean says "Its md5 must remain unchanged" but does NOT record the actual hash value. The hash `7f125b5f5f26122cc9e97c39522a4d03` is only in CspCoreExperiment.lean (line 70).
**Fix:** "Core.lean's headnote records the invariant; the specific hash is recorded in CspCoreExperiment.lean."

### L2. "Partial-write fencing having no prior type-system treatment"

§1 line 105. Asserted novelty claim. Plausible — byte-granularity fence-as-type-system is novel — but no systematic survey cited. Separation logic with byte-level permissions (CompCert, VST) could be argued as partially overlapping.
**Fix:** "to our knowledge, partial-write fencing has no prior type-system treatment."

### L3. Diverge partition uniqueness

§2.1 (line 155): "the only way a well-typed program partitions a participant set." True by inspection of the inductive (only `diverge` produces two sub-groups), but no inversion lemma proves uniqueness.
**Fix:** No code change needed. Consider adding: "as can be verified by inspection of the inductive: no other typing rule produces two sub-groups from a single group."

### L4. "+108 lines" from intermediate commit state

§1 and §8 quote "+108 lines" which reflects the state at the port commits, not the current files (which added metatheory corollaries). Table 4's caption discloses this but the intro/conclusion don't.
**Fix:** Add "(at the port commit; current files include subsequent metatheory additions)" or verify the current net delta.

### L5. Linearity of discharge cost — single data point

§1 Contribution (4): "the caller-side discharge cost is linear in the product of nested-cases depth and parametric-rule count." Demonstrated at M=2, D=7 (yielding 14 = 2×7). This is one data point, not a proved general property.
**Fix:** "empirically linear" or "observed to be linear."

### L6. Scaling extrapolation D definition ambiguous

§6.5 line 561 defines D as "total nesting depth across all existing inversion theorems." But the extrapolation at line 563 ("M=3, D≈15 → ~90 discharges") implies 3×15=45, not 90. The 90 only works if D counts per-domain depth across 2 domains (total 30).
**Fix:** Clarify whether D is total or per-domain, and verify the arithmetic.

---

## ASSERTED Claims — Accepted Risks

These are inherently unprovable (negative claims about the literature) but appropriately hedged or low-risk:

| # | Claim | Hedge | Risk |
|---|-------|-------|------|
| A1 | No prior type system tracks lane masks (§7.3) | "To our knowledge" | LOW (hedged) |
| A2 | Lean 4 cases-site unifier doesn't unfold @[reducible] (§5.4, §6.4) | None | LOW (artifact design proves it indirectly) |
| A3 | Session types orthogonal to participant sets (§7.1) | CSP instance as evidence | LOW |
| A4 | Rust borrow checker doesn't reason about set complements (§7.2) | None | LOW (factually correct) |
| A5 | Framework doesn't subsume session types (§7.1) | Explicit enumeration of missing features | LOW |
| A6 | Kobayashi is closest overlap (§7.1) | None | LOW (scholarly judgment) |
| A7 | Fractional permissions parallel is "suggestive" (§7.2) | Explicitly qualified | LOW |
| A8 | Amortization at N≥5 (§6.3) | "rough" | LOW |
| A9 | "Col := PSet 4 is the smallest halving width" (§4.4) | None | LOW (n=2 has trivial partition, n=4 is first non-trivial) |
| A10 | Bug witnesses model specific real-world bugs (§4.1) | Cites issue numbers | MEDIUM (correspondence not mechanized) |

---

## Recommended Fix Priority

**Before submission — DONE (commit `89518395f`):**
1. ~~Add GPU positive-typability theorem (C1)~~ — `merge_then_shuffle_typable` in Basic.lean
2. ~~Hedge the two unhedged novelty claims (H2, H3)~~ — "to our knowledge" added
3. ~~Fix "all four" now holds (C1)~~ — GPU witness added
4. ~~Qualify "five CVE-class" (C4)~~ — "four real-world, one synthetic"

**Same editing pass — DONE (commit TBD):**
5. ~~Resolve "six" vs "four" internal contradiction (C2/M6)~~ — done in prior commit
6. ~~Add sentence about bug-witness correspondence being illustrative (C3)~~ — done in prior commit
7. Soften "instantiated at four domains" framing (H1) — **DEFERRED** (body is transparent, optional polish)
8. ~~Correct discharge-block and line counts (M1, M2, M3)~~ — 18 total (14 helper + 4 inversion), ~50 lines, pre-port helper was 10 lines
9. ~~Use exact CoreMetatheory line count (M4)~~ — 1362
10. ~~Clarify D in scaling formula (L6)~~ — D redefined as helper depth, inversion blocks stated separately
11. ~~Qualify MPST analog claim (H4)~~ — "no direct MPST analog, though..."

**Optional polish (6 items):**
12. md5 location (L1)
13. Hedge "no prior type-system treatment" for fencing (L2)
14. Diverge-partition uniqueness note (L3)
15. "+108 lines" commit-state disclosure (L4)
16. "empirically linear" for discharge cost (L5)
17. Table 4 rationale helpers count (M5)
