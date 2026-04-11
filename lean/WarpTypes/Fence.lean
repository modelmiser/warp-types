import WarpTypes.Generic

/-
  Fence / Partial-Write Domain Extension (Level 2c — experiment B)

  Third instance of the complemented typestate framework, testing the §9 claim
  from research/complemented-typestate-framework.md:

    "the same mechanism that covers GPU divergence (Level 2a) and CSP
     protocol compliance (Level 2b) also covers fence/partial-write safety"

  Domain interpretation:
  - PSet n    : write mask over an n-byte buffer (bit i = byte i has been written)
  - group s   : linear permission to write exactly the bytes in s
  - diverge   : split a write region into disjoint sub-regions
  - merge     : combine sub-region permissions (requires IsComplement)
  - write     : perform a bulk write on a permission (threads handle unchanged)
  - fence     : barrier; requires full-buffer permission (same gate as collective)

  No topology. No receive. Writes are permission-typed, not location-typed.

  EXPERIMENT: verify that the 9 core typing rules transfer unchanged from
  Csp.lean via copy-rename, and that IsComplement is the right merge-gate
  and PSet.all is the right fence-gate for the partial-write domain.
-/

-- ============================================================================
-- ByteBuf: PSet 8 instantiation (one 8-byte word — concrete, small, testable)
-- ============================================================================

/-- An 8-bit write mask: one bit per byte of a word-sized buffer. -/
abbrev ByteBuf := PSet 8

namespace ByteBuf

def all : ByteBuf := PSet.all 8
def none : ByteBuf := PSet.none 8
def lowNibble : ByteBuf := 0x0F#8   -- bytes 0-3
def highNibble : ByteBuf := 0xF0#8  -- bytes 4-7

end ByteBuf

/-- Low and high nibbles are complements within All.
    Structural analog of leftCol_rightCol_complement (CSP) and
    even_odd_complement (GPU). -/
theorem nibble_complement :
    PSet.IsComplementAll ByteBuf.lowNibble ByteBuf.highNibble := by
  unfold PSet.IsComplementAll PSet.IsComplement
  unfold PSet.Disjoint PSet.Covers
  unfold ByteBuf.lowNibble ByteBuf.highNibble PSet.all PSet.none
  constructor <;> decide

-- ============================================================================
-- Fence Types (Level 2c)
-- ============================================================================

inductive FenceTy (n : Nat)
  | group (s : PSet n)   -- write permission for exactly these bytes
  | data                  -- payload to write
  | unit                  -- fence barrier result
  | pair (a b : FenceTy n)

-- ============================================================================
-- Fence Expressions (Level 1 core + Level 2c extensions)
-- ============================================================================

inductive FenceExpr (n : Nat)
  -- Core (Level 1 — copy-rename from CspExpr)
  | groupVal (s : PSet n)
  | dataVal
  | unitVal
  | var (name : String)
  | diverge (g : FenceExpr n) (pred : PSet n)
  | merge (g1 g2 : FenceExpr n)
  | letBind (name : String) (val body : FenceExpr n)
  | pairVal (a b : FenceExpr n)
  | fst (e : FenceExpr n)
  | snd (e : FenceExpr n)
  | letPair (e : FenceExpr n) (name1 name2 : String) (body : FenceExpr n)
  -- Level 2c: fence-specific
  | write (g payload : FenceExpr n)
  | fence (g : FenceExpr n)

-- ============================================================================
-- Fence Context (linear)
-- ============================================================================

def FenceCtx (n : Nat) := List (String × FenceTy n)

namespace FenceCtx

def lookup {n : Nat} (ctx : FenceCtx n) (name : String) : Option (FenceTy n) :=
  ctx.find? (fun p => p.1 == name) |>.map Prod.snd

def remove {n : Nat} (ctx : FenceCtx n) (name : String) : FenceCtx n :=
  ctx.filter (fun p => p.1 != name)

end FenceCtx

-- ============================================================================
-- Fence Typing Rules (no topology parameter)
-- ============================================================================

/-- Linear typing judgement for the fence domain: Γ ⊢ e : τ ⊣ Γ' -/
inductive FenceHasType {n : Nat} :
    FenceCtx n → FenceExpr n → FenceTy n → FenceCtx n → Prop

  -- ── Core rules (Level 1 — copy-rename from CspHasType, no topology) ──

  | groupVal (ctx : FenceCtx n) (s : PSet n) :
      FenceHasType ctx (.groupVal s) (.group s) ctx
  | dataVal (ctx : FenceCtx n) :
      FenceHasType ctx .dataVal .data ctx
  | unitVal (ctx : FenceCtx n) :
      FenceHasType ctx .unitVal .unit ctx
  | var (ctx : FenceCtx n) (name : String) (t : FenceTy n) :
      ctx.lookup name = some t →
      FenceHasType ctx (.var name) t (ctx.remove name)
  | diverge (ctx ctx' : FenceCtx n) (g : FenceExpr n) (s pred : PSet n) :
      FenceHasType ctx g (.group s) ctx' →
      FenceHasType ctx (.diverge g pred)
        (.pair (.group (s &&& pred)) (.group (s &&& ~~~pred))) ctx'
  | merge (ctx ctx' ctx'' : FenceCtx n) (g1 g2 : FenceExpr n)
      (s1 s2 parent : PSet n) :
      FenceHasType ctx g1 (.group s1) ctx' →
      FenceHasType ctx' g2 (.group s2) ctx'' →
      PSet.IsComplement s1 s2 parent →          -- THE gate (same as GPU, same as CSP)
      FenceHasType ctx (.merge g1 g2) (.group parent) ctx''
  | letBind (ctx ctx' ctx'' : FenceCtx n) (name : String)
      (val body : FenceExpr n) (t1 t2 : FenceTy n) :
      FenceHasType ctx val t1 ctx' →
      ctx'.lookup name = none →
      FenceHasType ((name, t1) :: ctx') body t2 ctx'' →
      ctx''.lookup name = none →
      FenceHasType ctx (.letBind name val body) t2 ctx''
  | pairVal (ctx ctx' ctx'' : FenceCtx n) (a b : FenceExpr n) (t1 t2 : FenceTy n) :
      FenceHasType ctx a t1 ctx' →
      FenceHasType ctx' b t2 ctx'' →
      FenceHasType ctx (.pairVal a b) (.pair t1 t2) ctx''
  | fstE (ctx ctx' : FenceCtx n) (e : FenceExpr n) (t1 t2 : FenceTy n) :
      FenceHasType ctx e (.pair t1 t2) ctx' →
      FenceHasType ctx (.fst e) t1 ctx'
  | sndE (ctx ctx' : FenceCtx n) (e : FenceExpr n) (t1 t2 : FenceTy n) :
      FenceHasType ctx e (.pair t1 t2) ctx' →
      FenceHasType ctx (.snd e) t2 ctx'
  | letPairE (ctx ctx' ctx'' : FenceCtx n) (e : FenceExpr n) (name1 name2 : String)
      (body : FenceExpr n) (t1 t2 t : FenceTy n) :
      FenceHasType ctx e (.pair t1 t2) ctx' →
      name1 ≠ name2 →
      ctx'.lookup name1 = none →
      ctx'.lookup name2 = none →
      FenceHasType ((name2, t2) :: (name1, t1) :: ctx') body t ctx'' →
      ctx''.lookup name1 = none →
      ctx''.lookup name2 = none →
      FenceHasType ctx (.letPair e name1 name2 body) t ctx''

  -- ── Fence-specific rules (Level 2c) ──

  /-- Write: perform a bulk write on a permission handle. Consumes payload,
      threads the group handle unchanged. No topology. No activeness check
      on the permission — any bytes in `s` are writable by definition
      (the permission *is* the write authorization). -/
  | write (ctx ctx' ctx'' : FenceCtx n) (g payload : FenceExpr n) (s : PSet n) :
      FenceHasType ctx g (.group s) ctx' →
      FenceHasType ctx' payload .data ctx'' →
      FenceHasType ctx (.write g payload) (.group s) ctx''

  /-- Fence: barrier requiring full-buffer permission. Consumes the group,
      returns unit. Structurally identical gate to CspHasType.collective —
      both require `group (PSet.all n)`. -/
  | fence (ctx ctx' : FenceCtx n) (g : FenceExpr n) :
      FenceHasType ctx g (.group (PSet.all n)) ctx' →
      FenceHasType ctx (.fence g) .unit ctx'

-- ============================================================================
-- Theorem: fence diverge partition (delegates to generic — unchanged)
-- ============================================================================

/-- The diverge partition theorem transfers to the fence domain unchanged.
    Third instance of the homomorphism (GPU, CSP, Fence all share it). -/
theorem fence_diverge_partition {n : Nat} (s pred : PSet n) :
    PSet.Disjoint (s &&& pred) (s &&& ~~~pred) ∧
    PSet.Covers (s &&& pred) (s &&& ~~~pred) s :=
  diverge_partition_generic s pred

-- ============================================================================
-- Theorem: fence requires all bytes written (parallel to shuffle/collective)
-- ============================================================================

/-- Fence typing requires the group to contain ALL bytes.
    Structurally identical to csp_collective_requires_all and the GPU
    shuffle_requires_all. -/
theorem fence_requires_all {n : Nat}
    {ctx ctx' : FenceCtx n} {g : FenceExpr n} :
    FenceHasType ctx (.fence g) .unit ctx' →
    FenceHasType ctx g (.group (PSet.all n)) ctx' := by
  intro h
  cases h with
  | fence _ _ _ hg => exact hg

-- ============================================================================
-- Helper: fst of diverge on groupVal produces the masked sub-group
-- ============================================================================

private theorem fence_fst_diverge_groupval_type {n : Nat}
    {s pred : PSet n} {t : FenceTy n} {ctx' : FenceCtx n}
    (ht : FenceHasType [] (.fst (.diverge (.groupVal s) pred)) t ctx') :
    t = .group (s &&& pred) := by
  cases ht with
  | fstE _ _ _ _ _ he =>
    cases he with
    | diverge _ _ _ _ _ hg =>
      cases hg with
      | groupVal _ _ => rfl

-- ============================================================================
-- NEGATIVE instance: fence after partial write is untypable
-- ============================================================================

/-- Fencing after writing only a proper sub-region of the buffer is untypable.

    Fence-domain analog of `collective_after_diverge_untypable` (CSP)
    and `shuffle_diverged_untypable` (GPU bugs 1-5).
    Same mechanism, same proof structure, third domain. -/
theorem fence_after_partial_write_untypable {n : Nat}
    (s pred : PSet n)
    (hne : s &&& pred ≠ PSet.all n) :
    ¬ ∃ ctx', FenceHasType []
      (.fence (.fst (.diverge (.groupVal s) pred)))
      .unit ctx' := by
  intro ⟨ctx', ht⟩
  have hg := fence_requires_all ht
  have heq := fence_fst_diverge_groupval_type hg
  simp only [FenceTy.group.injEq] at heq
  exact absurd heq.symm hne

/-- Concrete ByteBuf instance: fencing after writing only the low nibble
    is untypable. Parallel to j1_collective_after_column_split. -/
theorem bytebuf_fence_after_low_nibble_only :
    ¬ ∃ ctx', FenceHasType []
      (.fence
        (.fst (.diverge (.groupVal ByteBuf.all) ByteBuf.lowNibble)))
      .unit ctx' :=
  fence_after_partial_write_untypable ByteBuf.all ByteBuf.lowNibble (by decide)

-- ============================================================================
-- POSITIVE instance: merge two nibble permissions and fence
-- ============================================================================

/-- Writing the low nibble, writing the high nibble, merging the permissions
    via IsComplement, and fencing is well-typed. This exercises the full
    write/merge/fence round-trip and validates that IsComplement is the
    right merge-gate for the fence domain. -/
theorem fence_after_full_write_typable :
    ∃ ctx', FenceHasType ([] : FenceCtx 8)
      (.fence
        (.merge
          (.write (.groupVal ByteBuf.lowNibble) .dataVal)
          (.write (.groupVal ByteBuf.highNibble) .dataVal)))
      .unit ctx' := by
  refine ⟨[], ?_⟩
  apply FenceHasType.fence
  exact FenceHasType.merge [] [] [] _ _
    ByteBuf.lowNibble ByteBuf.highNibble (PSet.all 8)
    (FenceHasType.write [] [] [] _ _ ByteBuf.lowNibble
      (FenceHasType.groupVal [] ByteBuf.lowNibble)
      (FenceHasType.dataVal []))
    (FenceHasType.write [] [] [] _ _ ByteBuf.highNibble
      (FenceHasType.groupVal [] ByteBuf.highNibble)
      (FenceHasType.dataVal []))
    nibble_complement
