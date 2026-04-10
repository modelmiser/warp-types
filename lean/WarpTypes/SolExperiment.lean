import WarpTypes.Basic
import WarpTypes.Metatheory

/-
  Sol Experiment: Can an LLM construct Lean 4 proofs from specifications?

  Phase 1 (trivial/medium bitvector): 6 theorems, all passed in ≤3 attempts.
  Phase 2 (dependent types): 3 harder theorems requiring structural induction,
  compositionality, and operational semantics reasoning.

  All bitvector theorems are width-generic (PSet n). Concrete instances stay at n=32.
-/

-- ============================================================================
-- Phase 1: Bitvector Properties (all passed, now width-generic)
-- ============================================================================

theorem xor_self_cancel {n : Nat} (s : PSet n) :
    s ^^^ s = PSet.none n := by
  unfold PSet.none; ext i; simp

theorem xor_involution {n : Nat} (s m : PSet n) :
    (s ^^^ m) ^^^ m = s := by
  ext i; simp [BitVec.xor_assoc]

theorem xor_cancel (a b m : BitVec n) (h : a ^^^ m = b ^^^ m) :
    a = b := by
  have ha : (a ^^^ m) ^^^ m = a := by ext i; simp [BitVec.xor_assoc]
  have hb : (b ^^^ m) ^^^ m = b := by ext i; simp [BitVec.xor_assoc]
  rw [← ha, ← hb, h]

theorem xor_surjective (m j : BitVec n) :
    ∃ i, i ^^^ m = j :=
  ⟨j ^^^ m, by ext i; simp [BitVec.xor_assoc]⟩

theorem xor_permute_roundtrip (s m : BitVec n) (i : Nat) :
    (s ^^^ m ^^^ m).getLsbD i = s.getLsbD i := by
  simp [BitVec.xor_assoc]

theorem xor_is_involution (m : BitVec n) :
    ∀ s : BitVec n, (s ^^^ m) ^^^ m = s := by
  intro s; ext i; simp [BitVec.xor_assoc]

-- ============================================================================
-- Phase 2, Spec 4 (Medium-Hard): Nested complement composition
-- Now width-generic using PSet n.
-- ============================================================================

/-- If A and B are complements within S, and C and D are complements
    within A, then:
    1. C ||| D = A (from C,D complementing within A)
    2. A ||| B = S (from A,B complementing within S)
    3. C, D, B are pairwise disjoint
    4. C ||| D ||| B = S (the three sets partition S)

    This formalizes nested diverge: diverge S by P to get A,B, then
    diverge A by Q to get C,D. The three leaves C,D,B cover S. -/

-- Bitvector algebra helpers (width-generic)
private theorem bv_absorb_l {n : Nat} (x y : PSet n) : x &&& (x ||| y) = x := by
  ext i; simp_all

private theorem bv_and_assoc {n : Nat} (x y z : PSet n) : x &&& y &&& z = x &&& (y &&& z) := by
  ext i; simp_all; cases x[i] <;> simp

private theorem bv_and_zero {n : Nat} (x : PSet n) : x &&& PSet.none n = PSet.none n := by
  ext i; simp [PSet.none]

theorem nested_complement_partition {n : Nat}
    {s a b c d : PSet n}
    (hab : PSet.IsComplement a b s)
    (hcd : PSet.IsComplement c d a) :
    -- C, D, B are pairwise disjoint
    PSet.Disjoint c b ∧
    PSet.Disjoint d b ∧
    -- and they cover S
    (c ||| d ||| b = s) := by
  obtain ⟨hab_disj, hab_cov⟩ := hab
  obtain ⟨hcd_disj, hcd_cov⟩ := hcd
  unfold PSet.Disjoint PSet.Covers at *
  unfold PSet.none at *
  -- Key lemma: c ⊆ a (absorption from c ||| d = a)
  -- Absorption: c &&& a = c (since c ||| d = a)
  have hca : c &&& a = c := by rw [← hcd_cov]; exact bv_absorb_l c d
  refine ⟨?_, ?_, ?_⟩
  · -- C &&& B = 0
    calc c &&& b = (c &&& a) &&& b := by rw [hca]
      _ = c &&& (a &&& b) := bv_and_assoc c a b
      _ = c &&& (0#n) := by rw [hab_disj]
      _ = 0#n := bv_and_zero c
  · -- D &&& B = 0
    have hda : d &&& a = d := by
      rw [← hcd_cov]
      -- Goal: d &&& (c ||| d) = d
      have : d &&& (d ||| c) = d := bv_absorb_l d c
      rw [BitVec.or_comm] at this; exact this
    calc d &&& b = (d &&& a) &&& b := by rw [hda]
      _ = d &&& (a &&& b) := bv_and_assoc d a b
      _ = d &&& (0#n) := by rw [hab_disj]
      _ = 0#n := bv_and_zero d
  · -- C ||| D ||| B = S
    rw [hcd_cov, hab_cov]

-- ============================================================================
-- Phase 2, Spec 5 (Hard): Diverge tree with structural induction
-- Now width-generic using PSet n.
-- ============================================================================

/-- A diverge tree represents a sequence of diverge operations starting
    from a root active set. Leaves are un-diverged sub-warps. -/
inductive DivTree (n : Nat) : PSet n → Type
  | leaf (s : PSet n) : DivTree n s
  | branch (s : PSet n) (pred : PSet n)
      (left : DivTree n (s &&& pred))
      (right : DivTree n (s &&& ~~~pred)) : DivTree n s

/-- Collect all leaf masks from a diverge tree. -/
def DivTree.leaves {n : Nat} {s : PSet n} : DivTree n s → List (PSet n)
  | .leaf s => [s]
  | .branch _ _ l r => l.leaves ++ r.leaves

-- Auxiliary: foldl over associative op distributes over init
private theorem foldl_assoc_init {n : Nat} (f : PSet n → PSet n → PSet n)
    (h_assoc : ∀ a b c, f (f a b) c = f a (f b c))
    (a b : PSet n) (xs : List (PSet n)) :
    List.foldl f (f a b) xs = f a (List.foldl f b xs) := by
  induction xs generalizing b with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl]; rw [h_assoc a b x]; exact ih (f b x)

-- Auxiliary: foldl over monoid splits across append
private theorem foldl_or_append {n : Nat} (xs ys : List (PSet n)) :
    List.foldl (· ||| ·) (PSet.none n) (xs ++ ys) =
    List.foldl (· ||| ·) (PSet.none n) xs ||| List.foldl (· ||| ·) (PSet.none n) ys := by
  rw [List.foldl_append]
  have h_assoc : ∀ a b c : PSet n, (a ||| b) ||| c = a ||| (b ||| c) := by
    intro a b c; ext i; simp [Bool.or_assoc]
  have h_left_id : ∀ a : PSet n, PSet.none n ||| a = a := by
    intro a; ext i; simp [PSet.none]
  induction ys generalizing xs with
  | nil => ext i; simp [List.foldl, PSet.none]
  | cons y ys ih =>
    simp only [List.foldl]
    rw [foldl_assoc_init _ h_assoc _ y ys, h_left_id y]

/-- The OR of all leaves in a diverge tree equals the root.
    This is the partition property: no lanes are lost or created. -/
theorem DivTree.leaves_cover_root {n : Nat} {s : PSet n} : (t : DivTree n s) →
    t.leaves.foldl (· ||| ·) (PSet.none n) = s
  | .leaf s => by simp [leaves, List.foldl, PSet.none]
  | .branch s pred l r => by
    simp only [leaves]
    rw [foldl_or_append]
    rw [leaves_cover_root l, leaves_cover_root r]
    exact (diverge_partition s pred).2

-- ============================================================================
-- Phase 2, Spec 6 (Very Hard): Operational correspondence
-- Now width-generic: works for any n.
-- ============================================================================

/-- The term `merge (fst (diverge (warpVal s) p)) (snd (diverge (warpVal s) p))`
    reduces to `warpVal s` via the small-step semantics.

    This connects the bitvector partition theorem to the operational semantics:
    diverge produces a pair, fst/snd extract the halves, merge recombines.
    The result is the original warp — verified by the Lean kernel. -/
theorem diverge_merge_reduces_to_identity {n : Nat} (s pred : PSet n) :
    Star Step
      (.merge
        (.fst (.diverge (.warpVal s) pred))
        (.snd (.diverge (.warpVal s) pred)))
      (.warpVal s) := by
  -- Step 1: diverge (warpVal s) pred → pairVal (warpVal (s &&& pred)) (warpVal (s &&& ~~~pred))
  -- Step 2: fst (pairVal ...) → warpVal (s &&& pred)
  -- Step 3: snd (pairVal ...) → warpVal (s &&& ~~~pred)  (in right position after merge left reduces)
  -- Step 4: merge (warpVal (s &&& pred)) (warpVal (s &&& ~~~pred)) → warpVal ((s &&& pred) ||| (s &&& ~~~pred))
  -- Step 5: (s &&& pred) ||| (s &&& ~~~pred) = s (by diverge_partition)
  apply Star.step
  · -- merge (fst (diverge ...)) (snd (diverge ...))
    -- → merge (fst (pairVal ...)) (snd (diverge ...))   [diverge reduces in left/fst]
    exact Step.mergeLeft _ _ _
      (Step.fstCong _ _ (Step.divergeVal s pred))
  apply Star.step
  · -- merge (fst (pairVal (warpVal _) (warpVal _))) (snd (diverge ...))
    -- → merge (warpVal (s &&& pred)) (snd (diverge ...))   [fst reduces]
    exact Step.mergeLeft _ _ _
      (Step.fstVal (.warpVal (s &&& pred)) (.warpVal (s &&& ~~~pred)) rfl rfl)
  apply Star.step
  · -- merge (warpVal (s &&& pred)) (snd (diverge (warpVal s) pred))
    -- → merge (warpVal (s &&& pred)) (snd (pairVal ...))   [diverge reduces in right/snd]
    exact Step.mergeRight _ _ _ rfl
      (Step.sndCong _ _ (Step.divergeVal s pred))
  apply Star.step
  · -- merge (warpVal (s &&& pred)) (snd (pairVal (warpVal _) (warpVal _)))
    -- → merge (warpVal (s &&& pred)) (warpVal (s &&& ~~~pred))   [snd reduces]
    exact Step.mergeRight _ _ _ rfl
      (Step.sndVal (.warpVal (s &&& pred)) (.warpVal (s &&& ~~~pred)) rfl rfl)
  apply Star.step
  · -- merge (warpVal (s &&& pred)) (warpVal (s &&& ~~~pred))
    -- → warpVal ((s &&& pred) ||| (s &&& ~~~pred))
    exact Step.mergeVal (s &&& pred) (s &&& ~~~pred)
  -- Now goal: Star Step (warpVal ((s &&& pred) ||| (s &&& ~~~pred))) (warpVal s)
  -- Need: (s &&& pred) ||| (s &&& ~~~pred) = s
  have hcov := (diverge_partition s pred).2
  unfold PSet.Covers at hcov
  rw [hcov]
  exact Star.refl
