/-
Axiom scan: which axioms does every theorem of the library depend on?

  cd lean && lake build && lake env lean AxiomScan.lean

Walks every theorem constant declared in a module that `WarpTypes.lean`
imports (module name `WarpTypes` or `WarpTypes.*`), runs `Lean.collectAxioms`
on each, and prints the count per module, the total, and the axioms found.
It FAILS (non-zero exit) if any theorem depends on an axiom outside Lean's
standard three — `propext`, `Classical.choice`, `Quot.sound` — which is also
how a `sorry` shows up (`sorryAx`).

Not part of the library: nothing imports this file. Modules that
`WarpTypes.lean` does not import (the `*Experiment` files built on their own)
are outside the scan. The committed output is `axiom-scan.txt`;
`scripts/check-axiom-scan.sh` regenerates it and compares.
-/
import WarpTypes
import Lean

open Lean Elab Command

def allowedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

run_cmd do
  let env ← getEnv
  let mods := env.header.moduleNames
  -- theorem constants of WarpTypes modules, with their module
  let thms : Array (Name × Name) := env.constants.map₁.fold (init := #[]) fun acc name info =>
    match info, env.getModuleIdxFor? name with
    | .thmInfo _, some idx =>
      let m := mods[idx.toNat]!
      if (`WarpTypes).isPrefixOf m then acc.push (m, name) else acc
    | _, _ => acc
  let mut perModule : Std.HashMap Name Nat := {}
  let mut uses : Std.HashMap Name Nat := {}
  let mut axiomFree := 0
  let mut bad : Array (Name × Array Name) := #[]
  for (m, name) in thms do
    perModule := perModule.insert m (perModule.getD m 0 + 1)
    let axs ← liftCoreM (collectAxioms name)
    if axs.isEmpty then axiomFree := axiomFree + 1
    for a in axs do
      uses := uses.insert a (uses.getD a 0 + 1)
    let extra := axs.filter (fun a => !allowedAxioms.contains a)
    unless extra.isEmpty do bad := bad.push (name, extra)
  let sortByName (xs : Array (Name × Nat)) := xs.qsort (fun a b => a.1.toString < b.1.toString)
  IO.println s!"axiom scan — Lean {Lean.versionString}"
  IO.println "theorem constants per module:"
  for (m, n) in sortByName perModule.toArray do
    IO.println s!"  {m}: {n}"
  IO.println s!"total theorem constants: {thms.size}"
  IO.println s!"depending on no axiom: {axiomFree}"
  IO.println "axioms used (number of theorem constants depending on each):"
  for (a, n) in sortByName uses.toArray do
    IO.println s!"  {a}: {n}"
  if bad.isEmpty then
    IO.println "RESULT: every theorem constant depends only on propext, Classical.choice, Quot.sound"
  else
    for (name, extra) in bad do
      IO.println s!"  OUTSIDE THE STANDARD THREE: {name} uses {extra}"
    throwError "axiom scan FAILED: {bad.size} theorem constant(s) depend on other axioms"
