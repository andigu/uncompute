/-
# The certificate

590 hull divsteps at `Mstar` empty the state. This is the counterpart
of `example590` in `hddivsteps590.v`, checked here by `native_decide` on
a task-parallel evaluator and transferred to the specification along
`certEmptyPar_eq`: `processDivstepPar` computes `processDivstep` with
the per-key work of each iteration spawned as `Task`s, so compiled
evaluation uses all cores. `Task.spawn fn` is definitionally `⟨fn ()⟩`
and `Task.get` is the projection, so the parallel evaluator equals the
specification by unfolding — the parallelism affects speed only.

The check takes about 4 minutes; the result stays cached until
`Defs.lean` or this file changes.
-/
import Hddivsteps590.Defs

namespace Hddivsteps

/-! ## The task-parallel evaluator -/

/-- Stage-1 body of `processDivstep`, per entry, with the two branch-image
sets pre-computed (as tasks in the parallel version). -/
@[inline] private def joinBody (acc : State) (k : Int)
    (even odd : DSet) : State :=
  let acc := joinState (2 + k) even acc
  if 0 < k then joinState (2 - k) odd acc
  else joinState (2 + k) odd acc

/-- `processDivstep` with per-key task parallelism. -/
def processDivstepPar (M : Int) (s : State) : State :=
  let spawned := s.map fun kv =>
    (kv.1,
     Task.spawn fun _ => DSet.ofList (kv.2.map evenTrans),
     Task.spawn fun _ =>
       if 0 < kv.1 then DSet.ofList (kv.2.map oddPosTrans)
       else DSet.ofList (kv.2.map oddNonposTrans))
  let s1 : State := spawned.foldl (init := []) fun acc kt =>
    joinBody acc kt.1 kt.2.1.get kt.2.2.get
  let hulled := s1.map fun kv =>
    (kv.1, Task.spawn fun _ => (narrow M kv.2, chainHull kv.2))
  hulled.filterMap fun kt =>
    if (kt.2.get).1 then none else some (kt.1, (kt.2.get).2)

def certEmptyPar (M : Int) (n : Nat) : Bool :=
  (Nat.repeat (processDivstepPar M) n state0).isEmpty

/-- The parallel evaluator computes exactly the frozen `processDivstep`:
tasks are logically transparent (`(Task.spawn fn).get = fn ()`). -/
theorem processDivstepPar_eq : processDivstepPar = processDivstep := by
  funext M s
  simp only [processDivstepPar, processDivstep, Task.spawn,
    List.foldl_map, List.filterMap_map, joinBody]
  congr 1
  exact congrArg (fun f => List.foldl f ([] : State) s)
    (by funext x y; by_cases h : 0 < y.1 <;> simp [h])

theorem certEmptyPar_eq (M : Int) (n : Nat) :
    certEmptyPar M n = certEmpty M n := by
  rw [certEmptyPar, certEmpty, processDivstepPar_eq]

/-! ## The certificate -/

set_option maxHeartbeats 0 in
theorem example590_par : certEmptyPar Mstar 590 = true := by native_decide

theorem example590 : certEmpty Mstar 590 = true :=
  certEmptyPar_eq Mstar 590 ▸ example590_par

end Hddivsteps
