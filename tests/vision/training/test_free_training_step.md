# tests/vision/training/test_free_training_step.cpp


Suites: `Small_FreeTrainingStep`, `Medium_FreeTrainingStep` and, in a build with the profiler at level 1 or above,
`Medium_FreeTrainingZones`.

Tests `FreeTrainingStep` of `free_training_step.hpp` on the two-level case of
`free_two_level_case.hpp`: a model of two levels with both copies, a batch of 2 and an SGD
optimizer. The cases train on the CPU SYCL device through `EngineTest`, so they are medium.
`test_detection_truth_check.cpp` tests the refusals of the truth; this file keeps what the step
does with them.

### TwoRunsOfOneSeedEndWithEqualWeights

Two cases built from one seed and run for two steps end with equal weights in every entry, and at
least one weight moved by more than 1e-9. Each run's total is finite, above 0 and equal across the
two cases. The step counts two runs.

### TheSumLeafHoldsTheStagesTargetSum

The `free.target_scores_sum` leaf holds the stage's sum, which is at least 1.

### TheProjectionLeafHoldsTheBinIndices

The projection leaf holds 0 and 1.

### AOneToOneCopyMovesEveryWeight

With a one-to-one copy, every state entry has a weight that moved after two steps.

### ABranchWeightIsReadBackFromItsLeaf

The one-to-many weight starts at 0.8. After it is set to 0.25 the two weight leaves hold 0.25 and
0.75, and the one-to-one sum leaf is at least 1.

### TheTermsOfAOneToOneRunAreFinite

Every run's total is finite.

### ARefusedTruthCountsNoRun

A truth of one image against a batch of 2 makes `run` fail and counts no run. The reason of the
refusal belongs to `check_truth_layout`.

### MakeRefusesAPoolWithoutTheInputLeaf

`make` over an empty pool, whose input leaf has no tensor, fails with `InvalidDescription` naming
"input leaf" and "bind_parameters". The case is in `Small_FreeTrainingStep`, on a
`HostExecutionBackend`, because the refusal comes before any task.

### AStepPutsReadbackAssignAndLossInItsPeriod

Only in a profiled build of level 1 or above. One step's `copy.input`, `device.run`,
`host.assign`, `host.loss` and `host.readback` zones fall in the period that opened it, in the
groups `copy`, `device` and `host`.
