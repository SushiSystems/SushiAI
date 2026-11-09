# tests/vision/training/test_anchor_training_loop.cpp


Suite: `Medium_AnchorTrainingLoop`.

Tests `run_anchor_training` of `anchor_training_loop.hpp`, the loop that runs an anchor training
step over a dataset of truths. The fixture is a two-head model with its anchor step, an SGD
optimizer made at the rate `K_MADE_RATE`, and a dataset of 3 batches of 2 seeded images. The cases
train on the CPU SYCL device through `EngineTest`, so they are medium.

`test_detection_run_check.cpp` tests why a dataset or a run plan is refused. The three refusal
cases below keep what only the loop owns: a refused run takes no update.

### EachUpdateTakesTheRateOfItsIteration

A step schedule from 0.1 that halves from update 2 and again from update 5, over 2 epochs of 3
batches, gives six updates numbered 1 to 6 across the epochs 1, 1, 1, 2, 2, 2 and the batches 0, 1,
2, 0, 1, 2, at the rates 0.1, 0.05, 0.05, 0.05, 0.025 and 0.025 to 16 double unit roundoffs. The
report counts 6 updates, its last rate and the optimizer's rate are 0.025, its first and last costs
are those of the first and last update, the optimizer took 6 steps, and the step compiled once.

### ACosineScheduleSetsTheRateOfEachUpdate

A cosine schedule from 0.1 to 0.01 over 6 iterations gives each update the rate the schedule
states for its iteration, and the report's last rate is the rate at iteration 6.

### ARunWithoutAScheduleKeepsTheOptimizersRate

A run with no schedule reports the optimizer's made rate as its last rate and leaves it there.

### ADatasetTheStepCannotReadIsRefusedBeforeAnyUpdate, OptionsARunCannotUseAreRefused, AStartTheDatasetLacksIsRefused

A dataset of another batch size, a run of no epoch and a first batch past the end each make the
call fail, and after it the optimizer has taken no step and the step has counted no run.

### ARunStartsAtTheEpochAndTheBatchItsOptionsState

With a first epoch of 3 and a first batch of 2 over 2 epochs, the dataset is read at the epochs 3
and 4, the first pass reads its last batch alone and the second reads all three, so four updates
are reported, the first as epoch 1 batch 2 and the second as epoch 2 batch 0.

### ARestoredRunCountIsTheCountOfBothParts

A step restored to 4 runs reports 4 for the step and for its lead part, and after a run of 3
batches both report 7.

### ACountBelowTheCurrentRunIsRefused

After that run, restoring a count of 1 fails with `InvalidDescription` naming "before its first
run" and "run 7".

### ACallbackThatThrowsEndsTheRunAsAFailure

A callback that throws `std::runtime_error("stop here")` ends the run as a failure whose message
holds that text, after the optimizer took one step.
