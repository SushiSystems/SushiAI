# tests/vision/training/test_free_training_loop.cpp

Suites: `Medium_FreeTrainingLoop`.

Tests `run_free_training` of `free_training_loop.hpp` on the two-level model of `free_two_level_case.hpp`, over a dataset of 3 batches of 2 images with the three truth roles. `detection_loop_support.hpp` holds the dataset and the epoch recorder that this file and `test_anchor_training_loop.cpp` share.

### EachUpdateTakesTheRateOfItsIteration, ACosineScheduleSetsTheRateOfEachUpdate, ARunWithoutAScheduleKeepsTheOptimizersRate

Two epochs under a step schedule give the rates 0.1, 0.05, 0.05, 0.05, 0.025, 0.025 and 6 updates, 6 optimizer steps and a run count of 6. A cosine schedule gives the rate it states at each iteration. Without a schedule every update keeps the optimizer rate of 0.01.

### TheReportHoldsTheTermsOfTheFirstAndTheLastUpdate

The first and last terms of the report equal those the callback saw for the first and the last update, and every total is finite and above 0.

### ADatasetTheStepCannotReadIsRefusedBeforeAnyUpdate, OptionsARunCannotUseAreRefused, AStartTheDatasetLacksIsRefused

A batch of 3 against a step traced for 2, a dataset of no batch, a dataset without the `boxes` role, 0 epochs, a step schedule of 2 steps and 1 scale, an epoch index of -1 and a first batch of 3 in a dataset of 3 are all refused, and the optimizer takes no step.

### ARunStartsAtTheEpochAndTheBatchItsOptionsState

With 2 epochs from epoch 3 and batch 2, the first pass reads one batch and the second reads three, and the dataset begins the epochs 3 and 4.

### TheOneToManyWeightFallsOverTheEpochsOfTheRun

Over 3 epochs the weight is the start of the objective at the first epoch, the mean of start and end at the second and the end at the third.

### ARestoredRunCountCountsOn, ACallbackThatThrowsEndsTheRunAsAFailure

A restored count of 4 reaches 7 after a pass of 3 batches. A callback that throws ends the run as a failure that holds its message, after one optimizer step.
