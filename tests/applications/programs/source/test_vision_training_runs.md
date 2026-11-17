# tests/applications/programs/source/test_vision_training_runs.cpp

Suite: `Medium_VisionTrainingRuns`, in `sushiai_programs_tests`.

Tests the two run rows of `vision_training_runs.hpp` over a `PreparedRun` of the `tiny_yolo`
training file (`prepared_run_harness.hpp`): the setup, the optimizer and the starting state the
program builds, then the row alone. The fixture is anchor based, so `run_anchor` runs over it and
`run_anchor_free` is tested on its refusal; the example models of `applications/examples` run an
anchor-free training.

## TheAnchorRowTakesOneUpdateForEveryBatchOfEveryEpoch

One epoch of two batches leaves the optimizer at 2 steps.

## TheAnchorRowCompilesItsTrainingStepOnce

The summary reports one compile of the training step.

## TheAnchorRowReportsWhereTheRunStands

The summary carries the seed of the file, 2 runs of the step, and the start of epoch 1 at batch 0.

## TheAnchorRowPrintsTheProgressLineOfItsFirstEpoch

The row prints the line of epoch 1 of 1 with its loss.

## TheAnchorRowStartsItsBiasesFromTheModelFileWhenStartingFromNothing, TheAnchorRowLeavesItsBiasesAloneWhenWeightsAreRead

The bias start reads the model file. With a model path that names no file, a run that reads neither
a checkpoint nor weights fails as `FileNotFound` naming the file, and a run that reads weights does
not touch the file and succeeds.

## TheAnchorRowRefusesAStartBatchTheDataLacks

A start at batch 99 of an epoch of two batches fails as `Incompatible` naming "batch 99" and takes
no update.

## TheAnchorFreeRowRefusesAModelWhoseTaskIsAnchorBased

The anchor-free row over the anchor fixture fails, prints nothing and takes no update.
