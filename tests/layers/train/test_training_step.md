# tests/layers/train/test_training_step.cpp


Suites: `Medium_TrainingStep`.

Tests how `Train::TrainingStep` places the optimizer, splits at a host stage and counts runs.

Every case compiles a step on the CPU SYCL device through `EngineTest`, so the file is medium.
The refusals of a host stage the graph cannot be cut at, and the refusal of host work by a step
with no host stage, are pinned in `tests/vision/training/test_anchor_training_step.cpp`.

## ClassifierParts

`ClassifierParts` holds a Linear, ReLU, Linear classifier of 6 features, 8 hidden units and 3
classes over a batch of 4, its training graph from `build_classifier`, its state, the pool and
the parameter slots. `prepare` traces, binds and initialises it and sets `ready`.
`host_stage_options` returns options whose host stage reads the logits.

Each case keeps the parts in its body and declares the optimizer and the step after them, so the
step is destroyed first and the engine, which the fixture resets after the body, is destroyed
last.

## InPlanRefusesAnOptimizerThatIsNotReplayInvariant

An AdamW with separate update kernels is not replay invariant. Under the default placement,
`InPlan`, `make` returns InvalidDescription naming "cannot be compiled into the step" and
`PerStep`.

## PerStepKeepsTheOptimizerOutOfThePlan

The same optimizer under `PerStep` is accepted. `optimizer_is_in_plan()` is false, `optimizer()`
is the optimizer passed, and two runs leave a run count of 2.

## TheDefaultStepHoldsTheOptimizerInOnePart

A fused SGD under the default options is recorded into the plan, the step has no host stage and
no lead part, and one run leaves a run count of 1.

## AHostStageRunsTheHostWorkBetweenTheTwoParts

With a host stage that reads the logits the step has a lead part. `run` with host work calls the
work once; at that moment the lead part has run once and the rest has not run. After the call
the rest has run once.

## RestoringTheRunCountSetsEveryPart

On a fresh step with a host stage, `restore_run_count(5)` sets the run count of the lead part
and of the rest to 5.
