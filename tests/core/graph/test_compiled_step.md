# tests/core/graph/test_compiled_step.cpp


Suites: `Medium_CompiledStep`, `Medium_CompiledStepAttribution`.

Tests `CompiledStep::restore_run_count`, `CompiledStep::run_at` and the node range of `CompiledStep::compile` in
`modules/core/graph/include/SushiAI/graph/compiled_step.hpp`. The fixture is an `EngineTest`.
Each case compiles one dropout of rate 0.25 over a leaf of 256 ones with a fixed site seed, so
the step's output is the mask: 4/3 where `dropout_keeps` keeps an element and 0 elsewhere,
compared within 1e-6.

## ARestoredRunCountIsTheIndexOfTheNextRun

One step runs three times. A second step, with a run count of 0, is restored to 2 and runs
once. Its run count is then 3, its output equals the first step's third output, and that
output is the mask `dropout_keeps` gives for run 2.

## RunDrawsTheMaskOfItsRunCount

A step runs three times; after run `r` its output is the mask `dropout_keeps` gives for run
`r`, so a single-machine step draws what it drew before `run_at` existed.

## RunAtDrawsTheMaskOfItsIndexAndCountsTheRun

`run_at(7)` leaves the mask of run 7 and a run count of 1. The `run()` after it draws the mask
of run 1.

## ARestoreAfterARunIsRefusedWithTheCount

After one run, `restore_run_count(4)` fails with `InvalidDescription`, "before its first run"
and "run 1", and the run count stays 1.

## ASecondRestoreIsRefused

`restore_run_count(0)` and then `restore_run_count(5)` succeed. A third call with 7 fails with
`InvalidDescription` and "run 5", and the run count stays 5.

## TheSecondRangeReadsWhatThePoolHoldsAfterTheFirst

A graph scales a leaf of four elements by 2 and the result by 3. The nodes up to the producer
of the doubled value and the nodes behind it compile into two steps over one pool, each with
at least one task. After the first step the doubled value holds 2, 4, 6, 8. The host then writes 10 into every element of it, and
after the second step the tripled value holds 30 four times.

## ARangeOutsideTheGraphIsRefused

A range that ends one node behind the last fails with `InvalidDescription` and "does not lie
in a graph", and the range [1, 0) with `InvalidDescription` and "[1, 0)".

## Attribution

The `Medium_CompiledStepAttribution` cases test the label and the node origins
`CompiledStep::compile` gives its runtime graph. Each adds a `SushiRuntime::RunObserver` to the
runtime before it compiles, runs the step once and reads the `NodeTable` of the observation. The
step is a relu, a reshape and a relu over a leaf of 8 elements; the reshape is a view and records
no task. A case that needs trailing work gives the step a recorder that copies the first
activation over the leaf.

### EveryTaskOfANodeCarriesItsOrigin

No row of the table has an empty origin, and each origin is `<node id>:<operation name>` of a
node of the graph. Both relu nodes appear, and `step.trailing` does not.

### TwoStepsHaveTwoLabels

Two steps of the same graph, so with equal node indices, are compiled with the labels
`first.step` and `second.step` and run in turn. The two tables carry their own label, their graph
ids differ and are not 0, and the origins of the two tables are equal, so only the graph keeps
the rows apart.

### TheOptimizerCarriesTheTrailingOrigin

With a trailing recorder, exactly one row carries the origin `step.trailing` and none has an
empty origin.

### AViewLeavesNoOriginOpen

No row carries the origin of the reshape node, none has an empty origin, and the second relu
carries its own origin, so the origin a view's rule opened was closed again.
