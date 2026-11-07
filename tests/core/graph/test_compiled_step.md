# tests/core/graph/test_compiled_step.cpp


Suites: `Small_CompiledStepChecks`, `Medium_CompiledStep`,
`Medium_CompiledStepAttribution`.

Tests `CompiledStep::restore_run_count`, `CompiledStep::run_at` and the node range of `CompiledStep::compile` in
`modules/core/graph/include/SushiAI/graph/compiled_step.hpp`, and `Graph::check_compilable`. The `Small_` cases need no device. The
`Medium_` fixture is an `EngineTest`, and each case passes its `backend()` to `CompiledStep::compile`.
Each case compiles one dropout of rate 0.25 over a leaf of 256 ones with a fixed site seed, so
the step's output is the mask: 4/3 where the published rule keeps an element and 0 elsewhere. The
test derives the rule itself, from SplitMix64 and the top 53 bits of a draw, and compares within
`K_TOLERANCE`, 32 unit roundoffs of float, the smallest power of two that covers 1e-6 for a
FLOAT32 scale read back as a double.

## ARestoredRunCountIsTheIndexOfTheNextRun

One step runs three times. A second step, with a run count of 0, is restored to 2 and runs
once. Its run count is then 3, its output equals the first step's third output, and that
output is the mask the rule gives for run 2.

## RunDrawsTheMaskOfItsRunCount

A step runs three times; after run `r` its output is the mask the rule gives for run
`r`, so a single-machine step draws what it drew before `run_at` existed.

## RunAtDrawsTheMaskOfItsIndexAndCountsTheRun

`run_at(7)` leaves the mask of run 7 and a run count of 1. The `run()` after it draws the mask
of run 1.

## ARestoreAfterARunIsRefusedWithTheCount

After one run, `restore_run_count(4)` fails with `InvalidDescription`, "before its first run"
and "run 1", and the run count stays 1.

## AGraphWithADroppedRefusalIsNotCompilable

A graph with a scaled leaf passes `check_compilable`. After a `reshape` to an element count the
value does not have is refused and its result dropped, `check_compilable` fails with
`InvalidDescription` and "refused operation".

## TheSecondRangeReadsWhatThePoolHoldsAfterTheFirst

A graph scales a leaf of four elements by 2 and the result by 3. The nodes up to the producer
of the doubled value and the nodes behind it compile into two steps over one pool, each with
at least one task. After the first step the doubled value holds 2, 4, 6, 8. The host then writes 10 into every element of it, and
after the second step the tripled value holds 30 four times.

## ARangeOutsideTheGraphIsRefused

A range that ends one node behind the last fails with `InvalidDescription` and "does not lie
in a graph", and the range [1, 0) with `InvalidDescription` and "[1, 0)".

## AGraphWithADroppedRefusalIsNotCompiled

A `reshape` to an element count the value does not have is refused and its result dropped. The
graph's `has_refusal()` is true, and `compile` fails with `InvalidDescription` and "refused
operation".

## Attribution

The `Medium_CompiledStepAttribution` cases test the label and the node origins
`CompiledStep::compile` gives its runtime graph. Each adds a `SushiRuntime::RunObserver` to the
runtime before it compiles, runs the step once and reads the `NodeTable` of the observation. The
step is a relu, a reshape and a relu over a leaf of 8 elements, on a pool that adopted the
graph's memory plan with reuse off; the plan makes the reshape a view, so it records no task. A
case that needs trailing work gives the step a recorder that copies the first activation over
the leaf; the activation is materialised before the recorder is bound to it.

### EveryTaskOfANodeCarriesItsOrigin

No row of the table has an empty origin, and each origin is `<node id>:<operation name>` of a
node of the graph. Both relu nodes appear, and `step.trailing` does not.

### TwoStepsHaveTwoLabels

Two steps of the same graph, so with equal node indices, are compiled with the labels
`first.step` and `second.step` and run in turn. The two tables carry their own label.

### TwoStepsHaveTwoGraphIds

The graph ids of the two runs differ and the first is not 0, so the graph keeps the rows apart.

### TwoStepsOfEqualGraphsCarryEqualOrigins

The origins of the two tables are equal, so only the graph id, not the origins, tells the steps
apart.

### TheOptimizerCarriesTheTrailingOrigin

With a trailing recorder, exactly one row carries the origin `step.trailing` and none has an
empty origin.

### AViewLeavesNoOriginOpen

No row carries the origin of the reshape node, none has an empty origin, and the second relu
carries its own origin, so the origin a view's rule opened was closed again.
