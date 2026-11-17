# tests/layers/train/test_in_memory_objective.cpp

Suites: `Small_InMemoryObjective`, `Small_InMemoryObjectiveFeed` (leaves allocated on the engine of a
`HostExecutionBackend` and bound in a `TensorPool`, so no case needs a device).

Tests `Train::InMemoryObjective`, the objective that stands in for every hand-written `Objective`
in the tests of the repository. Each case scripts one promise and reads it back through the
`Objective` interface.

## DeclareAddsTheScriptedLeavesAndRoles

A leaf scripted by id and a required role scripted by description are in the `ObjectiveLeaves`
after `declare`, and `declare_count()` is 1.

## ADeclaredInputLeafIsAnInputOfTheGraph

`script_input_leaf` of a [2, 3] float64 leaf makes `declare` ask the builder for an input of that
shape and element type, and record it under its role.

## LossReturnsTheScriptedValue, AnUnscriptedLossIsRefused and AScriptedFailureReplacesTheLoss

`loss` returns the scripted value id and counts the call. With no value scripted it fails with
`InvalidDescription` and "no loss value". A scripted failure replaces the value.

## ALossEmitterBuildsTheLoss

A scripted emitter is called with the builder, the graph, the outputs and the leaves, and its value
is the loss.

## NamesItselfAndIsFedByNoBatch and ScriptedFedByBatchIsReturned

The name is "in-memory". `is_fed_by_batch` is false until a test scripts it to true, as the base
answers.

## FeedFillsTheScriptedLeafAndSkipsTheOther

Two float32 leaves of three elements hold 7. After a feed scripted to fill `targets` with 1,
`targets` holds 1, 1, 1 and `weights` still holds 7, 7, 7: a leaf no fill names is skipped, which is
how a test builds a feed that leaves a leaf unwritten. `feed_count()` is 1.

## FeedRefusesAFillOfAnUndeclaredLeaf

A fill of `targets` when no leaf is declared fails with `InvalidDescription` and the role, from
`find_leaf_tensor`.

## AnUnscriptedFeedIsUnsupportedAndCounted

With no fill and no failure scripted, `feed` fails with `Unsupported` and the name "in-memory", as
the base does, and `feed_count()` is 1.

## AScriptedFeedFailureIsReturned

A scripted failure is the result of `feed`.

## MeasureAppendsTheScriptedMetrics and MeasureAppendsNothingWhenNothingIsScripted

Two scripted metrics, `accuracy` 0.25 and `recall` 0.75, are appended in order and
`measure_count()` is 1. With none scripted, `measure` succeeds and appends nothing.

## StatedNamesNeedNotBeTheAppendedOnes and NoNameIsStatedWhenNoneIsScripted

`metric_names` returns the names given to `script_stated_names`, whatever `measure` appends, so a
test can state a name it does not append. With none scripted it is empty, as the base is.
