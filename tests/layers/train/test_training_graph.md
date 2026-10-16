# tests/layers/train/test_training_graph.cpp


Suites: `Small_TrainingGraph`.

Tests the values a `StepSpec`, a `ParameterBinding` and a `TrainingGraph` start with.

This file needs no device. The header declares data only; what `build_training_graph` writes into
a `TrainingGraph` is pinned in `test_steps.cpp`. The defaults matter because a loop that reads
a value nobody set must see `K_NO_VALUE` and not a valid id.

## ADefaultSpecNamesNoValue

The inputs, targets, logits and loss of a default `StepSpec` are each `K_NO_VALUE`.

## ADefaultBindingIsAWeightWithNoValues

A default `ParameterBinding` has an empty name, no value, no gradient and the role `Weight`.

## ADefaultGraphIsEmptyAndUnscaled

A default `TrainingGraph` holds a graph with no node and no value, no parameter, no pinned
value, no loss scale, a spec with no loss and no leaf under `labels`.
