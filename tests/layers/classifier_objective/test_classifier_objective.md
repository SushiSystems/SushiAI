# tests/layers/classifier_objective/test_classifier_objective.cpp


Suites: `Small_ClassifierObjective`, `Small_ClassifierObjectiveDType`.

Tests `classifier_objective` in
`modules/layers/classifier_objective/include/SushiAI/classifier_objective/classifier_objective.hpp`.
Each case traces a 6-to-3 `NN::Linear` with `Train::build_classifier` over a batch of 4, which
needs no device, and compares the objective the helper returns with the traced graph.

## declared_targets

Declares the leaves of an objective on a fresh graph and returns the value of its "targets" leaf.

## TheObjectiveDeclaresTheTracedTargetShape

The rebuilt objective declares a [4, 3] target leaf of the shape of the leaf `spec.targets` of the
traced graph.

## DTypeRow

Names the element type a classifier is traced in, for the rows of
`Small_ClassifierObjectiveDType`.

## Small_ClassifierObjectiveDType

Runs `TheObjectiveKeepsTheTracedElementType` once for each row: float32 and float64.

## TheObjectiveKeepsTheTracedElementType

A classifier traced in a row's element type yields an objective whose target leaf has that type,
the type of the leaf `spec.targets` of the traced graph.
