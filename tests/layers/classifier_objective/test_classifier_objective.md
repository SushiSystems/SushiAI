# tests/layers/classifier_objective/test_classifier_objective.cpp


Suites: `Small_ClassifierObjective`.

Tests `classifier_objective` in
`modules/layers/classifier_objective/include/SushiAI/classifier_objective/classifier_objective.hpp`.
Each case traces a 6-to-3 `NN::Linear` with `Train::build_classifier` over a batch of 4, which
needs no device, and compares the objective the helper returns with the traced graph.

## declared_targets

Declares the leaves of an objective on a fresh graph and returns the value of its "targets" leaf.

## TheObjectiveDeclaresTheTracedTargetLeaf

The rebuilt objective declares a [4, 3] target leaf of the shape and the element type of the
leaf `spec.targets` of the traced graph, and its name is "cross_entropy".

## TheObjectiveKeepsTheTracedElementType

A classifier traced in float64 yields an objective whose target leaf is float64.
