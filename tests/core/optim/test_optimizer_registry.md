# tests/core/optim/test_optimizer_registry.cpp


Suites: `Small_OptimizerRegistry`.

Tests the optimizer each row of the optimizer registry builds.

This file needs no device. Both cases run once per row of `K_OPTIMIZER_REGISTRY` on a
`HostExecutionBackend`. `tests/core/data/test_dataset_registry.cpp` covers an unknown type and an
unknown key.

## Small_OptimizerRegistry

The fixture's `make_slots` returns one slot "weight" of four elements.

## attributes_from

`attributes_from` parses JSON text as the attributes of an optimizer.

## ARowBuildsAnOptimizerOfItsType

`make_optimizer` with the row's type and empty attributes returns an optimizer whose type name is
that type and which holds the one slot.

## TheStatedRateReachesTheOptimizer

The row's factory, given `{"learning_rate": 0.25}`, returns an optimizer whose learning rate is
0.25 exactly.
