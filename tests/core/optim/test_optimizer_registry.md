# tests/core/optim/test_optimizer_registry.cpp


Suites: `Small_OptimizerRegistry`, `Medium_OptimizerRegistry`.

Tests the rows of the optimizer registry and the optimizer each row builds.

The table case needs no device. Building an optimizer allocates its state through an engine, so
the other cases are medium. `tests/core/data/test_dataset_registry.cpp` covers an unknown type
and an unknown key.

## Medium_OptimizerRegistry

The fixture's `make_slots` returns one slot "weight" of four elements.

## attributes_from

`attributes_from` parses JSON text as the attributes of an optimizer.

## EachRowNamesItsTypeAndFactory

`K_OPTIMIZER_REGISTRY` has two rows: `K_ADAMW_TYPE` with `make_adamw`, then `K_SGD_TYPE` with
`make_sgd`.

## EveryTypeBuildsAnOptimizerOfThatType

For each row, `make_optimizer` with the row's type and empty attributes returns an optimizer
whose type name is that type and which holds the one slot.

## TheStatedRateReachesTheOptimizer

Each row's factory, given `{"learning_rate": 0.25}`, returns an optimizer whose learning rate
is 0.25 exactly.
