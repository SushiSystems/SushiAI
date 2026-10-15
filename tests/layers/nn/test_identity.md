# tests/layers/nn/test_identity.cpp


That the Identity layer returns its input, emits nothing and holds no state.

Suites: `Small_Identity`.

Tests that the Identity layer returns its input, emits nothing and holds no state.

This file needs no device. `NN::Identity` is in `modules/layers/nn/include/SushiAI/nn/identity.hpp`.

Three `static_assert` lines at the top of the file hold that `Identity` is a `Module`, is not
a `ModuleContainer`, and that `Identity::fields()` is an empty tuple.

## ForwardReturnsItsInput

`forward` over a [2, 8, 6, 3] input returns the id it was given.

## ForwardEmitsNothing

The graph's node count and value count are the same before and after `forward`.

## OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

## TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.
