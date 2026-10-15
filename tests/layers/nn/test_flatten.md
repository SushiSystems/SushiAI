# tests/layers/nn/test_flatten.cpp


The Flatten layer's trace, shape rule, rejections and parameter walk.

Suites: `Unit_Flatten`.

Tests the Flatten layer's trace, shape rule, rejections and parameter walk.

This file needs no device. `NN::Flatten` in `modules/layers/nn/include/SushiAI/nn/flatten.hpp` has no field and
no parameter. It folds every axis after the batch axis into one.

## traced_shape

`traced_shape` traces a `Flatten` over a new input of the given batched shape and returns the
result's shape.

## SatisfiesTheModuleTrait

Two `static_assert` lines require `Flatten` to be a `Module` and not a `ModuleContainer`.

## ARank4InputTracesToOneReshape

A [2, 4, 5, 3] input traces to a [2, 60] value. The trace adds exactly one node, a K_RESHAPE
with the input as its one operand and the result as its one output.

## ARank3InputFoldsItsTwoInnerAxes

A [2, 3, 4] input traces to [2, 12].

## ARank2InputPassesThroughWithNoNode

For a [2, 7] input `forward()` returns the id it was given. The node count and the value count
are unchanged, so a flatten placed after a layer that already yields feature vectors adds
nothing to the step.

## AnNHWCInputYieldsAnNHWCResult

The input is asserted to be NHWC first, and the flattened result is NHWC too.

## ForwardRejectsAnNCHWInput

The layout tag of a [2, 3, 4, 5] input is set to NCHW by hand, and `forward()` throws.

## OutputShapeAgreesWithTheTracedShape

`output_shape()` maps [4, 5, 3] to [60], [3, 4] to [12] and [7] to [7]. For each, the traced
shape of a batch equals the batch extent followed by that one extent.

## ForwardRejectsARank1Input

`forward()` throws on a [6] input, which has a batch axis and no per-sample axis.

## OutputShapeRejectsARank0Shape

`output_shape()` throws on an empty shape.

## TheParameterWalkVisitsNothing

`parameter_count` is 0 before a trace. After one the walk visits nothing and
`parameter_elements` is 0.
