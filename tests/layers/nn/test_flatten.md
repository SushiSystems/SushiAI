# tests/layers/nn/test_flatten.cpp


The Flatten layer's trace, shape rule, rejections and parameter walk.

Suites: `Small_Flatten`, `Small_FlattenRank`, `Small_FlattenRefused`.

Tests the Flatten layer's trace, shape rule, rejections and parameter walk.

This file needs no device. `NN::Flatten` in `modules/layers/nn/include/SushiAI/nn/flatten.hpp` has no field and
no parameter. It folds every axis after the batch axis into one.

## traced_shape

`traced_shape` traces a `Flatten` over a new input of the given batched shape and returns the
result's shape.

## ARank4InputTracesToOneReshape

A [2, 4, 5, 3] input traces to a [2, 60] value. The trace adds exactly one node, a K_RESHAPE
with the input as its one operand and the result as its one output.

## Small_FlattenRank.TheInnerAxesFold

`output_shape` of the per-sample shape is the flattened shape, and the shape traced over the
batched input is the batch extent followed by the flattened extent. The rows are AnImage
([2, 4, 5, 3], per sample [4, 5, 3] to [60]), AMatrix ([6, 3, 4], per sample [3, 4] to [12]) and
AVector ([2, 7], per sample [7] to [7]).

## ARank2InputPassesThroughWithNoNode

For a [2, 7] input `forward()` returns the id it was given. The node count and the value count
are unchanged, so a flatten placed after a layer that already yields feature vectors adds
nothing to the step.

## AnNHWCInputYieldsAnNHWCResult

The input is asserted to be NHWC first, and the flattened result is NHWC too.

## Small_FlattenRefused.ForwardRefuses

`forward` returns `InvalidDescription`. The rows are AnNCHWInput (a [2, 3, 4, 5] input tagged
NCHW) and ARank1Input ([6], which has no per-sample axis).

## OutputShapeRejectsARank0Shape

`output_shape()` fails on an empty shape.

