# tests/layers/nn/test_global_avg_pool.cpp


The shapes GlobalAvgPool traces and reports, and the inputs it refuses.

Suites: `Unit_GlobalAvgPool`.

Tests the shapes GlobalAvgPool traces and reports, and the inputs it refuses.

This file needs no device. `NN::GlobalAvgPool` in `modules/layers/nn/include/SushiAI/nn/global_avg_pool.hpp` has
no field and no parameter. It reduces the two spatial axes of an image to their mean.

## traced_shape

`traced_shape` traces the layer over an [N, H, W, C] input built from its four arguments and
returns the result's shape.

## SatisfiesTheModuleTrait

Three `static_assert` lines require `GlobalAvgPool` to be a `Module`, not to be a
`ModuleContainer`, and to have a `forward()` that is callable on a const object and returns
`Graph::ValueID`.

## ForwardReducesTheSpatialAxes

A [4, 7, 5, 16] input traces to [4, 16].

## ForwardEmitsOneSpatialMeanNode

The trace adds exactly one node, a K_SPATIAL_MEAN with the input as its one operand and the
result as its one output.

## ForwardKeepsTheNHWCLayoutTag

The result of an NHWC input is tagged NHWC, and `finalise()` accepts the graph.

## OutputShapeKeepsOnlyTheChannels

`output_shape([7, 5, 16])` is [16].

## OutputShapeAgreesWithForward

For a square [7, 7, 512] sample and an oblong [3, 9, 2] one, `output_shape()` is rank 1 and
the traced shape is the batch extent followed by its one extent.

## OutputShapeRejectsAFeatureVector

`output_shape([16])` throws.

## OutputShapeRejectsARankTwoShape

`output_shape([7, 16])` throws.

## OutputShapeRejectsAShapeWithTheBatchAxis

`output_shape([4, 7, 5, 16])` throws.

## ForwardRejectsAFeatureMatrix

`forward()` throws on a [4, 16] input.

## ForwardRejectsAnInputWithoutTheBatchAxis

`forward()` throws on a [7, 5, 16] input.

## ForwardRejectsAnNCHWInput

A [1, 4, 4, 3] input is converted to NCHW, and `forward()` throws on the converted value.

## TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace, and `parameter_elements` is 0 after it.
