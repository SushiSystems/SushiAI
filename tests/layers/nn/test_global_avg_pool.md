# tests/layers/nn/test_global_avg_pool.cpp


The shapes GlobalAvgPool traces and reports, and the inputs it refuses.

Suites: `Small_GlobalAvgPool`, `Small_GlobalAvgPoolShapeRefusal`, `Small_GlobalAvgPoolForwardRefusal`.

Tests the shapes GlobalAvgPool traces and reports, and the inputs it refuses.

This file needs no device. `NN::GlobalAvgPool` in `modules/layers/nn/include/SushiAI/nn/global_avg_pool.hpp` has
no field and no parameter. It reduces the two spatial axes of an image to their mean.

## traced_shape

`traced_shape` traces the layer over an [N, H, W, C] input built from its four arguments and
returns the result's shape.

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

## Small_GlobalAvgPoolShapeRefusal.OutputShapeRefuses

`output_shape` returns InvalidDescription for a per-sample shape that is not rank 3. The rows are
AFeatureVector ([16]), ARankTwoShape ([7, 16]) and AShapeWithTheBatchAxis ([4, 7, 5, 16]).

## Small_GlobalAvgPoolForwardRefusal.ForwardRefuses

`forward()` returns InvalidDescription. The rows are AFeatureMatrix ([4, 16]),
AnInputWithoutTheBatchAxis ([7, 5, 16]) and AnNCHWInput (a [1, 4, 4, 3] input converted to
NCHW).
