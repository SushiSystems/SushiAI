# tests/layers/nn/test_unflatten.cpp


Suites: `Small_Unflatten`, `Small_UnflattenImage`, `Small_UnflattenExtent`.

Tests the trace, the shape rule and the rejected inputs of the Unflatten layer.

This file needs no device. `NN::Unflatten` in `modules/layers/nn/include/SushiAI/nn/unflatten.hpp` holds a
height, a width and a channel count and no parameter. It reshapes an
[N, height * width * channels] value to [N, height, width, channels]. Images enter a model
flat, as [N, features], so this layer is what gives a convolution its rank-4 input.

## per_sample

`per_sample` returns the shape of a graph value with the batch axis removed.

## TracesTheImageShape

`Unflatten{4, 5, 3}` traces a FLOAT64 [2, 60] input to a FLOAT64 [2, 4, 5, 3] value.

## TagsTheResultNHWC

The result is rank 4 and tagged NHWC, and `finalise()` accepts the graph.

## Small_UnflattenImage.OutputShapeAgreesWithTheTrace

`output_shape` of the traced per-sample input equals the traced per-sample result, and
`output_shape` of the flat feature count is the image. The rows are AnRGBImage (4 by 5 by 3, a
batch of 2) and AGreyImage (28 by 28 by 1, a batch of 7).

## Small_UnflattenExtent.ForwardRejectsTheExtent

`forward()` over a [2, 60] input returns InvalidDescription with "Unflatten needs positive
extents". The comment of the case notes that the count check or the shape constructor would also
refuse the input, without naming the extents. The rows are AZeroExtent (4, 0, 3) and
NegativeExtents (-4, -5, 3).

## Small_UnflattenExtent.OutputShapeRejectsTheExtent

`output_shape([60])` of the same layers returns InvalidDescription with the same message.

## ForwardRejectsAnInputThatIsNotRankTwo

A [2, 60, 1, 1] input returns `InvalidDescription`. Its second extent is 60 and it holds 120 elements, so only the
rank check can refuse it.

## ForwardRejectsAMismatchedElementCount

A [3, 40] input to `Unflatten{4, 5, 3}` returns `InvalidDescription` with a message containing "Unflatten(4, 5, 3)"
and "inner extent is 60". `reshape` would also reject the input, in a message that does not
name the layer, so the case requires the layer's own.

## ForwardRejectsAnNCHWInput

The layout tag of a [2, 60] input is set to NCHW by hand, and `forward()` returns `InvalidDescription`.

## OutputShapeRejectsAShapeThatIsNotRankOne

`output_shape([60, 1])` returns InvalidDescription.

## OutputShapeRejectsAMismatchedElementCount

`output_shape([59])` returns InvalidDescription.

## HasNoParameters

`parameter_count` is 0 before and after a trace and `parameter_elements` is 0.
