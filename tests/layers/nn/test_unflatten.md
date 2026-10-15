# tests/layers/nn/test_unflatten.cpp


Suites: `Small_Unflatten`.

Tests the trace, the shape rule and the rejected inputs of the Unflatten layer.

This file needs no device. `NN::Unflatten` in `modules/layers/nn/include/SushiAI/nn/unflatten.hpp` holds a
height, a width and a channel count and no parameter. It reshapes an
[N, height * width * channels] value to [N, height, width, channels]. Images enter a model
flat, as [N, features], so this layer is what gives a convolution its rank-4 input.

## per_sample

`per_sample` returns the shape of a graph value with the batch axis removed.

## SatisfiesTheModuleContract

Three `static_assert` lines require `Unflatten` to be a `Module`, not to be a
`ModuleContainer`, and to have no parameter walk of its own, so the walk reaches it through
`fields()`.

## TracesTheImageShape

`Unflatten{4, 5, 3}` traces a FLOAT64 [2, 60] input to a FLOAT64 [2, 4, 5, 3] value.

## TagsTheResultNHWC

The result is rank 4 and tagged NHWC, and `finalise()` accepts the graph.

## EmitsOneReshapeNode

The trace adds exactly one node. The producer of the result is a K_RESHAPE whose one operand is
the input.

## OutputShapeAgreesWithTheTraceForAnRGBImage

`output_shape()` of the input's per-sample shape equals the per-sample shape of the traced
result, and `output_shape([60])` is [4, 5, 3].

## OutputShapeAgreesWithTheTraceForAGreyImage

The same two assertions hold for `Unflatten{28, 28, 1}` over a [7, 784] input, the MNIST
geometry: `output_shape([784])` is [28, 28, 1].

## ForwardRejectsAZeroExtent

`Unflatten{4, 0, 3}` throws with a message containing "Unflatten needs positive extents". The
element-count check would also throw for this layer, in a message that does not name the
extents, so the case requires the layer's own message.

## ForwardRejectsANegativeExtent

`Unflatten{-4, -5, 3}` throws the same message. The two negative extents multiply to a
positive count of 60, which matches the input, so no later check refuses the layer by count;
`Shape`'s constructor would throw, in a message that does not name the layer.

## ForwardRejectsAnInputThatIsNotRankTwo

A [2, 60, 1, 1] input throws. Its second extent is 60 and it holds 120 elements, so only the
rank check can refuse it.

## ForwardRejectsAMismatchedElementCount

A [3, 40] input to `Unflatten{4, 5, 3}` throws with a message containing "Unflatten(4, 5, 3)"
and "inner extent is 60". `reshape` would also reject the input, in a message that does not
name the layer, so the case requires the layer's own.

## ForwardRejectsAnNCHWInput

The layout tag of a [2, 60] input is set to NCHW by hand, and `forward()` throws.

## OutputShapeRejectsAZeroExtent

`Unflatten{4, 5, 0}.output_shape([60])` throws with a message containing "Unflatten needs
positive extents".

## OutputShapeRejectsANegativeExtent

`Unflatten{-4, -5, 3}.output_shape([60])` throws the same message.

## OutputShapeRejectsAShapeThatIsNotRankOne

`output_shape([60, 1])` throws.

## OutputShapeRejectsAMismatchedElementCount

`output_shape([59])` throws.

## HasNoParameters

`parameter_count` is 0 before and after a trace and `parameter_elements` is 0. The trace
raises the value count by one, the reshaped value, so it declares no parameter.
