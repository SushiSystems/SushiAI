# tests/layers/nn/test_leaky_relu.cpp


Suites: `Small_LeakyReLU`, `Small_LeakyReLUKept`, `Small_LeakyReLUSample`, `Small_LeakyReLUSlope`.

Tests the LeakyReLU layer's trace, shape inference and slope range.

This file needs no device. `NN::LeakyReLU` in `modules/layers/nn/include/SushiAI/nn/leaky_relu.hpp` holds a
slope and no parameter, and the slope must lie strictly inside (0, 1). The section on
`test_ops.cpp`, under LeakyReLUDifferentiatesAtItsOutput, says why a slope of 0 or below is
excluded: the gradient reads the layer's output, which is a substitution only for a positive
slope.

Three `static_assert` lines at file scope require `LeakyReLU` to be a `Module`, not to be a
`ModuleContainer`, and to list no parameter field in `fields()`.

## is_batch_of

`is_batch_of` reports whether one shape is another with a single batch axis in front. It
compares every per-sample axis, so it serves a feature vector and an image alike.

## is_slope_rejected

`is_slope_rejected` traces a layer of the given slope over a [2, 4] input and reports whether
`forward` returned `InvalidDescription`, so a failure of another code does not count as a
rejection.

## ForwardEmitsOneLeakyReLUNodeCarryingTheSlope

The trace of a layer with slope 0.2 adds exactly one node, a K_LEAKY_RELU with the input as its
one operand and the result as its one output. Attribute slot 0 of the node holds 0.2.

## Small_LeakyReLUKept.ForwardKeepsTheShapeTypeAndLayoutOfItsInput

The result has the shape, the element type and the layout of the input. The rows are
AFeatureBatch ([2, 4]), AnImageBatch ([2, 8, 6, 3], which carries the NHWC tag) and
AnNCHWImageBatch (the same input converted to NCHW, which carries the NCHW tag).

## Small_LeakyReLUSample.OutputShapeAgreesWithForward

`output_shape` of the per-sample shape is that shape, and the traced shape of the batch is the
per-sample shape with one batch axis in front. The rows are AFeatureVector ([7] in a [5, 7]
batch) and AnImage ([8, 6, 4] in a [3, 8, 6, 4] batch).

## Small_LeakyReLUSlope.ForwardRefusesTheSlope

`forward` returns InvalidDescription, and the message holds "LeakyReLU" and "slope". The rows are
the slopes outside the open range (0, 1): Zero, One, Negative (-0.01), AboveOne (1.5) and NaN.

## Small_LeakyReLUSlope.OutputShapeRefusesTheSlope

`output_shape([4])` of a layer with the slope of the row returns InvalidDescription, and the
message holds "LeakyReLU" and "slope".

## ForwardAcceptsSlopesJustInsideTheOpenRange

The smallest positive float32 and one minus the float32 epsilon are both accepted.

## ARejectedSlopeEmitsNothing

After `forward()` returns InvalidDescription for a slope of 1, the node count and the value count are unchanged.

