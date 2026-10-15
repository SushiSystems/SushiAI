# tests/layers/nn/test_max_pool2d.cpp


Suites: `Medium_MaxPool2d`.

Tests the 2-D max pooling layer's trace, shape inference and rejections.

This file needs no device. `NN::MaxPool2d` in `modules/layers/nn/include/SushiAI/nn/max_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter. `GraphBuilder::maxpool2d` returns the pooled value and
an INT64 argmax together; the layer returns the first.

Two `static_assert` lines at file scope require `MaxPool2d` to be a `Module` and not a
`ModuleContainer`.

## make_pool

`make_pool` builds a layer from a kernel and a stride per axis and an optional top and left
padding, which default to 0.

## traced_shape

`traced_shape` traces a layer over a batch of the given per-sample shape and returns the
result's shape.

## with_batch

`with_batch` prepends a batch extent to a per-sample [H, W, C] shape.

## ForwardHalvesTheSpatialExtentsOfAnEvenInput

A 2x2 stride-2 pool traces a [2, 8, 6, 3] input to [2, 4, 3, 3].

## ForwardTagsItsResultNHWC

The traced result is rank 4 and tagged NHWC.

## ForwardReturnsThePooledValueNotTheArgmax

The result has the input's element type, and that type is not INT64. A layer that returned
the second result of the node would hand the next layer the index buffer.

## OutputShapeOfAnUnpaddedSquareWindow

`output_shape([8, 6, 3])` of a 2x2 stride-2 pool is [4, 3, 3].

## OutputShapeAgreesWithForwardForASquareWindow

For the same pool and sample, the traced shape of a batch of two equals `output_shape()` with
the batch extent in front.

## OutputShapeAgreesWithForwardForAPaddedAnisotropicWindow

A 3x2 window with stride 2 by 1 and a top padding of 1 maps [5, 7, 4] to [3, 6, 4], and the
traced shape agrees.

## OutputShapeAgreesWithForwardWhenTheStrideLeavesARemainder

A 3x3 stride-2 window maps [8, 10, 1] to [3, 4, 1], and the traced shape agrees. Neither
extent divides evenly, so the case covers the rounding down.

## OutputShapeAgreesWithForwardForADilatedAnisotropicWindow

A 2x2 unit-stride window with dilations 2 by 3 maps [7, 9, 2] to [5, 6, 2], and the traced
shape agrees.

## ForwardRejectsAnInputThatIsNotRankFour

`forward()` throws on a [2, 16] input.

## ForwardRejectsAWindowThatDoesNotFit

`forward()` throws for a 5x5 window over a [2, 4, 4, 3] input.

## ForwardRejectsAnNCHWInput

A [2, 8, 8, 3] input is converted to NCHW, and `forward()` throws on the converted value.

## ForwardNamesTheLayoutOfAnNCHWInputNarrowerThanTheWindow

A [1, 8, 8, 2] input converted to NCHW has 2 on the axis an NHWC reader takes for the height,
which a 3x3 window does not fit. The message contains "NHWC", so the layout check runs before
the window is fitted.

## ForwardRejectsAZeroStride

`forward()` with a vertical stride of 0 throws with a message containing "MaxPool2d" and
"stride 0,2".

## ForwardRejectsANegativePadding

`forward()` with a left padding of -1 throws with a message containing "MaxPool2d" and
"padding 0,-1".

## OutputShapeRejectsAZeroStride

`output_shape()` throws when the vertical stride is 0.

## OutputShapeNamesAZeroStride

The message for the same layer contains "MaxPool2d" and "stride 0,2", so it reports the
stride and not a window that does not fit.

## OutputShapeRejectsANegativePadding

`output_shape()` with a left padding of -1 throws with a message containing "MaxPool2d" and
"padding 0,-1".

## OutputShapeRejectsAFeatureVector

`output_shape([16])` throws.

## OutputShapeRejectsABatchedShape

`output_shape([2, 8, 8, 3])` throws.

## OutputShapeRejectsAWindowTallerThanTheInput

A 5x2 window over [4, 8, 3] throws. The window fits the width, so the height check refuses.

## OutputShapeRejectsAWindowWiderThanTheInput

A 2x5 window over [8, 4, 3] throws. The window fits the height, so the width check refuses.

## TrailingRowKeepsTheExtentOfAStrideOnePool

A 2x2 pool of stride 1 over [13, 13, 4] reports [12, 12, 4]. With a trailing pad of 1 on both
axes it reports [13, 13, 4], and a trace over a batch of 2 has that shape behind the batch.

## TrailingPadAtStrideTwoGivesSevenFromThirteen

A 2x2 pool of stride 2 with a trailing pad of 1 on both axes reports [7, 7, 4] over
[13, 13, 4], and its trace over a batch of 1 is [1, 7, 7, 4].

## TrailingPadGivesTwoHundredEightFromFourHundredSixteen

The same pool reports [208, 208, 16] over [416, 416, 16].

## TrailingPadAppliesToOneAxisOnly

A 2x2 pool of stride 1 with `trailing_pad_h` 1 alone reports [3, 2, 2] over [3, 3, 2].

## TrailingPadLetsAWindowFitAShortInput

A 2x2 pool of stride 1 throws `Error` from `output_shape` over [1, 4, 3]. With a trailing pad
of 1 on both axes it reports [1, 4, 3] and traces to [1, 1, 4, 3].

## OutputShapeRejectsANegativeTrailingPad

With `trailing_pad_w` -1, `output_shape` raises an error holding "MaxPool2d",
"non-negative trailing padding" and "0,-1".

## ForwardRejectsANegativeTrailingPad

With `trailing_pad_h` -2, `forward` raises an error holding "MaxPool2d",
"non-negative trailing padding" and "-2,0", and the graph's node count does not change.

## OutputShapeRejectsATrailingPadThatOverflows

With `trailing_pad_h` at the largest `int64_t`, `output_shape` raises an error holding
"MaxPool2d" and "within 64 bits".

## OutputShapeRejectsATrailingPadThatLeavesAWindowNoTap

A 2x2 pool with `trailing_pad_h` 2 raises from `output_shape` an error holding "MaxPool2d",
"every window a tap inside the input" and "2,0".

## ForwardRejectsATrailingPadThatLeavesAWindowNoTap

A 2x2 pool of stride 2 with `trailing_pad_w` 3 raises from `forward` an error holding
"MaxPool2d", "every window a tap inside the input" and "0,3", and the graph's node count does
not change.

## TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace.
