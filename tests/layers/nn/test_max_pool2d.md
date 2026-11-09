# tests/layers/nn/test_max_pool2d.cpp


Suites: `Small_MaxPool2d`, `Small_MaxPool2dExtent`, `Small_MaxPool2dForwardRefusal`,
`Small_MaxPool2dShapeRefusal`, `Medium_MaxPoolTrailing`.

Tests the 2-D max pooling layer's trace, shape inference and rejections.

The small cases need no device; the trailing-window cases run on an engine. `NN::MaxPool2d` in `modules/layers/nn/include/SushiAI/nn/max_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter. `GraphBuilder::maxpool2d` returns the pooled value and
an INT64 argmax together; the layer returns the first.

Two `static_assert` lines at file scope require `MaxPool2d` to be a `Module` and not a
`ModuleContainer`.

## with_trailing and with_dilation

Each returns a copy of a layer with a trailing pad of rows and columns, or a dilation per axis.

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

## Small_MaxPool2dExtent.OutputShapeAgreesWithForward

One row per window geometry. `output_shape(sample)` returns the extent of the row, and the shape of
a trace over a batch of that sample is the extent with the batch in front. Ten rows:

| Row | Window | Sample | Extent |
| --- | --- | --- | --- |
| ASquareWindow | 2x2, stride 2 | [8, 6, 3] | [4, 3, 3] |
| APaddedAnisotropicWindow | 3x2, stride 2 by 1, top pad 1 | [5, 7, 4] | [3, 6, 4] |
| AStrideThatLeavesARemainder | 3x3, stride 2 | [8, 10, 1] | [3, 4, 1] |
| ADilatedAnisotropicWindow | 2x2, stride 1, dilation 2 by 3 | [7, 9, 2] | [5, 6, 2] |
| WithoutATrailingRowAtStrideOne | 2x2, stride 1 | [13, 13, 4] | [12, 12, 4] |
| ATrailingRowKeepsTheExtentAtStrideOne | 2x2, stride 1, one trailing row and column | [13, 13, 4] | [13, 13, 4] |
| ATrailingPadAtStrideTwoGivesSevenFromThirteen | 2x2, stride 2, one trailing row and column | [13, 13, 4] | [7, 7, 4] |
| ATrailingPadGivesTwoHundredEightFromFourHundredSixteen | 2x2, stride 2, one trailing row and column | [416, 416, 16] | [208, 208, 16] |
| ATrailingPadOnOneAxisOnly | 2x2, stride 1, one trailing row | [3, 3, 2] | [3, 2, 2] |
| ATrailingPadLetsAWindowFitAShortInput | 2x2, stride 1, one trailing row and column | [1, 4, 3] | [1, 4, 3] |

## Small_MaxPool2dForwardRefusal.ForwardRefuses

One row per input or geometry `forward()` refuses with `InvalidDescription`. Every row also checks
that the graph gains no node. Eight rows: a rank-2 input, a window of 5x5 over 4x4, an NCHW input, an
NCHW input narrower than the window (the message names "NHWC"), a zero vertical stride ("MaxPool2d",
"stride 0,2"), a negative padding ("padding 0,-1"), a negative trailing pad ("non-negative trailing
padding", "-2,0") and a trailing pad of 3 columns on a 2x2 window ("every window a tap inside the
input", "0,3").

## Small_MaxPool2dShapeRefusal.OutputShapeRefuses

One row per shape or geometry `output_shape()` refuses with `InvalidDescription`. Ten rows: a zero
vertical stride ("MaxPool2d", "stride 0,2"), a negative padding ("padding 0,-1"), a feature vector, a
shape that still carries the batch axis, a window taller than the input, a window wider than the
input, a [1, 4, 3] input without a trailing pad, a negative trailing pad ("non-negative trailing
padding", "0,-1"), a trailing pad of `INT64_MAX` rows ("within 64 bits") and a trailing pad as tall as
the window ("every window a tap inside the input", "2,0").

## Medium_MaxPoolTrailing.TheElementIsTheMaximumOfItsTruncatedTaps

A `MaxPool2d` with a 2x2 window, a stride of one and one trailing row and column pools a
`[1, 3, 3, 1]` input into nine values on the engine. One row per element of two inputs, 18 rows. A
maximum selects an input value, so the bound is zero: the pooled element must equal the maximum of its
truncated window exactly.

- The input `[[5, -1, 2], [-3, 8, 0], [4, -7, 6]]` pools to `[[8, 8, 2], [8, 8, 6], [4, 6, 6]]`.
- The input runs from -18 to -10. The result is `[[-14, -13, -13], [-11, -10, -10], [-11, -10, -10]]`;
  a pad that took part in the maximum as zero would show as 0.
