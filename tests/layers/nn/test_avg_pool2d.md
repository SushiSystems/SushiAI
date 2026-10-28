# tests/layers/nn/test_avg_pool2d.cpp


The shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

Suites: `Small_AvgPool2d`.

Tests the shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

This file needs no device. `NN::AvgPool2d` in `modules/layers/nn/include/SushiAI/nn/avg_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter, so a case either traces the layer into a host-side
graph or calls `output_shape()` on a per-sample shape.

## window

`window` returns a geometry with the given kernel and stride on each axis, dilation 1 and no
padding. A case that needs padding or dilation sets the field afterwards.

## traced_shape

`traced_shape` traces a layer over a batch of two images of the given per-sample shape and
returns the shape of the result, batch axis included.

## per_sample

`per_sample` drops the batch axis of a rank-4 shape.

## SatisfiesTheModuleTrait

Four `static_assert` lines hold the layer's place in the layer library: `AvgPool2d` is a
`Module`, `const AvgPool2d` is one too, which requires a const `forward()`, it is not a
`ModuleContainer`, and it has no parameter walk of its own, so the walk reaches it through
`fields()`.

## ForwardTracesThePooledShapeInNHWC

A 3x3 window with stride 3 over a [2, 6, 6, 3] input traces a [2, 2, 2, 3] value tagged NHWC.

## OutputShapeAgreesWithTheTracedShapeForASquareWindow

`output_shape([6, 6, 3])` is [2, 2, 3], and it equals the traced shape with the batch axis
dropped. The literal is asserted beside the comparison so the two cannot agree on a wrong
shape. The next two cases have the same form.

## OutputShapeAgreesWithTheTracedShapeForAPaddedUnevenWindow

A 3x2 window with stride 2 by 1 and one pixel of padding on each axis maps [7, 5, 4] to
[4, 6, 4].

## OutputShapeAgreesWithTheTracedShapeForADilatedWindow

A 3x3 unit-stride window dilated by 2 on the height axis maps [9, 8, 1] to [5, 6, 1]. The two
axes have different dilations, so an implementation that applied one axis's dilation to both
gives another shape.

## OutputShapeRejectsAFeatureVector

`output_shape([8])` fails, because a per-sample shape must be rank 3.

## OutputShapeRejectsABatchedShape

`output_shape([2, 4, 4, 3])` returns InvalidDescription. A shape that still carries the batch axis
is rank 4.

## OutputShapeRejectsAWindowLargerThanTheImage

A 5x5 window over a [4, 4, 3] sample returns InvalidDescription.

## OutputShapeRejectsAZeroStride

A geometry with stride 0 on both axes returns InvalidDescription.

## OutputShapeNamesAZeroStride

The message for the same geometry contains "AvgPool2d" and "stride 0,0". The case holds the message
to the stride, where a check ordered differently would report a window that does not fit.

## OutputShapeRejectsANegativePadding

A top padding of -1 returns InvalidDescription with a message containing "AvgPool2d" and "padding
-1,0".

## ForwardRejectsARankTwoInput

`forward()` returns InvalidDescription on a [2, 8] input.

## ForwardRejectsAnNCHWInput

A [1, 6, 6, 3] input is converted with `layout_convert` to NCHW, and `forward()` returns InvalidDescription for the
converted value.

## ForwardNamesTheLayoutOfAnNCHWInputNarrowerThanTheWindow

A [1, 8, 8, 2] input converted to NCHW has 2 on the axis an NHWC reader takes for the height,
which a 3x3 window does not fit. The message contains "NHWC", so the layout check runs before
the window is fitted.

## ForwardRejectsAZeroStride

`forward()` with stride 0 on both axes returns InvalidDescription with a message containing "AvgPool2d" and
"stride 0,0".

## ForwardRejectsANegativePadding

`forward()` with a top padding of -1 returns InvalidDescription with a message containing "AvgPool2d" and
"padding -1,0".

## ForwardRejectsAWindowLargerThanTheImage

`forward()` returns InvalidDescription for a 5x5 window over a [2, 4, 4, 3] input.

## ATracedLayerHasNoParameter

After a trace the parameter walk visits nothing and `parameter_count` is 0. The graph holds
two values, the input and the pooled result.
