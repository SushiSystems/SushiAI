# tests/layers/nn/test_avg_pool2d.cpp


The shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

Suites: `Small_AvgPool2d`, `Small_AvgPool2dShape`, `Small_AvgPool2dShapeRefusal`, `Small_AvgPool2dForwardRefusal`.

Tests the shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

This file needs no device. `NN::AvgPool2d` in `modules/layers/nn/include/SushiAI/nn/avg_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter, so a case either traces the layer into a host-side
graph or calls `output_shape()` on a per-sample shape.

## window

`window` returns a geometry with the given kernel and stride on each axis, dilation 1 and no
padding. A case that needs padding or dilation sets the field afterwards.

## padded_top_left, padded_top, dilated

Each returns a copy of a geometry with the top and left padding, the top padding or the dilation set.

## traced_shape

`traced_shape` traces a layer over a batch of two images of the given per-sample shape and
returns the shape of the result, batch axis included.

## per_sample

`per_sample` drops the batch axis of a rank-4 shape.

## ForwardTracesThePooledShapeInNHWC

A 3x3 window with stride 3 over a [2, 6, 6, 3] input traces a [2, 2, 2, 3] value tagged NHWC.

## Small_AvgPool2dShape.OutputShapeAgreesWithTheTracedShape

`output_shape` of the sample is the expected shape, and it equals the traced shape with the batch
axis dropped. The literal is asserted beside the comparison so the two cannot agree on a wrong
shape. The rows are Square (a 3x3 window with stride 3 maps [6, 6, 3] to [2, 2, 3]),
PaddedUneven (a 3x2 window with stride 2 by 1 and one pixel of padding at the top and the left
maps [7, 5, 4] to [4, 6, 4]) and Dilated (a 3x3 unit-stride window dilated by 2 on the height and
1 on the width maps [9, 8, 1] to [5, 6, 1]; the two axes differ, so an implementation that
applied one axis's dilation to both gives another shape).

## Small_AvgPool2dShapeRefusal.OutputShapeRefuses

`output_shape` returns InvalidDescription, and the message holds the row's substrings. The rows
are AFeatureVector ([8], a per-sample shape must be rank 3), ABatchedShape ([2, 4, 4, 3] is rank
4), AWindowLargerThanTheImage (5x5 over [4, 4, 3]), AZeroStride (stride 0 on both axes; the
message holds "AvgPool2d" and "stride 0,0", so a check ordered differently would report a window
that does not fit) and ANegativePadding (a top padding of -1; "AvgPool2d" and "padding -1,0").

## Small_AvgPool2dForwardRefusal.ForwardRefuses

`forward()` returns InvalidDescription, and the message holds the row's substrings. The rows are
ARankTwoInput ([2, 8]), AnNCHWInput ([1, 6, 6, 3] converted with `layout_convert` to NCHW),
AnNCHWInputNarrowerThanTheWindowNamesTheLayout ([1, 8, 8, 2] converted to NCHW has 2 on the axis
an NHWC reader takes for the height, which a 3x3 window does not fit; the message holds "NHWC",
so the layout check runs before the window is fitted), AZeroStride and ANegativePadding (the
messages of the shape refusal), and AWindowLargerThanTheImage (5x5 over [2, 4, 4, 3]).

## ForwardAddsOneValue

After a trace the graph holds two values, the input and the pooled result.

## ForwardEmitsOneAveragePoolNodeThatCarriesTheWindow

A layer with a 3 by 2 kernel, stride 2 by 1, dilation 2 by 1 and a top and left padding of 1 traces one
new node of the operation `Ops::K_AVGPOOL2D` over an NHWC [2, 8, 8, 3] input. The node reads the input
and writes the returned value, and its attribute slots 0 to 7 hold the kernel, stride, dilation and
padding in that order: 3, 2, 2, 1, 2, 1, 1, 1. The case states a promise of `avg_pool2d.hpp`, that the
window reaches the node whole, and the node is the only place it can be observed.
