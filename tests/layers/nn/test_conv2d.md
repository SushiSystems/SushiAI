# tests/layers/nn/test_conv2d.cpp


Suites: `Small_Conv2d`, `Small_Conv2dShape`, `Small_Conv2dShapeRefusal`,
`Small_Conv2dForwardRefusal`.

Tests the 2-D convolution layer's trace, parameters, shape inference and rejections.

This file needs no device. `NN::Conv2d` in `modules/layers/nn/include/SushiAI/nn/conv2d.hpp` declares a weight
and a bias on its first trace and hands the node sequence to `GraphBuilder::conv2d`, so every
case reads a host-side graph, the layer's two parameters or the result of `output_shape()`.
`test_spatial_shapes.cpp` covers the nodes the builder emits; this file covers the layer.

Two `static_assert` lines at file scope require `Conv2d` to be a `Module` and not a
`ModuleContainer`.

## make_conv

`make_conv` builds a layer with a square window and one stride and one padding for both axes.
The stride defaults to 1 and the padding to 0. A case that needs an anisotropic window or a
dilation sets the geometry field afterwards.

## with_batch

`with_batch` prepends a batch extent to a per-sample [H, W, C] shape.

## traced_shape

`traced_shape` traces a copy of a layer over a batch of the given per-sample shape and returns
the result's shape. It takes the layer by value, so a const layer can be traced although
`forward()` declares parameters.

## walk

`walk` returns the parameter walk's output as "name:extents" strings.

## count_op

`count_op` counts the nodes of one operation in the whole graph.

## dilated

`dilated` returns a copy of a layer with the given dilation on each axis.

## expect_gradient_of

`expect_gradient_of` checks that a gradient exists and has the shape of its parameter.

## ForwardOfAPaddedThreeByThreeKeepsTheSpatialExtents

A 3x3 window with padding 1, one input channel and eight filters traces a [4, 28, 28, 1]
input to a [4, 28, 28, 8] value tagged NHWC.

## TheParametersAndTheResultTakeTheInputElementType

With a FLOAT64 input the weight, the bias and the result are FLOAT64.

## TheParameterWalkNamesTheWeightAndTheBias

The walk is empty before the trace. After it the walk yields "weight:[8, 3, 3, 1]" then
"bias:[8]", the names `Linear` uses, and `parameter_elements` is 8 * 3 * 3 * 1 + 8.

## TheWeightOfAnAnisotropicWindowIsOutHeightWidthIn

A 2x4 window with three input channels and five filters gives a weight of shape [5, 2, 4, 3].
The four extents differ, so a weight with two axes swapped fails the comparison.

## ALayerWithABiasTracesOneBiasAdd

The traced graph holds exactly one K_BIAS_ADD node.

## ABiasFreeLayerHasNoBiasEntryAndNoBiasAdd

With `use_bias` false the walk yields the weight alone, the bias is not declared, the graph
holds no K_BIAS_ADD node, and the result shape is still [4, 28, 28, 8].

## TheWeightAndTheBiasRecordTheirFans

For a 3x5 window with four input channels and six filters, both parameters record a fan-in of
3 * 5 * 4 and a fan-out of 3 * 5 * 6. The bias carries the same pair as the weight.

## TheParametersRecordTheirRoles

The weight has role WEIGHT and the bias has role BIAS.

## TracingTwiceDoesNotDuplicateParameters

After a second trace into the same graph the weight and the bias keep their value ids and
`parameter_count` is 2.

## Small_Conv2dShape.OutputShapeAgreesWithForward

`output_shape` of the sample is the expected shape, and the shape traced over a batch is the
expected shape with that batch in front. The rows are AStridedWindow (3x3, stride 2, [9, 12, 3]
to [4, 5, 16]), APaddedWindow (5x5 padded by 2, [7, 10, 2] to [7, 10, 4]), ADilatedStridedWindow
(3x3, stride 2, padding 1, dilation 2 by 3, [11, 13, 2] to [5, 5, 3]), APointwiseWindow
([5, 7, 4] to [5, 7, 6]) and, for `same_padded` at stride 1, SamePaddedAtStrideOne (3x3,
[7, 10, 4] to itself), SamePaddedPointwise and SamePaddedFiveByFive ([9, 6, 2] to [9, 6, 8]).

## SamePaddedPadsByHalfTheKernelAndDropsTheBias

`Conv2d::same_padded(3, 16, 5, 2)` has 3 input and 16 output channels, a 5x5 kernel, stride 2
on both axes, a pad of 2 on top and left, dilation 1 and `use_bias` false.

## SamePaddedAtStrideTwoHalvesAnEvenExtent

`same_padded(3, 16, 3, 2)` maps [640, 320, 3] to [320, 160, 16] in `output_shape()` and in the
trace.

## SamePaddedTracesNoBias

A traced `same_padded(2, 8, 3, 1)` layer walks as "weight:[8, 3, 3, 2]" alone. Its bias is not
declared and the graph holds no K_BIAS_ADD node.

## ANonZeroBiasInitReachesTheBiasParameter

A layer with `bias_init` set to -4.5 traces a bias whose `init()` is a constant of -4.5 and
whose role is `ParameterRole::Bias`. -4.5 is exact in binary, so the constant is compared for equality.

## TheDefaultBiasInitIsTheConstantZero

A layer left at the default traces a bias whose `init()` is a constant of 0.0.

## BiasInitLeavesTheWeightFanScaled

With `bias_init` set to 1.25 the weight's `init()` is still `ParameterInit::Kind::FanScaled`
and its role is `WEIGHT`.

## Small_Conv2dShapeRefusal.OutputShapeRefuses

`output_shape` returns `InvalidDescription`, and the message holds the row's substrings. The rows
are ANonPositiveInputChannelCount ("Conv2d", "in=0", "out=8"), ANonPositiveOutputChannelCount
("in=3", "out=-1"), AWrongChannelCount ([8, 8, 4] for a Conv2d(3, 8): "[8, 8, 4]", "[H, W, 3]"),
AFeatureVector ([3]), ABatchedShape ([2, 8, 3, 3], whose third extent equals the input channels
so only the rank check can refuse it), AWindowThatDoesNotFit (5x5 over [4, 4, 3]: "5x5"),
AZeroStride ("stride 0,0") and ANegativePadding ("padding -1,-1").

## Small_Conv2dForwardRefusal.ForwardRefuses

`forward` returns `InvalidDescription`, and the message holds the row's substrings. The graph
gains no node and no value, and neither the weight nor the bias is declared. The rows are
AWrongChannelCount ([2, 8, 8, 4]), AnInputThatIsNotRankFour ([2, 3]: "rank-4", "[2, 3]"),
ANonPositiveChannelCount (out=0), AWindowThatDoesNotFit and AnNCHWInput (a [2, 3, 3, 3] input
converted to NCHW; every extent is 3, so only the layout tag refuses it: "NHWC").

## DifferentiatingALossYieldsAGradientForTheWeightAndTheBias

The layer's result is reduced with `spatial_mean` and passed to `NN::CrossEntropyLoss`.
`Autograd::differentiate` returns two gradients; the weight's gradient and the bias's gradient
each exist and have the shape of their parameter. The case computes no number.
