# tests/layers/nn/test_conv2d.cpp


Suites: `Small_Conv2d`.

Tests the 2-D convolution layer's trace, parameters, shape inference and rejections.

This file needs no device. `NN::Conv2d` in `modules/layers/nn/include/SushiAI/nn/conv2d.hpp` declares a weight
and a bias on its first trace and hands the node sequence to `GraphBuilder::conv2d`, so every
case reads a host-side graph, the layer's two parameters or the result of `output_shape()`.
`test_spatial_shapes.cpp` covers the nodes the builder emits; this file covers the layer.

Two `static_assert` lines at file scope require `Conv2d` to be a `Module` and not a
`ModuleContainer`.

## K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

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

## OutputShapeAgreesWithForwardForAStridedWindow

A 3x3 stride-2 unpadded window with sixteen filters maps [9, 12, 3] to [4, 5, 16], and the
traced shape of a batch of two equals that shape with the batch extent in front. The four
cases from here assert the literal and the agreement together.

## OutputShapeAgreesWithForwardForAPaddedWindow

A 5x5 window with padding 2 maps [7, 10, 2] to [7, 10, 4].

## OutputShapeAgreesWithForwardForADilatedStridedWindow

A 3x3 window with stride 2, padding 1 and dilations 2 by 3 maps [11, 13, 2] to [5, 5, 3].

## OutputShapeAgreesWithForwardForAPointwiseWindow

A 1x1 window with six filters maps [5, 7, 4] to [5, 7, 6]. This is the geometry for which
`GraphBuilder::conv2d` emits no K_IM2COL.

## SamePaddedPadsByHalfTheKernelAndDropsTheBias

`Conv2d::same_padded(3, 16, 5, 2)` has 3 input and 16 output channels, a 5x5 kernel, stride 2
on both axes, a pad of 2 on top and left, dilation 1 and `use_bias` false.

## SamePaddedAtStrideOneKeepsTheSpatialExtents

`same_padded(4, 4, 3, 1)` maps [7, 10, 4] to [7, 10, 4] in `output_shape()` and traces
[2, 7, 10, 4] over a batch of two.

## SamePaddedAtStrideOneKeepsTheExtentsForAPointwiseAndAFiveByFiveKernel

Over [9, 6, 2], `same_padded(2, 8, 1, 1)` and `same_padded(2, 8, 5, 1)` both report [9, 6, 8].

## SamePaddedAtStrideTwoHalvesAnEvenExtent

`same_padded(3, 16, 3, 2)` maps [640, 320, 3] to [320, 160, 16] in `output_shape()` and in the
trace.

## SamePaddedTracesNoBias

A traced `same_padded(2, 8, 3, 1)` layer walks as "weight:[8, 3, 3, 2]" alone. Its bias is not
declared and the graph holds no K_BIAS_ADD node.

## ANonZeroBiasInitReachesTheBiasParameter

A layer with `bias_init` set to -4.5 traces a bias whose `init()` is a constant of -4.5 and
whose role is `ParameterRole::Bias`. -4.5 is exact in binary.

## TheDefaultBiasInitIsTheConstantZero

A layer left at the default traces a bias whose `init()` is a constant of 0.0.

## BiasInitLeavesTheWeightFanScaled

With `bias_init` set to 1.25 the weight's `init()` is still `ParameterInit::Kind::FanScaled`
and its role is `WEIGHT`.

## OutputShapeRejectsANonPositiveInputChannelCount

A layer with `in_channels` 0 throws with a message containing "Conv2d", "in=0" and "out=8".

## OutputShapeRejectsANonPositiveOutputChannelCount

A layer with `out_channels` -1 throws with a message containing "Conv2d", "in=3" and "out=-1".

## OutputShapeRejectsAWrongChannelCount

A three-channel layer given [8, 8, 4] throws. The message contains "Conv2d(3, 8)", the shape
"[8, 8, 4]" and the expected form "[H, W, 3]".

## OutputShapeRejectsAFeatureVector

`output_shape([3])` throws with a message containing "Conv2d(3, 8)", "[3]" and "[H, W, 3]".

## OutputShapeRejectsABatchedShape

`output_shape([2, 8, 3, 3])` throws with a message containing the shape and "[H, W, 3]". The
third extent equals `in_channels`, so only the rank check can refuse this shape.

## OutputShapeRejectsAWindowThatDoesNotFit

A 5x5 window over [4, 4, 3] throws with a message containing "Conv2d(3, 8)", "[4, 4, 3]" and
"5x5".

## OutputShapeRejectsAZeroStride

A stride of 0 throws with a message containing "Conv2d(3, 8)" and "stride 0,0".

## OutputShapeRejectsANegativePadding

A padding of -1 throws with a message containing "Conv2d(3, 8)" and "padding -1,-1".

## ForwardRejectsAWrongChannelCount

`forward()` on a [2, 8, 8, 4] input throws with a message containing "Conv2d(3, 8)",
"[8, 8, 4]" and "[H, W, 3]", the three strings OutputShapeRejectsAWrongChannelCount expects.
The weight is not declared afterwards.

## ForwardRejectsAnInputThatIsNotRankFour

`forward()` on a [2, 3] input throws with a message containing "Conv2d(3, 8)", "rank-4" and
"[2, 3]", and the weight is not declared afterwards.

## ForwardRejectsNonPositiveChannelCounts

`forward()` on a layer with `out_channels` 0 throws with a message containing "Conv2d",
"in=3" and "out=0".

## ForwardRejectsAWindowThatDoesNotFit

`forward()` with a 5x5 window over a [2, 4, 4, 3] input throws with a message containing
"Conv2d(3, 8)", "[4, 4, 3]" and "5x5".

## ForwardRejectsAnNCHWInput

The input is [2, 3, 3, 3] converted to NCHW. Every extent is 3, so the permuted shape passes
every check but the one on the layout tag. The message contains "Conv2d" and "NHWC". The node
count and the value count are unchanged and neither parameter is declared, so the layer
refuses before it writes anything to the graph.

## DifferentiatingALossYieldsAGradientForTheWeightAndTheBias

The layer's result is reduced with `spatial_mean` and passed to `NN::CrossEntropyLoss`.
`Autograd::differentiate` returns two gradients, neither is `K_NO_VALUE`, and each has the shape
of its parameter. The case checks that a gradient exists and has the right shape; it computes
no number.
