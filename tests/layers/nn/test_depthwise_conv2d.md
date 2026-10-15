# tests/layers/nn/test_depthwise_conv2d.cpp


The DepthwiseConv2d layer's one DEPTHWISE_CONV2D node, its [kernel_h, kernel_w, channels] weight with role and fans, shape inference against the trace, `same_padded`, and the inputs it rejects.

Suites: `Small_DepthwiseConv2d`.

Tests `NN::DepthwiseConv2d`. Most cases use four channels, a 3x3 window and one pixel of
padding.

Two `static_assert` lines at the top of the file hold that `DepthwiseConv2d` is a `Module` and
is not a `ModuleContainer`.

## ForwardEmitsOneDepthwiseNodeAndNoBiasAdd

After `forward` over a [2, 8, 8, 4] input the graph holds one `Ops::K_DEPTHWISE_CONV2D` node
and no `Ops::K_BIAS_ADD` node.

## TheEmittedNodeReadsTheInputAndTheWeight

The depthwise node has two inputs: the value passed in, and a value of the shape [3, 3, 4]. Its
one output is the value `forward` returned.

## ForwardKeepsTheSpatialExtentsAndTagsNHWC

A padded 3x3 window at stride 1 over [2, 8, 8, 4] yields [2, 8, 8, 4], tagged
`TensorLayout::NHWC`.

## StrideTwoHalvesAnEvenExtent

At stride 2 a [2, 8, 6, 4] input yields [2, 4, 3, 4].

## TheWeightOfAnAnisotropicWindowIsHeightWidthChannels

With a window 2 high and 4 wide over three channels, the weight's shape after `forward` is
[2, 4, 3].

## TheParameterWalkNamesOnlyTheWeight

The parameter walk is empty before `forward`. After it the walk yields "weight:[3, 3, 4]"
alone, and `NN::parameter_elements` is 36.

## TheWeightRecordsTheWeightRole

After `forward`, `weight.role()` is `ParameterRole::Weight`.

## TheWeightRecordsItsWindowAsBothFans

After `forward` with a 3x3 window, `weight.fan_in()` and `weight.fan_out()` are both 9.

## TheWeightAndTheResultTakeTheInputElementType

Over a `Float64` input, the weight's element type and the result's element type are `Float64`.

## OutputShapeAtStrideOneKeepsTheExtents

`output_shape([8, 6, 4])` is [8, 6, 4] for a padded 3x3 window at stride 1.

## OutputShapeAtStrideTwoHalvesTheExtents

`output_shape([8, 6, 4])` is [4, 3, 4] for a padded 3x3 window at stride 2.

## OutputShapeAgreesWithForwardAtStrideOneAndTwo

For the sample [8, 6, 4] in a batch of 3, the traced shape equals `output_shape` with the batch
extent in front, at stride 1 and at stride 2.

## OutputShapeAgreesWithForwardOverAnOddExtent

At stride 2, `output_shape([7, 9, 4])` is [4, 5, 4], and the traced shape of a batch of 2 is
that shape with the batch extent in front.

## SamePaddedBuildsAPaddedLayerOfTheGivenStride

`same_padded(4, 4, 5, 2)` gives 4 channels, a 5x5 window, stride 2 on both axes, a top and left
padding of 2, and dilation 1 on both axes.

## SamePaddedRefusesUnequalChannelCounts

`same_padded(4, 8, 3, 1)` throws an error naming "keeps its channel count".

## ForwardRejectsAnInputThatIsNotRankFour

`forward` over a [2, 16] input throws an error naming "rank-4".

## ForwardRejectsAnNCHWInput

`forward` over a value converted to NCHW throws an error naming "NHWC".

## ForwardRejectsAChannelCountThatDiffersFromTheLayer

A four-channel layer over a [2, 8, 8, 3] input throws an error naming "[H, W, 4]".

## ForwardRejectsANonPositiveChannelCount

A layer with `channels` 0 throws an error naming "positive channel count".

## ForwardRejectsAWindowThatDoesNotFit

An unpadded 5x5 window over a [2, 4, 4, 4] input throws `Error`.

## OutputShapeRejectsAChannelCountThatDiffersFromTheLayer

`output_shape([8, 8, 5])` on a four-channel layer throws an error naming "[H, W, 4]".

## OutputShapeRejectsASampleOfTheWrongRank

`output_shape([8, 8])` throws an error naming "[H, W, 4]".
