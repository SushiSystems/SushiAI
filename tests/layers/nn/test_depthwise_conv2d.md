# tests/layers/nn/test_depthwise_conv2d.cpp


The DepthwiseConv2d layer's one DEPTHWISE_CONV2D node, its [kernel_h, kernel_w, channels] weight with role and fans, shape inference against the trace, `same_padded`, and the inputs it rejects.

Suites: `Small_DepthwiseConv2d`, `Small_DepthwiseConv2dExtents`, `Small_DepthwiseConv2dForwardRefusal`,
`Small_DepthwiseConv2dShapeRefusal`, `Medium_DepthwiseConv2dDeathTest`.

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

## Small_DepthwiseConv2dExtents.OutputShapeAgreesWithForward

`output_shape` of the sample is the expected shape, and the shape traced over a batch is the
expected shape with that batch in front. The rows, all for a padded 3x3 window over four
channels, are StrideOneKeepsTheExtents ([8, 6, 4] to itself), StrideTwoHalvesTheExtents
([8, 6, 4] to [4, 3, 4]) and StrideTwoDropsTheRemainderOfAnOddExtent ([7, 9, 4] to [4, 5, 4]).

## SamePaddedBuildsAPaddedLayerOfTheGivenStride

`same_padded(4, 4, 5, 2)` gives 4 channels, a 5x5 window, stride 2 on both axes, a top and left
padding of 2, and dilation 1 on both axes.

## SamePaddedRefusesUnequalChannelCounts

The case is in the suite `Medium_DepthwiseConv2dDeathTest`, because a death test starts a child
process. `same_padded(4, 8, 3, 1)` stops the process with a report containing "keeps its channel
count".

## Small_DepthwiseConv2dForwardRefusal.ForwardRefuses

`forward` returns `InvalidDescription`, and the message holds the row's substring. The rows are
AnInputThatIsNotRankFour ([2, 16]: "rank-4"), AnNCHWInput (a [2, 8, 8, 4] input converted to
NCHW: "NHWC"), AChannelCountThatDiffersFromTheLayer ([2, 8, 8, 3] for four channels:
"[H, W, 4]"), ANonPositiveChannelCount (a layer of zero channels: "positive channel count") and
AWindowThatDoesNotFit (5x5 over [2, 4, 4, 4], the code alone).

## Small_DepthwiseConv2dShapeRefusal.OutputShapeRefuses

`output_shape` of a four-channel layer returns `InvalidDescription` with a message containing
"[H, W, 4]". The rows are AChannelCountThatDiffersFromTheLayer ([8, 8, 5]) and
ASampleOfTheWrongRank ([8, 8]).
