# tests/layers/nn/test_conv2d/test_groups.cpp

That the `groups` of the conv2d layer split the channels into independent convolutions, that a layer whose groups equal both channel counts is depthwise, and that both give the values a hand calculation gives and the gradients central differences give.

Suites: `Small_Conv2dGroupsShape`, `Small_Conv2dGroupsRefusal`, `Small_Conv2dGroups`, `Medium_Conv2dGroupsGradient`.

Tests `NN::Conv2d` with its `groups` field and the `groups` key of `NN::make_conv2d`. The values
cases evaluate the traced graph on the host in double; every number is an integer, so the
comparison bound is zero.

## Small_Conv2dGroupsShape.TheWeightAndTheResultFollowTheGroups

After a trace over a [2, 6, 6, in] input the weight has the shape of the row, the result has the
per-sample shape of the row, and `output_shape` agrees. The rows are two groups of two filters
(weight [4, 3, 3, 2]), two groups over six input channels ([4, 3, 3, 3]), a depthwise layer
([3, 3, 4]), a channel multiplier of two with four groups ([8, 3, 3, 1]) and one group, the dense
layer ([4, 3, 3, 4]).

## Small_Conv2dGroupsRefusal.OutputShapeRefusesTheGroups

`InvalidDescription` for zero groups, for three groups over four input channels, for three groups
over four filters, and for a depthwise layer with a trailing pad ("depthwise, which has no
trailing padding").

## Small_Conv2dGroups.AGroupedLayerTracesOneConvolutionPerGroup

Three groups of 3x3 windows over six channels trace three `K_IM2COL` nodes, three `K_MATMUL` nodes and one
`K_CONCAT_CHANNELS`, and no depthwise node.

## Small_Conv2dGroups.ADepthwiseLayerTracesTheDepthwiseNodeAndABroadcastBias

A depthwise layer traces one `K_DEPTHWISE_CONV2D`, one `K_BROADCAST_ADD` for its bias and no
`K_IM2COL`; its parameters are weight [3, 3, 4] and bias [4].

## Small_Conv2dGroups.ABiasFreeDepthwiseLayerAddsNoBroadcast

Without a bias the layer traces no `K_BROADCAST_ADD` and holds the weight alone.

## Small_Conv2dGroups.AGroupedLayerHoldsOneWeightAndOneBias

Two groups over four channels and six filters hold weight [6, 3, 3, 2] and bias [6]. The weight's
fan in is 3 * 3 * 2 and its fan out 3 * 3 * 3, one group's share.

## Small_Conv2dGroups.TheValuesOfTwoGroupsMatchTheHandResults

Four channels in two groups of two filters, a 1 by 1 window, the weights (1, 2), (3, 4), (5, 6),
(7, 8) and the biases 10, 20, 30, 40. The pixel (1, 2, 3, 4) gives 15, 31, 69, 93 and the pixel
(0, 1, 0, -1) gives 12, 24, 24, 32.

## Small_Conv2dGroups.TheValuesOfADepthwiseLayerMatchTheHandResults

Two channels read 1, 2, 4 and 3, 5, 7 through a window of two rows. The taps are (1, 2) and
(10, 20) and the biases 100 and 200, so the result is 105, 330, 110, 390 in (row, channel) order.

## Small_Conv2dGroups.ABiasFreeGroupedLayerGivesTheProductsAlone

The first case without biases over the pixel (1, 2, 3, 4) gives 5, 11, 39, 53.

## Small_Conv2dGroups.TheRowKeyGroupsBuildsAGroupedLayer

`make_conv2d` over [8, 8, 4] with 6 filters, kernel 3 and groups 2 gives a layer whose result is
[6, 6, 6].

## Small_Conv2dGroups.TheRowRefusesGroupsThatDoNotDivideTheChannels

The same row with groups 3 fails with the context "layer 1 'conv2d'" and "groups that divide both
channel counts; got 3".

## Small_Conv2dGroups.TheRowDefaultsToOneGroup

A row with no "groups" key gives the dense layer, whose result is [6, 6, 6].

## Medium_Conv2dGroupsGradient.TheInputGradientOfAGroupedLayerMatchesCentralDifferences

The gradient with respect to a [2, 2, 1, 4] input of a two-group layer, at a point whose 16
elements differ, matches central differences in FLOAT64.

## Medium_Conv2dGroupsGradient.TheWeightGradientOfAGroupedLayerMatchesCentralDifferences

The gradient with respect to the [4, 1, 1, 2] weight matches central differences. Every group's
slice of the weight is a view of the one parameter, so each group's gradient reaches it.

## Medium_Conv2dGroupsGradient.TheInputGradientOfADepthwiseLayerMatchesCentralDifferences

The gradient with respect to a [2, 3, 1, 2] input of a depthwise layer with a window of two rows
matches central differences.
