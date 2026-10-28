# tests/core/autograd/rules/test_spatial.cpp


Suites: `Small_SpatialShapes`.

Tests the shapes the builder infers for spatial operations and the layouts it refuses. A refused
call returns `StatusCode::InvalidDescription` with the substrings the case names.

## Im2ColProducesTheColumnMatrixAGEMMExpects

[N,H,W,C] with a 3x3 stride-1 pad-1 window over 5x5 gives 5x5 windows per image. Each row is
one window's kernel_h * kernel_w * C taps with channels innermost, so a sub-group reads a
contiguous run (CV-3).

## Im2ColHonoursStrideAndDilation

A dilated 3x3 covers 5 pixels, so an unpadded 7 yields 3 windows, and stride 2 over the same
input yields 2. Both numbers are wrong under the naive (H - K)/S + 1 that forgets dilation.

## Im2ColRejectsAWindowLargerThanThePaddedInput

The oversized window produces a zero extent, which Shape refuses anyway. The message a caller
gets should still name the geometry, not the Shape invariant it happened to trip.

## Im2ColRejectsANonSpatialInput

`im2col` over a [4, 8] input is refused.

## Col2ImReturnsToTheImageItWasAskedFor

col2im cannot derive its output extents. ConvGeometry carries pad_top and pad_left but no
pad_bottom or pad_right, so an asymmetric padding would be silently mis-shaped. The caller
states the image shape instead. It always has it, because it held the value im2col consumed.

## Col2ImRejectsAColumnMatrixThatDoesNotMatchTheImage

`col2im` of the column matrix of a [2, 5, 5, 3] image with a padded 3x3 window, asked for the
image shape [2, 6, 6, 3], is refused.

## Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward

A 3x3 stride-1 window with padding only on the bottom and right of a 13x13 input still yields
13x13 windows, which is not what the symmetric formula (13-3)/1+1 = 11 would say. The forward
pass (`im2col_with_extents`) states 13x13 outright. The backward rule must be able to state the
same numbers back, because it is the one holding them, and must not re-derive them from
ConvGeometry alone.

The geometry leaves pad_top and pad_left at 0. Padding was applied on the bottom and right
only, which ConvGeometry cannot express and does not need to.

## Col2ImDerivingStillRejectsThatSameAsymmetricPair

This case is the point of having `col2im_with_extents` as a twin. The deriving col2im assumes
symmetric padding and is not a substitute for it. Feeding it the exact cols/image pair that
`col2im_with_extents` accepted in Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward must
still be refused, which proves the twin is needed.

## Im2ColRejectsAnNCHWInput

im2col reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it would read channels
as height and produce a plausible but silently wrong shape, where it should refuse it.

## MaxPoolProducesAValueAndItsArgmaxTogether

The indices are a second result of the node, not a second output tensor the caller allocates.
The backward rule needs exactly the indices this forward produced, and a pair returned together
cannot be mismatched the way two separate calls could.

## MaxPoolHandlesYOLOTinysAsymmetricThirteenToThirteen

YOLOv3-tiny's last pool is 2x2 stride 1 over 13x13, padded only on the bottom and right, staying
13x13. ConvGeometry cannot express that padding, so the caller states the extents. This case
proves the explicit overload is needed.

## AvgPoolProducesOnlyAValue

avgpool2d records no indices: its backward divides by a constant window area and needs nothing
the forward recorded (CV-6).

## UpsampleMultipliesTheSpatialExtentsOnly

`upsample2d` of [2, 3, 5, 7] by 2 and 2 gives [2, 6, 10, 7].

## UpsampleRejectsANonPositiveScale

`upsample2d` with a height scale of 0 is refused.

## MaxPoolRejectsAnNCHWInput

maxpool2d reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it must refuse it and
must not pool the channel axis.

## AvgPoolRejectsAnNCHWInput

`avgpool2d` over a value converted to NCHW is refused.

## UpsampleRejectsAnNCHWInput

`upsample2d` over a value converted to NCHW is refused.

## SpatialMeanReducesToNAndC

The result is the [N,C] that the numeric Probe's loss needs, which is why every convolutional
gradient test in this plan ends with this call.

## SpatialMeanRejectsAnNCHWInput

`spatial_mean` over a value converted to NCHW is refused.

## ConcatSumsTheChannelAxisAndNothingElse

`concat_channels` of [2, 8, 8, 16] and [2, 8, 8, 32] gives [2, 8, 8, 48].

## ConcatRejectsOperandsThatDisagreeOffTheChannelAxis

`concat_channels` of [2, 8, 8, 16] and [2, 4, 8, 16] is refused.

## ConcatRejectsAnNCHWOperand

`concat_channels` of two values converted to NCHW is refused.

## SliceTakesAChannelRangeAndRefusesToOverrun

Over [2, 8, 8, 48], `slice_channels(x, 16, 32)` gives [2, 8, 8, 32] and
`slice_channels(x, 32, 32)` is refused.

## SliceRejectsAnNCHWInput

`slice_channels` over a value converted to NCHW is refused.

## ChannelAffineKeepsTheShapeAndChecksTheParameters

`channel_affine` of [2, 8, 8, 16] with a [16] scale and a [16] shift gives [2, 8, 8, 16].
With an [8] scale it is refused.

## ChannelAffineRejectsAnNCHWInput

`channel_affine` over a value converted to NCHW is refused.

## BatchNorm2dKeepsItsShapeAndDeclaresThreePerChannelStatistics

`GraphBuilder::batchnorm2d` over a [2, 8, 8, 16] FLOAT64 input returns four values: the output,
of the input's shape, and `saved_mean`, `saved_inv_std` and `batch_variance`, each [16]. They
are the four outputs of one K_BATCHNORM2D node in that order, and attribute 0 of the node is the
epsilon 1e-5. Each of the four is FLOAT64 with the lifetime `ACTIVATION`. A gamma of [8]
is refused.

A non-positive epsilon is a division by the standard deviation itself, which is exactly zero
for a constant channel.

## BatchNorm2dRejectsAnNCHWInput

The layout check runs before any extent is read, so the operand that is wrong is the one the
message names.

## BatchNormFoldYieldsAPerChannelScaleAndShift

`GraphBuilder::batchnorm_fold` over a [16] gamma, beta, running mean and running variance adds
one node. Its operation is `Ops::K_BATCHNORM_FOLD`, its inputs are those four values in that
order, its outputs are `fold.scale` and `fold.shift`, and attribute 0 is the epsilon 1e-3.
Both results are [16] FLOAT64 values of lifetime `ACTIVATION`, the operands' type. A
`channel_affine` of a [2, 8, 8, 16] input with the two results keeps that shape.

## BatchNormFoldRejectsAnOperandThatDoesNotMatchGamma

Five calls are refused with "batchnorm_fold" and the operand at fault: a [8] mean ("mean", "[16]",
"[8]"), a [8] beta ("beta"), a FLOAT64 variance among FLOAT32 operands ("variance", "float32",
"float64"), a [4, 4] gamma ("rank-1") and four INT64 operands ("int64"). The graph's node count
and value count are unchanged after all five.

## BatchNormFoldRejectsANonPositiveEpsilon

An epsilon of 0.0 and one of -1e-3 are each refused with a message containing "batchnorm_fold",
"epsilon" and "positive". An epsilon of 1e-3 is accepted.

## SumChannelsFoldsEverythingButTheChannelAxis

`sum_channels` of [2, 8, 8, 16] gives [16], and over the same value converted to NCHW it
is refused.

## ReshapeRefusesToChangeTheElementCount

A reshape has one failure mode. It moves no data, so a mismatched count is not a resize; it is
a read past the end.

## Convolution, window extents and upsampled extents

The cases from here cover three additions to `modules/core/graph/include/SushiAI/graph/builder.hpp`.
`GraphBuilder::conv2d` emits a convolution as a node sequence: a K_RESHAPE of the rank-4 weight
to [C_out, taps], an K_IM2COL of the input, a K_MATMUL of the two with its second operand
transposed, an optional K_BIAS_ADD, and a K_RESHAPE of the rows to [N, out_h, out_w, C_out].
`Graph::window_extents` computes the output extents of a window and validates its geometry,
and `conv2d`, `im2col`, `maxpool2d` and `avgpool2d` all derive their extents through it.
`Graph::upsampled_extents` does the same for K_UPSAMPLE2D.

## op_names_from

`op_names_from` returns the operation names of a graph's nodes from a given index on. A case
records the node count after it has declared its operands and passes that count, so the list
holds only what the emitter under test added.

## square_window

`square_window` returns a unit-stride geometry with a square kernel and the same padding on
both axes.

## window

`window` returns a unit-stride unpadded geometry with the given kernel height and width.

## Conv2dReturnsAnNHWCValueWithOneChannelPerFilter

A [2, 5, 5, 3] input, an [8, 3, 3, 3] weight and an [8] bias under a padded 3x3 window give a
[2, 5, 5, 8] value tagged NHWC.

## Conv2dDividesTheOutputExtentsByItsStride

A padded 3x3 window with stride 2 over a [2, 8, 6, 3] input gives [2, 4, 3, 4]. The call
passes `K_NO_VALUE` as the bias.

## Conv2dWithABiasEmitsIm2colMatmulAndBiasAddBetweenTwoReshapes

The nodes the call adds are, in order, "sushiai.reshape", "sushiai.im2col", "sushiai.matmul",
"sushiai.bias_add" and "sushiai.reshape". The sequence holds no operation of its own, which is
decision CV-1: `test_ops.cpp` asserts under ThereIsNoConv2dIdentity that no CONV2D identity
exists.

## Conv2dWithoutABiasEmitsNoBiasAdd

With `K_NO_VALUE` as the bias the same call adds four nodes: reshape, im2col, matmul, reshape.

## Conv2dContractsTheColumnMatrixAgainstTheTransposedFilterMatrix

The third node added is a K_MATMUL. Its first operand has shape [2 * 5 * 5, 3 * 3 * 3], one row
per window, and its second has shape [8, 3 * 3 * 3], one row per filter. Flag 0 is false and
flag 1 is true, so the filter matrix is the transposed operand and the product has one column
per filter.

## Conv2dPointwiseEmitsNoIm2col

A 1x1 unit-stride unpadded window adds reshape, reshape, matmul, reshape: the input is
reshaped to one row per pixel and no K_IM2COL is emitted. The result is [2, 5, 5, 8].

## Conv2dStridedOneByOneKernelStillEmitsIm2col

A 1x1 window with stride 2 over a [2, 6, 6, 3] input adds reshape, im2col, matmul, reshape and
gives [2, 3, 3, 8]. A stride above one selects pixels, which a reshape cannot do, so the
shortcut of the previous case must not apply.

## Conv2dTakesANonSquareKernelInHeightThenWidthOrder

An [8, 1, 3, 3] weight under a 1x3 window over a [2, 5, 5, 3] input gives [2, 5, 3, 8]. Axis
1 of the weight is the kernel height and axis 2 the kernel width.

## Conv2dRejectsAWeightWhoseKernelExtentsDifferFromTheWindow

An [8, 3, 1, 3] weight under the same 1x3 window is refused. It holds as many elements as
[8, 1, 3, 3], so the reshape that follows accepts it and only the check in `conv2d` refuses
the transposed kernel.

## Conv2dRejectsAWeightWhoseInputChannelsDifferFromTheImage

An [8, 3, 3, 4] weight over a three-channel input is refused with a message containing "conv2d".
The reshape rejects the same weight by element count in a message that does not name the
convolution, so the case requires the name.

## Conv2dRejectsAFlattenedWeight

An [8, 27] weight is refused. The weight enters `conv2d` as rank 4 and is flattened inside it.

## Conv2dRejectsARankThreeInput

A [5, 5, 3] input, which lacks the batch axis, is refused.

## Conv2dPointwiseRejectsAnNCHWInput

A [2, 5, 5, 3] input is converted to NCHW and paired with an [8, 1, 1, 5] weight, whose
input-channel extent matches the last axis of the converted shape. The pointwise path skips
K_IM2COL, which is the emitter that checks the layout, so `conv2d` has to check it itself. The
call is refused.

## Conv2dRejectsABiasWhoseExtentDiffersFromTheFilterCount

A [4] bias with eight filters is refused.

## Conv2dRejectsAWindowLargerThanThePaddedInput

A 5x5 unpadded window over a [1, 2, 2, 3] input is refused.

## Conv2dEmitsNothingWhenItRejectsItsOperands

Three calls are refused: an oversized window, a weight with four input channels over a
three-channel input, and a [4] bias with eight filters. The node count afterwards equals the
count taken before the calls. `conv2d` validates every operand before it emits its first
reshape, so a refused call leaves no orphan node in the graph.

## Conv2dBuildsAGraphThatFinaliseAccepts

A biased 3x3 convolution feeds a pointwise one, and `Graph::finalise()` does not throw. Both
paths therefore leave every value with a layout tag the per-node proof accepts.

## Conv2dDifferentiatesToAWeightAndABiasGradient

The convolution's result goes through `spatial_mean`, `log_softmax` and `nll_loss`.
`Autograd::differentiate` returns two gradients other than `K_NO_VALUE`, of shapes [8, 3, 3, 3]
and [8]. The weight gradient has the rank-4 shape of the parameter and not the [8, 27] shape
of the reshaped operand the K_MATMUL read.

## WindowExtentsPadBothSidesOfAnAxisByTheLeadingPad

A 3x5 window with a top pad of 1 and a left pad of 2 over a 7 by 9 input yields 7 by 9. The
extents are unchanged only if each pad is applied to both sides of its axis.

## WindowExtentsHonourStrideAndDilationPerAxis

A 3x3 window with stride 2 by 3 and a height dilation of 2 over an 11 by 16 input yields 4 by
5. The two axes differ in stride and in dilation, so a value taken from the wrong axis changes
one of the two numbers.

## WindowExtentsOfAWindowThatExactlyFitsAreOneByOne

A 5x5 window with padding 1 over a 3 by 3 input yields 1 by 1. The padded input is exactly as
large as the window.

## WindowExtentsRejectANonPositiveKernel

A kernel of 0 by 3 and one of 3 by -2 are refused. Each message contains "positive kernel" and the
kernel as given, "kernel 0x3" and "kernel 3x-2".

## WindowExtentsRejectANonPositiveStrideAndNameIt

A vertical stride of 0 and a horizontal stride of -1 are refused with "positive stride" and the
pair, "stride 0,1" and "stride 1,-1". A zero stride is a division by zero in the extent
formula, and the message names the stride and not a window that does not fit.

## WindowExtentsRejectANonPositiveDilation

A horizontal dilation of 0 is refused with "positive dilation" and "dilation 1,0".

## WindowExtentsRejectANegativePadding

A padding of -1 on both axes is refused with "padding -1,-1", and a left pad of -1 alone is refused
with "padding 0,-1".

## WindowExtentsRejectAWindowThatDoesNotFit

A 5x5 window over a 4 by 6 input is refused with a message containing "5x5 window" and "4x6
input". The window fits the width and not the height.

## UpsampledExtentsScaleEachAxisByItsOwnFactor

A 3 by 5 input with scales 2 and 4 yields 6 by 20.

## UpsampledExtentsRejectAScaleBelowOne

Scales of 0 by 2 and of 2 by -1 are refused with "at least one" and the pair, "0x2" and "2x-1".

## UpsampledExtentsRejectAProductBeyondSixtyFourBits

With a scale of 2^62, an extent of 5 on the height axis is refused with "fit in 64 bits", and so
does an extent of 5 on the width axis. Each call overflows one axis and leaves the other at
1, so both products are checked.

## UpsampledExtentsAcceptTheLargestProductThatFits

Half of the largest int64, rounded down, scaled by 2 yields the largest int64 minus 1. The
overflow check therefore refuses nothing that 64 bits hold.

## UpsampleRejectsAScaledExtentBeyondSixtyFourBits

`GraphBuilder::upsample2d` with a horizontal scale of 2^62 over a width of 5 is refused with "fit
in 64 bits", and the node count is unchanged.

## TheDerivingEmittersRejectANegativePadding

`im2col`, `conv2d`, `maxpool2d` and `avgpool2d` are each called with a padding of -1, and each
message contains "padding". These are the four emitters that derive their extents from the
geometry.

## Col2ImRejectsANegativePadding

`col2im` with a padding of -1 is refused with a message containing "padding".

## Col2ImRejectsAnImageShapeThatIsNotRankFour

`col2im` asked for the image shape [5, 5, 3] is refused with a message containing "col2im" and
"rank-4".

## TheDerivingEmittersNameTheLayoutBeforeTheWindow

A [1, 8, 8, 2] input is converted to NCHW, which leaves 2 on the axis an NHWC reader takes
for the height, and a 3x3 window does not fit that. `im2col`, `maxpool2d` and `avgpool2d` each
return a failure whose message contains their own name and "NHWC". A caller who passed the
wrong layout is told so, and is not told that the window is too large.
