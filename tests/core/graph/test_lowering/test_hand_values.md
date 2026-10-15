# tests/core/graph/test_lowering/test_hand_values.cpp


Suites: `Integration_Forward`, `Integration_ForwardDeathTest`.

Every lowered operation is checked against values worked out by hand. This tier catches a
transposed gemm, a bias broadcast down the wrong axis, and a log-softmax whose stabilising
shift did not cancel. All three produce plausible numbers and no error.

## LinearLayerMatchesHandComputedValues

`X * W^T + b` for a `[2, 2]` input and a `[3, 2]` weight gives `{-1, 1.5, 4, 1, 3.5, 8}`, and the
ReLU after it turns the -1 into 0, both within 1e-5.

## SumRowsProducesColumnSums

`sum_rows` of the `[3, 2]` value `{1, 2, 3, 4, 5, 6}` gives 9 and 12.

## ReshapeReinterpretsRowMajorLinearOrder

A `[3, 4]` value holding 0 to 11 reshaped to `[2, 6]` still reads 0 to 11 in order.

## LayoutConvertPermutesChannelsToTheFront

In NCHW the channels are outermost of the spatial block. For each c the two W positions follow
in order: (c=0, w=0..1) = 0, 3; (c=1, w=0..1) = 1, 4; (c=2, w=0..1) = 2, 5.

## ElementwiseAndScaleMatchTheirDefinitions

`add`, `sub`, `mul`, a `scale` by -2.5 and `exp` over two `[2, 2]` values equal the numbers worked
out by hand, within 1e-5.

## ActivationsMatchTheirDefinitionsAndLeaveTheInputAlone

Out-of-place semantics matter. The backward rules for relu and gelu read the forward input, so
an in-place activation would hand them the output.

The leaky_relu values are computed by hand at slope 0.1, with inputs on both sides of zero so
that both branches run: -2 gives -0.2, -0.5 gives -0.05, 0.5 gives 0.5, 2 gives 2.

## LogSoftmaxRowsSumToOneAfterExponentiating

The second row is the first offset by a large constant. Log-softmax must be invariant to the
offset, and the max shift is what makes it so.

## CrossEntropyOnUniformLogitsIsExactlyLogC

With `[4, 5]` logits of zero and one-hot targets, the loss is ln(5) within 1e-5.

## CrossEntropyMatchesAHandComputedSingleRow

For the logits `{1, 2, 3}` and the target class 2, the loss is `log(1 + exp(-1) + exp(-2))` within
1e-5.

## AccumulateAddsInPlaceAndAliasesItsResult

`accumulate` of `{1, 2, 3, 4}` and `{10, 20, 30, 40}` gives `{11, 22, 33, 44}`, and the accumulator
read back holds the same values as the result.

## ZeroFillsItsBufferExactlyOnce

Replaying must not accumulate. The bias broadcast writes into its result and does not add, and
the zero buffer is not re-filled per step.

## Im2colGathersOneWindowInTapOrderChannelInnermost

A gradcheck says the backward is consistent with whatever the forward computes. It says
nothing about whether the forward computes the right thing. This case is the forward's own
proof, worked out by hand.

N=1, H=2, W=2, C=1 with a 2x2 kernel and no padding covers the whole image in one window. The
column matrix's single row is the input read in tap order (r, s) with the channel innermost,
which for one channel is row-major (h, w).

## Col2imSumsEveryWindowThatOverlapsAPixel

`col2im` claims, and `im2col_backward` relies on it, that a pixel covered by several windows
receives the sum of their contributions and not the last one written. N=1, H=3, W=3, C=1 with a
2x2 kernel and no padding has four windows, with origins (0,0), (0,1), (1,0) and (1,1). The
corner pixels sit under one window, the edge pixels under two, and the centre pixel under all
four. Feeding every column entry a 1 turns the expected image into that overlap count:

```
1 2 1
2 4 2
1 2 1
```

## AnAsymmetricallyPaddedIm2colUsesTheExtentsItWasGivenNotDerivedOnes

A symmetric re-derivation gets this case wrong without an error. A [1, 4, 5, 1] input with a
3x3 window, stride 1, `pad_top` 1 and `pad_left` 1, padded 2 below and 0 on the right, has

```
out_h = (4 + 1 + 2 - 3) + 1 = 5     out_w = (5 + 1 + 0 - 3) + 1 = 4
```

Re-deriving the extents from the geometry alone assumes symmetric padding and yields 4x5, the
transpose of the real window grid. Both grids have 20 windows, so the row-count check agrees
and both sets of window origins stay inside the padded input. Nothing throws; the numbers are
wrong. That is why the extents travel in the node's attributes and are not recomputed.

The input is 0..19 in row-major (h, w) order. A window with origin (oh - 1, ow - 1) reads the
pixels h*5 + w it covers and a zero wherever it hangs off the image. Row oh*4 + ow of the
result is that window in tap order. By that rule the last row (oh = 4, ow = 3, origin (3, 2))
covers only the input's bottom-right three pixels, 17 18 19, and is padding everywhere else.
The transposed grid would put 13 14 0 18 19 0 0 0 0 there.

## MaxPool2dTakesEachWindowsMaximum

This is the forward's own proof, for the same reason as the im2col case. N=1, H=2, W=2, C=1
holds the values 0, 1, 2, 3 in row-major (h, w) order. One 2x2 window covers the whole image,
and the maximum is 3.

## ALayoutConvertRoundTripActuallyPermutesAndComesBack

This case closes the gap that let a dropped layout tag hide. Every other layout test stops at the
tag. The numerical claim behind them is that a tag reaching `lower_layout_convert`
(`modules/core/graph/source/lowering/shape.cpp`) intact makes a transpose happen, and nothing tested
that half.

A round trip alone would not test it either. Two no-ops compose to the identity, and the final
assertion would pass. So the intermediate is read out and asserted, by value, to be the
permuted buffer. That assertion is the test; the round trip is the control.

The passes run before compilation, so the case also pins tag survival by its consequence and
not by inspecting the field.

The input is N=1, H=2, W=2, C=2 with values 1..8 in NHWC order, at index ((h*2)+w)*2+c. The
values are distinct because a tensor of repeats would make a permutation indistinguishable
from a copy.

[N, H, W, C] seen as [N, C, H, W] has the same extents here only because H, W and C are all 2.
The buffer differs.

The NCHW index is ((c*2)+h)*2+w, so channel 0 collects the odd values and channel 1 the even
ones. The expected buffer is worked out by hand and not read back from the implementation.

## AMaxPoolSurvivesTheFullPassPipelineAndStillRoutesItsGradient

K_MAXPOOL2D is the first operation whose results are not all one element type. Beside the pooled
tensor it writes an INT64 argmax, and its backward reads that as a table of offsets.
`insert_casts()` derived one precision per node from `outputs[0]` and forced every operand and
result to it. The argmax then read as a value in the wrong type, and was either retyped to
float in place or handed to a cast that refuses anything but real float to real float.

Neither half showed up in a shape check. The case runs the passes in the order
`Trainer::build` runs them and executes the result. A routing table that has been through a
float is not a routing table, and the gradient is where that shows.

The graph is assembled by hand and not through `Trainer::build`, because no NN layer emits a
pool yet and the trainer cannot reach this operation.

The input is N=1, H=2, W=2, C=2 with values 0..7 in NHWC order and one 2x2 window. Per channel
the maximum is in the last row: tap 6 (c=0, value 6) and tap 7 (c=1, value 7). Every other tap
must come back with no gradient.

## AvgPool2dDividesByTheFullWindowArea

A 2x2 average pool over the `[1, 2, 2, 1]` values 0, 1, 2 and 3 gives one value, 1.5.

## Upsample2dReplicatesEachPixelOverItsBlock

One pixel of value 5 upsampled by 2 in each direction becomes four values of 5.

## SpatialMeanAveragesEveryHWPlane

The spatial mean of the `[1, 2, 2, 1]` values 0, 1, 2 and 3 is one value, 1.5.

## ConcatChannelsPlacesEachOperandAtItsOwnOffset

Two one-channel values holding 10 and 20 concatenate to `{10, 20}`.

## SliceChannelsExtractsTheMiddleRangeUnchanged

Two channels from channel 1 of `{1, 2, 3, 4}` are `{2, 3}`.

## ChannelAffineScalesAndShiftsPerChannelNotPerRow

The input is N=1, H=1, W=2, C=2. Two pixels sharing one pair of parameters separate a
per-channel broadcast from a per-row one: K_BIAS_ADD's broadcast would give the second pixel the
second channel's numbers.

## BatchNorm2dNormalisesEachChannelOverTheWholeBatch

The input is N=2, H=2, W=1, C=2, so each channel is folded over four elements. The two channels
have different scales on purpose, because a normalisation that reduced over the wrong axis
would still produce four finite numbers.

Channel 0 sees 1, 2, 3, 4: mean 2.5, biased variance 1.25. Channel 1 sees 10, 20, 30, 40: mean
25, biased variance 125.

The variance is biased, dividing by N*H*W and not by N*H*W - 1. The forward normalises by that
value and the backward reads the saved statistics, so an unbiased divisor in the test would be
a different function.

## AnUnboundLeafIsRefusedRatherThanFilledWithGarbage

Compiling a graph whose input leaf has no tensor in the pool is a fatal stop. The message contains
"Leaf value", 'x' and "has no tensor bound".

## AnUnimplementedOperationNamesItself

A node with an operation identity that has no lowering makes the compile a fatal stop whose message
contains "No lowering for operation".
