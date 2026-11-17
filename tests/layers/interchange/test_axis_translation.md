# tests/layers/interchange/test_axis_translation.cpp

Suites: `Small_ChannelsLastAxis`, `Small_AxisTranslation`.

Tests the functions of `modules/layers/interchange/include/SushiAI/interchange/axis_translation.hpp`.
The oracle is the layout rule written out by hand: the channel axis (1) moves to the end, every
axis after it moves one place down, a negative axis counts from the end first. For rank 4 that is
0 to 0, 1 to 3, 2 to 1 and 3 to 2. A filter [out, in, kh, kw] stored [out, kh, kw, in] has its
element at `((o * kh + y) * kw + x) * in + i`; the test fills the weight with
`o * 100 + i * 10 + y * 2 + x`, so each element names where it came from.

## MovesTheAxisAsTheLayoutRuleSays

A table over rank 1, 2, 3 and 4 with positive and negative axes, including -4, -3 and -1 of rank 4.

## RefusesAnAxisOutOfRange

Axis 4 and -5 of rank 4 fail with `InvalidDescription`.

## RewritesTheAxesOfConcatSoftmaxSliceAndReduce

Concat axis 1 gives 3, Softmax axis -1 gives 2, Slice axes 2 and 3 give 1 and 2, and ReduceMean
axes 2 and 3 give 1 and 2, all through `translate_axis_attribute`.

## LeavesAttributesThatHoldNoAxisAlone

A Conv's `strides` passes unchanged and its kind is `None`.

## RewritesATransposePermutation

The identity stays the identity; the swap of the two spatial axes [0, 1, 3, 2] becomes the swap
of the two middle axes [0, 2, 1, 3]; the move of the channels to the end [0, 2, 3, 1] stays
[0, 2, 3, 1].

## RefusesAPermutationThatRepeatsOrOverflows

[0, 1, 1, 3] fails naming the repeated axis; [0, 1, 2, 4] fails naming the entry 4 and rank 4.

## RewritesAReshapeTarget

[0, 3, 8, -1] becomes [0, 8, -1, 3], and [1, 3, 8, 8] becomes [1, 8, 8, 3] through the dispatch.

## RewritesAFlattenAxisOnlyWhereItKeepsItsMeaning

Axes 0, 1, -3 and 4 of rank 4 give 0, 1, 1 and 4; axis 2 fails with `Unsupported`.

## PermutesAnOIHWWeightToTheRecordLayout

A [2, 3, 2, 2] weight becomes [2, 2, 2, 3]; every element is compared against its formula.

## PermutesADepthwiseWeightToKernelHeightWidthChannel

A [3, 1, 2, 2] weight becomes [2, 2, 3]; every element is compared against its formula.

## RefusesAFilterThatIsNotAConvolutionWeight

Three extents, a zero extent, a depthwise weight that reads two channels and five floats for eight
elements each fail with `InvalidDescription` naming the cause.

## MovesTheInputsOfAnInnerProductWeightToChannelsLast

Two units over 2 channels and a 2 by 3 image: the input at channel c, row y and column x of
unit u is numbered `u * 100 + c * 10 + y * 3 + x`, and the moved weight holds it at
`u * 12 + (y * 3 + x) * 2 + c`. Every element is compared against that formula. This is the
permutation the Caffe InnerProduct and the ONNX Gemm translators share.

## AnInnerProductWeightOfOneChannelIsUnchanged

With one channel the [C, H, W] and the [H, W, C] orders agree, so the weight comes back as it
went in.

## RefusesAnInnerProductWeightThatDoesNotFitItsExtents

A zero channel count fails with "positive extents", and five floats for two units of three inputs
fail with "5 floats" and "need 6"; both are `InvalidDescription`.
