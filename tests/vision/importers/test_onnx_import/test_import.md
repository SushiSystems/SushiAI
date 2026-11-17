# tests/vision/importers/test_onnx_import/test_import.cpp

Suites: `Small_OnnxImport`, `Small_OnnxImportBinary`, `Small_OnnxImportSoftmax`, `Small_OnnxImportTranspose`, `Medium_OnnxImportFixture`.

The Small cases build an `Interchange::OnnxGraph` in code, so no file or engine is involved. The
Medium cases read `fixtures/onnx/silu_resize.onnx`, which `tools/reference/make_onnx_fixture.py`
writes, with `silu_resize.expected.json` beside it: the fixed input and the output that
`onnx.reference.ReferenceEvaluator` computed for it.

## EveryUnsupportedNodeIsListedTogether

A graph of `Relu`, `Erf`, `Pow` and `Where` fails with `Unsupported`. The one message names the
origin, says "3 nodes" and lists `'gauss' (Erf)`, `'power' (Pow)` and `'choose' (Where)`. The `Pow`
reads the output of the refused `Erf` and is still listed, because its operator has no row at all.

## AGroupedConvolutionWritesTheGroupsKeyAndTheGroupWeight

Two filters over four channels in two groups, with a window of 1 by 2, write one `conv2d` row
whose `groups` is 2 and the weight `0.weight` of [2, 1, 2, 2]. The ONNX weight holds
`o * 100 + i * 10 + x`, so the record, ordered (o, x, i), reads 0, 10, 1, 11, 100, 110, 101, 111.

## ADepthwiseConvolutionStoresItsTapsKernelHeightWidthChannel

Two channels with one filter each (`group` 2) store the weight as [1, 2, 2]. Channel 0 holds the
taps 10, 11 and channel 1 the taps 20, 21, so the record reads 10, 20, 11, 21.

## AConvolutionOfOneGroupStatesNoGroupsKey

A `Conv` without `group` writes no `groups` key.

## AGroupThatDoesNotDivideTheFiltersIsRefused

Three filters in two groups fail with `InvalidDescription` naming `'grouped' (Conv)`, "3 filters"
and "2 groups".

## AWeightThatDoesNotReadTheChannelsPerGroupIsRefused

A weight of 3 channels in two groups over 4 channels fails with `InvalidDescription` naming the
4 channels of the value.

## ADepthwiseConvolutionWithUnequalPadsIsRefused

A depthwise `Conv` with `pads` 0, 0, 1, 1 fails with `Unsupported` naming "depthwise", since the
depthwise row takes no trailing pad.

## TheOperatorBecomesOneRowReadingItsOperandsInOrder

Suite `Small_OnnxImportBinary`, rows `Sub`, `Div` and `Mul`. The operator over a `Relu` and the
input, in that order, writes one `sub`, `div` or `mul` row after the `relu` row, whose sources are row 0 then
the model input.

## ValuesOfDifferentExtentsAreRefused

Suite `Small_OnnxImportBinary`. The input [3, 4, 4] and a pooled value [3, 2, 2] fail with
`Unsupported` naming the node, "different extents" and "<row> row does not broadcast".

## AConstantOperandIsRefused

Suite `Small_OnnxImportBinary`. An initializer as an operand fails with `Unsupported` naming the
node and the constant.

## TheAxisMovesToTheChannelLastOrder

Suite `Small_OnnxImportSoftmax`, over [1, 3, 4, 4]. The channels (axis 1 or -3) are last in a
sample [H, W, C] and state no `axis`; the rows (2 or -2) state `axis` 0 and the columns (3 or -1)
state `axis` 1.

## ASoftmaxWithNoAxisNormalisesTheColumns

With no `axis` the default -1 of opset 13 normalises the columns, so the row states `axis` 1.

## ASoftmaxOverAVectorStatesNoAxis

A `Softmax` after a `Flatten` writes `flatten`, `softmax`, and the second row states no `axis`.

## ASoftmaxOverTheBatchAxisIsRefused

`axis` 0 fails with `Unsupported` naming `'soft' (Softmax)` and the batch axis.

## ASoftmaxOverAnAxisBeyondTheTensorIsRefused

`axis` 4 over a tensor of rank 4 fails with `InvalidDescription` naming axis 4.

## ThePermutationMovesToTheChannelLastOrder

Suite `Small_OnnxImportTranspose`, over [1, 2, 3, 4]. The channel-first tensor [N, C, H, W] is
the physical [N, H, W, C], so a `perm` of 0, 2, 3, 1 becomes the row's `perm` 1, 2, 0 over a
sample [H, W, C], 0, 1, 3, 2 becomes 1, 0, 2 and 0, 3, 1, 2 becomes 2, 0, 1.

## ATransposeThatMovesNoAxisWritesNoRow

`perm` 0, 1, 2, 3 followed by a `Relu` writes the `relu` row alone.

## TheExtentsAfterATransposeAreThePermutedOnes

A `Transpose` of [2, 3, 4] by 0, 2, 3, 1, then a `Flatten` and a `Gemm` of 24 features, writes the
rows `transpose`, `flatten` and `linear`; the `Gemm` finds the 24 features of the permuted extents.

## ATransposeOfTheBatchAxisIsRefused

A `Transpose` with no `perm` reverses the axes and so moves the batch axis; it fails with
`Unsupported` naming the batch axis.

## ATransposeThatListsTooFewAxesIsRefused

A `perm` of three entries over four axes fails with `InvalidDescription` naming "3 axes".

## ATransposeOfAVectorIsRefused

A `Transpose` of a flattened vector fails with `Unsupported` naming the rank of the value.

## AMaxPoolWithEqualPadsStatesPadding

`pads` 1, 1, 1, 1 write the key `padding` 1 and no `leading_padding` or `end_padding`.

## AMaxPoolWithUnequalPadsStatesLeadingAndEndPadding

`pads` 1, 0, 0, 2, which list the pads before the rows and the columns and then after them,
write `leading_padding` [1, 0] and `end_padding` [0, 2] and no `padding`.

## AnOldOpsetIsRefused

A graph of opset 12 fails with `Unsupported`, naming "opset 12" and 13.

## TwoFilesAreRefused

`import_onnx_files` over two paths fails with `InvalidDescription` naming `'onnx'` and "2 files",
before either file is opened.

## AMissingFileIsNotFound

A path that is not there fails with `FileNotFound` naming the file.

## AReluGraphIsOneRowOverAChannelLastInput

An input `[1, 3, 5, 7]` gives the model input `[5, 7, 3]`, one `relu` row, the output published as
`y`, and no weights.

## AFoldedConstantShapeMakesTheReshapeAFlatten

A `Constant` holding the int64 tensor `[1, -1]` feeds a `Reshape` of an image `[1, 2, 3, 3]`. The
folding removes the `Constant`; the `Reshape` writes one `flatten` row.

## AnEqualSplitOfTheChannelsIsOneSliceRowPerOutput

A `Split` along axis 1 of 4 channels into two outputs writes two `slice_channels` rows, at offsets
0 and 2, each of count 2.

## AGemmWeightAfterAFlattenFollowsTheChannelLastOrder

An image of 2 channels, 1 row and 2 columns is flattened and passed to a `Gemm` with `transB` 1.
ONNX orders the four features c0w0, c0w1, c1w0, c1w1, so the weight 10, 20, 30, 40 gives
the imported weight `1.weight` of shape `[1, 4]` the values 10, 30, 20, 40: the order of the
flatten row, c0w0, c1w0, c0w1, c1w1. The oracle is that index rule, worked by hand.

## TheFixtureImportsToTheRowsOfItsPattern

Medium. The fixture imports to the rows `conv2d`, `batchnorm2d`, `silu`, `upsample`, `concat`,
`conv2d`, `flatten`, with the model input `[8, 8, 3]`. The weights are `0.weight` `[4, 3, 3, 3]` (the
filter `[out, kh, kw, in]`), `0.bias`, `1.weight`, `1.bias`, `1.running_mean`, `1.running_var` and
`5.weight` `[2, 1, 1, 7]`, in that order. No `sigmoid` row appears: the `Sigmoid` and `Mul` became
the one `silu`.

## TheForwardEqualsTheReferenceEvaluator

Medium. The fixture is built, its weights loaded by name, and one evaluation step run on the fixed
input moved from `[1, 3, 8, 8]` to `[1, 8, 8, 3]`. Each of the 128 output values equals the value
`onnx.reference` computed for the same channel and pixel, where the model lists the row as (H, W, C)
and the reference as (C, H, W).

The bound of one value is `46 u` times the largest magnitude of the reference output, and a value
passes within the helper's threshold of 16 times that. The 46 is the rounded operations on the
longest path: 27 products of the 3x3 convolution over 3 channels, 4 of the batch normalisation, 8 of
the SiLU and 7 products of the 1x1 convolution over 7 channels.
