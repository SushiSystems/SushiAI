# tests/vision/blocks/test_conv_norm.cpp


Suites: `Small_BlockConvNorm`.

Tests ConvNorm's trace, its state names against YOLO26n row 0, and make_conv_node.

This file needs no device. `NN::ConvNorm` in `modules/vision/blocks/include/SushiAI/vision/blocks/conv_norm.hpp` is a
bias-free same-padded `Conv2d`, a `BatchNorm2d` and an activation, traced in that order.
`NN::Conv` is `ConvNorm<Conv2d, SiLU>` and `NN::ConvLinear` is `ConvNorm<Conv2d, Identity>`.
`NN::make_conv_node` builds the node of a `conv` row of a model file.

The expected state names are typed into the test from
`tests/vision/registration/fixtures/yolo26/MANIFEST.json`. No case opens a fixture file.

## Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`, the epsilon and momentum `MANIFEST.json`
records for YOLO26n. `K_CONTEXT` is "layer 0 'conv'", the row name handed to the factory.

## StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

## record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

## walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

## count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

## first_node

`first_node` returns the index of the first node of a graph that carries one operation.

## one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

## attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

## OutputShapeHalvesAnEvenExtentAtStrideTwo

`Conv(3, 16, 3, 2)` maps [640, 640, 3] to [320, 320, 16].

## OutputShapeOfAPointwiseBlockKeepsTheExtents

`Conv(32, 64, 1, 1)` maps [7, 9, 32] to [7, 9, 64].

## ATrainingTraceEmitsConvolutionNormThenSiLU

A training trace of `Conv(3, 16, 3, 2)` over [2, 8, 8, 3] emits one K_MATMUL, one K_BATCHNORM2D
and one K_SILU, in that order, and no K_BIAS_ADD. The K_SILU's output is the returned value, of
shape [2, 4, 4, 16].

## ATrainingTraceCarriesTheNormOptions

The K_BATCHNORM2D node carries the epsilon 1e-3 in attribute 0, and both requested updates carry
the momentum 0.03.

## AnEvaluationTraceFoldsTheNorm

An evaluation trace emits one K_BATCHNORM_FOLD, one K_CHANNEL_AFFINE, one K_SILU and no K_BATCHNORM2D,
and returns a [2, 4, 4, 16] value.

## ConvLinearEmitsNoActivation

A training trace of `ConvLinear(3, 16, 1, 1)` emits one K_BATCHNORM2D and no K_SILU. The returned
value is output 0 of the K_BATCHNORM2D node.

## StateMatchesManifestRow0

The walk of a traced `Conv(3, 16, 3, 2)` under "model.0" yields five records in this order:
"model.0.conv.weight" [16, 3, 3, 3], "model.0.bn.weight" [16] and "model.0.bn.bias" [16] as
parameters, then "model.0.bn.running_mean" [16] and "model.0.bn.running_var" [16] as buffers.

## ADepthwiseTraceEmitsDepthwiseConvolutionNormThenSiLU

A training trace of `DWConv(16, 16, 3, 1)` over [2, 8, 8, 16] holds one DEPTHWISE_CONV2D, one
BATCHNORM2D and one SILU in that order and no MATMUL. The convolution reads the input, the
SILU writes the result, and the result has the input's shape.

## DWConvLinearEmitsNoActivation

A training trace of `DWConvLinear(16, 16, 3, 1)` holds one DEPTHWISE_CONV2D, one BATCHNORM2D
and no SILU, and the BATCHNORM2D writes the result.

## ADepthwiseOutputShapeKeepsTheWidth

For the input [80, 80, 64], a 3x3 `DWConv` reports [80, 80, 64] at stride 1 and [40, 40, 64]
at stride 2.

## DepthwiseStateMatchesTheManifestPositionalConvolution

After a training trace of `DWConvLinear(128, 128, 3, 1)`, the state walk under
"model.10.m.0.attn.pe" gives five records in order: the parameters `conv.weight` [3, 3, 128],
`bn.weight` [128] and `bn.bias` [128], then the buffers `bn.running_mean` [128] and
`bn.running_var` [128].

## ADepthwiseBlockRefusesUnequalWidths

Constructing `DWConv(16, 32, 3, 1)` raises an error holding "keeps its channel count",
"in=16" and "out=32".

## AnUntracedBlockYieldsNoState

The walk of a block that was never traced is empty.

## TheFactoryBuildsRow0

`make_conv_node` over [640, 640, 3] with 16 filters, kernel 3, stride 2, `bn_epsilon` 0.001 and
`bn_momentum` 0.03 returns a node with one port named `NN::K_SINGLE_PORT_NAME` of shape
[320, 320, 16]. A training trace over [1, 640, 640, 3] appends one value of shape
[1, 320, 320, 16], the K_BATCHNORM2D carries 1e-3, and the first update carries 0.03. The walk
under "model.0" yields five records, the first "model.0.conv.weight" [16, 3, 3, 3] and the
last "model.0.bn.running_var" [16].

## TheFactoryDefaultsKernelAndStrideToOne

A row stating only 8 filters over [6, 10, 3] has a port of shape [6, 10, 8]. After a trace its
first state record is "conv.weight" [8, 1, 1, 3].

## TheFactoryRefusesAnUnknownKey

The key `groups` fails with `InvalidDescription` and a message containing "layer 0 'conv'",
"'groups'" and "padding", the last because it is among the accepted keys.

## TheFactoryDefaultsPaddingToHalfTheKernel

A row of 16 filters, kernel 3, stride 2 over [640, 640, 3] has a port of shape [320, 320, 16].

## TheFactoryHonoursAStatedPadding

A row of 16 filters, kernel 6, stride 2, padding 2 over [640, 640, 3] has a port of shape
[320, 320, 16], and its evaluation trace over [1, 640, 640, 3] yields [1, 320, 320, 16].

## TheFactoryAcceptsAPaddingOfZero

A row of 4 filters, kernel 3, padding 0 over [8, 8, 3] has a port of shape [6, 6, 4].

## TheFactoryRefusesAPaddingThatIsNotAWholeNumberOfZeroOrMore

A `padding` of -1 fails with "'padding'", "zero or more" and "-1". A `padding` given as the
string "1" fails with "'padding'" and "whole number".

## TheFactoryRefusesARowWithoutFilters

A row without `filters` fails with a message containing the context, "'filters'" and "absent".

## TheFactoryRefusesAWindowThatIsNotAPositiveWholeNumber

A `stride` of 0 fails with "'stride'", "positive" and "0". A `kernel` given as the string "3"
fails with "'kernel'" and "whole number".

## TheFactoryRefusesNormOptionsOutOfRange

A `bn_epsilon` of 0.0 fails with "'bn_epsilon'" and "positive", and one given as a string
fails with "'bn_epsilon'" and "number". A `bn_momentum` of 1.5 fails with "'bn_momentum'" and
"[0, 1]", and one of -0.1 fails with those and "-0.1".

## TheFactoryRefusesInputsThatAreNotOneFeatureMap

Two input shapes fail with a message containing the context, "one input" and "2". The one
shape [192] fails with "[192]" and "rank-3".
