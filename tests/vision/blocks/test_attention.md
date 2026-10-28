# tests/vision/blocks/test_attention.cpp


Suites: `Small_BlockAttention`.

Tests the `Attention` block at the width of row 10 of YOLO26n: 128 channels, two heads, a query and
key width of 32 and a value width of 64 per head. The traced cases run the block in training mode
over a [2, 4, 4, 128] input, which is 16 positions per sample. The norm options are the epsilon 1e-3
and the momentum 0.03 that `tests/vision/registration/fixtures/yolo26/MANIFEST.json` records.

## TwoHeadsOver128ChannelsGiveAHeadOf64AndAKeyOf32

`Attention(128, 2)` has `heads` 2, `head_dim` 64 and `key_dim` 32. Its `qkv` convolution has
256 output channels, and its `pe` convolution has a kernel height of 3 and a top padding of 1.

## OutputShapeIsTheInputShape

`output_shape([20, 20, 128])` is [20, 20, 128].

## TwoHeadsEmitSixSlicesTwoScalesFourProductsTwoSoftmaxesAndOneAdd

The training trace holds 6 SLICE_CHANNELS, 2 SCALE, 4 MATMUL over rank-3 operands,
2 LOG_SOFTMAX, 2 EXP, 1 ADD, 1 DEPTHWISE_CONV2D, 2 CONCAT_CHANNELS and 3 BATCHNORM2D nodes, and
no SILU node. The result has the shape [2, 4, 4, 128].

## TheScoreProductReadsScaledQueriesThenTransposedKeys

For each head, the first of its two products has transpose flags 0 and 1 and yields
[2, 16, 16]. Its first operand is a SCALE of the head's queries and its second is the head's
keys. Each is a RESHAPE to [2, 16, 32] of a slice of `qkv`'s normalised result, the queries
from channel `head * 128` and the keys 32 channels further on.

## TheValueProductReadsTheSoftmaxOfTheScoresThenTheValues

For each head, the second product has both transpose flags 0 and yields [2, 16, 64]. Its first
operand is the RESHAPE, EXP, LOG_SOFTMAX, RESHAPE chain over the head's scores, with
LOG_SOFTMAX reading a [32, 16] value. Its second operand is a RESHAPE to [2, 16, 64] of a
64-channel slice of `qkv`'s result that starts at channel `head * 128 + 64`.

## TheQueriesAreScaledByTheInverseRootOfTheKeyWidth

Both SCALE nodes carry 1 / sqrt(32) in attribute 0, which is 0.17677669529663687, to 1e-15
absolute.

## TheAddReadsTheAttendedHeadsThenThePositionalConvolution

The one ADD reads two values. The first is a CONCAT_CHANNELS of two [2, 4, 4, 64] maps, each a
RESHAPE of a head's value product. The second is the BATCHNORM2D of the DEPTHWISE_CONV2D over a
CONCAT_CHANNELS of both heads' value slices, a [2, 4, 4, 128] value. One RESHAPE reads the ADD,
and the block's result comes out of a BATCHNORM2D.

## StateMatchesManifestRecords175To189

Walked under the prefix "model.10.m.0.attn", the traced block yields fifteen records in order:
a convolution weight and four norm records each for `qkv` ([256, 1, 1, 128], width 256), `proj`
([128, 1, 1, 128], width 128) and `pe` ([3, 3, 128], width 128). In each group the norm's
weight and bias are parameters and its running mean and variance are buffers.

## AnEvaluationTraceFoldsTheThreeNorms

An evaluation trace holds no BATCHNORM2D node, three BATCHNORM_FOLD nodes and four rank-3
products, and the result has the shape [2, 4, 4, 128].

## AWidthTheHeadCountDoesNotDivideIsRefused

`Attention(100, 3)` stops the process naming "head count divides", "100 channels" and "3 heads".
`Attention(128, 0)` stops the process one naming "128 channels" and "0 heads".

## OneChannelPerHeadIsRefused

`Attention(2, 2)` stops the process naming "at least 2 channels per head", "2 channels" and
"2 heads".
