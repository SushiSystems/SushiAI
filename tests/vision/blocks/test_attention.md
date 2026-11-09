# tests/vision/blocks/test_attention.cpp


Suites: `Small_BlockAttention`, `Small_BlockAttentionHeads` and `Medium_BlockAttentionDeath`.

Tests the `Attention` block at the width of row 10 of YOLO26n: 128 channels, two heads, a query and
key width of 32 and a value width of 64 per head. The traced cases run the block in training mode
over a [2, 4, 4, 128] input, which is 16 positions per sample. The norm options are the epsilon 1e-3
and the momentum 0.03 that `tests/vision/registration/fixtures/yolo26/MANIFEST.json` records.

## TwoHeadsOver128ChannelsGiveAHeadOf64AndAKeyOf32

`Attention(128, 2)` has `heads` 2, `head_dim` 64 and `key_dim` 32. Its `qkv` convolution has
256 output channels, and its `pe` convolution has a kernel height of 3 and a top padding of 1.

## OutputShapeIsTheInputShape

`output_shape([20, 20, 128])` is [20, 20, 128].

## ATrainingTraceKeepsTheInputShapeAndUpdatesTheThreeNorms

The training trace yields a value of the shape [2, 4, 4, 128] and requests six buffer updates, a
running mean and a running variance for each of the norms of `qkv`, `proj` and `pe`.

## EachHeadScoresEveryPositionAgainstEveryOther

Two rank-3 products yield [2, 16, 16], one per head.

## EachHeadWeighsItsValuesAsABatchOfTokenRows

Two rank-3 products yield [2, 16, 64], one per head.

## TheQueriesAreScaledByTheInverseRootOfTheKeyWidth

The factors of the SCALE nodes equal one 1 / sqrt(32) per head, exactly. The first is also within
two roundings, in `TestSupport::measure_relative_error`, of the literal 0.17677669529663687.

## TheAttendedHeadsAndThePositionalConvolutionAddUpToTheInputShape

The one ADD yields a value of the shape [2, 4, 4, 128].

## StateMatchesManifestRecords175To189

Walked under the prefix "model.10.m.0.attn", the traced block yields fifteen records in order:
a convolution weight and four norm records each for `qkv` ([256, 1, 1, 128], width 256), `proj`
([128, 1, 1, 128], width 128) and `pe` ([3, 3, 128], width 128). In each group the norm's
weight and bias are parameters and its running mean and variance are buffers.

## AnEvaluationTraceLeavesNoRunningStatisticsNode

An evaluation trace holds no BATCHNORM2D node and yields the shape [2, 4, 4, 128].

## TheHeadsOfRow10AreAccepted

`check_attention_heads(128, 2)` is a success.

## ALayoutThatDoesNotFitIsRefusedNamingTheCounts

`check_attention_heads` refuses three layouts with InvalidDescription, each naming its counts:
`AWidthTheHeadCountDoesNotDivide` (100 channels, 3 heads, "head count divides"), `NoHeads`
(128 channels, 0 heads) and `OneChannelPerHead` (2 channels, 2 heads,
"at least 2 channels per head").

## AWidthTheHeadCountDoesNotDivideIsFatal

`Attention(100, 3)` stops the process naming "head count divides", "100 channels" and "3 heads".
This is a death test, so it is `Medium_`.

## OneChannelPerHeadIsFatal

`Attention(2, 2)` stops the process naming "at least 2 channels per head", "2 channels" and
"2 heads".

## EachHeadAttendsOverItsOwnValues

Runs on `Graph::HostEvaluator`. `Attention(4, 2)` over two tokens (1, 2, 3, 4) and (5, 6, 7, 8) as [1, 1, 2, 4], with queries and keys zero, norms that are the identity, a projection that is the identity and a zero positional convolution. Head 0 takes its values from channels 0 and 1 and head 1 from channels 2 and 3, and every token weighs both tokens 1/2, so each output token is (3, 4, 5, 6); two heads that read each other's values would give (5, 6, 3, 4).
