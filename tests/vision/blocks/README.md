# tests/vision/blocks

The tests of the module `vision/blocks`, one file per header of `modules/vision/blocks/include/`. The suite prefix of a case gives its ctest label.

## test_attention.cpp

Suites: `Unit_BlockAttention`.

Tests the `Attention` block at the width of row 10 of YOLO26n: 128 channels, two heads, a
query and key width of 32 and a value width of 64 per head. The traced cases run the block in
training mode over a [2, 4, 4, 128] input, which is 16 positions per sample. The norm options
are the epsilon 1e-3 and the momentum 0.03 that `tests/fixtures/yolo26/MANIFEST.json` records.

### TwoHeadsOver128ChannelsGiveAHeadOf64AndAKeyOf32

`Attention(128, 2)` has `heads` 2, `head_dim` 64 and `key_dim` 32. Its `qkv` convolution has
256 output channels, and its `pe` convolution has a kernel height of 3 and a top padding of 1.

### OutputShapeIsTheInputShape

`output_shape([20, 20, 128])` is [20, 20, 128].

### TwoHeadsEmitSixSlicesTwoScalesFourProductsTwoSoftmaxesAndOneAdd

The training trace holds 6 SLICE_CHANNELS, 2 SCALE, 4 MATMUL over rank-3 operands,
2 LOG_SOFTMAX, 2 EXP, 1 ADD, 1 DEPTHWISE_CONV2D, 2 CONCAT_CHANNELS and 3 BATCHNORM2D nodes, and
no SILU node. The result has the shape [2, 4, 4, 128].

### TheScoreProductReadsScaledQueriesThenTransposedKeys

For each head, the first of its two products has transpose flags 0 and 1 and yields
[2, 16, 16]. Its first operand is a SCALE of the head's queries and its second is the head's
keys. Each is a RESHAPE to [2, 16, 32] of a slice of `qkv`'s normalised result, the queries
from channel `head * 128` and the keys 32 channels further on.

### TheValueProductReadsTheSoftmaxOfTheScoresThenTheValues

For each head, the second product has both transpose flags 0 and yields [2, 16, 64]. Its first
operand is the RESHAPE, EXP, LOG_SOFTMAX, RESHAPE chain over the head's scores, with
LOG_SOFTMAX reading a [32, 16] value. Its second operand is a RESHAPE to [2, 16, 64] of a
64-channel slice of `qkv`'s result that starts at channel `head * 128 + 64`.

### TheQueriesAreScaledByTheInverseRootOfTheKeyWidth

Both SCALE nodes carry 1 / sqrt(32) in attribute 0, which is 0.17677669529663687, to 1e-15
absolute.

### TheAddReadsTheAttendedHeadsThenThePositionalConvolution

The one ADD reads two values. The first is a CONCAT_CHANNELS of two [2, 4, 4, 64] maps, each a
RESHAPE of a head's value product. The second is the BATCHNORM2D of the DEPTHWISE_CONV2D over a
CONCAT_CHANNELS of both heads' value slices, a [2, 4, 4, 128] value. One RESHAPE reads the ADD,
and the block's result comes out of a BATCHNORM2D.

### StateMatchesManifestRecords175To189

Walked under the prefix "model.10.m.0.attn", the traced block yields fifteen records in order:
a convolution weight and four norm records each for `qkv` ([256, 1, 1, 128], width 256), `proj`
([128, 1, 1, 128], width 128) and `pe` ([3, 3, 128], width 128). In each group the norm's
weight and bias are parameters and its running mean and variance are buffers.

### AnEvaluationTraceFoldsTheThreeNorms

An evaluation trace holds no BATCHNORM2D node, three BATCHNORM_FOLD nodes and four rank-3
products, and the result has the shape [2, 4, 4, 128].

### AWidthTheHeadCountDoesNotDivideIsRefused

`Attention(100, 3)` throws an error naming "head count divides", "100 channels" and "3 heads".
`Attention(128, 0)` throws one naming "128 channels" and "0 heads".

### OneChannelPerHeadIsRefused

`Attention(2, 2)` throws an error naming "at least 2 channels per head", "2 channels" and
"2 heads".

## test_bottleneck.cpp

Suites: `Unit_BlockBottleneck`.

Tests Bottleneck's trace, its state names against YOLO26n row 2, and its factory.

This file needs no device. `NN::Bottleneck` in `modules/vision/blocks/include/SushiAI/vision/blocks/bottleneck.hpp` is
two `Conv` blocks, `cv1` and `cv2`, and an optional K_ADD of the block's input to the second
one's result. `NN::make_bottleneck_node` builds the node of a `bottleneck` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 4 'bottleneck'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`Bottleneck(16, 24, 8, true)` maps [5, 7, 16] to [5, 7, 24].

### AShortcutOverEqualWidthsAddsTheInput

A training trace of `Bottleneck(16, 16, 8, true)` reports `adds_input` true and emits two
K_MATMUL, two K_BATCHNORM2D, two K_SILU and one K_ADD. The K_ADD's first operand is the block's input and
its second is produced by a K_SILU. Its output is the returned value, of shape [2, 4, 4, 16].

### NoShortcutTracesNoAdd

With the shortcut argument false, `adds_input` is false and the trace holds two K_MATMUL and no
K_ADD.

### AShortcutOverUnequalWidthsTracesNoAdd

`Bottleneck(16, 24, 8, true)` reports `adds_input` false and traces no K_ADD, although a shortcut
was asked for. The result has the shape [2, 4, 4, 24].

### StateMatchesManifestRow2Inner

The walk of a traced `Bottleneck(16, 16, 8)` under "model.2.m.0" yields ten records. The five
of `cv1` come first: "cv1.conv.weight" [8, 3, 3, 16], then "cv1.bn.weight", "cv1.bn.bias",
"cv1.bn.running_mean" and "cv1.bn.running_var", each [8]. The five of `cv2` follow in the same
order with the weight [16, 3, 3, 8] and the rest [16].

### EachKernelSetsItsOwnConvolution

`Bottleneck(16, 16, 8, false, K_YOLO_NORM, 1, 3)` traces a `cv1` weight of [8, 1, 1, 16] and a
`cv2` weight of [16, 3, 3, 8].

### TheFactoryBuildsTheBlockAndDefaultsTheShortcutToTrue

`make_bottleneck_node` over [4, 4, 16] with 16 filters and a hidden width of 8 has one port of
shape [4, 4, 16]. Its training trace holds one K_ADD, so a row without `shortcut` adds its input.
The walk under "model.2.m.0" yields ten records; record 0 is the `cv1` weight [8, 3, 3, 16]
and record 5 the `cv2` weight [16, 3, 3, 8].

### TheFactoryReadsTheShortcut

With `"shortcut": false` the trace holds no K_ADD.

### TheFactoryRefusesARowThatDoesNotFit

The key `kernel` fails with a message containing "layer 4 'bottleneck'" and "'kernel'". A row
without `hidden` fails with "'hidden'" and "absent". A `shortcut` given as the string "yes"
fails with "'shortcut'" and "true or false".

### TheFactoryRefusesAnInputThatIsNotAFeatureMap

The input shape [256] fails with "[256]" and "rank-3".

### TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.

## test_c2psa.cpp

Suites: `Unit_BlockC2psa`.

Tests the `C2PSA` block and its factory `Vision::Blocks::make_c2psa_node` at row 10 of YOLO26n:
256 channels in and out and a hidden width of 128. The traced cases use a [2, 4, 4, 256] input
and the norm options `tests/fixtures/yolo26/MANIFEST.json` records, epsilon 1e-3 and momentum
0.03.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`output_shape([20, 20, 256])` is [20, 20, 256].

### Row10EmitsEightSlicesThreeAddsAndThreeJoins

A training trace holds 8 SLICE_CHANNELS, 3 ADD, 3 CONCAT_CHANNELS, 3 SILU and 7 BATCHNORM2D
nodes, and the update list holds 14 updates. The result has the shape [2, 4, 4, 256]. Every
BATCHNORM2D node carries the epsilon 1e-3 and every update the momentum 0.03.

### TheJoinKeepsTheFirstHalfAndChainsTheSecondOffTheSecondSlice

The first two slices read the same value, at the offsets 0 and 128. The second ADD of the
graph reads the second slice as its first input. The third CONCAT_CHANNELS reads the first
slice and then the third ADD's result, and yields [2, 4, 4, 256].

### EachRepeatReadsTheResultOfThePreviousOne

With two repeats the block holds two inner blocks and an evaluation trace holds six ADD nodes.
The fifth ADD reads the third ADD's result as its first input.

### StateMatchesManifestRow10

Walked under the prefix "model.10" after a training trace, the block yields the 35 records the
file lists by name, shape, kind and order, from `model.10.cv1.conv.weight` on.

### TheFactoryBuildsRow10

`make_c2psa_node` over a [20, 20, 256] input, with the row's filters, hidden width, repeats,
heads and norm constants, returns a node with one port of the shape [20, 20, 256]. Its training
trace holds 7 BATCHNORM2D nodes with the row's epsilon and 14 updates with the row's momentum,
and its state walk yields the same 35 records.

### TheFactoryDefaultsToOneBlockAndHiddenOver64Heads

A row with only "filters" 256 and "hidden" 128 traces to two LOG_SOFTMAX nodes and three ADD
nodes, and its state walk yields 35 records.

### TheFactoryGivesOneHeadForAHiddenWidthUnder128

A row with "filters" 64 and "hidden" 64 over a 64-channel input traces to one LOG_SOFTMAX node.

### TheFactoryRefusesAFilterWidthThatIsNotTheInput

"filters" 128 over a 256-channel input fails with `InvalidDescription`, and the message holds
"'filters'", "128" and "256 channels".

### TheFactoryRefusesAHiddenWidthTheHeadsDoNotSplit

A hidden width of 128 with 3 heads, and with 128 heads, each fail with `InvalidDescription`,
"'hidden'" and "parts of at least 2 channels".

### TheFactoryRefusesARowThatDoesNotFit

An unknown key "inner" fails with a message holding "layer 10 'c2psa'" and "'inner'". A row
without "hidden" fails with "'hidden'" and "absent". A [512] input fails with "[512]" and
"rank-3".

## test_c3k.cpp

Suites: `Unit_BlockC3k`.

Tests C3k's trace, its state names against YOLO26n row 6, and its factory.

This file needs no device. `NN::C3k` in `modules/vision/blocks/include/SushiAI/vision/blocks/c3k.hpp` holds three `Conv`
blocks, `cv1`, `cv2` and `cv3`, and a `ModuleList` of `Bottleneck`s named `m`. The cases show
its trace: a chain of bottlenecks on one branch, `cv2` on the other, one K_CONCAT_CHANNELS of the
two, and `cv3` after it. `NN::make_c3k_node` builds the node of a `c3k` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 5 'c3k'", the row name handed
to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`C3k(64, 96, 2, true)` maps [5, 7, 64] to [5, 7, 96].

### TheChainHoldsOneBottleneckPerRepeat

`C3k(64, 64, 3, true)` has `m.size()` 3.

### ATrainingTraceEmitsSevenConvBlocksTwoAddsAndOneConcat

A training trace of `C3k(64, 64, 2, true)` over [2, 4, 4, 64] emits seven each of K_MATMUL,
K_BATCHNORM2D and K_SILU, two K_ADD, one K_CONCAT_CHANNELS and no K_SLICE_CHANNELS. The list holds 14
updates and the result has the shape [2, 4, 4, 64]. Seven is three `Conv` blocks of the C3k
plus two in each of the two bottlenecks.

### TheJoinReadsTheChainThenTheSecondBranch

The K_CONCAT_CHANNELS node has two inputs. The first is produced by an K_ADD, the end of the last
bottleneck, and the second by a K_SILU, the end of `cv2`. Each is [2, 4, 4, 32] and the joined
value is [2, 4, 4, 64].

### NoShortcutTracesNoAdd

With the shortcut argument false, an evaluation trace holds seven K_MATMUL and no K_ADD.

### TheHiddenWidthIsHalfTheOutputWidthRoundedDown

`C3k(8, 7, 1, true)` has a hidden width of 3. The traced weights are [3, 1, 1, 8] for `cv1`
and `cv2`, [7, 1, 1, 6] for `cv3` and [3, 3, 3, 3] for the bottleneck's `cv1`. The result is
[2, 4, 4, 7], and `output_shape([4, 4, 8])` is [4, 4, 7].

### StateMatchesManifestRow6Inner

The walk of a traced `C3k(64, 64, 2)` under "model.6.m.0" yields 35 records: five each for
`cv1`, `cv2`, `cv3`, `m.0.cv1`, `m.0.cv2`, `m.1.cv1` and `m.1.cv2`, in that order. Each group
is the convolution weight, the norm's weight and bias as parameters, then its running mean and
variance as buffers. The weights are [32, 1, 1, 64] for `cv1` and `cv2`, [64, 1, 1, 64] for
`cv3` and [32, 3, 3, 32] for the four bottleneck convolutions.

### TheFactoryDefaultsToOneBottleneckWithAShortcut

`make_c3k_node` over [4, 4, 64] with only 32 filters has one port of shape [4, 4, 32]. Its
evaluation trace appends one [2, 4, 4, 32] value and holds five K_MATMUL and one K_ADD: one
bottleneck that adds its input.

### TheFactoryReadsRepeatsAndShortcut

With 64 filters, `repeats` 2 and `shortcut` false, the trace holds seven K_MATMUL and no K_ADD. The
walk under "model.6.m.0" yields 35 records, and record 30 is "model.6.m.0.m.1.cv2.conv.weight"
[32, 3, 3, 32].

### TheFactoryRefusesAWidthThatLeavesNoHiddenChannel

`filters` 1 fails with `InvalidDescription` and a message containing "layer 5 'c3k'",
"'filters'" and "at least 2".

### TheFactoryRefusesARowThatDoesNotFit

The key `hidden` fails with the context and "'hidden'". A row without `filters` fails with
"'filters'" and "absent". `repeats` 0 fails with "'repeats'" and "positive".

### TheFactoryRefusesTwoInputs

Two input shapes fail with a message containing "one input" and "2".

### TheFactoryHandsTheNormOptionsToEveryNorm

A row with `repeats` 2, `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces seven K_BATCHNORM2D
nodes and 14 updates, and `expect_norm_options` finds those two values on all of them.

## test_c3k2.cpp

Suites: `Unit_BlockC3k2`.

Tests C3k2's trace, its state names against YOLO26n rows 2 and 6, and its factories.

This file needs no device. `NN::C3k2<Inner>` in `modules/vision/blocks/include/SushiAI/vision/blocks/c3k2.hpp` holds a
stem `cv1`, a `ModuleList<Inner>` named `m` and a closing `cv2`. The cases show its trace: the
stem's result is cut into two halves by two K_SLICE_CHANNELS, the inner blocks are chained off
the second half, and one K_CONCAT_CHANNELS joins both halves and every inner block's result
before `cv2`. `Inner` is `Bottleneck` or `C3k`. `NN::make_c3k2_node` reads the row's `inner`
key and selects a factory from `NN::K_C3K2_INNER`.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 2 'c3k2'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### row2_records

`row2_records` lists the 20 records `MANIFEST.json` holds under "model.2": five for `cv1`
(weight [32, 1, 1, 32]), five for `cv2` (weight [64, 1, 1, 48]), then five each for `m.0.cv1`
(weight [8, 3, 3, 16]) and `m.0.cv2` (weight [16, 3, 3, 8]).

### row6_records

`row6_records` lists the 45 records `MANIFEST.json` holds under "model.6": five for `cv1`
(weight [128, 1, 1, 128]), five for `cv2` (weight [128, 1, 1, 192]), then the 35 records of
the inner C3k under "model.6.m.0" that `test_block_c3k.cpp` lists.

### make_row2

`make_row2` builds `C3k2<Bottleneck>(32, 64, 16)` over one `Bottleneck(16, 16, 8, true)`.

### make_row6

`make_row6` builds `C3k2<C3k>(128, 128, 64)` over one `C3k(64, 64, 2, true)`.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

Row 2 maps [160, 160, 32] to [160, 160, 64].

### Row2EmitsFourConvBlocksTwoSlicesAndOneConcat

A training trace of row 2 over [2, 4, 4, 32] emits four each of K_MATMUL, K_BATCHNORM2D and K_SILU,
two K_SLICE_CHANNELS, one K_ADD and one K_CONCAT_CHANNELS. The list holds eight updates, the result
is [2, 4, 4, 64], and every norm carries `K_YOLO_NORM`.

### TheJoinReadsBothHalvesThenTheChainOffTheSecondHalf

The K_CONCAT_CHANNELS of row 2 has three inputs. The first two are produced by K_SLICE_CHANNELS
nodes that read the same value, with the offsets 0 and 16 in attribute 0, and each is
[2, 4, 4, 16]. The third is produced by the bottleneck's K_ADD, whose first operand is the second
half. The joined value is [2, 4, 4, 48].

### TwoInnerBlocksAreChainedAndBothJoined

With two bottlenecks the join has four inputs. The producer of input 2 reads input 1, and the
producer of input 3 reads input 2. The `cv2` weight is [64, 1, 1, 64], four parts of 16
channels, and the second bottleneck's `cv1` weight is [8, 3, 3, 16].

### Row6EmitsNineConvBlocksAndTwoConcats

A training trace of row 6 over [2, 4, 4, 128] emits nine each of K_MATMUL, K_BATCHNORM2D and K_SILU,
two K_SLICE_CHANNELS, two K_ADD and two K_CONCAT_CHANNELS. The list holds 18 updates, the result is
[2, 4, 4, 128], every norm carries `K_YOLO_NORM`, and `output_shape([40, 40, 128])` is
[40, 40, 128].

### StateMatchesManifestRow2

The walk of traced row 2 under "model.2" equals `row2_records()`.

### StateMatchesManifestRow6

The walk of traced row 6 under "model.6" equals `row6_records()`.

### TheInnerTableListsEachInnerBlockWithItsFactory

`K_C3K2_INNER` has three entries: "bottleneck" with `&make_c3k2_of_bottleneck`, "c3k" with
`&make_c3k2_of_c3k` and "bottleneck_psa" with `&make_c3k2_of_bottleneck_psa`.

### TheFactoryBuildsRow22

`make_c3k2_node` over the input [20, 20, 384], with `filters` 256, `hidden` 128, `repeats` 1,
`inner` "bottleneck_psa" and `shortcut` true, gives one port of shape [20, 20, 256]. Its
training trace holds 9 BATCHNORM2D, 5 SILU, 4 ADD, 8 SLICE_CHANNELS and 3 CONCAT_CHANNELS
nodes and 18 buffer updates, every norm carries the row's epsilon and momentum, and the state
walk under "model.22" equals `row22_records()`.

### TheFactoryRefusesAPSAHeadCountThatDoesNotDivideHidden

With `hidden` 200 and `inner` "bottleneck_psa", `make_c3k2_node` fails with
`InvalidDescription`, "'hidden'" and "must split into 3 parts".

### TheFactoryBuildsRow2

`make_c3k2_node` over [160, 160, 32] with 64 filters, hidden 16, one repeat, inner
"bottleneck", shortcut true and the two norm keys has one port of shape [160, 160, 64]. Its
training trace holds one K_ADD, four K_BATCHNORM2D and eight updates carrying `K_YOLO_NORM`, and
its walk under "model.2" equals `row2_records()`.

### TheFactoryBuildsRow6

`make_c3k2_node` over [40, 40, 128] with 128 filters, hidden 64, one repeat, inner "c3k",
shortcut true and the two norm keys has a port of shape [40, 40, 128]. Its training trace
holds two K_CONCAT_CHANNELS, two K_ADD, nine K_BATCHNORM2D and 18 updates carrying `K_YOLO_NORM`,
and its walk under "model.6" equals `row6_records()`. The inner C3k therefore has two
bottlenecks that add their input.

### TheFactoryReadsTheShortcutOfAC3kInnerBlock

With inner "c3k" and shortcut false the evaluation trace holds nine K_MATMUL and no K_ADD.

### TheFactoryDefaultsToOneBottleneckWithAShortcut

A row stating only 64 filters and hidden 16 traces four K_MATMUL, one K_ADD and one
K_CONCAT_CHANNELS.

### TheFactoryReadsRepeatsAndShortcut

With `repeats` 3 and `shortcut` false the trace holds eight K_MATMUL and no K_ADD. The walk yields
40 records: record 5 is "cv2.conv.weight" [64, 1, 1, 80] and record 30 is "m.2.cv1.conv.weight"
[8, 3, 3, 16].

### TheFactoryRefusesAnInnerBlockItDoesNotList

`"inner": "bottleneck_psa"` fails with `InvalidDescription` and a message containing "layer 2
'c3k2'", "'bottleneck_psa'" and "bottleneck, c3k". `"inner": 2` fails with "'inner'" and
"string".

### TheFactoryRefusesAHiddenWidthOfOne

`hidden` 1 fails with "'hidden'" and "at least 2" for the default inner block and for inner
"c3k".

### TheFactoryRefusesARowThatDoesNotFit

The key `heads` fails with the context and "'heads'". A row without `hidden` fails with
"'hidden'" and "absent". The input shape [512] fails with "[512]" and "rank-3".

### AnInnerFactoryRefusesARowThatNamesAnotherInnerBlock

`make_c3k2_of_bottleneck` given `"inner": "c3k"` fails with a message containing the context,
"'inner' is 'c3k'" and "builds 'bottleneck'". `make_c3k2_of_c3k` given `"inner": "bottleneck"`
fails with "'inner' is 'bottleneck'" and "builds 'c3k'". `make_c3k2_of_bottleneck` given
`"inner": 2` fails with "'inner'" and "string". Each factory accepts a row that names its own
block.

### TheBottleneckPSAFactoryRefusesARowThatNamesAnotherInnerBlock

`make_c3k2_of_bottleneck_psa` fails for a row whose `inner` is "c3k" with
`InvalidDescription`, "'inner' is 'c3k'" and "builds 'bottleneck_psa'". It builds a row whose
`inner` is "bottleneck_psa".

## test_conv_norm.cpp

Suites: `Unit_BlockConvNorm`.

Tests ConvNorm's trace, its state names against YOLO26n row 0, and make_conv_node.

This file needs no device. `NN::ConvNorm` in `modules/vision/blocks/include/SushiAI/vision/blocks/conv_norm.hpp` is a
bias-free same-padded `Conv2d`, a `BatchNorm2d` and an activation, traced in that order.
`NN::Conv` is `ConvNorm<Conv2d, SiLU>` and `NN::ConvLinear` is `ConvNorm<Conv2d, Identity>`.
`NN::make_conv_node` builds the node of a `conv` row of a model file.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`, the epsilon and momentum `MANIFEST.json`
records for YOLO26n. `K_CONTEXT` is "layer 0 'conv'", the row name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### OutputShapeHalvesAnEvenExtentAtStrideTwo

`Conv(3, 16, 3, 2)` maps [640, 640, 3] to [320, 320, 16].

### OutputShapeOfAPointwiseBlockKeepsTheExtents

`Conv(32, 64, 1, 1)` maps [7, 9, 32] to [7, 9, 64].

### ATrainingTraceEmitsConvolutionNormThenSiLU

A training trace of `Conv(3, 16, 3, 2)` over [2, 8, 8, 3] emits one K_MATMUL, one K_BATCHNORM2D
and one K_SILU, in that order, and no K_BIAS_ADD. The K_SILU's output is the returned value, of
shape [2, 4, 4, 16].

### ATrainingTraceCarriesTheNormOptions

The K_BATCHNORM2D node carries the epsilon 1e-3 in attribute 0, and both requested updates carry
the momentum 0.03.

### AnEvaluationTraceFoldsTheNorm

An evaluation trace emits one K_BATCHNORM_FOLD, one K_CHANNEL_AFFINE, one K_SILU and no K_BATCHNORM2D,
and returns a [2, 4, 4, 16] value.

### ConvLinearEmitsNoActivation

A training trace of `ConvLinear(3, 16, 1, 1)` emits one K_BATCHNORM2D and no K_SILU. The returned
value is output 0 of the K_BATCHNORM2D node.

### StateMatchesManifestRow0

The walk of a traced `Conv(3, 16, 3, 2)` under "model.0" yields five records in this order:
"model.0.conv.weight" [16, 3, 3, 3], "model.0.bn.weight" [16] and "model.0.bn.bias" [16] as
parameters, then "model.0.bn.running_mean" [16] and "model.0.bn.running_var" [16] as buffers.

### ADepthwiseTraceEmitsDepthwiseConvolutionNormThenSiLU

A training trace of `DWConv(16, 16, 3, 1)` over [2, 8, 8, 16] holds one DEPTHWISE_CONV2D, one
BATCHNORM2D and one SILU in that order and no MATMUL. The convolution reads the input, the
SILU writes the result, and the result has the input's shape.

### DWConvLinearEmitsNoActivation

A training trace of `DWConvLinear(16, 16, 3, 1)` holds one DEPTHWISE_CONV2D, one BATCHNORM2D
and no SILU, and the BATCHNORM2D writes the result.

### ADepthwiseOutputShapeKeepsTheWidth

For the input [80, 80, 64], a 3x3 `DWConv` reports [80, 80, 64] at stride 1 and [40, 40, 64]
at stride 2.

### DepthwiseStateMatchesTheManifestPositionalConvolution

After a training trace of `DWConvLinear(128, 128, 3, 1)`, the state walk under
"model.10.m.0.attn.pe" gives five records in order: the parameters `conv.weight` [3, 3, 128],
`bn.weight` [128] and `bn.bias` [128], then the buffers `bn.running_mean` [128] and
`bn.running_var` [128].

### ADepthwiseBlockRefusesUnequalWidths

Constructing `DWConv(16, 32, 3, 1)` raises an error holding "keeps its channel count",
"in=16" and "out=32".

### AnUntracedBlockYieldsNoState

The walk of a block that was never traced is empty.

### TheFactoryBuildsRow0

`make_conv_node` over [640, 640, 3] with 16 filters, kernel 3, stride 2, `bn_epsilon` 0.001 and
`bn_momentum` 0.03 returns a node with one port named `NN::K_SINGLE_PORT_NAME` of shape
[320, 320, 16]. A training trace over [1, 640, 640, 3] appends one value of shape
[1, 320, 320, 16], the K_BATCHNORM2D carries 1e-3, and the first update carries 0.03. The walk
under "model.0" yields five records, the first "model.0.conv.weight" [16, 3, 3, 3] and the
last "model.0.bn.running_var" [16].

### TheFactoryDefaultsKernelAndStrideToOne

A row stating only 8 filters over [6, 10, 3] has a port of shape [6, 10, 8]. After a trace its
first state record is "conv.weight" [8, 1, 1, 3].

### TheFactoryRefusesAnUnknownKey

The key `padding` fails with `InvalidDescription` and a message containing "layer 0 'conv'",
"'padding'" and "filters".

### TheFactoryRefusesARowWithoutFilters

A row without `filters` fails with a message containing the context, "'filters'" and "absent".

### TheFactoryRefusesAWindowThatIsNotAPositiveWholeNumber

A `stride` of 0 fails with "'stride'", "positive" and "0". A `kernel` given as the string "3"
fails with "'kernel'" and "whole number".

### TheFactoryRefusesNormOptionsOutOfRange

A `bn_epsilon` of 0.0 fails with "'bn_epsilon'" and "positive", and one given as a string
fails with "'bn_epsilon'" and "number". A `bn_momentum` of 1.5 fails with "'bn_momentum'" and
"[0, 1]", and one of -0.1 fails with those and "-0.1".

### TheFactoryRefusesInputsThatAreNotOneFeatureMap

Two input shapes fail with a message containing the context, "one input" and "2". The one
shape [192] fails with "[192]" and "rank-3".

## test_detect_head.cpp

Suites: `Unit_BlockDetectHead`.

Tests `DetectHead` and its factory `Vision::Blocks::make_detect_node` at row 23 of YOLO26n:
three feature levels of 64, 128 and 256 channels with the sides 80, 40 and 20, and 80 classes.
The traced cases feed a batch of 2 with the sides 8, 4 and 2.

### ThePortsCarryTheFixtureNamesAndShapes

The head takes three inputs and has twelve ports. They are named
`<copy>.level<n>.<role>`, with the copy "one2many" before "one2one", the role "boxes" before
"scores", and the levels in order inside a role. A boxes port has 4 channels and a scores port
80, at its level's side.

### StateMatchesManifestRecords390To593

Walked under the prefix "model.23" after a training trace, the head yields 204 records in
order: the box branches under `cv2`, the class branches under `cv3`, then the same two groups
under `one2one_cv2` and `one2one_cv3`.

### ATrainingTraceDetachesEachFeatureOnce

A training trace holds three STOP_GRADIENT nodes, whose inputs are the three features in level
order. `forward` appends twelve values, each of its port's channel count at the traced side of
its level, with the batch extent 2 in front.

### AnEvaluationTraceEmitsNoStopGradient

An evaluation trace holds no STOP_GRADIENT and no BATCHNORM2D node, 36 BATCHNORM_FOLD nodes and
12 DEPTHWISE_CONV2D nodes, and appends twelve values.

### TheFirstLevelSetsTheBranchWidths

A head over one [4, 4, 128] level and 3 classes has 68 state records after a trace. Among them
are `cv2.0.0.conv.weight` of [32, 3, 3, 128], `cv2.0.2.weight` of [4, 1, 1, 32],
`cv3.0.0.1.conv.weight` of [128, 1, 1, 128] and `cv3.0.2.weight` of [3, 1, 1, 128]. Port 1 has
the shape [4, 4, 3].

### ForwardRefusesAWrongInputCount

`forward` given two values for three levels fails with `InvalidDescription`, "input count is 3"
and "given is 2", and appends nothing.

### CreateRefusesLevelsOrClassesThatDoNotFit

`create` fails with `InvalidDescription` for no level ("at least one feature level"), for a
rank-2 second level ("level 1", "[40, 128]", "rank-3") and for a class count of zero
("at least one class", "0").

### TheFactoryBuildsRow23

`make_detect_node` over the three levels, with "classes" 80 and the norm constants, returns a
node with three inputs and twelve ports. Port 0 is "one2many.level0.boxes" of [80, 80, 4] and
port 11 is "one2one.level2.scores" of [20, 20, 80]. A training trace holds 36 BATCHNORM2D nodes
that carry the row's epsilon, every update carries the row's momentum, and the state walk
yields 204 records.

### TheFactoryRefusesARowThatDoesNotFit

An unknown key "filters" fails with a message holding "layer 23 'detect'" and "'filters'". A
row without "classes" fails with "'classes'" and "absent". An empty level list fails with
"layer 23 'detect'" and "at least one feature level".

## test_psa_block.cpp

Suites: `Unit_BlockPsaBlock`.

Tests `PSABlock` at the width of row 10 of YOLO26n, 128 channels over two heads, traced in
training mode over a [2, 4, 4, 128] input.

### OutputShapeIsTheInputShape

`output_shape([20, 20, 128])` is [20, 20, 128].

### EmitsThreeAddsOneSiLUFiveNormsAndTenUpdates

The trace holds three ADD nodes, one SILU node and five BATCHNORM2D nodes, and the buffer
update list holds ten updates. The result has the shape [2, 4, 4, 128].

### TheResidualAddsReadTheInputThenAttentionThenTheFeedForward

The second ADD of the graph reads the block's input and a BATCHNORM2D result. The third ADD
reads the second ADD's result and a BATCHNORM2D result, and its output is the value `forward`
returned.

### StateMatchesManifestRow10Block

Walked under the prefix "model.10.m.0", the traced block yields 25 records in order: five each
for `attn.qkv`, `attn.proj`, `attn.pe`, `ffn.0` and `ffn.1`. `ffn.0` has a [256, 1, 1, 128]
weight and `ffn.1` a [128, 1, 1, 256] weight.

### AWidthTheHeadCountDoesNotDivideThrows

`PSABlock(100, 3)` throws an error naming "head count divides", "100 channels" and "3 heads".

## test_segment_head.cpp

Suites: `Unit_BlockSegmentHead`.

Tests `Vision::Blocks::SegmentHead` and `make_segment_head_node` in
`modules/vision/blocks/include/SushiAI/vision/blocks/segment_head.hpp`. Every head takes 16
input channels and five classes, with an epsilon of 1e-3 and a momentum of 0.03 for its
normalisation.

### OutputShapeScalesTheExtentsAndSetsTheClasses

For the input [6, 10, 16], a head of scale 4 reports [24, 40, 5] and a head of scale 1 reports
[6, 10, 5].

### TheTraceEndsAtTheUpsample

A training trace of a head with 8 hidden channels and scale 2 over [2, 4, 4, 16] holds one
BATCHNORM2D, one SILU and one UPSAMPLE2D. The result has the shape [2, 8, 8, 5] and the
UPSAMPLE2D node produces it.

### StateIsListedUnderCv1AndCv2

After a training trace the state walk gives seven records in this order: `cv1.conv.weight`
[8, 3, 3, 16], `cv1.bn.weight`, `cv1.bn.bias`, `cv1.bn.running_mean` and `cv1.bn.running_var`,
each [8], then `cv2.weight` [5, 1, 1, 8] and `cv2.bias` [5].

### TheFactoryBuildsTheRow

Over the input [4, 4, 16], the row with `classes` 5, `hidden` 8 and `scale` 2 gives a node with
one port of shape [8, 8, 5]. The row with only `classes` 5 gives one port of shape [4, 4, 5];
after an evaluation trace it has seven pieces of state, the first being `cv1.conv.weight`
[16, 3, 3, 16].

### TheFactoryRefusesARowThatDoesNotFit

`make_segment_head_node` fails with `InvalidDescription` for a row with the key `mode`
("layer 3 'segment_head'", "'mode'"), for a row without `classes` ("'classes'", "absent") and
for `scale` 0 ("'scale'", "positive").

## test_sppf.cpp

Suites: `Unit_BlockSppf`.

Tests SPPF's trace, its state names against YOLO26n row 9, and its factory.

This file needs no device. `NN::SPPF` in `modules/vision/blocks/include/SushiAI/vision/blocks/sppf.hpp` holds a
`ConvLinear` named `cv1`, a `MaxPool2d` named `pool` and a `Conv` named `cv2`. The cases show
its trace: `cv1` narrows the input, the pool is applied a number of times in a chain, one
K_CONCAT_CHANNELS joins `cv1`'s result and every pooled value, `cv2` follows, and an optional K_ADD
closes the block. `NN::make_sppf_node` builds the node of an `sppf` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 9 'sppf'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`SPPF(256, 256, 5, 3, true)` maps [20, 20, 256] to [20, 20, 256], and `SPPF(64, 96, 5, 3,
true)` maps [5, 7, 64] to [5, 7, 96].

### ThePoolKeepsTheExtents

The pool of `SPPF(256, 256, 5, 3, true)` has a 5x5 kernel, stride 1 and a pad of 2 on top and
left, and its `output_shape([20, 20, 128])` is [20, 20, 128].

### Row9EmitsThreePoolsOneConcatAndOneAdd

A training trace over [2, 4, 4, 256] emits two K_MATMUL, two K_BATCHNORM2D, one K_SILU, three
K_MAXPOOL2D, one K_CONCAT_CHANNELS and one K_ADD. The list holds four updates and the result is
[2, 4, 4, 256]. The one K_SILU is `cv2`'s.

### TheJoinReadsTheNarrowedValueThenEachPoolInAChain

The K_CONCAT_CHANNELS has four inputs. The first is produced by a K_BATCHNORM2D, because `cv1` is
a `ConvLinear` and emits no activation. Each of the other three is produced by a K_MAXPOOL2D
that reads the input before it, and each is [2, 4, 4, 128]. The joined value is
[2, 4, 4, 512].

### TheClosingAddReadsTheResultThenTheInput

`adds_input` is true. The K_ADD's first operand is produced by a K_SILU and its second is the
block's input, and its output is the returned value. The operand order is the reverse of
`Bottleneck`'s.

### NoAddWithoutAShortcutOrOverUnequalWidths

`SPPF(64, 64, 5, 3, false)` and `SPPF(64, 96, 5, 3, true)` both report `adds_input` false and
trace no K_ADD.

### ThePoolCountSetsThePoolsAndTheJoinedWidth

`SPPF(64, 64, 3, 1, false)` traces one K_MAXPOOL2D. Its `cv1` weight is [32, 1, 1, 64] and its
`cv2` weight is [64, 1, 1, 64], two parts of 32 channels.

### StateMatchesManifestRow9

The walk of a traced `SPPF(256, 256, 5, 3)` under "model.9" yields ten records. The five of
`cv1` come first, with the weight [128, 1, 1, 256] and the rest [128]. The five of `cv2`
follow, with the weight [256, 1, 1, 512] and the rest [256]. The pool has no state.

### TheFactoryBuildsRow9

`make_sppf_node` over [20, 20, 256] with 256 filters, kernel 5, 3 pools and shortcut true has
one port of shape [20, 20, 256]. Its training trace holds three K_MAXPOOL2D and one K_ADD. The walk
under "model.9" yields ten records; record 0 is the `cv1` weight and record 5 the `cv2`
weight.

### TheFactoryDefaultsToThreePoolsOfFiveAndNoShortcut

A row stating only 64 filters traces three K_MAXPOOL2D and no K_ADD, and its `cv2` weight is
[64, 1, 1, 128].

### TheFactoryRefusesAnEvenKernel

`kernel` 4 fails with `InvalidDescription` and a message containing "layer 9 'sppf'",
"'kernel'", "odd" and "4".

### TheFactoryRefusesAnInputOfOneChannel

The input shape [4, 4, 1] fails with "[4, 4, 1]" and "at least 2 channels".

### TheFactoryRefusesARowThatDoesNotFit

The key `hidden` fails with the context and "'hidden'". A row without `filters` fails with
"'filters'" and "absent". `pools` 0 fails with "'pools'" and "positive".

### TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.

### TheFactoryReadsKernelAndPools

With `kernel` 3 and `pools` 1 the trace holds one K_MAXPOOL2D whose attributes 0, 1 and 6, which
are kernel_h, kernel_w and pad_top, are 3, 3 and 1. The `cv2` weight is [64, 1, 1, 64].
