# tests/vision/blocks/test_c3k2.cpp


Suites: `Unit_BlockC3k2`.

Tests C3k2's trace, its state names against YOLO26n rows 2 and 6, and its factories.

This file needs no device. `NN::C3k2<Inner>` in `modules/vision/blocks/include/SushiAI/vision/blocks/c3k2.hpp` holds a
stem `cv1`, a `ModuleList<Inner>` named `m` and a closing `cv2`. The cases show its trace: the
stem's result is cut into two halves by two K_SLICE_CHANNELS, the inner blocks are chained off
the second half, and one K_CONCAT_CHANNELS joins both halves and every inner block's result
before `cv2`. `Inner` is `Bottleneck` or `C3k`. `NN::make_c3k2_node` reads the row's `inner`
key and selects a factory from `NN::K_C3K2_INNER`.

The expected state names are typed into the test from
`tests/vision/registration/fixtures/yolo26/MANIFEST.json`. No case opens a fixture file.

## Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 2 'c3k2'", the row
name handed to the factory.

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

## expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

## row2_records

`row2_records` lists the 20 records `MANIFEST.json` holds under "model.2": five for `cv1`
(weight [32, 1, 1, 32]), five for `cv2` (weight [64, 1, 1, 48]), then five each for `m.0.cv1`
(weight [8, 3, 3, 16]) and `m.0.cv2` (weight [16, 3, 3, 8]).

## row6_records

`row6_records` lists the 45 records `MANIFEST.json` holds under "model.6": five for `cv1`
(weight [128, 1, 1, 128]), five for `cv2` (weight [128, 1, 1, 192]), then the 35 records of
the inner C3k under "model.6.m.0" that `test_block_c3k.cpp` lists.

## make_row2

`make_row2` builds `C3k2<Bottleneck>(32, 64, 16)` over one `Bottleneck(16, 16, 8, true)`.

## make_row6

`make_row6` builds `C3k2<C3k>(128, 128, 64)` over one `C3k(64, 64, 2, true)`.

## OutputShapeKeepsTheExtentsAndSetsTheWidth

Row 2 maps [160, 160, 32] to [160, 160, 64].

## Row2EmitsFourConvBlocksTwoSlicesAndOneConcat

A training trace of row 2 over [2, 4, 4, 32] emits four each of K_MATMUL, K_BATCHNORM2D and K_SILU,
two K_SLICE_CHANNELS, one K_ADD and one K_CONCAT_CHANNELS. The list holds eight updates, the result
is [2, 4, 4, 64], and every norm carries `K_YOLO_NORM`.

## TheJoinReadsBothHalvesThenTheChainOffTheSecondHalf

The K_CONCAT_CHANNELS of row 2 has three inputs. The first two are produced by K_SLICE_CHANNELS
nodes that read the same value, with the offsets 0 and 16 in attribute 0, and each is
[2, 4, 4, 16]. The third is produced by the bottleneck's K_ADD, whose first operand is the second
half. The joined value is [2, 4, 4, 48].

## TwoInnerBlocksAreChainedAndBothJoined

With two bottlenecks the join has four inputs. The producer of input 2 reads input 1, and the
producer of input 3 reads input 2. The `cv2` weight is [64, 1, 1, 64], four parts of 16
channels, and the second bottleneck's `cv1` weight is [8, 3, 3, 16].

## Row6EmitsNineConvBlocksAndTwoConcats

A training trace of row 6 over [2, 4, 4, 128] emits nine each of K_MATMUL, K_BATCHNORM2D and K_SILU,
two K_SLICE_CHANNELS, two K_ADD and two K_CONCAT_CHANNELS. The list holds 18 updates, the result is
[2, 4, 4, 128], every norm carries `K_YOLO_NORM`, and `output_shape([40, 40, 128])` is
[40, 40, 128].

## StateMatchesManifestRow2

The walk of traced row 2 under "model.2" equals `row2_records()`.

## StateMatchesManifestRow6

The walk of traced row 6 under "model.6" equals `row6_records()`.

## TheInnerTableListsEachInnerBlockWithItsFactory

`K_C3K2_INNER` has three entries: "bottleneck" with `&make_c3k2_of_bottleneck`, "c3k" with
`&make_c3k2_of_c3k` and "bottleneck_psa" with `&make_c3k2_of_bottleneck_psa`.

## TheFactoryBuildsRow22

`make_c3k2_node` over the input [20, 20, 384], with `filters` 256, `hidden` 128, `repeats` 1,
`inner` "bottleneck_psa" and `shortcut` true, gives one port of shape [20, 20, 256]. Its
training trace holds 9 BATCHNORM2D, 5 SILU, 4 ADD, 8 SLICE_CHANNELS and 3 CONCAT_CHANNELS
nodes and 18 buffer updates, every norm carries the row's epsilon and momentum, and the state
walk under "model.22" equals `row22_records()`.

## TheFactoryRefusesAPSAHeadCountThatDoesNotDivideHidden

With `hidden` 200 and `inner` "bottleneck_psa", `make_c3k2_node` fails with
`InvalidDescription`, "'hidden'" and "must split into 3 parts".

## TheFactoryBuildsRow2

`make_c3k2_node` over [160, 160, 32] with 64 filters, hidden 16, one repeat, inner
"bottleneck", shortcut true and the two norm keys has one port of shape [160, 160, 64]. Its
training trace holds one K_ADD, four K_BATCHNORM2D and eight updates carrying `K_YOLO_NORM`, and
its walk under "model.2" equals `row2_records()`.

## TheFactoryBuildsRow6

`make_c3k2_node` over [40, 40, 128] with 128 filters, hidden 64, one repeat, inner "c3k",
shortcut true and the two norm keys has a port of shape [40, 40, 128]. Its training trace
holds two K_CONCAT_CHANNELS, two K_ADD, nine K_BATCHNORM2D and 18 updates carrying `K_YOLO_NORM`,
and its walk under "model.6" equals `row6_records()`. The inner C3k therefore has two
bottlenecks that add their input.

## TheFactoryReadsTheShortcutOfAC3kInnerBlock

With inner "c3k" and shortcut false the evaluation trace holds nine K_MATMUL and no K_ADD.

## TheFactoryDefaultsToOneBottleneckWithAShortcut

A row stating only 64 filters and hidden 16 traces four K_MATMUL, one K_ADD and one
K_CONCAT_CHANNELS.

## TheFactoryReadsRepeatsAndShortcut

With `repeats` 3 and `shortcut` false the trace holds eight K_MATMUL and no K_ADD. The walk yields
40 records: record 5 is "cv2.conv.weight" [64, 1, 1, 80] and record 30 is "m.2.cv1.conv.weight"
[8, 3, 3, 16].

## TheFactoryRefusesAnInnerBlockItDoesNotList

`"inner": "bottleneck_psa"` fails with `InvalidDescription` and a message containing "layer 2
'c3k2'", "'bottleneck_psa'" and "bottleneck, c3k". `"inner": 2` fails with "'inner'" and
"string".

## TheFactoryRefusesAHiddenWidthOfOne

`hidden` 1 fails with "'hidden'" and "at least 2" for the default inner block and for inner
"c3k".

## TheFactoryRefusesARowThatDoesNotFit

The key `heads` fails with the context and "'heads'". A row without `hidden` fails with
"'hidden'" and "absent". The input shape [512] fails with "[512]" and "rank-3".

## AnInnerFactoryRefusesARowThatNamesAnotherInnerBlock

`make_c3k2_of_bottleneck` given `"inner": "c3k"` fails with a message containing the context,
"'inner' is 'c3k'" and "builds 'bottleneck'". `make_c3k2_of_c3k` given `"inner": "bottleneck"`
fails with "'inner' is 'bottleneck'" and "builds 'c3k'". `make_c3k2_of_bottleneck` given
`"inner": 2` fails with "'inner'" and "string". Each factory accepts a row that names its own
block.

## TheBottleneckPSAFactoryRefusesARowThatNamesAnotherInnerBlock

`make_c3k2_of_bottleneck_psa` fails for a row whose `inner` is "c3k" with
`InvalidDescription`, "'inner' is 'c3k'" and "builds 'bottleneck_psa'". It builds a row whose
`inner` is "bottleneck_psa".
