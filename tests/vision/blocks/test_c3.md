# tests/vision/blocks/test_c3.cpp

Suites: `Small_BlockC3`.

Tests the trace of C3, its state names and shapes, and its factory.

This file needs no device. `Vision::Blocks::C3` in `modules/vision/blocks/include/SushiAI/vision/blocks/c3.hpp` is
`C3k`, built with a stated hidden width and the windows 1 and 3: `cv1`, `cv2`, `cv3` and a
`ModuleList` of `Bottleneck`s named `m`. `make_c3_node` builds the node of a `c3` row. The cases
use the sizes `in` 8, `filters` 12 and `hidden` 4, so the hidden width is not half of the output
width. No case runs a graph; the value case reads the wiring of a traced [1, 2, 2, 8] input.

## Constants

`K_TEST_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 4 'c3'", the row name handed to
the factory.

## Helpers

`record`, `walk`, `count_nodes`, `first_node`, `one_input`, `attributes_of` and
`expect_norm_options` are the helpers of `test_c3k.md`. `conv_records` writes the five records of
one `Conv` under a prefix and `append` joins two lists.

## OutputShapeKeepsTheExtentsAndSetsTheWidth

`output_shape([5, 7, 8])` is [5, 7, 12].

## ATrainingTraceEmitsSevenConvBlocksTwoAddsAndOneConcat

With two bottlenecks and a shortcut, a training trace over [2, 4, 4, 8] emits seven each of
K_MATMUL, K_BATCHNORM2D and K_SILU, two K_ADD, one K_CONCAT_CHANNELS and no K_SLICE_CHANNELS. The
list holds 14 updates, the result is [2, 4, 4, 12] and `expect_norm_options` finds the constants.

## TheJoinReadsTheChainThenTheSecondBranch

Over a [1, 2, 2, 8] input the K_CONCAT_CHANNELS node has two inputs. The first is produced by a
K_ADD, the end of the last bottleneck, whose input is the result of the first bottleneck's K_ADD.
The second is produced by a K_SILU, the end of `cv2`. Each is [1, 2, 2, 4] and the join is
[1, 2, 2, 8].

## NoShortcutTracesNoAdd

With the shortcut argument false the trace holds seven K_MATMUL and no K_ADD.

## StateNamesAndShapesFollowTheTwoBranchLayout

The walk yields 35 records in the order `cv1` [4, 1, 1, 8], `cv2` [4, 1, 1, 8], `cv3`
[12, 1, 1, 8], then `m.0.cv1`, `m.0.cv2`, `m.1.cv1` and `m.1.cv2`. Each bottleneck's `cv1` is
[4, 1, 1, 4] and its `cv2` is [4, 3, 3, 4].

## TheFactoryDefaultsToOneBottleneckWithAShortcut

`make_c3_node` over [4, 4, 8] with `filters` 12 and `hidden` 4 has one port of [4, 4, 12]. Its
evaluation trace holds five K_MATMUL and one K_ADD.

## TheFactoryReadsRepeatsAndShortcut

With `repeats` 2 and `shortcut` false the trace holds seven K_MATMUL and no K_ADD. The walk
yields 35 records; record 25 is "c.m.1.cv1.conv.weight" [4, 1, 1, 4] and record 30 is
"c.m.1.cv2.conv.weight" [4, 3, 3, 4].

## TheFactoryRefusesARowThatDoesNotFit

The key `inner` fails with "layer 4 'c3'" and "'inner'". Rows without `hidden` or without
`filters` fail with the key and "absent". `repeats` 0 fails with "'repeats'" and "positive".

## TheFactoryRefusesTwoInputs

Two input shapes fail with a message containing "one input" and "2".

## TheFactoryHandsTheNormOptionsToEveryNorm

A row with `repeats` 2, `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces seven K_BATCHNORM2D
nodes and 14 updates, and `expect_norm_options` finds those two values on all of them.
