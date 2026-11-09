# tests/vision/blocks/test_c3.cpp

Suites: `Small_BlockC3`.

Tests the trace of C3, its state names and shapes, and its factory.

This file needs no device. `Vision::Blocks::C3` in `modules/vision/blocks/include/SushiAI/vision/blocks/c3.hpp` is
`C3k`, built with a stated hidden width and the windows 1 and 3: `cv1`, `cv2`, `cv3` and a
`ModuleList` of `Bottleneck`s named `m`. `make_c3_node` builds the node of a `c3` row. The cases
use the sizes `in` 8, `filters` 12 and `hidden` 4, so the hidden width is not half of the output
width. No case runs a graph; the join case traces a [1, 2, 2, 8] input.


## Constants

`K_TEST_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 4 'c3'", the row name handed to
the factory.

## Helpers

`record`, `walk`, `count_nodes`, `first_node`, `one_input`, `attributes_of` and
`expect_norm_options` are the helpers of `test_c3k.md`. `conv_records` writes the five records of
one `Conv` under a prefix, `append` joins two lists, `count_norms` counts the running means a walk
lists and `input_shapes` lists the shape of each input of a node.

## OutputShapeKeepsTheExtentsAndSetsTheWidth

`output_shape([5, 7, 8])` is [5, 7, 12].

## ATrainingTraceUpdatesEveryNormAndKeepsTheOptions

A training trace of `C3` with two bottlenecks and a shortcut, over [2, 4, 4, 8], yields [2, 4, 4, 12]. The list holds two updates, a mean and a variance, for each norm the state walk lists, and `expect_norm_options` finds the constants.

## TheJoinTakesTheChainAndTheSecondBranch

Over a [1, 2, 2, 8] input the one K_CONCAT_CHANNELS node has two inputs of [1, 2, 2, 4] each and yields [1, 2, 2, 8].

## NoShortcutTracesNoAdd

With the shortcut argument false the trace holds no K_ADD.

## StateNamesAndShapesFollowTheTwoBranchLayout

The walk yields 35 records in the order `cv1` [4, 1, 1, 8], `cv2` [4, 1, 1, 8], `cv3`
[12, 1, 1, 8], then `m.0.cv1`, `m.0.cv2`, `m.1.cv1` and `m.1.cv2`. Each bottleneck's `cv1` is
[4, 1, 1, 4] and its `cv2` is [4, 3, 3, 4].

## TheFactoryDefaultsToOneBottleneckWithAShortcut

`make_c3_node` over [4, 4, 8] with `filters` 12 and `hidden` 4 has one port of [4, 4, 12]. Its evaluation trace yields [2, 4, 4, 12] and holds one K_ADD.

## TheFactoryReadsRepeatsAndShortcut

With `repeats` 2 and `shortcut` false the trace holds no K_ADD. The walk yields 35 records; record 25 is "c.m.1.cv1.conv.weight" [4, 1, 1, 4] and record 30 is "c.m.1.cv2.conv.weight" [4, 3, 3, 4].

## TheFactoryRefusesAnUnknownKey

The key `inner` fails with "layer 4 'c3'" and "'inner'".

## TheFactoryRefusesAnAbsentHiddenWidth

A row without `hidden` fails with "'hidden'" and "absent".

## TheFactoryRefusesAnAbsentFilterCount

A row without `filters` fails with "'filters'" and "absent".

## TheFactoryRefusesZeroRepeats

`repeats` 0 fails with "'repeats'" and "positive".

## TheFactoryRefusesTwoInputs

Two input shapes fail with a message containing "one input" and "2".

## TheFactoryHandsTheNormOptionsToEveryNorm

A row with `repeats` 2, `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces seven K_BATCHNORM2D
nodes and 14 updates, and `expect_norm_options` finds those two values on all of them.
