# tests/vision/blocks/test_c2f.cpp

Suites: `Small_BlockC2f`.

Tests the trace of C2f, its state names and shapes, and its factory.

This file needs no device. `Vision::Blocks::C2f` in `modules/vision/blocks/include/SushiAI/vision/blocks/c2f.hpp` is
`C3k2<Bottleneck>`: `cv1`, `cv2` and a `ModuleList` of `Bottleneck`s named `m`, each of the full
hidden width. `make_c2f_node` builds the node of a `c2f` row. The cases use the sizes `in` 8,
`filters` 12 and `hidden` 4. No case runs a graph; the join cases trace a [1, 2, 2, 8] input.


## Constants

`K_TEST_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 4 'c2f'", the row name handed to the
factory.

## Helpers

`record`, `walk`, `count_nodes`, `first_node`, `one_input`, `attributes_of` and
`expect_norm_options` are the helpers of `test_c3k.md`. `conv_records` writes the five records of
one `Conv` under a prefix, `append` joins two lists, `count_norms` counts the running means a walk
lists, `slice_offsets` lists the offset each K_SLICE_CHANNELS node stores, `input_shapes` lists the
shape of each input of a node, and `make_chain` builds the bottlenecks of
width 4 that a case hands to `C2f`.

## OutputShapeKeepsTheExtentsAndSetsTheWidth

`output_shape([5, 7, 8])` is [5, 7, 12].

## ATrainingTraceUpdatesEveryNormAndKeepsTheOptions

A training trace of `C2f(8, 12, 4)` over two bottlenecks with a shortcut, over [2, 4, 4, 8], yields [2, 4, 4, 12]. The list holds two updates, a mean and a variance, for each norm the state walk lists, and `expect_norm_options` finds the constants.

## TheInputIsCutIntoTwoHalvesAtChannelsZeroAndFour

Over a [1, 2, 2, 8] input the K_SLICE_CHANNELS nodes store the offsets 0 and 4, compared exactly because a slice stores its offset as a whole number.

## TheJoinTakesTheTwoHalvesAndEachBottleneckResult

Over a [1, 2, 2, 8] input the one K_CONCAT_CHANNELS node has four inputs of [1, 2, 2, 4] each and yields [1, 2, 2, 16].

## NoShortcutTracesNoAdd

With the shortcut argument false the trace holds no K_ADD.

## StateNamesAndShapesFollowTheSplitLayout

The walk yields 30 records in the order `cv1` [8, 1, 1, 8], `cv2` [12, 1, 1, 16], `m.0.cv1`,
`m.0.cv2`, `m.1.cv1` and `m.1.cv2`, the last four [4, 3, 3, 4]. Each group is the weight, the
norm's weight and bias, then its running mean and variance. `cv2` reads (2 + 2) * 4 channels.

## TheFactoryDefaultsToOneBottleneckWithAShortcut

`make_c2f_node` over [4, 4, 8] with `filters` 12 and `hidden` 4 has one port of [4, 4, 12]. Its evaluation trace yields [2, 4, 4, 12] and holds one K_ADD.

## TheFactoryReadsRepeatsAndShortcut

With `repeats` 2 and `shortcut` false the trace holds no K_ADD. The walk yields 30 records, and record 25 is "c.m.1.cv2.conv.weight" [4, 3, 3, 4].

## TheFactoryRefusesAnUnknownKey

The key `inner` fails with "layer 4 'c2f'" and "'inner'".

## TheFactoryRefusesAnAbsentHiddenWidth

A row without `hidden` fails with "'hidden'" and "absent".

## TheFactoryRefusesAnAbsentFilterCount

A row without `filters` fails with "'filters'" and "absent".

## TheFactoryRefusesZeroRepeats

`repeats` 0 fails with "'repeats'" and "positive".

## TheFactoryRefusesTwoInputs

Two input shapes fail with a message containing "one input" and "2".

## TheFactoryHandsTheNormOptionsToEveryNorm

A row with `repeats` 2, `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces six K_BATCHNORM2D nodes
and 12 updates, and `expect_norm_options` finds those two values on all of them.
