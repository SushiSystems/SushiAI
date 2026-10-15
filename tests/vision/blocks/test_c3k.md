# tests/vision/blocks/test_c3k.cpp


Suites: `Unit_BlockC3k`.

Tests C3k's trace, its state names against YOLO26n row 6, and its factory.

This file needs no device. `NN::C3k` in `modules/vision/blocks/include/SushiAI/vision/blocks/c3k.hpp` holds three `Conv`
blocks, `cv1`, `cv2` and `cv3`, and a `ModuleList` of `Bottleneck`s named `m`. The cases show
its trace: a chain of bottlenecks on one branch, `cv2` on the other, one K_CONCAT_CHANNELS of the
two, and `cv3` after it. `NN::make_c3k_node` builds the node of a `c3k` row.

The expected state names are typed into the test from
`tests/vision/registration/fixtures/yolo26/MANIFEST.json`. No case opens a fixture file.

## Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 5 'c3k'", the row name handed
to the factory.

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

## OutputShapeKeepsTheExtentsAndSetsTheWidth

`C3k(64, 96, 2, true)` maps [5, 7, 64] to [5, 7, 96].

## TheChainHoldsOneBottleneckPerRepeat

`C3k(64, 64, 3, true)` has `m.size()` 3.

## ATrainingTraceEmitsSevenConvBlocksTwoAddsAndOneConcat

A training trace of `C3k(64, 64, 2, true)` over [2, 4, 4, 64] emits seven each of K_MATMUL,
K_BATCHNORM2D and K_SILU, two K_ADD, one K_CONCAT_CHANNELS and no K_SLICE_CHANNELS. The list holds 14
updates and the result has the shape [2, 4, 4, 64]. Seven is three `Conv` blocks of the C3k
plus two in each of the two bottlenecks.

## TheJoinReadsTheChainThenTheSecondBranch

The K_CONCAT_CHANNELS node has two inputs. The first is produced by an K_ADD, the end of the last
bottleneck, and the second by a K_SILU, the end of `cv2`. Each is [2, 4, 4, 32] and the joined
value is [2, 4, 4, 64].

## NoShortcutTracesNoAdd

With the shortcut argument false, an evaluation trace holds seven K_MATMUL and no K_ADD.

## TheHiddenWidthIsHalfTheOutputWidthRoundedDown

`C3k(8, 7, 1, true)` has a hidden width of 3. The traced weights are [3, 1, 1, 8] for `cv1`
and `cv2`, [7, 1, 1, 6] for `cv3` and [3, 3, 3, 3] for the bottleneck's `cv1`. The result is
[2, 4, 4, 7], and `output_shape([4, 4, 8])` is [4, 4, 7].

## StateMatchesManifestRow6Inner

The walk of a traced `C3k(64, 64, 2)` under "model.6.m.0" yields 35 records: five each for
`cv1`, `cv2`, `cv3`, `m.0.cv1`, `m.0.cv2`, `m.1.cv1` and `m.1.cv2`, in that order. Each group
is the convolution weight, the norm's weight and bias as parameters, then its running mean and
variance as buffers. The weights are [32, 1, 1, 64] for `cv1` and `cv2`, [64, 1, 1, 64] for
`cv3` and [32, 3, 3, 32] for the four bottleneck convolutions.

## TheFactoryDefaultsToOneBottleneckWithAShortcut

`make_c3k_node` over [4, 4, 64] with only 32 filters has one port of shape [4, 4, 32]. Its
evaluation trace appends one [2, 4, 4, 32] value and holds five K_MATMUL and one K_ADD: one
bottleneck that adds its input.

## TheFactoryReadsRepeatsAndShortcut

With 64 filters, `repeats` 2 and `shortcut` false, the trace holds seven K_MATMUL and no K_ADD. The
walk under "model.6.m.0" yields 35 records, and record 30 is "model.6.m.0.m.1.cv2.conv.weight"
[32, 3, 3, 32].

## TheFactoryRefusesAWidthThatLeavesNoHiddenChannel

`filters` 1 fails with `InvalidDescription` and a message containing "layer 5 'c3k'",
"'filters'" and "at least 2".

## TheFactoryRefusesARowThatDoesNotFit

The key `hidden` fails with the context and "'hidden'". A row without `filters` fails with
"'filters'" and "absent". `repeats` 0 fails with "'repeats'" and "positive".

## TheFactoryRefusesTwoInputs

Two input shapes fail with a message containing "one input" and "2".

## TheFactoryHandsTheNormOptionsToEveryNorm

A row with `repeats` 2, `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces seven K_BATCHNORM2D
nodes and 14 updates, and `expect_norm_options` finds those two values on all of them.
