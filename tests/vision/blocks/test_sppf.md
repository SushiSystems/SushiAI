# tests/vision/blocks/test_sppf.cpp


Suites: `Small_BlockSppf` and `Small_BlockSppfNoAdd`.

Tests SPPF's trace, its state names against YOLO26n row 9, and its factory.

This file needs no device. `NN::SPPF` in `modules/vision/blocks/include/SushiAI/vision/blocks/sppf.hpp` holds a
`ConvLinear` named `cv1`, a `MaxPool2d` named `pool` and a `Conv` named `cv2`. The cases show
its trace: `cv1` narrows the input, the pool is applied a number of times in a chain, one
K_CONCAT_CHANNELS joins `cv1`'s result and every pooled value, `cv2` follows, and an optional K_ADD
closes the block. `NN::make_sppf_node` builds the node of an `sppf` row.

The expected state names are typed into the test from
`tests/vision/registration/fixtures/yolo26/MANIFEST.json`. No case opens a fixture file.


## Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 9 'sppf'", the row
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

## count_norms

`count_norms` counts the running means a walk lists, one per norm.

## input_shapes

`input_shapes` lists the shape of each input of a node, in order.

## count_adds_reading

`count_adds_reading` counts the K_ADD nodes of a graph that read one value as either operand.

## conv_records

`conv_records` writes the five records of one `Conv` under a prefix: the weight, then the norm's weight, bias, running mean and running variance.

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

`SPPF(256, 256, 5, 3, true)` maps [20, 20, 256] to [20, 20, 256], and `SPPF(64, 96, 5, 3,
true)` maps [5, 7, 64] to [5, 7, 96].

## TheKernelSizeIsTheOneTheConstructorTook

`SPPF(256, 256, 5, 3, true, ...)` reports a kernel size of 5 and `SPPF(64, 64, 3, 1, false, ...)` one of 3.

## ThePoolKeepsTheExtents

The `output_shape([20, 20, 128])` of the pool of `SPPF(256, 256, 5, 3, true)` is [20, 20, 128].

## Row9KeepsTheShapeAndUpdatesEveryNorm

A training trace over [2, 4, 4, 256] returns [2, 4, 4, 256]. The list holds two updates, a mean and a variance, for each norm the state walk lists.

## TheJoinTakesTheNarrowedValueAndEachPool

The one K_CONCAT_CHANNELS has four inputs of [2, 4, 4, 128] each, the narrowed value and three pooled ones, and yields [2, 4, 4, 512].

## AShortcutOverEqualWidthsAddsTheInputOnce

`adds_input` is true, exactly one K_ADD reads the block's input, and the returned value is produced by a K_ADD.

## NoAddIsTracedWithoutAShortcutOrOverUnequalWidths

`SPPF(64, 64, 5, 3, false)` and `SPPF(64, 96, 5, 3, true)` both report `adds_input` false and trace no K_ADD. The suite is `Small_BlockSppfNoAdd`, one row each.

## ThePoolCountSetsTheJoinedWidthCv2Reads

The state walk under "b" of a traced `SPPF(64, 64, 3, 1, false)` lists ten records: `cv1` with a [32, 1, 1, 64] weight and `cv2` with a [64, 1, 1, 64] weight, which reads two parts of 32 channels.

## StateMatchesManifestRow9

The walk of a traced `SPPF(256, 256, 5, 3)` under "model.9" yields ten records. The five of
`cv1` come first, with the weight [128, 1, 1, 256] and the rest [128]. The five of `cv2`
follow, with the weight [256, 1, 1, 512] and the rest [256]. The pool has no state.

## TheFactoryBuildsRow9

`make_sppf_node` over [20, 20, 256] with 256 filters, kernel 5, 3 pools and shortcut true has one port of shape [20, 20, 256]. The state is pinned by `StateMatchesManifestRow9`.

## TheFactoryDefaultsToThreePoolsOfFiveAndNoShortcut

A row stating only 64 filters traces no K_ADD, and its `cv2` weight is [64, 1, 1, 128]: four parts of 32 channels, which is three pools.

## TheFactoryRefusesAnEvenKernel

`kernel` 4 fails with `InvalidDescription` and a message containing "layer 9 'sppf'",
"'kernel'", "odd" and "4".

## TheFactoryRefusesAnInputOfOneChannel

The input shape [4, 4, 1] fails with "[4, 4, 1]" and "at least 2 channels".

## TheFactoryRefusesAnUnknownKey

The key `hidden` fails with the context and "'hidden'".

## TheFactoryRefusesAnAbsentFilterCount

A row without `filters` fails with "'filters'" and "absent".

## TheFactoryRefusesZeroPools

`pools` 0 fails with "'pools'" and "positive".

## TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.

## TheFactoryReadsThePoolCount

With `kernel` 3 and `pools` 1 the walk yields ten records and the `cv2` weight is [64, 1, 1, 64], two parts. The value of `kernel` shows only in the pool node's attributes and in values, so no case here pins it.
