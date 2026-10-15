# tests/vision/blocks/test_bottleneck.cpp


Suites: `Small_BlockBottleneck`.

Tests Bottleneck's trace, its state names against YOLO26n row 2, and its factory.

This file needs no device. `NN::Bottleneck` in `modules/vision/blocks/include/SushiAI/vision/blocks/bottleneck.hpp` is
two `Conv` blocks, `cv1` and `cv2`, and an optional K_ADD of the block's input to the second
one's result. `NN::make_bottleneck_node` builds the node of a `bottleneck` row.

The expected state names are typed into the test from
`tests/vision/registration/fixtures/yolo26/MANIFEST.json`. No case opens a fixture file.

## Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 4 'bottleneck'", the row
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

## OutputShapeKeepsTheExtentsAndSetsTheWidth

`Bottleneck(16, 24, 8, true)` maps [5, 7, 16] to [5, 7, 24].

## AShortcutOverEqualWidthsAddsTheInput

A training trace of `Bottleneck(16, 16, 8, true)` reports `adds_input` true and emits two
K_MATMUL, two K_BATCHNORM2D, two K_SILU and one K_ADD. The K_ADD's first operand is the block's input and
its second is produced by a K_SILU. Its output is the returned value, of shape [2, 4, 4, 16].

## NoShortcutTracesNoAdd

With the shortcut argument false, `adds_input` is false and the trace holds two K_MATMUL and no
K_ADD.

## AShortcutOverUnequalWidthsTracesNoAdd

`Bottleneck(16, 24, 8, true)` reports `adds_input` false and traces no K_ADD, although a shortcut
was asked for. The result has the shape [2, 4, 4, 24].

## StateMatchesManifestRow2Inner

The walk of a traced `Bottleneck(16, 16, 8)` under "model.2.m.0" yields ten records. The five
of `cv1` come first: "cv1.conv.weight" [8, 3, 3, 16], then "cv1.bn.weight", "cv1.bn.bias",
"cv1.bn.running_mean" and "cv1.bn.running_var", each [8]. The five of `cv2` follow in the same
order with the weight [16, 3, 3, 8] and the rest [16].

## EachKernelSetsItsOwnConvolution

`Bottleneck(16, 16, 8, false, K_YOLO_NORM, 1, 3)` traces a `cv1` weight of [8, 1, 1, 16] and a
`cv2` weight of [16, 3, 3, 8].

## TheFactoryBuildsTheBlockAndDefaultsTheShortcutToTrue

`make_bottleneck_node` over [4, 4, 16] with 16 filters and a hidden width of 8 has one port of
shape [4, 4, 16]. Its training trace holds one K_ADD, so a row without `shortcut` adds its input.
The walk under "model.2.m.0" yields ten records; record 0 is the `cv1` weight [8, 3, 3, 16]
and record 5 the `cv2` weight [16, 3, 3, 8].

## TheFactoryReadsTheShortcut

With `"shortcut": false` the trace holds no K_ADD.

## TheFactoryRefusesARowThatDoesNotFit

The key `kernel` fails with a message containing "layer 4 'bottleneck'" and "'kernel'". A row
without `hidden` fails with "'hidden'" and "absent". A `shortcut` given as the string "yes"
fails with "'shortcut'" and "true or false".

## TheFactoryRefusesAnInputThatIsNotAFeatureMap

The input shape [256] fails with "[256]" and "rank-3".

## TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.
