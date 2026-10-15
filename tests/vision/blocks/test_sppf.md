# tests/vision/blocks/test_sppf.cpp


Suites: `Unit_BlockSppf`.

Tests SPPF's trace, its state names against YOLO26n row 9, and its factory.

This file needs no device. `NN::SPPF` in `modules/vision/blocks/include/SushiAI/vision/blocks/sppf.hpp` holds a
`ConvLinear` named `cv1`, a `MaxPool2d` named `pool` and a `Conv` named `cv2`. The cases show
its trace: `cv1` narrows the input, the pool is applied a number of times in a chain, one
K_CONCAT_CHANNELS joins `cv1`'s result and every pooled value, `cv2` follows, and an optional K_ADD
closes the block. `NN::make_sppf_node` builds the node of an `sppf` row.

The expected state names are typed into the test from
`tests/vision/registration/fixtures/yolo26/MANIFEST.json`. No case opens a fixture file.

## Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 9 'sppf'", the row
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

`SPPF(256, 256, 5, 3, true)` maps [20, 20, 256] to [20, 20, 256], and `SPPF(64, 96, 5, 3,
true)` maps [5, 7, 64] to [5, 7, 96].

## ThePoolKeepsTheExtents

The pool of `SPPF(256, 256, 5, 3, true)` has a 5x5 kernel, stride 1 and a pad of 2 on top and
left, and its `output_shape([20, 20, 128])` is [20, 20, 128].

## Row9EmitsThreePoolsOneConcatAndOneAdd

A training trace over [2, 4, 4, 256] emits two K_MATMUL, two K_BATCHNORM2D, one K_SILU, three
K_MAXPOOL2D, one K_CONCAT_CHANNELS and one K_ADD. The list holds four updates and the result is
[2, 4, 4, 256]. The one K_SILU is `cv2`'s.

## TheJoinReadsTheNarrowedValueThenEachPoolInAChain

The K_CONCAT_CHANNELS has four inputs. The first is produced by a K_BATCHNORM2D, because `cv1` is
a `ConvLinear` and emits no activation. Each of the other three is produced by a K_MAXPOOL2D
that reads the input before it, and each is [2, 4, 4, 128]. The joined value is
[2, 4, 4, 512].

## TheClosingAddReadsTheResultThenTheInput

`adds_input` is true. The K_ADD's first operand is produced by a K_SILU and its second is the
block's input, and its output is the returned value. The operand order is the reverse of
`Bottleneck`'s.

## NoAddWithoutAShortcutOrOverUnequalWidths

`SPPF(64, 64, 5, 3, false)` and `SPPF(64, 96, 5, 3, true)` both report `adds_input` false and
trace no K_ADD.

## ThePoolCountSetsThePoolsAndTheJoinedWidth

`SPPF(64, 64, 3, 1, false)` traces one K_MAXPOOL2D. Its `cv1` weight is [32, 1, 1, 64] and its
`cv2` weight is [64, 1, 1, 64], two parts of 32 channels.

## StateMatchesManifestRow9

The walk of a traced `SPPF(256, 256, 5, 3)` under "model.9" yields ten records. The five of
`cv1` come first, with the weight [128, 1, 1, 256] and the rest [128]. The five of `cv2`
follow, with the weight [256, 1, 1, 512] and the rest [256]. The pool has no state.

## TheFactoryBuildsRow9

`make_sppf_node` over [20, 20, 256] with 256 filters, kernel 5, 3 pools and shortcut true has
one port of shape [20, 20, 256]. Its training trace holds three K_MAXPOOL2D and one K_ADD. The walk
under "model.9" yields ten records; record 0 is the `cv1` weight and record 5 the `cv2`
weight.

## TheFactoryDefaultsToThreePoolsOfFiveAndNoShortcut

A row stating only 64 filters traces three K_MAXPOOL2D and no K_ADD, and its `cv2` weight is
[64, 1, 1, 128].

## TheFactoryRefusesAnEvenKernel

`kernel` 4 fails with `InvalidDescription` and a message containing "layer 9 'sppf'",
"'kernel'", "odd" and "4".

## TheFactoryRefusesAnInputOfOneChannel

The input shape [4, 4, 1] fails with "[4, 4, 1]" and "at least 2 channels".

## TheFactoryRefusesARowThatDoesNotFit

The key `hidden` fails with the context and "'hidden'". A row without `filters` fails with
"'filters'" and "absent". `pools` 0 fails with "'pools'" and "positive".

## TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.

## TheFactoryReadsKernelAndPools

With `kernel` 3 and `pools` 1 the trace holds one K_MAXPOOL2D whose attributes 0, 1 and 6, which
are kernel_h, kernel_w and pad_top, are 3, 3 and 1. The `cv2` weight is [64, 1, 1, 64].
