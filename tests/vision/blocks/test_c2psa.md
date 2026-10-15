# tests/vision/blocks/test_c2psa.cpp


Suites: `Small_BlockC2psa`.

Tests the `C2PSA` block and its factory `Vision::Blocks::make_c2psa_node` at row 10 of YOLO26n: 256
channels in and out and a hidden width of 128. The traced cases use a [2, 4, 4, 256] input and the
norm options `tests/vision/registration/fixtures/yolo26/MANIFEST.json` records, epsilon 1e-3 and
momentum 0.03.

## OutputShapeKeepsTheExtentsAndSetsTheWidth

`output_shape([20, 20, 256])` is [20, 20, 256].

## Row10EmitsEightSlicesThreeAddsAndThreeJoins

A training trace holds 8 SLICE_CHANNELS, 3 ADD, 3 CONCAT_CHANNELS, 3 SILU and 7 BATCHNORM2D
nodes, and the update list holds 14 updates. The result has the shape [2, 4, 4, 256]. Every
BATCHNORM2D node carries the epsilon 1e-3 and every update the momentum 0.03.

## TheJoinKeepsTheFirstHalfAndChainsTheSecondOffTheSecondSlice

The first two slices read the same value, at the offsets 0 and 128. The second ADD of the
graph reads the second slice as its first input. The third CONCAT_CHANNELS reads the first
slice and then the third ADD's result, and yields [2, 4, 4, 256].

## EachRepeatReadsTheResultOfThePreviousOne

With two repeats the block holds two inner blocks and an evaluation trace holds six ADD nodes.
The fifth ADD reads the third ADD's result as its first input.

## StateMatchesManifestRow10

Walked under the prefix "model.10" after a training trace, the block yields the 35 records the
file lists by name, shape, kind and order, from `model.10.cv1.conv.weight` on.

## TheFactoryBuildsRow10

`make_c2psa_node` over a [20, 20, 256] input, with the row's filters, hidden width, repeats,
heads and norm constants, returns a node with one port of the shape [20, 20, 256]. Its training
trace holds 7 BATCHNORM2D nodes with the row's epsilon and 14 updates with the row's momentum,
and its state walk yields the same 35 records.

## TheFactoryDefaultsToOneBlockAndHiddenOver64Heads

A row with only "filters" 256 and "hidden" 128 traces to two LOG_SOFTMAX nodes and three ADD
nodes, and its state walk yields 35 records.

## TheFactoryGivesOneHeadForAHiddenWidthUnder128

A row with "filters" 64 and "hidden" 64 over a 64-channel input traces to one LOG_SOFTMAX node.

## TheFactoryRefusesAFilterWidthThatIsNotTheInput

"filters" 128 over a 256-channel input fails with `InvalidDescription`, and the message holds
"'filters'", "128" and "256 channels".

## TheFactoryRefusesAHiddenWidthTheHeadsDoNotSplit

A hidden width of 128 with 3 heads, and with 128 heads, each fail with `InvalidDescription`,
"'hidden'" and "parts of at least 2 channels".

## TheFactoryRefusesARowThatDoesNotFit

An unknown key "inner" fails with a message holding "layer 10 'c2psa'" and "'inner'". A row
without "hidden" fails with "'hidden'" and "absent". A [512] input fails with "[512]" and
"rank-3".
