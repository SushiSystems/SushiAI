# tests/vision/blocks/test_c2psa.cpp


Suites: `Small_BlockC2psa`, `Small_BlockC2psaDefaultHeads` and `Small_BlockC2psaHeads`.

Tests the `C2PSA` block and its factory `Vision::Blocks::make_c2psa_node` at row 10 of YOLO26n: 256
channels in and out and a hidden width of 128. The traced cases use a [2, 4, 4, 256] input and the
norm options `tests/vision/registration/fixtures/yolo26/MANIFEST.json` records, epsilon 1e-3 and
momentum 0.03.


## OutputShapeKeepsTheExtentsAndSetsTheWidth

`output_shape([20, 20, 256])` is [20, 20, 256].

## Row10KeepsTheShapeUpdatesEveryNormAndKeepsTheOptions

A training trace of row 10 over [2, 4, 4, 256] yields [2, 4, 4, 256]. The list holds two updates, a mean and a variance, for each norm the state walk lists, and `expect_norm_options` finds the row's epsilon and momentum on all of them.

## TheStemIsCutIntoTwoHalvesAtChannelsZeroAndOneHundredTwentyEight

The K_SLICE_CHANNELS nodes that yield 128 channels store the offsets 0 and 128, compared exactly because a slice stores its offset as a whole number. The attention's slices yield 32 or 64 channels, so they are told apart by their result width, not by their place in the trace.

## TheJoinTakesTwoHalvesOf128ChannelsAndYieldsTheOutputWidth

Exactly one K_CONCAT_CHANNELS node yields [2, 4, 4, 256], and its two inputs are each [2, 4, 4, 128].

## TwoRepeatsHoldTwoBlocksAndKeepTheInputShape

`C2PSA(256, 256, 128, 2, 2)` holds two blocks in `m`, and its evaluation trace yields [2, 4, 4, 256].

## StateMatchesManifestRow10

Walked under the prefix "model.10" after a training trace, the block yields the 35 records the
file lists by name, shape, kind and order, from `model.10.cv1.conv.weight` on.

## TheFactoryBuildsOnePortOfTheInputShape

`make_c2psa_node` over [20, 20, 256] with `filters` 256, `hidden` 128, `repeats` 1 and `heads` 2 has one port of shape [20, 20, 256].

## TheFactoryHandsTheNormOptionsToEveryNorm

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces a block whose every norm and update carries those two values.

## TheFactoryDefaultsToOneBlock

A row stating neither `repeats` nor `heads` builds one block: the walk yields 35 records.

## ARowThatStatesNoHeadCountGetsTheHiddenWidthOver64OrOne

`default_attention_heads` returns max(hidden / 64, 1): 2 for 128, 3 for 192, 1 for 127 and 1 for 64. The suite is `Small_BlockC2psaDefaultHeads`.

## TheFactoryRefusesAFilterWidthThatIsNotTheInput

"filters" 128 over a 256-channel input fails with `InvalidDescription`, and the message holds
"'filters'", "128" and "256 channels".

## AHiddenWidthTheHeadsDoNotSplitIsRefused

A hidden width of 128 with 3 heads, and with 128 heads, fails with "Attention needs" and, for 3 heads, "its head count divides", for 128 heads, "at least 2 channels per head": the refusals of `check_attention_heads`. The suite is `Small_BlockC2psaHeads`, one row each.

## TheFactoryRefusesAnUnknownKey

The key `inner` fails with "layer 10 'c2psa'" and "'inner'".

## TheFactoryRefusesAnAbsentHiddenWidth

A row without `hidden` fails with "'hidden'" and "absent".

## TheFactoryRefusesAFlatInput

The input shape [512] fails with "[512]" and "rank-3".
