# tests/layers/nn/test_max_pool2d/test_end_padding.cpp

That the maxpool2d row can pad the end of an axis less or more than its start through the keys `leading_padding` and `end_padding`, that the window count follows the two pads, that a pooled value drops the windows the missing pad removes, and that the keys are refused in a form the layer cannot state.

Suites: `Small_MaxPool2dEndPaddingExtent`, `Small_MaxPool2dEndPaddingRefusal`, `Small_MaxPool2dEndPadding`, `Medium_MaxPool2dEndPaddingGradient`.

Tests `NN::MaxPool2d` with its signed `end_pad_h` and `end_pad_w` fields and the two keys of
`NN::make_maxpool2d`. The values cases pool a 3 by 3 input holding 9, 1, 2 over 3, 8, 4 over 5, 6,
7 on the host in double, and compare exactly.

## Small_MaxPool2dEndPaddingExtent.TheExtentFollowsTheLeadingAndTheEndPadding

Over a [5, 5, 3] input with a window of 3 at stride 1, the window count along an axis is 5 plus the
leading pad plus the end pad minus 2. The rows are the pads 1 and 1 (5), 1 and 0 (4), 0 and 2 (5),
2 and 0 (5), and the pairs [1, 0] and [0, 2] per axis ([4, 5, 3]).

## Small_MaxPool2dEndPaddingRefusal.TheFactoryRefusesTheRow

`InvalidDescription` with the context "layer 2 'maxpool2d'": one of the two keys alone ("needs both
or neither"); the pair beside "padding" or "trailing_padding" ("takes one form of the two"); a
negative pad in either key ("0 or more"); a negative "trailing_padding" ("non-negative trailing
padding").

## Small_MaxPool2dEndPadding.ALeadingPadWithNoEndPadDropsTheWindowsPastTheEnd

A window of two with a pad of one above and left and none below and right gives 9, 9, 2, 9, 9, 8,
5, 8, 8: row i reads the input rows i - 1 and i that exist, and column j the columns j - 1 and j.

## Small_MaxPool2dEndPadding.AnEndPadWithNoLeadingPadKeepsTheLastWindows

A window of two with no pad above and left and one below and right gives 9, 8, 4, 8, 8, 7, 6, 7, 7:
row i reads the input rows i and i + 1 that exist.

## Small_MaxPool2dEndPadding.TheRowKeysSetTheLeadingPadAndTheSignedEndPad

The pair 1 and 0, which sets an end pad of -1, gives a [3, 3, 1] result from a [3, 3, 1] input at a window of 2 and stride 1, as
does the pair 0 and 1.

## Small_MaxPool2dEndPadding.AnEndPadBelowMinusTheLeadingPadIsRefused

An end pad of -2 under a leading pad of 1 fails with `InvalidDescription`, "MaxPool2d" and "end
padding of 0 or more".

## Medium_MaxPool2dEndPaddingGradient.TheGradientMatchesCentralDifferences

The gradient of a loss over the pooled 3 by 3 input with a leading pad of 1 and an end pad of -1,
at the input above, whose nine elements differ, matches central differences in FLOAT64.
