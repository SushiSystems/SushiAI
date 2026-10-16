# tests/vision/detect/free/test_level_rows.cpp


Suites: `Small_DetectLevelRows`.

Tests that `join_level` writes only its own rows and that `split_level` returns what a join
placed.

This file needs no device. The cases of `Small_DetectLevelRows` in
`tests/vision/detect/free/test_anchors.cpp` check where a join places a level, what a split reads
from a counting array, and the two refusals. Their joined array starts at zero, so a join that
also wrote zeros elsewhere would pass them. This file closes that gap and checks the round trip.

## K_EXACT, K_UNTOUCHED and expect_values

`K_EXACT` is the tolerance 0 of a copied value. `K_UNTOUCHED` is -7, a value no join of these
cases writes. `expect_values` compares two arrays value for value.

## AJoinKeepsTheRowsOfOtherLevels

One image of four anchors with one value each starts at `K_UNTOUCHED`. Joining a level that
holds anchors 1 and 2 writes 5 and 6 there and leaves anchors 0 and 3 at `K_UNTOUCHED`.

## ASplitAfterAJoinReturnsTheLevel

A level of the last two of six anchors, three values each, in two images, is joined and then
split out again. The split returns the level value for value.
