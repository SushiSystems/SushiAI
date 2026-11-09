# tests/vision/detect/free/test_level_rows.cpp


Suites: `Small_DetectLevelRows`.

Tests that `join_level` writes only its own rows and that `split_level` returns what a join
placed.

This file needs no device. It is the one unit that tests `join_level` and `split_level`:
`test_anchors.cpp` only reads `split_level` to cut a level out of the anchor leaves. The joined
array of the join cases starts at `K_UNTOUCHED`, so a join that wrote anything outside its level
fails them.

## K_UNTOUCHED and expect_values

`K_UNTOUCHED` is -7, a value no join of these cases writes. `expect_values` compares two arrays
value for value with equality, since a copied value is exact.

## AJoinKeepsTheRowsOfOtherLevels

One image of four anchors with one value each starts at `K_UNTOUCHED`. Joining a level that
holds anchors 1 and 2 writes 5 and 6 there and leaves anchors 0 and 3 at `K_UNTOUCHED`.

## ASplitAfterAJoinReturnsTheLevel

A level of the last two of six anchors, three values each, in two images, is joined and then
split out again. The level is written out as twelve literal values, from 0.5 to 11.5, and the
split returns it value for value.

## ASplitReadsTheLevelAtItsAnchorOffset

Two images of five anchors with two values each hold their own index in every value. `split_level`
for anchors 1 to 3 returns values 2 to 7 of image 0 and 12 to 17 of image 1.

## AJoinRefusesALevelPastTheLastAnchor

A level of two anchors at offset 4 of five returns `InvalidDescription` with a message containing
"2 anchors", "offset 4" and "5 anchors".

## ASplitRefusesALevelArrayOfTheWrongSize

`split_level` into an array of 10 values, where the placement needs 12, returns
`InvalidDescription` with a message containing "12 level values", "20 joined values" and "got 10
and 20".
