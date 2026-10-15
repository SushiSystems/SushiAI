# tests/vision/detect/free/test_anchors.cpp


Suites: `Medium_DetectAnchors`, `Small_DetectAnchorLeaves`, `Small_DetectLevelRows`.

Tests the anchor grid, the per-image anchor leaves and the copies between a level and the joined
anchor axis.

This file needs no device. Its three suites, `Medium_DetectAnchors`, `Small_DetectAnchorLeaves` and
`Small_DetectLevelRows`, cover `modules/vision/detect/include/SushiAI/vision/detect/free/anchors.hpp`,
`modules/vision/detect/include/SushiAI/vision/detect/free/anchor_leaves.hpp` and `modules/vision/detect/include/SushiAI/vision/detect/free/level_rows.hpp`.
`modules/vision/detect/README.md` quotes the reference lines each unit ports. One case reads
`loss.sacp` through `modules/layers/reference_files/include/SushiAI/reference_files/reference_file.hpp` and reports itself skipped when the
file is absent.

## K_EXACT

`K_EXACT` is the tolerance 0. Every expected value in the file is a multiple of 0.5, which
float32 holds exactly.

## K_REFERENCE_LEVELS

`K_REFERENCE_LEVELS` holds the three levels
`tests/vision/registration/fixtures/yolo26/MANIFEST.json` records for a 640 px input: 80 by 80 at
stride 8, 40 by 40 at stride 16 and 20 by 20 at stride 32.

## K_SMALL_LEVELS

`K_SMALL_LEVELS` is a grid of 2 by 3 cells at stride 8 followed by 1 by 2 cells at stride 16,
eight anchors in all, so a case can write every expected value out.

## CountsOneAnchorPerCell

`count_anchors` returns 8 for `K_SMALL_LEVELS` and 8400 for `K_REFERENCE_LEVELS`.

## OrdersPointsByLevelThenRowThenColumn

`make_anchors` over `K_SMALL_LEVELS` writes the sixteen point values and the eight strides the
case lists: the first level row by row, x before y, then the second level.

## ScalesEachPointByItsStride

`scale_points` multiplies both coordinates of a point by its anchor's stride. The centres 0.5,
1.5 and 2.5 become 4, 12 and 20 on the first level, and 0.5 and 1.5 become 8 and 24 on the
second.

## ScaleRejectsStridesThatDoNotMatchThePoints

Seven strides for eight points throw with a message containing "[A] strides" and "got 16, 7 and
16 values".

## RejectsALevelWithoutCellsOrStride

`count_anchors` throws on a level of height 0 with a message containing "height 0" and
"width 3", and on a level of stride 0 with a message containing "stride 0".

## RejectsOutputsOfTheWrongSize

`make_anchors` given a point array of 15 values throws with a message containing "16 point
values" and "got 15".

## MatchesTheReferenceAnchorPoints

The points of `K_REFERENCE_LEVELS`, scaled to pixels, equal the tensor `anchors.points` of
`loss.sacp` at tolerance 0, and that tensor has shape [8400, 2]. The case is skipped when
`loss.sacp` is absent.

## GivesEachLevelItsStrideOverItsAnchorRange

The strides of `K_REFERENCE_LEVELS` are 8 over anchors 0 to 6399, 16 over 6400 to 7999 and 32
over 8000 to 8399. The offsets 0, 6400 and 8000 are the `anchor_offset` entries of the manifest.

## RepeatsThePointsAndTheStridesForEveryImage

`build_anchor_leaves` for a batch of two writes the eight points of `K_SMALL_LEVELS` twice. It
writes the strides with four values per anchor: 24 values of 8 and 8 values of 16, also twice.

## SplitsIntoALevelsPointsAndStrides

`split_level` cuts the second level, anchors 6 and 7, out of both leaves. The points come out as
[2, 1, 2, 2] and the strides as sixteen values of 16.

## RejectsABatchWithoutImagesAndAModelWithoutLevels

A batch of 0 throws with a message containing "positive batch" and "batch 0 and 2 levels". An
empty level list throws with a message containing "at least one level" and "batch 1 and 0
levels".

## JoinPlacesALevelAtItsAnchorOffset

`join_level` with two images, five anchors of two values and a level at anchors 3 and 4 writes
the level's rows into values 6 to 9 of each image and leaves the other values at 0.

## SplitReturnsTheRowsJoinPlaced

The joined array holds its own index in every value. `split_level` for anchors 1 to 3 returns
values 2 to 7 of image 0 and 12 to 17 of image 1.

## RejectsALevelPastTheLastAnchor

A level of two anchors at offset 4 of five throws with a message containing "2 anchors",
"offset 4" and "5 anchors".

## SplitRejectsALevelArrayOfTheWrongSize

`split_level` into an array of 10 values, where the placement needs 12, throws with a message
containing "12 level values", "20 joined values" and "got 10 and 20".
