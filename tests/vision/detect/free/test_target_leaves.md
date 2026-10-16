# tests/vision/detect/free/test_target_leaves.cpp


Suites: `Small_DetectTargetLeaves`.

Tests the constants of the loss leaves, the extents of every leaf, the rebuild of used leaves
and the geometry `build_target_leaves` refuses.

This file needs no device. The cases of `Small_DetectTargetLeaves` in
`tests/vision/detect/free/test_assigner.cpp` check the values of the leaves: the weights, the
scaled boxes and distances, the safe box of an unassigned row and the floor of the score sum.
This file covers the shape of the leaves and the inputs they refuse.

## K_EXACT

`K_EXACT` is the tolerance 0 of a constant or a copied value.

## make_idle_assignment and make_points

`make_idle_assignment` returns an `Assignment` of the given extents with every array sized and
no anchor assigned. `make_points` returns one row of cell centres, x then y, one per anchor.

## TheSafeBoxIsOneCellAndTheScoreSumStartsAtItsFloor

`K_SAFE_BOX` is 0, 0, 1, 1, one cell in grid units. `K_MIN_TARGET_SCORES_SUM` is 1, and a default
`TargetLeaves` holds it as its score sum.

## EveryLeafHasTheRowsOfTheAssignment

An assignment of 2 images, 3 anchors and 2 classes gives six rows: `weight` and
`normalized_weight` hold 6 values and the five box leaves hold 24.

## RebuildingFromASmallerAssignmentKeepsNoEarlierRow

Leaves built from 2 images with one foreground row are built again from 1 idle image. The box
mask then holds 12 zeros and the score sum is back at its floor of 1.

## RefusesGeometryOfAnotherAnchorCountAndNoImageSize

Strides for 2 anchors beside an assignment of 3 are refused, and the message names the expected
`[3, 2]` points and `[3]` strides. An image height of 0 is refused as no positive image size.
