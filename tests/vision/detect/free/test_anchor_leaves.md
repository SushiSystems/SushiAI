# tests/vision/detect/free/test_anchor_leaves.cpp


Suites: `Small_DetectAnchorLeaves`.

Tests `build_anchor_leaves` on a one-image batch and its rebuild of used leaves for a smaller
batch.

This file needs no device. The cases of `Small_DetectAnchorLeaves` in
`tests/vision/detect/free/test_anchors.cpp` check two images over two levels, the split into one
level, and the refused batch and level list. This file adds the smallest batch and the reuse of
one `AnchorLeaves` across batches.

## K_EXACT and expect_values

`K_EXACT` is the tolerance 0: every point is a cell centre ending in .5 and every stride a
whole number, which float32 holds exactly. `expect_values` compares two arrays value for value.

## OneImageHoldsTheCellCentresAndTheirStride

One 1 by 2 level at stride 8 and a batch of 1 give the points (0.5, 0.5) and (1.5, 0.5) and eight
strides of 8, four per anchor.

## RebuildingForASmallerBatchShrinksTheLeaves

Leaves built for 3 images of five anchors are built again for 1 image. `points` then holds 10
values and `strides` 20, the sizes of one image. Arrays that kept their earlier length would hand
a graph leaf of the wrong batch.
