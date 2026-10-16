# tests/vision/detect/test_ground_truth.cpp


Suites: `Small_DetectGroundTruth`.

Tests the empty `GroundTruthBatch`, the refill of a used batch by `pad_ground_truth`, and the
row arrays it refuses.

This file needs no device. The suite `Medium_DetectGroundTruth` in
`tests/vision/detect/free/test_assigner.cpp` checks the padding itself: rows per image, the
conversion to pixel corners, a row without extent, a batch without rows, a row outside the batch
and the reference labels. This file covers what those cases leave open.

## K_EXACT

`K_EXACT` is the tolerance 0 of a corner that is a multiple of 8, which float32 holds exactly.

## ADefaultBatchIsEmpty

A default `GroundTruthBatch` has no images, no rows per image and three empty arrays.

## PaddingIntoAUsedBatchKeepsNoEarlierRow

A batch is padded from two images with three rows in the first, then the same object is padded
from one image with one box. The result has one image of one row: label 6, valid, and the corners
8, 8, 24 and 24 of a 16 by 16 box centred at (16, 16) on a 32 by 32 input. Arrays that kept their
earlier length would leave the first batch's rows behind.

## RefusesNoImagesAndRowArraysOfUnequalLengths

A batch of 0 images is refused, and so is a call with two image indices but one class. The
message names the expected `[n], [n] and [n, 4]` shapes and the lengths it got.
