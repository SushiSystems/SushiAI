# tests/vision/detect/test_ground_truth.cpp


Suite: `Small_DetectGroundTruth`.

Tests the refill of a used batch by `pad_ground_truth` in
`modules/vision/detect/include/SushiAI/vision/detect/ground_truth.hpp`, the scaling of the corners
and the rows it refuses. The earlier case of a default batch is deleted: it compared a struct's
default fields with literals.

This file needs no device. The suite `Medium_DetectGroundTruth` in
`tests/vision/detect/free/test_assigner.cpp` checks the padding itself: rows per image, the
conversion to pixel corners, a row without extent, a batch without rows, a row outside the batch
and the reference labels. This file covers what those cases leave open. Every corner here is a
multiple of 8, which float32 holds exactly, so the arrays are compared for equality.

## PaddingIntoAUsedBatchKeepsNoEarlierRow

A batch is padded from two images with three rows in the first, then the same object is padded
from one image with one box. The result has one image of one row: label 6, valid, and four box
values. Arrays that kept their earlier length would leave the first batch's rows behind.

## PaddingIntoAUsedBatchWritesTheCornersOfTheNewRow

After the same two paddings the corners are 8, 8, 24 and 24: a 16 by 16 box centred at (16, 16) on
a 32 by 32 input.

## ScalesXByTheWidthAndYByTheHeight

On a 64 by 32 input the centre (0.5, 0.25) and the size (0.25, 0.5) give the corners 24, 0, 40
and 16. Every other case passes 32 for both sides, so only this one tells a swapped width from a
height.

## RefusesNoImages

A batch of 0 images is refused as `InvalidDescription`, naming the positive batch and the batch 0.

## RefusesRowArraysOfUnequalLengths

A call with two image indices but one class is refused. The message names the expected
`[n], [n] and [n, 4]` shapes and the lengths it got.

## RefusesARowThatNamesANegativeImage

A row that names image -1 in a batch of 2 is refused as `InvalidDescription`, naming the image and
the batch, as a row past the batch is.
