# tests/vision/detect/anchor/test_anchor_size.cpp


Suites: `Small_DetectAnchorSize`.

Tests `shape_iou`: the hand values, the order of the two arguments, the bounds and an operand
that is not a number. The default `AnchorSize` is not tested: its two zeros are a field default.

This file needs no device. The three hand values moved here from `test_anchor_layout.cpp`,
which keeps the zero-area case. The random sizes come from `reference_splitmix64` in
`support/splitmix64.hpp` and do not name `SeededSequence`.

## K_SEED, K_PAIRS and draw_ious

`K_SEED` fixes the sequence and `K_PAIRS` is the 256 pairs of sizes drawn from it. `draw_ious`
returns the IoU of every pair in both orders, so each property case compares whole lists.
`K_IOU_TOLERANCE` is 32 times the float unit roundoff, for the hand values that float rounds
once per term.

## ASizeWithItselfHasAnIoUOfOne

`shape_iou` of 0.3 x 0.7 with itself is 1.

## AContainedSizeHasTheRatioOfTheAreasAsItsIoU

The IoU of 0.2 x 0.4 and the size 0.1 x 0.2 inside it is 0.25 in either order.

## CrossedSizesCountTheSharedRectangle

The IoU of 0.4 x 0.1 and 0.1 x 0.4 is 1/7: the shared rectangle is 0.01 and the union is
0.04 + 0.04 - 0.01.

## TheIoUIsTheSameInBothOrders

For 256 drawn pairs, `shape_iou(a, b)` equals `shape_iou(b, a)` bit for bit. Both orders form
the same products and the same sums, so no tolerance applies.

## TheIoULiesBetweenZeroAndOne

The IoU of each of the 256 drawn pairs lies in [0, 1].

## ASideThatIsNotANumberGivesNoOverlap

A size whose width is a quiet NaN gives an IoU of 0 in either argument position. The header
promises 0 when the union is not a number, and `std::min` places the NaN differently in the two
orders, so both are checked.
