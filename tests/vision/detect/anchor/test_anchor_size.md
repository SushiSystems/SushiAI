# tests/vision/detect/anchor/test_anchor_size.cpp


Suites: `Small_DetectAnchorSize`.

Tests the default `AnchorSize` and the bounds and the symmetry of `shape_iou`.

This file needs no device. `tests/vision/detect/anchor/test_anchor_layout.cpp` holds the
hand-computed IoU cases: a size with itself, a contained size, a wide and a tall size, and a
zero area. This file adds what those cases leave open: the default value, the order of the two
arguments, and an operand that is not a number.

## K_SEED and draw_size

`draw_size` takes two unit draws of a `SeededSequence` as the width and the height. `K_SEED`
fixes the sequence, so every run draws the same `K_PAIRS` pairs.

## ADefaultSizeHasNoExtent

A default `AnchorSize` has a width and a height of 0.

## TheIoUIsSymmetricAndBetweenZeroAndOne

For 256 random pairs, `shape_iou(a, b)` equals `shape_iou(b, a)` bit for bit and lies in [0, 1].
Both orders form the same products and the same sums, so no tolerance applies.

## ASideThatIsNotANumberGivesNoOverlap

A size whose width is a quiet NaN gives an IoU of 0 in either argument position. The header
promises 0 when the union is not a number, and `std::min` places the NaN differently in the two
orders, so both are checked.
