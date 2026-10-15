# tests/vision/detect/anchor/test_anchor_layout.cpp


Suites: `Unit_DetectAnchorLayout`.

Tests `shape_iou` over two `AnchorSize` values and `validate_anchor_layout` over a layout of
six anchors split between two heads.

## ShapeIoUOfASizeWithItselfIsOne

`shape_iou` of 0.3 x 0.7 with itself is 1 to 1e-6.

## ShapeIoUOfAContainedSizeIsTheRatioOfTheAreas

`shape_iou` of 0.2 x 0.4 and 0.1 x 0.2 is 0.25 in either order.

## ShapeIoUOfCrossedSizesCountsTheSharedRectangle

`shape_iou` of 0.4 x 0.1 and 0.1 x 0.4 is 1/7.

## ShapeIoUWithAZeroAreaIsZero

`shape_iou` is exactly 0 when either size has a zero side, and for two default sizes.

## AcceptsOneHeadPerGroupOfAnchors

`validate_anchor_layout` is ok for the two-head layout.

## AcceptsASharedAnchorAndAnAnchorNoHeadUses

With the masks {3, 4, 5} and {1, 2, 3}, where anchor 3 is shared and anchor 0 unused, the
layout is accepted.

## RefusesAnIndexPastTheAnchors

A mask entry of 6 with six anchors fails with `InvalidDescription`.

## RefusesANegativeIndex

A mask entry of -1 fails with `InvalidDescription`.

## RefusesAnEmptyHead

A head with an empty mask fails with `InvalidDescription`.

## RefusesALayoutWithoutAnchorsOrWithoutHeads

A layout with no anchor and a layout with no head each fail with `InvalidDescription`.

## RefusesAnAnchorWithoutAFinitePositiveSide

An anchor with a zero width, one with a NaN height and one with an infinite width each fail
with `InvalidDescription`.
