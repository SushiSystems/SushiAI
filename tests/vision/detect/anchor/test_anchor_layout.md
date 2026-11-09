# tests/vision/detect/anchor/test_anchor_layout.cpp


Suites: `Small_DetectAnchorLayout`.

Tests `validate_anchor_layout` over a layout of six anchors split between two heads, and
`shape_iou` of a size with no area. The hand values of `shape_iou` are in
`test_anchor_size.cpp`.

## ShapeIoUWithAZeroAreaIsZero

`shape_iou` is exactly 0 when either size has a zero side, and for two default sizes.

## AcceptsOneHeadPerGroupOfAnchors

`validate_anchor_layout` is ok for the two-head layout.

## AcceptsASharedAnchor

With the masks {0, 1, 2, 3} and {3, 4, 5}, where anchor 3 is shared and every anchor is used,
the layout is accepted.

## AcceptsAnAnchorNoHeadUses

With the masks {3, 4, 5} and {1, 2}, where anchor 0 is unused and none is shared, the layout is
accepted.

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
