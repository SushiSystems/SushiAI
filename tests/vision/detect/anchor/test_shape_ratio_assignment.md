# tests/vision/detect/anchor/test_shape_ratio_assignment.cpp

Suites: `Small_DetectAnchorShapeRatioAssignment`.

Tests `ShapeRatioAssignment` and `validate_shape_ratio_options` in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp`. Every
case runs the default settings on one image, a 64-pixel input and an 8 by 8 head over anchors
0.25 and 1.0 of the input, 2 and 8 cells wide. Truths are square; every value is exact in float.

## ACentreOnTheMiddleTakesItsCellAlone

A 16-pixel truth centred at pixel 36, grid 4.5, takes slot 0 of cell (4, 4) alone, with the
target (0.5, 0.5, 2, 2).

## ACentreNearTheTopLeftGainsTheLeftAndTopCells

A centre at grid 4.25 takes cells (3, 4), (4, 3) and (4, 4), with targets (0.25, 1.25),
(1.25, 0.25) and (0.25, 0.25).

## ACentreNearTheBottomRightGainsThoseCells

A centre at grid 4.75 takes cells (4, 4), (4, 5) and (5, 4), with targets (0.75, 0.75),
(-0.25, 0.75) and (0.75, -0.25).

## ACentreAtTheBorderGainsNoNeighbour

A centre at grid 0.25 lies near the left and top edges but not above 1, so it takes cell (0, 0)
alone.

## AnAnchorAtTheRatioLimitIsRejected

A truth 2 cells wide against the 8-cell anchor has a ratio of exactly 4 and is not assigned to
slot 1.

## ACollisionKeepsTheTruthOfHigherShapeIoU

Row 0, 3 cells wide, and row 1, 2 cells wide, share a centre. Slot 0 goes to row 1, whose
`shape_iou` is 1 against row 0's 4/9; slot 1 goes to row 0, which row 1's ratio of 4 cannot
take.

## ACollisionOfEqualTruthsKeepsTheLowerRow

Two equal truths on one slot leave row 0 there.

## RefusesBadSettingsNamingTheKey

A `ratio_limit` of 1 and a `neighbour_offset` of 0.75 are refused with the key named; the
defaults pass.
