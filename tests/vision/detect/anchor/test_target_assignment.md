# tests/vision/detect/anchor/test_target_assignment.cpp


Suites: `Small_DetectAnchorTargetAssignment`.

Tests `TargetAssignment`, `AssignmentInput` and the `AssignmentRule` interface of
`anchor/target_assignment.hpp`. The interface has one implementation, so its cases call
`ShapeRatioAssignment` through an `AssignmentRule` reference; the rule's own placement cases are in
`test_shape_ratio_assignment.cpp`. The file needs no device.

The layout has one head over the anchors 0.25 and 1.0 of the input. The head is 8 by 8 cells over a
64 pixel input, with three classes. The truth has two images of one valid 16 pixel box each: image 0
centred at (36, 36) with the label 1, image 1 centred at (20, 12) with the label 2. Each centre lies
in the middle of its cell, so each box takes that cell alone, and only the anchor 0.25 fits its side
ratio.

## ADefaultAssignmentIsZero

Every field of a default `TargetAssignment` is 0.

## ADefaultInputNamesNoLayoutAndNoTruth

A default `AssignmentInput` has a null layout and truth, head 0, no classes, an empty head shape and
a network size of 0 by 0.

## ARuleOrdersTheEntriesByImageThroughTheInterface

The result holds two entries. The first is image 0, slot 0, row 4, column 4, label 1, truth row 0,
with the box target (0.5, 0.5) and the size 2 by 2 grid cells. The second is image 1, slot 0, row 1,
column 2, label 2 and truth row 0. Box values hold to 1e-6.

## ARuleReplacesWhatTheResultHeld

A result of five entries, the first with the label 99, holds the two entries of the rule afterwards,
and the first has the label 1.

## ABatchWithoutAValidRowAssignsNothing

A truth whose rows are all marked invalid leaves a result that held three entries empty.

## AHeadOutsideTheLayoutIsRefused

Head 1 of a layout with one head fails with `InvalidDescription`; the message holds "head 1" and
"layout's 1".

## AGridWithoutACellIsRefused

A head shape of 0 rows fails with `InvalidDescription`, and the message holds "holds no cell".
