# tests/vision/detect/test_suppression_criterion.cpp


Suites: `Small_DetectSuppressionCriterion` and `Medium_DetectSuppressionCriterionDeathTest`.

Tests `Detect::measure_suppression`, the six-argument `Detect::suppress_overlaps`, and the
criterion fields of `Detect::NMSConfig`. Boxes are corner pairs, and most cases measure against
one 10 by 10 square. Measures are compared within 32 times the float unit roundoff.
The two cases that the review moved are not here: the disjoint and touching-edge IoU is held by
`test_iou.cpp`, and the IoU criterion at its threshold by `test_detections_nms.cpp`.

## IoUOfOverlappingBoxesIsTheRatioOfAreas

The square and a copy shifted by 1 along x have an IoU of 90 / 110, and the criterion measure
equals `intersection_over_union` on the same boxes.

## IoUOfTwoEmptyBoxesIsZeroNotNaN

Two boxes of zero area at one point have an IoU that is not NaN and is exactly 0.

## DistanceIoUSubtractsTheNormalisedCentreDistance

For the square and a copy shifted down by 6, with the exponent 0.6, the measure is
0.25 - (36 / 356)^0.6.

## DistanceIoUOfDisjointBoxesIsNegative

For the square and a box 20 units to its right, the measure is -(0.4^0.6).

## ZeroExponentSubtractsOne

With the exponent 0 the measure of the square and its shifted copy is 0.25 - 1.

## DistanceIoUOfIdenticalBoxesIsOne

The square measured against itself gives 1.

## DistanceIoUOfCoincidentPointsIsTheIoU

Two boxes of zero area at one point measure exactly 0, and the value is not NaN.

## DistanceIoUIsSymmetric

The measure of the square and a diagonally shifted box is the same in both orders.

## DistanceIoUKeepsTheFartherOfTwoEqualIoUBoxes

Of three boxes scored 0.9, 0.8 and 0.7, at a threshold of 0, the DistanceIoU criterion keeps
rows 0 and 1. The IoU criterion keeps row 0 alone.

## ZeroScoreRowsAreNeverKept

Of three boxes that do not overlap, scored 0.9, 0 and 0.5, the rows 0 and 2 are kept.

## EmptyInputKeepsNothing

With no box and no score, a `kept` vector that held one entry comes back empty.

## FourArgumentFormEqualsTheIoUCriterion

For four boxes that do not overlap, scored 0.1, 0.9, 0.5 and 0.5, the four-argument
`suppress_overlaps` keeps the rows 1, 2, 3, 0 in that order, and the six-argument form with the
IoU criterion keeps the same.

## SelectionAppliesTheConfiguredCriterion

`select_nms_detections` with the DistanceIoU criterion, a confidence threshold of 0.1 and an
IoU threshold of 0 over the three boxes of the equal-IoU case, given as centre and size,
returns one image with two detections: the corners (0, 0, 10, 10) scored 0.9 and
(0, 6, 10, 16) scored 0.8, both of class 0.

## NMSConfigDefaultsToTheIoUCriterion

A default `NMSConfig` over two 10 by 10 boxes shifted by 1.5 suppresses the second: their IoU,
85 / 115 = 0.739, is above the default threshold 0.7, while their DistanceIoU with the default
exponent, about 0.677, is below it. The case asserts the behaviour of the default, not its field
values.

## RejectsBoxesOfTheWrongSizeFatally

A death test, so its suite is medium. Five box values with two scores stop the process with a message holding "a suppression needs"
and "got 5 and 2 values".

## RejectsAnUnknownCriterionFatally

A death test, so its suite is medium. A criterion cast from 7 stops the process with a message holding "must be IoU or DistanceIoU"
and "got 7".
