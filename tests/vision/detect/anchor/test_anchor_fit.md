# tests/vision/detect/anchor/test_anchor_fit.cpp


Suites: `Small_DetectAnchorFit`.

Tests `measure_anchor_fit`, which measures how well the anchors of an `AnchorLayout` fit a list
of box sizes. Every number is compared to 1e-9.

## OneBoxPerAnchorGivesOneEverywhere

One box at each of three anchors' sizes gives 3 boxes, a mean best IoU, a best possible recall
and a tenth percentile of 1, an unowned share of 0 and a share of 1/3 per anchor.

## HalfSizedBoxesGiveAQuarterAndAreRecalledAtAQuarter

Boxes at half each anchor's sides give a mean best IoU and a tenth percentile of 0.25. The best
possible recall is 1 with `recall_iou` 0.25 and 0 with `recall_iou` 0.3.

## V4TinyMasksLeaveASixthOfTheBoxesUnowned

For a six-anchor layout whose two heads list five of the anchors, 100 boxes at each anchor's
size give 600 boxes, an unowned share of 1/6, a best possible recall of 5/6, a mean best IoU of
1 and a share of 1/6 per anchor.

## MasksOverEveryAnchorLeaveNoBoxUnowned

With the heads listing all six anchors, the same boxes give an unowned share of 0 and a best
possible recall of 1.

## FourNumbersEqualTheBoxByBoxSums

For four anchors, of which no head lists anchor 0, and twelve boxes, the case finds each box's
best anchor with `shape_iou` and sums by hand. The mean best IoU, the recall at 0.5, the unowned
share, the tenth percentile, which is the second smallest best IoU, and the four anchor shares
equal those sums. The case also asserts that some boxes are unowned, some recalled and some
neither.

## TenthPercentileOfTenBoxesIsTheSmallest

Nine boxes at the anchor's size and one at half its sides give a tenth percentile of 0.25 and a
mean best IoU of 0.925.

## EqualAnchorsGiveTheBoxToTheLowerIndex

With two equal anchors, of which the one head lists anchor 1, one box at that size gives the
shares 1 and 0, an unowned share of 1 and a best possible recall of 0.

## RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.

## RefusesALayoutThatDoesNotValidate

A layout whose head names an anchor past the list fails with `InvalidDescription`.

## RefusesARecallIoUOutsideZeroToOne

A `recall_iou` of -0.1, of 1.5 and of NaN each fail with `InvalidDescription`.
