# tests/vision/detect/anchor/test_anchor_fit.cpp


Suites: `Small_DetectAnchorFit`.

Tests `measure_anchor_fit`, which measures how well the anchors of an `AnchorLayout` fit a list
of box sizes. Every number is compared within 16777216 times the double unit roundoff.

## OneBoxPerAnchorGivesOneEverywhere

One box at each of three anchors' sizes gives 3 boxes, a mean best IoU, a best possible recall
and a tenth percentile of 1, an unowned share of 0 and a share of 1/3 per anchor.

## HalfSizedBoxesGiveAQuarter

Boxes at half each anchor's sides give a mean best IoU and a tenth percentile of 0.25.

## HalfSizedBoxesAreRecalledAtAQuarter

With `recall_iou` 0.25 the best possible recall of the same boxes is 1.

## HalfSizedBoxesAreNotRecalledAboveAQuarter

With `recall_iou` 0.3 the best possible recall of the same boxes is 0.

## V4TinyMasksLeaveASixthOfTheBoxesUnowned

For a six-anchor layout whose two heads list five of the anchors, 100 boxes at each anchor's
size give 600 boxes, an unowned share of 1/6, a best possible recall of 5/6, a mean best IoU of
1 and a share of 1/6 per anchor.

## MasksOverEveryAnchorLeaveNoBoxUnowned

With the heads listing all six anchors, the same boxes give an unowned share of 0 and a best
possible recall of 1.

## FourNumbersAndTheSharesMatchHandValues

Four anchors, 0.25 x 0.125 (listed by no head), 0.5 x 0.5, 0.5 x 0.125 and 0.125 x 0.5, and six
boxes: one at each anchor's size, one of 0.0625 x 0.0625 and one of 0.5 x 0.0625. Every side is
a power of two, so the best IoUs are exact: 1, 1, 1, 1, 1/8 and 1/2. The mean best IoU is
4.625 / 6, the recall at 0.5 is 4/6, the unowned share is 2/6, the tenth percentile is the
smallest best IoU, 1/8, and the anchor shares are 2/6, 1/6, 2/6 and 1/6.

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

## ARecallIoUOutsideZeroToOneIsRefused

One row per `recall_iou`: -0.1, 1.5 and NaN. Each fails with `InvalidDescription`.
