# tests/vision/detect/anchor/test_anchor_review.cpp


Suites: `Unit_DetectAnchorReview`.

Tests `review_anchors`, which measures a current `AnchorLayout` on a box list, clusters a
proposal with the same counts per head and returns `AnchorVerdict::Keep` or
`AnchorVerdict::Propose`. The boxes are a ladder of six sizes with 400 boxes per rung, and the
shifted ladder is the same one scaled by 0.75.

## FittedLayoutIsKeptOnItsOwnBoxes

A layout fitted to the ladder and reviewed on the ladder gets `Keep`. Its best possible recall
is 1, and the proposal's mean best IoU equals the current one to 1e-12.

## ShiftedBoxesGetAProposalWithBothFits

Reviewed on the shifted ladder, the layout gets `Propose`. The reported current fit and
proposed fit equal `measure_anchor_fit` called directly on each layout, and their mean best IoU
and recall equal sums the case makes by hand to 1e-6. The proposal's gain in mean best IoU is
above `min_gain`.

## ProposalRecoversTheShiftedRungs

The proposal has six anchors, and each lies within 5 % of the width and the height of its
shifted rung.

## ProposalKeepsTheCountsPerHead

For a current layout with 2 and 4 anchors under the strides 32 and 16, the proposal has the
same anchor count and the masks {4, 5} and {0, 1, 2, 3}.

## ProposalIsTheClusteringAssignedToTheHeads

With `clustering.count` set to 99 in the options, the proposal still equals the layout fitted
directly to the shifted boxes with three anchors per head.

## InputLayoutIsUnchanged

After a review that returns `Propose`, the reviewed layout holds the anchors and masks it held
before.

## RecallAndGainEachDecideTheVerdict

With `min_gain` 10 the shifted review has a current recall of 1 and returns `Keep`. With
`recall_iou` raised to 0.6 as well, the current recall is 0, which equals the hand sum, and the
verdict is `Propose`.

## RefusesOptionsOutsideTheirRange

A `min_gain` of -0.01 and of NaN, a `min_recall` of 1.5 and of -0.1, and a `recall_iou` of 2
each fail with `InvalidDescription`.

## RefusesALayoutWithoutHeads

A layout with one anchor and no head fails with `InvalidDescription`.

## RefusesBoxesStridesAndCountsThatDisagree

Three sizes with a layout of masks {0} and {1, 2} under two strides are accepted. An empty box
list, four copies of one box, one stride for two heads, and a layout whose masks {0, 1} and
{1, 2} hold four slots for three anchors each fail with `Incompatible`.
