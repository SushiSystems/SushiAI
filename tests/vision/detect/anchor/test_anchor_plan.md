# tests/vision/detect/anchor/test_anchor_plan.cpp


Suites: `Small_DetectAnchorPlan`.

Tests `plan_anchors`, which chooses or takes a count per head, clusters the boxes, assigns the
anchors to the heads by stride, measures the fit and derives the assignment thresholds. The
file uses a ladder of six sizes with 400 boxes per rung, one blob of 2000 boxes and a set of
two sizes.

## LadderWithTwoStridesPlansTwoHeadsOfThreeAnchors

The ladder under the strides 16 and 32 plans six anchors with the masks {0, 1, 2} and
{3, 4, 5}. The fit counts every box, has a best possible recall of 1, an unowned share of 0 and
a mean best IoU above 0.95. The thresholds equal `derive_assignment_thresholds` of the fit: an
ignore threshold of 0.7 and a match threshold at the fit's tenth percentile. There are four
trials, the last for 4 per head.

## ChosenCountEqualsTheStepsMadeByHand

The plan equals, to 1e-9, a plan the case builds from `choose_anchor_count`,
`assign_anchors_to_heads`, `measure_anchor_fit` and `derive_assignment_thresholds` called in
that order.

## GivenCountTriesNoCount

With `per_head` 3 the plan has no trial, and its layout is the one `cluster_anchors` to six
anchors and `assign_anchors_to_heads` give. Each head has three anchors and the best possible
recall is 1.

## GivenCountIgnoresTheBoundsOfTheCountChoice

With `per_head` 3, a plan made with a `min_per_head` of 0, a `max_per_head` of -1, a NaN
`min_gain` and a clustering count of 17 is identical to one made with the default count
options.

## StridesInAnyOrderGiveEachHeadTheAnchorsOfItsStride

Under the strides 32 and 16 the masks are {3, 4, 5} and {0, 1, 2}, and with the two masks
swapped the plan is identical to the plan for 16 and 32. Under 16, 64 and 32 with two per head,
the masks are {0, 1}, {4, 5} and {2, 3}.

## TwoCallsGiveTheSamePlan

Two calls with the seed 0x5EED and the same arguments return identical plans.

## BlobPlansOneAnchorPerHead

The blob under the strides 8, 16 and 32 plans the masks {0}, {1} and {2}, with two trials and a
best possible recall of 1.

## RefusesNoStrideAndAStrideOfZero

An empty stride list and the strides 16 and 0 each fail with `InvalidDescription`, with a
chosen count and with `per_head` 1.

## RefusesOptionsOutsideTheirRange

Six option sets each fail with `InvalidDescription`: a `per_head` of -1 and of the largest
`int64_t`, a `min_per_head` of 0, a clustering `max_iterations` of 0 with and without a given
`per_head`, and a `recall_iou` of 2.

## RefusesBoxesThatDisagreeWithTheCount

An empty box list, with a chosen and with a given count, fails with `Incompatible`. So do two
sizes under three strides with a chosen count, and two sizes under two strides with two per
head.
