# tests/vision/detect/anchor/test_anchor_clustering.cpp


Suites: `Small_DetectAnchorClustering`, `Small_DetectAnchorClusteringLadder` and
`Small_DetectAnchorClusteringOptions`.

Tests `cluster_anchors`. The clustering cases use a ladder of six box sizes with 400 boxes per
rung, 2400 boxes in all, each jittered by 2 %. The jitter is drawn from `reference_splitmix64`
in `support/splitmix64.hpp`, so the file does not name `SeededSequence`; its own cases are in
`test_seeded_sequence.cpp`. A rung counts as recovered when an anchor lies within 5 % of it.

## EverySeedRecoversEveryRung

One row per clustering seed: the default seed and 0x1234. With a count of 6, every rung is
recovered. The mean best IoU is above 0.95 and not below its value before refinement, 2400 boxes
are used and none rejected, and the iteration and sweep counts stay within the options' limits.

## AnchorsAreInAscendingArea

Each anchor's area is larger than the area of the anchor before it.

## OutliersLeaveEveryCentreWithinFivePercent

With 24 extra boxes of 0.9 x 0.9, 1 % of the ladder, 2424 boxes are used, every rung is still
recovered, and the mean best IoU is not below its value before refinement.

## TwoRunsAreBitIdentical

Two runs over the same boxes and options return anchors whose widths and heights have the same
bits, the same bits in both mean IoU numbers, and equal iteration and sweep counts.

## ZeroSweepsKeepTheFitBeforeRefinement

With `max_refinement_sweeps` 0 the result reports zero sweeps, and its mean best IoU equals the
value before refinement within 16777216 times the double unit roundoff.

## RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.

## RefusesFewerDistinctBoxesThanAnchors

Two distinct sizes, each repeated ten times, with a count of 3 fail with `Incompatible`.

## LeavesOutAZeroAndANaNSide

Three sizes repeated five times, plus one box with a zero side and one with a NaN side, give
2 rejected and 15 used boxes. The three anchors equal the three sizes within 32 times the float
unit roundoff and the mean best IoU is 1 within the double mean tolerance.

## LeavesOutABoxAtTheSmallestSide

With `min_side` 0.125, the boxes 0.125 x 0.5 and 0.5 x 0.125 are rejected and 0.25 x 0.5 is the
one box used and the one anchor.

## AnOptionOutsideItsRangeIsRefused

One row per refused option: a count of 0, a `max_iterations` of 0, a `max_refinement_sweeps`
of -1, a `min_side` of -0.5 and a NaN `min_side`. Each fails with `InvalidDescription`.
