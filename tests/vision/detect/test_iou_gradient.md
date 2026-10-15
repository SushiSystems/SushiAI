# tests/vision/detect/test_iou_gradient.cpp


Suites: `Small_DetectIouGradient`.

Tests the box gradient declared in
`modules/vision/detect/include/SushiAI/vision/detect/iou_gradient.hpp` in its two forms,
`BoxGradient::Analytic` and `BoxGradient::DarknetCompatible`. A box is a centre and a size.
The analytic form is compared with a central difference of step 1e-6, within 1e-6 relative
plus 1e-7 absolute. Hand values are compared to 1e-5.

## AnalyticMatchesTheDifferenceOnOverlappingBoxes

The analytic gradient equals the central difference for two pairs of boxes that overlap
without one holding the other.

## AnalyticMatchesTheDifferenceOnDisjointBoxes

The same holds for a pair apart along one axis and a pair apart along both.

## AnalyticMatchesTheDifferenceOnNestedBoxes

The same holds for a box inside its target and for a box around its target.

## AnalyticMatchesTheDifferenceOnSeededPairs

The same holds for 100 pairs drawn from a seeded sequence, with centres in [0.2, 0.8] and sides
in [0.1, 0.5]. A pair that lies near a kink of the measure is drawn again and not counted.

## DarknetCompatibleMatchesTheDumpedDeltas

For each of the twelve cells in `K_DUMPED_CELLS`, the compatible gradient times the box weight
0.05 equals each of the cell's deltas to 1e-4 relative.

## DarknetCompatibleDoublesTheSizeSlopeOfTheOverlap

For a 0.2 square centred in a 0.4 square, the analytic width and height components are 1.25
and the compatible ones 2.5. Both x components are 0.

## DarknetCompatibleFlipsTheEnclosingTermOfTheCentre

For two disjoint equal squares, the x component is 0.18 / 0.2809 in the analytic form and
0.88 / 0.2809 in the compatible form. Both width components are 0.7 * 0.25 / 0.2809, and the
compatible y and height components are 0.

## DarknetCompatibleSimplifiesTheAspectTerm

For a 0.2 by 0.1 box centred in a 0.4 square, the case computes the aspect term and its
trade-off in double, with `K_ASPECT_TRADE_OFF_EPS`, and expects the width and height components
of each form from them.

## BothFormsAreZeroForTwoBoxesAtOnePoint

For two boxes of zero size at one point, every component of both forms is exactly 0.
