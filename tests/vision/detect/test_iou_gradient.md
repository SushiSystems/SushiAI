# tests/vision/detect/test_iou_gradient.cpp


Suites: `Small_DetectIouGradient`, `Small_DetectIouGradientAnalytic`,
`Small_DetectIouGradientFixed`, `Small_DetectIouGradientDumped` and
`Small_DetectIouGradientPoint`.

Tests the box gradient declared in
`modules/vision/detect/include/SushiAI/vision/detect/iou_gradient.hpp` in its three forms,
`BoxGradient::Analytic`, `BoxGradient::DarknetCompatible` and `BoxGradient::FixedTradeOff`.
A box is a centre and a size. The analytic form is compared with a central difference of step
1e-6, within 32 times the float unit roundoff relative plus 2 times the float unit roundoff
absolute. Hand values are compared within 256 times the float unit roundoff.

The analytic and fixed forms, the dumped cells and the one-point case run over rows; each row has
a name that states its input.

## MatchesTheDifference

Suite `Small_DetectIouGradientAnalytic`. The analytic gradient equals the central difference on
each row. The `Hand` rows are two pairs that overlap without one holding the other, a pair apart
along one axis and a pair apart along both, and a box inside its target and a box around its
target. The `Seeded` rows are 100 pairs drawn from a seeded sequence, with centres in [0.2, 0.8]
and sides in [0.1, 0.5]. A pair that lies near a kink of the measure is drawn again and gets no
row. The rows are built before the run by `make_seeded_rows`.

## MatchesTheDifferenceWithAFrozenTradeOff

Suite `Small_DetectIouGradientFixed`. The fixed form equals a central difference of a host
measure whose trade-off is computed once, at the box under test, and passed as a constant to
every shifted evaluation. Rows: one overlapping pair, one disjoint along both axes, one nested.
Tolerances are those of the analytic form.

## FixedTradeOffDropsTheSlopeOfTheTradeOff

For an overlapping pair of different aspects, the width component of the fixed form differs
from the analytic one by more than the hand tolerance.

## DarknetCompatibleMatchesTheDumpedDeltas

Suite `Small_DetectIouGradientDumped`. One row for each of the three cells in `K_DUMPED_CELLS`:
the compatible gradient times the box weight 0.05 equals each of the cell's four deltas within
2048 times the float unit roundoff, relative. `expect_dumped_entry` states the comparison for one
entry and the case calls it for the four entries.

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
of each form from them. The expected values restate the aspect formula, so the oracle is partly
the formula of the implementation.

## IsZeroForTwoBoxesAtOnePoint

Suite `Small_DetectIouGradientPoint`. For two boxes of zero size at one point, the gradient of
each form is exactly zero in all four components. Rows: `Analytic`, `DarknetCompatible` and
`FixedTradeOff`.
