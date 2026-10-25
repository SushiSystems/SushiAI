# tests/vision/detect/anchor/test_shape_ratio_delta.cpp

Suites: `Small_DetectAnchorShapeRatioDelta`.

Tests the loss, the objectness targets, the factors and the delta in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/shape_ratio_delta.hpp`. Assignments
are written by hand, ordered as `AssignmentRule::assign` orders them.

## TheDeltaIsTheCentralDifferenceOfTheLoss

One image, two levels of two anchors each (4 by 4 and 2 by 2 cells, 3 classes, a 64-pixel
input), two truths: the first takes two neighbour cells of slot 0 on level 0, the second slot 1
of level 0 and slot 0 of level 1. Head logits follow `0.8 sin(0.37 i + 0.1)`. Every delta equals
minus the central difference of the level's loss by that logit, step 1e-3, with the objectness
targets held at their base values, within 1e-4 absolute.

## ObjectnessTargetAndBalanceOnAHandWorkedCase

Level 2 of three, 2 by 2 cells of one anchor one cell wide, one class, a 640-pixel input, zero
logits. The assigned slot decodes to its truth exactly, so its target is 1 and every other
target 0. The objectness factor is 1 x 1 x 3/3 x 0.4 over 4 slots, which gives a delta of 0.05
at the assigned slot and -0.05 elsewhere.

## TheBalanceFollowsTheLevelCount

Three levels give 4, 1, 0.4; two, four and five levels read 4, 1, 0.25, 0.06, 0.02.

## TheFactorsDeriveFromLevelsClassesAndInputSize

Level 1 of two, 40 classes, a 320-pixel input: box 0.05 x 1.5, class 0.5 x 0.5 x 1.5,
objectness 1 x 0.25 x 1.5 x 1.

## RefusesAnActivatedHeadAndAnAssignmentOutsideIt

An activated head fails with `InvalidDescription` and "logits"; an assignment to row 2 of a
2-row head fails with "outside head 1".
