# tests/vision/detect/anchor/test_head_delta.cpp


Suites: `Unit_DetectAnchorDeltaIou`.

Tests the delta of one anchor head when more than one anchor may take a truth and the box
deltas come from the IoU gradient. Truths are given in pixels of a 64-pixel network side. A box
delta is compared with a value the file computes in double, within 1e-5 absolute plus 1e-5
relative.

## TwoAnchorsAboveTheMatchThresholdAreBothAssigned

With a match threshold of 0.5, the best slot and a second slot of the truth's cell both get the
box deltas of their own decoded box against the truth, an objectness delta of 0.5 and a class
delta of 1. The other ten slots get an objectness delta of -0.5 and no box delta. The result
counts two assignments, and only the best slot records the truth in `assigned_row` and
`assigned_class`.

## AFurtherAssignmentAddsItsIoUToTheSum

With zero logits in both assigned slots, the IoU sum is 1 + 0.2 / 0.29 to 1e-6.

## AMatchThresholdOfOneAssignsTheBestAnchorAlone

At the default threshold the best slot gets its box deltas, the second slot gets none and an
objectness delta of -0.5, and the result counts one assignment.

## AnAnchorBelowTheMatchThresholdIsNotAssigned

With a match threshold of 0.75, above the second anchor's shape IoU of 0.2 / 0.29, the result
counts one assignment and the second slot has no box delta.

## AHeadWithoutTheBestAnchorStillTakesAMatchingTruth

Head 1, which does not own the truth's best anchor, gets box deltas, an objectness delta of 0.5
and a class delta of 1 at a match threshold of 0.5. The result counts one assignment, and
`assigned_row` and `assigned_class` stay -1.

## TwoPositiveClassDeltasHalveTheBoxDeltasOfTheSlot

Two truths of different classes in one slot, at a match threshold of 0.5 and a box weight of
0.07, leave the slot with the sum of the two box deltas divided by 2. Both class deltas are
0.5, and the result counts two assignments.

## AMatchThresholdOfOneDividesNothing

At the default threshold the same slot holds the undivided sum of the two box deltas.

## TheDivisionCoversSquaredErrorBoxes

With `BoxLoss::SquaredError`, a match threshold of 0.5 and two truths that are the anchor at
the cell centre, the width delta is 2 * k * (0 - 0.5) divided by 2, with
k = (2 - 0.5 * 0.5) * 0.75, and the other three box deltas are 0.

## ABoxDeltaBeyondTheBoundIsClipped

With `max_delta` 0.5, each box delta whose magnitude exceeds 0.5 is 0.5 with its sign, and the
others are unchanged. The case asserts that at least one component is clipped and at least one
is not.

## ANonFiniteBoxComponentAddsZero

With a width logit of 200, which decodes to infinity, the width delta is exactly 0, all four
box deltas are finite and the objectness delta is 0.5.

Suites: `Unit_DetectAnchorDeltaActivated`.

Tests the delta of one anchor head whose entries arrive activated. The case holds two slots
and two classes over a 64-pixel network side. Slot 0 is assigned the truth. Slot 1 overlaps the
truth with an IoU of 0.8 and is not assigned. Both publish an objectness of 0.3.

## AnIgnoredSlotIsPulledTowardItsBestIoU

Slot 1's objectness delta is 0.8 - 0.3, to 1e-6, and its `assigned_row` is -1.

## TheObjectWeightScalesAnIgnoredSlot

With an object weight of 0.5, slot 1's objectness delta is 0.5 * (0.8 - 0.3).

## TheTargetOneLeavesAnIgnoredSlotAlone

With `ObjectnessTarget::One`, slot 1's objectness delta is exactly 0.

## ASlotWithoutAGatedClassIsNotIgnored

With both class entries of slot 1 at 0.2, its objectness delta is 0 - 0.3.

## AnAssignedSlotKeepsADeltaThatIsNotZero

Slot 0's `assigned_row` is 0 and its objectness delta is 0 - 0.3.

## AnAssignedSlotWithAZeroDeltaIsPulledTowardOne

With slot 0's objectness set to exactly 0, its objectness delta is 1.

## TheTargetOnePullsAnAssignedSlotTowardOne

With `ObjectnessTarget::One`, slot 0's objectness delta is 1 - 0.3.

## TheBoxDeltaIsTheWeightedGradientWithNoSizeFactor

With `scale_xy` 2 and a box written into slot 0, the four box deltas equal the gradient at the
decoded box against the truth, weighted by 0.75, within 1e-5 absolute plus 1e-5 relative. The
expected width entry is above 0.01 in magnitude. Slot 1's four box deltas are exactly 0.

## TheClassDeltasReadTheEntriesAsProbabilities

Slot 0's class deltas are 1 - 0.6 and 0 - 0.2. Slot 1's class deltas are exactly 0.

Suites: `Unit_DetectAnchorDeltaGradient`.

Tests the two forms of the box gradient a head's training options select. Each case decodes
one slot to a chosen box against one truth on a 20-pixel network side and reads the four box
deltas. Hand values are compared to 1e-5.

## TheDefaultFormIsAnalytic

With `box_gradient` taken from a default `HeadTrainingOptions`, a 0.2 square centred in a 0.4
square gets a width delta and a height delta of 1.25.

## TheCompatibleFormDoublesTheSizeSlopeOfTheOverlap

For the same boxes the analytic form writes 1.25 into width and height and the compatible form
writes 2.5. Both write 0 into x.

## TheCompatibleFormFlipsTheEnclosingTermOfTheCentre

For two disjoint equal squares the x delta is 0.18 / 0.2809 in the analytic form and
0.88 / 0.2809 in the compatible form. Both width deltas are 0.7 * 0.25 / 0.2809, and the
compatible form's y and height deltas are 0.

## TheCompatibleFormSimplifiesTheAspectTerm

For a 0.2 by 0.1 box centred in a 0.4 square, the case computes the aspect term and its
trade-off in double and expects the width and height deltas of each form from them. The
compatible form's are 1.25 and 2.5 moved by the trade-off times the slope. The analytic form's
scale the overlap part by one minus the squared trade-off.

## TheCompatibleFormReproducesTheDumpedDeltas

For each of the twelve cells in `K_DUMPED_CELLS`, with a box weight of 0.05, a `max_delta` of 2,
a match threshold of 0.2 and `ObjectnessTarget::BestIoU`, the compatible form's IoU sum equals
the cell's IoU to 1e-6 and each box delta equals the cell's to 1e-4 relative.

## TheAnalyticFormDepartsFromTheDumpedSizeDeltas

For the first dumped cell, the analytic form's width and height deltas each differ from the
cell's by more than 10 % of the cell's value.

## anchor/test_head_delta.cpp, anchor/test_head_training_options.cpp


The delta an anchor head with Logit coding receives: the assigned cell and slot, the IoU sum, the ignore rule and its class gate, two truths in one slot, padding rows, and the training options refused.

Suites: `Unit_DetectAnchorDelta`.

Tests the delta of one anchor head with the Logit coding, and `validate_head_training`. The
cases place truths in pixels of a 64-pixel network side and compare the whole delta with an
array built by hand, to 1e-6.

## DefaultTrainingOptionsValidate

`validate_head_training` is ok for a default `HeadTrainingOptions`.

## TrainingOptionsOutsideTheirRangeAreRefused

Nine option sets each fail with `InvalidDescription`: a match threshold of 0 and of 1.5, an
ignore threshold of -0.1, a class gate of 1.5, a box weight of -1 and of infinity, a NaN
object weight, a `max_delta` of -2, and a `box_gradient` that is no enumerator.

## OneTruthWritesTheCellAndSlotOfItsBestAnchor

One truth of the size (0.5, 0.5) centred at (0.75, 0.25) on a 2x2 grid with three anchors is
assigned to row 0, column 1, slot 1. That slot's box deltas are k times the differences in x, y,
log width and log height, with k = (2 - 0.5 * 0.5) * 0.75. Its objectness delta is 0.5 and its
class deltas are -0.5 and 0.5. Every other slot has an objectness delta of -0.5 and zeros
elsewhere. The result counts one assignment, and `assigned_row` and `assigned_class` are 0 and
1 at that slot and -1 at the others.

## AnAssignedSlotThatCoversItsTruthAddsOneToTheIoUSum

A one-cell head of zero logits whose anchor is the truth's box gives one assignment and an IoU
sum of 1 to 1e-6.

## ATruthWhoseBestAnchorAnotherHeadOwnsAssignsNothing

A truth whose size is the anchor head 1 owns leaves head 0 with no assignment, an IoU sum of 0,
`assigned_row` and `assigned_class` of -1 everywhere, and an objectness delta of -0.5 in each of
its four slots.

## AnOverlappingSlotIsIgnoredOnlyWithAClassAboveTheGate

Two images hold a slot that overlaps the truth with an IoU of 0.81 and is not assigned. Where
the class logit is -1.0, the objectness delta is exactly 0. Where it is -1.2, the objectness
delta is -0.5.

## AnOverlapBelowTheIgnoreThresholdIsNotIgnored

With an ignore threshold of 0.9, the same overlap of 0.81 leaves the objectness delta at -0.5.

## TwoTruthsInOneSlotUpdateOneClassOnly

Two truths with the same box and the classes 0 and 2 land in one slot. The box deltas are
twice those of one truth, the objectness delta is 0.5, and the class deltas are 0.5, -0.5 and
0.5. The result counts two assignments.

## AnEmptyTruthListGivesObjectnessDeltasAlone

Over random logits in [-3, 3] with an object weight of 2 and no truth, each objectness entry
holds 2 * (0 - objectness) and every other entry is 0. No slot is assigned.

## APaddingRowAssignsNothing

A truth row whose `valid` flag is 0 gives no assignment. The slot's objectness delta is -0.5
and its class delta is exactly 0.
