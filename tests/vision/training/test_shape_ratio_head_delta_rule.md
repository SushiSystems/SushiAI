# tests/vision/training/test_shape_ratio_head_delta_rule.cpp

Suites: `Small_ShapeRatioHeadDeltaRule` and `Small_ShapeRatioHeadDeltaRuleLevel`.

Tests `ShapeRatioHeadDeltaRule`. The heads are logits with squared sigmoid sides.

### CheckAcceptsLogitHeadsOfSquaredSigmoidSides

`check` succeeds for the fixture.

### CheckRefusesAnActivatedHead, CheckRefusesAHeadOfExponentialSides

A second head with the activated coding fails with "heads of logits" and "head 'head.1'". A first head with exponential sides fails with "squared sigmoids" and "head 'head.0'". Both are `InvalidDescription`.

### CheckRefusesALayoutOfMoreLevelsThanTheBalanceWeighs

A layout of 6 heads fails with `InvalidDescription`; the message holds "at most 5 levels" and "layout has 6".

### ComputeFollowsAssignmentThenLossThenDeltaForTheLevel

Rows: level 0 and level 1. The delta, assigned rows, assigned classes and assignment count of the level equal those of `ShapeRatioAssignment`, `build_objectness_targets` and `compute_shape_ratio_delta` run by hand on that level. The expectation is the rule's own composition, so the case shows that the rule runs the three steps in order for each level and keeps each level apart. The values of the fixture are not derived by hand.

### ComputeAssignsTheTruth

At least one slot is assigned over the two levels.

### TheCostSharesAreTheSumsOfTheLevelLosses

The total equals the sum of the level loss totals, the box share the sum of the level box losses times their box factor and the batch, and the non-box share the sum of the remainders, each to 1e-12.

### TheStoredCostEqualsTheTotal

The total is above 0 and the stored cost equals it.

### TheSharesAddUpToTheTotal

The box and non-box shares add up to the total, to 1e-12 of the total.

### ComputeRefusesATruthOfAnotherBatch

A truth of 2 images against heads of 1 fails with `InvalidDescription`; the message holds "1 images" and "batch of 2".
