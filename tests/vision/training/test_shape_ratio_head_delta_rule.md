# tests/vision/training/test_shape_ratio_head_delta_rule.cpp

Suites: `Small_ShapeRatioHeadDeltaRule`.

Tests `ShapeRatioHeadDeltaRule`. The heads are logits with squared sigmoid sides.

### CheckAcceptsLogitHeadsOfSquaredSigmoidSides

`check` succeeds for the fixture.

### CheckRefusesAnActivatedHead, CheckRefusesAHeadOfExponentialSides

A second head with the activated coding fails with "heads of logits" and "head 'head.1'". A first head with exponential sides fails with "squared sigmoids" and "head 'head.0'". Both are `InvalidDescription`.

### CheckRefusesALayoutOfMoreLevelsThanTheBalanceWeighs

A layout of 6 heads fails with `InvalidDescription`; the message holds "at most 5 levels" and "layout has 6".

### ComputeFollowsAssignmentThenLossThenDeltaForEachLevel

For each level, the delta, assigned rows, assigned classes and assignment count equal those of `ShapeRatioAssignment`, `build_objectness_targets` and `compute_shape_ratio_delta` run by hand on that level. The cost total, box and non-box shares equal the sums of the level losses and the box factors to 1e-12.

### TheStoredCostEqualsTheTotalAndTheSharesAddUpToIt

The stored cost equals the total, which is above 0, and the box and non-box shares add up to it to 1e-12 relative.

### ComputeRefusesATruthOfAnotherBatch

A truth of 2 images against heads of 1 fails with `InvalidDescription`; the message holds "1 images" and "batch of 2".
