# tests/vision/training/test_best_shape_head_delta_rule.cpp

Suites: `Small_BestShapeHeadDeltaRule` and `Small_BestShapeHeadDeltaRuleLevel`.

Tests `BestShapeHeadDeltaRule`. The heads decode exponential sides.

### CheckAcceptsHeadsThatDecodeExponentials

`check` succeeds for the fixture.

### CheckRefusesAHeadOfSquaredSigmoidSides

With the second head changed to squared sigmoid sides, `check` fails with `InvalidDescription`; the message holds "exponentials" and "head 'head.1'".

### ComputeWritesTheDeltaOfTheHead

Rows: head 0 and head 1. The delta, assigned rows, assigned classes and assignment count of the head equal those of `compute_head_delta` run on that head alone.

### ComputeAssignsTheTruthAtACost

At least one slot is assigned over the two heads and the total cost is above 0.

### ComputeSumsTheCostOfEachHead

The total, box, non-box and stored shares equal the sum of `measure_head_cost` of the first head and of the second. The expectation is the rule's own building block, not a literal: the fixture's floats are not derivable by hand.

### ComputeReplacesTheDeltasItWasGiven

Deltas filled with 99 before the call hold `[B, H, W, A * (5 + C)]` values and `[B, H, W, A]` assigned rows after it.

### ComputeRefusesATruthOfAnotherBatch

A truth of 2 images against heads of 1 fails with `InvalidDescription`; the message holds "1 images" and "batch of 2".
