# tests/vision/training/test_best_shape_head_delta_rule.cpp

Suites: `Small_BestShapeHeadDeltaRule`.

Tests `BestShapeHeadDeltaRule`. The heads decode exponential sides.

### CheckAcceptsHeadsThatDecodeExponentials

`check` succeeds for the fixture.

### CheckRefusesAHeadOfSquaredSigmoidSides

With the second head changed to squared sigmoid sides, `check` fails with `InvalidDescription`; the message holds "exponentials" and "head 'head.1'".

### ComputeWritesTheDeltaOfEachHead

For each head, the delta, assigned rows, assigned classes and assignment count equal those of `compute_head_delta` run on that head alone, and at least one slot is assigned.

### ComputeSumsTheCostOfEachHead

The total, box, non-box and stored shares equal the sums of `measure_head_cost` over the two heads.

### ComputeReplacesTheDeltasItWasGiven

Deltas filled with 99 before the call hold `[B, H, W, A * (5 + C)]` values and `[B, H, W, A]` assigned rows after it.

### ComputeRefusesATruthOfAnotherBatch

A truth of 2 images against heads of 1 fails with `InvalidDescription`; the message holds "1 images" and "batch of 2".
