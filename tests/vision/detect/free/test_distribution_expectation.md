# tests/vision/detect/free/test_distribution_expectation.cpp


Suites: `Small_DetectDistributionExpectation`.

Hand distributions over four bins: uniform (mean 1.5), probabilities `1..4 / 10` (mean 2), their
reverse (mean 1), and uniform after a shift of 5 (mean 1.5). `K_EXPECTATION_TOLERANCE` is 128 u,
eight float32 steps at 4, where u is the float unit roundoff. One bin copies the logits; a wrong size returns
`InvalidDescription`.

## ABinCountBelowOneIsRefused

`expect_distances` with a bin count of 0 returns `InvalidDescription` with "reg_max must be
positive" and "got 0".
