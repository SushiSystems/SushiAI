# tests/vision/detect/free/test_distribution_expectation.cpp


Suites: `Small_DetectDistributionExpectation`.

Hand distributions over four bins: uniform (mean 1.5), probabilities `1..4 / 10` (mean 2), their
reverse (mean 1), and uniform after a shift of 5 (mean 1.5). `K_EXPECTATION_TOLERANCE` is 4e-6,
eight float32 steps at 4. One bin copies the logits; a wrong size throws.
