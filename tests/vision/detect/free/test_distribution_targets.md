# tests/vision/detect/free/test_distribution_targets.cpp


Suites: `Small_DetectDistributionTargets`.

Four bins, normalized weight 0.8, so each side's pair sums to 0.2. Targets 1.25, 0, 5 (clamped to
2.99) and -1 (clamped to 0); a second, background row stays 0. `K_WEIGHT_TOLERANCE` is 1e-6.
