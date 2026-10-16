# tests/layers/train/test_decision_threshold.cpp


Suites: `Small_DecisionThreshold`.

Tests the threshold constants and the range `Train::validate_threshold` accepts.

This file needs no device. `read_threshold` and `logit_of` are pinned under the same suite in
`test_binary_confusion.cpp`; the cases here cover the constants and the validator on its own.

## TheKeyIsThresholdAndTheDefaultIsOneHalf

`K_THRESHOLD_KEY` is `threshold`, `K_DEFAULT_THRESHOLD` is 0.5, and the validator accepts the
default.

## ProbabilitiesInsideTheOpenIntervalAreAccepted

1e-9, 0.25 and 1 - 1e-9 are each accepted, so the bounds are open and nothing inside them is
refused.

## TheBoundsAndANaNAreRefused

0, 1, -0.1, 2, infinity and a NaN are each refused with InvalidDescription naming "above 0 and
below 1". A NaN fails every comparison, so a check written as two rejections would let it pass;
this case holds the validator to the form that refuses it.
