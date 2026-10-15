# tests/vision/detect/anchor/test_assignment_thresholds.cpp


Suites: `Small_DetectAnchorThresholds`.

Tests `AssignmentThresholds` and `derive_assignment_thresholds`. Each fit in the file is given
by its tenth percentile and its mean best IoU, and thresholds are compared to 1e-6.

## DefaultsAreOneAndSevenTenths

A default `AssignmentThresholds` has a match threshold of exactly 1 and an ignore threshold of
exactly 0.7.

## FitInsideBothRangesGivesItsOwnNumbers

A tenth percentile of 0.35 and a mean of 0.62 give a match threshold of 0.35 and an ignore
threshold of 0.62.

## FourFitsFollowTheFormula

For the fits (0.35, 0.62), (0.21, 0.51), (0.5, 0.69) and (0.25, 0.55), the match threshold is
the tenth percentile and the ignore threshold is the mean.

## MatchThresholdIsClampedAtBothEnds

Tenth percentiles of 0.05, 0 and 0.2 give a match threshold of 0.2. Tenth percentiles of 1 and
1.3 give 1.

## IgnoreThresholdIsClampedAtBothEnds

Means of 0.3, 0 and 0.5 give an ignore threshold of 0.5. Means of 0.7 and 0.95 give 0.7.

## PerfectFitGivesOneAndSevenTenths

A fit of 1 in both numbers gives a match threshold of 1 and an ignore threshold of 0.7.
