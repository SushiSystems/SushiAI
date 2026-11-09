# tests/vision/detect/free/test_distribution_targets.cpp


Suites: `Small_DetectDistributionTargets` and `Small_DetectDistributionTargetsRows`.

Four bins, normalized weight 0.8, so each side's pair of bins sums to 0.2. `K_WEIGHT_TOLERANCE` is
32 u, eight float32 steps at 1, where u is the float unit roundoff.

## EachSideWeighsTheTwoBinsAroundItsTarget

Four rows of `Small_DetectDistributionTargetsRows`, each with one target on all four sides: 1.25
weighs bins 1 and 2 at 0.15 and 0.05, 0 weighs bin 0 at 0.2, 5 clamps to 2.99 and weighs bins 2 and
3 at 0.002 and 0.198, and -1 clamps to 0 and weighs bin 0 at 0.2.

## ABackgroundAnchorWeighsNoBin

A normalized weight of 0 leaves every weight at 0, compared with equality.

## OneBinIsRefused

One bin per side returns `InvalidDescription` with "at least 2" and "got 1".

## WeightsOfAnotherSizeAreRefused

A weight array of 15 values returns `InvalidDescription` naming "4, 1" and "15 values".

## RefusesANonFiniteTargetWithItsIndex

A NaN target at index 2 returns `InvalidDescription` with "target distance 2" and "finite".

## RefusesANonFiniteNormalizedWeightWithItsIndex

An infinite normalized weight at index 0 returns `InvalidDescription` with "normalized weight 0"
and "finite".
