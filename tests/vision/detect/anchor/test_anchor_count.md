# tests/vision/detect/anchor/test_anchor_count.cpp


Suites: `Small_DetectAnchorCount`, `Small_DetectAnchorCountOptions` and `Small_DetectAnchorCountHeads`.

Tests `choose_anchor_count`, which clusters to one more anchor per head until the gain in mean
best IoU falls below `min_gain`. The file uses three box sets: a ladder of six sizes with 400
boxes per rung, one blob of 2000 boxes, and a set of two sizes.

## LadderWithTwoHeadsChoosesThreePerHead

The ladder with two heads gives three anchors per head and six anchors. There are four trials,
for 1 to 4 per head. The gain of the second and of the third trial is at least `min_gain`, and
the gain of the fourth is below it.

## TrialsHoldTheFitOfEachCountClusteredDirectly

Each of the four trials' mean best IoU equals, within 16777216 times the double unit roundoff, that of `cluster_anchors` called with the trial's
count. The chosen clustering's mean equals `measure_mean_best_iou` over its anchors and the
mean of the chosen trial.

## BlobChoosesOnePerHead

The blob with three heads gives one anchor per head and three anchors. There are two trials,
and the gain of the second is below `min_gain`.

## MaxPerHeadOfTwoStopsAtTwo

With `max_per_head` 2 the ladder with two heads gives two per head, four anchors and two
trials, and the clustering's mean equals the second trial's.

## EqualBoundsMakeOneTrial

With `min_per_head` and `max_per_head` both 2 and three heads, the choice is two per head, six
anchors and one trial.

## TwoSizesTakeTheSecondAnchorAtTheDefaultGain

Two sizes with one head and `max_per_head` 2 give two per head. The first trial's mean is
11.625 / 21 and the second's is 1, both to 1e-6.

## TwoSizesKeepOneAnchorWhenTheGainIsBelowMinGain

With `min_gain` 0.5 the same boxes give one anchor per head and one anchor whose mean is
11.625 / 21. Both trials are returned, and the second's mean is 1.

## IgnoresTheCountOfTheClusteringOptions

With `clustering.count` set to 17, the two sizes still give two per head and two anchors.

## RefusesAMinPerHeadAboveMaxPerHead

`min_per_head` 3 with `max_per_head` 2 fails with `InvalidDescription`.

## AnOptionOutsideItsRangeIsRefused

One row per option: a `min_per_head` of 0, a `min_gain` of -0.01, a NaN `min_gain` and a
clustering `max_iterations` of 0. Each fails with `InvalidDescription`.

## AHeadCountOutsideItsRangeIsRefused

One row per head count: 0, -1 and the largest `int64_t`, whose product with the count overflows. Each fails with `InvalidDescription`.

## RefusesFewerDistinctBoxesThanTheFirstCount

Two sizes with three heads fail with `Incompatible`.

## RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.
