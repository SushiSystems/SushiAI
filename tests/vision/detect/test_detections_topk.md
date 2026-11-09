# tests/vision/detect/test_detections_topk.cpp


Suites: `Medium_DetectPostprocess`, `Small_DetectPostprocess` and
`Medium_DetectPostprocessReference`.

Tests `select_top_detections`, `count_top_detections` and `keep_confident_detections` in
`modules/vision/detect/include/SushiAI/vision/detect/detections_topk.hpp`. A detection row is the
four corners, the score and the class. The values the postprocess copies from its input are
exact, so every array is compared for equality.

The suite `Medium_DetectPostprocessReference` is a fixture. Its `SetUp` opens
`yolo26/forward.sacp` and reads the decoded boxes, the decoded scores and the detections; when the
file is absent it skips every case of the suite with the reason. The cases hold no skip guard.

## SelectsTheHighestScoringAnchorClassPairs

Three anchors of two classes, limit 2: anchors 1 and 0 lead with 0.9 and 0.7, anchor 1 wins with
class 0 and anchor 0 with class 1. The rows come best first.

## OneAnchorCanYieldTwoDetections

When two classes of one anchor score highest, that anchor gives both rows: 0.8 of class 0 and 0.7
of class 1.

## LimitsRowsToTheAnchorCount

Small. `count_top_detections` gives 3 for 3 anchors and a limit of 300, and 300 for 8400 anchors
and a limit of 300.

## SelectsEachImageFromItsOwnRows

Two images of two anchors and two classes, limit 1: each image gives its own best row, 0.6 of
class 0 and 0.9 of class 1.

## RejectsADetectionArrayOfTheWrongSize

Small. A detections array of 6 values for 1 image of 2 rows is refused as `InvalidDescription`,
naming `[1, 2, 6] detections` and `6 values`.

## ReproducesTheReferenceDetections

Fixture case. The 300 rows from the reference boxes and scores have the reference's score column
and class column exactly. The rows above the last score equal the reference's rows above it, each
set sorted, since rows that share the last score are interchangeable. Every row is the box of
some anchor with that anchor's score of the row's class; `is_every_row_an_anchor_class_pair`
states that over the rows.

## TheConfidenceFilterDropsAScoreAtTheThreshold

Rows at 0.5, 0.25, 0.26 and 0.1 with a threshold of 0.25 keep 0.5 and 0.26, in order.

## TheConfidenceFilterKeepsEachImagesRows

Two images, the first with one row at 0.9 and one at 0, the second with none above the validation
confidence: the first keeps its row and the second keeps nothing.

## TheConfidenceFilterRejectsRowsTheBatchDoesNotDivide

Three rows of six values for a batch of 2 are refused as `InvalidDescription`, naming
`[2, k, 6] detections` and `18 values`.

## TheConfidenceFilterRejectsABatchOfZero

A batch of 0 is refused as `InvalidDescription`, naming the positive batch and `got 0`. The case
is the second half of the earlier combined refusal case.

## TheConfidenceFilterKeepsTheRowsTheReferenceValidatorKeeps

Fixture case. At the validation confidence 0.001 no reference row is kept, as the manifest's
`validation.row_count.one2one` says, and at a threshold of 0 all 300 are, equal to the reference.

## CountingRefusesALimitBelowOne

Small. A limit of 0 is refused as `InvalidDescription`, naming the positive extents and
`a limit of 0`.

## CountingRefusesAnExtentThatIsNotPositive

Small. A shape of 0 anchors is refused as `InvalidDescription`, naming the positive extents and
`0 anchors`.
