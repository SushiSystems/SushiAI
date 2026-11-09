# tests/vision/detect/test_detections_nms.cpp


Suites: `Medium_DetectPostprocess`, `Small_DetectPostprocess` and
`Medium_DetectPostprocessSuppressionReference`.

Tests the suppression path of
`modules/vision/detect/include/SushiAI/vision/detect/detections_nms.hpp`: `suppress_overlaps` and
`select_nms_detections`. The top-k path has its own page, `test_detections_topk.md`, and the
selection of `select_top_k` has `test_top_k.md`.

The suppression path takes boxes as centre and size, so the hand cases write them that way and
expect corners. A postprocess copies its output values from its inputs, so every array is
compared for equality. The suite `Medium_DetectPostprocessSuppressionReference` is a fixture: its
`SetUp` opens `yolo26/forward.sacp`, reads the one2many boxes, scores and suppressed detections,
and skips every case of the suite with the reason when the file is absent. The cases hold no skip
guard.

Two of the reference comparisons cannot compare row by row. The reference orders equal scores in
an order torch does not specify, and the fixture's scores are nearly constant.
`modules/vision/detect/README.md`, under `top_k.hpp` and "Reference rows the suppression test
reads", gives the counts.

## K_SUPPRESSION_CONFIDENCE

`K_SUPPRESSION_CONFIDENCE` is 1.8e-4, the `suppression.arguments.conf_thres` of `forward.sacp`
in `tests/vision/registration/fixtures/yolo26/MANIFEST.json`.

## Helpers

`Row` is an array of six floats, one detection. `column` returns the values at one offset of every
row. `sorted_rows_above` returns the rows whose score exceeds a floor, sorted by their six values,
so two arrays that hold the same rows in different orders compare equal through it.
`make_nms_config` returns an `NMSConfig` with the given confidence threshold, IoU threshold,
label mode and class mode, and the default limits. `suppress_one_image` runs
`select_nms_detections` on one image of `[A, 4]` boxes and returns that image's rows.

## SuppressionReproducesTheReferenceRows

Fixture case. At the confidence 1.8e-4 the 1542 passing scores give 300 rows, the reference's
count. The score column equals the reference's exactly, and the rows above the last score equal
the reference's rows above it, each set sorted, since rows that share the last score are
interchangeable.

## NoReferenceScorePassesTheValidationConfidence

Fixture case. With a default `NMSConfig` no row is returned for the one2many scores, as
`validation.row_count.one2many` of the manifest says.

## SuppressionDefaultsMatchTheValidator

Small. A default `NMSConfig` is checked by what it passes, not by its field values. Two 10 by 10
squares shifted by 1 have an IoU of 81 / 119, 0.68, below the default threshold 0.7, so both
survive; a third box scoring 0.001, the default confidence, is dropped. The earlier case compared
each field with a literal.

## SuppressesTheLowerOfTwoOverlappingBoxes

Small. Boxes 0 and 1 share 81 of 119, 0.68, above 0.5, so the lower score goes and boxes 0 and 2
stay.

## ListsSurvivorsByDescendingScore

Small. Four boxes that do not overlap, scored 0.1, 0.9, 0.5 and 0.5, are listed 1, 2, 3, 0:
descending score, equal scores by index.

## KeepsABoxWhoseIoUEqualsTheThreshold

Small. Two boxes with an IoU of exactly 0.5 are both kept at a threshold of 0.5. At 0.25 only
the first is kept.

## OverlappingBoxesOfOneClassCollapse

Two equal boxes of class 0, scored 0.9 and 0.8, collapse to the higher one, with the corners 0
and 10.

## OverlappingBoxesOfDifferentClassesBothSurvive

Two equal boxes of classes 0 and 1, scored 0.9 and 0.8, both survive a class-aware pass. A
class-agnostic pass keeps the first alone.

## DropsAScoreEqualToTheConfidenceThreshold

A score of 0.25 at the threshold 0.25 is dropped, and a score of 0.5 stays.

## AnAnchorContributesEveryPassingClass

Under `MultiLabel` an anchor with the class scores 0.6, 0.1 and 0.7 gives two rows: 0.7 of class
2, then 0.6 of class 0. The earlier combined case is split in two.

## AnAnchorContributesItsBestClassUnderBestClass

Under `BestClass` the same anchor gives one row, 0.7 of class 2.

## LimitsDetectionsToTheHighestScores

With `max_detections` of 2, three boxes that are apart give the rows scored 0.9 and 0.7.

## LimitsCandidatesBeforeSuppression

With `max_candidates` of 1, the same three boxes give the row scored 0.9 alone.

## RejectsAThresholdOutsideTheUnitInterval

Small. A confidence of 1.5 is refused as `InvalidDescription`, naming `[0, 1]` and 1.5.

## AnImageBelowTheThresholdYieldsNoRows

Small. Scores at or below the threshold give an empty row array.

## SuppressesEachImageFromItsOwnRows

Two images of two boxes: image 0 holds one box twice, so 0.8 falls to 0.9; image 1 holds two
boxes that are apart and keeps both, 0.6 then 0.3.

## RejectsScoresThatDoNotFillTheShape

Small. Eight boxes values and six scores for a shape of 1 image, 2 anchors and 2 classes are
refused, naming `[1, 2, 2] scores` and `8 and 6 values`.

## SuppressionKeepsTheFiniteRowMaximumBesideANaN

A first class score of NaN beside 0.9 and 0.2 does not hide the maximum: the row has the score
0.9 and the class 1. The values are copied from the input, so both are compared for equality.

## SuppressionRefusesAnExtentThatIsNotPositive

Small. A shape of 0 anchors is refused as `InvalidDescription`, naming the positive extents and
`0 anchors`.

## SuppressionRefusesADetectionLimitBelowOne

Small. A `max_detections` of 0 is refused, naming that the limits must be positive and `got 0 and
30000`.

## SuppressionRefusesACandidateLimitBelowOne

Small. A `max_candidates` of 0 is refused, naming that the limits must be positive and `got 300
and 0`.
