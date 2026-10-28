# tests/vision/detect: test_detections_nms.cpp, test_detections_topk.cpp, test_top_k.cpp


The top-k selection, both postprocess paths and the overlap suppression.

Suites: `Small_DetectTopK`, `Medium_DetectPostprocess`.

Tests the top-k selection, both postprocess paths and the overlap suppression.

This file needs no device. `Small_DetectTopK` covers `modules/vision/detect/include/SushiAI/vision/detect/top_k.hpp`, and
`Medium_DetectPostprocess` covers `modules/vision/detect/include/SushiAI/vision/detect/detections_topk.hpp` and
`modules/vision/detect/include/SushiAI/vision/detect/detections_nms.hpp`. The suppression path takes boxes as centre and
size, so the hand cases write them that way and expect corners. Four cases read `forward.sacp`
and report themselves skipped when the file is absent.

Two of the reference comparisons cannot compare row by row. The reference orders equal scores
in an order torch does not specify, and the fixture's scores are nearly constant.
`modules/vision/detect/README.md`, under `top_k.hpp` and "Reference rows the suppression test
reads", gives the counts.

## K_EXACT

`K_EXACT` is the tolerance 0. A postprocess copies its output values from its inputs.

## K_SUPPRESSION_CONFIDENCE

`K_SUPPRESSION_CONFIDENCE` is 1.8e-4, the `suppression.arguments.conf_thres` of `forward.sacp`
in `tests/vision/registration/fixtures/yolo26/MANIFEST.json`.

## Row

`Row` is an array of six floats, one detection.

## sorted_rows_above

`sorted_rows_above` returns the rows whose score exceeds a floor, sorted by their six values.
Two detection arrays that hold the same rows in different orders compare equal through it.

## is_anchor_class_pair

`is_anchor_class_pair` reports whether a row's box is some anchor's box and its score is that
anchor's score in the row's class.

## make_nms_config

`make_nms_config` returns an `NMSConfig` with the given confidence threshold, IoU threshold,
label mode and class mode, and the default limits.

## suppress_one_image

`suppress_one_image` runs `select_nms_detections` on one image of `[A, 4]` boxes and returns
that image's rows.

## OrdersByValueThenIndex

`select_top_k` of the values 0.5, 0.9, 0.5, 0.1, 0.9 with k = 4 returns the indices 1, 4, 0, 2:
larger values first, equal values by ascending index. Asking for 6 of 5 values throws with a
message containing "6" and "5 values".

## SelectsTheHighestScoringAnchorClassPairs

Three anchors with two classes and a limit of 2 give the row of anchor 1 with score 0.9 and
class 0, then the row of anchor 0 with score 0.7 and class 1.

## OneAnchorCanYieldTwoDetections

Anchor 1 scores 0.8 and 0.7 in its two classes, both above anchor 0's 0.6. Both output rows
carry anchor 1's box, with classes 0 and 1.

## LimitsRowsToTheAnchorCount

`count_top_detections` returns 3 for three anchors and a limit of 300, and 300 for 8400 anchors.

## SelectsEachImageFromItsOwnRows

A batch of two images with two anchors each and a limit of 1 gives one row per image: anchor 1
of image 0 at 0.6 in class 0, and anchor 1 of image 1 at 0.9 in class 1.

## RejectsADetectionArrayOfTheWrongSize

An output of 6 values where the shape needs `[1, 2, 6]` throws with a message containing
"[1, 2, 6] detections" and "6 values".

## ReproducesTheReferenceDetections

`select_top_detections` on `one2one.decoded.boxes` and `one2one.decoded.scores` with
`K_MAX_DETECTIONS` is compared with `one2one.detections` in three ways. The score column and
the class column are equal row by row. The rows above the last score are equal as a set,
through `sorted_rows_above`. Every output row passes `is_anchor_class_pair`. The case is
skipped when `forward.sacp` is absent.

## TheConfidenceFilterDropsAScoreAtTheThreshold

`keep_confident_detections` at 0.25 keeps the rows at 0.5 and 0.26, in their order, and drops
the rows at 0.25 and 0.1.

## TheConfidenceFilterKeepsEachImagesRows

Four rows split between two images at `K_VALIDATION_CONFIDENCE`: image 0 keeps its row at 0.9
and image 1 keeps none.

## TheConfidenceFilterRejectsRowsTheBatchDoesNotDivide

Three rows for a batch of 2 throw with a message containing "[2, k, 6] detections" and "18
values". A batch of 0 throws with "positive batch" and "got 0".

## TheConfidenceFilterKeepsTheRowsTheReferenceValidatorKeeps

At `K_VALIDATION_CONFIDENCE` the filter keeps none of the 300 rows of `one2one.detections`,
which is `validation.row_count.one2one` of the manifest. At threshold 0 it keeps all of them
unchanged, since every reference score is a sigmoid above zero. The case is skipped when
`forward.sacp` is absent.

## SuppressionReproducesTheReferenceRows

`select_nms_detections` at `K_SUPPRESSION_CONFIDENCE` on the one-to-many decode returns as many
values as `one2many.suppressed.detections`. The score column is equal row by row and the rows
above the last score are equal as a set. The case is skipped when `forward.sacp` is absent.

## NoReferenceScorePassesTheValidationConfidence

With a default `NMSConfig` the one-to-many decode yields no row, which is
`validation.row_count.one2many` of the manifest. The case is skipped when `forward.sacp` is
absent.

## SuppressionDefaultsMatchTheValidator

A default `NMSConfig` has confidence 0.001, IoU 0.7, `LabelMode::MultiLabel`,
`ClassMode::ClassAware`, 300 detections, 30000 candidates and a class offset of 7680.

## SuppressesTheLowerOfTwoOverlappingBoxes

Boxes 0 and 1 share 81 of 119 px, an IoU of 0.68, and box 2 is apart. `suppress_overlaps` at 0.5
keeps the indices 0 and 2.

## ListsSurvivorsByDescendingScore

Four boxes that do not overlap, with scores 0.1, 0.9, 0.5 and 0.5, are kept in the order
1, 2, 3, 0.

## KeepsABoxWhoseIoUEqualsTheThreshold

Two boxes share 2 of 4 px, an IoU of exactly 0.5 in float32. At threshold 0.5 both are kept; at
0.25 only the first is.

## OverlappingBoxesOfOneClassCollapse

Two anchors hold the same box and score 0.9 and 0.8 in class 0. One row remains, with corners
(0, 0, 10, 10), score 0.9 and class 0.

## OverlappingBoxesOfDifferentClassesBothSurvive

The same two boxes score in classes 0 and 1. `ClassMode::ClassAware` keeps both rows, and
`ClassMode::ClassAgnostic` keeps the one at 0.9.

## DropsAScoreEqualToTheConfidenceThreshold

An anchor at 0.25 with the threshold at 0.25 yields no row. The anchor at 0.5 yields the one
row.

## AnAnchorContributesEveryPassingClassOrItsBest

One anchor scores 0.6, 0.1 and 0.7 in three classes. `LabelMode::MultiLabel` yields two rows,
class 2 at 0.7 then class 0 at 0.6. `LabelMode::BestClass` yields the row of class 2 alone.

## LimitsDetectionsToTheHighestScores

Three boxes that are apart score 0.5, 0.9 and 0.7. With `max_detections` 2 the rows at 0.9 and
0.7 remain.

## LimitsCandidatesBeforeSuppression

With `max_candidates` 1 the same input yields the row at 0.9 alone.

## RejectsAThresholdOutsideTheUnitInterval

A confidence threshold of 1.5 returns `InvalidDescription` with a message containing "[0, 1]" and "1.5".

## AnImageBelowTheThresholdYieldsNoRows

Scores of 0.25, 0.1, 0.2 and 0 at the threshold 0.25 yield an empty row vector.

## SuppressesEachImageFromItsOwnRows

A batch of two images with two anchors each. Image 0 holds one box twice, so the row at 0.9
remains. Image 1 holds two boxes that are apart, and its rows are 0.6 then 0.3.

## RejectsScoresThatDoNotFillTheShape

Six scores where the shape `[1, 2, 2]` needs four fail with a message containing "[1, 2, 2]
scores" and "8 and 6 values".
