# tests/vision/detect/anchor/test_head_training_section.cpp


Suites: `Small_DetectAnchorTrainingSection`.

Tests `read_head_training_section` and `write_head_training_section` in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/head_training_section.hpp`.
Options are compared field by field, every float bit for bit. The full section states the
type "anchor" and all nine option keys.

## TheDefaultsRoundTripThroughTheirSection

A default `HeadTrainingOptions` comes back unchanged after `write_head_training_section`,
`write_json`, `parse_json` and `read_head_training_section`.

## OptionsWithNoDefaultRoundTripThroughTheirSection

Options with the thresholds 0.213f and 0.7f, `CompleteIoU`, `DarknetCompatible`, `BestIoU`,
the weights 0.07f and 0.4f, a `max_delta` of 5.0f and a class gate of 0.3f come back unchanged
after the same four steps.

## AFullSectionRoundTripsThroughItsOptions

The full section, read and written back, gives the same JSON text.

## ReadsEveryFieldOfASection

The full section reads as the thresholds 0.25 and 0.5, `BoxLoss::CompleteIoU`,
`BoxGradient::DarknetCompatible`, `ObjectnessTarget::BestIoU`, a box weight of 0.0625, an
object weight of 1.5, a `max_delta` of 5.0 and a class gate of 0.125.

## AKeyLeftOutKeepsTheDefaultOfItsField

An empty section and a section with the type alone both read as the default options. A
section with `max_delta` 2.5 alone reads as the defaults with that one field changed.

## WritesTheTypeEveryKeyInOrderAndAFloatAsItsDouble

The written section has the keys `type`, `match_threshold`, `ignore_threshold`, `box_loss`,
`box_gradient`, `objectness_target`, `box_weight`, `object_weight`, `max_delta`, `class_gate`,
`box_gain`, `class_gain`, `objectness_gain` in that order. Its type is `K_ANCHOR_OBJECTIVE_TYPE` and its `box_loss` "complete_iou". The
match threshold equals `double(0.213f)` and the box weight `double(0.07f)`.

## ReadsAndWritesEachEnumeratorString

The writer gives "squared_error" and "complete_iou" for `BoxLoss`, "analytic" and
"darknet_compatible" for `BoxGradient`, and "one" and "best_iou" for `ObjectnessTarget`. A
section of "squared_error", "analytic" and "one" reads as `SquaredError`, `Analytic` and
`One`.

## ReadsAWholeNumberAsAWeight

An `object_weight` of 2 reads as 2.0f.

## RefusesASectionThatIsNotAnObject

An array and a null value both fail with `InvalidDescription`, "objective" and "an object".

## RefusesAnotherType

The type "cross_entropy" fails with `InvalidDescription`, "objective 'type'",
"'cross_entropy'" and "'anchor'". The type 3 fails with "objective 'type'" and "a string".

## RefusesAnUnknownKey

The key `iou_normalizer` fails with `InvalidDescription`, "objective", "'iou_normalizer'",
"'type', 'match_threshold'" and "'class_gate'". The key `thresholds` fails with "objective"
and "'thresholds'".

## RefusesAValueOfAnotherKind

The read fails with `InvalidDescription` for a `box_weight` given as a string
("objective 'box_weight'", "a number"), an `ignore_threshold` given as an array
("objective 'ignore_threshold'", "a number") and a `box_loss` of 1 ("objective 'box_loss'",
"a string").

## RefusesAStringNoEnumeratorHas

The read fails with `InvalidDescription` for the `box_loss` "ciou" ("objective 'box_loss'",
"'ciou'", "'squared_error', 'complete_iou'"), the `box_gradient` "exact"
("objective 'box_gradient'", "'exact'", "'analytic', 'darknet_compatible'") and the
`objectness_target` "iou" ("objective 'objectness_target'", "'iou'", "'one', 'best_iou'").

## RefusesANumberFloat32CannotHold

A `max_delta` of 1e39 fails with `InvalidDescription`, "objective 'max_delta'" and "float32".

## RefusesTheValuesTheOptionsCheckRefuses

The read fails with `InvalidDescription` and "objective" for a `match_threshold` of 0.0
("match threshold", "(0, 1]"), an `ignore_threshold` of 1.5 ("ignore threshold", "1.5"), a
`class_gate` of -0.25 ("class gate", "-0.25"), a `box_weight` of -1.0 ("box weight",
"-1"), a `box_gain` of -1.0 ("box gain", "-1"), a `class_gain` of -2.0 ("class gain", "-2")
and an `objectness_gain` of -3.0 ("objectness gain", "-3").

## WriterRefusesTheValuesTheOptionsCheckRefuses

`write_head_training_section` fails with `InvalidDescription` and "objective" for an object
weight of NaN ("object weight"), an infinite `max_delta` ("max_delta") and a `BoxLoss` of 7
("BoxLoss", "7").
