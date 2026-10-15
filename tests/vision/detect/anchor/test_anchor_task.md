# tests/vision/detect/anchor/test_anchor_task.cpp


Suites: `Unit_DetectAnchorTask`.

Tests `read_anchor_task` and `write_anchor_task` in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/anchor_task.hpp`. A task is
compared field by field, every float bit for bit. A section is compared by the JSON text
`write_json` gives it. The refusal cases start from a valid section of three classes, four
anchors, two heads of two anchors each and the criterion "iou", and replace one member.

## AThreeHeadTaskRoundTripsThroughItsSection

A task of 80 classes, nine anchors and three heads, with `DistanceIoU` and an exponent of
0.6f, comes back unchanged after `write_anchor_task`, `write_json`, `parse_json` and
`read_anchor_task`.

## ATwoHeadTaskRoundTripsThroughItsSection

A task of 3 classes, six anchors and two heads under the plain IoU criterion comes back
unchanged after the same four steps.

## AThreeHeadSectionRoundTripsThroughItsTask

A section of three heads over nine anchors with the criterion "distance_iou" is written back
with the same JSON text after `read_anchor_task` and `write_anchor_task`.

## ATwoHeadSectionRoundTripsThroughItsTask

A section of two heads over six anchors with the criterion "iou" is written back with the
same JSON text.

## ReadsEveryFieldOfASection

The three-head section reads as 80 classes, its nine anchors, the masks {6, 7, 8}, {3, 4, 5},
{0, 1, 2}, and the heads head_0 and head_1 with `Logit` and the scales 1.25 and 1.125 and
head_2 with `Activated` and 2.0. The criterion is `DistanceIoU` with the exponent 0.5.

## WritesTheTypeTheKeysInOrderAndEachFloatAsItsDouble

The written section has the keys `type`, `classes`, `anchors`, `heads`, `suppression` in that
order, the type `K_ANCHOR_TASK_TYPE` and 3 classes. A head has the keys `output`, `anchors`,
`coding`, `scale_xy`. The second head's scale equals `double(1.05f)` and the third anchor's
sides equal `double(0.09f)` and `double(0.14f)`.

## ReadsAndWritesEachCodingString

"logit" reads as `HeadCoding::Logit` and "activated" as `HeadCoding::Activated`, and each is
written back as the same string.

## ReadsAndWritesEachCriterionString

"iou" reads as `SuppressionCriterion::IoU` and "distance_iou", with an exponent of 0.75, as
`SuppressionCriterion::DistanceIoU`. Each is written back as the same string.

## TheExponentBelongsToTheCriterionThatReadsIt

A task under IoU with the exponent 0.25 writes a `suppression` object with the one key
`criterion`, and that section reads back with `K_DEFAULT_DISTANCE_EXPONENT`. Under
`DistanceIoU` the object has the keys `criterion`, `distance_exponent` and the exponent 0.25.

## ReadsAWholeNumberAsAScale

A `scale_xy` of 2 reads as 2.0f.

## AcceptsAnAnchorTwoHeadsShare

Heads with the anchors [1, 2] and [0, 1] read with those two masks.

## RefusesASectionThatIsNotAnObject

`read_anchor_task` fails with `InvalidDescription` and "task" for an array ("an array") and
for a null value ("null").

## RefusesAnotherTypeOrNone

The read fails with `InvalidDescription` for the type "anchor_free_detection" ("'type'",
"'anchor_free_detection'", "'anchor_detection'"), for the type 3 ("'type'",
"a whole number") and for a section without a type ("task has no 'type'").

## RefusesAnUnknownKeyAtEachLevel

The read fails with `InvalidDescription` for the key `clases` in the section ("'clases'",
"'classes'"), the key `scale` in a head ("'heads' entry 0", "'scale'", "'scale_xy'") and the
key `threshold` in `suppression` ("'suppression'", "'threshold'").

## RefusesASectionWithoutOneOfItsMembers

A section with all five members reads. Without `classes`, `anchors`, `heads` or `suppression`
it fails with `InvalidDescription` and "task has no" followed by the quoted key.

## RefusesAClassCountBelowOneOrOfAnotherKind

`classes` 0 fails with `InvalidDescription`, "'classes'", "at least 1" and "0". `classes` 3.5
fails with "'classes'" and "a whole number".

## RefusesAnAnchorThatIsNotAPairOfNumbers

The read fails with `InvalidDescription` for `anchors` given as a number ("'anchors'",
"an array"), an anchor of three values ("'anchors' entry 1", "[width, height]", "3 values"),
an anchor given as a number ("'anchors' entry 2", "[width, height]") and a height given as a
string ("'anchors' entry 0 height", "a string").

## RefusesAnAnchorSideThatIsNotPositiveAndFinite

The read fails with `InvalidDescription` for a width of 0.0 ("'anchors' entry 2",
"above zero"), a height of -0.5 ("'anchors' entry 1", "above zero"), a width of 1e-60
("'anchors' entry 0", "above zero"), a height of 1e60 ("'anchors' entry 3 height", "float32")
and an empty list ("'anchors'", "no anchor").

## RefusesNoHeadAndAHeadThatIsNotAnObject

The read fails with `InvalidDescription` for an empty `heads` ("'heads'", "no head"), for
`heads` given as an object ("'heads'", "an object") and for a second head given as the number
7 ("'heads' entry 1", "a whole number").

## RefusesAMaskIndexThatNamesNoAnchor

Over four anchors the read fails with `InvalidDescription` for the index 4
("'heads' entry 1 'anchors' entry 2", "anchor 4", "4 anchors"), the index -1
("'heads' entry 0 'anchors' entry 0", "anchor -1"), the index 1.5
("'heads' entry 0 'anchors' entry 1", "a whole number") and an empty mask
("'heads' entry 0 'anchors'", "no anchor").

## RefusesAHeadWithABadOutputCodingOrScale

The read fails with `InvalidDescription` for an empty `output` ("'heads' entry 0 'output'",
"empty string"), a head without `output` ("'heads' entry 0", "no 'output'"), the coding
"linear" ("'heads' entry 1 'coding'", "'linear'", "'logit', 'activated'"), a `scale_xy` of 0.0
("'heads' entry 0 'scale_xy'", "above zero") and a `scale_xy` given as a string
("'heads' entry 0 'scale_xy'", "a string").

## RefusesABadCriterionOrExponent

The read fails with `InvalidDescription` for the criterion "giou"
("'suppression' 'criterion'", "'giou'", "'iou', 'distance_iou'"), an exponent stated under
"iou" ("'suppression' 'distance_exponent'", "'iou'", "does not read"), "distance_iou" without
an exponent ("'suppression'", "no 'distance_exponent'") and `suppression` given as an array
("'suppression'", "an array").

## WriterRefusesAHeadOfAnotherClassCount

A task of 3 classes whose second head has 4 fails in `write_anchor_task` with
`InvalidDescription`, "'heads' entry 1", "4 classes", "'classes'" and "3".

## WriterRefusesTheValuesTheReaderRefuses

`write_anchor_task` fails with `InvalidDescription` for 0 classes ("'classes'",
"at least 1"), an anchor height of NaN ("'anchors' entry 4", "above zero"), a mask index of 6
over six anchors ("'heads' entry 0 'anchors' entry 1", "anchor 6", "6 anchors"), no head
("'heads'", "no head"), one head for two masks ("'heads'", "1 heads", "2 masks"), an empty
output ("'heads' entry 0 'output'", "empty string") and an infinite scale
("'heads' entry 1 'scale_xy'", "finite").

## WriterRefusesAnEnumeratorWithoutAString

`write_anchor_task` fails with `InvalidDescription` for a `HeadCoding` of 7
("'heads' entry 1 'coding'", "7", "'logit', 'activated'") and for a `SuppressionCriterion` of
9 ("'suppression' 'criterion'", "9", "'iou', 'distance_iou'").

## WriterRefusesAnExponentThatIsNotFiniteUnderDistanceIoU

With an infinite exponent the task writes under IoU. Under `DistanceIoU` the write fails with
`InvalidDescription`, "'suppression' 'distance_exponent'" and "finite".
