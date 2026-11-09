# tests/vision/detect/anchor/test_anchor_task.cpp


Suites: `Small_DetectAnchorTask` and its row suites `Small_DetectAnchorTaskRoundTrip`,
`Small_DetectAnchorSectionRoundTrip`, `Small_DetectAnchorTaskCoding`,
`Small_DetectAnchorTaskCriterion` and `Small_DetectAnchorTaskMissing`.

Tests `read_anchor_task` and `write_anchor_task` in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/anchor_task.hpp`. A task is
compared field by field, every float bit for bit. A section is compared by the JSON text
`write_json` gives it. The refusal cases start from a valid section of three classes, four
anchors, two heads of two anchors each and the criterion "iou", and replace one member.

## ATaskRoundTripsThroughItsSection

One row per task. `ThreeHeadsOverNineAnchors`, 80 classes, nine anchors and three heads under
`DistanceIoU` with an exponent of 0.6f, and `TwoHeadsOverSixAnchors`, 3 classes, six anchors and
two heads under the plain IoU criterion, each come back unchanged after `write_anchor_task`,
`write_json`, `parse_json` and `read_anchor_task`.

## ASectionRoundTripsThroughItsTask

One row per section. `ThreeHeads`, three heads over nine anchors with the criterion
"distance_iou", and `TwoHeads`, two heads over six anchors with "iou", are each written back
with the same JSON text after `read_anchor_task` and `write_anchor_task`.

## ReadsEveryFieldOfASection

The three-head section reads as 80 classes, its nine anchors, the masks {6, 7, 8}, {3, 4, 5},
{0, 1, 2}, and the heads head_0 and head_1 with `Logit` and the scales 1.25 and 1.125 and
head_2 with `Activated` and 2.0. The criterion is `DistanceIoU` with the exponent 0.5.

## WritesTheTypeAndTheClassCount

The written section states the type `K_ANCHOR_TASK_TYPE` and 3 classes.

## WritesTheKeysInOrder

The written section has the keys `type`, `classes`, `anchors`, `heads`, `suppression` in that
order, and a head has the keys `output`, `anchors`, `coding`, `scale_xy`.

## WritesEachFloatAsItsDouble

The second head's scale equals `double(1.05f)` and the third anchor's sides equal
`double(0.09f)` and `double(0.14f)`.

## ReadsAndWritesSquaredSigmoidAsTheSizeOfALogitHead

"size": "squared_sigmoid" on a logit head reads as `SizeForm::SquaredSigmoid` and is written
back as the same string.

## AnExponentialSizeIsAnAbsentMember

A head without "size" reads as `SizeForm::Exponential` and is written without the key.

## ASizeOnAnActivatedHeadIsRefused

"size" on an activated head fails naming "'size'" and "logit head".

## AnUnknownSizeIsRefused

The form "cubic" fails naming it and the known forms.

## AStringOfCodingIsReadAndWrittenAsItsEnumerator

One row per string. "logit" reads as `HeadCoding::Logit` and "activated" as
`HeadCoding::Activated`, and each is written back as the same string.

## AStringOfCriterionIsReadAndWrittenAsItsEnumerator

One row per string. "iou" reads as `SuppressionCriterion::IoU` and "distance_iou", with an
exponent of 0.75, as `SuppressionCriterion::DistanceIoU`. Each is written back as the same
string.

## TheExponentIsNotWrittenUnderIoU

A task under IoU with the exponent 0.25 writes a `suppression` object with the one key
`criterion`.

## AnExponentTheSectionLeavesOutReadsAsTheDefault

The section of that task reads back with `K_DEFAULT_DISTANCE_EXPONENT`.

## TheExponentIsWrittenUnderDistanceIoU

Under `DistanceIoU` the object has the keys `criterion`, `distance_exponent` and the exponent
0.25.

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

## RefusesAnUnknownKeyOfTheSection

The read fails with `InvalidDescription` for the key `clases` in the section ("'clases'",
"'classes'").

## RefusesAnUnknownKeyOfAHead

The read fails with `InvalidDescription` for the key `scale` in a head ("'heads' entry 0",
"'scale'", "'scale_xy'").

## RefusesAnUnknownKeyOfTheSuppression

The read fails with `InvalidDescription` for the key `threshold` in `suppression`
("'suppression'", "'threshold'").

## AcceptsASectionOfAllFiveMembers

A section with all five members reads.

## ASectionWithoutAMemberIsRefused

One row per member: without `type`, `classes`, `anchors`, `heads` or `suppression` the read
fails with `InvalidDescription` and "task has no" followed by the quoted key.

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
("'anchors' entry 0", "above zero") and a height of 1e60 ("'anchors' entry 3 height",
"float32").

## RefusesAListOfNoAnchor

An empty `anchors` list fails with `InvalidDescription`, "'anchors'" and "no anchor".

## RefusesNoHead

An empty `heads` fails with `InvalidDescription`, "'heads'" and "no head".

## RefusesAHeadThatIsNotAnObject

The read fails with `InvalidDescription` for `heads` given as an object ("'heads'",
"an object") and for a second head given as the number 7 ("'heads' entry 1",
"a whole number").

## RefusesAMaskIndexThatNamesNoAnchor

Over four anchors the read fails with `InvalidDescription` for the index 4
("'heads' entry 1 'anchors' entry 2", "anchor 4", "4 anchors"), the index -1
("'heads' entry 0 'anchors' entry 0", "anchor -1"), the index 1.5
("'heads' entry 0 'anchors' entry 1", "a whole number") and an empty mask
("'heads' entry 0 'anchors'", "no anchor").

## RefusesAHeadWithoutAnOutputName

The read fails with `InvalidDescription` for an empty `output` ("'heads' entry 0 'output'",
"empty string") and for a head without `output` ("'heads' entry 0", "no 'output'").

## RefusesAnUnknownCoding

The coding "linear" fails ("'heads' entry 1 'coding'", "'linear'", "'logit', 'activated'").

## RefusesAScaleThatIsNotAPositiveNumber

A `scale_xy` of 0.0 ("'heads' entry 0 'scale_xy'", "above zero") and a `scale_xy` given as a
string ("'heads' entry 0 'scale_xy'", "a string") each fail with `InvalidDescription`.

## RefusesABadCriterion

The read fails with `InvalidDescription` for the criterion "giou" ("'suppression' 'criterion'",
"'giou'", "'iou', 'distance_iou'") and for `suppression` given as an array ("'suppression'",
"an array").

## RefusesAnExponentUnderTheWrongCriterion

The read fails with `InvalidDescription` for an exponent stated under "iou"
("'suppression' 'distance_exponent'", "'iou'", "does not read") and for "distance_iou" without
an exponent ("'suppression'", "no 'distance_exponent'").

## WriterRefusesAHeadOfAnotherClassCount

A task of 3 classes whose second head has 4 fails in `write_anchor_task` with
`InvalidDescription`, "'heads' entry 1", "4 classes", "'classes'" and "3".

## WriterRefusesAClassCountBelowOne

0 classes fail with `InvalidDescription`, "'classes'" and "at least 1".

## WriterRefusesAnAnchorSideThatIsNotANumber

An anchor height of NaN fails ("'anchors' entry 4", "above zero").

## WriterRefusesAMaskIndexPastTheAnchors

A mask index of 6 over six anchors fails ("'heads' entry 0 'anchors' entry 1", "anchor 6",
"6 anchors").

## WriterRefusesATaskWithoutAHead

No head fails ("'heads'", "no head").

## WriterRefusesFewerHeadsThanMasks

One head for two masks fails ("'heads'", "1 heads", "2 masks").

## WriterRefusesAHeadWithoutAnOutputName

An empty output fails ("'heads' entry 0 'output'", "empty string").

## WriterRefusesAScaleThatIsNotFinite

An infinite scale fails ("'heads' entry 1 'scale_xy'", "finite").

## WriterRefusesAnEnumeratorWithoutAString

`write_anchor_task` fails with `InvalidDescription` for a `HeadCoding` of 7
("'heads' entry 1 'coding'", "7", "'logit', 'activated'") and for a `SuppressionCriterion` of
9 ("'suppression' 'criterion'", "9", "'iou', 'distance_iou'").

## WriterAcceptsAnExponentThatIsNotFiniteUnderIoU

With an infinite exponent the task writes under IoU.

## WriterRefusesAnExponentThatIsNotFiniteUnderDistanceIoU

Under `DistanceIoU` the same task fails with `InvalidDescription`,
"'suppression' 'distance_exponent'" and "finite".
