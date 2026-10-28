# tests/vision/detect/free/test_assigner.cpp


Suites: `Medium_DetectAssigner`, `Medium_DetectGroundTruth`, `Small_DetectTargetLeaves`.

Tests the task-aligned assigner, its presets, the padded ground truth and the target leaves.

This file needs no device. Its three suites are `Medium_DetectAssigner`, `Medium_DetectGroundTruth`
and `Small_DetectTargetLeaves`, over `modules/vision/detect/include/SushiAI/vision/detect/free/assigner.hpp`,
`modules/vision/detect/include/SushiAI/vision/detect/free/assigner_config.hpp`, `modules/vision/detect/include/SushiAI/vision/detect/ground_truth.hpp` and
`modules/vision/detect/include/SushiAI/vision/detect/free/target_leaves.hpp`. Most cases build one image on a 4 by 4 grid by
hand, with `alpha` and `beta` both 1, so the metric is the class score times the overlap and
each expected score is a fraction a comment derives. Three cases read `loss.sacp` and report
themselves skipped when the file is absent. `modules/vision/detect/README.md`, under
"Reference tensors the assigner tests read", lists the tensors.

## Tolerances

`K_EXACT` is 0, for copied values and multiples of 0.5. `K_SCORE_TOLERANCE` is 1e-6, eight
float32 steps at 1, for a target score computed by hand. `K_REFERENCE_SCORE_TOLERANCE` is 1e-5
for a reference target score, where `std::pow` and `std::atan` may round apart from torch.
`K_REFERENCE_BOX_TOLERANCE` is 1.3e-4, two float32 steps at 640, for a reference box in pixels.

## The hand-built grid

`K_GRID_SIDE` is 4 cells, `K_GRID_ANCHORS` is 16, `K_GRID_STRIDE` is 8 px and `K_CLASSES` is 2.
`K_FAR_BOX` is the box (100, 100, 101, 101), which no hand-built ground truth touches.

## Scene

`Scene` holds one image on that grid: anchor `a` sits in column `a % 4` and row `a / 4`, with
pixel centres 4, 12, 20 and 28. Its constructor gives every anchor one score for both classes
and the prediction `K_FAR_BOX`. `add_truth` appends a valid ground-truth box, `predict` sets one
anchor's predicted box, and `inputs` returns the scene as an `AssignerInputs`.

## make_hand_config

`make_hand_config` returns an `AssignerConfig` with the given `topk` and `topk2`, two classes,
`alpha` 1, `beta` 1 and `stride_val` 8.

## foreground_anchors

`foreground_anchors` returns the rows an `Assignment` marks as foreground, in ascending order.

## make_crowded_scene

`make_crowded_scene` returns a scene with one box of class 1 over the whole 32 px image. Every
anchor predicts that box and scores 0.1, except anchors 5, 10 and 2, which score 0.9, 0.6 and
0.3 in class 1.

## make_contested_scene

`make_contested_scene` returns a scene with two boxes, of classes 0 and 1, that both contain
anchor 5. Anchors 0, 1 and 4 predict the first box, anchors 6, 9 and 10 the second, and anchor 5
predicts the box the caller passes.

## box_row

`box_row` returns the four values of one row of a `[rows, 4]` array.

## make_idle_assignment

`make_idle_assignment` returns an `Assignment` of the given batch and anchor count with no
foreground and every array at zero.

## set_foreground

`set_foreground` marks one row of an `Assignment` as foreground and gives it a pixel box and two
class scores.

## Reference constants

`K_REFERENCE_LEVELS` holds the manifest's three levels for a 640 px input,
`K_REFERENCE_IMAGE_SIDE` is 640, and `K_SMALL_BOX_ROW` is 3, the ground-truth row of the
fixture that is smaller than a stride.

## pad_reference_truth

`pad_reference_truth` reads `batch.batch_idx`, `batch.cls` and `batch.bboxes` from `loss.sacp`
and pads them with `pad_ground_truth` for one 640 px image. It returns an empty string, or the
reason a tensor could not be read.

## count_holders

`count_holders` counts the foreground anchors of a mask whose target index is a given
ground-truth row. It is a template over the mask's element type, so it takes the assigner's
mask and the reference's.

## expect_reference_assignment

`expect_reference_assignment` runs one branch's assigner on the reference's scores, boxes and
anchor points and compares all five outputs: the mask, the indices and the labels for equality,
the boxes within `K_REFERENCE_BOX_TOLERANCE` and the scores within
`K_REFERENCE_SCORE_TOLERANCE`. It then compares the foreground count and the count of anchors
that hold box 3 with the two numbers the caller passes, and the second with the same count
taken from the reference. Equality of the mask and the indices relies on the fixture having no
tied metrics. The helper skips when `loss.sacp` is absent.

## OneToManyPresetMatchesTheReferenceSettings

`one_to_many_assigner_config(80, {8, 16, 32})` has `topk` 10, `topk2` 10, 80 classes, `alpha`
0.5, `beta` 6, `stride_val` 16 and `eps` 1e-9. These are the `assigner.one2many` values of
`loss.sacp` in `tests/vision/registration/fixtures/yolo26/MANIFEST.json`.

## OneToOnePresetMatchesTheReferenceSettings

`one_to_one_assigner_config(80, {8, 16, 32})` has `topk` 7, `topk2` 1, `alpha` 0.5, `beta` 6 and
`stride_val` 16, the `assigner.one2one` values of the manifest.

## PresetTakesStrideValFromTheSecondLevel

`stride_val` is 12 for the strides 4 and 12, 4 for the one stride 4, and 16 for an empty stride
list.

## ConfigDefaultsMatchTheReferenceConstructor

A default `AssignerConfig` has `topk` 13, `topk2` 13, 80 classes, `alpha` 1, `beta` 6,
`stride_val` 16 and `eps` 1e-9, the arguments of the reference's `TaskAlignedAssigner.__init__`.

## RejectsAConfigWithoutCandidatesOrWithANegativeMargin

Constructing a `TaskAlignedAssigner` returns `InvalidDescription` for `topk` 0, with a message containing "positive
topk" and "got 0, 3 and 2", for `stride_val` -1, with "stride_val", "at least 0" and "-1", and
for `eps` -1, with "eps of at least 0" and "-1".

## TheMetricRaisesTheScoreToAlphaAndTheOverlapToBeta

The case uses the one-to-many preset, so `alpha` is 0.5 and `beta` is 6. Anchor 5 predicts the
box exactly with score 0.25, so its metric is sqrt(0.25) times 1, which is 0.5. Anchor 10
predicts a 24 px box with score 0.81; its overlap is 576 / 1024 - 32 / 2048, which is 35 / 64,
and its metric is 0.9 times (35 / 64)^6, about 0.024. With `topk` 1 the foreground is anchor 5
alone, and with `topk` 2 it is anchors 5 and 10. The target score of anchor 5 is 1 and that of
anchor 10 is its metric over 0.5. An assigner that used the scores or the overlaps unraised
gives another second score.

## NormalisesByTheLargestOverlapOfTheBox

Anchors 0 and 1 both predict a box offset by 8 px from a 16 px truth, with scores 0.8 and 0.4.
The overlap is 64 / 448 - 128 / 1152, which is 2 / 63. The target scores are that overlap and
half of it: each metric over the largest metric, times the largest overlap of the box.

## APredictionApartFromItsBoxHasNoOverlap

Anchors 0 and 1 score 0.9 and predict `K_FAR_BOX`, whose complete IoU with the box is negative
and is clamped to 0. With `topk` 2 the foreground is anchors 4 and 5, which predict the box.
With `topk` 3 the third place goes to anchor 0, the first candidate of metric zero, and its
target score is 0.

## AContestedAnchorOfEqualOverlapsGoesToTheFirstBox

Two ground-truth rows hold the same box with classes 0 and 1, and anchor 5 predicts it. Anchor 5
is the one foreground anchor, its target index and label are 0, its score in class 0 is 1 and
its score in class 1 is 0.

## AContestedAnchorCanGoToABoxThatDidNotKeepIt

With `topk` 1, boxes 0 and 1 both keep anchor 5 and box 2 keeps anchor 2. Anchor 5 predicts box
2 exactly, so its largest overlap is with box 2 and it goes there, with target index 2 and
label 1. Its score in class 1 is 0.1 / 0.9 and its score in class 0 is 0. The case pins that
the maximum overlap is taken over every box, including one that did not keep the anchor.

## AnImageWithoutGroundTruthGetsNoForeground

A scene with no ground truth has no foreground anchor. Every target label is 2, the class
count, and every index, box and score is 0.

## APaddingRowAssignsNothing

The crowded scene's one box is marked invalid. No anchor is foreground and every target score
is 0, although the box contains all sixteen anchors.

## KeepsTheTopKCandidatesWhenABoxHoldsMore

In the crowded scene every overlap is 1, so the metric is the class score. With `topk` 3 the
foreground is anchors 2, 5 and 10, with target scores 0.3 / 0.9, 1 and 0.6 / 0.9 in class 1 and
0 in class 0. Every target label is 1, the label of ground-truth row 0, on and off the
foreground.

## TheSecondTopKKeepsTheBestAnchorOfABox

With `topk` 3 and `topk2` 1 the crowded scene keeps anchor 5 alone, with target score 1.

## AContestedAnchorGoesToTheHigherOverlap

Anchor 5 predicts the second box of the contested scene. The foreground is anchors 0, 1, 4, 5,
6, 9 and 10, and the target index and label of anchor 5 are 1. Its score is 1 in class 1 and 0
in class 0, and its target box starts at x = 8.

## AContestedAnchorCanGoToTheFirstBox

The same scene with anchor 5 predicting the first box gives anchor 5 the target index 0, score
1 in class 0 and 0 in class 1.

## ASmallBoxIsEnlargedToStrideVal

A 2 px box centred at 14 contains no anchor centre. With `stride_val` 16 it is tested as the
span 6 to 22, which holds the centres 12 and 20 on both axes, so anchors 5, 6, 9 and 10 are
foreground. With `stride_val` 1 there is no foreground. The target box of anchor 5 still starts
at 13: the enlargement applies to the candidate test and not to the target.

## AssignsEachImageFromItsOwnRows

A batch of two holds an image with one invalid row and the crowded scene. The foreground rows
are 18, 21 and 26, which are anchors 2, 5 and 10 of image 1. The target label of row 0 is 0 and
that of row 16 is 1.

## RejectsTooFewAnchorsAndAnUnknownClass

`assign` with `topk` 17 on sixteen anchors returns `InvalidDescription` with a message containing "16 anchors" and
"topk 17". A valid box of class 2 with two classes returns `InvalidDescription` with "class 2 of 2".

## PadsLabelRowsPerImage

Three label rows, two for image 0 and one for image 2, are padded to a batch of three 64 by 32
images with `max_boxes` 2. The boxes are converted from normalised centre and size to pixel
corners, the labels are 3, 1, 0, 0, 7, 0 and the valid flags are 1, 1, 0, 0, 1, 0.

## MarksARowWithoutExtentInvalid

A row whose four values are 0 is padded in with its label 4 and the valid flag 0. The row after
it is valid.

## PadsNoRowsToABatchWithoutBoxes

With no label rows and a batch of 2, `max_boxes` is 0 and the labels, boxes and flags are empty.

## RejectsARowOutsideTheBatch

A row naming image 3 in a batch of 3 returns `InvalidDescription` with a message containing "row 0 names image 3" and
"batch of 3".

## BuildsTheLossLeavesOfAnAssignment

The crowded scene is assigned with `topk` 3 and its leaves are built for an image 64 px wide and
32 px high. The three weights are 1, 2/3 and 1/3, so `target_scores_sum` is 2 and the
normalised weights of anchors 5 and 10 are 1/2 and 1/3. Row 5 has a box mask of four ones, the
target box (0, 0, 4, 4) in grid units, the target distances (1.5, 1.5, 2.5, 2.5) and an
`l1_scale` of 8 / 64 * 0.25 * 0.5 across and 8 / 32 * 0.25 * 0.5 down. The unequal width and
height separate the two `l1_scale` values.

## ReadsEachRowsAnchorAcrossImagesAndLevels

Two levels give five anchors, and a hand-built assignment for two images marks rows 1 and 9.
Row 1 is anchor 1 of image 0, at point (1.5, 0.5) on stride 8. Row 9 is anchor 4 of image 1, at
point (0.5, 0.5) on stride 16. Each row's target box, distances and `l1_scale` are computed
from its own point and stride, and `target_scores_sum` is 1.5. Row 4, the same anchor as row 9
in image 0, has a box mask of zeros.

## AnUnassignedRowCarriesTheSafeBoxAndNoWeight

Row 0 of the crowded scene is off the foreground. Its target box is (0, 0, 1, 1), and its box
mask, `l1_scale`, target distances, weight and normalised weight are 0. `safe_boxes` holds
(0, 0, 1, 1) in row 5 as well, which is a foreground row.

## TheScoreSumIsAtLeastOne

With no ground truth, `target_scores_sum` is 1.

## PadsTheReferenceLabelsToItsTargetBoxes

The reference labels pad to four valid rows. Gathering the padded boxes and labels by the
reference's `one2many.assignment.target_indices` gives its `target_boxes` within
`K_REFERENCE_BOX_TOLERANCE` and its `target_labels` exactly. The case is skipped when
`loss.sacp` is absent.

## ReproducesTheReferenceOneToManyAssignment

`expect_reference_assignment` runs the one-to-many preset and expects 31 foreground anchors,
the manifest's `foreground_anchor_count.one2many`, of which 4 hold box 3. The case is skipped
when `loss.sacp` is absent.

## ReproducesTheReferenceOneToOneAssignment

The same helper runs the one-to-one preset and expects 4 foreground anchors, the manifest's
`foreground_anchor_count.one2one`, of which 1 holds box 3. The case is skipped when `loss.sacp`
is absent.
