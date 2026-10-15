# tests/vision/detect/test_average_precision.cpp


Suites: `Unit_AveragePrecision`.

Tests detection matching, the 101-point average precision and the validation table.

This file needs no device and reads no fixture. Every case in `Unit_AveragePrecision` builds
detection rows and ground-truth boxes by hand and calls `match_detections`,
`integrate_precision` or the `AveragePrecision` accumulator of
`modules/vision/detect/include/SushiAI/vision/detect/average_precision.hpp`. A comment in each case derives the expected
number from the recall and precision the detections produce.

## K_TOLERANCE

`K_TOLERANCE` is 1e-12, the absolute tolerance of a metric, about 4500 float64 steps at 1.

## K_FULL_AP

`K_FULL_AP` is 0.995, the area under an envelope that is 1 up to recall 1. The integration
samples 101 recall points and the envelope is 0 at the last one, so the area is
0.99 + 0.01 / 2 and not 1.

## The boxes

`K_BOX_A` is the 8 by 8 box at the origin. `K_BOX_B` and `K_BOX_FAR` are 8 by 8 boxes that
touch neither it nor each other. `K_BOX_LOOSE` is the 8 by 5 box whose IoU with `K_BOX_A` is
40 / 64, which is 0.625, and `K_BOX_NEAR` is the 8 by 7 box whose IoU with it is 56 / 64, which
is 0.875. `K_NO_VALUES` and `K_NO_CLASSES` are empty arrays.

## make_row

`make_row` returns one detection row: a box, then a score and a class.

## join

`join` lays several arrays end to end.

## match

`match` calls `match_detections` for one image and returns the flags. It sizes the array from
`K_DETECTION_VALUES` and fills it with 7 first, so a flag the function does not write is
neither 0 nor 1.

## add_image

The file-local `add_image` passes three vectors to `AveragePrecision::add_image`.

## summarize_image

`summarize_image` returns the table of an accumulator that holds one image.

## HoldsTheTenReferenceThresholds

`K_IOU_THRESHOLDS` has ten entries, and each equals a hexadecimal float literal that carries
the float32 bits of `torch.linspace(0.5, 0.95, 10)`. The comparison is exact, so a threshold
that rounds to a neighbouring float32 fails it.

## IntegratesAnEmptyCurveToOneHalf

`integrate_precision` of two empty curves is 0.5. With no points the envelope is the line from
1 at recall 0 to 0 at recall 1.

## IntegratesUnderThePrecisionEnvelope

Recall 0.5, 0.5, 1 with precision 1, 0.5, 2/3 integrates to
0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3. The envelope lifts the dip to 0.5 up to the 2/3 that
follows it at higher recall.

## RejectsCurvesOfUnequalLength

Two recall values with one precision value throw with a message containing "1 and 2".

## PerfectDetectorScoresTheInterpolationCeiling

Two images each hold one box, of classes 0 and 1, and one detection that equals it. Both rows
have precision 1, recall 1 and `K_FULL_AP` in both AP columns, and the four means are the same
numbers.

## NoDetectionsScoreZero

One box of class 0 and no detection give one row, with label 0 and one instance. Precision,
recall and both AP columns are 0 in the row and in the means.

## FalsePositiveAboveTheTruePositiveHalvesPrecision

A miss at score 0.9 ranks above a hit at 0.6. Recall is 0 then 1 and precision 0 then 0.5, so
the AP is 0.99 * 0.5 + 0.01 * 0.25, which is 0.4975, at every threshold. The mean F1 is flat
below confidence 0.6, so the first confidence point is read: precision 0.5 and recall 1.

## ReadsPrecisionAndRecallAtTheBestF1Confidence

A hit at 0.9 ranks above a miss at 0.3. F1 rises from 2/3 at confidence 0.3 to 1 at 0.9 and is 0
above it, so the smoothed curve is largest at point 849 of 1000. Precision there is
1 - 0.5 * (0.9 - 849 / 999) / (0.9 - 0.3), and the case also compares it with the literal
0.9582082255854392. Recall is 1. The row and the means hold the same values. Reading the first
confidence point gives precision 0.5.

## TwoDetectionsOnOneBoxCountOneMatch

Two detections equal `K_BOX_A` and a third equals `K_BOX_B`, all of class 0. The flags are 1
for the first, 0 for the second and 1 for the third at all ten thresholds. The AP at IoU 0.5 is
0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3, precision is 2/3 and recall is 1. Three matches give
0.995.

## ClassWithoutGroundTruthIsLeftOutOfTheMean

A detection of class 3 has no ground truth of its class. The table has one row, for class 0,
and the means are `K_FULL_AP`, 1 and 1. A row for class 3 halves every mean.

## ClassWithoutDetectionsCountsAsZeroInTheMean

Classes 0 and 1 both have a box and only class 0 has a detection. The row of class 1 has AP 0,
and the means are 0.4975 for both AP columns and 0.5 for precision and recall.

## EqualScoresKeepTheOrderTheyWereAddedIn

A miss and a hit share the score 0.5. Added miss first, the AP at IoU 0.5 is 0.4975; added hit
first, it is `K_FULL_AP`. The ranking is therefore stable.

## LooseDetectionCountsAtTheLowThresholdsOnly

A detection at IoU 0.625 has the flags 1, 1, 1 and then seven zeros. Its AP at IoU 0.5 is
`K_FULL_AP` and its mean over the ten thresholds is 3 * 0.995 / 10, which is 0.2985.

## ImageWithoutGroundTruthAddsFalsePositives

The second image has no ground truth and one detection at score 0.9, which ranks above the
first image's hit at 0.6. The summary counts 2 images and 1 instance, the class row counts 1
image, the AP at IoU 0.5 is 0.4975, precision is 0.5 and recall is 1.

## ImageWithoutDetectionsLowersRecall

The second image has a box and no detection. `image_count()` is 2, the class row counts 2
images and 2 instances, recall is 0.5, precision is 1, and both AP columns are 0.49 + 0.005,
which is 0.495.

## DetectionWhoseBestBoxIsTakenStaysUnmatched

The first detection takes `K_BOX_A`. The second overlaps `K_BOX_A` by 0.875 and `K_BOX_LOOSE`
by 40 / 56, about 0.714. Its flags are ten zeros: a detection is matched to its box of largest
IoU or to none.

## DetectionTakesTheBoxOfLargestIoU

The boxes are listed `K_BOX_LOOSE` then `K_BOX_A`, and one detection equals `K_BOX_NEAR`. Its
flags are eight ones and two zeros, which is the IoU 0.875 it has with the box listed second. A
matcher that took the first box passing 0.5 gives five ones.

## DetectionWithNaNBoxDoesNotMatch

A detection whose x1 is NaN has a NaN IoU with `K_BOX_A`, and its flags are ten zeros. The row
is built by hand because `make_row` takes a box constant.

## BoxGoesToItsFirstDetectionAtEachThreshold

A detection at IoU 0.625 ranks above one at IoU 1. The first holds the box at 0.5, 0.55 and
0.6; the second gets it from 0.65 up. The flags are 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 and
0, 0, 0, 1, 1, 1, 1, 1, 1, 1.

## DetectionOfAnotherClassDoesNotMatch

A detection of class 1 that equals a box of class 0 has ten zero flags.

## NoGroundTruthGivesNoRows

An empty accumulator has no rows. After an image with one detection and no ground truth it
counts 1 image, has no rows, and its four means are 0.

## ClearDropsEveryImage

An image with a miss is added and `clear()` is called. `image_count()` is 0. An image with a
hit added afterwards gives 1 image, 1 instance and AP `K_FULL_AP`, so nothing of the first
image remains.

## RejectsMalformedImages

`add_image` throws for a detection array of 5 values ("[N, 6]", "got 5"), for one box with two
classes ("4 values", "2 classes"), for a ground-truth class of -1 ("box 0", "negative class
-1"), for a detection class of 0.5 ("detection 0", "not a class index") and for a NaN score
("detection 0", "score"). The accumulator still holds no image and no rows afterwards.
`match_detections` throws for a box without a class ("0 classes") and for a flag array of 9
values ("need 10 values", "got 9").
