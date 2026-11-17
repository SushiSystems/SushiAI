# tests/applications/programs/source/test_detection_reports.cpp

Suite: `Medium_DetectionReports`, in `sushiai_programs_tests`.

Tests the three rows of `detection_reports.hpp` over a `PreparedInference` of the `tiny_yolo`
fixture (`prepared_inference_harness.hpp`): the anchor model the fixture is, with initialised
weights. The fixture holds no anchor-free and no YOLOX model, so those two rows are tested on what
they refuse before they read a head: a task section of another type, an input of the wrong rank and,
for YOLOX, a floor above the threshold. Their decoding runs in the example models of
`applications/examples`.

## TheAnchorReportNamesItsHeadsAndClassesAndTotalsEveryImage, TheAnchorReportEndsWithTheMeanAveragePrecision

The report succeeds, names "1 heads, 2 classes", ends the detections with "detections over 4
images." and prints the mean average precision line. The test states no detection count and no
precision: an untrained model fixes neither.

## TheAnchorReportMeasuresItsBoxesInTheSourceImageOfEachFile

The legend names the source image of each file, since the data publishes `fit`.

## TheAnchorReportReadsItsThresholdsFromTheEvaluationFile, TheAnchorReportKeepsItsDefaultThresholdsWhenTheFileStatesNone

A score threshold of 0.75 and a suppression of 0.3 stated by the file appear in the legend. With
none stated the legend carries the defaults, 0.25 and 0.45.

## TheAnchorReportRefusesAScoreThresholdBelowTheConfidenceFloor, TheAnchorReportRefusesAnInputThatIsNotHeightWidthDepth, TheAnchorReportRefusesAHeadThatNamesNoOutputOfTheModel

Each is refused before the report prints anything: a threshold of 0.1 under a floor of 0.5 names the
floor, a rank 2 input names the anchor model, and a head that names `head.9`, which the model does
not publish, fails.

## TheAnchorFreeReportRefusesATaskSectionOfAnotherType, TheAnchorFreeReportRefusesAnInputThatIsNotHeightWidthDepth

The anchor task of the fixture is refused naming `'type'` and `'anchor_free_detection'`. With an
anchor-free task and a rank 2 input the refusal names "an anchor-free model".

## TheYoloxReportRefusesATaskSectionOfAnotherType, TheYoloxReportRefusesAnInputThatIsNotHeightWidthDepth, TheYoloxReportRefusesAScoreThresholdBelowTheConfidenceFloor

The anchor task of the fixture is refused naming `'type'` and `'yolox_detection'`. With a YOLOX
task, a rank 2 input names "a YOLOX model", and a floor of 0.75 over the task's threshold of 0.5
names the floor.
