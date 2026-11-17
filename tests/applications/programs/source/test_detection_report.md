# tests/applications/programs/source/test_detection_report.cpp

Suites: `Small_DetectionReport` and `Medium_DetectionReport`, in `sushiai_programs_tests`.

Tests `detection_report.hpp`: the legend, the three checks and the batch loop every detection
report shares. The batch loop runs over the `tiny_yolo` fixture, four images in two batches of two,
through the `PreparedInference` of `prepared_inference_harness.hpp`. A scripted decoder stands for
the model: it hands the same detections to every batch and counts its calls, so each case states
the detections it expects to see printed.

## TheLegendStatesTheThresholdsAndTheFrameOfTheBoxes

The legend of a score of 0.5, a suppression of 0.25 and the frame "16 by 16 input" is the whole line
and a blank line.

## AnInputThatIsNotHeightWidthDepthIsRefused, AnInputOfHeightWidthDepthPasses

A rank 2 input is refused as `InvalidDescription`; the message names the model file, the model kind
and `[height, width, channels]`. A rank 3 input passes.

## AScoreThresholdBelowTheConfidenceFloorIsRefused, AScoreThresholdEqualToTheConfidenceFloorPasses

A threshold of 0.25 under a floor of 0.5 is refused as `InvalidDescription` naming both. Equal
values pass, since only a threshold below the floor could already have dropped a printed detection.

## BoxesAreMeasuredInTheSourceImageWhenTheDataPublishesFit, BoxesAreMeasuredInTheInputWhenTheDataHidesFit

Data that publishes `fit` is described as "source image of each file". With `fit` hidden, an input
of 20 by 30 is described as "30 by 20 input": width, then height.

## TheDecoderRunsOncePerBatch

Over two batches the decoder is called twice and the report succeeds.

## EachImagePrintsOnlyTheDetectionsAboveTheScoreThreshold

Image 0 holds rows of score 0.9 and 0.1, image 1 one row of score 0.4, and the plan keeps scores
above 0.5. Each batch prints "image N: 1 detections" for its first image and "image N: 0
detections" for its second, numbered across batches 0 to 3, and no row of score 0.1 or 0.4.

## ADetectionPrintsItsClassItsScoreAndItsBox

A row of class 1, score 0.9 and corners 10, 20, 30, 40 prints as one line with the class right
aligned to 3, the score to 4 decimals and each corner to 2 decimals, right aligned to 8.

## TheTotalCountsEveryPrintedDetectionOverEveryImage

Two kept rows in image 0 of each of two batches print "4 detections over 4 images."

## AMappingPlanClipsTheBoxToTheSourceImageOfItsFile, AnUnmappedPlanPrintsTheBoxAsDecoded

A box far larger than the image, `(-1e6, -1e6, 1e6, 1e6)`, prints as `(0, 0, width, height)` of the
source image when the plan maps to the source, the size read from the `fit` role of batch 0, and as
given when it does not.

## DataWithBoxesEndsWithTheMeanAveragePrecision, DataWithoutBoxesStatesThatNothingWasScored, AReportOfDetectionsOnlyScoresNothing

Data with the `boxes` role ends with the "Mean average precision at IoU 0.5" line over "boxes of 4
images". With the role hidden, and with `Comparison::DetectionsOnly` over data that has it, the line
reads "Mean average precision : not computed".

## AFailingDecoderStopsTheReportWithItsFailure

A decoder that fails on its second call returns its `DeviceFailed` through the report, is not called
again and the total line is not printed.
