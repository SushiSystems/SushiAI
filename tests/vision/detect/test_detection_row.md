# tests/vision/detect/test_detection_row.cpp


Suites: `Small_DetectDetectionRow`.

Tests the columns of a detection row and the reference's limits on the rows of an image.

This file needs no device. `modules/vision/detect/include/SushiAI/vision/detect/detection_row.hpp`
holds constants only, so each case compares them with the values the header documents. A change
to any of them changes every reader of a detection row.

## K_EXACT

`K_EXACT` is the tolerance 0 of a float constant compared with the literal that defines it.

## ARowHoldsFourCornersThenTheScoreThenTheClass

`K_DETECTION_VALUES` is 6, `K_DETECTION_SCORE` is 4 and `K_DETECTION_CLASS` is 5, so a row is
x1 y1 x2 y2, then the score, then the class, and the class column lies inside the row.

## TheLimitsAreTheReferenceValues

`K_MAX_DETECTIONS` is 300 rows per image and `K_VALIDATION_CONFIDENCE` is 0.001.
