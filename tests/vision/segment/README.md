# tests/vision/segment

The tests of the module `vision/segment`, one file per header of `modules/vision/segment/include/`. The suite prefix of a case gives its ctest label.

## test_confusion_matrix.cpp

Suites: `Unit_SegmentConfusionMatrix`.

Tests `Vision::Segment::ConfusionMatrix` in
`modules/vision/segment/include/SushiAI/vision/segment/confusion_matrix.hpp`. Every `add` call
passes the ignore index 255.

### ANewMatrixCountsNothing

A matrix made for 3 classes reports 3 classes, 0 pixels and a count of 0 at (2, 1).

### AddCountsByTruthAndPrediction

For the predictions 0, 1, 1, 2, 2, 2 against the truth 0, 0, 1, 2, 1, 2, `count(truth,
predicted)` is 1 at (0, 0), (0, 1), (1, 1) and (1, 2), 2 at (2, 2) and 0 at (1, 0).
`truth_pixels(1)` is 2, `predicted_pixels(2)` is 3 and `pixel_count` is 6.

### AnIgnoredPixelIsNotCounted

Of three pixels, two have the truth 255, one of them with the prediction 7. The matrix counts
one pixel, at (0, 0), and `predicted_pixels(1)` is 0.

### CountsAccumulateOverBatches

The same three pixels added twice give 2 at (0, 0), (0, 1) and (1, 1) and a pixel count of 6.

### MergeAddsAndClearEmpties

Two matrices each count the predictions 1, 1 against the truth 0, 1. After `first.merge(second)`
and `second.clear()`, the first holds 2 at (0, 1) and at (1, 1) and the second counts 0 pixels.

### MakeRefusesAClassCountOutsideItsBounds

`ConfusionMatrix::make(0)` fails with `InvalidDescription`, "ConfusionMatrix" and "0 classes".
`make(K_MAX_CONFUSION_CLASSES + 1)` fails with "4097 classes" and "4096".

### AddRefusesWithoutCounting

Over 2 classes, `add` fails with `Incompatible` for spans of two and three pixels
("2 predicted", "3 true"), a truth of 2 at pixel 1 ("true class of pixel 1", "is 2",
"below 2", "255") and a prediction of -1 at pixel 1 ("predicted class of pixel 1", "is -1").
The pixel count stays 0 after the three calls.

### MergeRefusesAnotherClassCount

Merging a matrix of 3 classes into one of 2 fails with `Incompatible`, "2 classes" and "of 3".
