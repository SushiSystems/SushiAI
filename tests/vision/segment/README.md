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

## test_intersection_over_union.cpp, test_pixel_accuracy.cpp, test_argmax_classes.cpp

IoU per class and its mean over the classes that occur, pixel accuracy, an empty matrix, and the class pick with its tie.

Suites: `Unit_SegmentMetrics`.

Tests `class_iou`, `class_occurs` and `mean_iou` in
`modules/vision/segment/include/SushiAI/vision/segment/intersection_over_union.hpp`,
`pixel_accuracy` in `modules/vision/segment/include/SushiAI/vision/segment/pixel_accuracy.hpp`
and `argmax_classes` in
`modules/vision/segment/include/SushiAI/vision/segment/argmax_classes.hpp`. Each ratio is
compared within 1e-15, and the matrices count with the ignore index 255.

### ClassIoUIsIntersectionOverUnion

For the predictions 0, 1, 1, 2, 2, 2 against the truth 0, 0, 1, 2, 1, 2, `class_iou` is 1/2
for class 0, 1/3 for class 1 and 2/3 for class 2, and `mean_iou` is the mean of the three.

### TheMeanSkipsAClassThatDoesNotOccur

For the predictions 0, 0, 1, 2 against the truth 0, 0, 1, 1, `class_occurs` is true for class
0 and false for class 2, `class_iou` of class 2 is 0, and `mean_iou` is (1 + 1/2) / 2.

### PixelAccuracyIsTheDiagonalOverTheCountedPixels

For seven pixels, the last with the truth 255, `pixel_accuracy` is 4/6.

### AnEmptyMatrixGivesZero

A matrix whose one pixel was ignored gives 0 for `mean_iou` and for `pixel_accuracy`.

### ArgmaxPicksTheLargestScoreAndTheLowestClassOnATie

Over three pixels of three scores each, (0.1, 0.9, 0.0), (2.0, -1.0, 2.0) and
(-3.0, -2.0, -1.0), `argmax_classes` writes 1, 0, 2.

## test_rasterise_outlines.cpp, test_fill_outline.cpp

The even-odd fill by pixel centres, a shared edge, a hole, clipping, the later outline winning an overlap, and the refusals.

Suites: `Unit_SegmentRasterise`.

Tests `fill_outline` in `modules/vision/segment/include/SushiAI/vision/segment/fill_outline.hpp`
and `rasterise_outlines` in
`modules/vision/segment/include/SushiAI/vision/segment/rasterise_outlines.hpp`. Every mask
starts filled with the background byte 9.

### ARectangleOnPixelEdgesFillsThePixelsItBounds

On a mask of 4 rows and 5 columns, the rectangle from (1, 1) to (4, 3) filled with 1 sets the
columns 1 to 3 of the rows 1 and 2 and leaves every other pixel 9.

### APixelIsFilledWhenItsCentreIsInside

On a mask of one row and three columns, a rectangle spanning x from 0.6 to 2.4 gives 9, 1, 9.

### ACentreOnASharedEdgeBelongsToOneSide

On a mask of one row and three columns, the rectangle from x = 0 to 1.5 filled with 1 gives
1, 9, 9. The rectangle from x = 1.5 to 3 filled with 2 then gives 1, 2, 2.

### ATriangleFillsTheCentresUnderItsSlope

On a 4 by 4 mask, the triangle (0, 0), (4, 0), (0, 4) fills the first 3, 2, 1 and 0 columns of
the rows 0 to 3.

### EveryPartOfAnOutlineIsFilled

On a mask of one row and five columns, an outline of the parts x from 0 to 1 and x from 3 to 5
gives 1, 9, 9, 1, 1.

### APartInsideAnotherCutsAHole

On a 3 by 3 mask, an outline of the rectangle from (0, 0) to (3, 3) and the rectangle from
(1, 1) to (2, 2) fills every pixel but the centre one.

### AnOutlineIsClippedToTheMask

On a 2 by 2 mask, the rectangle from (-5, -5) to (1, 50) fills column 0 of both rows.

### AnOutlineWithoutAreaFillsNothing

An outline of no part and an outline whose one part has two vertices both leave a 2 by 2 mask
at 9.

### FillRefusesABadMaskAndAVertexThatIsNotFinite

`fill_outline` fails with `InvalidDescription` for an unsized mask ("fill_outline",
"0 by 0"), for a mask of three channels ("3 channels") and for a second part whose second
vertex has a NaN coordinate ("vertex 1 of part 1"). After that last failure the mask still
holds 9 in every pixel.

### TheLaterOutlineWinsAnOverlap

`rasterise_outlines` of class 1 over x from 0 to 3 and class 2 over x from 2 to 4, at a size
of one row and five columns, gives a mask of that size and one channel holding 1, 1, 2, 2, 9.

### RasteriseRefusesABadSizeAndNamesTheOutline

`rasterise_outlines` fails with `InvalidDescription` for a size of 0 by 4
("rasterise_outlines", "0 by 4") and for a second outline with an infinite vertex
("outline 1", "vertex 0 of part 0").
