# tests/vision/imaging/test_boxes.cpp


Suites: `Unit_ImagingBoxes`.

Tests the two box forms, their conversions and the candidate filter.

This file needs no device. `Unit_ImagingBoxes` covers `modules/vision/imaging/include/SushiAI/vision/imaging/boxes.hpp`: a
`NormalizedBox` is a centre and an extent in fractions of the image, a `PixelBox` is two corners
in pixels, and `is_box_candidate` decides whether a box survives an augmentation.

## Tolerances and K_SIZE

`K_PIXEL_TOLERANCE` is 1e-3 px and `K_FRACTION_TOLERANCE` is 1e-6 of the image.
`K_EPSILON_TOLERANCE` is 1e-22, for the filter's default epsilon of 1e-16. `K_SIZE` is 480 rows
by 640 columns, the size the hand cases use.

## expect_pixel_box

`expect_pixel_box` compares the four corners of a `PixelBox` within `K_PIXEL_TOLERANCE`.

## expect_normalized_box

`expect_normalized_box` compares the four values of a `NormalizedBox` within
`K_FRACTION_TOLERANCE`.

## box_of_sides

`box_of_sides` returns a box with its left top corner at the origin and the given width and
height.

## ToPixelBoxPlacesCornersFromCentreAndExtent

The normalized box (0.5, 0.25, 0.5, 0.25) in `K_SIZE` is the pixel box (160, 60, 480, 180).

## ToPixelBoxUsesWidthForXAndHeightForY

A box covering a 100 by 200 image is (0, 0, 200, 100). The two sides differ, so a conversion
that swapped them gives other corners.

## ToPixelBoxDoesNotClip

A box centred on the left edge converts to (-50, 75, 50, 125). The conversion leaves the
negative corner as it is.

## ToNormalizedBoxDividesCentreAndExtentByTheImageSize

The pixel box (160, 60, 480, 180) in `K_SIZE` is the normalized box (0.5, 0.25, 0.5, 0.25).

## NormalizedToPixelAndBackReturnsTheBox

Four normalized boxes, one of them all zeros, come back from the two conversions within
`K_FRACTION_TOLERANCE`.

## PixelToNormalizedAndBackReturnsTheBox

The pixel box (12.5, 7.25, 630, 471.75) comes back within `K_PIXEL_TOLERANCE`.

## ScaleBoxMultipliesEachAxisByItsOwnFactor

`scale_box` by 2 across and 0.5 down maps (10, 20, 30, 40) to (20, 10, 60, 20).

## TranslateBoxAddsEachAxisItsOwnOffset

`translate_box` by 5 across and -25 down maps (10, 20, 30, 40) to (15, -5, 35, 15).

## ClipBoxClampsEachEdgeToTheImage

`clip_box` maps (-10, 20, 700, 500) in `K_SIZE` to (0, 20, 640, 480).

## ClipBoxLeavesABoxInsideTheImageAlone

The box (0, 0, 640, 480) is unchanged.

## ClipBoxCollapsesABoxOutsideTheImageOntoTheEdge

A box to the right of the image clips to (640, 10, 640, 20) and one above it to
(10, 0, 20, 0). Both have area 0.

## BoxAreaIsWidthTimesHeight

`box_area` of (1, 2, 4, 6) is 12.

## PolygonFilterLowersOnlyTheAreaRatioToOneHundredth

`make_polygon_filter` returns a `CandidateFilter` whose `min_area_ratio` is 0.01, which is
`K_POLYGON_MIN_AREA_RATIO`. Its `min_side`, `max_aspect_ratio` and `epsilon` equal those of a
default filter.

## PolygonFilterKeepsABoxThePlainFilterRejects

A box that goes from 100 by 100 to 50 by 10 keeps 5 percent of its area. The default filter
rejects it and the polygon filter keeps it.

## CandidateFilterDefaultsAreTheReferenceThresholds

A default `CandidateFilter` has `min_side` 2, `max_aspect_ratio` 100, `min_area_ratio` 0.10 and
`epsilon` 1e-16.

## CandidateNeedsAWidthAboveTheMinimumSide

A width of 2 is rejected and a width of 2.25 is kept. The comparison is strict.

## CandidateNeedsAHeightAboveTheMinimumSide

A height of 2 is rejected and a height of 2.25 is kept.

## CandidateNeedsMoreThanTheMinimumAreaRatio

Against a 100 by 100 box, a 10 by 100 box has a ratio of 0.10 in float and is rejected. A width
of 10.5 is kept and a width of 9.5 is rejected.

## CandidateAreaRatioFollowsTheFilterField

With `min_area_ratio` set to 0.01, the 10 by 100 box is kept and a 10 by 10 box, whose ratio is
0.01, is rejected.

## CandidateNeedsAnAspectRatioBelowTheMaximum

A 300 by 3 box and a 3 by 300 box have an aspect ratio of 100 in float and are rejected. Boxes
of 297 by 3 and 3 by 297 are kept. The filter takes the larger of the two ratios.

## CandidateMeasuresSidesNotPositions

A 50 by 40 box moved to another position is kept. The filter reads widths and heights and no
coordinate.

## CandidateEpsilonIsAddedToTheAreaBefore

With `epsilon` 100 and `min_area_ratio` 0.6, a 10 by 10 box against itself has the ratio
100 / (100 + 100), which is 0.5, and is rejected. Without the epsilon in the denominator the
ratio is 1.

## CandidateEpsilonIsAddedToTheHeightOfAWideBox

With `epsilon` 1, a 300 by 3 box has the ratio 300 / (3 + 1), which is 75, and is kept.

## CandidateEpsilonIsAddedToTheWidthOfATallBox

With `epsilon` 1, a 3 by 300 box has the ratio 75 and is kept.

## CandidateWithAnEmptyBeforeBoxIsKeptAndAnEmptyAfterBoxIsNot

An empty box before and a 10 by 10 box after is kept. A 10 by 10 box before and an empty box
after is rejected.
