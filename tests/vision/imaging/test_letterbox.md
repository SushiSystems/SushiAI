# tests/vision/imaging/test_letterbox.cpp


Suites: `Small_ImagingLetterbox`, `Medium_ImagingLetterbox` and the parameterised `Small_ImagingLetterbox<Family>` and `Medium_ImagingLetterboxAliasing`.

Tests the plan and the application of the letterbox to an image and its labels.

This file needs no device. The `Small_` suites cover
`modules/vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp`. `plan_letterbox` computes the ratio, the resized size
and the four paddings, and draws no random number. The `Medium_` suites hold the death tests, since each starts a child process. `apply_letterbox` resizes with
`resize_bilinear` of `modules/vision/imaging/include/SushiAI/vision/imaging/resize.hpp`, pads, and moves boxes and polygons
by the same plan.

## K_DEFAULT_PADDING

`K_DEFAULT_PADDING` is 114, the padding level of the default settings.

## make_counted_image, make_flat_image and make_destination

`make_counted_image` returns an image whose byte at offset i is a first value plus i, modulo
256. `make_flat_image` returns an image with every byte at one level. `make_destination` returns an empty image for side 0 and otherwise a square image of that side held at 255.

## is_content

`is_content` reports whether a pixel of the target lies inside the plan's resized image.

## make_padded_copy and expect_same_image

`make_padded_copy` returns the source placed inside the padding of a plan that does not resize. `expect_same_image` expects two images to have the same size, channels and bytes.

## expect_two_levels

`expect_two_levels` expects every content byte of an image to be one level and every padding
byte another.

## expect_resized_content_between_padding, expect_box and expect_point

`expect_resized_content_between_padding` expects the content rectangle to hold the `resize_bilinear` result, the rest to hold 114 and the content not to be flat. `expect_box` and `expect_point` compare coordinates exactly.

## WideImageIsPaddedAboveAndBelowWithTheOddRowBelow

A 50 by 200 source in a 100 by 100 target has ratio 0.5 and resizes to 25 by 100. The 75 rows
of padding split into 37 above and 38 below, and the plan records the source and target sizes.

## TallImageIsPaddedLeftAndRightWithTheOddColumnRight

A 200 by 50 source resizes to 100 by 25 with 37 columns left and 38 right.

## ExactFitHasRatioOneAndNoPadding

A 64 by 64 source in a 64 by 64 target has ratio 1 and four paddings of 0.

## SmallImageIsOnlyPaddedWhileScaleUpIsOff

A 20 by 40 source in an 80 by 80 target keeps its size by default, with paddings 30, 30, 20, 20.

## ScaleUpEnlargesASmallImage

With `scale_up` set, the same source has ratio 2, a resized size of 40 by 80 and paddings of 20 above and below.

## ResizedSideRoundsAHalfToTheEvenInteger

At ratio 0.5 a height of 5 resizes to 2 and a height of 7 to 4: 2.5 and 3.5 round to the even
integer.

## RectangularStrideTargetPadsOnlyTheShortAxis

A 480 by 640 source in a 512 by 640 target has ratio 1 and 16 rows above and below.

## PlanCopiesThePaddingValueOfTheSettings

The plan's `padding_value` is that of the settings, at the default and at 7.

## PlanRejectsASideOutsideTheImageRange

Suite `Small_ImagingLetterboxSideRange`. A source of 0 by 8 and a target of 8 by -1 fail with "plan_letterbox" and the size; a source height and a target width of 16385 fail with the size and "16384".

## PlanRejectsAResizedSideOfZeroPixels

A 1 by 1000 source in an 8 by 8 target resizes to a height of 0, and the plan returns `InvalidDescription` with "zero
pixels".

## ImageWithoutResizeKeepsItsPixelsInsideThePadding

A 2 by 4 source in a 4 by 6 target is placed at row 1, column 1. Every content byte equals the
source byte and every other byte is the padding value.

## ExactFitCopiesEveryPixel

A plan without resize or padding copies the source.

## FlatImageIsResizedThenPaddedWithTheFillValue

Suite `Small_ImagingLetterboxFlatFit`. A flat 50 by 200 image at level 200 gives a 100 by 100 result that is 200 inside the content rectangle and 114 outside it; a flat one-channel 200 by 50 image at level 33 with a padding value of 7 gives 33 inside and 7 outside.

## ResizedContentEqualsTheBilinearResizeOfTheSource

Suite `Small_ImagingLetterboxResizedContent`. For a counted 50 by 200 image into an empty destination, and for a counted 200 by 50 image into a 100 by 101 target with a 120 by 120 destination held at 255, every content byte equals the byte `resize_bilinear` gives for the plan's resized size and every padding byte is 114. The content is not flat, since a flat rectangle cannot tell a resize from a fill.

## SourceWithoutPixelsIsRejected

Suite `Medium_ImagingLetterbox`. `apply_letterbox` on a default view stops the process with "apply_letterbox" and "no pixels".

## ViewInsideTheDestinationIsRejected

Suite `Medium_ImagingLetterboxAliasing`. `apply_letterbox` with the image as source and destination, and with a view of the two lower rows of the destination, stops the process with "destination holds the source's pixels".

## DestinationIsFullyRewrittenWhenItIsReused

A destination that was 9 by 9 and filled with 255 becomes 4 by 6 with content 50 and padding
114.

## ImageOfAnotherSizeThanThePlanIsRejected

Suite `Medium_ImagingLetterbox`. A 2 by 4 source with a plan made for 4 by 4 stops the process with "plan was made for".

## PlanWhosePaddingsDoNotTileTheTargetIsRejected

Suite `Medium_ImagingLetterbox`. A plan whose right padding is raised by 1 stops the process with "tile".

## BoxesMoveByTheRatioThenTheLeftAndTopPadding

Every value is a dyadic fraction, so the comparison is exact. In the tall plan a box maps by x * 0.5 + 37 and y * 0.5, and in the wide plan by x * 0.5 and
y * 0.5 + 37. Classes and order are kept.

## BoxesOutsideTheImageAreNeitherClippedNorRemoved

A box that extends past the source and a box of zero size are both moved by the plan and kept.

## ExactFitLeavesBoxesWhereTheyWere

With ratio 1 and no padding a box keeps its four coordinates.

## PolygonMovesByTheRatioThenTheLeftAndTopPadding

In the tall plan the vertices (10, 20), (30, 40) and (-4, 210) move to (42, 10), (52, 20) and
(35, 105). An object without a polygon still has none.
