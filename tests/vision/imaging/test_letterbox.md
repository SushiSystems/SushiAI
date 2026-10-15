# tests/vision/imaging/test_letterbox.cpp


Suites: `Unit_ImagingLetterbox`.

Tests the plan and the application of the letterbox to an image and its labels.

This file needs no device. `Unit_ImagingLetterbox` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp`. `plan_letterbox` computes the ratio, the resized size
and the four paddings, and draws no random number. `apply_letterbox` resizes with
`resize_bilinear` of `modules/vision/imaging/include/SushiAI/vision/imaging/resize.hpp`, pads, and moves boxes and polygons
by the same plan.

## K_BOX_TOLERANCE

`K_BOX_TOLERANCE` is 1e-4 px.

## make_counted_image and make_flat_image

`make_counted_image` returns an image whose byte at offset i is a first value plus i, modulo
256. `make_flat_image` returns an image with every byte at one level.

## is_content

`is_content` reports whether a pixel of the target lies inside the plan's resized image.

## expect_two_levels

`expect_two_levels` expects every content byte of an image to be one level and every padding
byte another.

## WideImageIsPaddedAboveAndBelowWithTheOddRowBelow

A 50 by 200 source in a 100 by 100 target has ratio 0.5 and resizes to 25 by 100. The 75 rows
of padding split into 37 above and 38 below, and the plan records the source and target sizes.

## TallImageIsPaddedLeftAndRightWithTheOddColumnRight

A 200 by 50 source resizes to 100 by 25 with 37 columns left and 38 right.

## ExactFitHasRatioOneAndNoPadding

A 64 by 64 source in a 64 by 64 target has ratio 1 and four paddings of 0.

## SmallImageIsOnlyPaddedUnlessScaleUpIsSet

A 20 by 40 source in an 80 by 80 target keeps its size by default, with paddings 30, 30, 20,
20. With `scale_up` set the ratio is 2, the resized size is 40 by 80 and the paddings are 20
above and below.

## ResizedSideRoundsAHalfToTheEvenInteger

At ratio 0.5 a height of 5 resizes to 2 and a height of 7 to 4: 2.5 and 3.5 round to the even
integer.

## RectangularStrideTargetPadsOnlyTheShortAxis

A 480 by 640 source in a 512 by 640 target has ratio 1 and 16 rows above and below.

## PlanCopiesThePaddingValueOfTheSettings

The plan's `padding_value` is that of the settings, at the default and at 7.

## PlanRejectsASideThatIsNotPositive

A source of 0 by 8 and a target of 8 by -1 throw with "plan_letterbox" and the size.

## PlanRejectsASideAboveTheImageBound

A source height and a target width of 16385 throw with the size and "16384".

## ValidateAcceptsEverySetting

`validate_letterbox` does not throw with `scale_up` set and a padding value of 255, or for the
defaults.

## PlanRejectsAResizedSideOfZeroPixels

A 1 by 1000 source in an 8 by 8 target resizes to a height of 0, and the plan throws with "zero
pixels".

## ImageWithoutResizeKeepsItsPixelsInsideThePadding

A 2 by 4 source in a 4 by 6 target is placed at row 1, column 1. Every content byte equals the
source byte and every other byte is the padding value.

## ExactFitCopiesEveryPixel

A plan without resize or padding copies the source.

## WideImageIsResizedThenPaddedWithTheFillValue

A flat 50 by 200 image at level 200 gives a 100 by 100 result that is 200 inside the content
rectangle and 114, the default padding value, outside it.

## TallImageIsResizedThenPaddedWithACustomFillValue

A flat one-channel 200 by 50 image at level 33 with a padding value of 7 gives 33 inside and 7
outside.

## ResizedContentEqualsTheBilinearResizeOfTheSource

For a counted 50 by 200 image, every content byte equals the byte `resize_bilinear` gives for
the plan's resized size, and every padding byte is 114. The case also expects the content not
to be flat, since a flat rectangle cannot tell a resize from a fill.

## SourceWithoutPixelsIsRejected

`apply_letterbox` on a default view throws with "apply_letterbox" and "no pixels".

## DestinationThatIsTheSourceIsRejected

`apply_letterbox` with the image as source and destination throws with "destination holds the
source's pixels", and the image keeps its size.

## ViewInsideTheDestinationIsRejected

A view of two rows of the destination is refused with the same message, and the image keeps its
size and its bytes.

## ResizedContentBesideSidePaddingEqualsTheBilinearResize

A counted 200 by 50 image in a 100 by 101 target has 38 columns of padding on each side. The
destination starts as a 120 by 120 image filled with 255. After the call no byte differs from
the bilinear resize placed between two bands of 114.

## DestinationIsFullyRewrittenWhenItIsReused

A destination that was 9 by 9 and filled with 255 becomes 4 by 6 with content 50 and padding
114.

## ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 4 source with a plan made for 4 by 4 throws with "plan was made for".

## PlanWhosePaddingsDoNotTileTheTargetIsRejected

A plan whose right padding is raised by 1 throws with "tile".

## BoxesMoveByTheRatioThenTheLeftAndTopPadding

In the tall plan a box maps by x * 0.5 + 37 and y * 0.5, and in the wide plan by x * 0.5 and
y * 0.5 + 37. Classes and order are kept.

## BoxesOutsideTheImageAreNeitherClippedNorRemoved

A box that extends past the source and a box of zero size are both moved by the plan and kept.

## ExactFitLeavesBoxesWhereTheyWere

With ratio 1 and no padding a box keeps its four coordinates.

## PolygonMovesByTheRatioThenTheLeftAndTopPadding

In the tall plan the vertices (10, 20), (30, 40) and (-4, 210) move to (42, 10), (52, 20) and
(35, 105). An object without a polygon still has none.
