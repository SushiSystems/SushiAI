# tests/vision/imaging/test_resize.cpp


Suites: `Small_ImagingResize`, `Medium_ImagingResize` and the parameterised `Small_ImagingResizeConstant`, `Small_ImagingResizeLongSide` and `Medium_ImagingResizeAliasing`.

Tests the bilinear resize and the fit of an image's long side.

This file needs no device. The `Medium_` suites hold the death tests, since each starts a child process. The `Small_` suites cover `modules/vision/imaging/include/SushiAI/vision/imaging/resize.hpp`:
`resize_bilinear`, `fit_long_side` and `resize_long_side`. The kernel samples at pixel centres
with weights rounded to `K_BILINEAR_WEIGHT_BITS` bits and rounds its result half up. The first
cases compare whole images with values a comment derives.

## Bytes

`Bytes` is a vector of bytes, an image's contents in height, width, channel order.

## make_image and bytes_of

`make_image` returns an image holding the given bytes, and `bytes_of` returns an image's bytes.

## resized_bytes

`resized_bytes` resizes a one-channel source to a size and returns the bytes. It also expects
the destination to have that size and one channel.

## Helpers that hold a loop

`make_constant_image` and `expect_constant_image` build and check an image whose pixels hold one level per channel. `expect_row_monotone_at_every_width`, `expect_column_monotone_at_every_height`, `expect_rows_uniform_at_every_height` and `expect_long_side_at_image_size` run a check over a bounded range of sizes. `make_row_levels_source` returns the 4 by 3 source whose rows are each one level.

## TwoByTwoToFourByFourMatchesTheHandComputedImage

Along one axis the four outputs of a pair (a, b) are a, (3a + b) / 4, (a + 3b) / 4 and b. The
case lists all sixteen output bytes, among them 58.75 rounded to 59 and 76.25 rounded to 76.

## FourByFourToTwoByTwoAveragesEachTwoByTwoBlock

Each output reads position 0.5 or 2.5 on both axes, so it is the mean of a 2 by 2 block. The
sums 142, 220, 460 and 541 give 36, 55, 115 and 135: 35.5 rounds up and 135.25 rounds down.

## ResizeToTheSameSizeCopiesEveryPixel

A 3 by 3 image resized to 3 by 3 is unchanged.

## AnEnlargedRowClampsAtBothEdges

A row (10, 250) enlarged to four columns starts at 10 and ends at 250. Its outer columns read
positions outside the two pixel centres.

## AReducedRowUsesWeightsRoundedToElevenBits

Eight columns reduced to three read positions 0.8333, 3.5 and 6.1667, with upper weights 1707,
1024 and 341 of 2048. The case computes the three bytes from those weights, checks they are 3,
41 and 209, and compares the resize with them.

## EachChannelIsInterpolatedOnItsOwn

A two-pixel row of three channels enlarged to four pixels gives each channel the values of its
own pair. Channel 2 falls from 255 to 3 while the others rise.

## AConstantImageStaysConstant

Suite `Small_ImagingResizeConstant`. A 5 by 7 image of the colour (17, 114, 255) enlarged to 13 by 23 holds that colour in every pixel, and a 31 by 17 image at level 201 reduced to 4 by 5 holds 201 in every byte.

## AHorizontalGradientStaysMonotoneAtEveryWidth

A rising row of eight levels resized to every width from 1 to 40 never falls from one column to
the next.

## AnEnlargedGradientKeepsItsEndValues

The same row enlarged to 20 columns starts at 0 and ends at 255.

## AVerticalGradientStaysMonotone

A 4 by 3 image whose rows are each one level, resized to every height from 1 to 17 at width 5, never falls from one output row to the next.

## EachRowOfAVerticalGradientStaysUniform

The same resize gives rows that each hold one level, at every height from 1 to 17.

## TheDestinationIsResetToTheRequestedSizeAndChannels

A destination that was 9 by 9 with one channel becomes 2 by 2 with the source's three channels.

## ResizeRejectsASourceWithoutPixels

Suite `Medium_ImagingResize`. `resize_bilinear` on a default view stops the process with "resize_bilinear", "source" and "no pixels".

## ResizeRejectsASizeOutsideTheImageBounds

Destination sizes of 0 by 2, 2 by -1 and 2 by 16385 fail with "destination size" and the size.

## ResizeRejectsASourceWithExtentsOutsideTheImageBounds

Suite `Medium_ImagingResize`. A source view of 0 by 2 and one of 2 by 16385 stop the process with "source size" and the size.

## ResizeRejectsASourceWithAnUnsupportedChannelCount

Suite `Medium_ImagingResize`. A source view of five channels stops the process with "source channel count" and 5.

## ResizeRejectsADestinationThatHoldsTheSource

Suite `Medium_ImagingResize`. `resize_bilinear` with the image as source and destination stops the process with "destination holds the source's pixels".

## FitLongSideLeavesAnImageWhoseLongSideAlreadyFits

`fit_long_side` at 640 returns 480 by 640, 640 by 427 and 640 by 640 unchanged.

## FitLongSideReducesALargeImage

1080 by 1920 becomes 360 by 640, and 1920 by 1080 becomes 640 by 360.

## FitLongSideEnlargesASmallImage

100 by 50 becomes 640 by 320, and 500 by 375 becomes 640 by 480.

## FitLongSideRoundsTheShortSideUp

333 by 500 becomes 427 by 640, 7 by 3 becomes 640 by 275 and 3 by 1000 becomes 2 by 640: the
short side is the ceiling of 426.24, 274.29 and 1.92.

## FitLongSideCapsBothSidesAtTheImageSize

1281 by 1280 becomes 640 by 640. The short side would be the ceiling of 639.5.

## FitLongSideAlwaysPutsTheLongSideAtTheImageSize

Over heights and widths from 1 to 1499 in steps of 7 and 11, the long side of the result is 640
and the short side lies in [1, 640]. The loop is in `expect_long_side_at_image_size`.

## ResizeLongSideCopiesAnImageThatAlreadyFits

A 2 by 4 image with image size 4 comes out as a 2 by 4 image with the same bytes under another
data pointer. The destination, which was 3 by 3 with three channels, takes the source's one
channel.

## ResizeLongSideGivesTheResizeToTheFittedSize

Suite `Small_ImagingResizeLongSide`. A 1 by 2 image with image size 4 becomes 2 by 4, and both rows are 0, 25, 75, 100. The 4 by 4 image of the block-average case with image size 2 becomes the 2 by 2 image 36, 55, 115, 135.

## ResizeLongSideRejectsAnImageSizeOutsideTheImageBounds

Image sizes of 0 and 16385 fail with "resize_long_side" and the value.

## ResizeLongSideRejectsASourceWithoutPixels

Suite `Medium_ImagingResize`. A default view stops the process with "resize_long_side" and "no pixels".

## ResizeRejectsAViewOfTheDestination

Suite `Medium_ImagingResizeAliasing`. `resize_long_side` of the whole image into itself, `resize_bilinear` of a view of its two lower rows and `resize_long_side` of that view each stop the process with "destination holds the source's pixels".

## ResizeLongSideRejectsASourceWithExtentsOutsideTheImageBounds

Suite `Medium_ImagingResize`. A source view of 0 by 2 stops the process with "source size".

## ResizeLongSideRejectsASourceWithAnUnsupportedChannelCount

Suite `Medium_ImagingResize`. A source view of five channels stops the process with "source channel count".

## NearestRepeatsEachPixelWhenEnlarged

`resize_nearest` of the 2 by 2 image 1, 2, 3, 4 to 4 by 4 gives one channel with the rows
1 1 2 2, 1 1 2 2, 3 3 4 4, 3 3 4 4.

## NearestReadsThePixelUnderEachCentreWhenReduced

The row 10, 11, 12, 13, 14 resized to two columns is 11, 13.

## NearestKeepsClassIndicesUnblended

The row 0, 255 resized to seven columns is 0, 0, 0, 255, 255, 255, 255.

## NearestCopiesEveryChannelOfAPixel

An RGB row of the pixels (1, 2, 3) and (4, 5, 6), resized to 2 by 2 into a destination that
was 9 by 9 with one channel, gives a 2 by 2 image of three channels holding that row twice.

## NearestRejectsASourceWithoutPixels

Suite `Medium_ImagingResize`. `resize_nearest` of an empty view stops the process with "resize_nearest", "source" and "no pixels".

## NearestRejectsADestinationSizeWithAZeroSide

Suite `Medium_ImagingResize`. A destination size of 0 by 2 stops the process with "Image::reset" and "0x2".

## NearestRejectsADestinationThatHoldsTheSource

Suite `Medium_ImagingResize`. A destination that is the source image stops the process with "resize_nearest" and "destination holds the source's pixels".
