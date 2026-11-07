# tests/vision/imaging/test_affine_nearest.cpp


Suites: `Small_ImagingAffineNearest`.

Tests that `apply_affine_nearest` resets its destination to the plan's output and fills with the
plan's own border value.

This file needs no device. Its cases cover a used destination, another border value, a scale
that rounds a half up, the channels of one pixel, and each refusal. The plans of the scale,
channel and refusal cases scale by one factor, translate and have the border value 255.

## make_plan, make_image_of and expect_bytes

`make_plan` returns a plan that scales by one factor and translates, with the border value 255. `make_image_of` returns an image holding the listed bytes in order, and `expect_bytes` checks that an image holds them.

## make_shift_plan and counting_image

`make_shift_plan` returns a plan from one size to itself that shifts by a number of columns, with
a given border value. `counting_image` returns a one-channel image whose byte i holds i + 1, so no
byte equals a border value of 0 or 17.

## ADestinationOfAnotherShapeTakesThePlanOutput

A destination of 5 by 5 pixels and three channels, filled with 9, receives the identity warp of a
2 by 3 one-channel image. It ends 2 by 3 with one channel.

## TheIdentityWarpIntoAUsedDestinationCopiesTheSourceBytes

The same used destination, after the identity warp, holds the bytes of the source.

## ThePlansBorderValueFillsWhatTheSourceDoesNotCover

A shift of one column with the border value 17 fills column 0 with 17, and columns 1 and 2 hold
source columns 0 and 1.

## AScaleRepeatsPixelsAndBlendsNone

The row 3, 50, 7, 120 warped at scale 2 into 8 columns is 3, 50, 50, 7, 7, 120, 120, 255.

## EveryChannelOfAPixelMovesTogether

An RGB row of the pixels (1, 2, 3) and (4, 5, 6) translated by one column is (255, 255, 255),
(1, 2, 3).

## RefusesAViewWithoutPixelsAndAPlanOfAnotherSize

`apply_affine_nearest` fails with `InvalidDescription` and "holds no image" for an empty view. A plan whose source size is 3 by 3 over a 2 by 2 image fails with `Incompatible`, "3x3" and "2x2".

## RefusesADestinationThatSharesThePixelsOfTheSource

An output that is the source image fails with `InvalidDescription` and "share pixels".

## RefusesAShearedMatrix

A matrix with a shear term of 0.5 fails with `InvalidDescription` and "off-diagonal".
