# tests/vision/imaging/test_affine_nearest.cpp


Suites: `Small_ImagingAffineNearest`.

Tests that `apply_affine_nearest` resets its destination to the plan's output and fills with the
plan's own border value.

This file needs no device. The cases of `Small_ImagingAffineNearest` in
`tests/vision/imaging/test_mask_pipeline.cpp` check the identity, a translation, a scale that
rounds a half up, the channels of one pixel, and each refusal. Their destinations start empty and
their plans all use the border value 255. This file covers a used destination and another border
value.

## make_shift_plan and counting_image

`make_shift_plan` returns a plan from one size to itself that shifts by a number of columns, with
a given border value. `counting_image` returns a one-channel image whose byte i holds i + 1, so no
byte equals a border value of 0 or 17.

## ADestinationOfAnotherShapeTakesThePlanOutput

A destination of 5 by 5 pixels and three channels, filled with 9, receives the identity warp of a
2 by 3 one-channel image. It ends 2 by 3 with one channel and holds the source bytes.

## ThePlansBorderValueFillsWhatTheSourceDoesNotCover

A shift of one column with the border value 17 fills column 0 with 17, and columns 1 and 2 hold
source columns 0 and 1.
