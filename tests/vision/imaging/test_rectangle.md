# tests/vision/imaging/test_rectangle.cpp


Suites: `Small_ImagingRectangle`.

Tests the plan that orders a dataset by aspect ratio and gives each batch one shape.

This file needs no device. `Small_ImagingRectangle` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/rectangle.hpp`. The expected shapes are what lines 405 to 427 of the
reference's `data/base.py` give for the same sizes.

## make_sizes

`make_sizes` returns seven image sizes with height over width below, at and above 1. Images 0
and 5 share the ratio 0.75.

## make_settings

`make_settings` returns `RectangleSettings` with a batch size, an image size and a pad, at the
default stride of 32.

## SettingsDefaultToTheReferenceValidationRun

Default settings are image size 640, batch size 32, stride 32 and pad 0.5.

## OrderRisesWithTheAspectRatioAndKeepsTies

`plan_rectangular_batches` orders the seven images as 6, 3, 0, 5, 2, 1, 4. Images 0 and 5 keep
their dataset order.

## ShapesMatchTheReferenceArithmetic

With batch 3 and pad 0.5 the shapes are 512 by 672, 672 by 672 and 672 by 448. With batch 2 and
pad 0 they are 448 by 640, 480 by 640, 640 by 640 and 640 by 448. With one batch of 7 at image
size 320 the shape is 352 by 352.

## ShapeAtNamesTheBatchOfAPosition

`shape_at` returns the first batch's shape for positions 0 and 2, the second's for position 3
and the third's for position 6.

## EverySideIsAMultipleOfTheStride

With stride 64 and image size 600, both sides of all four shapes are multiples of 64 and at
least 64.

## OneWideImageGivesOneWideShape

One 480 by 640 image gives the order {0} and the shape 512 by 672:
ceil(0.75 * 640 / 32 + 0.5) * 32 and ceil(20.5) * 32.

## RejectsSettingsAndSizesOutsideTheirRanges

`validate_rectangle` throws for a batch size, a stride or an image size of 0 and for a pad that
is negative or NaN. `plan_rectangular_batches` throws for an empty size list ("no image") and
for a size of 0 by 500 ("image 2", "0x500"). Default settings pass.

## RejectsAShapeAboveTheImageBound

An image size of `K_MAX_IMAGE_SIDE` with pad 0.5 gives a side above the bound, and the call
throws with "batch side" and "16384".
