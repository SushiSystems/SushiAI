# tests/vision/imaging/test_mask_pipeline.cpp


Suites: `Small_ImagingAffineNearest`, `Small_ImagingMaskPipeline`.

Tests `apply_affine_nearest` in
`modules/vision/imaging/include/SushiAI/vision/imaging/affine_nearest.hpp` and
`TrainingMaskPipeline` in
`modules/vision/imaging/include/SushiAI/vision/imaging/mask_pipeline.hpp`. Every plan scales by
one factor, translates and has the border value 255. The pipeline cases take grey RGB images
whose one box of the image's index covers the whole image, in an `InMemorySampleSource`, masks held in an `InMemoryMaskSource`, and the seed
20261011.

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

## AFlippedSampleHasItsMaskMirrored

With no mosaic, an identity warp and a flip probability of 1 over an 8 by 8 image, the trace
of sample 0 is flipped and has no mosaic. `make_mask` over that trace gives an 8 by 8 mask
whose pixel (y, x) holds the source mask's pixel (y, 7 - x).

## AMosaicMaskFollowsTheBoxesOfItsSample

Four images of four sizes have masks filled with their own index, and the training size is 32
with the default settings. For each sample of the epochs 0, 1 and 2 the trace has the mosaic
enabled and the mask is 32 by 32 with one channel. Every mask pixel at least 2 pixels inside a
box of the sample holds that box's class, and at least one pixel is compared.

## RefusesAMaskOfSeveralChannels

`make_mask` fails with `Incompatible` and "3 channels" for a mask of three channels.

## RefusesAMaskOfAnotherSizeThanItsImage

`make_mask` fails with `Incompatible`, "8x4" and "8x8" for a mask of 8 rows and 4 columns under an 8 by 8 image.

## ReturnsTheFailureOfAMaskThatCannotBeLoaded

When the mask source fails it returns that failure: `InvalidFile` and "mask 0 is unreadable".
