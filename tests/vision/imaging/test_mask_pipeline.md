# tests/vision/imaging/test_mask_pipeline.cpp


Suites: `Small_ImagingAffineNearest`, `Small_ImagingMaskPipeline`.

Tests `apply_affine_nearest` in
`modules/vision/imaging/include/SushiAI/vision/imaging/affine_nearest.hpp` and
`TrainingMaskPipeline` in
`modules/vision/imaging/include/SushiAI/vision/imaging/mask_pipeline.hpp`. Every plan scales by
one factor, translates and has the border value 255. The pipeline cases take grey RGB images
whose one box of the image's index covers the whole image, masks held in memory, and the seed
20261011.

## AnIdentityPlanCopiesTheSource

A plan of scale 1 and no translation over a mask of 3 rows and 4 columns gives an image of
that size and one channel holding the source's bytes.

## ATranslationFillsWhatItUncoversWithTheBorderValue

A translation of 2 columns and 1 row over a mask of 2 rows and 4 columns gives a first row of
255. In the second row the first two columns are 255 and column x holds the source's row 0,
column x - 2.

## AScaleRepeatsPixelsAndBlendsNone

The row 3, 50, 7, 120 warped at scale 2 into 8 columns is 3, 50, 50, 7, 7, 120, 120, 255.

## EveryChannelOfAPixelMovesTogether

An RGB row of the pixels (1, 2, 3) and (4, 5, 6) translated by one column is (255, 255, 255),
(1, 2, 3).

## RefusesWhatItCannotWarp

`apply_affine_nearest` fails with `InvalidDescription` for an empty view ("holds no image"),
for an output that is the source image ("share pixels") and for a matrix with a shear term of
0.5 ("off-diagonal"). A plan whose source size is 3 by 3 over a 2 by 2 image fails with
`Incompatible`, "3x3" and "2x2".

## AFlippedSampleHasItsMaskMirrored

With no mosaic, an identity warp and a flip probability of 1 over an 8 by 8 image, the trace
of sample 0 is flipped and has no mosaic. `make_mask` over that trace gives an 8 by 8 mask
whose pixel (y, x) holds the source mask's pixel (y, 7 - x).

## AMosaicMaskFollowsTheBoxesOfItsSample

Four images of four sizes have masks filled with their own index, and the training size is 32
with the default settings. For each sample of the epochs 0, 1 and 2 the trace has the mosaic
enabled and the mask is 32 by 32 with one channel. Every mask pixel at least 2 pixels inside a
box of the sample holds that box's class, and at least one pixel is compared.

## RefusesAMaskItCannotReplay

`make_mask` fails with `Incompatible` for a mask of three channels ("3 channels") and for a
mask of 8 rows and 4 columns under an 8 by 8 image ("8x4", "8x8"). When the mask source fails
it returns that failure: `InvalidFile` and "mask 0 is unreadable".
