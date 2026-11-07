# tests/vision/imaging/test_mask_pipeline.cpp


Suites: `Small_ImagingMaskPipeline`.

Tests `TrainingMaskPipeline` in
`modules/vision/imaging/include/SushiAI/vision/imaging/mask_pipeline.hpp`. The cases take grey RGB images
whose one box of the image's index covers the whole image, in an `InMemorySampleSource`, masks held in an `InMemoryMaskSource`, and the seed
20261011.

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
