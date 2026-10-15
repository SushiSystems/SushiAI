# tests/vision/imaging/test_pipeline.cpp


Suites: `Unit_ImagingPipeline`.

Tests the training and the validation pipeline that compose the imaging transforms into a
sample.

This file needs no device and reads no file. `Unit_ImagingPipeline` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/pipeline.hpp`. A `TrainingPipeline` loads an image from a
`SampleSource`, draws and applies mosaic, affine, colour jitter and flip, and returns a
`TrainingTrace` of the plans it used. A `ValidationPipeline` resizes the long side and
letterboxes. The cases replay a sample's draws on a copy of its generator and rebuild the
sample from the trace with the transform functions, so the pipeline is compared with its parts
and not with stored images.

## Constants

`K_SIDE` is 32, the training size of the cases, so a mosaic canvas is 64 by 64.
`K_REPLAY_TOLERANCE` is 0: a replayed draw repeats the same arithmetic. `K_BOX_TOLERANCE` is
1e-4 px. `K_MIN_SCALE_GAP` is 2^-24, one step of the 24-bit uniform over the default scale
range of width 1.

## MemorySource

`MemorySource` is a `SampleSource` that holds images and labels in memory and records the index
of every `load`. `loads` returns the recorded indices and `clear_loads` forgets them.

## FixedPartners

`FixedPartners` is a `PartnerSource` that returns the three partners it was constructed with
and makes no draw.

## make_textured, make_constant and make_flat

`make_textured` returns a three-channel image whose every byte depends on its row, column,
channel and a salt. `make_constant` returns a three-channel image of one colour. `make_flat`
returns an image of a given channel count with every byte at one level.

## one_object

`one_object` returns the `FileLabels` of one object of a given class and box.

## fill_source

`fill_source` adds five textured images to a source: 32 by 64, 64 by 64, 64 by 32, 16 by 32 and
32 by 32. The first four carry one object each and the last carries none.

## make_settings

`make_settings` returns default `TrainingSettings` with the image size `K_SIDE`.

## make_still_settings

`make_still_settings` returns settings under which nothing moves: one epoch with
`close_mosaic` 1, so epoch 0 has no mosaic, affine ranges of 0, HSV gains of 0 and a flip
probability of 0.

## whole_dataset

`whole_dataset` returns a `DatasetPartners` over every image of a source.

## images_match

`images_match` returns a GoogleTest assertion result that says whether two images have the same
size, channel count and bytes, and how they differ when they do not.

## expect_same_labels

`expect_same_labels` expects two label sets to hold the same classes, box edges and polygon
vertices at `K_REPLAY_TOLERANCE`.

## expect_same_affine and expect_same_hsv

`expect_same_affine` compares the sizes and the three drawn values of two `AffinePlan` objects.
`expect_same_hsv` compares the three gains of two `HSVPlan` objects.

## load_patch

`load_patch` loads one image the way a pipeline does: `resize_long_side` to `K_SIDE`,
`expand_gray` when the image has one channel, and `to_pixel_labels` of the resampled polygons.

## compose_from_trace

`compose_from_trace` builds a sample from a `TrainingTrace` by calling the transforms in the
reference's order, mosaic, affine, colour jitter, flip (`data/augment.py`, lines 2870 to 2912),
each with the plan the trace holds.

## SettingsDefaultToTheReferenceRun

Default `TrainingSettings` have image size 640, 100 epochs, `close_mosaic` 10, batch size 16
and seed 0. Default `ValidationSettings` have image size 640 and a letterbox that does not
scale up.

## MosaicClosesForTheLastTenEpochsOfAHundred

`is_mosaic_closed` is false for epochs 0 and 89 of 100 with `close_mosaic` 10 and true for 90
and 99. With `close_mosaic` 0 it is false for epoch 99, and with 5 epochs and `close_mosaic`
10 it is true for epoch 0.

## SampleRandomDerivesFromTheSeedTheEpochAndTheIndex

`make_sample_random(2026, 3, 17)` equals `Random(2026).derive(3).derive(17)`. Another seed,
epoch or index gives another generator, and so do the epoch and the index swapped.

## TrainingDrawsMosaicThenAffineThenHSVThenFlip

The trace of sample 2 in epoch 5 equals a replay on `make_sample_random(41, 5, 2)`:
`draw_mosaic`, then `plan_affine` from the 64 by 64 canvas to 32 by 32, then `plan_hsv`, then
`plan_flip`.

## TrainingLoadsTheSampleThenItsPartnersInDrawOrder

The source records four loads: index 3, then the trace's three partners in order.

## TrainingAppliesItsPlansInTheReferenceOrder

For every index of the source in epochs 0 and 1, the sample's image and labels equal what
`compose_from_trace` builds from the trace.

## TrainingAppliesAForcedFlipAndSkipsAForbiddenOne

Two pipelines with one seed and flip probabilities of 1 and 0 give images that are mirrors of
each other, pixel for pixel. The flipped sample also equals `compose_from_trace` of its trace.

## TrainingProducesTheTrainingSizeAndTheSizesOfItsOwnImage

The sample is 32 by 32 with `K_RGB_CHANNELS` channels. `original_size` is 32 by 64, the stored
size of image 0, and `loaded_size` is 16 by 32, its size after the long-side resize. The affine
plan maps 64 by 64 to 32 by 32.

## MosaicSampleKeepsItsLargestQuadrantBoxInsideTheTrainingSize

Four 32 by 32 images each carry a box over the whole image, and the affine ranges are 0, so
the warp shows canvas rows and columns 16 to 48. Over eight seeds every box of the sample lies
inside the 32 by 32 output with positive width and height, and exactly one box equals the
quadrant that is at least 16 by 16, computed from the trace's centre.

## ClosedEpochSkipsTheMosaicAndItsDraws

With 10 epochs and `close_mosaic` 3, epoch 7 is closed. The trace says so, the mosaic is not
enabled, the source records one load, and the affine, HSV and flip plans equal a replay that
starts with `plan_affine` on the sample's generator. No mosaic draw precedes them. The sample
equals `compose_from_trace`.

## SampleWithoutMosaicKeepsTheBoxOfItsOwnImage

In a closed epoch with affine ranges of 0, the box (8, 4, 24, 12) of the 16 by 32 loaded image
becomes (8, 12, 24, 20) in the 32 by 32 output: the image is centred, which adds 8 to y. The
box is symmetric about the centre column, so a flip leaves it where it is.

## EpochBeforeTheClosedOnesStillMakesAMosaic

Epoch 6 of the same settings is open: the mosaic is enabled and the source records four loads.

## ClosedGateInAnOpenEpochWarpsTheLoadedImage

With a mosaic probability of 0 in an open epoch, the trace is not closed and the mosaic is not
enabled. The affine plan's source is the 32 by 16 loaded image, the source records one load,
and the sample equals `compose_from_trace`.

## SampleDependsOnTheSeedTheEpochAndTheIndexAlone

The same seed, epoch and index give the same image, labels, affine plan and partners a second
time. Another seed and another epoch each give another image and an affine scale that differs
by more than `K_MIN_SCALE_GAP`.

## EarlierSamplesDoNotShiftTheDrawsOfALaterOne

A pipeline that already made a mosaic sample and a closed-epoch sample, which differ by six
draws, gives the same sample for index 0 of epoch 0 as a new pipeline does.

## DefaultPartnersComeFromTheImagesThePipelineLoaded

A pipeline built without a partner source gives the first sample, index 3, the partners
3, 3, 3. The second sample's partners are each 3 or 1, the two indices loaded so far.

## GivenPartnerSourceChoosesThePartners

With `FixedPartners` of 4, 4, 0 the trace holds those partners and the source records the loads
2, 4, 4, 0.

## TrainingRejectsEverySettingOutOfRangeAtConstruction

Constructing a `TrainingPipeline` throws for an image size of 0 or 8193, 0 epochs, a
`close_mosaic` of -1, a batch size of 0, a mosaic probability of 1.5, an affine scale range of
1, a negative HSV gain and a NaN flip probability, each under the name of the validator that
found it. No rejected construction loads an image, and `validate_training` accepts the test
settings.

## TrainingRejectsAnEmptySourceAndANullPartnerSource

An empty source throws with "TrainingPipeline" and "no image", with or without a partner
source. A null partner source throws with "partner source is null".

## TrainingRejectsIndicesAndEpochsOutsideTheirRange

`make_sample` throws for index 5 of five images and for epochs -1 and 100 of 100.

## TrainingExpandsAGreyImageToRGB

A one-channel image at level 77 under the still settings gives a 32 by 32 sample of three
channels whose centre pixel is (77, 77, 77).

## TrainingComposesAMosaicOfGreyAndColourImages

A source of two one-channel images and one colour image, with fixed partners, gives a
three-channel 32 by 32 sample that equals `compose_from_trace`.

## TrainingRejectsAnImageOfFourChannels

A four-channel image throws from `make_sample` with "4 channels" and "image 0".

## TrainingCarriesAPolygonAndTakesTheBoxFromIt

Under the still settings a triangle in a 32 by 32 image keeps its place. The sample's one
object has class 5, the box (8, 8, 24, 24) and a polygon of `K_POLYGON_POINTS` points that
starts at (8, 8).

## ValidationResizesTheLongSideThenOnlyPads

An 8 by 16 image of one colour is loaded as 16 by 32 and letterboxed into 32 by 32 with ratio 1
and 8 rows above and below. The sample's rows 8 to 23 hold the colour and the other rows hold
114. The source records one load.

## ValidationMovesBoxesByThePaddingWithoutClipping

The box (8, 4, 24, 12) of the loaded image becomes (8, 12, 24, 20): the top padding adds 8 to
y.

## ValidationIntoARectangularTargetAddsNoPadding

With a 16 by 32 target for a 16 by 32 image the four paddings are 0 and the sample's image
equals the source image.

## ValidationRepeatsItsSampleAfterAnotherIndex

Sample 0 made again after sample 2 has the same image and labels as the first time, and sample
2 has another image.

## ValidationExpandsAGreyImageToRGB

A one-channel 8 by 16 image at level 61 gives a three-channel 32 by 32 sample with 61 in the
content rows and 114 in the padding.

## ValidationCarriesAPolygonThroughTheLetterbox

A triangle's polygon is resampled to `K_POLYGON_POINTS` points, and its first vertex, (8, 4) on
the loaded image, becomes (8, 12).

## ValidationRejectsSettingsAndIndicesOutsideTheirRange

Constructing a `ValidationPipeline` throws for an image size of 0 ("validate_validation") and
for an empty source ("no image"). `make_sample` throws for index 9 of five images.
