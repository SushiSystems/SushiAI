# tests/vision/datasets/test_fit_mask.cpp


Suites: `Small_DatasetsFitMask`, `Small_DatasetsFitMaskRefusal`,
`Small_DatasetsLoadSampleMaskRefusal` and `Medium_DatasetsLoadSampleMask`.

Tests `fit_mask` in `modules/vision/datasets/include/SushiAI/vision/datasets/fit_mask.hpp` and
`load_sample_mask` in
`modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_mask.hpp`. The
`load_sample_mask` case that reads a path is `Medium_`; the one-channel decode case of the
earlier file is deleted, as `test_load_sample_mask.cpp` holds that promise.

## AResizeRepeatsTheNearestClass

A 2 by 2 mask of 0, 1, 2, 3 resized to 4 by 4 has one channel and holds each class in a 2 by 2
block: the rows 0 0 1 1, 0 0 1 1, 2 2 3 3, 2 2 3 3.

## AReductionTakesThePixelUnderEachCentre

A 4 by 4 mask of the bytes 0 to 15 resized to 2 by 2 holds 5, 7, 13, 15.

## ALetterboxPadsWithTheVoidByte

A mask of 2 rows and 4 columns letterboxed to 4 by 4 keeps its two rows in the middle, with a
row of `K_MASK_VOID` above and below.

## ALetterboxThatReducesSamplesTheNearestPixel

A mask of 4 rows and 8 columns holding the bytes 0 to 31, letterboxed to 4 by 4, has the rows
9, 11, 13, 15 and 25, 27, 29, 31 between two rows of `K_MASK_VOID`.

## AMaskMustMatchItsImage

`Small_DatasetsFitMaskRefusal`, one row each: `fit_mask` fails with `Incompatible` and
"fit_mask" for a mask of three channels ("the mask has 3 channels") and for a 2 by 2 mask whose
image is 2 by 4 ("the mask is 2 by 2 pixels; its image is 2 by 4").

## ATargetSizeOfNoPixelIsRefused

A target size of 0 by 4 fails with `InvalidDescription`, "fit_mask" and "0 by 4 pixels".

## ASampleMustNameAMaskOfTheList

`Small_DatasetsLoadSampleMaskRefusal`, one row each. Over a list of one file and one sample,
`load_sample_mask` fails with `InvalidDescription` for sample 1
("load_sample_mask", "sample 1 is outside a list of 1 samples"), for a sample without a mask
("sample 0 has no mask") and for the mask index 4
("sample 0 names the mask file 4 of a list of 1 files").

## AnAbsentMaskFileIsReported

`Medium_DatasetsLoadSampleMask`, since it reaches a path on disk: a mask file that does not exist
fails with `FileNotFound` and "absent.png".
