# tests/vision/datasets/test_idx.cpp


Suites: `Small_DatasetsIdx`.

Tests `read_idx` in `modules/vision/datasets/include/SushiAI/vision/datasets/idx.hpp` over IDX
files the cases write under the system's temporary directory. The training split of the first
two cases is three images of 2 rows and 3 columns holding the bytes 100 to 117, with the
labels 4, 0 and 2.

## ASampleHasItsLabelAndThePlaceOfItsPixels

The list has one file, `train-images-idx3-ubyte`, no class names, a class count of 5 and three
samples. Sample i names file 0, has the class of its label and no object, and its stored
pixels start at byte 16 + 6i, are 2 by 3 with one channel and are `PlaneLayout::Interleaved`.

## ASamplesPixelsReadBackFromTheImageFile

`read_stored_pixels` over the third sample gives the bytes 112 to 117.

## TheTestSplitReadsTheT10kFiles

With the split "test", `t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` are read: one
sample of class 7 and a class count of 8.

## NamedFilesAreReadForAnySplit

With the split "balanced" and the files `images.idx` and `labels.idx` named, two samples are
read. The image file has four dimensions, the last being 3; the second sample has three
channels and its pixels start at byte 23.

## ASplitWithoutAFileNameIsRefused

The split "validation" with no file named fails with `InvalidDescription` and "validation".

## AnAbsentFileIsNotFound

With the image file alone present, the read fails with `FileNotFound` and
"train-labels-idx1-ubyte".

## CountsThatDifferAreRefused

Two images with three labels fail with `InvalidFile`, "train-images-idx3-ubyte", "2 images",
"train-labels-idx1-ubyte" and "3 labels".

## AnImageFileOfAnotherFormIsRefused

An image file of two dimensions fails with `InvalidFile` and "2 dimensions", and one of
element code 0x0C with `InvalidFile` and "code 12". One whose fourth dimension is 5 fails with
`Unsupported` and "5 channels". Each message names "train-images-idx3-ubyte".

## AnImageFileWithoutAnImageIsRefused

An image file whose first extent is 0 fails with `InvalidFile`, "train-images-idx3-ubyte" and
"no image".
