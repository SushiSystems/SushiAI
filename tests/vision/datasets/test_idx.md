# tests/vision/datasets/test_idx.cpp


Suites: `Small_DatasetsIdx`, `Small_DatasetsIdxForm` and `Medium_DatasetsIdx`.

Tests `read_idx` in `modules/vision/datasets/include/SushiAI/vision/datasets/idx.hpp`. The `Small_`
cases read the files from a `MemorySource` under the folder `data`; `AnAbsentFileIsNotFound`
reaches the file system and is `Medium_`. The training split of the first three cases is three
images of 2 rows and 3 columns holding the bytes 100 to 117, with the labels 4, 0 and 2.

## ASampleHasTheClassOfItsLabel

The three samples have the classes 4, 0 and 2, the class count is 5 and the list has no class
names.

## ASampleRefersToThePlaceOfItsPixels

The list has one file, `train-images-idx3-ubyte`. Every sample names file 0 and has no object; its
stored pixels are 2 by 3 with one channel and are `PlaneLayout::Interleaved`, and they start at
bytes 16, 22 and 28.

## ASamplesPixelsReadBackFromTheImageFile

`read_stored_pixels` over the third sample, from a stream holding the image file, gives the bytes
112 to 117.

## TheTestSplitReadsTheT10kFiles

With the split "test", `t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` are read: one
sample of class 7 and a class count of 8.

## NamedFilesAreReadForAnySplit

With the split "balanced" and the files `images.idx` and `labels.idx` named, two samples are
read. The image file has four dimensions, the last being 3; the second sample has three
channels and its pixels start at byte 23.

## ASplitWithoutAFileNameIsRefused

The split "validation" with no named file fails with `InvalidDescription` and "validation", over an
empty source.

## AnAbsentFileIsNotFound

With the image file written to a scratch folder and the label file missing, `read_idx` fails with
`FileNotFound` and "train-labels-idx1-ubyte".

## CountsThatDifferAreRefused

Two images against three labels fail with `InvalidFile` naming "train-images-idx3-ubyte",
"2 images", "train-labels-idx1-ubyte" and "3 labels".

## AnImageFileOfAnotherFormIsRefused

One row each: an image file of two dimensions fails with `InvalidFile` and "2 dimensions"; one of
element code 12 fails with `InvalidFile` and "code 12"; one of five channels fails with
`Unsupported` and "5 channels". Every message names "train-images-idx3-ubyte".

## AnImageFileWithoutAnImageIsRefused

An image file of zero images fails with `InvalidFile`, "train-images-idx3-ubyte" and "no image".
