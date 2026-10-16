# tests/vision/datasets/test_mnist.cpp


Suites: `Small_DatasetsMNIST`, `Medium_DatasetsMNIST`.

Tests `MNISTOptions`, `try_load_mnist` and `make_mnist` in
`modules/vision/datasets/include/SushiAI/vision/datasets/mnist.hpp` over IDX files the test
writes, so no MNIST download is needed. The refusals of the factory (an unknown key, a missing
directory, an unknown split or layout, absent files) are tested through the dataset registry
in `tests/vision/registration/test_registration.cpp`, under `Medium_VisionMnist`; this file
tests what the loader reads.

## Medium_DatasetsMNIST

The fixture writes, under the system's temporary directory, "train-images-idx3-ubyte" with five
images of 2 by 3 pixels, where pixel `j` of image `i` holds `6 i + j`, and
"train-labels-idx1-ubyte" with the classes 3, 1, 4, 1, 5. `options` reads that directory in
batches of two with no cap.

## TheDefaultsReadACappedTrainingSplitAsRows

A default `MNISTOptions` reads "data/mnist", 64 samples per batch, at most 4096 samples, the
training split, in `MNISTLayout::Flat`.

## TheSamplesLoadAsWholeBatches

The split is available. Five samples in batches of two give two batches of samples of the shape
[6], so the fifth sample is left out. Batch 1 holds images 2 and 3: its twelve pixels are the
bytes 12 to 23 divided by 255, and its labels are 4 and 1. The test divides as the loader does,
so the pixels compare with a tolerance of 0.

## TheSampleCapLimitsTheBatches

A cap of three samples gives one batch.

## ABatchSizeThatIsNotPositiveIsRefused

A batch size of 0 fails with InvalidDescription naming "positive batch size".

## AnImageFileWithoutTheIDX3HeaderIsInvalid

An image file rewritten with the one-axis header of a label file fails with InvalidFile, "is not
an IDX3 image file".

## TheFactoryReadsTheImageLayout

`make_mnist` over the same directory with "layout" set to "image" gives samples of the shape
[2, 3, 1] in two batches.
