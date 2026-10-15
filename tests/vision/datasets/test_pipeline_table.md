# tests/vision/datasets: test_pipeline_table.cpp, test_fit_pipeline.cpp, test_augment_pipeline.cpp


The pipeline table, the fit the same in every epoch, a training sample identical alone, in a batch and reversed, the mosaic epochs, masks, and the refusals.

Suites: `Unit_DatasetsPipelineTable`, `Unit_DatasetsFitPipeline`,
`Unit_DatasetsAugmentPipeline`.

Tests `pipeline_table` and `find_pipeline`, `FitPipeline` and `AugmentPipeline` in
`modules/vision/datasets/include/SushiAI/vision/datasets/`. The pipeline cases read four raw
RGB images of the sizes 20 by 16, 16 by 20, 16 by 16 and 10 by 20, written under the system's
temporary directory; sample i holds one object of class i. The training pipelines have a size
of 16 and the seed 20261011.

## ItListsNoneThenDefault

`pipeline_table` has two rows named "none" and "default". `find_pipeline` returns each row for
its name and null for "mixup".

## ARowBuildsThePipelineOfItsName

For a request of 16 by 16 and three channels, each row's `make` gives a pipeline whose `name`
is the row's name.

## DefaultRefusesAnInputItDoesNotMake

The "default" row fails with `InvalidDescription` and "'default'" for a size of 4 by 6
("square", "4 by 6") and for one channel ("3 channels", "has 1").

## FitsEachSampleTheSameInEveryEpoch

A `FitPipeline` that letterboxes to 8 by 8 is named "none". For the samples 3 and 0 it gives
images of 8 by 8 and three channels with one object each, of the classes 3 and 0. Epoch 7
gives the same image, mask, file, classes and boxes as epoch 0.

## RefusesWhatItCannotFit

`FitPipeline::make` with a size of 0 by 8 fails with `InvalidDescription`, "FitPipeline" and
"0 by 8". `produce` with two sample indices and one fitted slot fails with
`InvalidDescription`, "2 samples" and "1 fitted".

## MakesSquareSamplesOfThreeChannels

An `AugmentPipeline` with 1000 mosaic epochs is named "default". In epoch 0 each of the four
samples is 16 by 16 with three channels and an empty mask, and every box lies within
[0, 16] on both axes.

## ASampleDependsOnItsIndexAndItsEpochAlone

In epoch 2, each sample comes out the same when the four are produced twice, when they are
produced in reverse order and when sample 2 is produced alone. In epoch 3 the image of at
least one sample differs from its image in epoch 2.

## PastTheMosaicEpochsASampleHoldsItsOwnObjects

With 0 mosaic epochs, in the epochs 0, 1 and 2 every object of sample i has the class i.

## AMaskTakesTheSizeOfItsSample

Each sample is given a mask of its image's size that holds its index. In epoch 1 every
produced mask is 16 by 16 with one channel, and each byte is below 4 or is `K_MASK_VOID`.

## AnImageThatCannotBeReadIsTheFailure

With the file of sample 1 replaced by a path that does not exist, `produce` fails with
`FileNotFound` and "gone.saim".

## RefusesWhatItCannotMake

`AugmentPipeline::make` fails with `InvalidDescription` for a size of 0 ("AugmentPipeline",
"a training size of 0") and for -1 mosaic epochs ("-1 epochs of mosaic"). `produce` fails
with `InvalidDescription` for the epoch -1 ("epoch -1") and for an empty list
("a list of 0 samples").
