# tests/vision/datasets: test_pipeline_table.cpp, test_fit_pipeline.cpp, test_augment_pipeline.cpp


The pipeline table, the fit the same in every epoch, a training sample identical alone, in a batch and reversed, the mosaic epochs, masks, and the refusals.

Suites: `Small_DatasetsPipelineTable`, `Small_DatasetsPipelineTableName`,
`Small_DatasetsPipelineTableRefusal`, `Small_DatasetsFitPipeline`,
`Medium_DatasetsFitPipelineSample`, `Small_DatasetsAugmentPipeline`,
`Medium_DatasetsAugmentPipeline`, `Medium_DatasetsAugmentPipelineSample`,
`Medium_DatasetsAugmentPipelineEpochSample` and `Small_DatasetsAugmentPipelineMakeRefusal`.

Tests `pipeline_table` and `find_pipeline`, `FitPipeline` and `AugmentPipeline` in
`modules/vision/datasets/include/SushiAI/vision/datasets/`. The pipeline cases that read files
are `Medium_`. They read four raw RGB images of the sizes 20 by 16, 16 by 20, 16 by 16 and 10 by 20, written under the system's
temporary directory; sample i holds one object of class i. The training pipelines have a size
of 16 and the seed 20261011.

## ItListsNoneThenDefault

`pipeline_table` has two rows, named "none" and then "default".

## FindPipelineFindsEachNameInTheTable

`Small_DatasetsPipelineTableName`, one row each for "none" and "default": `find_pipeline`
returns the row of that name.

## AnUnknownNameFindsNoRow

`find_pipeline("mixup")` returns null.

## ARowBuildsThePipelineOfItsName

For the same rows and a request of 16 by 16 and three channels, the row's `make` gives a
pipeline whose `name` is the row's name.

## DefaultRefusesAnInputItDoesNotMake

`Small_DatasetsPipelineTableRefusal`, one row each: the "default" row fails with
`InvalidDescription` and "'default'" for a size of 4 by 6 ("square", "4 by 6") and for one
channel ("3 channels", "has 1").

## AFittedSampleHasTheInputSizeAndThreeChannels

`Medium_DatasetsFitPipelineSample`, one row each for the samples 3 and 0. A `FitPipeline` that
letterboxes to 8 by 8 is named "none" and gives an image of 8 by 8 and three channels.

## AFittedSampleKeepsTheClassOfItsRow

For the same rows, the fitted sample holds one object, of the class of its row, 3 or 0.

## TheFitIsTheSameInEveryEpoch

For the same rows, epoch 7 gives the same image, mask, file, classes and boxes as epoch 0.

## MakeRefusesASizeOfNoPixel

`FitPipeline::make` with a size of 0 by 8 fails with `InvalidDescription`, "FitPipeline" and
"0 by 8".

## ProduceRefusesCountsThatDiffer

`produce` with two sample indices and one fitted slot fails with `InvalidDescription`,
"2 samples" and "1 fitted".

## ASampleHasTheTrainingSizeAndThreeChannels

`Medium_DatasetsAugmentPipelineSample`, one row each for the samples 0 to 3. An
`AugmentPipeline` with 1000 mosaic epochs is named "default". In epoch 0 the sample is 16 by 16
with three channels.

## TheBoxesOfASampleLieInsideTheImage

For the same rows, in epoch 0 every box of the sample lies within [0, 16] on both axes.

## ASampleWithoutAMaskHasAnEmptyMask

For the same rows, in epoch 0 the mask of the sample is empty.

## ASampleDependsOnItsIndexAndItsEpochAlone

For the same rows, in epoch 2 the sample comes out the same when the four are produced twice,
when they are produced in reverse order and when it is produced alone.

## ANewEpochChangesTheImages

In epoch 3 the image of at least one of the four samples differs from its image in epoch 2.

## PastTheMosaicEpochsASampleHoldsItsOwnObjects

`Medium_DatasetsAugmentPipelineEpochSample`, one row per epoch 0, 1 and 2 and sample 0 to 3.
With 0 mosaic epochs, the objects of sample i all have the class i.

## AMaskTakesTheSizeOfItsSample

For the sample rows, each sample is given a mask of its image's size that holds its index. In
epoch 1 the produced mask is 16 by 16 with one channel, and each byte is below 4 or is
`K_MASK_VOID`.

## AnImageThatCannotBeReadIsTheFailure

`Medium_DatasetsAugmentPipeline`. With the file of sample 1 replaced by a path that does not
exist, `produce` fails with `FileNotFound` and "gone.saim".

## MakeRefusesWhatItCannotBuild

`Small_DatasetsAugmentPipelineMakeRefusal`, one row each: `AugmentPipeline::make` fails with
`InvalidDescription` for a size of 0 ("AugmentPipeline", "a training size of 0") and for -1
mosaic epochs ("-1 epochs of mosaic").

## ProduceRefusesANegativeEpoch

`produce` fails with `InvalidDescription` and "epoch -1" for the epoch -1. The list names a file
that is never read.

## ProduceRefusesAnEmptyList

`produce` fails with `InvalidDescription` and "a list of 0 samples" for an empty list.
