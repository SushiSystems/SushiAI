# tests/vision/datasets/test_reader_dataset.cpp


Suites: `Small_DatasetsReaderDataset`, `Small_DatasetsReaderDatasetOptions`,
`Small_DatasetsReaderDatasetRange`, `Medium_DatasetsReaderDataset` and
`Medium_DatasetsReaderDatasetEpoch`.

Tests `ReaderDataset` in
`modules/vision/datasets/include/SushiAI/vision/datasets/reader_dataset.hpp`. Every suite derives
from `ReaderBench`, which holds a `HostExecutionBackend`, so batches are host tensors and no
device or process-wide engine is reached. The `Small_` cases name a list that is never read. The
`Medium_` cases read it: the list stores gray samples of 2 by 2 pixels in one file `pixels.bin` of
rising bytes, written under the system's temporary directory, so sample i opens with the byte 4i
and has the class i modulo 2. The default options are batches of two such inputs under the name
"tiny", with a classification collator over two classes.

## ItDescribesItsBatches

Over five samples the dataset reports a batch size of 2, the sample shape [2, 2, 1], two
batches, the name "tiny" and one target of the role `K_ROLE_LABELS`.

## ABatchHoldsItsSamplesInListOrder

Over four samples, batch 1 holds the samples 2 and 3, in that order.

## TheInputsOfABatchAreScaledBy255

Over four samples, batch 1 holds the inputs (8 + i) / 255 for i from 0 to 7, each within the
typed tolerance of a byte divided by 255.

## TheLabelsOfABatchFollowTheList

Over four samples, batch 1 holds the labels 0 and 1.

## ADetectionBatchLetterboxesTheImages

Two raw RGB files of 2 rows and 4 columns at level 255 are letterboxed to 4 by 4, with a
detection collator of 3 classes and 2 slots. The first input value is 114 / 255 and value 12
is 1.0.

## ADetectionBatchMovesTheBoxesWithTheImage

For the same batch, the box of the second image's object is (1, 1, 3, 3) within the typed
tolerance of a box coordinate.

## ADetectionBatchFlagsTheObjectRowAndItsClass

For the same batch, valid is 0, 0, 1, 0 and the class at slot 2 is 2.

## MakeRefusesOptionsOutsideTheirBounds

`Small_DatasetsReaderDatasetOptions`, one row each: `ReaderDataset::make` fails with
`InvalidDescription` for a size of 0 by 2 ("0 by 2"), five channels ("5 channels"), a batch
size of 0 ("batch size of 0") and a batch size of 3 over two samples ("2 samples",
"one batch of 3").

## MakeRefusesANullCollator

`make` fails with `InvalidDescription`, "ReaderDataset", "tiny" and "no collator".

## ABatchOutsideTheEpochStopsTheProcess

Over two samples, `read(1, batch)` is fatal with "ReaderDataset", "1 batches" and "batch 1".

## AReadIntoAFilledBatchHoldsTheValuesOfTheSecondRead

After `read(0, batch)` and `read(1, batch)` into one batch, the inputs are (8 + i) / 255 and the
labels are 0 and 1. The pointer of the tensors is an allocation detail and is not checked.

## AReadDoesNotWriteATensorSomethingElseHolds

With a copy of the input tensor kept after `read(0, batch)`, `read(1, batch)` gives inputs at
another data pointer. The kept tensor still opens with 0 and the batch opens with 8 / 255.

## ItIsADatasetATrainerCanRead

Through a `Data::Dataset` reference, `read(1, batch)` gives as many targets as `targets()`
lists, the first with the same role, and inputs whose first extent is `batch_size()`.

## AnImageOfAnotherChannelCountNamesItsFile

With the options asking for three channels over the gray list, `read(0, batch)` fails with
`Incompatible`, "pixels.bin", "ReaderDataset", "1 channels" and "reads 3".

## AFileAbsentAtReadTimeIsNamed

With the list's file replaced by a path that does not exist, `make` succeeds and
`read(0, batch)` fails with `InvalidFile` and "gone.bin".

## AShuffledEpochReadsEverySampleOnce

With `shuffle` on and the seed 42 over eight samples, the rows read across the four batches are
a permutation of the eight samples.

## AShuffledEpochFollowsTheOrderOfTheSeed

For the same dataset, the rows read are the samples in the order
`Data::EpochPermutation(42, 0, 8)` gives.

## BeginEpochSelectsTheOrderOfTheEpoch

After `Data::begin_epoch(dataset, 3)` the samples are read in the order of
`EpochPermutation(42, 3, 8)`, which differs from the order of epoch 0.

## BeginningTheFirstEpochAgainRestoresItsOrder

After `begin_epoch(dataset, 3)` and then `begin_epoch(dataset, 0)` the samples are read in the
order of epoch 0.

## WithoutShuffleEveryEpochKeepsTheOrderOfTheList

`Medium_DatasetsReaderDatasetEpoch`, one row each for the epochs 0 and 5. Without `shuffle`, six
samples are read as 0, 1, 2, 3, 4, 5 after `begin_epoch(dataset, epoch)`.

## ARangeReadHoldsTheRowsTheSeedOrderNames

With `shuffle` on and epoch 1 begun over eight samples in batches of 4, `read_rows(1, {1, 2})`
gives inputs of 2 rows. They hold the bytes 4s to 4s + 3, divided by 255, of the samples s at the
positions 5 and 6 of `EpochPermutation(42, 1, 8)`.

## ARangeReadOpensNoSampleOutsideIt

Over a list whose second file is absent, `read_rows(0, {1, 1})` succeeds with inputs of 1 row,
and `read(0)` fails with `InvalidFile` naming "gone.bin".

## MakeRefusesANullPipeline

The form of `make` that takes a pipeline fails for a null one with `InvalidDescription`,
"ReaderDataset", "tiny" and "no pipeline".

## RegressionABadRangeIsRefusedWithAStatus

`Small_DatasetsReaderDatasetRange`, one row each. Over eight samples in batches of 4,
`read_rows(1, {3, 2})` fails with `InvalidDescription`, "Rows 3 to 5" and "of batch 1", and
`read_rows(99, {0, 1})` with `InvalidDescription` and "Batch 99 is out of range".
