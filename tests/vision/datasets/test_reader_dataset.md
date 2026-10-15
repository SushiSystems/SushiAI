# tests/vision/datasets/test_reader_dataset.cpp


Suites: `Unit_DatasetsReaderDataset`.

Tests `ReaderDataset` in
`modules/vision/datasets/include/SushiAI/vision/datasets/reader_dataset.hpp`. The fixture is an
`EngineTest`. Its list stores gray samples of 2 by 2 pixels in one file `pixels.bin` of rising
bytes, written under the system's temporary directory, so sample i opens with the byte 4i and
has the class i modulo 2. The default options are batches of two such inputs under the name
"tiny", with a classification collator over two classes.

## ItDescribesItsBatches

Over five samples the dataset reports a batch size of 2, the sample shape [2, 2, 1], two
batches, the name "tiny" and one target of the role `K_ROLE_LABELS`.

## ABatchHoldsItsSamplesInListOrder

Over four samples, batch 1 holds the inputs (8 + i) / 255 for i from 0 to 7, each within 1e-6,
and the labels 0 and 1.

## ADetectionBatchLetterboxesImagesAndBoxes

Two raw RGB files of 2 rows and 4 columns at level 255 are letterboxed to 4 by 4, with a
detection collator of 3 classes and 2 slots. The first input value is 114 / 255 and value 12
is 1.0. The second image's object fills slot 2: valid is 0, 1, 0 at the slots 0, 2 and 3, the
class is 2 and the box is (1, 1, 3, 3) within 1e-4.

## MakeRefusesOptionsOutsideTheirBounds

`ReaderDataset::make` fails with `InvalidDescription` for a null collator ("ReaderDataset",
"tiny", "no collator"), a size of 0 by 2 ("0 by 2"), five channels ("5 channels"), a batch
size of 0 ("batch size of 0") and a batch size of 3 over two samples ("2 samples",
"one batch of 3").

## ABatchOutsideTheEpochStopsTheProcess

Over two samples, `read(1, batch)` is fatal with "ReaderDataset", "1 batches" and "batch 1".

## AReadReusesTheTensorsItFilledBefore

After `read(0, batch)` and `read(1, batch)` into one batch, the inputs and the labels have the
data pointers of the first read, and the first input value is 8 / 255.

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

## AShuffledEpochReadsEverySampleOnceInTheOrderOfTheSeed

With `shuffle` on and the seed 42 over eight samples, the rows read across the four batches
are the samples in the order `Data::EpochPermutation(42, 0, 8)` gives, and each of the eight
is read once.

## BeginEpochSelectsTheOrderOfTheEpoch

After `Data::begin_epoch(dataset, 3)` the samples are read in the order of
`EpochPermutation(42, 3, 8)`, which differs from the order of epoch 0. After
`begin_epoch(dataset, 0)` they are read in the order of epoch 0.

## WithoutShuffleEveryEpochKeepsTheOrderOfTheList

Without `shuffle`, six samples are read as 0, 1, 2, 3, 4, 5, and again in that order after
`begin_epoch(dataset, 5)`.

## MakeRefusesANullPipeline

The form of `make` that takes a pipeline fails for a null one with `InvalidDescription`,
"ReaderDataset", "tiny" and "no pipeline".
