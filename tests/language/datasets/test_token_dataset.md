# tests/language/datasets/test_token_dataset.cpp


Suites: `Medium_LanguageDatasetsTokenDataset`, `Medium_LanguageDatasetsTokenDatasetStride`,
`Medium_LanguageDatasetsTokenDatasetRefusal`, `Medium_LanguageDatasetsTokenDatasetRange`.

Tests `Language::Datasets::TokenDataset` in
`modules/language/datasets/include/SushiAI/language/datasets/token_dataset.hpp`. The fixture
provides an engine and writes `rising.satk` to a scratch directory: the ids 0 to 19 over a
vocabulary of 20, so each id equals its position. Unless a case says otherwise the dataset has a
next-token collator of four positions, a batch size of 2 and the name "tiny". Every case reads a
file and fills engine tensors, so none runs `Small_`. The windows and the file format are tested
small in `test_window_layout.cpp` and `test_token_file.cpp`.

## ItDescribesItsBatches

The dataset has a batch size of 2, the sample shape [4], the type `Int32`, two batches and the
name "tiny".

## ItDescribesItsVocabularyAndLabels

The dataset has a vocabulary size of 20. Its one target has the role `K_ROLE_LABELS`, the sample
shape [4] and 20 classes.

## ABatchHoldsItsWindowsAsInt32

`read(1)` fills `Int32` inputs of the shape [2, 4] and one `Int32` target of eight elements. The
inputs are the ids 8 to 15 and the labels the ids 9 to 16.

## EveryLabelIsTheTokenAfterItsInput

With a stride of 3 the dataset has three batches. One row per batch: batch 0 holds the windows
that begin at the ids 0 and 3, batch 1 at 6 and 9, batch 2 at 12 and 15. The inputs are those
windows and the labels are the same windows shifted by one id.

## MakeRefusesOptionsOutsideTheirBounds

One promise over six rows: `TokenDataset::make` fails with `InvalidDescription` for a null
collator ("TokenDataset", "tiny", "no collator"), a batch size of 0 ("batch size of 0"), a stride
of 0 ("TokenDataset", "tiny", "stride of 0") and a batch size of 5 ("rising.satk", "4 windows of
5", "one batch of 5"). It fails with `Incompatible` for a collator over 19 tokens ("rising.satk",
"vocabulary of 20", "one of 19") and with `FileNotFound` for a path that does not exist
("absent.satk"). The id outside the vocabulary of a file is tested in `test_token_file.cpp`.

## AFileCutAfterMakeNamesWhereItEnds

The file is resized to 64 bytes after the dataset is made. `read(0)` is ok and `read(1)` fails
with `InvalidFile` and a message containing "rising.satk", "ends at byte 64" and "end at byte
76".

## ABatchOutsideTheEpochStopsTheProcess

`read(2)` on a dataset of two batches stops the process with a report containing "TokenDataset",
"2 batches" and "batch 2".

## AReadReusesTheTensorsItFilledBefore

After `read(0)` and `read(1)` into the same batch, the inputs and the labels have the data
pointers they had after the first read, and the first input is 8.

## AReadDoesNotWriteATensorSomethingElseHolds

A copy of the inputs tensor is kept after `read(0)`. After `read(1)` the batch's inputs have
another data pointer, the first id of the kept tensor is 0 and the first id of the batch is 8.

## ItIsADatasetATrainerCanRead

Read through a `Data::Dataset` reference, `read(1)` fills as many targets as `targets()` lists,
the first target has the role the description states, and the first extent of the inputs equals
`batch_size()`.

## ARangeReadHoldsTheRowsOfTheWholeBatch

`read_rows(1, {1, 1})` gives inputs of 1 row, and that row and the labels equal row 1 of
`read(1)`.

## ARangeReadReadsNoWindowOutsideIt

After the file is cut to 76 bytes, which keep the ids 0 to 12, `read_rows(1, {0, 1})` succeeds
with inputs of 1 row, and `read(1)` fails with `InvalidFile` naming "rising.satk".

## RegressionABadRangeIsRefusedWithAStatus

One promise over two rows: `read_rows` returns `InvalidDescription` rather than stopping, for the
rows 1 to 3 of batch 1 ("Rows 1 to 3", "of batch 1", "a batch of 2") and for batch 99 ("Batch 99
is out of range").
