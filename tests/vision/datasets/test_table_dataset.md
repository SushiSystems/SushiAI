# tests/vision/datasets/test_table_dataset.cpp


Suites: `Small_DatasetsTableDataset`, `Small_DatasetsTableDatasetMake`,
`Small_DatasetsTableDatasetRange` and `Medium_DatasetsTableDataset`.

Tests `TableDataset` in
`modules/vision/datasets/include/SushiAI/vision/datasets/table_dataset.hpp`. Every suite derives
from `TableBench`, which holds a `HostExecutionBackend`, so batches are host tensors and no file
or process-wide engine is reached; only the case that stops its process is `Medium_`. The list names the file `rows.csv` and holds rows of two features and one
target, row i being 10i, 10i + 1 and 10i + 2. The default options are batches of two rows
under the name "rows", with a `RegressionCollator` of two features and one target.

## ItDescribesItsBatches

Over five rows the dataset reports the name "rows", a batch size of 2, two batches, the sample
shape [2] and one target of the role `K_ROLE_VALUES` and the sample shape [1].

## ABatchHoldsItsRowsInListOrder

Through a `Data::Dataset` reference, batch 1 holds the inputs 20, 21, 30, 31 and the values
22, 32.

## ASecondReadHoldsItsOwnRows

After `read(1, batch)`, `read(0, batch)` into the same batch holds the inputs 0, 1, 10, 11. The
data pointer of the tensors is an allocation detail and is not checked.

## MakeRefusesWhatItCannotBatch

`Small_DatasetsTableDatasetMake`, one row each: `TableDataset::make` fails with `InvalidDescription` for a null collator ("TableDataset",
"'rows' has no collator"), a list without a table ("a table of 0 features"), a table one
number short ("4 samples and 11 numbers", "a row holds 3"), a batch size of 0
("a batch size of 0") and one row for batches of two ("1 samples", "one batch of 2").

## ARangeReadHoldsTheRowsOfTheWholeBatch

Over six rows in batches of 3, `read_rows(1, {2, 1})` gives inputs of 1 row, and that row and
the targets equal row 2 of `read(1)`.

## ABatchOutsideTheEpochStopsTheProcess

Over two rows, `read(1, batch)` is fatal with "TableDataset", "1 batches" and "batch 1".

## TheFailureOfTheCollatorIsReturned

With a collator of two features and two targets over the list of one target, `make` succeeds
and `read(0, batch)` fails with `InvalidDescription`, "rows.csv", "RegressionCollator" and
"2 features and 1 targets".

## RegressionABadRangeIsRefusedWithAStatus

`Small_DatasetsTableDatasetRange`, one row each. Over six rows in batches of 3,
`read_rows(1, {2, 2})` fails with `InvalidDescription`, "Rows 2 to 4" and "of batch 1", and
`read_rows(99, {0, 1})` with `InvalidDescription` and "Batch 99 is out of range".
