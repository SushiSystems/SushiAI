# tests/vision/datasets/test_table_dataset.cpp


Suites: `Medium_DatasetsTableDataset`.

Tests `TableDataset` in
`modules/vision/datasets/include/SushiAI/vision/datasets/table_dataset.hpp`. The fixture is an
`EngineTest`. Its list names the file `rows.csv` and holds rows of two features and one
target, row i being 10i, 10i + 1 and 10i + 2. The default options are batches of two rows
under the name "rows", with a `RegressionCollator` of two features and one target.

## ItDescribesItsBatches

Over five rows the dataset reports the name "rows", a batch size of 2, two batches, the sample
shape [2] and one target of the role `K_ROLE_VALUES` and the sample shape [1].

## ABatchHoldsItsRowsInListOrder

Through a `Data::Dataset` reference, batch 1 holds the inputs 20, 21, 30, 31 and the values
22, 32. A following `read(0, batch)` writes the same input tensor, whose first value is then 0.

## MakeRefusesWhatItCannotBatch

`TableDataset::make` fails with `InvalidDescription` for a null collator ("TableDataset",
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
