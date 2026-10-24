# tests/core/data/test_dataset.cpp


Suites: `Medium_Dataset`.

Tests what `Data::SyntheticClassification` states and how it reads a batch, and which options
`Data::SyntheticClassification` and `Data::Rings` refuse.

The fixture needs an engine. `small_synthetic` gives 2 batches of 4 samples, 3 features and 2
classes. The labels and the shapes of `Data::Rings` are pinned by the `Medium_Registries` cases,
and the labels of the synthetic set by `test_target.cpp`.

## SyntheticClassificationStatesItsExtents

The synthetic set reports a batch size of 4, 2 batches and the name
"synthetic-classification". Batch 1 holds [4, 3] inputs.

## AReadSharesTheHeldBatch

Two reads of batch 0 hand out inputs with the same data pointer.

## TheSeedFixesTheSyntheticInputs

The inputs of batch 1 are equal for two sets seeded 5 and differ for a set seeded 6.

## SyntheticClassificationRefusesANonPositiveExtent

A feature count of 0 fails with `InvalidDescription`, naming "positive extents" and
"features=0".

## SyntheticClassificationReadsTheRowsOfARange

`read_rows(1, {1, 2})` gives inputs of 2 rows, and those rows and the `labels` role hold, byte for
byte, rows 1 and 2 of what `read(1)` gives.

## RingsReadsTheRowsOfARange

`read_rows(0, {3, 1})` of a rings set of 2 batches of 4 gives inputs of 1 row, equal to row 3 of
`read(0)`, and the same for its `labels`.

## RingsRefusesAnOptionOutOfRange

`Data::Rings::make` fails with `InvalidDescription` for a distractor count of -1, a ring width
of 0 and a position noise of -0.5, and each message names the option.
