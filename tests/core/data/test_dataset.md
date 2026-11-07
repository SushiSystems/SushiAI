# tests/core/data/test_dataset.cpp


Suites: `Small_SyntheticLabels`, `Small_SyntheticMeans`, `Medium_Dataset`, `Medium_DatasetExtent`, `Medium_RingsOption`, `Medium_DatasetSharing`,
`Medium_DatasetRange`, `Medium_DatasetRefusal`.

Tests what `Data::SyntheticClassification` generates, how every dataset reads a batch or a range of
its rows, and which options the dataset factories refuse.

The fixtures need an engine. `small_synthetic` gives 2 batches of 4 samples, 3 features and 2
classes, and `small_rings` gives 3 batches of 4 samples. The contract suites run on a table of
dataset kinds: `SyntheticClassification`, `Rings`, a `MaterialisedDataset` of two batches of 4
samples and a `ThreadedLoader` over a rings set. The labels and the shapes of `Data::Rings` are
pinned by the `Medium_Registries` cases, and the labels of the synthetic set by `test_target.cpp`.

## AReadGivesInputsOfBatchSizeByFeatureCount

Batch 1 of the synthetic set holds [4, 3] inputs.

## TheSeedFixesTheSyntheticInputs

The inputs of batch 1 are equal for two sets seeded 5 and differ for a set seeded 6.

## LabelsRunRoundRobinOverTheWholeSet

Rows: `fill_round_robin_labels` from position 0 over 3 classes gives 0, 1, 2, 0, 1, 2, 0; from
position 5 over 3 classes it gives 2, 0, 1, 2; over one class it gives 0 for every sample. These
are the labels the synthetic set and the rings set publish, and they need no device.

## EachRowTakesTheMeanOfItsClass

`copy_class_means` over the means (1, 2), (10, 20) and (100, 200) and the labels 2, 0, 2, 1 gives the rows
(100, 200), (1, 2), (100, 200) and (10, 20).

## AFactoryRefusesANonPositiveExtent

Rows: a synthetic set with 0 features, which names "positive" and "features=0", and a rings set
built through the registry with a batch size of 0, which names "positive". Both fail with
`InvalidDescription`.

## RingsRefusesAnOptionOutOfRange

Rows: a distractor count of -1, a ring width of 0 and a position noise of -0.5. `Data::Rings::make`
fails with `InvalidDescription` and the message names the option.

## AReadSharesTheHeldBatch

Kinds that hold their batches: the synthetic set, the rings set and the materialised set. Two reads
of batch 0 hand out the same input and target data pointers, and a read of batch 1 hands out other
inputs.

## ARangeReadSharesTheStorageOfTheHeldBatch

For the same kinds, `read_rows(1, {1, 2})` gives inputs on the storage that `read(1)` gives.

## ARangeReadGivesTheRowsOfTheWholeBatch

Rows: every kind, with `read_rows(1, {1, 2})` and `read_rows(0, {3, 1})`. The part has `count` rows
and its inputs and every role hold, byte for byte, the rows of what `read` gives.

## ABadRangeIsRefusedWithAStatus

For every kind, `read_rows(1, {3, 2})` fails with `InvalidDescription` naming "Rows 3 to 5",
"of batch 1" and "a batch of 4", and `read_rows(99, {0, 1})` names "Batch 99 is out of range".
