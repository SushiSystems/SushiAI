# tests/core/data/test_epoch.cpp


Suites: `Small_DataEpochSequence`, `Small_DataEpoch`.

Tests that `Data::begin_epoch` reaches a dataset that implements `EpochAware` and leaves one that
does not.

This file needs no device. The datasets are `Data::InMemoryDataset`, which journals the indices
it reads, and `Data::InMemoryEpochDataset`, which adds `Data::InMemoryEpochAware` to it and so
holds the epoch last begun and counts the calls.

## EveryCallReachesAnEpochAwareDataset

Rows: the epoch sequences 4, and 0, 1, 0. After each sequence the dataset holds the last epoch of
it and counts one call per epoch.

## ADatasetThatIsNotEpochAwareIsLeftAlone

`Data::begin_epoch` with epoch 4 on an `InMemoryDataset` leaves its journal empty.

## ADatasetThatIsNotEpochAwareStillReads

After `Data::begin_epoch` with epoch 4 on an `InMemoryDataset` of two batches, `read(1, batch)`
succeeds and the journal holds the index 1 alone.
